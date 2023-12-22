# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import numpy as np
import copy
import gradio as gr

from collections import OrderedDict
from typing import Union, Tuple, List, Dict, Any
import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import TypeHelper
import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal

# Structure of this extension adapted from Automatic's reference Unet extension for TensorRT:
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt, and by extension:
# https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT

class OrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[ORT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        if shared.sd_model.is_sdxl:
            raise ValueError(
                    "SD XL models are not supported with this version of the DirectML extension."
                )        
        return OrtUnet(self.filename)


input_to_ort_names_map = {
    "x": "sample",
    "timesteps": "timestep",
    "context":"encoder_hidden_states"
}


class CudaSession:
    """Inference Session with IO Binding for ONNX Runtime CUDA provider"""

    def __init__(self, ort_session: ort.InferenceSession, device: torch.device, enable_cuda_graph=False):
        self.ort_session = ort_session
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        self.io_name_to_numpy_type = TypeHelper.get_io_numpy_type_map(self.ort_session)
        self.io_binding = self.ort_session.io_binding()
        self.enable_cuda_graph = enable_cuda_graph

        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()
        self.device = device

    def __del__(self):
        del self.input_tensors
        del self.output_tensors
        del self.io_binding
        del self.ort_session

    def allocate_buffers(self, shape_dict: Dict[str, Union[Tuple[int], List[int]]]):
        """Allocate tensors for I/O Binding"""
        if self.enable_cuda_graph:
            for name, shape in shape_dict.items():
                if name in self.input_names:
                    # Reuse allocated buffer when the shape is same
                    if name in self.input_tensors:
                        if tuple(self.input_tensors[name].shape) == tuple(shape):
                            continue
                        raise RuntimeError("Expect static input shape for cuda graph")

                    numpy_dtype = self.io_name_to_numpy_type[name]
                    tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                        device=self.device
                    )
                    self.input_tensors[name] = tensor

                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        numpy_dtype,
                        list(tensor.size()),
                        tensor.data_ptr(),
                    )

        for name, shape in shape_dict.items():
            if name in self.output_names:
                # Reuse allocated buffer when the shape is same
                if name in self.output_tensors and tuple(self.output_tensors[name].shape) == tuple(shape):
                    continue

                numpy_dtype = self.io_name_to_numpy_type[name]
                tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                    device=self.device
                )
                self.output_tensors[name] = tensor

                self.io_binding.bind_output(
                    name,
                    tensor.device.type,
                    tensor.device.index,
                    numpy_dtype,
                    list(tensor.size()),
                    tensor.data_ptr(),
                )

    def infer(self, feed_dict: Dict[str, torch.Tensor]):
        """Bind input tensors and run inference"""
        for name, tensor in feed_dict.items():
            assert isinstance(tensor, torch.Tensor) and tensor.is_contiguous()
            if name in self.input_names:
                if self.enable_cuda_graph:
                    assert self.input_tensors[name].nelement() == tensor.nelement()
                    assert self.input_tensors[name].dtype == tensor.dtype
                    assert tensor.device.type == "cuda"
                    # Update input tensor inplace since cuda graph requires input and output has fixed memory address.
                    self.input_tensors[name].copy_(tensor)
                else:
                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        TypeHelper.torch_type_to_numpy_type(tensor.dtype),
                        [1] if len(tensor.shape) == 0 else list(tensor.shape),
                        tensor.data_ptr(),
                    )

        self.ort_session.run_with_iobinding(self.io_binding)

        return self.output_tensors

    @staticmethod
    def get_cuda_provider_options(device_id: int, enable_cuda_graph: bool) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }
    

class OrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.session = None
        self.sess_options = None
        #self.output_name = None
        self.buffers = None
        self.dimensions = {}
        self.static_dimension_dims = True

    def dimensionschanged(self,feed_dict):
        if ((self.dimensions["width"] != feed_dict['x'].shape[-1]) or
        (self.dimensions["height"] != feed_dict['x'].shape[-2]) or
        (self.dimensions["batch_size"] != feed_dict['x'].shape[-4])):
            gr.Info("Dimensions changed, restarting ORT inference session, first job may be slow. \
                    Select dynamic dimensions from Settings if you expect to change image size/batch size frequently")
            return True

        return False

    def infer(self, feed_dict):
        new_profile = shared.opts.data.get("ort_static_dims", False)
        if (self.session is None or
            new_profile != self.static_dimension_dims or
            (self.static_dimension_dims and self.dimensionschanged(feed_dict))):

            # Input has shape NxCxHxW
            self.dimensions["width"] = feed_dict['x'].shape[-1]
            self.dimensions["height"] = feed_dict['x'].shape[-2]
            self.dimensions["channel_size"] = feed_dict['x'].shape[-3]
            self.dimensions["batch_size"] = feed_dict['x'].shape[-4]
        
            self.initsession()
        self.static_dimension_dims = new_profile

        shape_dict = {
            "sample": list(feed_dict['x'].shape),
            "timestep": (1,),
            "encoder_hidden_states":  list(feed_dict['context'].shape),
            "out_sample": (self.dimensions["batch_size"], 4, self.dimensions["height"], self.dimensions["width"]),
        }
        # if shared.sd_model.is_sdxl:
        #     shape_dict = {
        #         "sample": list(feed_dict['x'].shape),
        #         "timestep": (1,),
        #         "encoder_hidden_states":  list(feed_dict['context'].shape),
        #         "text_embeds": (self.dimensions["batch_size"], 1280),
        #         "time_ids": list(feed_dict['time_ids'].shape),
        #         "out_sample": (self.dimensions["batch_size"], 4, self.dimensions["height"], self.dimensions["width"]),
        #     }
        print(shape_dict)
        self.session.allocate_buffers(shape_dict)

        inputs = {}
        for name, tensor in feed_dict.items():
            print(f"input name={name}, dtype={tensor.dtype}, shape={tensor.shape}")
            input_name = input_to_ort_names_map[name]
            
            if tensor.dtype != torch.float16:
                inputs[input_name] = tensor.clone().to(torch.float16)
            else:
                inputs[input_name] = tensor.clone()

        return self.session.infer(inputs)

    def forward(self, x, timesteps, context, *args, **kwargs):
        feed_dict = {
            "x": x,
            "timesteps": timesteps,
            "context": context,
        }

        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError(
                    "Input shape must be divisible by 64 in both dimensions."
                )

        outputs = self.infer(feed_dict)

        return outputs["out_sample"].to(dtype=x.dtype, device=devices.device)


    def activate(self):
        print("ORT activation delayed to inference time")

    def initsession(self):
        print("devices.device", devices.device)
        device_id = torch.cuda.current_device()
        enable_cuda_graph = shared.opts.data.get("ort_static_dims", False)
        providers = [
            ("CUDAExecutionProvider", CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)),
            "CPUExecutionProvider",
        ]
        print(providers)

        # Session Options
        self.sess_options = ort.SessionOptions()
        ort.set_default_logger_severity(3)
        # When the model has been optimized by onnxruntime, we can disable optimization to save session creation time.
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # if enable_cuda_graph:
        #     height = self.dimensions["height"]
        #     width = self.dimensions["width"]
        #     batch_size = self.dimensions["batch_size"]
        #     channels_size = self.dimensions["channel_size"]
        #     self.sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size)
        #     self.sess_options.add_free_dimension_override_by_name("unet_sample_channels", channels_size)
        #     self.sess_options.add_free_dimension_override_by_name("unet_sample_height", height)
        #     self.sess_options.add_free_dimension_override_by_name("unet_sample_width", width)
        #     self.sess_options.add_free_dimension_override_by_name("unet_time_batch", batch_size)
        #     self.sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size)
        #     self.sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

        onnx_path = self.filename
        print(f"Creating CUDA EP session for {onnx_path}")
        session = ort.InferenceSession(onnx_path, providers=providers, sess_options=self.sess_options)

        device = torch.device("cuda", device_id)
        self.session = CudaSession(session, device, enable_cuda_graph)

    def deactivate(self):
        devices.torch_gc()


def list_unets(unet_list):
    dml_dir = os.path.join(paths_internal.models_path, 'UNET-ORT')
    candidates = list(shared.walk_files(dml_dir, allowed_extensions=[".onnx"]))
    for filename in sorted(candidates, key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]

        opt = OrtUnetOption(filename, name)
        unet_list.append(opt)

def on_ui_settings():
        section = ('onnxruntime', "OnnxRuntime")
        shared.opts.add_option("ort_static_dims", shared.OptionInfo(
        False, "Enable ORT Unet Static Dimensions", gr.Checkbox, {"interactive": True}, section=section))

def on_ui_tabs():
    with gr.Blocks() as dml_interface:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with open(
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "help.md"),
                    "r",
                    encoding='utf-8',
                ) as f:
                    gr.Markdown(elem_id="ort_info", value=f.read())

    return [(dml_interface, "OnnxRuntime", "onnxruntime")]

script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)


