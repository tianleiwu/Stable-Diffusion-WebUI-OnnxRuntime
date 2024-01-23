# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import numpy as np
import onnxruntime as ort
import torch
from modules import devices, script_callbacks, scripts, sd_unet, shared
from onnxruntime.transformers.io_binding_helper import TypeHelper

import ui_ort
from ort_model_config import ModelType
from ort_model_manager import ORT_MODEL_DIR, modelmanager


class OrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, model_name: str, configs: List[dict]):
        self.label = "[ORT] " + configs[0]["filepath"][:-5]
        self.model_name = model_name
        self.configs = configs

    def create_unet(self):
        return OrtUnet(self.model_name, self.configs)


class CudaSession:
    """Inference Session with IO Binding for ONNX Runtime CUDA provider"""

    def __init__(
        self,
        ort_session: ort.InferenceSession,
        device: torch.device,
        enable_cuda_graph=False,
    ):
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

    def reset_io_binding(self):
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()

    def allocate_buffers(self, shape_dict: Dict[str, Union[Tuple[int], List[int]]], half_batch_size: bool):
        """Allocate tensors for I/O Binding"""
        if self.enable_cuda_graph:
            for name, shape in shape_dict.items():
                if name in self.input_names:
                    # Reuse allocated buffer when the shape is same
                    if name in self.input_tensors:
                        if list(self.input_tensors[name].shape) == shape:
                            continue
                        raise RuntimeError("Expect static input shape for cuda graph")

                    numpy_dtype = self.io_name_to_numpy_type[name]
                    tensor = torch.empty(
                        tuple(shape),
                        dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype),
                    ).to(device=self.device)
                    self.input_tensors[name] = tensor

                    # When CFG scale is 1.0, only use the first half since negative prompt is not used.
                    bind_shape = shape.copy()
                    if half_batch_size:
                        bind_shape[0] = bind_shape[0] // 2

                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        numpy_dtype,
                        bind_shape,
                        tensor.data_ptr(),
                    )

        for name, shape in shape_dict.items():
            if name in self.output_names:
                # Reuse allocated buffer when the shape is same
                if not (name in self.output_tensors and list(self.output_tensors[name].shape) == shape):
                    numpy_dtype = self.io_name_to_numpy_type[name]
                    tensor = torch.zeros(
                        tuple(shape),
                        dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype),
                    ).to(device=self.device)
                    self.output_tensors[name] = tensor

                # When CFG scale is 1.0, only use the first half since negative prompt is not used.
                bind_shape = shape.copy()
                if half_batch_size:
                    bind_shape[0] = bind_shape[0] // 2

                self.io_binding.bind_output(
                    name,
                    tensor.device.type,
                    tensor.device.index,
                    numpy_dtype,
                    bind_shape,
                    tensor.data_ptr(),
                )

    def infer(self, feed_dict: Dict[str, torch.Tensor], half_batch_size):
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
                    # When CFG scale is 1.0, only use the first half since negative prompt is not used.
                    bind_shape = list(tensor.shape)
                    if half_batch_size:
                        bind_shape[0] = bind_shape[0] // 2

                    self.io_binding.bind_input(
                        name,
                        tensor.device.type,
                        tensor.device.index,
                        TypeHelper.torch_type_to_numpy_type(tensor.dtype),
                        bind_shape,
                        tensor.data_ptr(),
                    )

        # TODO: Pass cuda stream to ORT so that we need not synchronize inputs and outputs
        # TODO: Also add disable_synchronize_execution_providers in run options, see https://github.com/microsoft/onnxruntime/pull/14088
        #self.io_binding.synchronize_inputs()
        
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        self.ort_session.run_with_iobinding(self.io_binding, run_options=run_options)

        # run_with_iobinding has synchronization and we binded the output buffer in device, so no need to synchronize_outputs.
        # self.io_binding.synchronize_outputs()

        return self.output_tensors

    @staticmethod
    def get_cuda_provider_options(device_id: int, enable_cuda_graph: bool, use_user_compute_stream:bool=True) -> Dict[str, Any]:
        options = {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }
        
        if use_user_compute_stream:
            options["user_compute_stream"] = str(torch.cuda.current_stream().cuda_stream)
            
        return options


class OrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.configs = configs
        self.profile_idx = 0

        self.session = None
        self.sess_options = None

        self.last_input_shape = None
        self.last_half_batch_size = None
        self.is_static_input_shape = True

        self.last_cuda_stream = None
        
        self.batch_size = 1
        self.use_cfg1_opt = True

        self.verbose = False
        if self.verbose:
            print("OrtUnet model_name:", model_name, "profiles:", configs)

    def get_onnx_path(self):
        if self.verbose:
            print(
                "get_onnx_path from configs with profile_idx={self.profile_idx}",
                self.configs,
            )
        assert len(self.configs) > 0
        assert self.profile_idx < len(self.configs)
        return os.path.join(ORT_MODEL_DIR, self.configs[self.profile_idx]["filepath"])

    def infer(self, feed_dict):
        is_static = shared.opts.data.get("ort_static_dims", False)

        input_shape = list(feed_dict["sample"].shape)

        half_batch_size = False
        if self.use_cfg1_opt and input_shape[0] == self.batch_size * 2:
            half_batch_size = True
            input_shape[0] = input_shape[0] // 2

        is_shape_changed = input_shape != self.last_input_shape and self.last_input_shape is not None
        is_half_changed = half_batch_size != self.last_half_batch_size and self.last_half_batch_size is not None
        
        if (
            self.session is None
            or is_static != self.is_static_input_shape
            or (self.is_static_input_shape and is_shape_changed)
        ):
            self.is_static_input_shape = is_static
            if self.session:
                gr.Info(
                    "Dimensions changed, restarting ORT inference session, first job may be slow. \
                    Disable `ORT Static Dimensions` from Settings if you expect to change image size/batch size frequently"
                )
            self.create_session()
        elif torch.cuda.current_stream().cuda_stream != self.last_cuda_stream:
            gr.Info("Cuda stream changed, restarting ORT inference session, first job may be slow.")
            print("cuda stream changed, restarting session")
            self.create_session()
        elif is_shape_changed or is_half_changed:
            if self.verbose:
                print("reset_io_binding")
            self.session.reset_io_binding()

        if input_shape != self.last_input_shape or is_half_changed:
            self.last_input_shape = input_shape
            self.last_half_batch_size = half_batch_size
            shape_dict = {
                "sample": list(feed_dict["sample"].shape),
                "timesteps": list(feed_dict["timesteps"].shape),
                "encoder_hidden_states": list(feed_dict["encoder_hidden_states"].shape),
                "latent": list(feed_dict["sample"].shape),
            }
            if "y" in feed_dict:
                shape_dict["y"] = list(feed_dict["y"].shape)

            self.session.allocate_buffers(shape_dict, half_batch_size)

            if self.verbose:
                print(shape_dict)

        inputs = {}
        for name, tensor in feed_dict.items():
            inputs[name] = tensor
        return self.session.infer(inputs, half_batch_size)

    def create_session(self):
        if self.verbose:
            print("create_session is called")
            print("devices.device", devices.device)

        device_id = torch.cuda.current_device()
        enable_cuda_graph = shared.opts.data.get("ort_static_dims", False)
        providers = [
            (
                "CUDAExecutionProvider",
                CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph, use_user_compute_stream=True),
            ),
            "CPUExecutionProvider",
        ]
        
        if self.verbose:
            print(providers)

        # Session Options
        self.sess_options = ort.SessionOptions()
        ort.set_default_logger_severity(3)
        # When the model has been optimized by onnxruntime, we can disable optimization to save session creation time.
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        onnx_path = self.get_onnx_path()
        if self.verbose:
            print("create session for", onnx_path, "with providers=", providers)

        session = ort.InferenceSession(onnx_path, providers=providers, sess_options=self.sess_options)
        device = torch.device("cuda", device_id)
        if self.session is not None:
            del self.session
        self.session = CudaSession(session, device, enable_cuda_graph)

        self.last_input_shape = None
        self.is_static_input_shape = enable_cuda_graph

        self.last_cuda_stream = torch.cuda.current_stream().cuda_stream
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        outputs = self.infer(feed_dict)

        out = outputs[self.session.output_names[0]]
        return out

    def switch_engine(self):
        if self.verbose:
            print("switch_engine is called")

        self.create_session()

    def activate(self):
        if self.verbose:
            print("activate is called")

        if self.session is None:
            self.create_session()

    def deactivate(self):
        if self.verbose:
            print("deactivate is called")

        if self.session is not None:
            del self.session
            self.session = None

        devices.torch_gc()

    def start_batch(self, batch_size: int, width: int, height: int, cfg_scale: float):
        self.batch_size = batch_size
        self.use_cfg1_opt = shared.opts.data.get("ort_cfg1_opt", False) and (cfg_scale <= 1.0)


class OnnxRuntimeScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        self.idx = None
        self.hr_idx = None
        self.torch_unet = False
        self.verbose = False

    def title(self):
        return "OnnxRuntime"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):  # 1
        # Check divisibilty
        if p.width % 64 or p.height % 64:
            gr.Error("Target resolution must be divisible by 64 in both dimensions.")

        if self.is_img2img:
            return
        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                gr.Error(
                    "HIRES Fix resolution must be divisible by 64 in both dimensions. Please change the upscale factor or disable HIRES Fix."
                )

    def get_profile_idx(self, p, model_name: str, model_type: ModelType) -> (int, int):
        best_hr = None

        if self.is_img2img:
            hr_scale = 1
        else:
            hr_scale = p.hr_scale if p.enable_hr else 1
        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            model_name,
            p.width,
            p.height,
            p.batch_size,
            77,
        )
        if len(valid_models) == 0:
            gr.Error(
                f"""No valid profile found for ({model_name}) LOWRES. Please go to the OnnxRuntime tab and generate an engine with the necessary profile.
                If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
            )
            return None, None
        best = idx[np.argmin(distances)]
        best_hr = best

        if hr_scale != 1:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                model_name,
                hr_w,
                hr_h,
                p.batch_size,
                77,
            )
            if len(valid_models_hr) == 0:
                gr.Error(
                    f"""No valid profile found for ({model_name}) HIRES. Please go to the OnnxRuntime tab and generate an engine with the necessary profile.
                    If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
                )
            merged_idx = [i for i, id in enumerate(idx) if id in idx_hr]
            if len(merged_idx) == 0:
                gr.Warning(
                    "No model available for both ({}) LOWRES ({}x{}) and HIRES ({}x{}). This will slow-down inference.".format(
                        model_name, p.width, p.height, hr_w, hr_h
                    )
                )
                return None, None
            else:
                _distances = [distances[i] for i in merged_idx]
                best_hr = merged_idx[np.argmin(_distances)]
                best = best_hr

        return best, best_hr

    def get_loras(self, p):
        if self.verbose:
            print("get_loras is called")

        # get lora from prompt
        _prompt = p.prompt
        extra_networks = re.findall(r"\<(.*?)\>", _prompt)
        loras = [net for net in extra_networks if net.startswith("lora")]

        if loras:
            print("Onnxruntime extension does not support LoRA. Falling back to torch.")
            self.torch_unet = True

    def process(self, p, *args):
        # before unet_init
        sd_unet_option = sd_unet.get_unet_option()
        if sd_unet_option is None:
            return

        model_name = p.sd_model_name

        if self.verbose:
            print(
                f"process sd_unet_option.model_name={sd_unet_option.model_name} p.sd_model_name={p.sd_model_name} p.sd_model_hash={p.sd_model_hash} sd_vae_name={p.sd_vae_name} sd_vae_hash={p.sd_vae_hash} cfg_scale={p.cfg_scale} refiner_checkpoint={p.refiner_checkpoint} args={args}"
            )

        if sd_unet_option.model_name != model_name:
            if p.sd_model_hash == shared.sd_model.sd_checkpoint_info.shorthash:
                # When the model is put in sub-directory like sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors, p.sd_model_name does not include sub-directory name
                model_name = shared.sd_model.sd_checkpoint_info.model_name
            else:
                gr.Error(
                    """Selected torch model ({}) does not match the selected OnnxRuntime U-Net ({}).
                    Please ensure that both models are the same or select Automatic from the SD UNet dropdown.""".format(
                        model_name, sd_unet_option.model_name
                    )
                )

        self.idx, self.hr_idx = self.get_profile_idx(p, model_name, ModelType.UNET)
        self.torch_unet = self.idx is None or self.hr_idx is None

        if self.verbose:
            print(f"idx={self.idx} hr_idx={self.hr_idx} torch_unet={self.torch_unet}")

        if not self.torch_unet:
            self.get_loras(p)

        self.apply_unet(sd_unet_option)

    def apply_unet(self, sd_unet_option):
        if self.verbose:
            print("apply_unet is called")
        if sd_unet_option == sd_unet.current_unet_option and sd_unet.current_unet is not None and not self.torch_unet:
            return

        if sd_unet.current_unet is not None:
            sd_unet.current_unet.deactivate()

        if self.torch_unet:
            print("Enabling PyTorch fallback as no engine was found.")
            gr.Warning("Enabling PyTorch fallback as no engine was found.")
            sd_unet.current_unet = None
            sd_unet.current_unet_option = sd_unet_option
            shared.sd_model.model.diffusion_model.to(devices.device)
            return
        else:
            shared.sd_model.model.diffusion_model.to(devices.cpu)
            devices.torch_gc()

        sd_unet.current_unet = sd_unet_option.create_unet()
        sd_unet.current_unet.profile_idx = self.idx
        sd_unet.current_unet.option = sd_unet_option
        sd_unet.current_unet_option = sd_unet_option

        print(f"Activating unet: {sd_unet.current_unet.option.label}")
        sd_unet.current_unet.activate()

    def process_batch(self, p, *args, **kwargs):
        if self.verbose:
            print(
                f"process_batch is called. batch_size={p.batch_size} w={p.width} height={p.height} steps={p.steps} cfg_scale={p.cfg_scale} sd_model_name={p.sd_model_name} args={args} kwargs={kwargs}"
            )

        if self.torch_unet or sd_unet.current_unet is None:
            return super().process_batch(p, *args, **kwargs)

        if self.idx != sd_unet.current_unet.profile_idx:
            sd_unet.current_unet.profile_idx = self.idx
            sd_unet.current_unet.switch_engine()

        if kwargs["batch_number"] == 0:
            sd_unet.current_unet.start_batch(p.batch_size, p.width, p.height, p.cfg_scale)

    def before_hr(self, p, *args):
        if self.verbose:
            print("before_hr is called. p.sd_model_name={p.sd_model_name} cfg_scale={p.cfg_scale} args={args}")

        if self.idx != self.hr_idx:
            sd_unet.current_unet.profile_idx = self.hr_idx
            sd_unet.current_unet.switch_engine()

        return super().before_hr(p, *args)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        if self.verbose:
            print(f"after_extra_networks_activate is called, torch_unet={self.torch_unet} args={args} kwargs={kwargs}")


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        l.append(OrtUnetOption(model_name=k, configs=v))


def on_ui_settings():
    section = ("onnxruntime", "OnnxRuntime")

    shared.opts.add_option(
        "ort_static_dims",
        shared.OptionInfo(
            default=False,
            label="ORT Static Dimensions",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "ort_cfg1_opt",
        shared.OptionInfo(
            default=True,
            label="ORT CFG Optimization",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_ort.on_ui_tabs)
