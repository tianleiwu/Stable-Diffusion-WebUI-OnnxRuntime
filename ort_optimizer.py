# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
ONNX Model Optimizer for Stable Diffusion
"""

import gc
import logging
from pathlib import Path

import onnx
import psutil
import torch
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

from unet_fusion_tracer import UnetFusionTracer

logger = logging.getLogger(__name__)


class OrtStableDiffusionOptimizer:
    def __init__(self, model_type: str = "unet"):
        assert model_type in ["unet"]
        self.model_type = model_type
        self.model_type_class_mapping = {
            "unet": UnetFusionTracer,
        }
        self.weight_packing_list = []

    def print_memory_usage(self, message):
        gb = float(1024**3)
        free, total = torch.cuda.mem_get_info()
        m = psutil.virtual_memory()
        logger.info(
            "%s: free gpu memory %.1f out of %.1f GB; free cpu memory %.1f out of %.1f GB ",
            message,
            free / gb,
            total / gb,
            m.free / gb,
            m.total / gb,
        )

    def optimize_by_ort(self, onnx_model, use_external_data_format, tmp_dir):
        # Save to a temporary file so that we can load it with Onnx Runtime.
        logger.info("Saving a temporary model to run OnnxRuntime graph optimizations...")
        tmp_model_path = Path(tmp_dir) / "model.onnx"
        onnx_model.save_model_to_file(str(tmp_model_path), use_external_data_format=use_external_data_format)

        del onnx_model
        gc.collect()

        ort_optimized_model_path = Path(tmp_dir) / "optimized.onnx"

        # Disable ConstantFolding to avoid folding the Transpose node and Conv weights, so that we can preserve the weights for LoRA refit.
        optimize_by_onnxruntime(
            onnx_model_path=str(tmp_model_path),
            use_gpu=True,
            optimized_model_path=str(ort_optimized_model_path),
            save_as_external_data=use_external_data_format,
            external_data_filename="optimized.onnx.data",
            provider="cuda",
            disabled_optimizers=["ConstantFolding"],
        )
        model = onnx.load(str(ort_optimized_model_path), load_external_data=True)
        return self.model_type_class_mapping[self.model_type](model)

    def optimize(
        self,
        input_fp32_onnx_path,
        optimized_onnx_path,
        tmp_dir,
        float16=True,
        keep_io_types=False,
        fp32_op_list=None,
        keep_outputs=None,
        optimize_by_ort=False,
        optimize_by_fusion=True,
        final_target_float16=True,
        use_external_data=True,
    ):
        """Optimize onnx model using ONNX Runtime transformers optimizer"""
        logger.info(f"Optimize {input_fp32_onnx_path}...")

        self.print_memory_usage("before optimize")

        model = onnx.load_model(input_fp32_onnx_path, load_external_data=True)
        m = self.model_type_class_mapping[self.model_type](model)

        if optimize_by_fusion:
            fusion_options = FusionOptions(self.model_type)

            if self.model_type in ["unet"] and not final_target_float16:
                fusion_options.enable_packed_kv = False
                fusion_options.enable_packed_qkv = False

            m.optimize(fusion_options)
            m.topological_sort()
            m.model.producer_name = "onnxruntime.transformers"

        if self.model_type == "unet":
            self.weight_packing_list = m.weight_packing_list

        if keep_outputs:
            m.prune_graph(outputs=keep_outputs)

        if optimize_by_ort:
            m = self.optimize_by_ort(m, use_external_data_format=use_external_data, tmp_dir=tmp_dir)

        if float16:
            m.convert_float_to_float16(
                keep_io_types=keep_io_types,
                op_block_list=fp32_op_list,
            )

        m.get_operator_statistics()
        m.get_fused_operator_statistics()
        m.save_model_to_file(optimized_onnx_path, use_external_data_format=use_external_data)
        logger.info(f"{self.model_type} is optimized: {optimized_onnx_path}")

        del model
        del m
        gc.collect()

    def optimize_step1(
        self,
        input_fp32_onnx_path,
        fusion_onnx_path,
        keep_outputs=None,
        final_target_float16=True,
        use_external_data=True,
        float16=True,
        keep_io_types=False,
        fp32_op_list=None,
    ):
        """Optimize onnx model using ONNX Runtime transformers optimizer"""
        logger.info(f"Optimize {input_fp32_onnx_path}...")

        self.print_memory_usage("before optimize")

        model = onnx.load_model(input_fp32_onnx_path, load_external_data=True)
        m = self.model_type_class_mapping[self.model_type](model)

        fusion_options = FusionOptions(self.model_type)

        # Shape inference has been done after onnx export. No need to run symbolic shape inference again.
        # fusion_options.enable_shape_inference = False

        # It is allowed float16=False and final_target_float16=True, for using fp32 as intermediate optimization step.
        # For rare fp32 use case, we can disable packed kv/qkv since there is no fp32 TRT fused attention kernel.

        # Disable packing for lora refit, or when optimized model is fp32
        if (self.model_type in ["unet"] and not final_target_float16):
            fusion_options.enable_packed_kv = False
            fusion_options.enable_packed_qkv = False

        m.optimize(fusion_options)
        self.weight_packing_list = m.weight_packing_list

        m.topological_sort()
        m.model.producer_name = "onnxruntime.transformers"

        if keep_outputs:
            m.prune_graph(outputs=keep_outputs)

        if float16:
            m.convert_float_to_float16(
                keep_io_types=keep_io_types,
                op_block_list=fp32_op_list,
            )

        m.save_model_to_file(fusion_onnx_path, use_external_data_format=use_external_data)

        m.get_operator_statistics()
        m.get_fused_operator_statistics()

        print(f"{self.model_type} after optimizing by fusion: {fusion_onnx_path}")

    def optimize_step2(self, fusion_onnx_path, optimized_onnx_path, use_external_data):
        optimize_by_onnxruntime(
            str(fusion_onnx_path),
            use_gpu=True,
            optimized_model_path=str(optimized_onnx_path),
            save_as_external_data=use_external_data,
            external_data_filename=Path(optimized_onnx_path).name + "data",
            disabled_optimizers=["ConstantFolding"],
            provider="cuda",
        )

        logger.info(f"{self.model_type} is optimized: {optimized_onnx_path}")
