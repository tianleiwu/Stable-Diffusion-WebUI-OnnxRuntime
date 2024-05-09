# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import os
import shutil
import time
from logging import getLogger
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

from ort_model_config import ProfileSettings
from ort_model_helper import UNetModel
from ort_optimizer import OrtStableDiffusionOptimizer

logger = getLogger(__name__)


def swap_sdpa(func):
    def wrapper(*args, **kwargs):
        swap_sdpa = hasattr(F, "scaled_dot_product_attention")
        old_sdpa = getattr(F, "scaled_dot_product_attention", None) if swap_sdpa else None
        if swap_sdpa:
            delattr(F, "scaled_dot_product_attention")
        ret = func(*args, **kwargs)
        if swap_sdpa and old_sdpa:
            F.scaled_dot_product_attention = old_sdpa
        return ret

    return wrapper


def check_model_uses_external_data(onnx_model: onnx.ModelProto) -> bool:
    for initializer in onnx_model.graph.initializer:
        if initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
            return True
    return False


@swap_sdpa
def export_onnx(
    temp_dir: str,
    onnx_path: str,
    modelobj: UNetModel,
    profile: ProfileSettings,
    opset: int = 17,
    shape_inference: bool = False,
    force_export: bool = False,
):
    if os.path.exists(onnx_path):
        if not force_export:
            logger.info("Skip exporting to ONNX since %s exists.", onnx_path)
            onnx_model = onnx.load(onnx_path, load_external_data=False)
            modelobj.use_external_data = check_model_uses_external_data(onnx_model)
            del onnx_model
            return
        else:
            os.remove(onnx_path)
            if os.path.exists(onnx_path + ".data"):
                os.remove(onnx_path + ".data")

    s = time.time()

    print("-" * 40)
    print("Exporting to ONNX...")
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    model = modelobj.unet
    path = Path(onnx_path)
    os.makedirs(temp_dir, exist_ok=True)
    tmp_path = os.path.join(temp_dir, "model.onnx")

    try:
        logger.info("Exporting ONNX to %s", tmp_path)
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=modelobj.get_input_names(),
                output_names=modelobj.get_output_names(),
                dynamic_axes=modelobj.get_dynamic_axes(),
            )
    except Exception as e:
        logger.exception(e)
        return

    os.makedirs(path.parent, exist_ok=True)
    onnx_model = onnx.load(tmp_path, load_external_data=False)
    if modelobj.use_external_data or check_model_uses_external_data(onnx_model):
        if shape_inference:
            shape_onnx_path = os.path.join(temp_dir, "model_with_shape.onnx")
            logger.info(
                "Running shape inference and save onnx to a temporary file %s",
                shape_onnx_path,
            )
            onnx.shape_inference.infer_shapes_path(tmp_path, shape_onnx_path)

        logger.info("ONNX model uses external data. Saving as external data.")
        onnx_model = onnx.load(shape_onnx_path if shape_inference else tmp_path, load_external_data=True)
        onnx.save(
            onnx_model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".data",
            size_threshold=1024,
        )

        modelobj.use_external_data = True
    else:
        if shape_inference:
            onnx_model = shape_inference.infer_shapes(onnx_model)
            onnx.save(onnx_model, str(path))
        else:
            shutil.move(tmp_path, str(path))

    e = time.time()
    print(f"Exported ONNX {path} in {int(e-s)} seconds")
    print("-" * 40)

    # shutil.rmtree(temp_dir)
    del onnx_model


def optimize_onnx(
    temp_dir: str,
    optimized_onnx_path: str,
    input_onnx_path: str,
    use_fp16: bool,
    model_type: str = "unet",
    use_external_data: bool = False,
    force_optimize: bool = False,
):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    s = time.time()
    optimizer = OrtStableDiffusionOptimizer(model_type)
    logger.debug(f"use_fp16={use_fp16}")

    is_ok = True
    try:
        if not use_external_data:
            optimizer.optimize(
                input_onnx_path,
                optimized_onnx_path,
                tmp_dir=temp_dir,
                float16=use_fp16,
                keep_io_types=True,
                fp32_op_list=None,
                optimize_by_ort=True,
                optimize_by_fusion=True,
                use_external_data=use_external_data,
            )
        else:
            fusion_onnx_path = input_onnx_path[:-5] + "_fusion.onnx"
            if os.path.exists(fusion_onnx_path) and not force_optimize:
                print("skip fusion since path exists", fusion_onnx_path)
            else:
                optimizer.optimize_step1(
                    input_onnx_path,
                    fusion_onnx_path,
                    final_target_float16=use_fp16,
                    use_external_data=use_external_data,
                    float16=use_fp16,
                    keep_io_types=True,
                    fp32_op_list=None,
                )

            gc.collect()
            torch.cuda.empty_cache()

            optimizer.optimize_step2(
                fusion_onnx_path,
                optimized_onnx_path,
                use_external_data=use_external_data,
            )

            # After we have the optimized onnx, we can remove the temporary onnx file from step 1.
            os.remove(fusion_onnx_path)
            data_file = fusion_onnx_path + ".data"
            if os.path.exists(data_file):
                os.remove(data_file)

        e = time.time()
        print(f"Optimized onnx {optimized_onnx_path} in {int(e-s)} seconds")
        print("-" * 40)

        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.exception(e)
        is_ok = False

    return is_ok, optimizer.weight_packing_list
