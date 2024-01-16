# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
from importlib.metadata import version

import launch
from modules import paths_internal


def install():
    ort_version = "1.16.3"
    if launch.is_installed("onnxruntime_gpu"):
        current_version = version("onnxruntime_gpu")
        if current_version != ort_version:
            launch.run(
                f'"{sys.executable}" -m pip uninstall -y onnxruntime-gpu',
                desc=f"onnxruntime-gpu {current_version} is removed",
                errdesc="cannot uninstall onnxruntime-gpu {current_version}",
                live=True,
            )

    if not launch.is_installed("onnxruntime_gpu"):
        ort = "onnxruntime_gpu"
        launch.run(
            f'"{sys.executable}" -m pip install {ort} --no-cache-dir',
            desc="onnxruntime-gpu is installed",
            errdesc=f"Couldn't install {ort}",
            live=True,
        )

    if not launch.is_installed("onnx"):
        print("Onnx is not installed! Installing...")
        launch.run_pip(
            "install onnx",
            "onnx",
            live=True,
        )

    dir = os.path.join(paths_internal.models_path, "Unet-ort")
    if not os.path.exists(dir):
        os.makedirs(dir)


install()
