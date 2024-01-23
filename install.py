# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
from importlib.metadata import version

import launch
from modules import paths_internal

from modules import paths_internal
WHEEL_DIR = os.path.join(paths_internal.data_path, "wheel_cuda12")

def install():
    is_cuda12 = False
    if launch.is_installed("torch"):
        import torch

        if torch.version.cuda.startswith("12."):
            is_cuda12 = True

    if is_cuda12:
        if launch.is_installed("onnxruntime_gpu"):
            print("pip uninstall -y onnxruntime-gpu")
            launch.run(
                f'"{sys.executable}" -m pip uninstall -y onnxruntime-gpu',
                desc="onnxruntime-gpu is removed",
                errdesc="cannot uninstall onnxruntime-gpu",
                live=True,
            )

        for name in ["coloredlogs", "flatbuffers", "packaging", "protobuf", "sympy"]:
            if not launch.is_installed(name):
                print(f"pip install {name}")
                launch.run(
                    f'"{sys.executable}" -m pip {name}',
                    desc=f"{name} is installed",
                    errdesc=f"Couldn't install {name}",
                    live=True,
                )

        cuda12_wheel = os.path.join(WHEEL_DIR, "onnxruntime_gpu-1.18.0-cp310-cp310-linux_x86_64.whl")
        if os.path.exists(cuda12_wheel):
            name = "ort-nightly-gpu"
            if launch.is_installed(name):
                print(
                    f"pip uninstall -y {name}"
                )
                launch.run(
                    f"pip uninstall -y {name}",
                    desc=f"{name} is removed",
                    errdesc=f"cannot uninstall {name}",
                    live=True,
                )
                
            print(
                f"pip install {cuda12_wheel} --no-cache-dir --force-reinstall"
            )
            launch.run(
                f'"{sys.executable}" -m pip install {cuda12_wheel} --no-cache-dir',
                desc=f"{cuda12_wheel} is installed",
                errdesc=f"Couldn't install {cuda12_wheel}",
                live=True,
            )
        else:
            name = "ort-nightly-gpu"
            if not launch.is_installed(name):
                print(
                    f"pip install {name} --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/ --no-cache-dir"
                )
                launch.run(
                    f'"{sys.executable}" -m pip install {name} --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/ --no-cache-dir',
                    desc=f"{name} is installed",
                    errdesc=f"Couldn't install {name}",
                    live=True,
                )
    else:
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
