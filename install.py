# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
from importlib.metadata import version

import launch
from modules import paths_internal


def pip_uninstall(name):
    if launch.is_installed(name.replace("-", "_")):
        print(f"pip uninstall -y {name}")
        launch.run(
            f'"{sys.executable}" -m pip uninstall -y {name} --no-cache-dir',
            desc=f"{name} is removed",
            errdesc=f"cannot uninstall {name}",
            live=True,
        )


def pip_install(name, options="--no-cache-dir"):
    if not launch.is_installed(name.replace("-", "_")):
        print(f"pip install {name}")
        launch.run(
            f'"{sys.executable}" -m pip install {name} {options}',
            desc=f"{name} is installed",
            errdesc=f"Couldn't install {name}",
            live=True,
        )


def pip_install_ort_wheel(version="1.18.0"):
    # You can put onnxruntime-gpu CUDA 12 wheel to ..\..\wheel\ relative to this file for testing purpose.
    dir = os.path.join(paths_internal.data_path, "wheel")
    ort_wheel = os.path.join(
        dir,
        f"onnxruntime_gpu-{version}-cp310-cp310-linux_x86_64.whl"
        if sys.platform == "linux"
        else f"onnxruntime_gpu-{version}-cp310-cp310-win_amd64.whl",
    )
    if os.path.exists(ort_wheel):
        pip_uninstall("ort-nightly-gpu")

        print(f"pip install {ort_wheel} --no-cache-dir --force-reinstall")
        launch.run(
            f'"{sys.executable}" -m pip install {ort_wheel} --no-cache-dir',
            desc=f"{ort_wheel} is installed",
            errdesc=f"Couldn't install {ort_wheel}",
            live=True,
        )
        return True

    return False


def install(force_reinstall=False, local_wheel=False):
    import torch
    if not torch.cuda.is_available():
        print(
            "Torch CUDA is not available! Please install Torch with CUDA and try again."
        )
        return
    
    is_cuda12 = False
    if launch.is_installed("torch"):
        import torch

        if torch.version.cuda.startswith("12."):
            is_cuda12 = True

    ort_version = "1.17.0"
    if launch.is_installed("onnxruntime_gpu"):
        current_version = version("onnxruntime_gpu")
        if force_reinstall or current_version != ort_version:
            pip_uninstall("onnxruntime-gpu")

    if not launch.is_installed("onnxruntime_gpu"):
        if is_cuda12:
            if not (local_wheel and pip_install_ort_wheel()):
                pip_install(
                    "onnxruntime-gpu",
                    options="--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ --no-cache-dir",
                )
        else:
            pip_install("onnxruntime-gpu")

    if not launch.is_installed("onnx"):
        pip_install("onnx")

    dir = os.path.join(paths_internal.models_path, "Unet-ort")
    if not os.path.exists(dir):
        os.makedirs(dir)


# Only need install once.
install()
