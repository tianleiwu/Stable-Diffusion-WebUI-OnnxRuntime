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


def pip_install(name):
    if not launch.is_installed(name.replace("-", "_")):
        print(f"pip install {name}")
        launch.run(
            f'"{sys.executable}" -m pip {name} --no-cache-dir',
            desc=f"{name} is installed",
            errdesc=f"Couldn't install {name}",
            live=True,
        )


def pip_install_ort_cuda_12_wheel():
    # You can put onnxruntime-gpu wheel to ..\..\wheel_cuda12\ relative to this file for testing purpose.
    dir = os.path.join(paths_internal.data_path, "wheel_cuda12")
    cuda12_wheel = os.path.join(
        dir,
        "onnxruntime_gpu-1.18.0-cp310-cp310-linux_x86_64.whl"
        if sys.platform == "linux"
        else "onnxruntime_gpu-1.18.0-cp310-cp310-win_amd64.whl",
    )
    if os.path.exists(cuda12_wheel):
        pip_uninstall("ort-nightly-gpu")

        print(f"pip install {cuda12_wheel} --no-cache-dir --force-reinstall")
        launch.run(
            f'"{sys.executable}" -m pip install {cuda12_wheel} --no-cache-dir',
            desc=f"{cuda12_wheel} is installed",
            errdesc=f"Couldn't install {cuda12_wheel}",
            live=True,
        )
        return True

    return False


def pip_install_ort_nightly_cuda_12():
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


def install():
    is_cuda12 = False
    if launch.is_installed("torch"):
        import torch

        if torch.version.cuda.startswith("12."):
            is_cuda12 = True

    if is_cuda12:
        pip_uninstall("onnxruntime-gpu")

        for name in ["coloredlogs", "flatbuffers", "packaging", "protobuf", "sympy"]:
            pip_install(name)

        if not pip_install_ort_cuda_12_wheel():
            pip_install_ort_nightly_cuda_12()
    else:
        ort_version = "1.17.0"
        if launch.is_installed("onnxruntime_gpu"):
            current_version = version("onnxruntime_gpu")
            if current_version != ort_version:
                pip_uninstall("onnxruntime_gpu")

        pip_install("onnxruntime-gpu")

    pip_install("onnx")
    dir = os.path.join(paths_internal.models_path, "Unet-ort")
    if not os.path.exists(dir):
        os.makedirs(dir)


install()
