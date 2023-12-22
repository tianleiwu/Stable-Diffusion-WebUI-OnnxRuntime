# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import launch
from importlib_metadata import version
from modules import paths_internal

def install():
    if launch.is_installed("onnxruntime_gpu"):
        if not version("onnxruntime_gpu") == "1.16.3":
            launch.run(["python","-m","pip","uninstall","-y","onnxruntime-gpu"], "removing old version of onnxruntime-gpu")
    
    if not launch.is_installed("onnxruntime_gpu"):
        print("onnxruntime-gpu is not installed! Installing...")
        #launch.run_pip("install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir", "nvidia-cudnn-cu11")
        #launch.run_pip("install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir", "tensorrt", live=True)
        #launch.run(["python","-m","pip","uninstall","-y","nvidia-cudnn-cu11"], "removing nvidia-cudnn-cu11")
        launch.run_pip("install onnxruntime-gpu==1.16.3 --no-cache-dir", "onnxruntime-gpu", live=True)

    ORT_UNET_MODEL_DIR = os.path.join(paths_internal.models_path, "UNET-ORT")
    if not os.path.exists(ORT_UNET_MODEL_DIR):
        os.makedirs(ORT_UNET_MODEL_DIR)

install()