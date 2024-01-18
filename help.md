# ONNX Runtime Extension

This extension enables optimized execution of Stable Diffusion on Nividia RTX GPUs using [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider.

Stable Diffusion versions 1.5, 2.1, SDXL 1.0, sd-turbo, and SDXL-Turbo are supported.

LoRA and ControlNet are not supported at this time.

## Getting Started
1. Click `Export and Optimize ONNX` button under the `OnnxRuntime Exporter` tab to generate ONNX models.
2. Go to Settings → User Interface → Quick Settings List, add  `sd_unet` and `ort_static_dims`. Click `Apply Settings` button, then `Reload UI` button.
3. Back in the main UI, select `Automatic` under `sd_unet` dropdown menu at the top of the page.
4. If your batch size, image width and height does not change frequently, check `ORT Static Dimensions` in quick setting to get better performance.

You can now start generating images accelerated by ONNX Runtime.

## More Information
This extension has conflict with the TensorRT extension. You can open `Extensions` tab and make sure only one of them is activated.

Image dimensions need to be specified as multiples of 64.

The first batch for a combination of image resolution and batch size will take longer. Additional batches will be much faster.

When `ORT Static Dimensions` is enabled, ONNX Runtime will enable CUDA graph to get better performance when image size or batch size are the same. However, if image size or batch size changes, ONNX Runtime will create a new session which causes extra latency in the first inference. If you enable `HiRes. fix` to upscale image, it is better to disable `ORT Static Dimensions` since image size changes in upscaling.

If you install dev branch of Automatic1111, please add the following line to webui-user.bat to use Torch of cuda 11.8 since installed onnxruntime-gpu does not support CUDA 12:
```
set TORCH_COMMAND=pip install torch --index-url https://download.pytorch.org/whl/cu118
```

For Turbo models, For Turbo, it is recommended to use CFG Scale=1.0 and `LCM` sampling method in dev branch of A1111. `ORT CFG Optimization` is enabled by default to get better performance for CFG Scale = 1.0. To eanble/disable it, you can use Settings → User Interface → Quick Settings, and add `ort_cfg1_opt`. Click `Apply Settings` button, then `Reload UI` button.


For more information, please visit the ONNX Runtime Extension GitHub page [here](https://github.com/tianleiwu/Stable-Diffusion-WebUI-OnnxRuntime).
</ol>