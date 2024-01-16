# ONNX Runtime Extension

This extension enables optimized execution of Stable Diffusion UNet model on Nividia GPU.

This extension uses [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider to run inference against these models.

Stable Diffusion versions 1.5, 2.1, SDXL 1.0, sd-turbo, and SDXL-Turbo are supported.

LoRA and ControlNet are not supported at this time.

## Getting Started
1. Click `Export and Optimize ONNX` button under the `OnnxRuntime Exporter` tab to generate ONNX models.
2. Go to Settings → User Interface → Quick Settings List, add  `sd_unet` and `ort_static_dims`. Apply these settings, then reload the UI.
3. Back in the main UI, select `Automatic` or the corresponding ORT model under `sd_unet` dropdown menu at the top of the page
4. If your batch size, image width and height does not change frequently, check `ORT Static Dimensions` in quick setting to get better performance.
</ol>