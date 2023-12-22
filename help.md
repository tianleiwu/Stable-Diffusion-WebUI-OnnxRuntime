# ONNX Runtime Extension for Automatic1111's SD WebUI

This extension enables optimized execution of Stable Diffusion UNet model on Nividia GPU.

As a pre-requisite, the base models need to be optimized through [Olive](https://github.com/microsoft/Olive) and added to the WebUI's model inventory, as described in the Setup
section. The extension uses [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider to run inference against these models.

Stable Diffusion versions 1.5, 2.0 and 2.1 are supported.

## Getting Started

1. Follow instructions [here](https://github.com/microsoft/Olive/tree/main/examples/stable_diffusion#prerequisitesn) to setup Olive.
2. Convert your SD model to ONNX, optimized by Olive, as described [here](https://github.com/microsoft/Olive/tree/main/examples/stable_diffusion#conversion-to-onnx-and-latency-optimization). For the simplest case (the v1.5 safetensors downloaded during WebUI setup), the following command will suffice:
    ```
    python stable_diffusion.py --provider cuda --optimize
    python stable_diffusion.py --provider cuda --model_id stabilityai/sd-turbo --optimize
    ```
3. The optimized Unet model will be stored under `\models\optimized\[model_id]\unet` (for example `\models\optimized\runwayml\stable-diffusion-v1-5\unet`). Copy this over, renaming to match the filename of the base SD WebUI model, to the WebUI's `models\Unet-dml` folder.
4. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
5. Back in the main UI, select the DML Unet model from the sd_unet dropdown menu at the top of the page, and get going.
</ol>