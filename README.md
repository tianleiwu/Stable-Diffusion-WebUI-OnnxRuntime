# ONNX Runtime Extension for Automatic1111's SD WebUI

This extension enables optimized execution of Stable Diffusion UNet model on Nividia GPU.

For non-CUDA compatible GPU, please use [DirectML Extension for SD WebUI](https://github.com/microsoft/Stable-Diffusion-WebUI-DirectML) instead.

As a pre-requisite, the models need to be optimized through [Olive](https://github.com/microsoft/Olive) and added to the WebUI's model inventory, as described in the Setup
section. This extension uses [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider to run inference against these models.

Stable Diffusion versions 1.5 is supported. We are still working on 2.1, sd-turbo, SDXL 1.0 and SDXL-Turbo.

## Getting Started

1. Follow instructions [here](https://github.com/microsoft/Olive/tree/main/examples/stable_diffusion#prerequisitesn) to setup Olive.
2. Convert your SD model to ONNX, optimized by Olive, as described [here](https://github.com/microsoft/Olive/tree/main/examples/stable_diffusion#conversion-to-onnx-and-latency-optimization). The following commands are for 1.5, 2.1 and sd-turbo:
    ```
    python stable_diffusion.py --provider cuda --optimize
    python stable_diffusion.py --provider cuda --optimize --model_id stabilityai/stable-diffusion-2-1
    python stable_diffusion.py --provider cuda --optimize --model_id stabilityai/sd-turbo    
    ```
3. The optimized Unet model will be stored under `models\optimized-cuda\[model_id]\unet` (for example `models\optimized-cuda\runwayml\stable-diffusion-v1-5\unet\`). Copy this over, renaming to match the filename of the base SD WebUI model, to the WebUI's `models\Unet-ort` folder.
4. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
5. Back in the main UI, select the ORT Unet model from the sd_unet dropdown menu at the top of the page, and get going.
</ol>

## Notes
Image dimensions need to be specified as multiples of 64.

Stable Diffusion XL is not supported at this time, nor are LoRA/ControlNet.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
