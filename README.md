# ONNX Runtime Extension for Stable Diffusion

This extension enables optimized execution of Stable Diffusion on Nividia RTX GPUs using [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider.

You need to install the extension and generate ONNX models before using the extension. Please follow the instructions below to set everything up. 

Supports Stable Diffusion 1.5, 2.1, SD-Turbo, SDXL 1.0, and SDXL-Turbo. For SDXL and SDXL Turbo, we recommend using a GPU with 12 GB or more VRAM for best performance.

## Notes
For non-CUDA compatible GPU, please use [DirectML Extension for SD WebUI](https://github.com/microsoft/Stable-Diffusion-WebUI-DirectML) instead.

LoRA and ControlNet are not supported at this time.

Image dimensions need to be specified as multiples of 64.

If you use dev branch of Automatic1111, please add the following line to webui-user.bat to use Torch of cuda 11.8:

```
set TORCH_COMMAND=pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Installation
Example instructions for Automatic1111:

* Start the webui-user.bat
* Select the Extensions tab and click on Install from URL
* Copy the link to this repository and paste it into URL for extension's git repository
* Click Install

## How to use

1. Click `Export and Optimize ONNX` button under the `OnnxRuntime` tab to generate ONNX models.
2. Go to Settings → User Interface → Quick Settings List, add  `sd_unet` and `ort_static_dims`. Apply these settings, then reload the UI.
3. Back in the main UI, select `Automatic` or corresponding ORT model under `sd_unet` dropdown menu at the top of the page.
4. If your batch size, image width and height are not changed frequently, select the `ORT Static Dimensions` to get better performance.

You can now start generating images accelerated by ONNX Runtime. 

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
