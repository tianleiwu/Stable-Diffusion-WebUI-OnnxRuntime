# ONNX Runtime Extension for Stable Diffusion

This extension enables optimized execution of Stable Diffusion on Nividia RTX GPUs using [ONNX Runtime](https://onnxruntime.ai/) CUDA execution provider.
For non-CUDA compatible GPUs, please use [DirectML Extension for SD WebUI](https://github.com/microsoft/Stable-Diffusion-WebUI-DirectML) instead.

You need to install the extension and generate ONNX models before using the extension. Please follow the instructions below to set everything up. 

Supports Stable Diffusion 1.5, 2.1, SD-Turbo, SDXL 1.0, and SDXL-Turbo. For SDXL and SDXL Turbo, we recommend using a GPU with 12 GB or more VRAM for best performance.

LoRA and ControlNet are not supported at this time.

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

## More Information
This extension has confliction with the TensorRT extension. You can open `Extensions` tab and make sure only one of them is activated.

Image dimensions need to be specified as multiples of 64.

The first batch for a combination of image resolution and batch size will take longer. Additional batches will be much faster.

When `ORT Static Dimensions` is enabled, ONNX Runtime will enable CUDA graph to get better performance when image size or batch size are the same. However, if image size or batch size changes, ONNX Runtime will create a new session which causes extra latency in the first inference. If you enable `HiRes. fix` to upscale image, it is better to disable `ORT Static Dimensions` since image size changes in upscaling.

If you install dev branch of Automatic1111, PyTorch is for CUDA 12 by default. There is an option to use CUDA 11.8 by adding the following line to webui-user.bat:
```
set TORCH_COMMAND=pip install torch --index-url https://download.pytorch.org/whl/cu118
```
If you add this option, this extension will install onnxruntime-gpu, and you will need install CUDA 11.8 and latest cuDNN for CUDA 11.8 to your machine.
If you use CUDA 12, the extension will install ort-nightly-gpu and you will need install CUDA 12.* and latest cuDNN for CUDA 12.* in your machine.

For Turbo models, `ORT CFG Optimization` is enabled by default to get better performance for CFG Scale=1.0. To disable it, you can use Settings → User Interface → Quick Settings, and add `ort_cfg1_opt`. Click `Apply Settings` button, then `Reload UI` button. For SDXL-Turbo or SD-Turbo, it is recommended to try `Euler a` or `LCM` sampling method (in dev branch of A1111).
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
