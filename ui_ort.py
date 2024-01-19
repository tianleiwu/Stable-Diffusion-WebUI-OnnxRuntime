# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
from collections import defaultdict

import gradio as gr
import torch
from modules import sd_hijack, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import FormRow, ToolButton

from ort_exporter import export_onnx, optimize_onnx
from ort_model_config import ProfilePrests
from ort_model_helper import UNetModel
from ort_model_manager import cc_major, cc_minor, modelmanager

logger = logging.getLogger(__name__)

profile_presets = ProfilePrests()

logging.basicConfig(level=logging.INFO)


def get_context_dim():
    if shared.sd_model.is_sd1:
        return 768
    elif shared.sd_model.is_sd2:
        return 1024
    elif shared.sd_model.is_sdxl:
        return 2048


def is_fp32():
    use_fp32 = False
    if cc_major * 10 + cc_minor < 61:
        use_fp32 = True
        print("FP16 has been disabled because your GPU does not support it.")
    return use_fp32


def export_unet_to_ort(force_export):
    sd_hijack.model_hijack.apply_optimizations("None")

    logging.basicConfig(level=logging.INFO)

    is_xl = shared.sd_model.is_sdxl
    info = shared.sd_model.sd_checkpoint_info

    # Assume that we put model under a subdirectory like sd/v1-5-pruned-emaonly.safetensors, the information is like the following:
    #   shorttitle=v1-5-pruned-emaonly [6ce0161689] shorthash=6ce0161689 model_name=sd_v1-5-pruned-emaonly hash=d7049739 name=sd/v1-5-pruned-emaonly.safetensors name_for_extra=v1-5-pruned-emaonly
    # Here we choose the name and hash to match the short title shown in the checkpoint list. Another benefit is that, when checkpoint is moved to sub-directory, onnx model name does not change.
    model_name = info.name_for_extra
    model_hash = info.shorthash
    logger.debug(
        f"Checkpoint info: shorttitle={info.short_title} shorthash={info.shorthash} model_name={info.model_name} hash={info.hash} name={info.name} name_for_extra={info.name_for_extra}"
    )

    use_fp32 = is_fp32()
    profile_settings = profile_presets.get_default(is_xl=is_xl)
    profile_settings.token_to_dim()
    print(f"Exporting checkpoint {info.short_title} to OnnxRuntime using profile setting: {profile_settings}")

    onnx_filename, onnx_path = modelmanager.get_onnx_path(model_name, model_hash)
    embedding_dim = get_context_dim()

    modelobj = UNetModel(
        shared.sd_model.model.diffusion_model,
        embedding_dim,
        is_xl=is_xl,
    )
    modelobj.apply_torch_model()

    export_onnx(
        onnx_path,
        modelobj,
        profile_settings,
    )
    gc.collect()
    torch.cuda.empty_cache()

    ort_engine_filename, ort_engine_path = modelmanager.get_engine_path(model_name, model_hash)

    if not os.path.exists(ort_engine_path) or force_export:
        print("Optimize ONNX for OnnxRuntime... This can take a while, please check the progress in the terminal.")
        gr.Info("Optimize ONNX for OnnxRuntime... This can take a while, please check the progress in the terminal.")

        # Unload model to CPU to free GPU memory so that ORT has enough memory to create session in GPU.
        free, total = torch.cuda.mem_get_info()
        need_unload_model_to_cpu = free < 12 * float(1024**3)
        if need_unload_model_to_cpu:
            model = shared.sd_model.cpu()
            torch.cuda.empty_cache()

        is_ok = optimize_onnx(
            ort_engine_path,
            onnx_path,
            use_fp16=not use_fp32,
            use_external_data=modelobj.use_external_data,
        )

        if need_unload_model_to_cpu:
            shared.sd_model = model.cuda()

        if not is_ok:
            return "## Export Failed due to unknown reason. See shell for more information. \n"
    else:
        print(
            "OnnxRuntime engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed."
        )

    # Allow it to add to JSON for recovering json file.
    profile = modelobj.get_input_profile(profile_settings)

    modelmanager.add_entry(
        info.model_name,  # pass the full model name to match in "Automatic": https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cf2772fab0af5573da775e7437e6acdca424f26e/modules/sd_unet.py#L24
        ort_engine_filename,
        profile,
        fp32=use_fp32,
        inpaint=modelobj.in_channels == 6,
        unet_hidden_dim=modelobj.in_channels,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return "## Exported Successfully \n"


def engine_profile_card():
    def get_md_table(
        h_min,
        h_opt,
        h_max,
        w_min,
        w_opt,
        w_max,
        b_min,
        b_opt,
        b_max,
        t_min,
        t_opt,
        t_max,
    ):
        md_table = (
            "|             	|   Min   	|   Opt   	|   Max   	| \n"
            "|-------------	|:-------:	|:-------:	|:-------:	| \n"
            "| Height      	| {h_min} 	| {h_opt} 	| {h_max} 	| \n"
            "| Width       	| {w_min} 	| {w_opt} 	| {w_max} 	| \n"
            "| Batch Size  	| {b_min} 	| {b_opt} 	| {b_max} 	| \n"
            "| Text-length 	| {t_min} 	| {t_opt} 	| {t_max} 	| \n"
        )

        return md_table.format(
            h_min=h_min,
            h_opt=h_opt,
            h_max=h_max,
            w_min=w_min,
            w_opt=w_opt,
            w_max=w_max,
            b_min=b_min,
            b_opt=b_opt,
            b_max=b_max,
            t_min=t_min,
            t_opt=t_opt,
            t_max=t_max,
        )

    available_models = modelmanager.available_models()

    model_md = defaultdict(list)
    for base_model, models in available_models.items():
        for _i, m in enumerate(models):
            assert isinstance(m, dict)
            s_min, s_opt, s_max = m["config"].profile.get("sample", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            t_min, t_opt, t_max = m["config"].profile.get("encoder_hidden_states", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            profile_table = get_md_table(
                s_min[2] * 8,
                s_opt[2] * 8,
                s_max[2] * 8,
                s_min[3] * 8,
                s_opt[3] * 8,
                s_max[3] * 8,
                max(s_min[0] // 2, 1),
                max(s_opt[0] // 2, 1),
                max(s_max[0] // 2, 1),
                (t_min[1] // 77) * 75,
                (t_opt[1] // 77) * 75,
                (t_max[1] // 77) * 75,
            )

            model_md[base_model].append(profile_table)

    return model_md


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ort_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"), gr.Tabs(elem_id="ort_tabs"):
                with gr.Tab(label="OnnxRuntime Exporter"):
                    gr.Markdown(
                        value="# Build ONNX model for CUDA",
                    )

                    with FormRow(elem_classes="checkboxes-row", variant="compact"):
                        force_rebuild = gr.Checkbox(
                            label="Force Rebuild.",
                            value=False,
                            elem_id="ort_force_rebuild",
                        )

                    button_export_default_unet = gr.Button(
                        value="Export and Optimize ONNX",
                        variant="primary",
                        elem_id="ort_export_default_unet",
                        visible=True,
                    )

            with gr.Column(variant="panel"), open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "help.md"),
                encoding="utf-8",
            ) as f:
                gr.Markdown(elem_id="ort_info", value=f.read())

        with gr.Row(equal_height=False), gr.Accordion("Output", open=True):
            ort_result = gr.Markdown(elem_id="ort_result", value="")

        def get_ort_profiles_markdown():
            profiles_md_string = ""
            engine_cards = engine_profile_card()
            for model, profiles in engine_cards.items():
                profiles_md_string += f"<details><summary>{model} ({len(profiles)} Profiles)</summary>\n\n"
                for i, profile in enumerate(profiles):
                    profiles_md_string += f"#### Profile {i} \n{profile}\n\n"
                profiles_md_string += "</details>\n"
            profiles_md_string += "</details>\n"

            return profiles_md_string

        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True, variant="compact"):
                button_refresh_profiles = ToolButton(value=refresh_symbol, elem_id="ort_refresh_profiles", visible=True)
                gr.Markdown(value="## Available ONNX models for CUDA Engine")

            with gr.Row(equal_height=True):
                ort_profiles_markdown = gr.Markdown(elem_id="ort_profiles_markdown", value=get_ort_profiles_markdown())

        button_refresh_profiles.click(
            lambda: gr.Markdown.update(value=get_ort_profiles_markdown()),
            outputs=[ort_profiles_markdown],
        )

        button_export_default_unet.click(
            export_unet_to_ort,
            inputs=[
                force_rebuild,
            ],
            outputs=[ort_result],
        )

    return [(ort_interface, "OnnxRuntime", "onnxruntime")]
