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
from safetensors.torch import save_file

from ort_exporter import export_onnx, optimize_onnx
from ort_lora import export_lora, export_weights_map, get_lora_checkpoints, save_packing_source_tensors
from ort_model_config import ProfilePrests, SDVersion
from ort_model_helper import UNetModel
from ort_model_manager import ONNX_MODEL_DIR, ORT_MODEL_DIR, cc_major, cc_minor, modelmanager

logger = logging.getLogger(__name__)

profile_presets = ProfilePrests()


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


def export_unet_to_ort(force_export, force_optimize):
    sd_hijack.model_hijack.apply_optimizations("None")
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

    onnx_path = modelmanager.get_onnx_path(model_name, model_hash)
    embedding_dim = get_context_dim()

    unet_model = UNetModel(
        shared.sd_model.model.diffusion_model,
        embedding_dim,
        is_xl=is_xl,
    )
    unet_model.apply_torch_model()

    temp_dir = os.path.join(ONNX_MODEL_DIR, "temp_export")

    export_onnx(
        temp_dir,
        onnx_path,
        unet_model,
        profile_settings,
        force_export=force_export,
    )
    gc.collect()
    torch.cuda.empty_cache()

    ort_engine_filename, ort_engine_path = modelmanager.get_engine_path(model_name, model_hash, provider=None)
    weight_map_path = modelmanager.get_weights_map_path(model_name, model_hash, provider=None)
    # weight_packing_path = modelmanager.get_weights_packing_path(model_name, model_hash, provider=None)
    packing_source_tensors_path = modelmanager.get_packing_source_tensors_path(model_name, model_hash, provider=None)

    if not (os.path.exists(ort_engine_path) and os.path.exists(weight_map_path)) or force_optimize:
        print("Optimize ONNX for OnnxRuntime... This can take a while.")
        gr.Info("Optimize ONNX for OnnxRuntime... This can take a while, please check the progress in the terminal.")

        # Unload model to CPU to free GPU memory so that ORT has enough memory to create session in GPU.
        free, total = torch.cuda.mem_get_info()
        need_unload_model_to_cpu = free < 12 * float(1024**3)
        if need_unload_model_to_cpu:
            model = shared.sd_model.cpu()
            torch.cuda.empty_cache()

        temp_dir = os.path.join(ONNX_MODEL_DIR, "temp_optimize")
        is_ok, weight_packing_list = optimize_onnx(
            temp_dir,
            ort_engine_path,
            onnx_path,
            use_fp16=not use_fp32,
            use_external_data=unet_model.use_external_data,
            force_optimize=force_optimize,
        )

        # Save the weight packing list for LoRA weight refitting later.
        # with open(weight_packing_path, "w") as fp:
        #     json.dump(weight_packing_list, fp)
        #     print(f"Saved weight packing list to {weight_packing_path}")

        if need_unload_model_to_cpu:
            shared.sd_model = model.cuda()

        if not is_ok:
            return "## Export Failed due to unknown reason. See shell for more information. \n"
    else:
        print(
            "OnnxRuntime engine found. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed."
        )

        # with open(weight_packing_path) as fp_weight_packing:
        #     print(f"[I] Loading weight packing: {weight_packing_path} ")
        #     weight_packing_list = json.load(fp_weight_packing)

    if os.path.exists(packing_source_tensors_path):
        os.remove(packing_source_tensors_path)
    logger.info("export packing source tensors from onnx %s to %s", onnx_path, packing_source_tensors_path)
    save_packing_source_tensors(weight_packing_list, onnx_path, packing_source_tensors_path)

    if os.path.exists(weight_map_path):
        os.remove(weight_map_path)
    logger.info("export weights map from onnx %s to %s", ort_engine_path, weight_map_path)
    export_weights_map(unet_model, ort_engine_path, weight_map_path, weight_packing_list)

    # Export raw weights map for testing
    # raw_weight_map_path = modelmanager.get_weights_map_path(model_name, model_hash, provider="raw")
    # if os.path.exists(raw_weight_map_path):
    #     os.remove(raw_weight_map_path)
    # if not os.path.exists(raw_weight_map_path):
    #     logger.info("export raw weights map from onnx %s to %s", onnx_path, raw_weight_map_path)
    #     export_weights_map(unet_model, onnx_path, raw_weight_map_path, [])

    # Allow it to add to JSON for recovering json file.
    profile = unet_model.get_input_profile(profile_settings)

    modelmanager.add_entry(
        info.model_name,  # pass the full model name to match in "Automatic": https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cf2772fab0af5573da775e7437e6acdca424f26e/modules/sd_unet.py#L24
        ort_engine_filename,
        profile,
        fp32=use_fp32,
        inpaint=unet_model.in_channels == 6,
        unet_hidden_dim=unet_model.in_channels,
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


# LoRA
def export_lora_to_ort(lora_name, force_export):
    logger.debug("export_lora_to_ort is called. lora_name=%s, force_export=%s", lora_name, force_export)
    if isinstance(lora_name, list):
        return "## Please select only one Lora to export"

    is_xl = shared.sd_model.is_sdxl
    available_lora_models = get_lora_checkpoints()

    lora_name = lora_name.split(" ")[0]
    lora_model = available_lora_models.get(lora_name, None)
    logger.debug("lora_name=%s, lora_model=%s", lora_name, lora_model)

    if lora_model is None:
        return f"## No LoRA model found for {lora_name}"

    version = lora_model.get("version", SDVersion.Unknown)
    if version == SDVersion.Unknown:
        logger.info("LoRA SD version couldn't be determined. Please ensure the correct SD Checkpoint is selected.")

    model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    model_hash = shared.sd_model.sd_checkpoint_info.shorthash
    logger.debug("model_name=%s, model_hash=%s", model_name, model_hash)

    if not version.match(shared.sd_model):
        print(
            f"""LoRA SD version ({version}) does not match the current SD version ({model_name}). Please ensure the correct SD Checkpoint is selected."""
        )

    profile_settings = profile_presets.get_default(is_xl=is_xl)
    logger.info("Exporting %s to OnnxRuntime using - %s", lora_name, profile_settings)
    profile_settings.token_to_dim()

    # raw_onnx_path = modelmanager.get_onnx_path(model_name, model_hash)

    _, onnx_opt_path = modelmanager.get_engine_path(model_name, model_hash)
    if not os.path.exists(onnx_opt_path):
        return f"## Please export the base model ({model_name} [{model_hash}]) first."

    embedding_dim = get_context_dim()

    unet_model = UNetModel(
        shared.sd_model.model.diffusion_model,
        embedding_dim,
        is_xl=is_xl,
    )
    unet_model.apply_torch_model()

    weights_map_path = modelmanager.get_weights_map_path(model_name, model_hash, provider=None)
    if not os.path.exists(weights_map_path):
        logger.info("export weights map from onnx %s to %s", onnx_opt_path, weights_map_path)
        export_weights_map(onnx_opt_path, weights_map_path)

    # Export raw weights map for comparison
    # raw_weights_map_path = modelmanager.get_weights_map_path(model_name, model_hash, provider="raw")
    # if not os.path.exists(raw_weights_map_path):
    #     logger.info("export raw weights map from onnx %s to %s", raw_onnx_path, weights_map_path)
    #     export_weights_map(raw_onnx_path, raw_weights_map_path)

    lora_ort_name = f"{lora_name}.lora"
    lora_ort_path = os.path.join(ORT_MODEL_DIR, lora_ort_name)

    if os.path.exists(lora_ort_path) and not force_export:
        logger.info(
            "File found: %s. Skipping build. You can enable Force Export in the Advanced Settings to force a rebuild if needed.",
            lora_ort_path,
        )
        return "## Exported Successfully \n"

    delta_dict = export_lora(
        unet_model,
        onnx_opt_path,
        weights_map_path,
        packing_source_tensors_path=modelmanager.get_packing_source_tensors_path(model_name, model_hash, provider=None),
        lora_name=lora_model["filename"],
        profile=profile_settings,
    )

    save_file(delta_dict, lora_ort_path)

    return "## Exported Successfully \n"


def get_valid_lora_checkpoints():
    available_lora_models = get_lora_checkpoints()
    return [f"{k} ({v['version']})" for k, v in available_lora_models.items()]


def disable_lora_export(lora):
    if lora is None:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ort_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="ort_tabs"):
                    with gr.Tab(label="OnnxRuntime Exporter"):
                        gr.Markdown(
                            value="# Build ONNX model for CUDA",
                        )

                        with FormRow(elem_classes="checkboxes-row-1", variant="compact"):
                            force_export = gr.Checkbox(
                                label="Force Export",
                                value=False,
                                elem_id="ort_force_export",
                            )
                        with FormRow(elem_classes="checkboxes-row-2", variant="compact"):
                            force_optimize = gr.Checkbox(
                                label="Force Optimize",
                                value=False,
                                elem_id="ort_force_optimize",
                            )
                        button_export_default_unet = gr.Button(
                            value="Export and Optimize ONNX",
                            variant="primary",
                            elem_id="ort_export_default_unet",
                            visible=True,
                        )

                    with gr.Tab(label="OnnxRuntime LoRA"):
                        gr.Markdown("# Apply LoRA checkpoint to OnnxRuntime model")
                        # lora_refresh_button = gr.Button(
                        #     value="Refresh",
                        #     variant="primary",
                        #     elem_id="ort_lora_refresh",
                        # )
                        with FormRow(elem_classes="droplist-row", variant="compact"):
                            ort_lora_dropdown = gr.Dropdown(
                                choices=get_valid_lora_checkpoints(),
                                elem_id="lora_model",
                                label="LoRA Model",
                                default=None,
                                multiselect=False,
                            )

                            lora_refresh_button = ToolButton(
                                value=refresh_symbol, elem_id="ort_lora_refresh", visible=True
                            )

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            ort_lora_force_rebuild = gr.Checkbox(
                                label="Force Rebuild.",
                                value=False,
                                elem_id="ort_lora_force_rebuild",
                            )

                        button_export_lora_unet = gr.Button(
                            value="Convert to OnnxRuntime",
                            variant="primary",
                            elem_id="ort_lora_export_unet",
                            visible=False,
                        )

                        lora_refresh_button.click(
                            get_valid_lora_checkpoints,
                            None,
                            ort_lora_dropdown,
                        )

                        ort_lora_dropdown.change(
                            disable_lora_export,
                            ort_lora_dropdown,
                            button_export_lora_unet,
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
                force_export,
                force_optimize,
            ],
            outputs=[ort_result],
        )

        button_export_lora_unet.click(
            export_lora_to_ort,
            inputs=[ort_lora_dropdown, ort_lora_force_rebuild],
            outputs=[ort_result],
        )

    return [(ort_interface, "OnnxRuntime", "onnxruntime")]
