# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
import json
import os
import sys
from collections import OrderedDict
from logging import getLogger
from typing import List, Tuple

import numpy as np
import onnx
import torch
from modules import sd_models  # , sd_hijack, shared
from modules.shared import cmd_opts
from onnx import numpy_helper
from safetensors.torch import load_file, save_file

from hash_helper import hash_as_fp16_numpy_array
from ort_model_config import ProfileSettings, SDVersion
from ort_model_helper import UNetModel

logger = getLogger(__name__)


# Create weight mapping source hashes
def weight_packing_source_hash_map(weight_packing_list: List):
    source_name_to_hash_shape_mapping = {}
    for weight_packing in weight_packing_list:
        source_list = weight_packing["source"]
        reshape_shape = (
            weight_packing["target"]["source_reshape"] if "source_reshape" in weight_packing["target"] else None
        )
        for source in source_list:
            source_name_to_hash_shape_mapping[source["name"]] = (source["hash"], source["shape"], reshape_shape)
    return source_name_to_hash_shape_mapping


def weight_packing_source_list_map(weight_packing_list: List):
    source_name_to_list_mapping = {}
    for weight_packing in weight_packing_list:
        source_list = weight_packing["source"]
        source_name_list = [source["name"] for source in source_list]
        for source in source_list:
            source_name_to_list_mapping[source["name"]] = source_name_list
    return source_name_to_list_mapping


# Create initializer mapping from name to fp16 data hash and shape
def initializer_hash_shape_map(onnx_opt_path: str):
    dir = os.path.dirname(os.path.abspath(onnx_opt_path))
    onnx_opt_model = onnx.load(onnx_opt_path)
    initializer_hash_shape_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(initializer, base_dir=dir)
        initializer_hash = hash_as_fp16_numpy_array(initializer_data)
        initializer_hash_shape_mapping[initializer.name] = (initializer_hash, initializer_data.shape)
    return initializer_hash_shape_mapping


# Create initializer mapping from name to fp16 data hash, and another map from name to numpy fp16 data
def initializer_hash_data_map(onnx_opt_path: str):
    dir = os.path.dirname(os.path.abspath(onnx_opt_path))
    onnx_opt_model = onnx.load(onnx_opt_path)
    initializer_hash_mapping = {}
    initializer_data_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(initializer, base_dir=dir).astype(np.float16)
        initializer_data_mapping[initializer.name] = initializer_data

        initializer_hash = hash_as_fp16_numpy_array(initializer_data)
        initializer_hash_mapping[initializer.name] = initializer_hash

    return initializer_hash_mapping, initializer_data_mapping


def save_packing_source_tensors(weight_packing_list: List, onnx_path: str, packing_source_tensors_path: str):
    """
    Save the original weights of packing source to a

    Args:
        weight_packing_list (List): a list of packed weights including the source and target name/shape etc.
        onnx_path (str): raw onnx model that exported from PyTorch
        packing_source_tensors_path(str): safe tensor file to store the original weights that used in packing.
    """
    source_name_to_hash_shape_mapping = weight_packing_source_hash_map(weight_packing_list)
    dir = os.path.dirname(os.path.abspath(onnx_path))
    onnx_model = onnx.load(onnx_path)

    source_tensors = {}
    for initializer in onnx_model.graph.initializer:
        if initializer.name not in source_name_to_hash_shape_mapping:
            continue

        source_hash, _shape, reshape_shape = source_name_to_hash_shape_mapping[initializer.name]
        initializer_data = numpy_helper.to_array(initializer, base_dir=dir).astype(np.float16)
        initializer_hash = hash_as_fp16_numpy_array(initializer_data)

        # The hash shall be matched with the one from weight_packing_list that is computed during optimization.
        if initializer_hash != source_hash:
            print(
                f"source name={initializer.name} hash={source_hash} shape={_shape} reshape_shape={reshape_shape} data.shape={initializer_data.shape} data.hash={initializer_hash}"
            )
            # for initializer2 in onnx_model.graph.initializer:
            #     initializer_data2 = numpy_helper.to_array(initializer2, base_dir=dir).astype(np.float16)
            #     initializer_hash2 = hash_as_fp16_numpy_array(initializer_data2)
            #     if  initializer_hash2 == source_hash:
            #         print(f"source name={initializer2.name} hash={source_hash} shape={_shape} reshape_shape={reshape_shape} data.shape={initializer_data2.shape} data.hash={initializer_hash2}")
            #         break
            raise RuntimeError(
                f"hash not matched for initializer {initializer.name} in {onnx_path}. Please force export the onnx model."
            )

        tensor = torch.tensor(initializer_data)
        # if reshape_shape is not None:
        #     tensor = tensor.reshape(reshape_shape)

        source_tensors[initializer.name] = tensor

    for name in source_name_to_hash_shape_mapping:
        if name not in source_tensors:
            raise RuntimeError(
                f"initializer name {name} not found in raw onnx file {onnx_path}. Please report the issue to the Stable-Diffusion-WebUI-OnnxRuntime git repository."
            )

    save_file(source_tensors, packing_source_tensors_path)


# Create a mapping of weights from torch to onnx
def export_weights_map(unet_model: UNetModel, onnx_opt_path: str, weights_map_path: str, weight_packing_list: List):
    """
    Create weight mapping from Torch to ONNX

    Args:
        unet_model (UNetModel): UNet model for export
        onnx_opt_path (str): optimized onnx model
        weights_map_path (str): output JSON file with the weight mapping
        weight_packing_list (List): a list of packed weights including the source and target name/shape etc.
    """
    state_dict = unet_model.unet.state_dict()

    initializer_hash_shape_mapping = initializer_hash_shape_map(onnx_opt_path)

    source_name_to_hash_shape_mapping = weight_packing_source_hash_map(weight_packing_list)

    weights_name_mapping = {}
    weights_shape_mapping = {}
    # A set to keep track of initializers already added to the name_mapping dict
    initializers_mapped = set()
    for weight_name, weight in state_dict.items():
        # Compute hashes of fp16 weight and transposed fp16 weight
        weight_fp16 = weight.cpu().detach().numpy().astype(np.float16)
        weight_hash = hash_as_fp16_numpy_array(weight_fp16)
        transposed_weight_hash = hash_as_fp16_numpy_array(np.transpose(weight_fp16))

        # Match with initializer in optimized onnx model
        for initializer_name, (
            initializer_hash,
            initializer_shape,
        ) in initializer_hash_shape_mapping.items():
            # Due to constant folding or graph optimization, some weights are transposed
            if weight_hash == initializer_hash or transposed_weight_hash == initializer_hash:
                # The assert below ensures there is a 1:1 mapping between PyTorch and ONNX weight names.
                # It can be removed in cases where 1:many mapping is found and name_mapping[weight_name] = list()
                assert initializer_name not in initializers_mapped
                weights_name_mapping[weight_name] = (initializer_name, False)
                initializers_mapped.add(initializer_name)
                is_transpose = weight_hash != initializer_hash
                weights_shape_mapping[weight_name] = (initializer_shape, is_transpose)

        # Match with source initializers that has been packed
        for initializer_name, (
            initializer_hash,
            initializer_shape,
            _reshape_shape,
        ) in source_name_to_hash_shape_mapping.items():
            # Due to constant folding, some weights are transposed during export
            if weight_hash == initializer_hash or transposed_weight_hash == initializer_hash:
                assert initializer_name not in initializers_mapped
                weights_name_mapping[weight_name] = (initializer_name, True)
                initializers_mapped.add(initializer_name)
                is_transpose = weight_hash != initializer_hash
                weights_shape_mapping[weight_name] = (
                    initializer_shape,  # reshape_shape if reshape_shape is not None else initializer_shape,
                    is_transpose,
                )

        # Sanity check: Were any weights not matched
        if weight_name not in weights_name_mapping:
            print(f"[I] PyTorch weight {weight_name} not matched with any ONNX initializer")
    print(f"[I] UNet: {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers")

    assert weights_name_mapping.keys() == weights_shape_mapping.keys()
    with open(weights_map_path, "w") as fp:
        json.dump(
            {
                "weight_name_map": weights_name_mapping,
                "weight_shape_map": weights_shape_mapping,
                "weight_packing": weight_packing_list,
            },
            fp,
        )


def apply_lora(model: torch.nn.Module, lora_path: str, inputs: Tuple[torch.Tensor]) -> torch.nn.Module:
    try:
        sys.path.append("extensions-builtin/Lora")
        networks = importlib.import_module("networks")
        _network = importlib.import_module("network")
        _lora_net = importlib.import_module("extra_networks_lora")
    except Exception as e:
        # Shall not reached here if A1111 1.7.0 or later is installed properly.
        logger.exception(e)
        logger.error("LoRA not found. Please install LoRA extension first")

    model.forward(*inputs)
    lora_name = os.path.splitext(os.path.basename(lora_path))[0]
    networks.load_networks([lora_name], [1.0], [1.0], [None])

    model.forward(*inputs)
    return model


def get_delta_weights(
    state_dict: dict,
    onnx_opt_path: str,
    packing_source_tensors_path: str,
    weight_name_mapping: dict,
    weight_shape_mapping: dict,
    weight_packing_list: List,
) -> dict:
    packing_source_tensors: dict = load_file(packing_source_tensors_path)
    delta_weights = OrderedDict()
    initializer_hash_mapping, initializer_data_mapping = initializer_hash_data_map(onnx_opt_path)

    source_name_to_hash_shape_mapping = weight_packing_source_hash_map(weight_packing_list)

    for torch_name, (initializer_name, is_packed) in weight_name_mapping.items():
        if not is_packed:
            initializer_hash = initializer_hash_mapping[initializer_name]
        else:
            initializer_hash = source_name_to_hash_shape_mapping[initializer_name][0]

        if torch_name not in state_dict:
            logger.error("%s not found in state_dict", torch_name)
            continue

        wt = state_dict[torch_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[torch_name]

        # if is_packed:
        #     print(f"torch_name={torch_name} initializer_name={initializer_name} initializer_hash={initializer_hash} initializer_shape={initializer_shape} wt.shape={wt.shape} is_transpose={is_transpose}")

        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash_as_fp16_numpy_array(wt.cpu().detach().numpy())

        if initializer_hash != wt_hash:
            tensor = (
                torch.tensor(initializer_data_mapping[initializer_name])
                if not is_packed
                else packing_source_tensors[initializer_name]
            )
            delta = wt - tensor.to(wt.device)
            delta_weights[initializer_name] = delta.contiguous()
            logger.debug(
                f"delta_weights name={initializer_name} hash={initializer_hash} shape={initializer_shape} transpose={is_transpose} is_packed={is_packed} torch_name={torch_name}"
            )

    return delta_weights


def export_lora(
    modelobj: UNetModel,
    onnx_opt_path: str,
    weights_map_path: str,
    packing_source_tensors_path: str,
    lora_name: str,
    profile: ProfileSettings,
) -> dict:
    logger.info("Exporting to ONNX...")
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    with open(weights_map_path) as fp_weight_map:
        logger.info("Loading weights map: %s", weights_map_path)
        data = json.load(fp_weight_map)
        weights_name_mapping = data["weight_name_map"]
        weights_shape_mapping = data["weight_shape_map"]
        weight_packing_list = data["weight_packing"]

    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet = apply_lora(modelobj.unet, os.path.splitext(lora_name)[0], inputs)

        delta_dict = get_delta_weights(
            modelobj.unet.state_dict(),
            onnx_opt_path,
            packing_source_tensors_path,
            weights_name_mapping,
            weights_shape_mapping,
            weight_packing_list,
        )

    return delta_dict


def merge_loras(loras: List[str], scales: List[str]) -> dict:
    refit_dict = {}
    for lora, scale in zip(loras, scales):
        lora_dict = load_file(lora)
        for k, v in lora_dict.items():
            if k in refit_dict:
                refit_dict[k] += scale * v
            else:
                refit_dict[k] = scale * v
    return refit_dict


def apply_loras(
    onnx_opt_path: str, packing_source_tensors_path: str, weight_packing_list: List, loras: List[str], scales: List[str]
) -> dict:
    refit_dict = merge_loras(loras, scales)

    packing_source_tensors: dict = load_file(packing_source_tensors_path)

    device = None
    packing_dict = {}
    # Make sure when packing tensors shall be added to refit_dict in a group.
    # For example, we need 3 tensors for packing, we cannot add only one or two of them to refit_dict, otherwise the refit_dict is incomplete.
    for weight_packing in weight_packing_list:
        source_list = weight_packing["source"]
        count = 0
        for source in source_list:
            if source["name"] in refit_dict:
                count += 1
                if device is None:
                    device = refit_dict[source["name"]].device

        if count == 0:
            continue

        tensors = []
        for source in source_list:
            name = source["name"]

            delta = refit_dict.pop(name, None)
            tensor = packing_source_tensors[name].to(device)

            if delta is not None:
                tensor = tensor + delta

            if "source_reshape" in weight_packing["target"]:
                tensor = tensor.reshape(weight_packing["target"]["source_reshape"])
            tensors.append(tensor)

        target_name = weight_packing["target"]["name"]
        transform = weight_packing["target"]["transform"]
        if transform in ["np.dstack([qw, kw, vw)])", "np.dstack([kw, vw)])"]:
            target = torch.dstack(tensors).reshape(weight_packing["target"]["shape"])
        elif transform == "np.stack((qw, kw, vw), axis=1)":
            target = torch.stack(tensors, dim=1).reshape(weight_packing["target"]["shape"])
        else:
            raise RuntimeError("unknown transform: " + transform)

        assert target_name not in refit_dict
        packing_dict[target_name] = target.contiguous()

    base = onnx.load(onnx_opt_path)
    onnx_opt_dir = os.path.dirname(onnx_opt_path)

    for initializer in base.graph.initializer:
        if initializer.name not in refit_dict:
            continue
        assert initializer.name not in packing_source_tensors

        delta = refit_dict[initializer.name]
        initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
        new_weight = torch.tensor(initializer_data).to(delta.device) + delta

        refit_dict[initializer.name] = new_weight.contiguous()

    refit_dict.update(packing_dict)

    return refit_dict


def get_version_from_filename(name):
    if "v1-" in name:
        return "1.5"
    elif "v2-" in name:
        return "2.1"
    elif "xl" in name:
        return "xl-1.0"
    else:
        return "Unknown"


def get_lora_checkpoints():
    available_lora_models = {}
    allowed_extensions = ["pt", "ckpt", "safetensors"]
    candidates = [p for p in os.listdir(cmd_opts.lora_dir) if p.split(".")[-1] in allowed_extensions]

    for filename in candidates:
        metadata = {}
        name, ext = os.path.splitext(filename)
        config_file = os.path.join(cmd_opts.lora_dir, name + ".json")

        if ext == ".safetensors":
            metadata = sd_models.read_metadata_from_safetensors(os.path.join(cmd_opts.lora_dir, filename))
        else:
            print(
                """LoRA {} is not a safetensor. This might cause issues when exporting to TensorRT.
                   Please ensure that the correct base model is selected when exporting.""".format(
                    name
                )
            )

        base_model = metadata.get("ss_sd_model_name", "Unknown")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            version = SDVersion.from_str(config["sd version"])

        else:
            version = SDVersion.Unknown
            print(f"No config file found for {name}. You can generate it in the LoRA tab.")

        available_lora_models[name] = {
            "filename": filename,
            "version": version,
            "base_model": base_model,
        }

    print("available_lora_models", available_lora_models)

    return available_lora_models
