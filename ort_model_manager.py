import copy
import json
import os
from logging import info, warning

import torch
from modules import paths_internal

from ort_model_config import ModelConfig, ModelConfigEncoder

# This directory caches all onnx models exported by torch.
ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-ort-export")
if not os.path.exists(ONNX_MODEL_DIR):
    os.makedirs(ONNX_MODEL_DIR)

# This directory contains all onnx models that optimized by ONNX Runtime.
ORT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-ort")
if not os.path.exists(ORT_MODEL_DIR):
    os.makedirs(ORT_MODEL_DIR)

MODEL_FILE = os.path.join(ORT_MODEL_DIR, "model.json")


def get_cc():
    cc_major = torch.cuda.get_device_properties(0).major
    cc_minor = torch.cuda.get_device_properties(0).minor
    return cc_major, cc_minor


cc_major, cc_minor = get_cc()


class ModelManager:
    def __init__(self, model_file=MODEL_FILE, default_provider="cuda") -> None:
        self.all_models = {}
        self.model_file = model_file

        # Default execution provider
        self.default_provider = default_provider

        if not os.path.exists(model_file):
            warning("model.json does not exist. Creating new one.")
        else:
            self.all_models = self.read_json()

        self.update()

    @staticmethod
    def get_onnx_path(model_name):
        onnx_filename = f"{model_name}.onnx"
        onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)
        return onnx_filename, onnx_path

    def get_engine_path(self, model_name, model_hash, provider=None):
        if provider is None:
            provider = self.default_provider
        engine_filename = "_".join([model_name, model_hash, provider]) + ".onnx"
        engine_path = os.path.join(ORT_MODEL_DIR, engine_filename)

        return engine_filename, engine_path

    def update(self):
        ort_engines = [ort_file for ort_file in os.listdir(ORT_MODEL_DIR) if ort_file.endswith(".onnx")]
        tmp_all_models = copy.deepcopy(self.all_models)
        for provider, base_models in tmp_all_models.items():
            for base_model, models in base_models.items():
                tmp_config_list = {}
                for model_config in models:
                    if model_config["filepath"] not in ort_engines:
                        info("Model config outdated. {} was not found".format(model_config["filepath"]))
                        continue
                    tmp_config_list[model_config["filepath"]] = model_config

                tmp_config_list = list(tmp_config_list.values())
                if len(tmp_config_list) == 0:
                    self.all_models[provider].pop(base_model)
                else:
                    self.all_models[provider][base_model] = tmp_config_list

        self.write_json()

    def __del__(self):
        self.update()

    def add_entry(self, model_name, model_hash, profile, fp32, inpaint, unet_hidden_dim, provider=None):
        if provider is None:
            provider = self.default_provider

        config = ModelConfig(profile, fp32, inpaint, unet_hidden_dim)
        ort_name, _ort_path = self.get_engine_path(model_name, model_hash)

        if provider not in self.all_models:
            self.all_models[provider] = {}

        if model_name not in self.all_models[provider]:
            self.all_models[provider][model_name] = []

        self.all_models[provider][model_name].append({"filepath": ort_name, "config": config})

        self.update()

    def write_json(self):
        with open(self.model_file, "w") as f:
            json.dump(self.all_models, f, indent=4, cls=ModelConfigEncoder)

    def read_json(self, encode_config=True):
        with open(self.model_file) as f:
            out = json.load(f)

        if not encode_config:
            return out

        for provider, models in out.items():
            for base_model, configs in models.items():
                for i in range(len(configs)):
                    out[provider][base_model][i]["config"] = ModelConfig(**configs[i]["config"])
        return out

    def available_models(self, provider=None):
        if provider is None:
            provider = self.default_provider

        available = self.all_models.get(provider, {})
        return available

    def get_valid_models(
        self,
        base_model: str,
        width: int,
        height: int,
        batch_size: int,
        max_embedding: int,
    ):
        valid_models = []
        distances = []
        idx = []
        models = self.available_models()
        for i, model in enumerate(models[base_model]):
            valid, distance = model["config"].is_compatible(width, height, batch_size, max_embedding)
            if valid:
                valid_models.append(model)
                distances.append(distance)
                idx.append(i)

        return valid_models, distances, idx


modelmanager = ModelManager()
