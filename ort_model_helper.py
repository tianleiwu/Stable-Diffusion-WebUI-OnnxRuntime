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

from typing import List, Tuple

import torch
from modules import sd_hijack, sd_unet

from ort_model_config import ProfileSettings


class UNetModel(torch.nn.Module):
    def __init__(
        self,
        unet,
        embedding_dim: int,
        is_xl: bool = False,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.is_xl = is_xl

        self.embedding_dim = embedding_dim

        self.y_embed_dim = 2816
        self.in_channels = self.unet.in_channels

        self.dynamic_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B", 1: "77N"},
            "timesteps": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
            "y": {0: "2B"},
        }

        # For onnx export and optimization
        self.use_external_data = is_xl

    def apply_torch_model(self):
        def disable_checkpoint(self):
            if getattr(self, "use_checkpoint", False) is True:
                self.use_checkpoint = False
            if getattr(self, "checkpoint", False) is True:
                self.checkpoint = False

        self.unet.apply(disable_checkpoint)
        self.set_unet("None")

    def set_unet(self, ckpt: str):
        sd_unet.apply_unet(ckpt)
        sd_hijack.model_hijack.apply_optimizations(ckpt)

    def get_input_names(self) -> List[str]:
        names = ["sample", "timesteps", "encoder_hidden_states"]
        if self.is_xl:
            names.append("y")
        return names

    def get_output_names(self) -> List[str]:
        return ["latent"]

    def get_dynamic_axes(self) -> dict:
        names = self.get_input_names() + self.get_output_names()
        dynamic_axes = {name: self.dynamic_axes[name] for name in names}
        return dynamic_axes

    def get_sample_input(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        text_len: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor]:
        return (
            torch.randn(
                batch_size,
                self.in_channels,
                latent_height,
                latent_width,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, dtype=dtype, device=device),
            torch.randn(
                batch_size,
                text_len,
                self.embedding_dim,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, self.y_embed_dim, dtype=dtype, device=device) if self.is_xl else None,
        )

    def get_input_profile(self, profile: ProfileSettings) -> dict:
        min_batch, opt_batch, max_batch = profile.get_a1111_batch_dim()
        (
            min_latent_height,
            latent_height,
            max_latent_height,
            min_latent_width,
            latent_width,
            max_latent_width,
        ) = profile.get_latent_dim()

        shape_dict = {
            "sample": [
                [min_batch, self.unet.in_channels, min_latent_height, min_latent_width],
                [opt_batch, self.unet.in_channels, latent_height, latent_width],
                [max_batch, self.unet.in_channels, max_latent_height, max_latent_width],
            ],
            "timesteps": [[min_batch], [opt_batch], [max_batch]],
            "encoder_hidden_states": [
                [min_batch, profile.t_min, self.embedding_dim],
                [opt_batch, profile.t_opt, self.embedding_dim],
                [max_batch, profile.t_max, self.embedding_dim],
            ],
        }
        if self.is_xl:
            shape_dict["y"] = [
                [min_batch, self.y_embed_dim],
                [opt_batch, self.y_embed_dim],
                [max_batch, self.y_embed_dim],
            ]

        return shape_dict
