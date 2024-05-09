# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Optional

from onnx import ModelProto
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_unet import UnetOnnxModel

from nhwc_conv_fusion import NhwcConvFusion
from unet_attention_fusion import UnetAttentionFusion

logger = logging.getLogger(__name__)


class UnetFusionTracer(UnetOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """
        Initialize UNet Fusion Tracer for LoRA weights refit

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)
        self.weight_packing_list = []

    def fuse_multi_head_attention(self, options: Optional[FusionOptions] = None):
        # Self Attention
        enable_packed_qkv = (options is None) or options.enable_packed_qkv
        self_attention_fusion = UnetAttentionFusion(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=False,
            enable_packed_qkv=enable_packed_qkv,
            enable_packed_kv=False,
            weight_packing_list=self.weight_packing_list,
        )
        self_attention_fusion.apply()

        # Cross Attention
        enable_packed_kv = (options is None) or options.enable_packed_kv
        cross_attention_fusion = UnetAttentionFusion(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=True,
            enable_packed_qkv=False,
            enable_packed_kv=enable_packed_kv,
            weight_packing_list=self.weight_packing_list,
        )
        cross_attention_fusion.apply()

    def convert_conv_to_nhwc(self):
        # conv_to_nhwc_conv = NhwcConvFusion(self, transpose_weight=False, weight_packing_list=self.weight_packing_list)
        conv_to_nhwc_conv = NhwcConvFusion(self, transpose_weight=True, weight_packing_list=None)
        conv_to_nhwc_conv.apply()
        # onnx.save(self.model, "unet_nhwc_conv.onnx", save_as_external_data=True, all_tensors_to_one_file=True)
