# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This file is modified from https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/fusion_attention_unet.py
# We will remove this file once the change is in nightly or release package.

from logging import getLogger
from typing import Union

import numpy as np
from onnx import NodeProto, TensorProto, helper
from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.fusion_utils import NumpyHelper
from onnxruntime.transformers.onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionAttentionUnet(Fusion):
    """
    Fuse Attention subgraph of UNet into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        is_cross_attention: bool,
        enable_packed_qkv: bool,
        enable_packed_kv: bool,
    ):
        super().__init__(
            model,
            "Attention" if is_cross_attention and enable_packed_qkv else "MultiHeadAttention",
            ["LayerNormalization"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.is_cross_attention = is_cross_attention

        # Note: pack Q/K/V or K/V weights into one tensor make it harder for updating initializers for LoRA.
        # TODO: It is better to use separated Q, K and V inputs in offline optimization,
        # and CUDA operator pre-packs those tensors to preferred format based on available kernels.
        # In this way, we can support LoRA and get optimal performance at same time.
        self.enable_packed_qkv = enable_packed_qkv
        self.enable_packed_kv = enable_packed_kv

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads(self, reshape_q: NodeProto) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0
        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, -1]
        q_shape_value = self.model.get_constant_value(reshape_q.input[1])
        if isinstance(q_shape_value, np.ndarray) and list(q_shape_value.shape) == [4]:
            num_heads = int(q_shape_value[2])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0

    def get_hidden_size(self, layernorm_node):
        """Detect hidden_size from LayerNormalization node.
        Args:
            layernorm_node (NodeProto): LayerNormalization node before Q, K and V
        Returns:
            int: hidden_size, or 0 if not found
        """
        layernorm_bias = self.model.get_initializer(layernorm_node.input[2])
        if layernorm_bias:
            return NumpyHelper.to_array(layernorm_bias).shape[0]

        return 0

    def create_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        is_self_attention = not self.is_cross_attention

        if is_self_attention:
            if q_matmul.input[0] != input or k_matmul.input[0] != input or v_matmul.input[0] != input:
                logger.debug(
                    "For self attention, input hidden state for q and k/v shall be same. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None
        else:
            if q_matmul.input[0] != input or (k_matmul.input[0] != v_matmul.input[0]) or (k_matmul.input[0] == input):
                logger.debug(
                    "For cross attention, input hidden state for q and k/v shall be different. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        # Sometimes weights are stored in fp16
        float_type = q_weight.data_type

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

        # assert q and k have same shape as expected
        if is_self_attention:
            if qw.shape != kw.shape or qw.shape != vw.shape:
                return None

            qw_in_size = qw.shape[0]

            if hidden_size > 0 and hidden_size != qw_in_size:
                raise ValueError(
                    f"Input hidden size ({hidden_size}) is not same as weight dimension of q,k,v ({qw_in_size}). "
                    "Please provide a correct input hidden size or pass in 0"
                )

            # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
            # For 2d weights, the shapes would be [in_size, out_size].
            # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
            qw_out_size = int(np.prod(qw.shape[1:]))

            if self.enable_packed_qkv:
                attention_node_name = self.model.create_node_name("MultiHeadAttention")

                c = qw_in_size
                n = num_heads
                h = qw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
                qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(
                    c, n * 3 * h
                )

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=float_type,
                    dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
                    vals=qkv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                self.add_initializer(
                    name=matmul_node_name + "_reshape_shape",
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 3, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[
                        matmul_node_name + "_out",
                        matmul_node_name + "_reshape_shape",
                    ],
                    outputs=[attention_node_name + "_qkv_input"],
                    name=matmul_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
                self.nodes_to_add.extend([matmul_node, reshape_node])
                self.nodes_to_remove.extend([q_matmul, k_matmul, v_matmul])

            else:
                qkv_weight = np.stack((qw, kw, vw), axis=1)
                qkv_weight_dim = 3 * qw_out_size

                attention_node_name = self.model.create_node_name("Attention")

                self.add_initializer(
                    name=attention_node_name + "_qkv_weight",
                    data_type=float_type,
                    dims=[qw_in_size, qkv_weight_dim],
                    vals=qkv_weight,
                )
        else:  # cross attention
            attention_node_name = self.model.create_node_name("MultiHeadAttention")
            if self.enable_packed_kv:
                if kw.shape != vw.shape:
                    return None

                kw_in_size = kw.shape[0]
                vw_in_size = vw.shape[0]
                assert kw_in_size == vw_in_size

                qw_out_size = qw.shape[1]
                kw_out_size = kw.shape[1]
                vw_out_size = vw.shape[1]
                assert qw_out_size == vw_out_size and kw_out_size == vw_out_size

                c = kw_in_size
                n = num_heads
                h = kw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 2, H] shape
                kv_weight = np.dstack([kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 2 * h)

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_KV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=float_type,
                    dims=[kv_weight.shape[0], kv_weight.shape[1]],
                    vals=kv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                self.add_initializer(
                    name=matmul_node_name + "_reshape_shape",
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 2, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[
                        matmul_node_name + "_out",
                        matmul_node_name + "_reshape_shape",
                    ],
                    outputs=[attention_node_name + "_kv_input"],
                    name=matmul_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
                self.nodes_to_add.extend([matmul_node, reshape_node])
                self.nodes_to_remove.extend([k_matmul, v_matmul])

        # No bias, use zeros
        qkv_bias = np.zeros([3, hidden_size], dtype=np.float32)
        qkv_bias_dim = 3 * hidden_size

        self.add_initializer(
            name=attention_node_name + "_qkv_bias",
            data_type=float_type,
            dims=[qkv_bias_dim],
            vals=qkv_bias,
        )

        if is_self_attention:
            if not self.enable_packed_qkv:
                attention_inputs = [
                    input,
                    attention_node_name + "_qkv_weight",
                    attention_node_name + "_qkv_bias",
                ]
            else:
                attention_inputs = [attention_node_name + "_qkv_input"]
        else:
            if not self.enable_packed_kv:
                attention_inputs = [
                    q_matmul.output[0],
                    k_matmul.output[0],
                    v_matmul.output[0],
                    attention_node_name + "_qkv_bias",
                ]
            else:
                attention_inputs = [
                    q_matmul.output[0],
                    attention_node_name + "_kv_input",
                ]

        attention_node = helper.make_node(
            "Attention" if (is_self_attention and not self.enable_packed_qkv) else "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = (
            "Attention (self attention)"
            if is_self_attention and not self.enable_packed_qkv
            else "MultiHeadAttention ({})".format(
                "self attention with packed qkv"
                if self.enable_packed_qkv
                else "cross attention with packed kv"
                if self.enable_packed_kv
                else "cross attention"
            )
        )
        self.increase_counter(counter_name)
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        entry_path = self.model.match_parent_path(normalize_node, ["Cast", "Add"], [0, 0])
        if entry_path is None:
            entry_path = self.model.match_parent_path(normalize_node, ["Cast", "Reshape"], [0, 0])
            if entry_path is None:
                return
        _cast, node_before_layernorm = entry_path

        root_input = node_before_layernorm.output[0]

        children_nodes = input_name_to_nodes[root_input]
        skip_add = None
        for node in children_nodes:
            if node.op_type == "Add":  # SkipLayerNormalization fusion is not applied yet
                skip_add = node
                break
        if skip_add is None:
            return

        match_qkv = self.match_qkv_a1111_torch2(root_input, skip_add)
        if match_qkv is not None:
            (
                is_torch2,
                reshape_qkv,
                transpose_qkv,
                reshape_q,
                matmul_q,
                matmul_k,
                matmul_v,
            ) = match_qkv

            cast_q = self.model.match_parent(matmul_q, "Cast", 0)
            cast_k = self.model.match_parent(matmul_k, "Cast", 0)
            cast_v = self.model.match_parent(matmul_v, "Cast", 0)
            if not (
                cast_q is not None
                and cast_k is not None
                and (cast_q == cast_k if not self.is_cross_attention else cast_q != cast_k)
                and cast_k == cast_v
            ):
                return
            if cast_q.input[0] != normalize_node.output[0]:
                return

            attention_last_node = reshape_qkv

            q_num_heads = self.get_num_heads(reshape_q)
            if q_num_heads <= 0:
                logger.debug("fuse_attention: failed to detect num_heads")
                return

            q_hidden_size = self.get_hidden_size(normalize_node)

            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            new_node = self.create_attention_node(
                matmul_q,
                matmul_k,
                matmul_v,
                q_num_heads,
                q_hidden_size,
                input=matmul_q.input[0],
                output=attention_last_node.output[0],
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

            # Use prune graph to remove nodes since they are shared by all attention nodes.
            self.prune_graph = True

    def match_qkv_a1111_torch2(self, root_input, skip_add):
        """Match Q, K and V paths exported by PyTorch 2.*"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "Einsum"],
            [another_input, None, None, 0, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, reshape_einsum, einsum_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(einsum_qkv, ["Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, _, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(
            einsum_qkv, ["Cast", "Cast", "Softmax", "Mul", "Einsum"], [0, 0, 0, 0, None]
        )
        if qk_nodes is not None:
            (_, _, _softmax_qk, _, einsum_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(einsum_qk, ["Reshape", "Transpose", "Reshape", "MatMul"], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (_, _transpose_q, reshape_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(einsum_qk, ["Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_, _, _, matmul_k) = k_nodes

        return True, reshape_qkv, transpose_qkv, reshape_q, matmul_q, matmul_k, matmul_v
