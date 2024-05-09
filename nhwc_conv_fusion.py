# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List, Optional

from onnx import helper, numpy_helper
from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.fusion_utils import FusionUtils
from onnxruntime.transformers.onnx_model import OnnxModel

from hash_helper import hash_as_fp16_numpy_array

logger = getLogger(__name__)


class NhwcConvFusion(Fusion):
    """Convert Conv to NhwcConv"""

    def __init__(self, model: OnnxModel, transpose_weight: bool = False, weight_packing_list: Optional[List] = None):
        super().__init__(
            model,
            fused_op_type="NhwcConv" if transpose_weight else "Conv",
            search_op_types=["Conv"],
            description="" if transpose_weight else "com.ms.internal.nhwc",
        )

        self.fusion_utils = FusionUtils(model)

        self.transpose_weight = transpose_weight

        # A list to trace weight packing or transform
        self.weight_packing_list = weight_packing_list

    def create_transpose_node(self, input_name: str, perm: List[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")

        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name

        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])

        return transpose_node

    def fuse(self, conv, input_name_to_nodes, output_name_to_node):
        if conv.domain == "com.ms.internal.nhwc":
            return

        # Make sure the weights is 4D
        weight_tensor = self.model.get_initializer(conv.input[1])
        if weight_tensor is None:
            return
        weight = numpy_helper.to_array(weight_tensor)
        if len(weight.shape) != 4:
            return

        # Add Transpose node to convert input from NCHW to NHWC
        input_transpose_node = self.create_transpose_node(conv.input[0], [0, 2, 3, 1])
        nodes_to_add = [input_transpose_node]

        node_name = self.model.create_node_name(self.fused_op_type)
        nhwc_output_name = node_name + "_out_nhwc"
        output_transpose_node = self.create_transpose_node(nhwc_output_name, [0, 3, 1, 2], conv.output[0])
        nodes_to_add.append(output_transpose_node)

        nhwc_conv_input = input_transpose_node.output[0]

        dtype = self.model.get_dtype(conv.input[0])
        if not (dtype is not None and weight_tensor.data_type == dtype):
            cast_node = self.fusion_utils.add_cast_node(
                input_name=nhwc_conv_input,
                to_type=weight_tensor.data_type,
                output_name_to_node=output_name_to_node,
            )
            nhwc_conv_input = cast_node.output[0]

        if self.transpose_weight and self.weight_packing_list:
            # Transpose weights from NCHW to NHWC
            transposed_weight = weight.transpose(0, 2, 3, 1)

            transposed_weight_name = node_name + "_weight_nhwc"

            weight_packing = {
                "source": [
                    {
                        "name": conv.input[1],
                        "hash": hash_as_fp16_numpy_array(weight),
                        "shape": list(weight.shape),
                    }
                ],
                "target": {
                    "name": transposed_weight_name,
                    "shape": list(transposed_weight.shape),
                    "format": "nchw_to_nhwc",
                    "transform": "transpose(0, 2, 3, 1)",
                },
            }
            self.weight_packing_list.append(weight_packing)

            self.add_initializer(
                name=transposed_weight_name,
                data_type=weight_tensor.data_type,
                dims=list(transposed_weight.shape),
                vals=transposed_weight,
            )

            nhwc_conv = helper.make_node(
                "NhwcConv",
                inputs=[nhwc_conv_input, transposed_weight_name] + conv.input[2:],
                outputs=[nhwc_output_name],
                name=node_name + "-" + conv.name,
            )
            nhwc_conv.attribute.extend(conv.attribute)
            nhwc_conv.domain = "com.microsoft"
            nodes_to_add.append(nhwc_conv)

            self.nodes_to_remove.append(conv)
        elif self.transpose_weight:
            weight_transpose_node = self.create_transpose_node(conv.input[1], [0, 2, 3, 1])
            transposed_weight_name = weight_transpose_node.output[0]
            nodes_to_add.append(weight_transpose_node)

            nhwc_conv = helper.make_node(
                "NhwcConv",
                inputs=[nhwc_conv_input, transposed_weight_name] + conv.input[2:],
                outputs=[nhwc_output_name],
                name=node_name + "-" + conv.name,
            )
            nhwc_conv.attribute.extend(conv.attribute)
            nhwc_conv.domain = "com.microsoft"
            nodes_to_add.append(nhwc_conv)

            self.nodes_to_remove.append(conv)
        else:
            conv.input[0] = nhwc_conv_input
            conv.domain = "com.ms.internal.nhwc"
            conv.output[0] = nhwc_output_name

        for node in nodes_to_add:
            self.node_name_to_graph_name[node.name] = self.this_graph_name
        self.nodes_to_add.extend(nodes_to_add)

        self.increase_counter("NhwcConv" if self.transpose_weight else "Conv@com.ms.internal.nhwc")
