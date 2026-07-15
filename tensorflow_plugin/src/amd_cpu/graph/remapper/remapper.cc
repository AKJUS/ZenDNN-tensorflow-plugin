/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_plugin/src/amd_cpu/graph/remapper/remapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/graph/remapper/constant_names.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_common_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/layout_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/pattern_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/symbolic_shapes.h"
#include "tensorflow_plugin/src/amd_cpu/util/node_def_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/strcat.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_id.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {
namespace graph {

bool HasDataType(const NodeDef* node, const DataType& expected,
                 const string& type_attr) {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == expected;
}

bool IsSupportedActivation(const NodeDef& node) {
  bool is_default_supported = IsRelu(node) || IsRelu6(node) || IsElu(node) ||
                              IsLeakyRelu(node) || IsSigmoid(node) ||
                              IsTanh(node);
  return is_default_supported;
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);
}

// Helper function to set fused op attributes with activation.
// `fused_ops` should not contain `activation`, it will add activation
// in this function.
void SetFusedOpAttributesWithActivation(
    NodeDef* fused, const NodeDef* activation,
    std::vector<absl::string_view> fused_ops, int num_args = 1) {
  // Handle special activation.
  if (activation != nullptr) {
    auto& activation_attr = activation->attr();

    if (IsLeakyRelu(*activation)) {
      AddNodeAttr("leakyrelu_alpha", activation_attr.at("alpha"), fused);
      fused_ops.push_back(activation->op());
    } else if (IsGelu(*activation)) {
      fused_ops.push_back(activation_attr.at("approximate").b()
                              ? "GeluApproximate"
                              : "GeluExact");
    } else {
      fused_ops.push_back(activation->op());
    }
  }

  SetFusedOpAttributes(fused, fused_ops, num_args);
}

Status GetTensorFromConstant(const NodeDef* node_def, Tensor* dst) {
  if (!dst->FromProto(node_def->attr().at("value").tensor())) {
    TF_CHECK_OK(errors::InvalidArgument(
        "Could not construct Tensor from TensorProto in node: ",
        node_def->name()));
  }

  return OkStatus();
}

namespace {

// FusedBatchNorm with activation.
struct FusedBatchNormEx {
  FusedBatchNormEx() = default;

  int fused_batch_norm = kMissingIndex;
  int activation = kMissingIndex;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add, int bias_port)
      : contraction(contraction), bias_add(bias_add), bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int bias_port = kMissingIndex;
};

struct BatchMatMulWithBiasAdd {
  BatchMatMulWithBiasAdd() = default;
  BatchMatMulWithBiasAdd(int batch_matmul, int bias_add, int bias_port)
      : batch_matmul(batch_matmul), bias_add(bias_add), bias_port(bias_port) {}

  int batch_matmul = kMissingIndex;
  int bias_add = kMissingIndex;
  int bias_port = kMissingIndex;
};

struct BatchMatMulWithBiasAddAndActivation {
  BatchMatMulWithBiasAddAndActivation() = default;
  BatchMatMulWithBiasAddAndActivation(int batch_matmul, int bias_add,
                                      int activation, int bias_port)
      : batch_matmul(batch_matmul),
        bias_add(bias_add),
        activation(activation),
        bias_port(bias_port) {}

  int batch_matmul = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(int contraction, int bias_add, int add,
                               int port_id, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd, Add and Relu.
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(int contraction, int bias_add, int add,
                                      int port_id, int activation,
                                      int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// MatMul node followed by a Activation.
struct ContractionWithActivation {
  ContractionWithActivation() = default;
  ContractionWithActivation(int contraction, int activation)
      : contraction(contraction), activation(activation) {}

  int contraction = kMissingIndex;
  int activation = kMissingIndex;
};

// _FusedMatMul -> Softplus -> Tanh -> Mul(x, tanh(softplus(x))) (decomposed
// Mish), fused to _FusedMatMul + Mish post-op.
struct FusedMatMulMishDecomposedMatch {
  int contraction = kMissingIndex;
  int mish_mul = kMissingIndex;
  int tanh = kMissingIndex;
  int softplus = kMissingIndex;
  // BF16 SrcT==DstT Cast nodes between _FusedMatMul and Softplus/Mul (removed
  // on fuse).
  std::vector<int> noop_bf16_cast_nodes;
  // Float->BF16 Cast on BF16 _FusedMatMul :0 when Softplus is BF16 (stale
  // AMP/BN shim); consumers rewired to matmul output then Cast removed in
  // AddFusedMatMulMishDecomposed.
  std::vector<int> redundant_f32_to_bf16_cast_on_bf16_matmul_out;
};

// safe_embedding_lookup_sparse subgraph matched by the remapper.
struct SafeEmbeddingLookupSparse {
  SafeEmbeddingLookupSparse() = default;

  int select_v2 = kMissingIndex;               // SelectV2 (zero_empty_rows)
  int tile = kMissingIndex;                    // Tile (broadcast indicator)
  int reshape = kMissingIndex;                 // Reshape (indicator reshape)
  int zeros_like = kMissingIndex;              // ZerosLike
  int sparse_segment = kMissingIndex;          // SparseSegment{Sum,Mean,SqrtN}
  int sparse_fill_empty_rows = kMissingIndex;  // SparseFillEmptyRows
  int strided_slice = kMissingIndex;           // StridedSlice (segment_ids)
  int embedding_table = kMissingIndex;  // Identity/Const (embedding table)

  // Upstream filter chain (GatherV2 + Where + GreaterEqual + Reshape).
  int gv2_indices = kMissingIndex;     // GatherV2 (reindex sparse indices)
  int gv2_values = kMissingIndex;      // GatherV2 (gather hashed values)
  int filter_reshape = kMissingIndex;  // Reshape(Where(...))
  int filter_where = kMissingIndex;    // Where(GreaterEqual(...))
  int filter_ge = kMissingIndex;       // GreaterEqual(FloorMod, 0)
  bool has_upstream_filter = false;

  // Downstream adjust_shape chain
  // (Shape + Slice + ConcatV2 + Reshape) that reshapes output to original dims.
  int adjust_shape_node = kMissingIndex;       // Shape (of fused output)
  int adjust_slice_node = kMissingIndex;       // Slice (extract embed dim)
  int adjust_concat_node = kMissingIndex;      // ConcatV2 (build target shape)
  int adjust_reshape_node = kMissingIndex;     // Reshape (final output reshape)
  int adjust_cast_slice_node = kMissingIndex;  // Slice (from Cast, batch dims)
  bool has_adjust_shape = false;
  string orig_dense_shape_input;  // Name of the node providing original dense
                                  // shape

  string combiner;  // "sum", "mean", or "sqrtn"
};

// Pad with `VALID` and 'EXPLICIT' padding followed by Depthwise/_Fused(Conv2D).
// Only `Pad` is supported rather than PadV2/MirrorPad.
struct PadWithContraction {
  PadWithContraction() = default;
  PadWithContraction(int pad, int contraction)
      : pad(pad), contraction(contraction) {}

  int pad = kMissingIndex;
  int contraction = kMissingIndex;
};

struct KerasDenseLayerFwd {
  KerasDenseLayerFwd() = default;
  KerasDenseLayerFwd(int matmul, int reshape, int bias, int activation)
      : matmul(matmul), reshape(reshape), bias(bias), activation(activation) {}

  int matmul = kMissingIndex;
  int reshape = kMissingIndex;
  int bias = kMissingIndex;
  int activation = kMissingIndex;
};

// BatchMatMul + Mul fusion.
struct ContractionWithMul {
  ContractionWithMul() = default;
  ContractionWithMul(int contraction, int mul, int scalar)
      : contraction(contraction), mul(mul), scalar(scalar) {}

  int contraction = kMissingIndex;
  int mul = kMissingIndex;
  int scalar = kMissingIndex;
};

// MatMul + BiasAdd + Mul(scale) + Add|AddV2(shift) -> _FusedMatMul (updated
// Consts).
struct MatMulBiasMulAddFoldMatch {
  MatMulBiasMulAddFoldMatch() = default;

  int matmul = kMissingIndex;
  int bias_add = kMissingIndex;
  int mul = kMissingIndex;
  int add = kMissingIndex;
  int scale_const = kMissingIndex;
  int shift_const = kMissingIndex;
  int bias_port = 1;
  // Optional Cast(BiasAdd|_FusedMatMul) -> Mul and Add|AddV2 ->
  // Cast(DstT=Tmatmul).
  int bias_cast = kMissingIndex;
  int tail_cast = kMissingIndex;
};

// MatMul + BiasAdd + Keras-exported inference BN -> _FusedMatMul (updated
// Consts). Two subgraph shapes:
//   (1) Add|AddV2(Mul(bias×Mul(Rsqrt,gamma)), Sub(β, Mul(…, same scale))) —
//   Keras-style. (2) Add|AddV2(Mul(Mul(Sub(BiasAdd, mean), Rsqrt), gamma),
//   beta) — linear/raw-op BN.
// mul_mean is only used for shape (1); kMissingIndex for shape (2).
struct MatMulBiasKerasBnFoldMatch {
  MatMulBiasKerasBnFoldMatch() = default;

  int matmul = kMissingIndex;
  int bias_add = kMissingIndex;
  int mul_1 = kMissingIndex;
  int keras_bn_add = kMissingIndex;
  int sub = kMissingIndex;
  int mul_scale = kMissingIndex;
  int mul_mean = kMissingIndex;
  int rsqrt = kMissingIndex;
  int add_eps = kMissingIndex;
  int gamma_const = kMissingIndex;
  int beta_const = kMissingIndex;
  int mean_const = kMissingIndex;
  int variance_const = kMissingIndex;
  float epsilon = 1e-3f;
  int bias_port = 1;
  int bias_cast = kMissingIndex;
  int tail_cast = kMissingIndex;
};

// _FusedMatMul + Mul(scale) + Add|AddV2(shift). Remapper runs before Zen
// layout, so the chain head is always _FusedMatMul here (layout renames it
// afterward).
struct FuseMatmulBNfoldMatch {
  FuseMatmulBNfoldMatch() = default;

  int fused_matmul = kMissingIndex;
  int mul = kMissingIndex;
  int add = kMissingIndex;
  int scale_const = kMissingIndex;
  int shift_const = kMissingIndex;
  int bias_cast = kMissingIndex;
  int tail_cast = kMissingIndex;
};

// A contiguous run of GatherV2 ops sharing the same table within a ConcatV2.
struct GroupEmbeddingRun {
  std::string table_name;
  std::vector<std::string> index_names;
  std::vector<int> nodes_to_remove;
  int start_pos = 0;
  int end_pos = 0;  // inclusive
  int gather_axis = 0;
};

// Horizontal fusion: contiguous runs of
// GatherV2 -> Cast -> IsInf/ZerosLike -> SelectV2 within a ConcatV2
// are replaced by _ZenGroupEmbedding ops.
struct GroupEmbedding {
  GroupEmbedding() = default;

  int concat = kMissingIndex;
  std::vector<GroupEmbeddingRun> runs;
  // When true, ALL ConcatV2 inputs matched and the entire ConcatV2 is
  // replaced by a single multi-table _ZenGroupEmbedding node.
  bool full_fusion = false;
  // Ordered elementwise post-op chain fused between each GatherV2 and the
  // ConcatV2, in KERNEL order (innermost/gather-side op first). The chain is
  // uniform across all inputs of the ConcatV2; captured from the first input.
  // Consumed by the _ZenGroupEmbedding kernel via the "fused_ops" attribute.
  std::vector<std::string> fused_ops;
};

bool IsAddWithNoBroadcast(const RemapperContext& ctx, const NodeDef& node) {
  if (!IsAdd(node)) return false;

  // Check if this is case of broadcasting - Add node supports broadcasting.
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node.name(), &props));
  if (props.size() == 2 &&
      ShapesSymbolicallyEqual(props[0].shape(), props[1].shape())) {
    return true;
  }
  return false;
}

// Generic function to check contraction kernel.
bool IsConvOrMatMul(const NodeDef& node) {
  return IsConv3D(node) || IsConv2D(node) || IsDepthwiseConv2dNative(node) ||
         IsMatMul(node);
}

// Returns true if one input to Add is Conv2D/3D or DepthwiseConv2dNative or
// MatMul, and the other input is semantically equivalent to BiasAdd.
bool IsBiasSemanticAdd(const RemapperContext& ctx,
                       const utils::MutableNodeView& node_view,
                       int* bias_port) {
  const auto* node_def = node_view.node();
  if (!IsAdd(*node_def) || node_view.NumRegularFanins() != 2) return false;

  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def->name(), &props));

  if (props.size() < 2) return false;

  const auto& regular_fanin_0 = node_view.GetRegularFanin(0);
  const auto* node_view_0 = regular_fanin_0.node_view();
  const auto* node_def_0 = node_view_0->node();
  const auto& regular_fanin_1 = node_view.GetRegularFanin(1);
  const auto* node_view_1 = regular_fanin_1.node_view();
  const auto* node_def_1 = node_view_1->node();

  // Currently supported data formats are NHWC.
  auto is_channel_last_format = [](const NodeDef& node) -> bool {
    if (node.attr().contains("data_format")) {
      const string data_format = node.attr().at("data_format").s();
      return (data_format == "NHWC");
    }
    return true;
  };

  if (IsConvOrMatMul(*node_def_0) && is_channel_last_format(*node_def_0)) {
    *bias_port = 1;
  } else if (IsConvOrMatMul(*node_def_1) &&
             is_channel_last_format(*node_def_1)) {
    *bias_port = 0;
  } else {
    return false;
  }

  const TensorShapeProto& contraction_shape = props[1 - *bias_port].shape();
  const TensorShapeProto& bias_shape = props[*bias_port].shape();

  if (contraction_shape.unknown_rank() || bias_shape.unknown_rank() ||
      contraction_shape.dim_size() < 1 || bias_shape.dim_size() < 1 ||
      IsUnknown(contraction_shape.dim(contraction_shape.dim_size() - 1)) ||
      IsUnknown(bias_shape.dim(bias_shape.dim_size() - 1)))
    return false;

  // Helper function to check Add/AddV2 could be replaced with BiasAdd.
  const auto is_supported_shape =
      [&](const TensorShapeProto& shape,
          const TensorShapeProto& bcast_shape) -> bool {
    int conv_channel_dim;
    conv_channel_dim = shape.dim(shape.dim_size() - 1).size();

    if (shape.dim_size() == 4 && bcast_shape.dim_size() > 4) return false;
    if (shape.dim_size() == 5 && bcast_shape.dim_size() > 5) return false;

    if (shape.dim_size() < 2) return false;
    // Check that the conv node's channel dim is equal to the 1-dim add node's
    // dim.
    if (conv_channel_dim != bcast_shape.dim(bcast_shape.dim_size() - 1).size())
      return false;

    // Check that add nodes dims are all 1's except the channel dim.
    for (int i = 0; i < bcast_shape.dim_size() - 1; i++) {
      if (1 != bcast_shape.dim(i).size()) return false;
    }
    return true;
  };

  if (ShapesSymbolicallyEqual(contraction_shape, bias_shape) ||
      !ShapesBroadcastable(contraction_shape, bias_shape))
    return false;

  return is_supported_shape(contraction_shape, bias_shape);
}

// Returns 0: left input scalar, 1: right input scalar, -1: no scalar inputs.
int GetMulScalarInputIndex(const RemapperContext& ctx,
                           const NodeDef& node_def) {
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def.name(), &props));
  if (props.size() != 2) return -1;

  bool left_is_scalar = IsScalar(props[0].shape());
  bool right_is_scalar = IsScalar(props[1].shape());
  if (left_is_scalar) {
    return 0;
  } else if (right_is_scalar) {
    return 1;
  } else {
    return -1;
  }
}

// The function to set shapes is used in TF Proper's fused op creation
// extensively, but is not necessary in fused op creation in plugin except for
// BatchNorm fusions.
// TODO(plugin) : Validate the necessity of the same.
void AddInputShapesAttr(const RemapperContext& ctx, int node_index) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  NodeDef* node_def = node_view->node();

  AttrValue attr_input_shape;
  std::vector<OpInfo_TensorProperties> tensor_properties;
  TF_ABORT_IF_ERROR(ctx.graph_properties.GetInputProperties(
      node_def->name(), &tensor_properties));
  for (const auto& tensor_property : tensor_properties) {
    TensorShapeProto* proto = attr_input_shape.mutable_list()->add_shape();
    *proto = tensor_property.shape();
  }

  // TODO(plugin): Validate if "input_shapes" is necessary for ZenDNN ops.
  if (!tensor_properties.empty()) {
    auto* attr = node_def->mutable_attr();
    SetAttrValue(attr_input_shape, &(*attr)["_input_shapes"]);
  }
}

bool FindKerasDenseLayerFwd(const RemapperContext& ctx, int node_index,
                            KerasDenseLayerFwd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a Reshape.
  // Find reshape + biasadd + activation.
  int bias_index = kMissingIndex;
  int activation_index = kMissingIndex;
  const auto* reshape = node_view->node();
  if (!reshape || !IsReshape(*reshape) || HasControlFaninOrFanout(*node_view) ||
      IsInPreserveSet(ctx, node_view->node()))
    return false;

  if (node_view->NumRegularFanouts() != 1) return false;
  const auto& reshape_fanout_0 = node_view->GetRegularFanouts()[0];
  if (reshape_fanout_0.size() != 1) return false;

  const auto* biasadd = reshape_fanout_0[0].node_view();
  if (!IsBiasAdd(*biasadd->node()) || HasControlFaninOrFanout(*biasadd) ||
      IsInPreserveSet(ctx, biasadd->node()))
    return false;
  bias_index = biasadd->node_index();
  if (biasadd->NumRegularFanouts() == 1) {
    const auto& biasadd_fanout_0 = biasadd->GetRegularFanouts()[0];
    if (biasadd_fanout_0.size() == 1) {
      const auto* relu = biasadd_fanout_0[0].node_view();
      if (IsSupportedActivation(*relu->node()) &&
          !HasControlFaninOrFanout(*relu) &&
          !IsInPreserveSet(ctx, relu->node())) {
        activation_index = relu->node_index();
      }
    }
  }

  int bias_dim = 0;
  int weight_dim = 0;
  if (biasadd->NumRegularFanins() != 2) return false;
  auto* readvariable = biasadd->GetRegularFanin(1).node_view();
  // If pb is frozen.
  if (IsConstant(*readvariable->node())) {
    const TensorProto& bias_val =
        readvariable->node()->attr().at("value").tensor();
    const TensorShape bias_shape(bias_val.tensor_shape());
    bias_dim = bias_shape.num_elements();
  }

  // _Arg -> ReadVariableOp -> (Cast) ->BiasAdd.
  if (bias_dim == 0) {
    if (IsCast(*readvariable->node())) {
      readvariable = readvariable->GetRegularFanin(0).node_view();
    }
    if (!IsReadVariableOp(*readvariable->node())) return false;
    const auto* arg_bias = readvariable->GetRegularFanin(0).node_view()->node();

    if (IsArg(*arg_bias)) {
      const AttrValue attr_bshape = arg_bias->attr().at("_handle_shapes");
      if (attr_bshape.list().shape().empty()) return false;
      const TensorShapeProto& bshape_proto = attr_bshape.list().shape(0);
      if (bshape_proto.unknown_rank()) return false;
      bias_dim = TensorShape(bshape_proto).dim_size(0);
    } else if (IsVarHandle(*arg_bias)) {
      const AttrValue attr_bshape = arg_bias->attr().at("shape");

      const TensorShapeProto& bshape_proto = attr_bshape.shape();
      if (bshape_proto.unknown_rank() ||
          IsUnknown(bshape_proto.dim(bshape_proto.dim_size() - 1)))
        return false;
      bias_dim = TensorShape(bshape_proto).dim_size(0);
    } else {
      return false;
    }
  }
  // Arg -> ReadVariableOp -> (Cast) ->MatMul -> reshape.
  if (node_view->NumRegularFanins() != 2) return false;
  const auto* matmul = node_view->GetRegularFanin(0).node_view();
  if (!IsMatMul(*matmul->node())) return false;
  auto* readvariable2 = matmul->GetRegularFanin(1).node_view();

  // If pb is frozen.
  if (IsConstant(*readvariable2->node())) {
    const TensorProto& weight_val =
        readvariable2->node()->attr().at("value").tensor();
    const TensorShape weight_shape(weight_val.tensor_shape());
    weight_dim = weight_shape.dim_size(1);
  }

  if (weight_dim == 0) {
    if (IsCast(*readvariable2->node())) {
      readvariable2 = readvariable2->GetRegularFanin(0).node_view();
    }
    if (!IsReadVariableOp(*readvariable2->node())) return false;
    const auto* arg_weight =
        readvariable2->GetRegularFanin(0).node_view()->node();
    if (IsArg(*arg_weight)) {
      const AttrValue attr_wshape = arg_weight->attr().at("_handle_shapes");
      if (attr_wshape.list().shape().empty()) return false;
      const TensorShapeProto& wshape_proto = attr_wshape.list().shape(0);
      if (!Is2D(wshape_proto)) return false;
      weight_dim = TensorShape(wshape_proto).dim_size(1);
    } else if (IsVarHandle(*arg_weight)) {
      const AttrValue attr_wshape = arg_weight->attr().at("shape");
      const TensorShapeProto& wshape_proto = attr_wshape.shape();
      if (!Is2D(wshape_proto)) return false;
      if (IsUnknown(wshape_proto.dim(wshape_proto.dim_size() - 1)))
        return false;
      weight_dim = TensorShape(wshape_proto).dim_size(1);
    } else {
      return false;
    }
  }

  if (bias_dim != weight_dim) return false;

  const KerasDenseLayerFwd pattern{matmul->node_index(),
                                   node_view->node_index(), bias_index,
                                   activation_index};
  *matched = pattern;
  return true;
}

bool FindBatchMatMulWithBiasAdd(const RemapperContext& ctx, int node_index,
                                BatchMatMulWithBiasAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  int bias_port = 1;
  const auto* node_def = node_view->node();
  if (!IsBiasAdd(*node_def)) return false;

  // Input to the BiasAdd must be a BatchMatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* batchmatmul_node_view = regular_fanin_0.node_view();
  const auto* batchmatmul_node_def = batchmatmul_node_view->node();

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*batchmatmul_node_view)) return false;

  // BatchMatMul.
  bool is_batchmatmul = IsAnyBatchMatMul(*batchmatmul_node_def);

  if (!is_batchmatmul || !HaveSameDataType(node_def, batchmatmul_node_def) ||
      !HasAtMostOneFanoutAtPort0(*batchmatmul_node_view) ||
      IsInPreserveSet(ctx, batchmatmul_node_def))
    return false;

  const BatchMatMulWithBiasAdd pattern{batchmatmul_node_view->node_index(),
                                       node_index, bias_port};
  // We successfully found a BatchMatMul+BiasAdd pattern.
  *matched = pattern;
  return true;
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  const auto* node_def = node_view->node();
  int bias_port = 1;
  if (!IsBiasAdd(*node_def) && !IsBiasSemanticAdd(ctx, *node_view, &bias_port))
    return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(1 - bias_port);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  // Conv, MatMul or DepthwiseConv2D.
  bool is_contraction = IsConvOrMatMul(*contraction_node_def);

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // TODO(plugin): ZenDNN does not support double dtype currently.
  if (HasDataType(contraction_node_def, DT_DOUBLE)) return false;

  // Defer MatMul+BiasAdd when BiasAdd feeds Mul→Add so MatMulBiasMulAddFold
  // can absorb scale/shift into weights/bias in one step. Only defer if the
  // Mul has a single fanout that is Add/AddV2 (complete BN fold pattern).
  // Plain scale-only (Mul without Add) should fuse normally.
  if (IsMatMul(*contraction_node_def)) {
    const auto& bf = node_view->GetRegularFanout(0);
    if (bf.size() == 1) {
      const auto* fo = bf[0].node_view();
      if (fo != nullptr && IsAnyMul(*(fo->node()))) {
        // Only defer if Mul has single fanout to Add/AddV2 (complete BN
        // pattern).
        const auto& mul_fo = fo->GetRegularFanout(0);
        if (mul_fo.size() == 1) {
          const auto* mul_fo_node = mul_fo[0].node_view();
          if (mul_fo_node != nullptr && IsAdd(*(mul_fo_node->node()))) {
            return false;
          }
        }
      }
      if (fo != nullptr && IsCast(*(fo->node())) &&
          fo->GetRegularFanout(0).size() == 1) {
        const auto* fo2 = fo->GetRegularFanout(0)[0].node_view();
        if (fo2 != nullptr && IsAnyMul(*(fo2->node()))) {
          // Only defer if Mul has single fanout to Add/AddV2 (complete BN
          // pattern).
          const auto& mul_fo = fo2->GetRegularFanout(0);
          if (mul_fo.size() == 1) {
            const auto* mul_fo_node = mul_fo[0].node_view();
            if (mul_fo_node != nullptr && IsAdd(*(mul_fo_node->node()))) {
              return false;
            }
          }
        }
      }
    }
  }

  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index, bias_port};
  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;
  return true;
}

// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const utils::MutableNodeView& add_node_view,
                                   const NodeDef& add_node_def, int port_id,
                                   ContractionWithBiasAdd* base) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;
  const auto& bias_add_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (bias_add_node_view == nullptr) return false;
  const auto* bias_add_node_def = bias_add_node_view->node();

  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), base,
                               /*check_device_compatible=*/false))
    return false;
  if (!HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(&add_node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;
  return true;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      const int node_index,
                                      ContractionWithBiasAddAndAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Fusion with AddN is supported only when it has two inputs.
  if (HasControlFaninOrFanout(*node_view) || node_view->NumRegularFanins() != 2)
    return false;

  // Root of the pattern must be a AddN or Add with same input shapes
  // (no broadcasting).
  const auto* node_def = node_view->node();

  if (!IsAddN(*node_def) && !IsAddWithNoBroadcast(ctx, *node_def)) return false;

  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  ContractionWithBiasAdd base;
  matched->port_id = 0;

  // Find the conv+bias pattern in specific port.
  if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                     matched->port_id, &base)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                       matched->port_id, &base)) {
      return false;
    }
  }

  // We do not yet have support for DepthWiseConv2D fusion.
  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsDepthwiseConv2dNative(*contraction_def)) return false;

  // We successfully found a Conv2D+BiasAdd+{AddN,Add} pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->bias_port = base.bias_port;
  matched->add = node_view->node_index();

  return true;
}

bool FindContractionWithActivation(const RemapperContext& ctx, int node_index,
                                   ContractionWithActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsRelu(*node_def)) return false;

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // Input to the Relu must be a MatMul.
  // We have not yet encountered other Contraction + Activation patterns.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  bool is_matmul = IsMatMul(*contraction_node_def);

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  if (!is_matmul || !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // TODO(plugin): ZenDNN does not support double dtype currently.
  if (HasDataType(contraction_node_def, DT_DOUBLE)) return false;

  const ContractionWithActivation pattern{contraction_node_view->node_index(),
                                          node_index};
  // We successfully found a Matmul + Relu pattern.
  *matched = pattern;
  return true;
}

bool FindContractionWithSigmoid(const RemapperContext& ctx, int node_index,
                                ContractionWithActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;

  if (!IsSigmoid(*node_def)) return false;
  if (HasControlFanin(*node_view)) return false;
  if (node_view->NumRegularFanins() < 1) return false;

  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  if (contraction_node_def->op() != "_FusedMatMul") return false;

  auto fused_ops = contraction_node_def->attr().at("fused_ops").list().s();
  if (fused_ops.size() != 1 || fused_ops.at(0) != "BiasAdd") return false;

  if (HasControlFanout(*contraction_node_view) ||
      !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def)) {
    return false;
  }

  const ContractionWithActivation pattern{contraction_node_view->node_index(),
                                          node_index};
  *matched = pattern;
  return true;
}

bool FindContractionWithMish(const RemapperContext& ctx, int node_index,
                             ContractionWithActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr || !IsMish(*node_def)) return false;
  if (HasControlFanin(*node_view) || node_view->NumRegularFanins() < 1) {
    return false;
  }

  const auto* contraction_node_view = node_view->GetRegularFanin(0).node_view();
  const auto* contraction_node_def = contraction_node_view->node();
  if (contraction_node_def->op() != "_FusedMatMul") return false;
  if (!contraction_node_def->attr().contains("fused_ops")) return false;

  auto fused_ops = contraction_node_def->attr().at("fused_ops").list().s();
  if (fused_ops.size() != 1 || fused_ops.at(0) != "BiasAdd") return false;
  if (HasControlFanout(*contraction_node_view) ||
      !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def)) {
    return false;
  }

  *matched = ContractionWithActivation{contraction_node_view->node_index(),
                                       node_index};
  return true;
}

// Defined later in this translation unit; required before
// FindFusedMatMulMishDecomposedPattern.
utils::MutableNodeView* SkipIdentityOnly(utils::MutableNodeView* v,
                                         int max_hops = 16);
bool GetCastDataTypes(const NodeDef& n, DataType* src_t, DataType* dst_t);
bool IsNoopBf16Cast(const NodeDef& n);
void MishDecomposedAppendUniqueCast(std::vector<int>* v, int node_index);
utils::MutableNodeView* SkipIdentityOrNoopBf16CastTowardProducer(
    utils::MutableNodeView* v, bool allow_bf16_noop_cast_on_chain,
    std::vector<int>* noop_bf16_cast_indices);
bool MishDecomposedForwardReachesSoftplusOrMul(
    utils::MutableNodeView* start, int target_node_index,
    bool allow_bf16_noop_cast_forward, const NodeDef* fused_def,
    const NodeDef* endpoint_dtype_node);
bool MishDecomposedIsStaleF32ToBf16CastOnBf16MatmulOutput(
    const NodeDef* fused_def, const utils::MutableNodeView* child);
bool MishDecomposedCastHasAnyRegularConsumer(const utils::MutableNodeView* v);
bool MishDecomposedVerifyMatmulOutputOnlySoftplusAndMul(
    utils::MutableNodeView* fused_view, utils::MutableNodeView* sp_view,
    int mish_mul_node_index, const NodeDef* fused_def, const NodeDef* mul_def);

bool FindFusedMatMulMishDecomposedPattern(
    const RemapperContext& ctx, int node_index,
    FusedMatMulMishDecomposedMatch* matched) {
  auto* mul_view = ctx.graph_view.GetNode(node_index);
  if (mul_view == nullptr) return false;
  const auto* mul_def = mul_view->node();
  if (mul_def == nullptr || !IsAnyMul(*mul_def)) return false;
  if (HasControlFanin(*mul_view)) return false;
  if (mul_view->NumRegularFanins() != 2) return false;

  for (int p = 0; p < 2; ++p) {
    std::vector<int> noop_bf16_casts_to_remove;
    std::vector<int> redundant_f32_to_bf16_cast_on_bf16_mm;
    utils::MutableNodeView* bypass = const_cast<utils::MutableNodeView*>(
        mul_view->GetRegularFanin(p).node_view());
    utils::MutableNodeView* branch = const_cast<utils::MutableNodeView*>(
        mul_view->GetRegularFanin(1 - p).node_view());
    if (bypass == nullptr || branch == nullptr) continue;
    bypass = SkipIdentityOrNoopBf16CastTowardProducer(
        bypass, HasDataType(mul_def, DT_BFLOAT16), &noop_bf16_casts_to_remove);
    branch = SkipIdentityOnly(branch);
    if (bypass == nullptr || branch == nullptr) continue;

    const NodeDef* branch_def = branch->node();
    if (branch_def == nullptr || !IsTanh(*branch_def)) continue;
    if (HasControlFaninOrFanout(*branch)) continue;
    if (branch->NumRegularFanins() < 1) continue;

    utils::MutableNodeView* sp_view = const_cast<utils::MutableNodeView*>(
        branch->GetRegularFanin(0).node_view());
    if (sp_view == nullptr) continue;
    sp_view = SkipIdentityOnly(sp_view);
    if (sp_view == nullptr || sp_view->node() == nullptr ||
        !IsSoftplus(*(sp_view->node())))
      continue;
    if (HasControlFaninOrFanout(*sp_view)) continue;
    if (sp_view->NumRegularFanins() < 1) continue;

    utils::MutableNodeView* from_sp = const_cast<utils::MutableNodeView*>(
        sp_view->GetRegularFanin(0).node_view());
    if (from_sp == nullptr) continue;
    from_sp = SkipIdentityOrNoopBf16CastTowardProducer(
        from_sp, HasDataType(sp_view->node(), DT_BFLOAT16),
        &noop_bf16_casts_to_remove);
    if (from_sp == nullptr || bypass->node_index() != from_sp->node_index()) {
      continue;
    }

    const NodeDef* fused_def = from_sp->node();
    if (fused_def == nullptr || fused_def->op() != "_FusedMatMul") continue;

    if (!fused_def->attr().contains("fused_ops")) continue;
    auto fused_ops = fused_def->attr().at("fused_ops").list().s();
    if (fused_ops.size() != 1 || fused_ops.at(0) != "BiasAdd") continue;
    if (HasDataType(fused_def, DT_DOUBLE)) continue;
    if (IsInPreserveSet(ctx, fused_def) || IsInPreserveSet(ctx, mul_def) ||
        IsInPreserveSet(ctx, branch_def) ||
        IsInPreserveSet(ctx, sp_view->node())) {
      continue;
    }
    if (!HaveSameDataType(mul_def, fused_def) ||
        !HaveSameDataType(branch_def, mul_def)) {
      continue;
    }

    if (HasDataType(fused_def, DT_BFLOAT16) &&
        HasDataType(sp_view->node(), DT_BFLOAT16)) {
      for (const auto& fo : from_sp->GetRegularFanout(0)) {
        const utils::MutableNodeView* cv = fo.node_view();
        if (!MishDecomposedIsStaleF32ToBf16CastOnBf16MatmulOutput(fused_def,
                                                                  cv)) {
          continue;
        }
        MishDecomposedAppendUniqueCast(&redundant_f32_to_bf16_cast_on_bf16_mm,
                                       cv->node_index());
      }
    }

    if (HasControlFanin(*from_sp) || HasControlFanout(*from_sp)) continue;
    if (!MishDecomposedVerifyMatmulOutputOnlySoftplusAndMul(
            from_sp, sp_view, node_index, fused_def, mul_def)) {
      continue;
    }
    if (!HasAtMostOneFanoutAtPort0(*sp_view) ||
        !HasAtMostOneFanoutAtPort0(*branch)) {
      continue;
    }

    const GraphDef* gpreserve = ctx.graph_view.graph();
    bool cast_in_preserve = false;
    for (int cix : noop_bf16_casts_to_remove) {
      if (cix >= 0 && cix < gpreserve->node_size() &&
          IsInPreserveSet(ctx, &gpreserve->node(cix))) {
        cast_in_preserve = true;
        break;
      }
    }
    if (!cast_in_preserve) {
      for (int cix : redundant_f32_to_bf16_cast_on_bf16_mm) {
        if (cix >= 0 && cix < gpreserve->node_size() &&
            IsInPreserveSet(ctx, &gpreserve->node(cix))) {
          cast_in_preserve = true;
          break;
        }
      }
    }
    if (cast_in_preserve) continue;

    matched->contraction = from_sp->node_index();
    matched->mish_mul = node_index;
    matched->tanh = branch->node_index();
    matched->softplus = sp_view->node_index();
    matched->noop_bf16_cast_nodes = std::move(noop_bf16_casts_to_remove);
    matched->redundant_f32_to_bf16_cast_on_bf16_matmul_out =
        std::move(redundant_f32_to_bf16_cast_on_bf16_mm);
    return true;
  }
  return false;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  // TODO: Add check for Convolution + BiasAdd + Sigmoid fusion.
  if (!IsSupportedActivation(*node_def)) return false;

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // And input to the activation node must match ContractionWithBiasAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), &base,
                               /*check_device_compatible=*/false) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      (!HaveSameDataType(node_def, bias_add_node_def) &&
       !(GetDataTypeFromAttr(*node_def, "T") == DT_FLOAT)) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // TODO(plugin): TF Proper doesn't have MatMul + LeakyRelu fusion, remove this
  // limitation once it's supported.
  const auto* contraction_node_view = ctx.graph_view.GetNode(base.contraction);
  const auto* contraction_def = contraction_node_view->node();

  // We have not encountered any other Contraction + BiasAdd + {Sigmoid}
  // pattern.
  if (IsSigmoid(*node_def) && !IsMatMul(*contraction_def)) return false;

  // Verify the inter node has control fanin&fanout or not.
  if (HasControlFaninOrFanout(*bias_add_node_view)) {
    return false;
  }

  // TODO(plugin): ZenDNN does not support double dtype currently.
  if (HasDataType(contraction_def, DT_DOUBLE)) return false;
  if (IsLeakyRelu(*node_def) && IsMatMul(*contraction_def)) return false;

  const ContractionWithBiasAddAndActivation pattern{
      base.contraction, base.bias_add, node_index, base.bias_port};

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindFusedBatchNormEx(const RemapperContext& ctx, int node_index,
                          FusedBatchNormEx* matched) {
  // Root of the pattern must be a Relu.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (!IsRelu(*node_def)) return false;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_batch_norm =
      [&](const utils::MutableNodeView& fused_batch_norm) -> bool {
    const auto* fused_batch_norm_node_def = fused_batch_norm.node();
    if (!IsFusedBatchNorm(*fused_batch_norm_node_def)) return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm_node_def, "T");

    // CPU supports float.
    if (t_dtype != DT_FLOAT) return false;

    string data_format;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kDataFormat, &data_format)
             .ok())
      return false;
    if (data_format != "NHWC" && data_format != "NCHW") return false;

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if ((fused_batch_norm_node_def->op() != "FusedBatchNorm") &&
        !HasDataType(fused_batch_norm_node_def, DT_FLOAT, "U"))
      return false;

    // Check that only one node consumes the 0-th output of a FusedBatchNorm.
    if (HasControlFaninOrFanout(fused_batch_norm) ||
        !HasAtMostOneFanoutAtPort0(fused_batch_norm) ||
        IsInPreserveSet(ctx, fused_batch_norm_node_def))
      return false;

    return true;
  };

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* relu_fanin_0_node_view = regular_fanin_0.node_view();

  // Input to a Relu can be a FusedBatchNorm.
  if (valid_batch_norm(*relu_fanin_0_node_view)) {
    matched->activation = node_index;
    matched->fused_batch_norm = regular_fanin_0.node_index();
    return true;
  }

  return false;
}

bool FindContractionWithBiasAndAddActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAndAddActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  if (HasControlFaninOrFanout(*node_view)) return false;
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsSupportedActivation(*node_def)) return false;

  // ZenDNN activation op only supports float and bfloat16 on CPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* add_node_view = regular_fanin_0.node_view();
  const auto* add_node_def = add_node_view->node();

  ContractionWithBiasAddAndAdd base;
  if (!FindContractionWithBiasAddAndAdd(ctx, add_node_view->node_index(),
                                        &base) ||
      !HasAtMostOneFanoutAtPort0(*add_node_view) ||
      !HaveSameDataType(node_def, add_node_def) ||
      IsInPreserveSet(ctx, add_node_def)) {
    return false;
  }

  // TODO(plugin): Public TF doesn't have MatMul + LeakyRelu fusion, remove this
  // limitation once it's supported.
  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsLeakyRelu(*node_def) && IsMatMul(*contraction_def)) return false;

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add,
      base.port_id,     node_index,    base.bias_port};
  *matched = pattern;

  return true;
}

// ============================================================================
// safe_embedding_lookup_sparse pattern matching.
//
// Matches the subgraph:
//   SelectV2
//     ├── Tile
//     │     └── Reshape
//     │           └── SparseFillEmptyRows:2 (empty_row_indicator)
//     ├── ZerosLike
//     │     └── SparseSegment{Sum,Mean,SqrtN}
//     └── SparseSegment{Sum,Mean,SqrtN}
//           ├── Identity|Const (embedding_table)
//           ├── SparseFillEmptyRows:1 (filled_values)
//           └── StridedSlice
//                 └── SparseFillEmptyRows:0 (filled_indices)
// ============================================================================
bool IsSparseSegmentOp(const NodeDef& node) {
  return node.op() == "SparseSegmentSum" || node.op() == "SparseSegmentMean" ||
         node.op() == "SparseSegmentSqrtN";
}

string GetCombinerFromSparseSegmentOp(const string& op) {
  if (op == "SparseSegmentSum") return "sum";
  if (op == "SparseSegmentMean") return "mean";
  return "sqrtn";
}

constexpr int kSparseFillEmptyRowsIndicatorPort = 2;

bool FindSafeEmbeddingLookupSparse(const RemapperContext& ctx, int node_index,
                                   SafeEmbeddingLookupSparse* matched) {
  const auto* select_node_view = ctx.graph_view.GetNode(node_index);
  const auto* select_node_def = select_node_view->node();

  // Root must be SelectV2.
  if (!IsSelect(*select_node_def)) return false;
  if (HasControlFaninOrFanout(*select_node_view)) return false;
  if (select_node_view->NumRegularFanins() != 3) return false;

  // Input 0: Tile (broadcast empty_row_indicator)
  const auto& fanin_0 = select_node_view->GetRegularFanin(0);
  const auto* tile_node_view = fanin_0.node_view();
  const auto* tile_node_def = tile_node_view->node();
  if (tile_node_def->op() != "Tile") return false;

  // Input 1: ZerosLike
  const auto& fanin_1 = select_node_view->GetRegularFanin(1);
  const auto* zeros_like_node_view = fanin_1.node_view();
  const auto* zeros_like_node_def = zeros_like_node_view->node();
  if (zeros_like_node_def->op() != "ZerosLike") return false;

  // Input 2: SparseSegment{Sum,Mean,SqrtN}
  const auto& fanin_2 = select_node_view->GetRegularFanin(2);
  const auto* sparse_seg_node_view = fanin_2.node_view();
  const auto* sparse_seg_node_def = sparse_seg_node_view->node();
  if (!IsSparseSegmentOp(*sparse_seg_node_def)) return false;

  // ZerosLike must feed from the SAME SparseSegment op.
  if (zeros_like_node_view->NumRegularFanins() < 1) return false;
  const auto& zl_fanin = zeros_like_node_view->GetRegularFanin(0);
  if (zl_fanin.node_view()->node_index() !=
      sparse_seg_node_view->node_index()) {
    return false;
  }

  // SparseSegment must have 3 inputs:
  //   0: embedding_table (Identity or Const)
  //   1: SparseFillEmptyRows:1 (filled_values)
  //   2: StridedSlice (segment_ids from SparseFillEmptyRows:0)
  if (sparse_seg_node_view->NumRegularFanins() < 3) return false;

  const auto& seg_fanin_0 = sparse_seg_node_view->GetRegularFanin(0);
  const auto* table_node_view = seg_fanin_0.node_view();
  const auto* table_node_def = table_node_view->node();

  const auto& seg_fanin_1 = sparse_seg_node_view->GetRegularFanin(1);
  const auto* sfr_values_node_view = seg_fanin_1.node_view();
  const auto* sfr_values_node_def = sfr_values_node_view->node();

  const auto& seg_fanin_2 = sparse_seg_node_view->GetRegularFanin(2);
  const auto* strided_slice_node_view = seg_fanin_2.node_view();
  const auto* strided_slice_node_def = strided_slice_node_view->node();

  // Validate: input 1 must come from SparseFillEmptyRows (output port 1).
  if (sfr_values_node_def->op() != "SparseFillEmptyRows") return false;
  if (seg_fanin_1.index() != 1) return false;

  // Validate: StridedSlice's input should come from SparseFillEmptyRows:0.
  if (strided_slice_node_def->op() != "StridedSlice") return false;
  if (strided_slice_node_view->NumRegularFanins() < 1) return false;
  const auto& ss_fanin = strided_slice_node_view->GetRegularFanin(0);
  if (ss_fanin.node_view()->node_index() !=
      sfr_values_node_view->node_index()) {
    return false;
  }
  if (ss_fanin.index() != 0) return false;

  // Validate: Tile -> Reshape -> SparseFillEmptyRows:2 (same SFR node).
  if (tile_node_view->NumRegularFanins() < 1) return false;
  const auto& tile_fanin_0 = tile_node_view->GetRegularFanin(0);
  const auto* reshape_node_view = tile_fanin_0.node_view();
  const auto* reshape_node_def = reshape_node_view->node();
  if (reshape_node_def->op() != "Reshape") return false;

  if (reshape_node_view->NumRegularFanins() < 1) return false;
  const auto& reshape_fanin_0 = reshape_node_view->GetRegularFanin(0);
  if (reshape_fanin_0.node_view()->node_index() !=
      sfr_values_node_view->node_index()) {
    return false;
  }
  if (reshape_fanin_0.index() != kSparseFillEmptyRowsIndicatorPort)
    return false;

  // Validate: embedding table is Identity or Const.
  if (table_node_def->op() != "Identity" && table_node_def->op() != "Const") {
    return false;
  }

  // All checks passed — populate the match.
  matched->select_v2 = select_node_view->node_index();
  matched->tile = tile_node_view->node_index();
  matched->reshape = reshape_node_view->node_index();
  matched->zeros_like = zeros_like_node_view->node_index();
  matched->sparse_segment = sparse_seg_node_view->node_index();
  matched->sparse_fill_empty_rows = sfr_values_node_view->node_index();
  matched->strided_slice = strided_slice_node_view->node_index();
  matched->embedding_table = table_node_view->node_index();
  matched->combiner = GetCombinerFromSparseSegmentOp(sparse_seg_node_def->op());

  // --- Try to absorb upstream GatherV2 + Where + GreaterEqual chain ---
  // Pattern: sfr.input(0) = GatherV2(SparseReshape:0, Reshape, Const)
  //          sfr.input(1) = GatherV2(FloorMod, Reshape, Const)
  //          Both GatherV2s share the same Reshape(Where(GreaterEqual(FloorMod,
  //          0)))
  if (sfr_values_node_view->NumRegularFanins() >= 2) {
    const auto& sfr_fanin_0 = sfr_values_node_view->GetRegularFanin(0);
    const auto& sfr_fanin_1 = sfr_values_node_view->GetRegularFanin(1);
    const auto* gv2_idx_view = sfr_fanin_0.node_view();
    const auto* gv2_val_view = sfr_fanin_1.node_view();
    const auto* gv2_idx_def = gv2_idx_view->node();
    const auto* gv2_val_def = gv2_val_view->node();

    if (gv2_idx_def->op() == "GatherV2" && gv2_val_def->op() == "GatherV2" &&
        gv2_idx_view->NumRegularFanins() >= 2 &&
        gv2_val_view->NumRegularFanins() >= 2) {
      // GatherV2[indices].input(1) and GatherV2[values].input(1) should be
      // the same Reshape node.
      const auto& gv2_idx_fanin1 = gv2_idx_view->GetRegularFanin(1);
      const auto& gv2_val_fanin1 = gv2_val_view->GetRegularFanin(1);
      const auto* filt_reshape_view = gv2_idx_fanin1.node_view();
      const auto* filt_reshape_def = filt_reshape_view->node();

      if (filt_reshape_def->op() == "Reshape" &&
          gv2_idx_fanin1.node_view()->node_index() ==
              gv2_val_fanin1.node_view()->node_index() &&
          filt_reshape_view->NumRegularFanins() >= 1) {
        // Reshape.input(0) = Where
        const auto& reshape_fanin = filt_reshape_view->GetRegularFanin(0);
        const auto* where_view = reshape_fanin.node_view();
        const auto* where_def = where_view->node();

        if (where_def->op() == "Where" && where_view->NumRegularFanins() >= 1) {
          // Where.input(0) = GreaterEqual
          const auto& where_fanin = where_view->GetRegularFanin(0);
          const auto* ge_view = where_fanin.node_view();
          const auto* ge_def = ge_view->node();

          if (ge_def->op() == "GreaterEqual" &&
              ge_view->NumRegularFanins() >= 1) {
            // GreaterEqual.input(0) should be the same as
            // GatherV2[values].input(0) (both are FloorMod).
            const auto& ge_fanin0 = ge_view->GetRegularFanin(0);
            const auto& gv2_val_fanin0 = gv2_val_view->GetRegularFanin(0);

            if (ge_fanin0.node_view()->node_index() ==
                gv2_val_fanin0.node_view()->node_index()) {
              // Full chain matched!
              matched->gv2_indices = gv2_idx_view->node_index();
              matched->gv2_values = gv2_val_view->node_index();
              matched->filter_reshape = filt_reshape_view->node_index();
              matched->filter_where = where_view->node_index();
              matched->filter_ge = ge_view->node_index();
              matched->has_upstream_filter = true;
            }
          }
        }
      }
    }
  }

  // --- Try to absorb downstream adjust_shape chain ---
  // Pattern: SelectV2 -> Shape -> Slice (embed dim)
  //          SelectV2 -> Reshape (final output)
  //          Cast(orig_dense_shape) -> Slice (batch dims)
  //          ConcatV2(batch_slice, embed_slice) -> Reshape.input(1)
  {
    // Check if SelectV2 has exactly 2 consumers: Shape and Reshape.
    const auto& select_fanouts = select_node_view->GetRegularFanouts();
    const utils::MutableNodeView* ds_shape_view = nullptr;
    const utils::MutableNodeView* ds_reshape_view = nullptr;

    // Collect all fanout consumers of output port 0.
    if (!select_fanouts.empty()) {
      for (const auto& consumer : select_fanouts[0]) {
        const auto* cv = consumer.node_view();
        if (cv->node()->op() == "Shape") ds_shape_view = cv;
        if (cv->node()->op() == "Reshape") ds_reshape_view = cv;
      }
    }

    if (ds_shape_view != nullptr && ds_reshape_view != nullptr) {
      // Shape must have exactly 1 consumer: a Slice node.
      bool shape_ok = false;
      const utils::MutableNodeView* ds_slice_embed_view = nullptr;
      const auto& shape_fanouts = ds_shape_view->GetRegularFanouts();
      if (!shape_fanouts.empty() && shape_fanouts[0].size() == 1) {
        ds_slice_embed_view = shape_fanouts[0][0].node_view();
        if (ds_slice_embed_view->node()->op() == "Slice") {
          shape_ok = true;
        }
      }

      if (shape_ok && ds_slice_embed_view != nullptr) {
        // Slice(embed) must feed into a ConcatV2.
        const utils::MutableNodeView* ds_concat_view = nullptr;
        const auto& slice_fanouts = ds_slice_embed_view->GetRegularFanouts();
        if (!slice_fanouts.empty()) {
          for (const auto& consumer : slice_fanouts[0]) {
            if (consumer.node_view()->node()->op() == "ConcatV2") {
              ds_concat_view = consumer.node_view();
            }
          }
        }

        if (ds_concat_view != nullptr) {
          // ConcatV2 must feed into the Reshape as shape input (input[1]).
          if (ds_reshape_view->NumRegularFanins() >= 2) {
            const auto& reshape_shape_fanin =
                ds_reshape_view->GetRegularFanin(1);
            if (reshape_shape_fanin.node_view()->node_index() ==
                ds_concat_view->node_index()) {
              // Find the batch-dims Slice in ConcatV2 inputs.
              // ConcatV2 has: Slice(batch_dims), Slice(embed_dims), axis_const
              const utils::MutableNodeView* ds_slice_batch_view = nullptr;
              for (int ci = 0; ci < ds_concat_view->NumRegularFanins(); ++ci) {
                const auto& cf = ds_concat_view->GetRegularFanin(ci);
                if (cf.node_view()->node()->op() == "Slice" &&
                    cf.node_view()->node_index() !=
                        ds_slice_embed_view->node_index()) {
                  ds_slice_batch_view = cf.node_view();
                  break;
                }
              }

              if (ds_slice_batch_view != nullptr) {
                // Slice(batch) input[0] should be Cast(orig_dense_shape).
                if (ds_slice_batch_view->NumRegularFanins() >= 1) {
                  const auto& batch_src =
                      ds_slice_batch_view->GetRegularFanin(0);
                  const auto* cast_view = batch_src.node_view();
                  // Accept Cast or direct dense_shape source.
                  string orig_ds_input;
                  if (cast_view->node()->op() == "Cast" &&
                      cast_view->NumRegularFanins() >= 1) {
                    // orig_dense_shape is Cast's input.
                    orig_ds_input = cast_view->node()->input(0);
                  } else {
                    orig_ds_input = ds_slice_batch_view->node()->input(0);
                  }

                  // All checks passed for adjust_shape.
                  matched->adjust_shape_node = ds_shape_view->node_index();
                  matched->adjust_slice_node =
                      ds_slice_embed_view->node_index();
                  matched->adjust_concat_node = ds_concat_view->node_index();
                  matched->adjust_reshape_node = ds_reshape_view->node_index();
                  matched->adjust_cast_slice_node =
                      ds_slice_batch_view->node_index();
                  matched->has_adjust_shape = true;
                  matched->orig_dense_shape_input = orig_ds_input;
                }
              }
            }
          }
        }
      }
    }
  }

  return true;
}

bool FindPadWithContraction(const RemapperContext& ctx, int node_index,
                            PadWithContraction* matched,
                            bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a Conv or FusedConv.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root node must be (_Fused)Conv2D/(_Fused)DepthwiseConv2dNative.
  const auto* node_def = node_view->node();
  const bool is_ok = IsConv2D(*node_def) || node_def->op() == kFusedConv2D ||
                     IsDepthwiseConv2dNative(*node_def) ||
                     node_def->op() == kFusedDepthwiseConv2dNative;
  if (!is_ok) {
    return false;
  }

  // Input to the contraction must be Pad.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* pad_node_view = regular_fanin_0.node_view();
  const auto* pad_node_def = pad_node_view->node();

  // Only Pad is allowed, PadV2 will be prevented.
  if (pad_node_def->op() != "Pad") return false;

  // Only fuse contraction with `VALID` and 'EXPLICIT' padding.
  // TODO(plugin): Support more padding type in future.
  string padding_str;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "padding", &padding_str));
  if (padding_str == "SAME") return false;

  // Only fuse contraction with INT32 padding.
  // TODO(plugin): support INT64 padding in future.
  if (!HasDataType(pad_node_def, DT_INT32, "Tpaddings")) return false;

  // If contraction has been fused, only fuse it with Pad if, Conv is fused with
  // only Bias.
  if (node_def->op() == kFusedConv2D) {
    int num_args;
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "num_args", &num_args));
    if (num_args != 1) return false;
  }

  if (!HaveSameDataType(node_def, pad_node_def) ||
      HasControlFaninOrFanout(*pad_node_view) ||
      !HasAtMostOneFanoutAtPort0(*pad_node_view) ||
      IsInPreserveSet(ctx, pad_node_def))
    return false;

  const PadWithContraction pattern{pad_node_view->node_index(), node_index};

  // We successfully found a Pad + (_Fused)Conv2D/DepthwiseConv2dNative pattern.
  *matched = pattern;

  return true;
}

inline bool VerifyConstants(RemapperContext* ctx,
                            std::map<string, int>* nodes_map,
                            std::map<string, float>* values_map) {
  using utils::MutableNodeView;
  for (auto it = values_map->begin(); it != values_map->end(); ++it) {
    int node_idx = nodes_map->at(it->first);
    MutableNodeView* node_view = ctx->graph_view.GetNode(node_idx);
    NodeDef* node_def = node_view->node();
    Tensor const_tensor;

    // Check if node is Const or Cast.
    if (node_def != nullptr &&
        (node_def->op() == "Cast" || node_def->op() == "Const")) {
      // If node is a Cast, look for Const in fan-ins.
      if (node_def->op() == "Cast") {
        const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
        const auto* regular_node_view = regular_fanin_0.node_view();
        node_def = regular_node_view->node();
      }
      // Verify if the node is a constant.
      if (node_def->op() == "Const") {
        TF_CHECK_OK(GetTensorFromConstant(node_def, &const_tensor));
        if (const_tensor.NumElements() == 1) {
          DataType dtype = const_tensor.dtype();
          if (!(dtype == DT_FLOAT || dtype == DT_BFLOAT16)) return false;
          auto const_value = (dtype == DT_FLOAT)
                                 ? const_tensor.flat<float>()(0)
                                 : const_tensor.flat<Eigen::bfloat16>()(0);
          // To compare float.
          if (std::abs(const_value - it->second) > 1e-2f) {
            return false;
          }
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool IsMatchedMatMulBiasAddAndGeluExact(
    RemapperContext& ctx, int node_index,
    std::map<string, int>* matched_nodes_map = nullptr,
    std::set<int>* remove_node_indices = nullptr, bool* expand_dims = nullptr) {
  auto* node_view = ctx.graph_view.GetNode(node_index);
  using utils::MatchingDirection;
  using utils::NodeStatus;
  int found_pattern_index = 0;  // Default = 0 means no pattern found.
  std::vector<utils::OpTypePattern> gelu_exact_patterns;
  // clang-format off
  // Pattern 1:
  //    Const: 1/sqrt(2)        Const: 1    Const: 1/2
  //                  \               \         \
  //  * --> BiasAdd --> Mul --> Erf --> AddV2 --> Mul --> Mul
  //        /       \____________________________________/
  //  MatMul
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_x_sqrt_one_half",
                     NodeStatus::kRemove,
                      {
                        {"BiasAdd", "bias_add", NodeStatus::kRemove},
                        {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                      }
                    }  // Mul: "bias_add_x_sqrt_one_half"
                  }
                },  // Erf: "erf"
                {"Cast|Const", "one", NodeStatus::kRemain}
              }  // Add|AddV2: "erf_plus_one"
            },
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: "erf_plus_one_times_one_half"
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
          {
            {"MatMul", "matmul", NodeStatus::kRemove},
            {"*", "bias", NodeStatus::kRemain}
          }
        }  // BiasAdd: "bias_add"
      }  // Mul: "output"
    });

  // Pattern 2:
  //  Cast|Const: 1/sqrt(2)    Cast|Const: 1
  //                  \               \
  //  * --> BiasAdd --> Mul --> Erf --> Add|AddV2 --> Mul
  //      /         \                                 /
  // MatMul           ----------------------------> Mul
  //                                                /
  //                                  Cast|Const: 1/2
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
          {
            {"Erf", "erf", NodeStatus::kRemove,
              {
                {"Mul", "bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"BiasAdd", "bias_add", NodeStatus::kRemove},
                    {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                  }
                }  // Mul: "bias_add_x_sqrt_one_half"
              }
            },  // Erf: "erf"
            {"Cast|Const", "one", NodeStatus::kRemain}
          }
        },  // Add|AddV2: "erf_plus_one"
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"BiasAdd", "bias_add", NodeStatus::kRemove,
              {
                {"MatMul", "matmul", NodeStatus::kRemove},
                {"*", "bias", NodeStatus::kRemain}
              }
            },  // BiasAdd: "bias_add"
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        }  // Mul: "erf_plus_one_times_one_half"
      }
    });  // Mul: "output"

  // Pattern 3:
  //    Const: expand_dims  1/sqrt(2)        Const: 1    Const: 1/2
  //                 \           \               \         \
  //  _FusedMatMul --> Reshape --> Mul --> Erf --> AddV2 --> Mul --> Mul
  //                           \____________________________________/
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                      {
                        {"Reshape", "reshape", NodeStatus::kRemove},
                        {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                      }
                    }  // Mul: "bias_add_x_sqrt_one_half"
                  }
                },  // Erf: "erf"
                {"Cast|Const", "one", NodeStatus::kRemain}
              }  // Add|AddV2: "erf_plus_one"
            },
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: "erf_plus_one_times_one_half"
        {"Reshape", "reshape", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Cast|Const", "expand_dims", NodeStatus::kRemain}
          }
        }
      }
    });  // Mul: "output"

  // Pattern 4: Erfc
  //                                     Const: 1/sqrt(2) Const: 1/2
  //                                      \                \
  //  _FusedMatMul --> Reshape --> Neg --> Mul --> Erfc --> Mul --> Mul
  //                            \____________________________________/
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "one_half_x_erfc", NodeStatus::kRemove,
          {
            {"Const|Cast", "one_half", NodeStatus::kRemain},
            {"Erfc", "erfc", NodeStatus::kRemove,
              {
                {"Mul", "neg_bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"Const|Cast", "sqrt_one_half", NodeStatus::kRemain},
                    {"Neg", "neg", NodeStatus::kRemove,
                      {{"Reshape", "reshape", NodeStatus::kRemove}}
                    },  // Neg: "neg"
                  }
                }  // Mul: "neg_bias_add_x_sqrt_one_half"
              }  // Erfc: "erfc"
            }
          }  // Mul: "one_half_x_erfc"
        },
        {"Reshape", "reshape", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Const|Cast", "expand_dims", NodeStatus::kRemain}
          }
        }
      }
    });  // Mul: "output"

  // clang-format on
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx.graph_view));
  // Find GeluExact
  std::map<string, int> dummy_matched_nodes_map;
  std::set<int> dummy_remove_node_indices;
  if (!matched_nodes_map) matched_nodes_map = &dummy_matched_nodes_map;
  if (!remove_node_indices) remove_node_indices = &dummy_remove_node_indices;
  bool found_gelu_exact = false;

  for (size_t pattern = 0; pattern < gelu_exact_patterns.size(); pattern++) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    if (graph_matcher.GetMatchedNodes(gelu_exact_patterns[pattern],
                                      ctx.nodes_to_preserve, node_view,
                                      matched_nodes_map, remove_node_indices)) {
      found_gelu_exact = true;
      found_pattern_index = pattern + 1;
      break;
    }
  }
  if (found_pattern_index == 3 || found_pattern_index == 4) {
    *expand_dims = true;
  }
  return found_gelu_exact;
}

// Gelu in python api generates a number of nodes in the graph. Depending on the
// parmeter `approximate={True/False}` different types of ops are generated. We
// distinguish them as `GeluExact` that uses Erf and `GeluApproximate` that
// uses Tanh.
bool FindMatMulBiasAddAndGelu(RemapperContext* ctx, int node_index,
                              std::map<string, int>* matched_nodes_map,
                              std::set<int>* remove_node_indices,
                              bool* is_gelu_approximate, bool* expand_dims) {
  using utils::MatchingDirection;
  using utils::NodeStatus;

  bool found_gelu_exact = false;
  bool found_gelu_approximate = false;
  int found_pattern_index = 0;
  std::vector<utils::OpTypePattern> gelu_approximate_patterns;

  // Find GeluExact
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_gelu_exact = IsMatchedMatMulBiasAddAndGeluExact(
      *ctx, node_index, matched_nodes_map, remove_node_indices, expand_dims);
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  // clang-format off
  // SubGraph gpu Pattern
  //            Const: 3  Empirical Const
  //                \      \
  //  FusedMatMul --> Pow --> Mul
  utils::OpTypePattern subgraph_gpu =
    {"Mul", "mul", NodeStatus::kRemove,
      {
        {"Pow", "pow", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Const", "three", NodeStatus::kRemain}
          }
        },
        {"Const", "empirical_const", NodeStatus::kRemain}
      }
    };

  // SubGraph cpu Pattern
  //            Empirical Const
  //                   |
  //             ------Mul------
  //            /               \
  //  FusedMatMul               Mul
  //            \               /
  //             -----Square----
  utils::OpTypePattern subgraph_cpu =
    {"Mul", "mul", NodeStatus::kRemove,
      {
        {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
          {
            {"Const", "empirical_const", NodeStatus::kRemain},
            {"_FusedMatMul", "matmul", NodeStatus::kRemove}
          }
        },
        {"Square", "square", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove}
          }
        }
      }
    };
  // clang-format on

  utils::MutableNodeView* node_view = ctx->graph_view.GetNode(node_index);
  const NodeDef* node_def = node_view->node();
  bool root_on_gpu = NodeIsOnGpu(node_def);
  utils::OpTypePattern* subgraph_pattern =
      root_on_gpu ? &subgraph_gpu : &subgraph_cpu;

  // clang-format off
  // Pattern 1:
  //             Const: 1/sqrt(2)       Const: 1    Const: 1/2
  //            SubGraph     \                \        \
  //                \         \                \        \
  //  FusedMatMul --> AddV2 --> Mul --> Tanh --> AddV2 --> Mul --> Mul
  //            \________________________________________________/
  gelu_approximate_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
              {
                {"Tanh", "tanh", NodeStatus::kRemove,
                  {
                    {"Mul",
                     "matmul_plus_mul_times_square_root_two_over_pi",
                     NodeStatus::kRemove,
                      {
                        {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                          {
                            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                            *subgraph_pattern
                          }
                        },  // Add|AddV2: matmul_plus_mul
                        {"Const",
                         "square_root_two_over_pi",
                         NodeStatus::kRemain}
                      }
                    }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                  }
                },  // Tanh: tanh
                {"Const", "one", NodeStatus::kRemain}
              }
            },  // Add|AddV2: tanh_plus_one
            {"Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: tanh_plus_one_times_one_half
        {"_FusedMatMul", "matmul", NodeStatus::kRemove}
      }
    });  // Mul: output

  // Pattern 2:
  //    Const:                      Emperical     1/sqrt(2)             1
  //                                   \                 \               \
  //  FusedMatMul --> Square --> Mul --> Mul --> AddV2 --> Mul --> Tanh -->
  //  AddV2  --> Mul
  //           \\\_______________/              /                            /
  //            \\_____________________________/                            /
  //             \______________________________________________________ Mul
  //                                                                      /
  //                                                                  Const: 1/2
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Const", "one_half", NodeStatus::kRemain},
          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
        {
          {"Tanh", "tanh", NodeStatus::kRemove,
            {
              {"Mul",
               "matmul_plus_mul_times_square_root_two_over_pi",
               NodeStatus::kRemove,
                {
                  {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                    {
                      {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                      {"Mul", "mul", NodeStatus::kRemove,
                        {
                          {"Const", "empirical_const", NodeStatus::kRemain},
                          {"Mul",
                           "empirical_const_times_matmul",
                           NodeStatus::kRemove,
                            {
                              {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                              {"Square", "square", NodeStatus::kRemove,
                                {
                                  {"_FusedMatMul", "matmul",
                                   NodeStatus::kRemove}
                                }
                              }  // Square: square
                            }
                          }  // Mul: empirical_const_times_matmul
                        }
                      }  // Mul: mul
                    }
                  },  // Add|AddV2: matmul_plus_mul
                  {"Const",
                   "square_root_two_over_pi",
                   NodeStatus::kRemain}
                }
              }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
            }
          },  // Tanh: tanh
          {"Const", "one", NodeStatus::kRemain}
        }
      }  // Add|AddV2: tanh_plus_one
    }
  });  // Mul: output

  // Pattern 3:
  //    Const:   ExpandDims                   Emperical        1/sqrt(2)
  //    Const: 1    Const: 1/2
  //                \                              \                 \
  //                \         \
  //  FusedMatMul --> Reshape --> Square --> Mul --> Mul --> AddV2 --> Mul -->
  //  Tanh --> AddV2 --> Mul --> Mul
  //                      \\\_______________/              /
  //                                            /
  //                       \\_____________________________/
  //                                            /
  //                        \______________________________________________/
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
            {
              {"Tanh", "tanh", NodeStatus::kRemove,
                {
                  {"Mul",
                   "matmul_plus_mul_times_square_root_two_over_pi",
                   NodeStatus::kRemove,
                    {
                      {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                        {
                          {"Reshape", "reshape", NodeStatus::kRemove},
                          {"Mul", "mul", NodeStatus::kRemove,
                            {
                              {"Mul",
                               "empirical_const_times_matmul",
                               NodeStatus::kRemove,
                                {
                                  {"Reshape", "reshape", NodeStatus::kRemove,
                                    {
                                      {"_FusedMatMul", "matmul",
                                       NodeStatus::kRemove},
                                      {"Cast|Const",
                                       "expand_dims",
                                       NodeStatus::kRemain}
                                    }
                                  },  // Reshape: reshape
                                  {"Square", "square", NodeStatus::kRemove,
                                    {
                                      {"Reshape", "reshape",
                                       NodeStatus::kRemove}
                                    }
                                  }  // Square: square
                                }
                              },  // Mul: empirical_const_times_matmul
                              {"Cast|Const",
                               "empirical_const",
                               NodeStatus::kRemain}
                            }
                          }  // Mul: mul
                        }
                      },  // Add|AddV2: matmul_plus_mul
                      {"Cast|Const",
                       "square_root_two_over_pi",
                       NodeStatus::kRemain}
                    }
                  }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                }
              },  // Tanh: tanh
              {"Cast|Const", "one", NodeStatus::kRemain}
            }
          },  // Add|AddV2: tanh_plus_one
          {"Cast|Const", "one_half", NodeStatus::kRemain}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"Reshape", "reshape", NodeStatus::kRemove}
    }
  });  // Mul: output

  // Pattern BF16:
  //    Const:                    Emperical                1/sqrt(2)
  //    1
  //                                   \                         \
  //                                   \
  //  FusedMatMul --> Cast --> Square --> Mul --> Mul --> AddV2 --> Mul -->
  //  Cast --> Tanh --> AddV2  --> Mul
  //            \      \\\______________________/       /
  //                                             /
  //             \      \\_____________________________/
  //                                             /
  //              \______________________________________________________ Mul
  //                                                                      /
  //                                                                  Const: 1/2
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Const", "one_half", NodeStatus::kRemain},
          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"AddV2", "tanh_plus_one", NodeStatus::kRemove,
        {
          {"Tanh", "tanh", NodeStatus::kRemove,
            {
              {"Cast", "cast1", NodeStatus::kRemove,
                {
                  {"Mul",
                   "matmul_plus_mul_times_square_root_two_over_pi",
                   NodeStatus::kRemove,
                    {
                      {"AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                        {
                          {"Cast", "cast", NodeStatus::kRemove,
                            {
                              {"_FusedMatMul", "matmul",
                               NodeStatus::kRemove}
                            }
                          },  // Cast: cast
                          {"Mul", "mul", NodeStatus::kRemove,
                            {
                              {"Cast", "cast", NodeStatus::kRemove,
                                {
                                  {"_FusedMatMul", "matmul",
                                   NodeStatus::kRemove}
                                }
                              },  // Cast: cast
                              {"Mul",
                               "empirical_const_times_matmul",
                               NodeStatus::kRemove,
                                {
                                  {"Const",
                                   "empirical_const",
                                   NodeStatus::kRemain},
                                  {"Square", "square", NodeStatus::kRemove,
                                    {
                                      {"Cast", "cast", NodeStatus::kRemove,
                                        {
                                          {"_FusedMatMul", "matmul",
                                           NodeStatus::kRemove}
                                        }
                                      }  // Cast: cast
                                    }
                                  }  // Square: square
                                }
                              }  // Mul: empirical_const_times_matmul
                            }
                          }  // Mul: mul
                        }
                      },  // AddV2: matmul_plus_mul
                      {"Const",
                       "square_root_two_over_pi",
                       NodeStatus::kRemain}
                    }
                  }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                }
              }  // Cast: cast1
            }
          },  // Tanh: tanh
          {"Const", "one", NodeStatus::kRemain}
        }
      }  // AddV2: tanh_plus_one
    }
  });  // Mul: output

  // clang-format on
  // Find GeluApproximate
  if (!found_gelu_exact) {
    for (size_t pattern = 0; pattern < gelu_approximate_patterns.size();
         pattern++) {
      matched_nodes_map->clear();
      remove_node_indices->clear();
      if (graph_matcher.GetMatchedNodes(
              gelu_approximate_patterns[pattern], ctx->nodes_to_preserve,
              node_view, matched_nodes_map, remove_node_indices)) {
        found_gelu_approximate = true;
        found_pattern_index = pattern + 1;
        break;
      }
    }
    if (found_pattern_index == 3) *expand_dims = true;
  }

  *is_gelu_approximate = found_gelu_approximate ? true : false;
  // Pattern matcher does subgraph matching based on op types only. The matcher
  // also does a sanity check on nodes tagged as `kRemove`, i.e., they do not
  // have any consumer outside the matched nodes. In order to replace the
  // subgraph, we need additional checks, for example, if the key ops have been
  // placed on CPU or GPU, desired data type, const has desired value etc. For
  // the following fusion: MatMul + BiasAdd + Gelu (disintegrated into smaller
  // ops), we check if (i) MatMul op is CpuCompatible or GpuComptible, (ii)
  // const nodes have desired values.
  if (found_gelu_exact) {
    // Check if the MatMul to be fused is device compatible.
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
      return false;
    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    bool cpu_ok = matmul_node->attr().contains("transpose_a") &&
                  !matmul_node->attr().at("transpose_a").b();
    if (!cpu_ok) return false;

    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"sqrt_one_half", 0.707106},
                                          {"one_half", 0.5}};

    // Gelu exact pattern with TF version 2.19 does not have the constant one.
    // Check is added to see if the matched pattern has this constant, if True
    // it is added. For TF version 2.18 it is added.
    if (matched_nodes_map->find("one") != matched_nodes_map->end()) {
      values_map["one"] = 1.0;
    }
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else if (*is_gelu_approximate) {
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    if (NodeIsOnCpu(matmul_node) &&
        matmul_node->attr().contains("transpose_a") &&
        matmul_node->attr().at("transpose_a").b()) {
      return false;
    }

    // Check if _FusedMatMul contains only BiasAdd
    auto fused_ops = matmul_node->attr().at("fused_ops").list().s();
    if (fused_ops.size() == 1) {
      if (fused_ops.at(0) != "BiasAdd") return false;
    } else {
      return false;
    }
    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"square_root_two_over_pi", 0.797884},
                                          {"one", 1.0},
                                          {"one_half", 0.5},
                                          {"empirical_const", 0.044715}};

    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else {
    return false;
  }

  return (found_gelu_exact || *is_gelu_approximate);
}

bool FindBatchMatMulBiasAddActivation(
    RemapperContext& ctx, int node_index,
    BatchMatMulWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an Relu node.
  // TODO (plugin): Add support for other activations.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;

  if (!IsRelu(*node_def)) return false;

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // And input to the activation node must match BatchMatMulWithBiasAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  BatchMatMulWithBiasAdd base;
  if (!FindBatchMatMulWithBiasAdd(ctx, bias_add_node_view->node_index(),
                                  &base) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      (!HaveSameDataType(node_def, bias_add_node_def) &&
       !(GetDataTypeFromAttr(*node_def, "T") == DT_FLOAT)) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // Verify the inter node has control fanin&fanout or not.
  if (HasControlFaninOrFanout(*bias_add_node_view)) {
    return false;
  }

  const auto* batchmatmul_node_view = ctx.graph_view.GetNode(base.batch_matmul);

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*batchmatmul_node_view)) return false;

  const BatchMatMulWithBiasAddAndActivation pattern{
      base.batch_matmul, base.bias_add, node_index, base.bias_port};

  // We successfully found a BatchMatMul+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindFusedBatchMatMul(RemapperContext* ctx, int node_index,
                          std::map<string, int>* matched_nodes_map,
                          std::set<int>* remove_node_indices,
                          std::vector<string>* input_node_names) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  // The fusion is only supported for FP32.
  if (!HasDataType(node_def, DT_FLOAT)) return false;
  using utils::MatchingDirection;
  using utils::NodeStatus;
  int found_pattern_index = 0;  // Default = 0 means no pattern found.
  std::vector<utils::OpTypePattern> fusion_patterns;
  // clang-format off
  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"Mul", "mul", NodeStatus::kRemove,
          {
            {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove},
            {"*", "multiplicand", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    });

  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove,
          {
            {"Mul", "mul", NodeStatus::kRemove,
              {
                {"*", "mul_input0", NodeStatus::kRemain},
                {"Const|Cast", "multiplicand", NodeStatus::kRemain}
              }
            },
            {"*", "bmm_input1", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    });

  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"*", "addend", NodeStatus::kRemain},
        {"Mul", "mul", NodeStatus::kRemove,
          {
            {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove},
            {"*", "multiplicand", NodeStatus::kRemain}
          }
        }
      }
    });
  // clang-format on

  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));

  bool found_op_type_match = false;

  for (size_t pattern_iterator = 0; pattern_iterator < fusion_patterns.size();
       pattern_iterator++) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = graph_matcher.GetMatchedNodes(
        fusion_patterns[pattern_iterator], ctx->nodes_to_preserve,
        ctx->graph_view.GetNode(node_index), matched_nodes_map,
        remove_node_indices);
    if (found_op_type_match) {
      found_pattern_index = pattern_iterator + 1;
      break;
    }
  }

  // ZenDNN is not optimized for all shapes with regard to binary-post ops
  // fusion. Allow limited cases only for now that are optimized, (i)
  // multiplicand is scalar, (ii) BatchMatmulV2 output is 4D tensor, and (iii)
  // addend is 4D tensor with shape same as BatchMatmulV2 output since
  // broadcasting is not supported.

  // Currently ZenDNN supports binary-post ops fusion for addend tensor with 2D
  // only.
  // TODO(plugin): Enable this when ZenDNN supports 3D and 4D addend tensor.
  if (!found_op_type_match) return false;
  if (!ctx->inferred_graph_properties) {
    Status s = ctx->graph_properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/false,
        /*include_output_tensor_values=*/true);
    if (!s.ok()) return false;
    ctx->inferred_graph_properties = true;
  }
  std::vector<OpInfo_TensorProperties> multiplicand_props;
  NodeDef* multiplicand_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("multiplicand"))->node();
  TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
      multiplicand_node_def->name(), &multiplicand_props));
  if (NumCoefficients(multiplicand_props[0].shape()) != 1) return false;

  NodeDef* batch_matmul_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("batch_matmul"))->node();
  DCHECK(IsAnyBatchMatMul(*batch_matmul_node_def));  // Expected BatchMatMul op.
  if (!NodeIsOnCpu(batch_matmul_node_def)) return false;

  std::vector<OpInfo_TensorProperties> batch_matmul_props, input_props,
      addend_props;
  TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
      batch_matmul_node_def->name(), &batch_matmul_props));
  if (Rank(batch_matmul_props[0].shape()) != 4) return false;

  TF_ABORT_IF_ERROR(ctx->graph_properties.GetInputProperties(
      batch_matmul_node_def->name(), &input_props));
  if (input_props.size() < 2 || Rank(input_props[0].shape()) != 4 ||
      Rank(input_props[1].shape()) != 4)
    return false;

  auto& lhs_shape = input_props[0].shape();
  auto& rhs_shape = input_props[1].shape();
  if (lhs_shape.dim(0).size() * lhs_shape.dim(1).size() !=
      rhs_shape.dim(0).size() * rhs_shape.dim(1).size())
    return false;

  // Validate addend shape since broadcasting is not supported
  NodeDef* addend_node_def = nullptr;
  if (matched_nodes_map->find("addend") != matched_nodes_map->end()) {
    addend_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("addend"))->node();
    TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
        addend_node_def->name(), &addend_props));
    auto& addend_shape = addend_props[0].shape();
    if (Rank(addend_shape) != 4 ||
        addend_shape.dim(0).size() * addend_shape.dim(1).size() !=
            batch_matmul_props[0].shape().dim(0).size() *
                batch_matmul_props[0].shape().dim(1).size())
      return false;
  }
  input_node_names->clear();
  input_node_names->resize(4);
  if (found_pattern_index == 1 || found_pattern_index == 3) {
    input_node_names->at(0) = batch_matmul_node_def->input(0);
  } else if (found_pattern_index == 2) {
    auto* mul_input0_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("mul_input0"))->node();
    input_node_names->at(0) = mul_input0_node_def->name();
  }
  input_node_names->at(1) = batch_matmul_node_def->input(1);
  input_node_names->at(2) = multiplicand_node_def->name();
  if (addend_node_def != nullptr) {
    input_node_names->at(3) = addend_node_def->name();
  } else {
    return false;  // Add fusion requires addend
  }
  return found_op_type_match;
}

// Fuse BatchMatMul and Mul into FusedBatchMatmul if the other input of
// Mul is a scalar. For example, we can optimize:
/*
              Mul
             /  \
    BatchMatMul scale*  ->       FusedBatchMatmul
       /   \                     /      |       \
   input1  input2             input1  input2   scale
*/
// *) scale must be a scalar.

bool FindContractionWithMul(const RemapperContext& ctx, int node_index,
                            ContractionWithMul* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsAnyMul(*node_def)) return false;
  // Mul has two inputs, one input should be scalar.
  int scalar_input_index = GetMulScalarInputIndex(ctx, *node_def);
  if (scalar_input_index == -1) return false;

  auto* const_node_view =
      node_view->GetRegularFanin(scalar_input_index).node_view();
  auto* contraction_node_view =
      node_view->GetRegularFanin(1 - scalar_input_index).node_view();

  auto* contraction_node_def = contraction_node_view->node();
  if (!IsAnyBatchMatMul(*contraction_node_def)) return false;
  auto* const_node_def = const_node_view->node();
  if (!(IsAnyConst(*const_node_def) || IsCast(*const_node_def))) return false;

  // The fusion is only supported for float and bfloat16 on CPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  if (!HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  const ContractionWithMul pattern{contraction_node_view->node_index(),
                                   node_index, const_node_view->node_index()};

  *matched = pattern;

  return true;
}

void CopyBatchMatMulAttributes(const NodeDef& batchmatmul,
                               NodeDef* fused_batch_matmul) {
  DCHECK(IsAnyBatchMatMul(batchmatmul)) << "Input node must be a BatchMatMul";

  auto* attr = fused_batch_matmul->mutable_attr();
  auto& src_attr = batchmatmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["adj_x"] = src_attr.at("adj_x");
  (*attr)["adj_y"] = src_attr.at("adj_y");

  // TODO(plugin): Validate if "input_shapes" is necessary for ZenDNN ops.
  auto input_shapes = src_attr.find("_input_shapes");
  if (input_shapes != src_attr.end()) {
    (*attr)["_input_shapes"] = input_shapes->second;
  }
}

void CopyConv2DAttributes(const NodeDef& conv2d, NodeDef* fused_conv2d,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsConv2D(conv2d)) << "Input node must be a Conv2D";
  auto* attr = fused_conv2d->mutable_attr();
  auto& src_attr = conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  int num_args = fused_conv2d->input_size() - 2;
  for (int i = 0; i < num_args; ++i) {
    (*attr)["TArgs"].mutable_list()->add_type(src_attr.at("T").type());
  }
  (*attr)["num_host_args"].set_i(0);
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
  // Copy LeakyRelu's attr alpha to FusedConv2D's attr leakyrelu_alpha.
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyDepthwiseConv2dNativeAttributes(const NodeDef& dw_conv2d,
                                         NodeDef* fused_dw_conv2d,
                                         const NodeDef* activation = nullptr) {
  DCHECK(IsDepthwiseConv2dNative(dw_conv2d))
      << "Input node must be a DepthwiseConv2dNative";

  auto* attr = fused_dw_conv2d->mutable_attr();
  auto& src_attr = dw_conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  if (HasNodeAttr(dw_conv2d, "explicit_paddings")) {
    (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  }
  // Copy LeakyRelu's attr alpha to FusedDepthwiseConv2d's attr leakyrelu_alpha.
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyMatMulAttributes(const NodeDef& matmul, NodeDef* fused_matmul,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsMatMul(matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = matmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["transpose_a"] = src_attr.at("transpose_a");
  (*attr)["transpose_b"] = src_attr.at("transpose_b");
  // Copy LeakyRelu's attr alpha to _FusedMatMul's attr leakyrelu_alpha.
  // TODO(plugin) : Enable this when supporting LeakyRelu as fused activation.
  // if (activation != nullptr && IsLeakyRelu(*activation)) {
  //   auto& activation_attr = activation->attr();
  //   (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  // }
}

// MatMul, _ZenMatMul, and BatchMatMul* share T on _FusedMatMul; BatchMatMul
// uses adj_x/adj_y which map to transpose_a/transpose_b on _FusedMatMul.
void CopyMatMulLikeAttributes(const NodeDef& src, NodeDef* fused_matmul) {
  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = src.attr();
  (*attr)["T"] = src_attr.at("T");
  if (IsAnyBatchMatMul(src) || src.op() == "_ZenBatchMatMul" ||
      src.op() == "_ZenBatchMatMulV2") {
    bool ax = src_attr.contains("adj_x") && src_attr.at("adj_x").b();
    bool ay = src_attr.contains("adj_y") && src_attr.at("adj_y").b();
    (*attr)["transpose_a"].set_b(ax);
    (*attr)["transpose_b"].set_b(ay);
    return;
  }
  if (src_attr.contains("transpose_a")) {
    (*attr)["transpose_a"] = src_attr.at("transpose_a");
  } else {
    (*attr)["transpose_a"].set_b(false);
  }
  if (src_attr.contains("transpose_b")) {
    (*attr)["transpose_b"] = src_attr.at("transpose_b");
  } else {
    (*attr)["transpose_b"].set_b(false);
  }
}

void CopyFusedBatchNormAttributes(const NodeDef& fused_batch_norm,
                                  NodeDef* fused_batch_norm_ex) {
  DCHECK(IsFusedBatchNorm(fused_batch_norm))
      << "Input node must be a FusedBatchNorm";

  CopyAllAttrs(fused_batch_norm, fused_batch_norm_ex);

  // FusedBatchNorm doesn't have an extra type parameter.
  if ((fused_batch_norm.op() == "FusedBatchNorm") ||
      (fused_batch_norm.op() == "FusedBatchNormGrad")) {
    AddNodeAttr("U", DT_FLOAT, fused_batch_norm_ex);
  }
}

void CopyReshapeAttributes(const NodeDef& reshape, NodeDef* node) {
  DCHECK(IsReshape(reshape)) << "Input node must be a Reshape";

  auto* attr = node->mutable_attr();
  auto& src_attr = reshape.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["Tshape"] = src_attr.at("Tshape");
}

Status AddKerasDenseLayerFwd(RemapperContext* ctx,
                             const KerasDenseLayerFwd& matched,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& matmul = graph->node(matched.matmul);
  const NodeDef& reshape = graph->node(matched.reshape);
  const NodeDef& bias = graph->node(matched.bias);
  NodeDef new_shape;
  NodeDef fused_node;
  if (matched.activation != kMissingIndex) {
    const NodeDef& activation = graph->node(matched.activation);
    fused_node.set_op(kFusedMatMul);
    fused_node.set_name(bias.name());
    fused_node.set_device(matmul.device());
    fused_node.add_input(matmul.input(0));
    fused_node.add_input(matmul.input(1));
    fused_node.add_input(bias.input(1));
    CopyMatMulAttributes(matmul, &fused_node);
    SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});
    NodeDef new_reshape;
    new_reshape.set_op(kReshape);
    new_reshape.set_device(reshape.device());
    new_reshape.set_name(activation.name());
    new_reshape.add_input(bias.name());
    new_reshape.add_input(reshape.input(1));
    CopyReshapeAttributes(reshape, &new_reshape);

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;

    mutation->AddNode(std::move(fused_node), &status);
    mutation->AddNode(std::move(new_reshape), &status);
    (*invalidated_nodes)[matched.activation] = true;
    (*invalidated_nodes)[matched.bias] = true;
    (*nodes_to_delete)[matched.reshape] = true;
    (*nodes_to_delete)[matched.matmul] = true;

    TF_ABORT_IF_ERROR(status);
    TF_ABORT_IF_ERROR(mutation->Apply());
    return OkStatus();
  } else {
    fused_node.set_op(kFusedMatMul);
    fused_node.set_name(reshape.name());
    fused_node.set_device(matmul.device());
    fused_node.add_input(matmul.input(0));
    fused_node.add_input(matmul.input(1));
    fused_node.add_input(bias.input(1));
    CopyMatMulAttributes(matmul, &fused_node);
    SetFusedOpAttributes(&fused_node, {"BiasAdd"});
    NodeDef new_reshape;
    new_reshape.set_op(kReshape);
    new_reshape.set_device(reshape.device());
    new_reshape.set_name(bias.name());
    new_reshape.add_input(reshape.name());
    new_reshape.add_input(reshape.input(1));
    CopyReshapeAttributes(reshape, &new_reshape);

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;

    mutation->AddNode(std::move(fused_node), &status);
    mutation->AddNode(std::move(new_reshape), &status);
    (*invalidated_nodes)[matched.bias] = true;
    (*invalidated_nodes)[matched.reshape] = true;
    (*nodes_to_delete)[matched.matmul] = true;

    TF_ABORT_IF_ERROR(status);
    TF_ABORT_IF_ERROR(mutation->Apply());
    return OkStatus();
  }
}

// Contraction + BiasAdd.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with BiasAdd");

  NodeDef fused_node;
  fused_node.set_name(bias_add.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));               // 0: input
  fused_node.add_input(contraction.input(1));               // 1: filter
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
    // TODO(plugin) : Explore if _ZenFusedBatchMatMul is a simple possibility.
  } else if (IsAnyBatchMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulLikeAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributes(&fused_node, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Add.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAddAndAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  // ZenDNN only supports fusion for Conv and MatMul.
  DCHECK(IsConvOrMatMul(contraction));

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with BiasAdd and Add");

  NodeDef fused_node;
  fused_node.set_name(add.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input(conv) / a (matmul)
  fused_node.add_input(contraction.input(1));  // 1: filter(conv) / b (matmul)
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  // Add OP has two inputs, one is conv+bias/matmul+bias pattern matched
  // previously, the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else if (IsAnyBatchMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulLikeAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributes(&fused_node, {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with BiasAdd and ", activation.op());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));               // 0: input
  fused_node.add_input(contraction.input(1));               // 1: filter
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else if (IsAnyBatchMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulLikeAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddFusedMatMulSigmoidNode(RemapperContext* ctx,
                                 const ContractionWithActivation& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction =
      graph->node(matched.contraction);  // _FusedMatMul (MatMul + BiasAdd).
  const NodeDef& activation = graph->node(matched.activation);  // Sigmoid.

  // Log the fusion operation.
  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with Sigmoid activation");

  // Create the new fused node.
  NodeDef fused_node;
  fused_node.set_name(activation.name());  // Name it after the Sigmoid node.
  fused_node.set_device(
      contraction.device());  // Use the device of the original fused matmul.
  fused_node.set_op(contraction.op());

  CopyAllAttrs(contraction, &fused_node);
  fused_node.clear_input();
  fused_node.add_input(
      contraction.input(0));  // Input tensor (e.g., from concat).
  fused_node.add_input(contraction.input(1));  // Filter (constant kernel).
  fused_node.add_input(contraction.input(2));  // Bias (constant bias).

  // Set the fused operations to include both BiasAdd and Sigmoid.
  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  // Add the new node to the graph.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  // Mark the original _FusedMatMul for deletion and the Sigmoid as invalidated.
  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddFusedMatMulMishNode(RemapperContext* ctx,
                              const ContractionWithActivation& matched,
                              std::vector<bool>* invalidated_nodes,
                              std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction =
      graph->node(matched.contraction);  // _FusedMatMul (MatMul + BiasAdd).
  const NodeDef& activation = graph->node(matched.activation);  // Mish.

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with Mish activation");

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));
  fused_node.add_input(contraction.input(1));
  fused_node.add_input(contraction.input(2));

  fused_node.set_op(kFusedMatMul);

  CopyMatMulAttributes(contraction, &fused_node);

  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddFusedMatMulMishDecomposed(
    RemapperContext* ctx, const FusedMatMulMishDecomposedMatch& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& mul = graph->node(matched.mish_mul);

  if (!matched.redundant_f32_to_bf16_cast_on_bf16_matmul_out.empty()) {
    const string& contraction_out_name = contraction.name();
    SafeTensorId safe_contraction_out(contraction_out_name, 0);
    const TensorId contraction_tid(safe_contraction_out);
    for (int cix : matched.redundant_f32_to_bf16_cast_on_bf16_matmul_out) {
      utils::MutableNodeView* cast_view = ctx->graph_view.GetNode(cix);
      if (cast_view == nullptr) continue;
      utils::Mutation* cast_mut = ctx->graph_view.GetMutationBuilder();
      for (const auto& fo_set : cast_view->GetRegularFanouts()) {
        for (const auto& fo : fo_set) {
          cast_mut->AddOrUpdateRegularFanin(fo.node_view(), fo.index(),
                                            contraction_tid);
        }
      }
      for (const auto& ctrl : cast_view->GetControlledFanouts()) {
        cast_mut->RemoveControllingFanin(ctrl.node_view(),
                                         cast_view->node()->name());
        cast_mut->AddControllingFanin(ctrl.node_view(), contraction_out_name);
      }
      TF_RETURN_IF_ERROR(cast_mut->Apply());
    }
  }

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(),
                                       ") with Mish (decomposed)");

  NodeDef fused_node;
  fused_node.set_name(mul.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));
  fused_node.add_input(contraction.input(1));
  fused_node.add_input(contraction.input(2));
  fused_node.set_op(kFusedMatMul);
  CopyMatMulAttributes(contraction, &fused_node);
  SetFusedOpAttributes(&fused_node, {"BiasAdd", "Mish"}, 1);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.softplus] = true;
  (*nodes_to_delete)[matched.tanh] = true;
  (*invalidated_nodes)[matched.mish_mul] = true;
  for (int cix : matched.noop_bf16_cast_nodes) {
    if (cix >= 0 && cix < graph->node_size()) {
      (*nodes_to_delete)[cix] = true;
    }
  }
  for (int cix : matched.redundant_f32_to_bf16_cast_on_bf16_matmul_out) {
    if (cix >= 0 && cix < graph->node_size()) {
      (*nodes_to_delete)[cix] = true;
    }
  }

  return OkStatus();
}

// Contraction + Activation.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithActivation& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", contraction.op(),
                                       " (", contraction.name(), ") with ",
                                       activation.op());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter

  if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else if (IsAnyBatchMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulLikeAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation, {}, 0);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Add + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndAddActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConvOrMatMul(contraction));
  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  zendnnl::error_handling::apilog_info(
      "Remapper: Fusing ", contraction.op(), " (", contraction.name(),
      ") with BiasAdd, Add and ", activation.op());

  NodeDef fused_node;
  fused_node.set_name(activation.name());

  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter
  fused_node.add_input(bias_add.input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
    // TODO(plugin) : Check if _ZenFusedBatchMatMul is a possibility.
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else if (IsAnyBatchMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulLikeAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation,
                                     {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.add] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return OkStatus();
}

Status AddFusedBatchNormExNode(RemapperContext* ctx,
                               const FusedBatchNormEx& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnl::error_handling::apilog_info("Remapper: Fusing FusedBatchNorm (",
                                       fused_batch_norm.name(), ") with ",
                                       activation.op());

  // Replace FusedBatchNorm with _FusedBatchNormEx + Activation.
  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormEx);
  fused_op.set_name(fused_batch_norm.name());
  fused_op.set_device(fused_batch_norm.device());

  fused_op.add_input(fused_batch_norm.input(0));  // 0: input
  fused_op.add_input(fused_batch_norm.input(1));  // 1: scale
  fused_op.add_input(fused_batch_norm.input(2));  // 2: offset
  fused_op.add_input(fused_batch_norm.input(3));  // 3: estimated_mean
  fused_op.add_input(fused_batch_norm.input(4));  // 4: estimated_var

  CopyFusedBatchNormAttributes(fused_batch_norm, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  SetAttrValue(activation.op(), &(*attrs)["activation_mode"]);

  AddNodeAttr("num_side_inputs", 0, &fused_op);

  // Turn activation node into Identity node.
  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation.name());
  identity_op.set_device(fused_batch_norm.device());
  identity_op.add_input(fused_batch_norm.name());
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  mutation->AddNode(std::move(identity_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddPadWithContractionNode(RemapperContext* ctx,
                                 const PadWithContraction& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& pad = graph->node(matched.pad);
  const auto& pad_view = ctx->graph_view.GetNode(matched.pad);
  const auto& contraction_view = ctx->graph_view.GetNode(matched.contraction);
  const NodeDef& contraction = graph->node(matched.contraction);

  string padding;
  TF_CHECK_OK(GetNodeAttr(contraction, "padding", &padding));

  // Get original explicit padding values if padding = EXPLICIT.
  std::vector<int32> explicit_paddings_orig = {};
  if (padding == "EXPLICIT") {
    TF_CHECK_OK(GetNodeAttr(pad, "explicit_paddings", &explicit_paddings_orig));
  }

  // Index 0 has the input data and Index 1 has the padding values (which is
  // needed).
  const auto& const_pad_val_node_view =
      pad_view->GetRegularFanin(1).node_view();
  const auto const_pad_val_node_def = const_pad_val_node_view->node();
  Tensor explicit_padding_tensor;
  std::vector<int32> explicit_paddings;
  if (explicit_padding_tensor.FromProto(
          const_pad_val_node_def->attr().at("value").tensor())) {
    // Number of elements in explicit_padding_tensor (should be 8).
    int length = explicit_padding_tensor.NumElements();
    // 'padding_1d_tensor' is an Eigen Tensor with datatype int32.
    auto padding_1d_tensor = explicit_padding_tensor.flat<int32>();
    // For dimension i (starting from 0), the padding values
    // will be at 2*i and 2*i + 1.
    for (int index_pad = 0; index_pad < length; index_pad++) {
      if (padding == "VALID") {
        explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                 padding_1d_tensor(index_pad));
      } else if (padding == "EXPLICIT") {
        explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                 padding_1d_tensor(index_pad) +
                                     explicit_paddings_orig.at(index_pad));
      }
    }
  }

  auto* conv2d_mutable_attr = contraction_view->node()->mutable_attr();
  SetAttrValue("EXPLICIT", &(*conv2d_mutable_attr)["padding"]);
  SetAttrValue(explicit_paddings, &(*conv2d_mutable_attr)["explicit_paddings"]);

  NodeDef pad_with_conv;
  pad_with_conv.set_name(contraction.name());
  pad_with_conv.set_device(contraction.device());
  pad_with_conv.add_input(pad.input(0));          // 0: input
  pad_with_conv.add_input(contraction.input(1));  // 1: filter
  // Add bias input if contraction is _FusedConv2D/_FusedDepthwiseConv2dNative.
  if (contraction.op() == kFusedConv2D ||
      contraction.op() == kFusedDepthwiseConv2dNative) {
    pad_with_conv.add_input(contraction.input(2));  // 2: bias
  }

  if (IsConv2D(contraction) || contraction.op() == kFusedConv2D) {
    pad_with_conv.set_op(contraction.op());
    CopyConv2DAttributes(contraction, &pad_with_conv);
  } else if (IsDepthwiseConv2dNative(contraction) ||
             contraction.op() == kFusedDepthwiseConv2dNative) {
    pad_with_conv.set_op(contraction.op());
    CopyDepthwiseConv2dNativeAttributes(contraction, &pad_with_conv);
  } else {
    CHECK(false);
  }
  if (HasNodeAttr(contraction, "fused_ops")) {
    std::vector<string> fused_ops;
    // Only bias is allowed with fused contraction.
    int num_args = 1;
    float epsilon = 0.0;
    TF_CHECK_OK(GetNodeAttr(contraction, "fused_ops", &fused_ops));
    auto* attr = pad_with_conv.mutable_attr();
    SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
    SetAttrValue(num_args, &(*attr)["num_args"]);
    SetAttrValue(epsilon, &(*attr)["epsilon"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(pad_with_conv), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*nodes_to_delete)[matched.pad] = true;

  return OkStatus();
}

// Fuse safe_embedding_lookup_sparse subgraph into
// _ZenSafeEmbeddingLookupSparse.
Status AddFusedSafeEmbeddingLookupSparse(
    RemapperContext* ctx, const SafeEmbeddingLookupSparse& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& select_v2 = graph->node(matched.select_v2);
  const NodeDef& sparse_segment = graph->node(matched.sparse_segment);
  const NodeDef& sfr = graph->node(matched.sparse_fill_empty_rows);
  const NodeDef& table_node = graph->node(matched.embedding_table);

  zendnnl::error_handling::apilog_info(
      "Remapper: Fusing safe_embedding_lookup_sparse at ", select_v2.name(),
      " with combiner=", matched.combiner);

  NodeDef fused_node;
  fused_node.set_name(select_v2.name());
  fused_node.set_op("_ZenSafeEmbeddingLookupSparse");
  fused_node.set_device(sparse_segment.device());

  // Input 0: params (embedding table)
  // If the table_node is Identity, pass through to its input (the Const/Var).
  if (table_node.op() == "Identity") {
    fused_node.add_input(table_node.input(0));
  } else {
    fused_node.add_input(table_node.name());
  }

  // Input 1: sp_indices — sparse indices
  // Input 2: sp_values — sparse values (embedding indices)
  if (matched.has_upstream_filter) {
    // Wire past the GatherV2 chain to the raw upstream inputs.
    // GatherV2[indices].input(0) = SparseReshape:0 (raw sparse indices)
    // GatherV2[values].input(0) = FloorMod (raw hashed values)
    const NodeDef& gv2_idx = graph->node(matched.gv2_indices);
    const NodeDef& gv2_val = graph->node(matched.gv2_values);
    fused_node.add_input(gv2_idx.input(0));  // SparseReshape:0
    fused_node.add_input(gv2_val.input(0));  // FloorMod output
  } else {
    // No upstream filter — wire directly from SFR inputs (pre-filtered).
    fused_node.add_input(sfr.input(0));
    fused_node.add_input(sfr.input(1));
  }

  // Input 3: sp_dense_shape — SparseFillEmptyRows input 2 (dense_shape)
  // This is SparseReshape:1
  fused_node.add_input(sfr.input(2));

  // Input 4: default_value — SparseFillEmptyRows input 3
  fused_node.add_input(sfr.input(3));

  // Input 5: orig_dense_shape — for adjust_shape, or same as sp_dense_shape.
  if (matched.has_adjust_shape) {
    fused_node.add_input(matched.orig_dense_shape_input);
    // Use the final Reshape node's name so its consumers are auto-redirected.
    fused_node.set_name(graph->node(matched.adjust_reshape_node).name());
  } else {
    // No adjust_shape — pass sp_dense_shape as orig_dense_shape (no-op
    // reshape).
    fused_node.add_input(sfr.input(2));
  }

  // Copy data type from the embedding table.
  auto* attr = fused_node.mutable_attr();
  if (!HasNodeAttr(sparse_segment, "T")) {
    VLOG(1) << "SafeEmbeddingLookupSparse fusion: sparse_segment node "
            << sparse_segment.name() << " missing 'T' attribute";
    return Status::OK();
  }
  (*attr)["T"] = sparse_segment.attr().at("T");

  // Set Tindices from the sparse segment's Tidx.
  if (!HasNodeAttr(sparse_segment, "Tidx")) {
    VLOG(1) << "SafeEmbeddingLookupSparse fusion: sparse_segment node "
            << sparse_segment.name() << " missing 'Tidx' attribute";
    return Status::OK();
  }
  (*attr)["Tindices"] = sparse_segment.attr().at("Tidx");

  // Set combiner attribute.
  SetAttrValue(matched.combiner, &(*attr)["combiner"]);
  SetAttrValue(matched.has_adjust_shape, &(*attr)["has_adjust_shape"]);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  // The SelectV2 node is replaced by the fused node.
  (*invalidated_nodes)[matched.select_v2] = true;

  // Mark intermediate nodes for deletion.
  (*nodes_to_delete)[matched.tile] = true;
  (*nodes_to_delete)[matched.reshape] = true;
  (*nodes_to_delete)[matched.zeros_like] = true;
  (*nodes_to_delete)[matched.sparse_segment] = true;
  (*nodes_to_delete)[matched.strided_slice] = true;
  // Only delete SparseFillEmptyRows if no other consumers need it.
  // Check if SFR outputs are used only by nodes we're deleting.
  const auto* sfr_node_view =
      ctx->graph_view.GetNode(matched.sparse_fill_empty_rows);
  bool sfr_safe_to_delete = true;
  for (const auto& fanout : sfr_node_view->GetRegularFanouts()) {
    for (const auto& consumer : fanout) {
      int idx = consumer.node_view()->node_index();
      if (idx != matched.sparse_segment && idx != matched.strided_slice &&
          idx != matched.reshape && idx != matched.select_v2 &&
          idx != matched.zeros_like && idx != matched.tile &&
          (!matched.has_upstream_filter ||
           (idx != matched.gv2_indices && idx != matched.gv2_values))) {
        sfr_safe_to_delete = false;
        break;
      }
    }
    if (!sfr_safe_to_delete) break;
  }
  if (sfr_safe_to_delete) {
    (*nodes_to_delete)[matched.sparse_fill_empty_rows] = true;
  }
  // Only delete Identity table node if it has no other consumers.
  if (table_node.op() == "Identity") {
    const auto* table_view = ctx->graph_view.GetNode(matched.embedding_table);
    bool table_safe_to_delete = true;
    for (const auto& fanout : table_view->GetRegularFanouts()) {
      for (const auto& consumer : fanout) {
        int idx = consumer.node_view()->node_index();
        if (idx != matched.sparse_segment) {
          table_safe_to_delete = false;
          break;
        }
      }
      if (!table_safe_to_delete) break;
    }
    if (table_safe_to_delete) {
      (*nodes_to_delete)[matched.embedding_table] = true;
    }
  }

  // Delete absorbed upstream filter nodes (GatherV2, Reshape, Where,
  // GreaterEqual) if they have no other consumers.
  if (matched.has_upstream_filter) {
    // Helper: check if a node's only consumers are in our deletion set.
    auto safe_to_delete = [&](int node_idx) -> bool {
      const auto* nv = ctx->graph_view.GetNode(node_idx);
      for (const auto& fanout : nv->GetRegularFanouts()) {
        for (const auto& consumer : fanout) {
          int cidx = consumer.node_view()->node_index();
          if (!(*nodes_to_delete)[cidx] && cidx != matched.select_v2) {
            return false;
          }
        }
      }
      return true;
    };

    // GatherV2 nodes feed only into SFR (which is already deleted).
    if (safe_to_delete(matched.gv2_indices))
      (*nodes_to_delete)[matched.gv2_indices] = true;
    if (safe_to_delete(matched.gv2_values))
      (*nodes_to_delete)[matched.gv2_values] = true;
    // Reshape feeds only into the two GatherV2s.
    if (safe_to_delete(matched.filter_reshape))
      (*nodes_to_delete)[matched.filter_reshape] = true;
    // Where feeds only into Reshape.
    if (safe_to_delete(matched.filter_where))
      (*nodes_to_delete)[matched.filter_where] = true;
    // GreaterEqual feeds only into Where.
    if (safe_to_delete(matched.filter_ge))
      (*nodes_to_delete)[matched.filter_ge] = true;

    zendnnl::error_handling::apilog_info(
        "Remapper: Absorbed upstream GatherV2/Where/GreaterEqual filter chain");
  }

  // Delete absorbed downstream adjust_shape nodes.
  if (matched.has_adjust_shape) {
    // The fused node now takes the name of the final Reshape, so
    // the original SelectV2 node (matched.select_v2) should be deleted.
    (*nodes_to_delete)[matched.select_v2] = true;
    (*invalidated_nodes)[matched.adjust_reshape_node] = true;

    (*nodes_to_delete)[matched.adjust_shape_node] = true;       // Shape
    (*nodes_to_delete)[matched.adjust_slice_node] = true;       // Slice (embed)
    (*nodes_to_delete)[matched.adjust_concat_node] = true;      // ConcatV2
    (*nodes_to_delete)[matched.adjust_cast_slice_node] = true;  // Slice (batch)

    zendnnl::error_handling::apilog_info(
        "Remapper: Absorbed downstream adjust_shape chain (Shape+Slice+"
        "ConcatV2+Reshape)");
  }

  return OkStatus();
}

// Contraction + Mul(scale).
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithMul& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& mul = graph->node(matched.mul);
  const NodeDef& scalar = graph->node(matched.scalar);

  NodeDef fused_op;
  fused_op.set_name(mul.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(scalar.name());         // 2: scale
  fused_op.set_op(kFusedBatchMatMulV2);

  CopyBatchMatMulAttributes(contraction, &fused_op);
  SetFusedOpAttributes(&fused_op, {kBinaryMul}, 1);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.mul] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  return OkStatus();
}

Status AddFusedMatMulBiasAddAndGelu(
    RemapperContext* ctx, const std::map<string, int>& matched_nodes_map,
    const std::set<int>& remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete,
    bool is_gelu_approximate, bool expand_dims) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("matmul"))->node();

  NodeDef fused_node;
  // Fused node should have the name of terminal node of the fusion.
  fused_node.set_name(output_node->name());
  fused_node.set_op("_FusedMatMul");
  fused_node.set_device(matmul_node->device());
  fused_node.add_input(matmul_node->input(0));
  fused_node.add_input(matmul_node->input(1));
  if (is_gelu_approximate || expand_dims) {
    fused_node.add_input(matmul_node->input(2));
  } else {
    auto* bias_add_node =
        ctx->graph_view.GetNode(matched_nodes_map.at("bias_add"))->node();
    fused_node.add_input(bias_add_node->input(1));
  }
  CopyMatMulAttributes(*matmul_node, &fused_node);
  if (is_gelu_approximate) {
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluApproximate"});
  } else {
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluExact"});
  }
  if (expand_dims) {
    auto* attr = fused_node.mutable_attr();
    SetAttrValue(true, &(*attr)["is_reshape"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return OkStatus();
}

Status AddFusedBatchMatMulBiasAddActivation(
    RemapperContext* ctx, const BatchMatMulWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& batch_matmul = graph->node(matched.batch_matmul);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnl::error_handling::apilog_info("Remapper: Fusing BatchMatMul (",
                                       batch_matmul.name(),
                                       ") with BiasAdd and ", activation.op());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(batch_matmul.device());
  fused_node.add_input(batch_matmul.input(0));              // 0: input
  fused_node.add_input(batch_matmul.input(1));              // 1: filter
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias
  // Note: The op and kernel definition for the "_FusedBatchMatMulV2" op is not
  // present. We are sure that it will be rewritten with
  // "_ZenFusedBatchMatMulV2" from zen layout pass.
  fused_node.set_op(kFusedBatchMatMulV2);

  CopyBatchMatMulAttributes(batch_matmul, &fused_node);
  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.batch_matmul] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddFusedBatchMatMul(RemapperContext* ctx,
                           const std::map<string, int>& matched_nodes_map,
                           const std::set<int>& remove_node_indices,
                           const std::vector<string>& input_node_names,
                           std::vector<bool>* invalidated_nodes,
                           std::vector<bool>* nodes_to_delete) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* batch_matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("batch_matmul"))->node();

  NodeDef fused_node;
  fused_node.set_name(output_node->name());
  // Note: The op and kernel definition for the "_FusedBatchMatMulV2" op is not
  // present. We are sure that it will be rewritten with
  // "_ZenFusedBatchMatMulV2" from zen layout pass.
  fused_node.set_op(kFusedBatchMatMulV2);

  fused_node.set_device(batch_matmul_node->device());
  for (const auto& name : input_node_names) fused_node.add_input(name);

  CopyBatchMatMulAttributes(*batch_matmul_node, &fused_node);
  SetFusedOpAttributes(&fused_node, {kBinaryMul, kAdd}, /*num_args=*/2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return OkStatus();
}

// Try to match a single ConcatV2 input as a GatherV2 wrapped in zero or more
// SafeCast layers.
//
// Supported patterns (all feed into ConcatV2):
//   BF16:  SelectV2(IsInf(Cast(GatherV2)), ZerosLike, Cast)
//   FP32:  SelectV2(IsInf(SelectV2(IsInf(GatherV2),..)),..,..)  (stacked)
//   Direct: GatherV2
//
// In the BF16 pattern the Cast is the SafeCast's else-input (and the IsInf
// input); it is unwrapped as part of the SafeCast, not as an independent layer.
// A Cast is therefore only accepted inside a verified SafeCast, so a bare
// GatherV2 -> Cast -> ConcatV2 is not matched.
//
// On success, sets table_name, index_name, gather_axis and appends
// intermediate node indices to remove_indices. If op_tokens is non-null, it is
// filled with the semantic post-op names in KERNEL order (gather-side op
// first), e.g. {"SafeCastCheck"} for GatherV2 -> Cast -> SelectV2(SafeCast
// idiom) -> ConcatV2. The fused Cast is deleted but not emitted as a token (its
// dtype conversion is folded into the kernel's gather copy), so a direct
// GatherV2 yields an empty token list.
bool MatchGatherInput(const RemapperContext& ctx,
                      const utils::MutableNodeView* inp_view,
                      std::string* table_name, std::string* index_name,
                      int* gather_axis, std::vector<int>* remove_indices,
                      std::vector<std::string>* op_tokens = nullptr) {
  const auto* cur_view = inp_view;
  // Collected outermost->innermost (ConcatV2 side first); reversed before
  // returning so the caller receives kernel order (gather side first).
  std::vector<std::string> walk_tokens;

  // Iteratively unwrap SafeCast layers until we reach the GatherV2:
  //   SelectV2(IsInf(X), ZerosLike(X), X) → unwrap to X
  // where X is either the next SafeCast (stacked FP32), the GatherV2, or - in
  // the BF16 pattern - a Cast wrapping the GatherV2, which is peeled within the
  // same iteration.
  //
  // Each loop iteration peels one SafeCast layer. We cap the walk with a
  // bounded limit rather than looping unbounded: in production the number of
  // layers is small and a hard cap protects against walking a
  // pathological/malformed graph. The cap is sized to allow up to 4 stacked
  // SafeCast layers with comfortable headroom.
  constexpr int kMaxUnwrapDepth = 20;
  int depth = 0;
  for (depth = 0; depth < kMaxUnwrapDepth; ++depth) {
    const auto* cur_def = cur_view->node();

    if (cur_def->op() == "SelectV2") {
      if (cur_view->NumRegularFanins() < 3) return false;

      const auto* cond_view = cur_view->GetRegularFanin(0).node_view();
      const auto* then_view = cur_view->GetRegularFanin(1).node_view();
      const auto* else_view = cur_view->GetRegularFanin(2).node_view();

      if (cond_view->node()->op() != "IsInf") return false;
      if (then_view->node()->op() != "ZerosLike") return false;

      if (cond_view->NumRegularFanins() < 1) return false;
      const auto* isinf_input_view = cond_view->GetRegularFanin(0).node_view();
      if (isinf_input_view->node_index() != else_view->node_index())
        return false;

      remove_indices->push_back(cur_view->node_index());
      remove_indices->push_back(cond_view->node_index());
      remove_indices->push_back(then_view->node_index());
      // Emit the semantic token "SafeCastCheck". This node represents the
      // SafeCast (clamp-infinities-to-zero) idiom.
      walk_tokens.push_back("SafeCastCheck");

      cur_view = else_view;

      // BF16 pattern: the SafeCast wraps a Cast
      //   SelectV2(IsInf(Cast(GatherV2)), ZerosLike, Cast(GatherV2))
      // i.e. the SafeCast's else-input (== the IsInf input, verified above) is
      // a Cast. Unwrap that Cast here as part of the SafeCast: it needs no
      // post-op token because its dtype conversion is folded into the kernel's
      // gather copy (static_cast<T_output>, output dtype captured in the
      // T_output attr). The node is still recorded for deletion. In FP32 the
      // else-input is the next SelectV2 (stacked) or the GatherV2 directly (no
      // Cast).
      if (IsCast(*else_view->node())) {
        remove_indices->push_back(else_view->node_index());
        if (else_view->NumRegularFanins() < 1) return false;
        cur_view = else_view->GetRegularFanin(0).node_view();
      }
      continue;
    }

    break;
  }

  // If we exhausted the depth budget without reaching a GatherV2, the chain is
  // deeper than we support (or malformed); log and bail so the caller can skip
  // fusing this input.
  if (depth == kMaxUnwrapDepth && !IsGather(*cur_view->node())) {
    zendnnl::error_handling::apilog_info(
        "MatchGatherInput: hit unwrap depth limit at ", cur_view->node()->op(),
        " node ", cur_view->node()->name());
  }

  if (!IsGather(*cur_view->node())) return false;

  const auto* gather_view = cur_view;
  const auto* gather_def = gather_view->node();

  if (gather_view->NumRegularFanins() < 3) return false;

  // Direct GatherV2 → ConcatV2 (depth == 0) is only safe when the kernel can
  // produce the correct output shape. The current kernel always emits rank-2
  // output [outer_size, features], which matches const rank-≤1 index patterns
  // (e.g., swipe_l1's sequential const indices). It CANNOT represent standard
  // embedding lookups (dynamic multi-dim indices, gather_axis=0, output rank
  // = indices_rank + 1), e.g., DIEN's [batch,seq] placeholders →
  // [batch,seq,emb]. Wrapped paths (depth > 0, SafeCast/Cast) remain
  // unrestricted (those are the original working BF16/FP32 embedding fusion
  // cases).
  if (depth == 0) {
    const auto* indices_view = gather_view->GetRegularFanin(1).node_view();
    const auto* indices_def = indices_view->node();

    // Reject dynamic indices (non-Const).
    if (!IsConstant(*indices_def)) {
      zendnnl::error_handling::apilog_info(
          "MatchGatherInput: rejected direct GatherV2→ConcatV2 with dynamic "
          "indices (",
          indices_def->op(), " ", indices_def->name(), ")");
      return false;
    }

    // Reject Const indices with rank > 1 (multi-dim).
    if (indices_def->attr().count("value")) {
      const auto& tensor_proto = indices_def->attr().at("value").tensor();
      const int indices_rank = tensor_proto.tensor_shape().dim_size();
      if (indices_rank > 1) {
        zendnnl::error_handling::apilog_info(
            "MatchGatherInput: rejected direct GatherV2→ConcatV2 with "
            "multi-dim Const indices (rank=",
            indices_rank, ")");
        return false;
      }
    }
  }

  const auto* axis_view = gather_view->GetRegularFanin(2).node_view();
  const auto* axis_def = axis_view->node();
  if (!IsConstant(*axis_def)) return false;
  Tensor axis_tensor;
  if (!axis_tensor.FromProto(axis_def->attr().at("value").tensor()))
    return false;
  int axis_val = 0;
  if (axis_tensor.dtype() == DT_INT32) {
    axis_val = axis_tensor.flat<int32>()(0);
  } else if (axis_tensor.dtype() == DT_INT64) {
    axis_val = static_cast<int>(axis_tensor.flat<int64>()(0));
  } else {
    return false;
  }

  *table_name = gather_def->input(0);
  *index_name = gather_def->input(1);
  *gather_axis = axis_val;
  remove_indices->push_back(gather_view->node_index());

  if (op_tokens != nullptr) {
    // Reverse to kernel order: gather-side op applied first.
    op_tokens->assign(walk_tokens.rbegin(), walk_tokens.rend());
  }
  return true;
}

// Find GatherV2 ops (via SafeCast chain) within a ConcatV2.
//
// Full-fusion path (preferred): when ALL ConcatV2 value inputs match the
// gather pattern with the same axis, build one table-group per contiguous
// same-table segment and set full_fusion=true.  The ConcatV2 will be
// replaced entirely by a single multi-table _ZenGroupEmbedding.
//
// Fallback path: when some inputs don't match, build contiguous same-table
// runs of size >= kMinRunSize and keep the ConcatV2 with reduced inputs.
constexpr int kMinRunSize = 2;

bool FindGroupEmbedding(const RemapperContext& ctx, int node_index,
                        GroupEmbedding* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;

  if (node_def->op() != "ConcatV2") return false;

  if (!HasNodeAttr(*node_def, "N")) return false;
  int n_values = node_def->attr().at("N").i();
  if (n_values < kMinRunSize) return false;

  // --- Pass 1: try to match every input. ---
  struct GatherInfo {
    std::string table_name;
    std::string index_name;
    int axis_val;
    std::vector<int> remove_indices;
    std::vector<std::string> op_tokens;
  };
  std::vector<GatherInfo> all_gathers(n_values);
  bool all_match = true;
  int common_axis = 0;

  for (int v = 0; v < n_values; ++v) {
    if (v >= node_view->NumRegularFanins()) {
      all_match = false;
      break;
    }

    const auto& fanin = node_view->GetRegularFanin(v);
    const auto* inp_view = fanin.node_view();

    bool ok = MatchGatherInput(
        ctx, inp_view, &all_gathers[v].table_name, &all_gathers[v].index_name,
        &all_gathers[v].axis_val, &all_gathers[v].remove_indices,
        &all_gathers[v].op_tokens);
    if (!ok) {
      zendnnl::error_handling::apilog_info(
          "Remapper: full-fusion check failed at input ", v, " of ", n_values,
          " (", inp_view->node()->op(), " ", inp_view->node()->name(), ")");
      all_match = false;
      break;
    }
    if (v == 0) {
      common_axis = all_gathers[v].axis_val;
    } else if (all_gathers[v].axis_val != common_axis) {
      zendnnl::error_handling::apilog_info(
          "Remapper: full-fusion check failed at input ", v,
          " axis=", all_gathers[v].axis_val, " != common_axis=", common_axis);
      all_match = false;
      break;
    }
  }

  // --- Full-fusion path ---
  if (all_match && n_values >= kMinRunSize) {
    std::vector<GroupEmbeddingRun> runs;
    GroupEmbeddingRun current_run;

    for (int v = 0; v < n_values; ++v) {
      const auto& gi = all_gathers[v];
      if (v > 0 && gi.table_name == current_run.table_name) {
        current_run.index_names.push_back(gi.index_name);
        current_run.nodes_to_remove.insert(current_run.nodes_to_remove.end(),
                                           gi.remove_indices.begin(),
                                           gi.remove_indices.end());
        current_run.end_pos = v;
      } else {
        if (v > 0) runs.push_back(std::move(current_run));
        current_run = GroupEmbeddingRun();
        current_run.table_name = gi.table_name;
        current_run.index_names.push_back(gi.index_name);
        current_run.nodes_to_remove = gi.remove_indices;
        current_run.start_pos = v;
        current_run.end_pos = v;
        current_run.gather_axis = gi.axis_val;
      }
    }
    runs.push_back(std::move(current_run));

    int total_gathers = 0;
    for (const auto& run : runs) {
      total_gathers += static_cast<int>(run.index_names.size());
    }

    matched->concat = node_index;
    matched->runs = std::move(runs);
    matched->full_fusion = true;
    // Chain is uniform across inputs; capture it from the first input.
    matched->fused_ops = all_gathers[0].op_tokens;

    zendnnl::error_handling::apilog_info(
        "Remapper: Found full-fusion GroupEmbedding with ",
        matched->runs.size(), " table groups (", total_gathers,
        " GatherV2 ops) in ConcatV2 (", node_def->name(), ")");
    return true;
  }

  // --- Fallback: contiguous same-table runs of size >= kMinRunSize ---
  std::vector<GroupEmbeddingRun> runs;
  GroupEmbeddingRun current_run;
  bool in_run = false;

  for (int v = 0; v < n_values; ++v) {
    if (v >= node_view->NumRegularFanins()) break;

    const auto& fanin = node_view->GetRegularFanin(v);
    const auto* inp_view = fanin.node_view();

    std::string table_name, index_name;
    int axis_val = 0;
    std::vector<int> remove_indices;
    std::vector<std::string> op_tokens;

    bool is_match = MatchGatherInput(ctx, inp_view, &table_name, &index_name,
                                     &axis_val, &remove_indices, &op_tokens);
    if (is_match && matched->fused_ops.empty()) {
      // Chain is uniform; capture from the first matched input.
      matched->fused_ops = op_tokens;
    }

    if (is_match && in_run && table_name == current_run.table_name &&
        axis_val == current_run.gather_axis) {
      current_run.index_names.push_back(index_name);
      current_run.nodes_to_remove.insert(current_run.nodes_to_remove.end(),
                                         remove_indices.begin(),
                                         remove_indices.end());
      current_run.end_pos = v;
    } else {
      if (in_run &&
          static_cast<int>(current_run.index_names.size()) >= kMinRunSize) {
        runs.push_back(std::move(current_run));
      }
      if (is_match) {
        current_run = GroupEmbeddingRun();
        current_run.table_name = table_name;
        current_run.index_names.push_back(index_name);
        current_run.nodes_to_remove = std::move(remove_indices);
        current_run.start_pos = v;
        current_run.end_pos = v;
        current_run.gather_axis = axis_val;
        in_run = true;
      } else {
        in_run = false;
      }
    }
  }
  if (in_run &&
      static_cast<int>(current_run.index_names.size()) >= kMinRunSize) {
    runs.push_back(std::move(current_run));
  }

  if (runs.empty()) return false;

  int total_gathers = 0;
  for (const auto& run : runs) {
    total_gathers += static_cast<int>(run.index_names.size());
  }

  matched->concat = node_index;
  matched->runs = std::move(runs);
  matched->full_fusion = false;

  zendnnl::error_handling::apilog_info(
      "Remapper: Found GroupEmbedding with ", matched->runs.size(), " runs (",
      total_gathers, " GatherV2 ops) in ConcatV2 (", node_def->name(), ")");
  return true;
}

// Infer table / index / output dtypes from the first run's intermediate nodes.
void InferGroupEmbeddingDtypes(const GraphDef& graph,
                               const GroupEmbeddingRun& first_run,
                               DataType* table_dtype, DataType* indices_dtype,
                               DataType* output_dtype) {
  *table_dtype = DT_FLOAT;
  *indices_dtype = DT_INT64;
  *output_dtype = DT_FLOAT;
  for (int idx : first_run.nodes_to_remove) {
    const NodeDef& n = graph.node(idx);
    if (IsGather(n)) {
      if (HasNodeAttr(n, "Tparams"))
        *table_dtype = n.attr().at("Tparams").type();
      if (HasNodeAttr(n, "Tindices"))
        *indices_dtype = n.attr().at("Tindices").type();
      break;
    }
  }
  // If a Cast exists (BF16 pattern), output dtype is the Cast's DstT.
  // Otherwise (FP32 pattern — no Cast), output dtype matches table dtype.
  bool found_cast = false;
  for (int idx : first_run.nodes_to_remove) {
    const NodeDef& n = graph.node(idx);
    if (IsCast(n) && HasNodeAttr(n, "DstT")) {
      *output_dtype = n.attr().at("DstT").type();
      found_cast = true;
      break;
    }
  }
  if (!found_cast) {
    *output_dtype = *table_dtype;
  }
}

// Only delete intermediate nodes whose consumers are ALL within the fusion.
// Shared nodes (with external consumers) must be kept alive.
void SafeDeleteFusionNodes(const RemapperContext& ctx,
                           const GroupEmbedding& matched,
                           std::vector<bool>* nodes_to_delete) {
  std::set<int> fusion_set;
  fusion_set.insert(matched.concat);
  for (const auto& run : matched.runs) {
    for (int idx : run.nodes_to_remove) {
      fusion_set.insert(idx);
    }
  }

  for (int idx : fusion_set) {
    if (idx == matched.concat) continue;
    const auto* node_view = ctx.graph_view.GetNode(idx);
    bool all_consumers_in_fusion = true;
    for (const auto& fanout_set : node_view->GetRegularFanouts()) {
      for (const auto& fanout : fanout_set) {
        if (fusion_set.count(fanout.node_view()->node_index()) == 0) {
          all_consumers_in_fusion = false;
          break;
        }
      }
      if (!all_consumers_in_fusion) break;
    }
    if (all_consumers_in_fusion) {
      (*nodes_to_delete)[idx] = true;
    }
  }
}

// Full-fusion path: replace the entire ConcatV2 with a single multi-table
// _ZenGroupEmbedding node.
Status AddFullFusionGroupEmbeddingNode(RemapperContext* ctx,
                                       const GroupEmbedding& matched,
                                       std::vector<bool>* invalidated_nodes,
                                       std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& concat_node = graph->node(matched.concat);

  DataType table_dtype, indices_dtype, output_dtype;
  InferGroupEmbeddingDtypes(*graph, matched.runs[0], &table_dtype,
                            &indices_dtype, &output_dtype);

  const std::string concat_name = concat_node.name();
  const std::string concat_device = concat_node.device();

  const int num_tables = static_cast<int>(matched.runs.size());
  int total_indices = 0;
  std::vector<int> gathers_per_table;
  gathers_per_table.reserve(num_tables);
  for (const auto& run : matched.runs) {
    int n = static_cast<int>(run.index_names.size());
    gathers_per_table.push_back(n);
    total_indices += n;
  }

  std::string emb_name = concat_name + "/_ZenGroupEmbedding";

  zendnnl::error_handling::apilog_info(
      "Remapper: Creating full-fusion ", emb_name, " with ", num_tables,
      " table groups, ", total_indices, " total gathers");

  NodeDef emb_node;
  emb_node.set_name(emb_name);
  emb_node.set_op(kZenGroupEmbedding);
  emb_node.set_device(concat_device);

  // Inputs: tables first, then all indices.
  for (const auto& run : matched.runs) {
    emb_node.add_input(run.table_name);
  }
  for (const auto& run : matched.runs) {
    for (const auto& idx_name : run.index_names) {
      emb_node.add_input(idx_name);
    }
  }

  auto* attr = emb_node.mutable_attr();
  SetAttrValue(num_tables, &(*attr)["num_tables"]);
  SetAttrValue(total_indices, &(*attr)["N"]);
  SetAttrValue(table_dtype, &(*attr)["T_table"]);
  SetAttrValue(indices_dtype, &(*attr)["T_indices"]);
  SetAttrValue(output_dtype, &(*attr)["T_output"]);
  SetAttrValue(-1, &(*attr)["embedding_dim"]);
  SetAttrValue(matched.runs[0].gather_axis, &(*attr)["gather_axis"]);
  SetAttrValue(gathers_per_table, &(*attr)["gathers_per_table"]);
  // Only emit fused_ops when there is at least one post-op. An empty list would
  // leave the kernel reading an empty list(string) attribute, which the op-def
  // default ([]) already represents as "no post-ops".
  if (!matched.fused_ops.empty()) {
    SetAttrValue(matched.fused_ops, &(*attr)["fused_ops"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(emb_node), &status);
  TF_RETURN_IF_ERROR(status);

  SafeDeleteFusionNodes(*ctx, matched, nodes_to_delete);

  // Replace the ConcatV2: point all its consumers to the new node.
  auto* concat_view = ctx->graph_view.GetNode(matched.concat);
  auto fanouts = concat_view->GetRegularFanouts();
  for (const auto& fanout_set : fanouts) {
    for (const auto& fanout : fanout_set) {
      mutation->AddOrUpdateRegularFanin(fanout.node_view(), fanout.index(),
                                        TensorId(emb_name, 0));
    }
  }

  // Also replace control outputs.
  for (const auto& ctrl_fanout : concat_view->GetControlledFanouts()) {
    mutation->RemoveControllingFanin(ctrl_fanout.node_view(),
                                     concat_view->node()->name());
    mutation->AddControllingFanin(ctrl_fanout.node_view(), emb_name);
  }

  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.concat] = true;
  (*invalidated_nodes)[matched.concat] = true;

  return OkStatus();
}

// Multi-table path: group runs into contiguous blocks separated by
// non-matching ConcatV2 inputs.  Each block becomes one _ZenGroupEmbedding.
// Non-matching inputs stay in their original position to preserve output
// ordering.  If only one trailing non-matching input remains at the very
// end and all gathers form a single block, absorb it as passthrough.
Status AddMultiTableGroupEmbeddingNode(RemapperContext* ctx,
                                       const GroupEmbedding& matched,
                                       std::vector<bool>* invalidated_nodes,
                                       std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& concat_node = graph->node(matched.concat);

  DataType table_dtype, indices_dtype, output_dtype;
  InferGroupEmbeddingDtypes(*graph, matched.runs[0], &table_dtype,
                            &indices_dtype, &output_dtype);

  const std::string concat_name = concat_node.name();
  const std::string concat_device = concat_node.device();
  const int n_values = concat_node.attr().at("N").i();

  std::vector<std::string> original_inputs;
  original_inputs.reserve(concat_node.input_size());
  for (int i = 0; i < concat_node.input_size(); ++i) {
    original_inputs.push_back(concat_node.input(i));
  }

  std::set<int> consumed_positions;
  for (const auto& run : matched.runs) {
    for (int p = run.start_pos; p <= run.end_pos; ++p) {
      consumed_positions.insert(p);
    }
  }

  // Group runs into contiguous blocks.  A new block starts when there
  // is a gap (non-matching position) between adjacent runs.
  struct RunBlock {
    std::vector<size_t> run_indices;
    int first_pos;
    int last_pos;
  };
  std::vector<RunBlock> blocks;
  for (size_t r = 0; r < matched.runs.size(); ++r) {
    const auto& run = matched.runs[r];
    bool start_new = blocks.empty();
    if (!start_new) {
      int prev_end = blocks.back().last_pos;
      for (int p = prev_end + 1; p < run.start_pos; ++p) {
        if (consumed_positions.count(p) == 0) {
          start_new = true;
          break;
        }
      }
    }
    if (start_new) {
      blocks.push_back({{r}, run.start_pos, run.end_pos});
    } else {
      blocks.back().run_indices.push_back(r);
      blocks.back().last_pos = run.end_pos;
    }
  }

  // Trailing passthrough: one unconsumed position at the end, all runs
  // in a single contiguous block.
  std::vector<int> unconsumed;
  for (int v = 0; v < n_values; ++v) {
    if (consumed_positions.count(v) == 0) unconsumed.push_back(v);
  }
  bool has_trailing_passthrough =
      (blocks.size() == 1 && unconsumed.size() == 1 &&
       unconsumed[0] == n_values - 1);

  // Create one _ZenGroupEmbedding per contiguous block.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  std::vector<std::string> block_emb_names;

  for (size_t b = 0; b < blocks.size(); ++b) {
    const auto& block = blocks[b];
    int num_tables = static_cast<int>(block.run_indices.size());
    int total_indices = 0;
    std::vector<int> gathers_per_table;
    for (size_t ri : block.run_indices) {
      int n = static_cast<int>(matched.runs[ri].index_names.size());
      gathers_per_table.push_back(n);
      total_indices += n;
    }

    std::string emb_name = concat_name + "/_ZenGroupEmbedding";
    if (blocks.size() > 1) emb_name += "_" + std::to_string(b);
    block_emb_names.push_back(emb_name);

    bool block_has_pt = (has_trailing_passthrough && blocks.size() == 1);

    zendnnl::error_handling::apilog_info(
        "Remapper: Creating ", emb_name, " with ", num_tables,
        " table groups, ", total_indices, " gathers",
        block_has_pt ? ", with trailing passthrough" : "");

    NodeDef emb_node;
    emb_node.set_name(emb_name);
    emb_node.set_op(kZenGroupEmbedding);
    emb_node.set_device(concat_device);

    for (size_t ri : block.run_indices)
      emb_node.add_input(matched.runs[ri].table_name);
    for (size_t ri : block.run_indices)
      for (const auto& idx : matched.runs[ri].index_names)
        emb_node.add_input(idx);
    if (block_has_pt) emb_node.add_input(original_inputs[unconsumed[0]]);

    auto* attr = emb_node.mutable_attr();
    SetAttrValue(num_tables, &(*attr)["num_tables"]);
    SetAttrValue(total_indices, &(*attr)["N"]);
    SetAttrValue(table_dtype, &(*attr)["T_table"]);
    SetAttrValue(indices_dtype, &(*attr)["T_indices"]);
    SetAttrValue(output_dtype, &(*attr)["T_output"]);
    SetAttrValue(-1, &(*attr)["embedding_dim"]);
    SetAttrValue(matched.runs[block.run_indices[0]].gather_axis,
                 &(*attr)["gather_axis"]);
    SetAttrValue(gathers_per_table, &(*attr)["gathers_per_table"]);
    SetAttrValue(block_has_pt ? 1 : 0, &(*attr)["num_passthrough"]);
    // Only emit fused_ops when there is at least one post-op. An empty list
    // would leave the kernel reading an empty list(string) attribute, which
    // the op-def default ([]) already represents as "no post-ops".
    if (!matched.fused_ops.empty()) {
      SetAttrValue(matched.fused_ops, &(*attr)["fused_ops"]);
    }

    mutation->AddNode(std::move(emb_node), &status);
    TF_RETURN_IF_ERROR(status);
  }

  SafeDeleteFusionNodes(*ctx, matched, nodes_to_delete);

  if (has_trailing_passthrough && blocks.size() == 1) {
    // Single block + trailing passthrough → replace ConcatV2 entirely.
    auto* concat_view = ctx->graph_view.GetNode(matched.concat);
    for (const auto& fanout_set : concat_view->GetRegularFanouts())
      for (const auto& fanout : fanout_set)
        mutation->AddOrUpdateRegularFanin(fanout.node_view(), fanout.index(),
                                          TensorId(block_emb_names[0], 0));
    for (const auto& ctrl : concat_view->GetControlledFanouts()) {
      mutation->RemoveControllingFanin(ctrl.node_view(),
                                       concat_view->node()->name());
      mutation->AddControllingFanin(ctrl.node_view(), block_emb_names[0]);
    }
    TF_RETURN_IF_ERROR(mutation->Apply());
    (*nodes_to_delete)[matched.concat] = true;
    (*invalidated_nodes)[matched.concat] = true;
  } else {
    // Keep ConcatV2 — replace each contiguous block with its
    // _ZenGroupEmbedding, preserving non-matching inputs in place.
    std::map<int, size_t> block_start_to_idx;
    for (size_t b = 0; b < blocks.size(); ++b)
      block_start_to_idx[blocks[b].first_pos] = b;

    std::vector<std::string> new_value_inputs;
    for (int v = 0; v < n_values; ++v) {
      auto it = block_start_to_idx.find(v);
      if (it != block_start_to_idx.end()) {
        new_value_inputs.push_back(block_emb_names[it->second]);
        v = blocks[it->second].last_pos;
      } else if (consumed_positions.count(v) == 0) {
        new_value_inputs.push_back(original_inputs[v]);
      }
    }

    int new_n = static_cast<int>(new_value_inputs.size());
    auto* concat_view = ctx->graph_view.GetNode(matched.concat);
    int total_fanins = concat_view->NumRegularFanins();
    int new_total_fanins = new_n + 1;

    for (int vi = 0; vi < new_n; ++vi) {
      const std::string& inp = new_value_inputs[vi];
      auto colon = inp.find(':');
      std::string node_name =
          (colon == std::string::npos) ? inp : inp.substr(0, colon);
      int port =
          (colon == std::string::npos) ? 0 : std::stoi(inp.substr(colon + 1));
      mutation->AddOrUpdateRegularFanin(concat_view, vi,
                                        TensorId(node_name, port));
    }
    {
      const std::string& axis_inp = original_inputs[n_values];
      auto colon = axis_inp.find(':');
      std::string node_name =
          (colon == std::string::npos) ? axis_inp : axis_inp.substr(0, colon);
      int port = (colon == std::string::npos)
                     ? 0
                     : std::stoi(axis_inp.substr(colon + 1));
      mutation->AddOrUpdateRegularFanin(concat_view, new_n,
                                        TensorId(node_name, port));
    }
    for (int fi = total_fanins - 1; fi >= new_total_fanins; --fi)
      mutation->RemoveRegularFanin(concat_view, fi);

    AttrValue n_attr;
    n_attr.set_i(new_n);
    mutation->AddOrUpdateNodeAttr(concat_view, "N", n_attr);

    TF_RETURN_IF_ERROR(mutation->Apply());
    (*invalidated_nodes)[matched.concat] = true;
  }

  return OkStatus();
}

Status AddGroupEmbeddingNode(RemapperContext* ctx,
                             const GroupEmbedding& matched,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete) {
  if (matched.full_fusion) {
    return AddFullFusionGroupEmbeddingNode(ctx, matched, invalidated_nodes,
                                           nodes_to_delete);
  }
  // Multi-table fusion: merge all runs into a single op, keep ConcatV2
  // for any remaining non-gather inputs.
  return AddMultiTableGroupEmbeddingNode(ctx, matched, invalidated_nodes,
                                         nodes_to_delete);
}

bool IsFusedMatMulBiasAddOnly(const NodeDef& node) {
  if (node.op() != kFusedMatMul && node.op() != "_ZenFusedMatMul") return false;
  if (!node.attr().contains("fused_ops")) return false;
  const auto& fused_ops = node.attr().at("fused_ops").list().s();
  if (fused_ops.size() != 1 || fused_ops[0] != "BiasAdd") return false;
  int num_args = 0;
  Status num_args_st = GetNodeAttr(node, "num_args", &num_args);
  if (num_args_st.ok()) {
    if (num_args != 1) return false;
  } else if (node.input_size() < 3) {
    // Some imported graphs omit num_args; three inputs (a, b, bias) imply one
    // arg.
    return false;
  }
  return node.input_size() >= 3;
}

bool FuseMatmulBNfoldTensorShapesOk(const Tensor& weight, const Tensor& bias,
                                    const Tensor& scale, const Tensor& shift,
                                    DataType t) {
  // MatMul T may be BF16 while frozen weights/bias stay FP32 Const (mixed
  // precision); fold in float then cast updated values to BF16 for the Const.
  const bool weight_bias_types_ok =
      (weight.dtype() == t && bias.dtype() == t) ||
      (t == DT_BFLOAT16 && weight.dtype() == DT_FLOAT &&
       bias.dtype() == DT_FLOAT);
  if (!weight_bias_types_ok) {
    return false;
  }
  if (scale.dtype() != t || shift.dtype() != t) {
    if (t != DT_BFLOAT16 || scale.dtype() != DT_FLOAT ||
        shift.dtype() != DT_FLOAT) {
      return false;
    }
  }
  if (t != DT_FLOAT && t != DT_BFLOAT16) return false;
  if (weight.dims() != 2) return false;
  const int64_t n_out = weight.dim_size(1);
  if (bias.NumElements() != n_out || scale.NumElements() != n_out ||
      shift.NumElements() != n_out) {
    return false;
  }
  return true;
}

void ApplyFuseMatmulBNfoldToTensors(Tensor* weight, Tensor* bias,
                                    const Tensor& scale, const Tensor& shift,
                                    DataType t) {
  const int64_t k_in = weight->dim_size(0);
  const int64_t n_out = weight->dim_size(1);
  if (t == DT_FLOAT) {
    auto w = weight->matrix<float>();
    auto bv = bias->flat<float>();
    auto sv = scale.flat<float>();
    auto hv = shift.flat<float>();
    for (int64_t j = 0; j < n_out; ++j) {
      const float s = sv(j);
      for (int64_t k = 0; k < k_in; ++k) w(k, j) *= s;
      bv(j) = bv(j) * s + hv(j);
    }
    return;
  }
  if (t == DT_BFLOAT16 && weight->dtype() == DT_FLOAT &&
      bias->dtype() == DT_FLOAT) {
    // Fold in-place on FP32 tensors, then allocate BF16 outputs once (no extra
    // full-matrix copies of W/B before the math).
    auto w_mat = weight->matrix<float>();
    auto bv_flat = bias->flat<float>();
    if (scale.dtype() == DT_FLOAT && shift.dtype() == DT_FLOAT) {
      auto sv = scale.flat<float>();
      auto hv = shift.flat<float>();
      for (int64_t j = 0; j < n_out; ++j) {
        const float s = sv(j);
        const float h = hv(j);
        for (int64_t k = 0; k < k_in; ++k) w_mat(k, j) *= s;
        bv_flat(j) = bv_flat(j) * s + h;
      }
    } else {
      auto sv = scale.flat<Eigen::bfloat16>();
      auto hv = shift.flat<Eigen::bfloat16>();
      for (int64_t j = 0; j < n_out; ++j) {
        const float s = static_cast<float>(sv(j));
        const float h = static_cast<float>(hv(j));
        for (int64_t k = 0; k < k_in; ++k) w_mat(k, j) *= s;
        bv_flat(j) = bv_flat(j) * s + h;
      }
    }
    Tensor w_out(DT_BFLOAT16, weight->shape());
    Tensor b_out(DT_BFLOAT16, bias->shape());
    auto w_dst = w_out.matrix<Eigen::bfloat16>();
    auto b_dst = b_out.flat<Eigen::bfloat16>();
    for (int64_t j = 0; j < n_out; ++j) {
      for (int64_t k = 0; k < k_in; ++k) {
        w_dst(k, j) = Eigen::bfloat16(w_mat(k, j));
      }
      b_dst(j) = Eigen::bfloat16(bv_flat(j));
    }
    *weight = std::move(w_out);
    *bias = std::move(b_out);
    return;
  }
  if (t == DT_BFLOAT16) {
    auto w = weight->matrix<Eigen::bfloat16>();
    auto bv = bias->flat<Eigen::bfloat16>();
    if (scale.dtype() == DT_FLOAT && shift.dtype() == DT_FLOAT) {
      auto sv = scale.flat<float>();
      auto hv = shift.flat<float>();
      for (int64_t j = 0; j < n_out; ++j) {
        const float s = sv(j);
        const float h = hv(j);
        for (int64_t k = 0; k < k_in; ++k) {
          w(k, j) = Eigen::bfloat16(static_cast<float>(w(k, j)) * s);
        }
        bv(j) = Eigen::bfloat16(static_cast<float>(bv(j)) * s + h);
      }
    } else {
      auto sv = scale.flat<Eigen::bfloat16>();
      auto hv = shift.flat<Eigen::bfloat16>();
      for (int64_t j = 0; j < n_out; ++j) {
        const float s = static_cast<float>(sv(j));
        const float h = static_cast<float>(hv(j));
        for (int64_t k = 0; k < k_in; ++k) {
          w(k, j) = Eigen::bfloat16(static_cast<float>(w(k, j)) * s);
        }
        bv(j) = Eigen::bfloat16(static_cast<float>(bv(j)) * s + h);
      }
    }
  }
}

// Follow Identity / StopGradient / Cast to the Const feeding a tensor input.
// Each hop must have a single data fanout and no control edges.
int PeelToConstProducerNodeIndex(const RemapperContext& ctx,
                                 utils::MutableNodeView* v) {
  for (int depth = 0; depth < 16 && v != nullptr; ++depth) {
    const NodeDef* nd = v->node();
    if (nd == nullptr) return -1;
    if (IsInPreserveSet(ctx, nd)) return -1;
    if (HasControlFaninOrFanout(*v)) return -1;
    if (!HasAtMostOneFanoutAtPort0(*v)) return -1;
    if (IsAnyConst(*nd)) return v->node_index();
    const string& op = nd->op();
    if ((op == "Identity" || op == "StopGradient" || op == kCast) &&
        v->NumRegularFanins() >= 1) {
      v = v->GetRegularFanin(0).node_view();
      continue;
    }
    // Frozen graphs often wrap variables as ReadVariableOp -> Const (or
    // Identity).
    if (op == "ReadVariableOp" && v->NumRegularFanins() >= 1) {
      v = v->GetRegularFanin(0).node_view();
      continue;
    }
    return -1;
  }
  return -1;
}

// TF 2.x often emits BatchMatMulV2 for tf.matmul. Also accept _ZenMatMul /
// _ZenBatchMatMul* when those ops are already present in the GraphDef fed to
// the remapper (the Zen plugin runs remapper before Zen layout in
// cpu_optimizer).
bool IsZenBatchMatMulLayoutOp(const NodeDef& node) {
  const auto& op = node.op();
  return op == "_ZenBatchMatMul" || op == "_ZenBatchMatMulV2";
}

bool IsMatMulZenOrBatchMatMul(const NodeDef& node) {
  return IsMatMul(node) || node.op() == "_ZenMatMul" ||
         IsAnyBatchMatMul(node) || IsZenBatchMatMulLayoutOp(node);
}

bool MatMulLikeNoTransposeOrAdj(const NodeDef& node) {
  if (IsMatMul(node) || node.op() == "_ZenMatMul") {
    bool ta = false, tb = false;
    if (node.attr().contains("transpose_a")) {
      ta = node.attr().at("transpose_a").b();
    }
    if (node.attr().contains("transpose_b")) {
      tb = node.attr().at("transpose_b").b();
    }
    return !ta && !tb;
  }
  if (IsAnyBatchMatMul(node) || IsZenBatchMatMulLayoutOp(node)) {
    bool ax = false, ay = false;
    if (node.attr().contains("adj_x")) {
      ax = node.attr().at("adj_x").b();
    }
    if (node.attr().contains("adj_y")) {
      ay = node.attr().at("adj_y").b();
    }
    return !ax && !ay;
  }
  return false;
}

// BN-fold rewrites the contraction into a strictly 2-D _FusedMatMul. A plain
// MatMul / _ZenMatMul is inherently rank-2, but BatchMatMul* (which TF 2.x can
// emit even for tf.matmul) may carry batch dimensions. Folding a genuine
// rank-3+ batched op into a 2-D _FusedMatMul would silently drop the batch
// dimension, so only allow batched ops whose operands are statically rank-2.
// Unknown rank (Rank() == -1) is rejected conservatively.
bool MatMulFoldRankOk(const RemapperContext& ctx, const NodeDef& node) {
  if (!IsAnyBatchMatMul(node) && !IsZenBatchMatMulLayoutOp(node)) {
    return true;
  }
  std::vector<OpInfo_TensorProperties> props;
  if (!ctx.graph_properties.GetInputProperties(node.name(), &props).ok()) {
    return false;
  }
  if (props.size() < 2) {
    return false;
  }
  return Rank(props[0].shape()) == 2 && Rank(props[1].shape()) == 2;
}

bool MatchMulScaleBiasAddTail(
    const RemapperContext& ctx, const utils::MutableNodeView* mul_view,
    const utils::MutableNodeView* shift_view,
    const utils::MutableNodeView** fused_or_bias_view_out,
    const utils::MutableNodeView** scale_view_out,
    const utils::MutableNodeView** shift_const_view_out) {
  if (mul_view == nullptr || shift_view == nullptr) return false;
  const auto* mul_def = mul_view->node();
  if (mul_def == nullptr) return false;
  if (!IsAnyMul(*mul_def)) return false;
  // Incoming control edges can prevent reordering; outgoing control edges are
  // common on TF executor graphs and are preserved when mutating fanins.
  if (HasControlFanin(*mul_view) || mul_view->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, mul_def)) {
    return false;
  }

  utils::MutableNodeView* shift_mutable = const_cast<utils::MutableNodeView*>(
      ctx.graph_view.GetNode(shift_view->node_index()));
  int shift_const_ix = PeelToConstProducerNodeIndex(ctx, shift_mutable);
  if (shift_const_ix < 0) return false;
  const auto* shift_const_view = ctx.graph_view.GetNode(shift_const_ix);
  // Match the scale_const checks below: a control edge on the shift Const would
  // be lost when the node is deleted, so reject it here too.
  if (HasControlFaninOrFanout(*shift_const_view) ||
      !HasAtMostOneFanoutAtPort0(*shift_const_view))
    return false;

  for (int mul_fm_port = 0; mul_fm_port < 2; ++mul_fm_port) {
    const auto* chain_view = mul_view->GetRegularFanin(mul_fm_port).node_view();
    utils::MutableNodeView* scale_mutable = const_cast<utils::MutableNodeView*>(
        mul_view->GetRegularFanin(1 - mul_fm_port).node_view());
    if (chain_view == nullptr || scale_mutable == nullptr) continue;
    const auto* chain_def = chain_view->node();
    if (chain_def == nullptr) continue;
    int scale_ix = PeelToConstProducerNodeIndex(ctx, scale_mutable);
    if (scale_ix < 0) continue;
    const auto* scale_const_view = ctx.graph_view.GetNode(scale_ix);
    if (HasControlFaninOrFanout(*scale_const_view) ||
        !HasAtMostOneFanoutAtPort0(*scale_const_view)) {
      continue;
    }
    *fused_or_bias_view_out = chain_view;
    *scale_view_out = scale_const_view;
    *shift_const_view_out = shift_const_view;
    return true;
  }
  return false;
}

bool GetCastDataTypes(const NodeDef& n, DataType* src_t, DataType* dst_t) {
  if (!IsCast(n)) return false;
  if (!n.attr().contains("SrcT") || !n.attr().contains("DstT")) return false;
  *src_t = n.attr().at("SrcT").type();
  *dst_t = n.attr().at("DstT").type();
  return true;
}

bool IsNoopBf16Cast(const NodeDef& n) {
  DataType st, dt;
  if (!GetCastDataTypes(n, &st, &dt)) return false;
  return st == DT_BFLOAT16 && dt == DT_BFLOAT16;
}

bool MishDecomposedIsStaleF32ToBf16CastOnBf16MatmulOutput(
    const NodeDef* fused_def, const utils::MutableNodeView* child) {
  if (fused_def == nullptr || child == nullptr || child->node() == nullptr) {
    return false;
  }
  if (!HasDataType(fused_def, DT_BFLOAT16)) return false;
  if (!IsCast(*(child->node()))) return false;
  if (HasControlFaninOrFanout(*child)) return false;
  DataType st = DT_INVALID, dt = DT_INVALID;
  if (!GetCastDataTypes(*(child->node()), &st, &dt)) return false;
  return st == DT_FLOAT && dt == DT_BFLOAT16;
}

bool MishDecomposedCastHasAnyRegularConsumer(const utils::MutableNodeView* v) {
  if (v == nullptr) return false;
  for (int i = 0; i < v->NumRegularFanouts(); ++i) {
    if (!v->GetRegularFanout(i).empty()) return true;
  }
  return false;
}

void MishDecomposedAppendUniqueCast(std::vector<int>* v, int node_index) {
  if (v == nullptr) return;
  if (std::find(v->begin(), v->end(), node_index) == v->end()) {
    v->push_back(node_index);
  }
}

utils::MutableNodeView* SkipIdentityOrNoopBf16CastTowardProducer(
    utils::MutableNodeView* v, bool allow_bf16_noop_cast_on_chain,
    std::vector<int>* noop_bf16_cast_indices) {
  for (int h = 0; h < 16 && v != nullptr; ++h) {
    NodeDef* nd = v->node();
    if (nd == nullptr) break;
    if (nd->op() == "Identity") {
      if (v->NumRegularFanins() < 1) break;
      v = v->GetRegularFanin(0).node_view();
      continue;
    }
    if (allow_bf16_noop_cast_on_chain && IsCast(*nd) && IsNoopBf16Cast(*nd)) {
      if (v->NumRegularFanins() < 1) break;
      MishDecomposedAppendUniqueCast(noop_bf16_cast_indices, v->node_index());
      v = v->GetRegularFanin(0).node_view();
      continue;
    }
    break;
  }
  return v;
}

bool MishDecomposedForwardReachesSoftplusOrMul(
    utils::MutableNodeView* start, int target_node_index,
    bool allow_bf16_noop_cast_forward, const NodeDef* fused_def,
    const NodeDef* endpoint_dtype_node) {
  const bool allow_cast = allow_bf16_noop_cast_forward &&
                          HasDataType(fused_def, DT_BFLOAT16) &&
                          HasDataType(endpoint_dtype_node, DT_BFLOAT16);
  utils::MutableNodeView* v = start;
  for (int h = 0; h < 16 && v != nullptr; ++h) {
    if (v->node_index() == target_node_index) return true;
    const NodeDef* nd = v->node();
    if (nd == nullptr) return false;
    const auto& outs = v->GetRegularFanout(0);
    if (outs.size() != 1) return false;
    utils::MutableNodeView* next = outs[0].node_view();
    if (nd->op() == "Identity") {
      v = next;
      continue;
    }
    if (allow_cast && IsCast(*nd)) {
      DataType st = DT_INVALID, dt = DT_INVALID;
      if (!GetCastDataTypes(*nd, &st, &dt)) return false;
      // BF16 no-op cast, or stale Float->BF16 cast on BF16 _FusedMatMul output
      // (same dtype policy as Mish decomposed removal of
      // redundant_f32_to_bf16_* casts).
      if ((st == DT_BFLOAT16 && dt == DT_BFLOAT16) ||
          (st == DT_FLOAT && dt == DT_BFLOAT16)) {
        v = next;
        continue;
      }
      return false;
    }
    return false;
  }
  return false;
}

bool MishDecomposedVerifyMatmulOutputOnlySoftplusAndMul(
    utils::MutableNodeView* fused_view, utils::MutableNodeView* sp_view,
    int mish_mul_node_index, const NodeDef* fused_def, const NodeDef* mul_def) {
  const auto& direct = fused_view->GetRegularFanout(0);
  int n_sp = 0;
  int n_mul = 0;
  const bool fbf16 = HasDataType(fused_def, DT_BFLOAT16);
  for (const auto& fo : direct) {
    utils::MutableNodeView* child = fo.node_view();
    if (child == nullptr || child->node() == nullptr) {
      return false;
    }
    if (MishDecomposedIsStaleF32ToBf16CastOnBf16MatmulOutput(fused_def,
                                                             child) &&
        !MishDecomposedCastHasAnyRegularConsumer(child)) {
      // Third consumer can be a dead stale Float->BF16 cast on BF16 matmul
      // output — not part of the Softplus vs Mish Mul fork.
      continue;
    }
    const bool ok_sp = MishDecomposedForwardReachesSoftplusOrMul(
        child, sp_view->node_index(),
        fbf16 && HasDataType(sp_view->node(), DT_BFLOAT16), fused_def,
        sp_view->node());
    const bool ok_mul = MishDecomposedForwardReachesSoftplusOrMul(
        child, mish_mul_node_index, fbf16 && HasDataType(mul_def, DT_BFLOAT16),
        fused_def, mul_def);
    if (ok_sp == ok_mul) {
      // A *live* stale Float->BF16 cast (one with real consumers) cannot be
      // safely folded away: those consumers want the matmul+BiasAdd output, but
      // after fusion the only surviving node produces Mish(matmul+BiasAdd) -- a
      // different value -- and the old matmul is deleted, so there is no valid
      // tensor to rewire them to. Only a *dead* stale cast (handled above) is
      // tolerable; bail otherwise so the matmul keeps its standalone output.
      return false;
    }
    if (ok_sp) {
      ++n_sp;
    } else {
      ++n_mul;
    }
  }
  if (n_sp != 1 || n_mul != 1) {
    return false;
  }
  return true;
}

utils::MutableNodeView* SkipIdentityOnly(utils::MutableNodeView* v,
                                         int max_hops) {
  for (int h = 0; h < max_hops && v != nullptr; ++h) {
    NodeDef* nd = v->node();
    if (nd == nullptr || nd->op() != "Identity") break;
    if (v->NumRegularFanins() < 1) break;
    v = v->GetRegularFanin(0).node_view();
  }
  return v;
}

double ReadTensorElemAsDouble(const Tensor& t, int64_t j) {
  switch (t.dtype()) {
    case DT_FLOAT:
      return static_cast<double>(t.flat<float>()(j));
    case DT_BFLOAT16:
      return static_cast<double>(
          static_cast<float>(t.flat<Eigen::bfloat16>()(j)));
    default:
      return 0.0;
  }
}

bool BuildKerasBnScaleShiftTensors(const Tensor& gamma, const Tensor& beta,
                                   const Tensor& mean, const Tensor& variance,
                                   float epsilon, Tensor* scale_dt_float,
                                   Tensor* shift_dt_float) {
  const int64_t n = gamma.NumElements();
  if (beta.NumElements() != n || mean.NumElements() != n ||
      variance.NumElements() != n) {
    return false;
  }
  Tensor S(DT_FLOAT, {n});
  Tensor H(DT_FLOAT, {n});
  for (int64_t j = 0; j < n; ++j) {
    const double g = ReadTensorElemAsDouble(gamma, j);
    const double b = ReadTensorElemAsDouble(beta, j);
    const double m = ReadTensorElemAsDouble(mean, j);
    const double v = ReadTensorElemAsDouble(variance, j);
    const double rad = v + static_cast<double>(epsilon);
    // Use !(rad > 0.0) rather than rad <= 0.0 so that a NaN radicand (e.g. from
    // a negative/garbage variance entry) is rejected too -- every comparison
    // with NaN is false, so `rad <= 0.0` would let NaN slip through and get
    // folded into the weights silently.
    if (!(rad > 0.0)) return false;
    const double sigma = std::sqrt(rad);
    if (!std::isfinite(sigma) || sigma <= 0.0) return false;
    const double s = g / sigma;
    const double h = b - m * s;
    if (!std::isfinite(s) || !std::isfinite(h)) return false;
    S.flat<float>()(j) = static_cast<float>(s);
    H.flat<float>()(j) = static_cast<float>(h);
  }
  *scale_dt_float = std::move(S);
  *shift_dt_float = std::move(H);
  return true;
}

// Upper bound on a plausible BatchNorm epsilon. Common values are 1e-5..1e-3
// (Keras default 1e-3); anything notably larger is treated as "not an epsilon".
constexpr float kMaxBatchNormEpsilon = 0.1f;

// BN folding rewrites the weight/bias Const tensors in place. We stamp this
// internal marker (attr names starting with '_' are ignored by TF op
// validation) on a Const once its values have been folded, so a later pass /
// recheck iteration can never apply the scale/shift a second time.
constexpr char kZenBnFoldedAttr[] = "_zen_bn_folded";

bool IsConstAlreadyBnFolded(const NodeDef* node) {
  return node != nullptr && node->attr().contains(kZenBnFoldedAttr);
}

void MarkConstBnFolded(utils::Mutation* mutation,
                       utils::MutableNodeView* const_view) {
  AttrValue marker;
  marker.set_b(true);
  mutation->AddOrUpdateNodeAttr(const_view, kZenBnFoldedAttr, marker);
}

bool ParseVarianceConstFromAddVariancePlusEps(
    const RemapperContext& ctx, utils::MutableNodeView* add_eps_view,
    float default_eps, float* epsilon_out, int* variance_const_ix_out) {
  *epsilon_out = default_eps;
  *variance_const_ix_out = -1;
  if (add_eps_view == nullptr || add_eps_view->NumRegularFanins() < 2)
    return false;
  bool have_multi_elem_const = false;
  for (int p = 0; p < 2; ++p) {
    const auto* fin = add_eps_view->GetRegularFanin(p).node_view();
    if (fin == nullptr) return false;
    utils::MutableNodeView* branch = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(fin->node_index()));
    if (branch == nullptr) return false;
    branch = SkipIdentityOnly(branch);
    int const_ix = PeelToConstProducerNodeIndex(ctx, branch);
    if (const_ix < 0) continue;
    const NodeDef* cdef = ctx.graph_view.GetNode(const_ix)->node();
    Tensor t;
    if (!GetTensorFromConstant(cdef, &t).ok()) continue;
    if (t.dtype() != DT_FLOAT && t.dtype() != DT_BFLOAT16) continue;
    if (t.NumElements() == 1) {
      // The scalar operand of (variance + epsilon) is the epsilon. Identify it
      // by structure (scalar) rather than by magnitude, but require a sane
      // finite, non-negative value within the BatchNorm epsilon range. An
      // out-of-range scalar means this is not a standard var+eps add, so reject
      // the whole match instead of silently folding with the wrong epsilon.
      const double eps = ReadTensorElemAsDouble(t, 0);
      if (!std::isfinite(eps) || eps < 0.0 || eps > kMaxBatchNormEpsilon) {
        return false;
      }
      *epsilon_out = static_cast<float>(eps);
    } else {
      // The multi-element operand is the variance: must be finite and
      // non-negative so that sqrt(var + eps) is real (guards against
      // NaN/garbage being folded into the weights).
      const int64_t n = t.NumElements();
      for (int64_t j = 0; j < n; ++j) {
        const double v = ReadTensorElemAsDouble(t, j);
        if (!std::isfinite(v) || v < 0.0) return false;
      }
      *variance_const_ix_out = const_ix;
      have_multi_elem_const = true;
    }
  }
  return have_multi_elem_const && *variance_const_ix_out >= 0;
}

// Linear decomposed inference BN (e.g. explicit tf.raw_ops after
// MatMul+BiasAdd):
//   Add|AddV2( Mul( Mul( Sub(BiasAdd(MatMul), mean), Rsqrt(var+ε) ), gamma ),
//   beta )
// Operand order may vary on Add and on each Mul.
bool FindMatMulBiasKerasBnFoldLinearChain(const RemapperContext& ctx,
                                          int node_index,
                                          MatMulBiasKerasBnFoldMatch* matched) {
  const auto* add_view = ctx.graph_view.GetNode(node_index);
  const auto* add_def = add_view->node();
  if (add_def == nullptr || (!IsAdd(*add_def) && add_def->op() != kAddV2)) {
    return false;
  }
  if (HasControlFanin(*add_view) || add_view->NumRegularFanins() != 2) {
    return false;
  }

  for (int mul_gamma_port = 0; mul_gamma_port < 2; ++mul_gamma_port) {
    const utils::MutableNodeView* mul_gamma_view =
        add_view->GetRegularFanin(mul_gamma_port).node_view();
    const utils::MutableNodeView* beta_side_view =
        add_view->GetRegularFanin(1 - mul_gamma_port).node_view();
    if (mul_gamma_view == nullptr || beta_side_view == nullptr) {
      continue;
    }
    const NodeDef* mul_gamma_def = mul_gamma_view->node();
    if (mul_gamma_def == nullptr || !IsAnyMul(*mul_gamma_def)) {
      continue;
    }
    if (HasControlFanin(*mul_gamma_view) ||
        mul_gamma_view->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, mul_gamma_def) ||
        !HasAtMostOneFanoutAtPort0(*mul_gamma_view)) {
      continue;
    }

    utils::MutableNodeView* beta_mut = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(beta_side_view->node_index()));
    beta_mut = SkipIdentityOnly(beta_mut);
    int beta_ix = PeelToConstProducerNodeIndex(ctx, beta_mut);
    if (beta_ix < 0) {
      continue;
    }
    Tensor Beta;
    if (!GetTensorFromConstant(ctx.graph_view.GetNode(beta_ix)->node(), &Beta)
             .ok()) {
      continue;
    }

    const utils::MutableNodeView* inner_mul_view = nullptr;
    int gamma_ix = kMissingIndex;
    for (int gp = 0; gp < 2; ++gp) {
      const auto* branch_mul = mul_gamma_view->GetRegularFanin(gp).node_view();
      const auto* branch_other =
          mul_gamma_view->GetRegularFanin(1 - gp).node_view();
      if (branch_mul == nullptr || branch_other == nullptr) {
        continue;
      }
      const NodeDef* bm_def = branch_mul->node();
      if (bm_def == nullptr || !IsAnyMul(*bm_def)) {
        continue;
      }
      utils::MutableNodeView* gamma_peel = const_cast<utils::MutableNodeView*>(
          ctx.graph_view.GetNode(branch_other->node_index()));
      gamma_peel = SkipIdentityOnly(gamma_peel);
      int g_ix = PeelToConstProducerNodeIndex(ctx, gamma_peel);
      if (g_ix < 0) {
        continue;
      }
      Tensor Gamma_probe;
      if (!GetTensorFromConstant(ctx.graph_view.GetNode(g_ix)->node(),
                                 &Gamma_probe)
               .ok()) {
        continue;
      }
      inner_mul_view = branch_mul;
      gamma_ix = g_ix;
      break;
    }
    if (inner_mul_view == nullptr || gamma_ix == kMissingIndex) {
      continue;
    }
    const NodeDef* inner_mul_def = inner_mul_view->node();
    if (HasControlFanin(*inner_mul_view) ||
        inner_mul_view->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, inner_mul_def) ||
        !HasAtMostOneFanoutAtPort0(*inner_mul_view)) {
      continue;
    }

    const utils::MutableNodeView* sub_view = nullptr;
    const utils::MutableNodeView* rsqrt_view = nullptr;
    for (int ip = 0; ip < 2; ++ip) {
      const auto* a = inner_mul_view->GetRegularFanin(ip).node_view();
      const auto* b = inner_mul_view->GetRegularFanin(1 - ip).node_view();
      if (a == nullptr || b == nullptr) {
        continue;
      }
      const NodeDef* ad = a->node();
      const NodeDef* bd = b->node();
      if (ad == nullptr || bd == nullptr) {
        continue;
      }
      if (IsSub(*ad) && IsRsqrt(*bd)) {
        sub_view = a;
        rsqrt_view = b;
        break;
      }
      if (IsSub(*bd) && IsRsqrt(*ad)) {
        sub_view = b;
        rsqrt_view = a;
        break;
      }
    }
    if (sub_view == nullptr || rsqrt_view == nullptr) {
      continue;
    }
    const NodeDef* sub_def = sub_view->node();
    const NodeDef* rsqrt_def = rsqrt_view->node();
    if (HasControlFanin(*sub_view) || sub_view->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, sub_def) ||
        !HasAtMostOneFanoutAtPort0(*sub_view)) {
      continue;
    }
    if (HasControlFanin(*rsqrt_view) || rsqrt_view->NumRegularFanins() != 1 ||
        IsInPreserveSet(ctx, rsqrt_def) ||
        !HasAtMostOneFanoutAtPort0(*rsqrt_view)) {
      continue;
    }

    // Sub(linear, mean): fanin 0 is the MatMul+BiasAdd chain (optional Cast).
    const utils::MutableNodeView* linear_side =
        sub_view->GetRegularFanin(0).node_view();
    const utils::MutableNodeView* mean_side =
        sub_view->GetRegularFanin(1).node_view();
    if (linear_side == nullptr || mean_side == nullptr) {
      continue;
    }

    const utils::MutableNodeView* bias_cast_view = nullptr;
    utils::MutableNodeView* bias_add_mut = nullptr;
    utils::MutableNodeView* linear_head = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(linear_side->node_index()));
    linear_head = SkipIdentityOnly(linear_head);
    if (linear_head == nullptr) {
      continue;
    }
    NodeDef* lh_def = linear_head->node();
    if (lh_def != nullptr && lh_def->op() == kCast) {
      if (HasControlFanin(*linear_head) ||
          linear_head->NumRegularFanins() < 1 || IsInPreserveSet(ctx, lh_def) ||
          !HasAtMostOneFanoutAtPort0(*linear_head)) {
        continue;
      }
      bias_cast_view = linear_head;
      linear_head = linear_head->GetRegularFanin(0).node_view();
      lh_def = linear_head != nullptr ? linear_head->node() : nullptr;
    }
    int bias_port = 1;
    if (linear_head == nullptr || lh_def == nullptr ||
        (!IsBiasAdd(*lh_def) &&
         !IsBiasSemanticAdd(ctx, *linear_head, &bias_port))) {
      continue;
    }
    if (HasControlFanin(*linear_head) || linear_head->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, lh_def) ||
        !HasAtMostOneFanoutAtPort0(*linear_head)) {
      continue;
    }
    bias_add_mut = linear_head;
    const auto* matmul_view =
        bias_add_mut->GetRegularFanin(1 - bias_port).node_view();
    const NodeDef* matmul_def =
        matmul_view != nullptr ? matmul_view->node() : nullptr;
    if (matmul_view == nullptr || matmul_def == nullptr) {
      continue;
    }
    if (!IsMatMulZenOrBatchMatMul(*matmul_def)) {
      continue;
    }
    if (!MatMulLikeNoTransposeOrAdj(*matmul_def)) {
      continue;
    }
    if (!MatMulFoldRankOk(ctx, *matmul_def)) {
      continue;
    }
    if (!NodeIsOnCpu(matmul_def)) {
      continue;
    }
    if (HasControlFanin(*matmul_view) ||
        !HasAtMostOneFanoutAtPort0(*matmul_view) ||
        IsInPreserveSet(ctx, matmul_def)) {
      continue;
    }
    if (HasDataType(matmul_def, DT_DOUBLE)) {
      continue;
    }

    utils::MutableNodeView* mean_mut = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(mean_side->node_index()));
    mean_mut = SkipIdentityOnly(mean_mut);
    int mean_ix = PeelToConstProducerNodeIndex(ctx, mean_mut);
    if (mean_ix < 0) {
      continue;
    }
    Tensor Mean;
    if (!GetTensorFromConstant(ctx.graph_view.GetNode(mean_ix)->node(), &Mean)
             .ok()) {
      continue;
    }

    const auto* rsqrt_in = rsqrt_view->GetRegularFanin(0).node_view();
    if (rsqrt_in == nullptr) {
      continue;
    }
    utils::MutableNodeView* add_eps_mut = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(rsqrt_in->node_index()));
    if (add_eps_mut == nullptr) {
      continue;
    }
    add_eps_mut = SkipIdentityOnly(add_eps_mut);
    NodeDef* add_eps_def = add_eps_mut->node();
    if (add_eps_def == nullptr ||
        (add_eps_def->op() != kAddV2 && !IsAdd(*add_eps_def))) {
      continue;
    }
    if (HasControlFanin(*add_eps_mut) || add_eps_mut->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, add_eps_def) ||
        !HasAtMostOneFanoutAtPort0(*add_eps_mut)) {
      continue;
    }

    float epsilon = 1e-3f;
    int var_const_ix = kMissingIndex;
    if (!ParseVarianceConstFromAddVariancePlusEps(ctx, add_eps_mut, 1e-3f,
                                                  &epsilon, &var_const_ix)) {
      continue;
    }

    Tensor Gamma;
    if (!GetTensorFromConstant(ctx.graph_view.GetNode(gamma_ix)->node(), &Gamma)
             .ok()) {
      continue;
    }
    const NodeDef* var_def = ctx.graph_view.GetNode(var_const_ix)->node();
    Tensor Variance;
    if (!GetTensorFromConstant(var_def, &Variance).ok()) {
      continue;
    }

    DataType t_mm = GetDataTypeFromAttr(*matmul_def, "T");
    DataType t_inner = GetDataTypeFromAttr(*inner_mul_def, "T");
    if (bias_cast_view != nullptr) {
      DataType c_src = DT_INVALID, c_dst = DT_INVALID;
      if (!GetCastDataTypes(*(bias_cast_view->node()), &c_src, &c_dst)) {
        continue;
      }
      if (c_src != t_mm || c_dst != t_inner) {
        continue;
      }
      if (!HaveSameDataType(add_def, mul_gamma_def)) {
        continue;
      }
    } else {
      if (!HaveSameDataType(add_def, matmul_def)) {
        continue;
      }
      if (!HaveSameDataType(inner_mul_def, matmul_def)) {
        continue;
      }
    }

    utils::MutableNodeView* w_mutable = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(ParseTensorName(matmul_def->input(1)).node()));
    utils::MutableNodeView* b_mutable =
        const_cast<utils::MutableNodeView*>(ctx.graph_view.GetNode(
            ParseTensorName(bias_add_mut->node()->input(bias_port)).node()));
    if (w_mutable == nullptr || b_mutable == nullptr) {
      continue;
    }
    int w_ix = PeelToConstProducerNodeIndex(ctx, w_mutable);
    int b_ix = PeelToConstProducerNodeIndex(ctx, b_mutable);
    if (w_ix < 0 || b_ix < 0) {
      continue;
    }
    if (IsConstAlreadyBnFolded(ctx.graph_view.GetNode(w_ix)->node()) ||
        IsConstAlreadyBnFolded(ctx.graph_view.GetNode(b_ix)->node())) {
      continue;
    }

    Tensor W;
    Tensor B;
    if (!GetTensorFromConstant(ctx.graph_view.GetNode(w_ix)->node(), &W).ok() ||
        !GetTensorFromConstant(ctx.graph_view.GetNode(b_ix)->node(), &B).ok()) {
      continue;
    }

    Tensor Sfloat;
    Tensor Hfloat;
    if (!BuildKerasBnScaleShiftTensors(Gamma, Beta, Mean, Variance, epsilon,
                                       &Sfloat, &Hfloat)) {
      continue;
    }
    if (!FuseMatmulBNfoldTensorShapesOk(W, B, Sfloat, Hfloat, t_mm)) {
      continue;
    }

    int tail_cast_ix = kMissingIndex;
    const auto& keras_add_fanouts = add_view->GetRegularFanout(0);
    if (keras_add_fanouts.size() == 1) {
      const auto* tail_view = keras_add_fanouts[0].node_view();
      const NodeDef* tail_def =
          tail_view != nullptr ? tail_view->node() : nullptr;
      if (tail_view != nullptr && tail_def != nullptr && IsCast(*tail_def) &&
          !HasControlFanin(*tail_view)) {
        DataType tc_src = DT_INVALID, tc_dst = DT_INVALID;
        if (GetCastDataTypes(*tail_def, &tc_src, &tc_dst) &&
            tc_src == GetDataTypeFromAttr(*add_def, "T") && tc_dst == t_mm) {
          tail_cast_ix = tail_view->node_index();
        }
      }
    }

    matched->matmul = matmul_view->node_index();
    matched->bias_add = bias_add_mut->node_index();
    matched->mul_1 = mul_gamma_view->node_index();
    matched->keras_bn_add = node_index;
    matched->sub = sub_view->node_index();
    matched->mul_scale = inner_mul_view->node_index();
    matched->mul_mean = kMissingIndex;
    matched->rsqrt = rsqrt_view->node_index();
    matched->add_eps = add_eps_mut->node_index();
    matched->gamma_const = gamma_ix;
    matched->beta_const = beta_ix;
    matched->mean_const = mean_ix;
    matched->variance_const = var_const_ix;
    matched->epsilon = epsilon;
    matched->bias_port = bias_port;
    matched->bias_cast = bias_cast_view != nullptr
                             ? bias_cast_view->node_index()
                             : kMissingIndex;
    matched->tail_cast = tail_cast_ix;
    return true;
  }
  return false;
}

bool FindMatMulBiasKerasBnFold(const RemapperContext& ctx, int node_index,
                               MatMulBiasKerasBnFoldMatch* matched) {
  if (FindMatMulBiasKerasBnFoldLinearChain(ctx, node_index, matched)) {
    return true;
  }
  const auto* add_view = ctx.graph_view.GetNode(node_index);
  const auto* add_def = add_view->node();
  if (add_def == nullptr || (!IsAdd(*add_def) && add_def->op() != kAddV2)) {
    return false;
  }
  if (HasControlFanin(*add_view) || add_view->NumRegularFanins() != 2) {
    return false;
  }

  const auto* in0 = add_view->GetRegularFanin(0).node_view();
  const auto* in1 = add_view->GetRegularFanin(1).node_view();
  if (in0 == nullptr || in1 == nullptr) return false;
  const NodeDef* d0 = in0->node();
  const NodeDef* d1 = in1->node();
  if (d0 == nullptr || d1 == nullptr) return false;

  const utils::MutableNodeView *mul_1_view = nullptr, *sub_view = nullptr;
  if (IsAnyMul(*d0) && IsSub(*d1)) {
    mul_1_view = in0;
    sub_view = in1;
  } else if (IsAnyMul(*d1) && IsSub(*d0)) {
    mul_1_view = in1;
    sub_view = in0;
  } else {
    return false;
  }

  const NodeDef* mul_1_def = mul_1_view->node();
  const NodeDef* sub_def = sub_view->node();
  if (HasControlFanin(*mul_1_view) || mul_1_view->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, mul_1_def) ||
      !HasAtMostOneFanoutAtPort0(*mul_1_view)) {
    return false;
  }
  if (HasControlFanin(*sub_view) || sub_view->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, sub_def)) {
    return false;
  }

  const utils::MutableNodeView* bias_cast_view = nullptr;
  utils::MutableNodeView* bias_add_mut = nullptr;

  auto resolve_mul1_bias_side =
      [&](const utils::MutableNodeView* chain_a,
          const utils::MutableNodeView* chain_b) -> bool {
    for (int chain_port = 0; chain_port < 2; ++chain_port) {
      const utils::MutableNodeView* chain =
          (chain_port == 0 ? chain_a : chain_b);
      const utils::MutableNodeView* other =
          (chain_port == 0 ? chain_b : chain_a);
      if (chain == nullptr || other == nullptr) {
        continue;
      }
      const NodeDef* od = other->node();
      if (od == nullptr || !IsAnyMul(*od)) {
        continue;
      }
      utils::MutableNodeView* head = const_cast<utils::MutableNodeView*>(
          ctx.graph_view.GetNode(chain->node_index()));
      head = SkipIdentityOnly(head);
      if (head == nullptr) {
        continue;
      }
      NodeDef* hd = head->node();
      if (hd == nullptr) {
        continue;
      }
      if (hd->op() == kCast) {
        if (HasControlFanin(*head) || head->NumRegularFanins() < 1 ||
            IsInPreserveSet(ctx, hd) || !HasAtMostOneFanoutAtPort0(*head)) {
          continue;
        }
        bias_cast_view = head;
        head = head->GetRegularFanin(0).node_view();
        hd = head != nullptr ? head->node() : nullptr;
      }
      int port_temp = 1;
      if (head == nullptr || hd == nullptr ||
          (!IsBiasAdd(*hd) && !IsBiasSemanticAdd(ctx, *head, &port_temp))) {
        continue;
      }
      if (HasControlFanin(*head) || head->NumRegularFanins() != 2 ||
          IsInPreserveSet(ctx, hd) || !HasAtMostOneFanoutAtPort0(*head)) {
        continue;
      }
      bias_add_mut = head;
      return true;
    }
    return false;
  };

  if (!resolve_mul1_bias_side(mul_1_view->GetRegularFanin(0).node_view(),
                              mul_1_view->GetRegularFanin(1).node_view())) {
    return false;
  }

  DCHECK(bias_add_mut != nullptr);
  const NodeDef* bias_add_def = bias_add_mut->node();
  int bias_port = 1;
  const auto& contraction_fanin = bias_add_mut->GetRegularFanin(1 - bias_port);
  const auto* matmul_view = contraction_fanin.node_view();
  const auto* matmul_def =
      matmul_view != nullptr ? matmul_view->node() : nullptr;
  if (matmul_view == nullptr || matmul_def == nullptr) {
    return false;
  }
  if (!IsMatMulZenOrBatchMatMul(*matmul_def)) {
    return false;
  }
  if (!MatMulLikeNoTransposeOrAdj(*matmul_def)) {
    return false;
  }
  if (!MatMulFoldRankOk(ctx, *matmul_def)) {
    return false;
  }
  if (!NodeIsOnCpu(matmul_def)) {
    return false;
  }
  if (HasControlFanin(*matmul_view) ||
      !HasAtMostOneFanoutAtPort0(*matmul_view) ||
      IsInPreserveSet(ctx, matmul_def)) {
    return false;
  }
  if (HasDataType(matmul_def, DT_DOUBLE)) {
    return false;
  }

  const utils::MutableNodeView* mul_scale_view = nullptr;
  for (int p = 0; p < 2; ++p) {
    const auto* cand = mul_1_view->GetRegularFanin(p).node_view();
    if (cand != nullptr && cand->node_index() != bias_add_mut->node_index() &&
        (bias_cast_view == nullptr ||
         cand->node_index() != bias_cast_view->node_index())) {
      mul_scale_view = cand;
      break;
    }
  }
  if (mul_scale_view == nullptr) {
    return false;
  }
  const NodeDef* mul_scale_def = mul_scale_view->node();
  if (!IsAnyMul(*mul_scale_def) || HasControlFanin(*mul_scale_view) ||
      mul_scale_view->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, mul_scale_def)) {
    return false;
  }
  const auto& mul_scale_fanouts = mul_scale_view->GetRegularFanout(0);
  if (mul_scale_fanouts.size() != 2) {
    return false;
  }

  const utils::MutableNodeView* rsqrt_view = nullptr;
  const utils::MutableNodeView* gamma_branch = nullptr;
  for (int p = 0; p < 2; ++p) {
    const auto* cand = mul_scale_view->GetRegularFanin(p).node_view();
    if (cand == nullptr) {
      return false;
    }
    const NodeDef* cd = cand->node();
    if (cd != nullptr && IsRsqrt(*cd)) {
      rsqrt_view = cand;
    } else {
      gamma_branch = cand;
    }
  }
  if (rsqrt_view == nullptr || gamma_branch == nullptr) {
    return false;
  }
  const NodeDef* rsqrt_def = rsqrt_view->node();
  if (HasControlFanin(*rsqrt_view) || rsqrt_view->NumRegularFanins() != 1 ||
      IsInPreserveSet(ctx, rsqrt_def) ||
      !HasAtMostOneFanoutAtPort0(*rsqrt_view)) {
    return false;
  }

  utils::MutableNodeView* add_eps_mut =
      const_cast<utils::MutableNodeView*>(ctx.graph_view.GetNode(
          rsqrt_view->GetRegularFanin(0).node_view()->node_index()));
  if (add_eps_mut == nullptr) {
    return false;
  }
  add_eps_mut = SkipIdentityOnly(add_eps_mut);
  NodeDef* add_eps_def = add_eps_mut->node();
  if (add_eps_def == nullptr ||
      (add_eps_def->op() != kAddV2 && !IsAdd(*add_eps_def))) {
    return false;
  }
  if (HasControlFanin(*add_eps_mut) || add_eps_mut->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, add_eps_def) ||
      !HasAtMostOneFanoutAtPort0(*add_eps_mut)) {
    return false;
  }

  float epsilon = 1e-3f;
  int var_const_ix = -1;
  if (!ParseVarianceConstFromAddVariancePlusEps(ctx, add_eps_mut, 1e-3f,
                                                &epsilon, &var_const_ix)) {
    return false;
  }

  utils::MutableNodeView* gamma_mut = const_cast<utils::MutableNodeView*>(
      ctx.graph_view.GetNode(gamma_branch->node_index()));
  int gamma_ix = PeelToConstProducerNodeIndex(ctx, gamma_mut);
  if (gamma_ix < 0) {
    return false;
  }
  const NodeDef* gamma_def = ctx.graph_view.GetNode(gamma_ix)->node();
  Tensor Gamma;
  if (!GetTensorFromConstant(gamma_def, &Gamma).ok()) {
    return false;
  }

  const NodeDef* var_def = ctx.graph_view.GetNode(var_const_ix)->node();
  Tensor Variance;
  if (!GetTensorFromConstant(var_def, &Variance).ok()) {
    return false;
  }

  const utils::MutableNodeView* mul_mean_view = nullptr;
  for (int p = 0; p < 2; ++p) {
    const auto* cand = sub_view->GetRegularFanin(p).node_view();
    if (cand == nullptr) {
      return false;
    }
    if (IsAnyMul(*(cand->node()))) {
      mul_mean_view = cand;
      break;
    }
  }
  if (mul_mean_view == nullptr) {
    return false;
  }
  const NodeDef* mul_mean_def = mul_mean_view->node();
  if (!IsAnyMul(*mul_mean_def) || HasControlFanin(*mul_mean_view) ||
      mul_mean_view->NumRegularFanins() != 2 ||
      IsInPreserveSet(ctx, mul_mean_def)) {
    return false;
  }

  bool shares_scale = false;
  for (int p = 0; p < 2; ++p) {
    const auto* x = mul_mean_view->GetRegularFanin(p).node_view();
    if (x != nullptr && x->node_index() == mul_scale_view->node_index()) {
      shares_scale = true;
      break;
    }
  }
  if (!shares_scale) {
    return false;
  }

  utils::MutableNodeView* mean_mut = nullptr;
  for (int p = 0; p < 2; ++p) {
    const auto* x = mul_mean_view->GetRegularFanin(p).node_view();
    if (x != nullptr && x->node_index() != mul_scale_view->node_index()) {
      mean_mut = const_cast<utils::MutableNodeView*>(
          ctx.graph_view.GetNode(x->node_index()));
      break;
    }
  }
  if (mean_mut == nullptr) {
    return false;
  }
  mean_mut = SkipIdentityOnly(mean_mut);
  int mean_ix = PeelToConstProducerNodeIndex(ctx, mean_mut);
  if (mean_ix < 0) {
    return false;
  }
  Tensor Mean;
  if (!GetTensorFromConstant(ctx.graph_view.GetNode(mean_ix)->node(), &Mean)
           .ok()) {
    return false;
  }

  utils::MutableNodeView* beta_mut = nullptr;
  for (int p = 0; p < 2; ++p) {
    const auto* x = sub_view->GetRegularFanin(p).node_view();
    if (x != nullptr && x->node_index() != mul_mean_view->node_index()) {
      beta_mut = const_cast<utils::MutableNodeView*>(
          ctx.graph_view.GetNode(x->node_index()));
      break;
    }
  }
  if (beta_mut == nullptr) {
    return false;
  }
  beta_mut = SkipIdentityOnly(beta_mut);
  int beta_ix = PeelToConstProducerNodeIndex(ctx, beta_mut);
  if (beta_ix < 0) {
    return false;
  }
  Tensor Beta;
  if (!GetTensorFromConstant(ctx.graph_view.GetNode(beta_ix)->node(), &Beta)
           .ok()) {
    return false;
  }

  DataType t_mm = GetDataTypeFromAttr(*matmul_def, "T");
  DataType t_mul = GetDataTypeFromAttr(*mul_1_def, "T");
  if (bias_cast_view != nullptr) {
    DataType c_src = DT_INVALID, c_dst = DT_INVALID;
    if (!GetCastDataTypes(*(bias_cast_view->node()), &c_src, &c_dst)) {
      return false;
    }
    if (c_src != t_mm || c_dst != t_mul) {
      return false;
    }
    if (!HaveSameDataType(add_def, mul_1_def)) {
      return false;
    }
  } else {
    if (!HaveSameDataType(add_def, matmul_def)) {
      return false;
    }
    if (!HaveSameDataType(mul_1_def, matmul_def)) {
      return false;
    }
  }

  utils::MutableNodeView* w_mutable = const_cast<utils::MutableNodeView*>(
      ctx.graph_view.GetNode(ParseTensorName(matmul_def->input(1)).node()));
  utils::MutableNodeView* b_mutable =
      const_cast<utils::MutableNodeView*>(ctx.graph_view.GetNode(
          ParseTensorName(bias_add_def->input(bias_port)).node()));
  if (w_mutable == nullptr || b_mutable == nullptr) {
    return false;
  }
  int w_ix = PeelToConstProducerNodeIndex(ctx, w_mutable);
  int b_ix = PeelToConstProducerNodeIndex(ctx, b_mutable);
  if (w_ix < 0 || b_ix < 0) {
    return false;
  }
  if (IsConstAlreadyBnFolded(ctx.graph_view.GetNode(w_ix)->node()) ||
      IsConstAlreadyBnFolded(ctx.graph_view.GetNode(b_ix)->node())) {
    return false;
  }

  Tensor W;
  Tensor B;
  if (!GetTensorFromConstant(ctx.graph_view.GetNode(w_ix)->node(), &W).ok() ||
      !GetTensorFromConstant(ctx.graph_view.GetNode(b_ix)->node(), &B).ok()) {
    return false;
  }

  Tensor Sfloat;
  Tensor Hfloat;
  if (!BuildKerasBnScaleShiftTensors(Gamma, Beta, Mean, Variance, epsilon,
                                     &Sfloat, &Hfloat)) {
    return false;
  }
  if (!FuseMatmulBNfoldTensorShapesOk(W, B, Sfloat, Hfloat, t_mm)) {
    return false;
  }

  int tail_cast_ix = kMissingIndex;
  const auto& keras_add_fanouts = add_view->GetRegularFanout(0);
  if (keras_add_fanouts.size() == 1) {
    const auto* tail_view = keras_add_fanouts[0].node_view();
    const NodeDef* tail_def =
        tail_view != nullptr ? tail_view->node() : nullptr;
    if (tail_view != nullptr && tail_def != nullptr && IsCast(*tail_def) &&
        !HasControlFanin(*tail_view)) {
      DataType tc_src = DT_INVALID, tc_dst = DT_INVALID;
      if (GetCastDataTypes(*tail_def, &tc_src, &tc_dst) &&
          tc_src == GetDataTypeFromAttr(*add_def, "T") && tc_dst == t_mm) {
        tail_cast_ix = tail_view->node_index();
      }
    }
  }

  matched->matmul = matmul_view->node_index();
  matched->bias_add = bias_add_mut->node_index();
  matched->mul_1 = mul_1_view->node_index();
  matched->keras_bn_add = node_index;
  matched->sub = sub_view->node_index();
  matched->mul_scale = mul_scale_view->node_index();
  matched->mul_mean = mul_mean_view->node_index();
  matched->rsqrt = rsqrt_view->node_index();
  matched->add_eps = add_eps_mut->node_index();
  matched->gamma_const = gamma_ix;
  matched->beta_const = beta_ix;
  matched->mean_const = mean_ix;
  matched->variance_const = var_const_ix;
  matched->epsilon = epsilon;
  matched->bias_port = bias_port;
  matched->bias_cast =
      bias_cast_view != nullptr ? bias_cast_view->node_index() : kMissingIndex;
  matched->tail_cast = tail_cast_ix;
  return true;
}

Status AddMatMulBiasKerasBnFold(RemapperContext* ctx,
                                const MatMulBiasKerasBnFoldMatch& matched,
                                std::vector<bool>* invalidated_nodes,
                                std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& matmul = graph->node(matched.matmul);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& keras_add = graph->node(matched.keras_bn_add);

  if (matched.gamma_const < 0 || matched.beta_const < 0 ||
      matched.mean_const < 0 || matched.variance_const < 0) {
    return errors::Internal("MatMulBiasKerasBnFold: missing const indices");
  }

  Tensor Gamma, Beta, Mean, Variance;
  TF_RETURN_IF_ERROR(GetTensorFromConstant(
      ctx->graph_view.GetNode(matched.gamma_const)->node(), &Gamma));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(
      ctx->graph_view.GetNode(matched.beta_const)->node(), &Beta));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(
      ctx->graph_view.GetNode(matched.mean_const)->node(), &Mean));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(
      ctx->graph_view.GetNode(matched.variance_const)->node(), &Variance));

  utils::MutableNodeView* w_mutable =
      ctx->graph_view.GetNode(ParseTensorName(matmul.input(1)).node());
  utils::MutableNodeView* b_mutable = ctx->graph_view.GetNode(
      ParseTensorName(bias_add.input(matched.bias_port)).node());
  int w_ix = PeelToConstProducerNodeIndex(*ctx, w_mutable);
  int b_ix = PeelToConstProducerNodeIndex(*ctx, b_mutable);
  if (w_ix < 0 || b_ix < 0) {
    return errors::Internal("MatMulBiasKerasBnFold: weight/bias const peel");
  }
  auto* w_view = ctx->graph_view.GetNode(w_ix);
  auto* b_view = ctx->graph_view.GetNode(b_ix);
  NodeDef* w_def = w_view->node();
  NodeDef* b_def = b_view->node();

  Tensor W;
  Tensor B;
  TF_RETURN_IF_ERROR(GetTensorFromConstant(w_def, &W));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(b_def, &B));

  Tensor Sfloat;
  Tensor Hfloat;
  if (!BuildKerasBnScaleShiftTensors(Gamma, Beta, Mean, Variance,
                                     matched.epsilon, &Sfloat, &Hfloat)) {
    return errors::Internal("MatMulBiasKerasBnFold: scale/shift build failed");
  }

  DataType t = GetDataTypeFromAttr(matmul, "T");
  if (!FuseMatmulBNfoldTensorShapesOk(W, B, Sfloat, Hfloat, t)) {
    return errors::Internal("MatMulBiasKerasBnFold: tensor shapes");
  }
  ApplyFuseMatmulBNfoldToTensors(&W, &B, Sfloat, Hfloat, t);

  AttrValue w_attr;
  AttrValue b_attr;
  W.AsProtoTensorContent(w_attr.mutable_tensor());
  B.AsProtoTensorContent(b_attr.mutable_tensor());

  NodeDef fused_node;
  fused_node.set_op(kFusedMatMul);
  fused_node.set_name(keras_add.name());
  fused_node.set_device(matmul.device());
  fused_node.add_input(matmul.input(0));
  fused_node.add_input(matmul.input(1));
  fused_node.add_input(bias_add.input(matched.bias_port));
  CopyMatMulLikeAttributes(matmul, &fused_node);
  SetFusedOpAttributes(&fused_node, {"BiasAdd"});

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", matmul.op(), " (",
                                       matmul.name(), ") with Keras BN fold");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddOrUpdateNodeAttr(w_view, "value", w_attr);
  mutation->AddOrUpdateNodeAttr(b_view, "value", b_attr);
  // Tag the rewritten Consts so a later pass/recheck cannot fold them twice.
  MarkConstBnFolded(mutation, w_view);
  MarkConstBnFolded(mutation, b_view);
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  const string& fused_out_name =
      ctx->graph_view.graph()->node(matched.keras_bn_add).name();
  if (matched.tail_cast != kMissingIndex) {
    utils::MutableNodeView* tail_cast_view =
        ctx->graph_view.GetNode(matched.tail_cast);
    if (tail_cast_view == nullptr) {
      return errors::Internal("MatMulBiasKerasBnFold: tail Cast not found");
    }
    utils::Mutation* tail_mut = ctx->graph_view.GetMutationBuilder();
    SafeTensorId safe_fused_out(fused_out_name, 0);
    const TensorId fused_tid(safe_fused_out);
    for (const auto& fo_set : tail_cast_view->GetRegularFanouts()) {
      for (const auto& fo : fo_set) {
        tail_mut->AddOrUpdateRegularFanin(fo.node_view(), fo.index(),
                                          fused_tid);
      }
    }
    for (const auto& ctrl : tail_cast_view->GetControlledFanouts()) {
      tail_mut->RemoveControllingFanin(ctrl.node_view(),
                                       tail_cast_view->node()->name());
      tail_mut->AddControllingFanin(ctrl.node_view(), fused_out_name);
    }
    TF_RETURN_IF_ERROR(tail_mut->Apply());
    (*nodes_to_delete)[matched.tail_cast] = true;
  }

  (*invalidated_nodes)[matched.keras_bn_add] = true;
  (*nodes_to_delete)[matched.matmul] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.mul_1] = true;
  (*nodes_to_delete)[matched.sub] = true;
  (*nodes_to_delete)[matched.mul_scale] = true;
  if (matched.mul_mean != kMissingIndex) {
    (*nodes_to_delete)[matched.mul_mean] = true;
  }
  (*nodes_to_delete)[matched.rsqrt] = true;
  (*nodes_to_delete)[matched.add_eps] = true;
  if (matched.bias_cast != kMissingIndex) {
    (*nodes_to_delete)[matched.bias_cast] = true;
  }

  return OkStatus();
}

bool FindMatMulBiasMulAddFold(const RemapperContext& ctx, int node_index,
                              MatMulBiasMulAddFoldMatch* matched) {
  const auto* add_view = ctx.graph_view.GetNode(node_index);
  const auto* add_def = add_view->node();
  if (add_def == nullptr || (!IsAdd(*add_def) && add_def->op() != kAddV2)) {
    return false;
  }
  // Session fetches often mark the output Add/AddV2 in nodes_to_preserve; we
  // still fuse because the replacement _FusedMatMul keeps add.name() (same
  // tensor name).
  if (HasControlFanin(*add_view) || add_view->NumRegularFanins() != 2) {
    return false;
  }

  for (int add_mul_port = 0; add_mul_port < 2; ++add_mul_port) {
    const auto* mul_view = add_view->GetRegularFanin(add_mul_port).node_view();
    const auto* shift_view =
        add_view->GetRegularFanin(1 - add_mul_port).node_view();

    const utils::MutableNodeView *chain_view = nullptr, *scale_view = nullptr,
                                 *shift_const_view = nullptr;
    if (!MatchMulScaleBiasAddTail(ctx, mul_view, shift_view, &chain_view,
                                  &scale_view, &shift_const_view)) {
      continue;
    }

    const utils::MutableNodeView* bias_cast_view = nullptr;
    const utils::MutableNodeView* bias_add_view = chain_view;
    const NodeDef* chain_head_def = chain_view->node();
    if (chain_head_def != nullptr && chain_head_def->op() == kCast) {
      bias_cast_view = chain_view;
      if (HasControlFanin(*bias_cast_view) ||
          bias_cast_view->NumRegularFanins() < 1 ||
          IsInPreserveSet(ctx, chain_head_def) ||
          !HasAtMostOneFanoutAtPort0(*bias_cast_view)) {
        continue;
      }
      bias_add_view = bias_cast_view->GetRegularFanin(0).node_view();
    }
    const auto* bias_add_def = bias_add_view->node();
    int bias_port = 1;
    if (bias_add_view == nullptr || bias_add_def == nullptr ||
        (!IsBiasAdd(*bias_add_def) &&
         !IsBiasSemanticAdd(ctx, *bias_add_view, &bias_port))) {
      continue;
    }
    if (bias_add_view->NumRegularFanins() < 2) {
      continue;
    }
    const auto& contraction_fanin =
        bias_add_view->GetRegularFanin(1 - bias_port);
    const auto* matmul_view = contraction_fanin.node_view();
    const auto* matmul_def = matmul_view->node();
    if (matmul_view == nullptr || matmul_def == nullptr) {
      continue;
    }
    if (!IsMatMulZenOrBatchMatMul(*matmul_def)) {
      continue;
    }
    if (!MatMulLikeNoTransposeOrAdj(*matmul_def)) {
      continue;
    }
    if (!MatMulFoldRankOk(ctx, *matmul_def)) {
      continue;
    }
    if (!NodeIsOnCpu(matmul_def)) {
      continue;
    }
    if (HasControlFanin(*bias_add_view) ||
        bias_add_view->NumRegularFanins() != 2 ||
        IsInPreserveSet(ctx, bias_add_def)) {
      continue;
    }
    if (HasControlFanin(*matmul_view) ||
        !HasAtMostOneFanoutAtPort0(*matmul_view) ||
        IsInPreserveSet(ctx, matmul_def)) {
      continue;
    }
    if (!HasAtMostOneFanoutAtPort0(*bias_add_view)) {
      continue;
    }

    if (HasDataType(matmul_def, DT_DOUBLE)) {
      continue;
    }

    const auto* mul_def = mul_view->node();
    DataType t_mm = GetDataTypeFromAttr(*matmul_def, "T");
    DataType t_mul = GetDataTypeFromAttr(*mul_def, "T");
    if (bias_cast_view != nullptr) {
      DataType c_src = DT_INVALID, c_dst = DT_INVALID;
      if (!GetCastDataTypes(*(bias_cast_view->node()), &c_src, &c_dst)) {
        continue;
      }
      if (c_src != t_mm || c_dst != t_mul) {
        continue;
      }
      if (!HaveSameDataType(add_def, mul_def)) {
        continue;
      }
    } else {
      if (!HaveSameDataType(add_def, matmul_def)) {
        continue;
      }
      if (!HaveSameDataType(mul_def, matmul_def)) {
        continue;
      }
    }

    utils::MutableNodeView* w_mutable = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(ParseTensorName(matmul_def->input(1)).node()));
    utils::MutableNodeView* b_mutable =
        const_cast<utils::MutableNodeView*>(ctx.graph_view.GetNode(
            ParseTensorName(bias_add_def->input(bias_port)).node()));
    if (w_mutable == nullptr || b_mutable == nullptr) {
      continue;
    }
    int w_ix = PeelToConstProducerNodeIndex(ctx, w_mutable);
    int b_ix = PeelToConstProducerNodeIndex(ctx, b_mutable);
    if (w_ix < 0 || b_ix < 0) {
      continue;
    }
    auto* w_node_view = ctx.graph_view.GetNode(w_ix);
    auto* b_node_view = ctx.graph_view.GetNode(b_ix);
    const NodeDef* w_def = w_node_view->node();
    const NodeDef* b_def = b_node_view->node();
    if (IsConstAlreadyBnFolded(w_def) || IsConstAlreadyBnFolded(b_def)) {
      continue;
    }

    const NodeDef* scale_def = scale_view->node();
    const NodeDef* shift_def = shift_const_view->node();

    Tensor W;
    Tensor B;
    Tensor S;
    Tensor H;
    if (!GetTensorFromConstant(w_def, &W).ok() ||
        !GetTensorFromConstant(b_def, &B).ok() ||
        !GetTensorFromConstant(scale_def, &S).ok() ||
        !GetTensorFromConstant(shift_def, &H).ok()) {
      continue;
    }
    if (!FuseMatmulBNfoldTensorShapesOk(W, B, S, H, t_mm)) {
      continue;
    }

    int tail_cast_ix = kMissingIndex;
    const auto& add_fanouts = add_view->GetRegularFanout(0);
    if (add_fanouts.size() == 1) {
      const auto* tail_view = add_fanouts[0].node_view();
      const NodeDef* tail_def =
          tail_view != nullptr ? tail_view->node() : nullptr;
      if (tail_view != nullptr && tail_def != nullptr && IsCast(*tail_def) &&
          !HasControlFanin(*tail_view)) {
        DataType tc_src = DT_INVALID, tc_dst = DT_INVALID;
        if (GetCastDataTypes(*tail_def, &tc_src, &tc_dst) &&
            tc_src == GetDataTypeFromAttr(*add_def, "T") && tc_dst == t_mm) {
          tail_cast_ix = tail_view->node_index();
        }
      }
    }

    zendnnl::error_handling::apilog_info(
        "Remapper: MatMulBiasMulAddFold full pattern matched (MatMul-like ",
        matmul_def->op(), " ", matmul_def->name(), " -> BiasAdd -> Mul -> ",
        add_def->op(), " ", add_def->name(), ")");

    matched->matmul = matmul_view->node_index();
    matched->bias_add = bias_add_view->node_index();
    matched->mul = mul_view->node_index();
    matched->add = node_index;
    matched->scale_const = scale_view->node_index();
    matched->shift_const = shift_const_view->node_index();
    matched->bias_port = bias_port;
    matched->bias_cast = bias_cast_view != nullptr
                             ? bias_cast_view->node_index()
                             : kMissingIndex;
    matched->tail_cast = tail_cast_ix;
    return true;
  }
  return false;
}

Status AddMatMulBiasMulAddFold(RemapperContext* ctx,
                               const MatMulBiasMulAddFoldMatch& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& matmul = graph->node(matched.matmul);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  utils::MutableNodeView* w_mutable =
      ctx->graph_view.GetNode(ParseTensorName(matmul.input(1)).node());
  utils::MutableNodeView* b_mutable = ctx->graph_view.GetNode(
      ParseTensorName(bias_add.input(matched.bias_port)).node());
  if (w_mutable == nullptr || b_mutable == nullptr) {
    return errors::Internal("MatMulBiasMulAddFold: missing weight/bias inputs");
  }
  int w_ix = PeelToConstProducerNodeIndex(*ctx, w_mutable);
  int b_ix = PeelToConstProducerNodeIndex(*ctx, b_mutable);
  if (w_ix < 0 || b_ix < 0) {
    return errors::Internal(
        "MatMulBiasMulAddFold: weight/bias must resolve to Const");
  }
  auto* w_view = ctx->graph_view.GetNode(w_ix);
  auto* b_view = ctx->graph_view.GetNode(b_ix);
  NodeDef* w_def = w_view->node();
  NodeDef* b_def = b_view->node();
  const NodeDef& scale_def = graph->node(matched.scale_const);
  const NodeDef& shift_def = graph->node(matched.shift_const);

  Tensor W;
  Tensor B;
  Tensor S;
  Tensor H;
  TF_RETURN_IF_ERROR(GetTensorFromConstant(w_def, &W));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(b_def, &B));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(&scale_def, &S));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(&shift_def, &H));

  DataType t = GetDataTypeFromAttr(matmul, "T");
  if (!FuseMatmulBNfoldTensorShapesOk(W, B, S, H, t)) {
    return errors::Internal("MatMulBiasMulAddFold: invalid tensor shapes");
  }

  ApplyFuseMatmulBNfoldToTensors(&W, &B, S, H, t);

  AttrValue w_attr;
  AttrValue b_attr;
  W.AsProtoTensorContent(w_attr.mutable_tensor());
  B.AsProtoTensorContent(b_attr.mutable_tensor());

  NodeDef fused_node;
  fused_node.set_op(kFusedMatMul);
  fused_node.set_name(add.name());
  fused_node.set_device(matmul.device());
  fused_node.add_input(matmul.input(0));
  fused_node.add_input(matmul.input(1));
  fused_node.add_input(bias_add.input(matched.bias_port));
  CopyMatMulLikeAttributes(matmul, &fused_node);
  SetFusedOpAttributes(&fused_node, {"BiasAdd"});

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", matmul.op(), " (",
                                       matmul.name(), ") with Mul+Add fold");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddOrUpdateNodeAttr(w_view, "value", w_attr);
  mutation->AddOrUpdateNodeAttr(b_view, "value", b_attr);
  // Tag the rewritten Consts so a later pass/recheck cannot fold them twice.
  MarkConstBnFolded(mutation, w_view);
  MarkConstBnFolded(mutation, b_view);
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  const string& fused_out_name =
      ctx->graph_view.graph()->node(matched.add).name();
  if (matched.tail_cast != kMissingIndex) {
    utils::MutableNodeView* tail_cast_view =
        ctx->graph_view.GetNode(matched.tail_cast);
    if (tail_cast_view == nullptr) {
      return errors::Internal("MatMulBiasMulAddFold: tail Cast not found");
    }
    utils::Mutation* tail_mut = ctx->graph_view.GetMutationBuilder();
    SafeTensorId safe_fused_out(fused_out_name, 0);
    const TensorId fused_tid(safe_fused_out);
    for (const auto& fo_set : tail_cast_view->GetRegularFanouts()) {
      for (const auto& fo : fo_set) {
        tail_mut->AddOrUpdateRegularFanin(fo.node_view(), fo.index(),
                                          fused_tid);
      }
    }
    for (const auto& ctrl : tail_cast_view->GetControlledFanouts()) {
      tail_mut->RemoveControllingFanin(ctrl.node_view(),
                                       tail_cast_view->node()->name());
      tail_mut->AddControllingFanin(ctrl.node_view(), fused_out_name);
    }
    TF_RETURN_IF_ERROR(tail_mut->Apply());
    (*nodes_to_delete)[matched.tail_cast] = true;
  }

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.matmul] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.mul] = true;
  // Do not delete matched.add: Mutation::AddNode overwrote that NodeDef in
  // place (same name as Add/AddV2); removing it would drop the fetch tensor.
  (*nodes_to_delete)[matched.scale_const] = true;
  (*nodes_to_delete)[matched.shift_const] = true;
  if (matched.bias_cast != kMissingIndex) {
    (*nodes_to_delete)[matched.bias_cast] = true;
  }

  return OkStatus();
}

bool FindFuseMatmulBNfold(const RemapperContext& ctx, int node_index,
                          FuseMatmulBNfoldMatch* matched) {
  const auto* add_view = ctx.graph_view.GetNode(node_index);
  const auto* add_def = add_view->node();
  if (add_def == nullptr || (!IsAdd(*add_def) && add_def->op() != kAddV2)) {
    return false;
  }
  // Same as FindMatMulBiasMulAddFold: allow fetch roots in nodes_to_preserve.
  if (HasControlFanin(*add_view) || add_view->NumRegularFanins() != 2) {
    return false;
  }

  for (int add_mul_port = 0; add_mul_port < 2; ++add_mul_port) {
    const auto* mul_view = add_view->GetRegularFanin(add_mul_port).node_view();
    const auto* shift_view =
        add_view->GetRegularFanin(1 - add_mul_port).node_view();

    const utils::MutableNodeView *fused_view = nullptr, *scale_view = nullptr,
                                 *shift_const_view = nullptr;
    if (!MatchMulScaleBiasAddTail(ctx, mul_view, shift_view, &fused_view,
                                  &scale_view, &shift_const_view)) {
      continue;
    }

    const utils::MutableNodeView* bias_cast_view = nullptr;
    const utils::MutableNodeView* fused_inner_view = fused_view;
    const NodeDef* chain_head_def = fused_view->node();
    if (chain_head_def != nullptr && chain_head_def->op() == kCast) {
      bias_cast_view = fused_view;
      if (HasControlFanin(*bias_cast_view) ||
          bias_cast_view->NumRegularFanins() < 1 ||
          IsInPreserveSet(ctx, chain_head_def) ||
          !HasAtMostOneFanoutAtPort0(*bias_cast_view)) {
        continue;
      }
      fused_inner_view = bias_cast_view->GetRegularFanin(0).node_view();
    }
    const auto* fused_def =
        fused_inner_view != nullptr ? fused_inner_view->node() : nullptr;
    if (fused_def == nullptr || !IsFusedMatMulBiasAddOnly(*fused_def)) continue;
    if (!NodeIsOnCpu(fused_def)) continue;
    if (HasControlFanin(*fused_inner_view) ||
        !HasAtMostOneFanoutAtPort0(*fused_inner_view) ||
        IsInPreserveSet(ctx, fused_def)) {
      continue;
    }

    bool transpose_a = false;
    bool transpose_b = false;
    if (fused_def->attr().contains("transpose_a")) {
      transpose_a = fused_def->attr().at("transpose_a").b();
    }
    if (fused_def->attr().contains("transpose_b")) {
      transpose_b = fused_def->attr().at("transpose_b").b();
    }
    if (transpose_a || transpose_b) continue;
    if (HasDataType(fused_def, DT_DOUBLE)) continue;

    const auto* mul_def = mul_view->node();
    DataType t_mm = GetDataTypeFromAttr(*fused_def, "T");
    DataType t_mul = GetDataTypeFromAttr(*mul_def, "T");
    if (bias_cast_view != nullptr) {
      DataType c_src = DT_INVALID, c_dst = DT_INVALID;
      if (!GetCastDataTypes(*(bias_cast_view->node()), &c_src, &c_dst)) {
        continue;
      }
      if (c_src != t_mm || c_dst != t_mul) continue;
      if (!HaveSameDataType(add_def, mul_def)) continue;
    } else {
      if (!HaveSameDataType(add_def, fused_def)) continue;
      if (!HaveSameDataType(mul_def, fused_def)) continue;
    }

    utils::MutableNodeView* w_mutable = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(ParseTensorName(fused_def->input(1)).node()));
    utils::MutableNodeView* b_mutable = const_cast<utils::MutableNodeView*>(
        ctx.graph_view.GetNode(ParseTensorName(fused_def->input(2)).node()));
    if (w_mutable == nullptr || b_mutable == nullptr) continue;
    int w_ix = PeelToConstProducerNodeIndex(ctx, w_mutable);
    int b_ix = PeelToConstProducerNodeIndex(ctx, b_mutable);
    if (w_ix < 0 || b_ix < 0) continue;
    auto* w_node_view = ctx.graph_view.GetNode(w_ix);
    auto* b_node_view = ctx.graph_view.GetNode(b_ix);
    const NodeDef* w_def = w_node_view->node();
    const NodeDef* b_def = b_node_view->node();
    if (IsConstAlreadyBnFolded(w_def) || IsConstAlreadyBnFolded(b_def)) {
      continue;
    }

    const NodeDef* scale_def = scale_view->node();
    const NodeDef* shift_def = shift_const_view->node();

    Tensor W;
    Tensor B;
    Tensor S;
    Tensor H;
    if (!GetTensorFromConstant(w_def, &W).ok() ||
        !GetTensorFromConstant(b_def, &B).ok() ||
        !GetTensorFromConstant(scale_def, &S).ok() ||
        !GetTensorFromConstant(shift_def, &H).ok()) {
      continue;
    }
    if (!FuseMatmulBNfoldTensorShapesOk(W, B, S, H, t_mm)) continue;

    int tail_cast_ix = kMissingIndex;
    const auto& add_fanouts = add_view->GetRegularFanout(0);
    if (add_fanouts.size() == 1) {
      const auto* tail_view = add_fanouts[0].node_view();
      const NodeDef* tail_def =
          tail_view != nullptr ? tail_view->node() : nullptr;
      if (tail_view != nullptr && tail_def != nullptr && IsCast(*tail_def) &&
          !HasControlFanin(*tail_view)) {
        DataType tc_src = DT_INVALID, tc_dst = DT_INVALID;
        if (GetCastDataTypes(*tail_def, &tc_src, &tc_dst) &&
            tc_src == GetDataTypeFromAttr(*add_def, "T") && tc_dst == t_mm) {
          tail_cast_ix = tail_view->node_index();
        }
      }
    }

    matched->fused_matmul = fused_inner_view->node_index();
    matched->mul = mul_view->node_index();
    matched->add = node_index;
    matched->scale_const = scale_view->node_index();
    matched->shift_const = shift_const_view->node_index();
    matched->bias_cast = bias_cast_view != nullptr
                             ? bias_cast_view->node_index()
                             : kMissingIndex;
    matched->tail_cast = tail_cast_ix;
    return true;
  }
  return false;
}

Status AddFuseMatmulBNfold(RemapperContext* ctx,
                           const FuseMatmulBNfoldMatch& matched,
                           std::vector<bool>* invalidated_nodes,
                           std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused = graph->node(matched.fused_matmul);
  const NodeDef& add = graph->node(matched.add);

  utils::MutableNodeView* w_mutable =
      ctx->graph_view.GetNode(ParseTensorName(fused.input(1)).node());
  utils::MutableNodeView* b_mutable =
      ctx->graph_view.GetNode(ParseTensorName(fused.input(2)).node());
  if (w_mutable == nullptr || b_mutable == nullptr) {
    return errors::Internal("FuseMatmulBNfold: missing weight/bias inputs");
  }
  int w_ix = PeelToConstProducerNodeIndex(*ctx, w_mutable);
  int b_ix = PeelToConstProducerNodeIndex(*ctx, b_mutable);
  if (w_ix < 0 || b_ix < 0) {
    return errors::Internal(
        "FuseMatmulBNfold: weight/bias must resolve to Const");
  }
  auto* w_view = ctx->graph_view.GetNode(w_ix);
  auto* b_view = ctx->graph_view.GetNode(b_ix);
  NodeDef* w_def = w_view->node();
  NodeDef* b_def = b_view->node();
  const NodeDef& scale_def = graph->node(matched.scale_const);
  const NodeDef& shift_def = graph->node(matched.shift_const);

  Tensor W;
  Tensor B;
  Tensor S;
  Tensor H;
  TF_RETURN_IF_ERROR(GetTensorFromConstant(w_def, &W));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(b_def, &B));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(&scale_def, &S));
  TF_RETURN_IF_ERROR(GetTensorFromConstant(&shift_def, &H));

  DataType t = GetDataTypeFromAttr(fused, "T");
  if (!FuseMatmulBNfoldTensorShapesOk(W, B, S, H, t)) {
    return errors::Internal("FuseMatmulBNfold: invalid tensor shapes");
  }

  ApplyFuseMatmulBNfoldToTensors(&W, &B, S, H, t);

  AttrValue w_attr;
  AttrValue b_attr;
  W.AsProtoTensorContent(w_attr.mutable_tensor());
  B.AsProtoTensorContent(b_attr.mutable_tensor());

  NodeDef new_node;
  new_node.set_op(fused.op());
  new_node.set_name(add.name());
  new_node.set_device(fused.device());
  CopyAllAttrs(fused, &new_node);
  for (int i = 0; i < fused.input_size(); ++i) {
    new_node.add_input(fused.input(i));
  }

  zendnnl::error_handling::apilog_info("Remapper: Fusing ", fused.op(), " (",
                                       fused.name(), ") with FuseMatmulBNfold");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddOrUpdateNodeAttr(w_view, "value", w_attr);
  mutation->AddOrUpdateNodeAttr(b_view, "value", b_attr);
  // Tag the rewritten Consts so a later pass/recheck cannot fold them twice.
  MarkConstBnFolded(mutation, w_view);
  MarkConstBnFolded(mutation, b_view);
  mutation->AddNode(std::move(new_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  const string& fused_out_name =
      ctx->graph_view.graph()->node(matched.add).name();
  if (matched.tail_cast != kMissingIndex) {
    utils::MutableNodeView* tail_cast_view =
        ctx->graph_view.GetNode(matched.tail_cast);
    if (tail_cast_view == nullptr) {
      return errors::Internal("FuseMatmulBNfold: tail Cast not found");
    }
    utils::Mutation* tail_mut = ctx->graph_view.GetMutationBuilder();
    SafeTensorId safe_fused_out(fused_out_name, 0);
    const TensorId fused_tid(safe_fused_out);
    for (const auto& fo_set : tail_cast_view->GetRegularFanouts()) {
      for (const auto& fo : fo_set) {
        tail_mut->AddOrUpdateRegularFanin(fo.node_view(), fo.index(),
                                          fused_tid);
      }
    }
    for (const auto& ctrl : tail_cast_view->GetControlledFanouts()) {
      tail_mut->RemoveControllingFanin(ctrl.node_view(),
                                       tail_cast_view->node()->name());
      tail_mut->AddControllingFanin(ctrl.node_view(), fused_out_name);
    }
    TF_RETURN_IF_ERROR(tail_mut->Apply());
    (*nodes_to_delete)[matched.tail_cast] = true;
  }

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.fused_matmul] = true;
  (*nodes_to_delete)[matched.mul] = true;
  // Same as AddMatMulBiasMulAddFold: new_node uses add.name() and overwrites
  // Add.
  (*nodes_to_delete)[matched.scale_const] = true;
  (*nodes_to_delete)[matched.shift_const] = true;
  if (matched.bias_cast != kMissingIndex) {
    (*nodes_to_delete)[matched.bias_cast] = true;
  }

  return OkStatus();
}
//------------------------------------------------------------------------------
// Expanded LayerNorm fusion.
//------------------------------------------------------------------------------

int ResolveNodeDataSourceIndex(utils::MutableNodeView* v) {
  for (int depth = 0; depth < 16 && v != nullptr; ++depth) {
    const string& op = v->node()->op();
    if ((op == "Identity" || op == "StopGradient" || op == kCast) &&
        v->NumRegularFanins() >= 1) {
      v = v->GetRegularFanin(0).node_view();
      continue;
    }
    break;
  }
  return v != nullptr ? v->node_index() : -1;
}

bool GetScalarFloatFromConstNode(const NodeDef* node_def, float* out) {
  if (node_def == nullptr || node_def->op() != "Const") return false;
  Tensor t;
  if (!GetTensorFromConstant(node_def, &t).ok() || t.NumElements() != 1) {
    return false;
  }
  if (t.dtype() == DT_FLOAT) {
    *out = t.flat<float>()(0);
    return true;
  }
  if (t.dtype() == DT_BFLOAT16) {
    *out = static_cast<float>(t.flat<Eigen::bfloat16>()(0));
    return true;
  }
  return false;
}

bool ExtractEpsilonFromAddEps(RemapperContext* ctx, int add_eps_idx,
                              float* eps) {
  auto* add_view = ctx->graph_view.GetNode(add_eps_idx);
  if (add_view->NumRegularFanins() < 2) return false;
  for (int i = 0; i < 2; ++i) {
    auto* fin = add_view->GetRegularFanin(i).node_view();
    if (IsMean(*fin->node())) continue;
    int src = ResolveNodeDataSourceIndex(fin);
    if (src >= 0 && GetScalarFloatFromConstNode(
                        ctx->graph_view.GetNode(src)->node(), eps)) {
      return true;
    }
  }
  return false;
}

bool GetMeanReductionAxes(RemapperContext* ctx, int mean_idx,
                          std::vector<int32>* axes) {
  auto* mean_view = ctx->graph_view.GetNode(mean_idx);
  if (mean_view->NumRegularFanins() < 2) return false;
  int axes_idx =
      ResolveNodeDataSourceIndex(mean_view->GetRegularFanin(1).node_view());
  if (axes_idx < 0) return false;
  const NodeDef* axes_node = ctx->graph_view.GetNode(axes_idx)->node();
  if (axes_node->op() != "Const") return false;
  Tensor t;
  if (!GetTensorFromConstant(axes_node, &t).ok()) return false;
  axes->clear();
  if (t.dtype() == DT_INT32) {
    auto e = t.flat<int32>();
    for (int i = 0; i < e.size(); ++i) axes->push_back(e(i));
  } else if (t.dtype() == DT_INT64) {
    auto e = t.flat<int64_t>();
    for (int i = 0; i < e.size(); ++i) {
      axes->push_back(static_cast<int32>(e(i)));
    }
  } else {
    return false;
  }
  return !axes->empty();
}

utils::OpTypePattern MakeLayerNormScaleSubtree() {
  using utils::NodeStatus;
  utils::OpTypePattern mean_mu = {"Mean", "mean_mu", NodeStatus::kRemove, {}};
  mean_mu.AddInput({"*", "input", NodeStatus::kRemain});
  mean_mu.AddInput({"*", "axes_mu", NodeStatus::kRemain});

  utils::OpTypePattern sqdiff = {
      "SquaredDifference", "sqdiff", NodeStatus::kRemove, {}};
  sqdiff.AddInput({"*", "input", NodeStatus::kRemain});
  sqdiff.AddInput(mean_mu);

  utils::OpTypePattern mean_var = {"Mean", "mean_var", NodeStatus::kRemove, {}};
  mean_var.AddInput(sqdiff);
  mean_var.AddInput({"*", "axes_var", NodeStatus::kRemain});

  utils::OpTypePattern add_eps = {
      "AddV2|Add", "add_eps", NodeStatus::kRemove, {}};
  add_eps.AddInput(mean_var);
  add_eps.AddInput({"*", "epsilon", NodeStatus::kRemain});

  utils::OpTypePattern rsqrt = {"Rsqrt", "rsqrt", NodeStatus::kRemove, {}};
  rsqrt.AddInput(add_eps);

  utils::OpTypePattern scale = {"Mul", "scale", NodeStatus::kRemove, {}};
  scale.AddInput({"*", "gamma", NodeStatus::kRemain});
  scale.AddInput(rsqrt);
  return scale;
}

std::pair<utils::OpTypePattern, utils::OpTypePattern>
MakeLayerNormSubMulBranches(const utils::OpTypePattern& scale) {
  using utils::NodeStatus;
  utils::OpTypePattern mul_x = {"Mul", "mul_x", NodeStatus::kRemove, {}};
  mul_x.AddInput({"*", "input", NodeStatus::kRemain});
  mul_x.AddInput(scale);

  utils::OpTypePattern mul_mu = {"Mul", "mul_mu", NodeStatus::kRemove, {}};
  utils::OpTypePattern mean_mu = {"Mean", "mean_mu", NodeStatus::kRemove, {}};
  mean_mu.AddInput({"*", "input", NodeStatus::kRemain});
  mean_mu.AddInput({"*", "axes_mu", NodeStatus::kRemain});
  mul_mu.AddInput(mean_mu);
  mul_mu.AddInput(scale);

  utils::OpTypePattern sub = {"Sub", "sub", NodeStatus::kRemove, {}};
  sub.AddInput({"*", "beta", NodeStatus::kRemain});
  sub.AddInput(mul_mu);
  return {sub, mul_x};
}

bool FindFusedLayerNorm(RemapperContext* ctx, int node_index,
                        std::map<string, int>* matched_nodes_map,
                        std::set<int>* remove_node_indices,
                        bool* matched_residual_out) {
  using utils::MatchingDirection;
  using utils::NodeStatus;

  const auto validate_matched = [&](RemapperContext* c,
                                    const std::map<string, int>& matched,
                                    bool with_residual_addn) -> bool {
    auto* sub_view = c->graph_view.GetNode(matched.at("sub"));
    const int ds_sub0 =
        ResolveNodeDataSourceIndex(sub_view->GetRegularFanin(0).node_view());
    const int ds_sub1 =
        ResolveNodeDataSourceIndex(sub_view->GetRegularFanin(1).node_view());
    const int ds_beta =
        ResolveNodeDataSourceIndex(c->graph_view.GetNode(matched.at("beta")));
    const int ds_mul_mu =
        ResolveNodeDataSourceIndex(c->graph_view.GetNode(matched.at("mul_mu")));
    if (ds_sub0 != ds_beta || ds_sub1 != ds_mul_mu) return false;

    std::vector<int32> axes_mu, axes_var;
    if (!GetMeanReductionAxes(c, matched.at("mean_mu"), &axes_mu) ||
        !GetMeanReductionAxes(c, matched.at("mean_var"), &axes_var) ||
        axes_mu != axes_var) {
      return false;
    }

    float eps = 0.f;
    if (!ExtractEpsilonFromAddEps(c, matched.at("add_eps"), &eps) ||
        eps <= 0.f) {
      return false;
    }

    const NodeDef* input_node =
        c->graph_view.GetNode(matched.at("input"))->node();
    DataType input_type = GetDataTypeFromAttr(*input_node, "T");
    if (input_type == DT_INVALID) {
      input_type = GetDataTypeFromAttr(*input_node, "dtype");
    }
    if (input_type != DT_FLOAT && input_type != DT_BFLOAT16) return false;

    if (!with_residual_addn) return true;

    const int residual_ds = ResolveNodeDataSourceIndex(
        c->graph_view.GetNode(matched.at("residual_in")));
    auto* out_view = c->graph_view.GetNode(matched.at("output"));
    int residual_fanin_matches = 0;
    for (int p = 0; p < 3; ++p) {
      if (ResolveNodeDataSourceIndex(
              out_view->GetRegularFanin(p).node_view()) == residual_ds) {
        residual_fanin_matches++;
      }
    }
    return residual_fanin_matches == 1;
  };

  utils::MutableNodeView* node_view = ctx->graph_view.GetNode(node_index);
  const NodeDef* node_def = node_view->node();
  if (HasControlFaninOrFanout(*node_view)) return false;

  const utils::OpTypePattern scale_subtree = MakeLayerNormScaleSubtree();
  const auto [sub_branch, mul_x_branch] =
      MakeLayerNormSubMulBranches(scale_subtree);

  const std::vector<utils::OpTypePattern> two_input_patterns = {
      {"AddN|AddV2|Add",
       "output",
       NodeStatus::kReplace,
       {sub_branch, mul_x_branch}},
      {"AddN|AddV2|Add",
       "output",
       NodeStatus::kReplace,
       {mul_x_branch, sub_branch}},
  };

  std::vector<utils::OpTypePattern> residual_patterns;
  residual_patterns.reserve(6);
  static constexpr std::array<std::array<int, 3>, 6> kFaninOrders = {{
      {0, 1, 2},
      {0, 2, 1},
      {1, 0, 2},
      {1, 2, 0},
      {2, 0, 1},
      {2, 1, 0},
  }};
  utils::OpTypePattern residual_branch{
      "*", "residual_in", NodeStatus::kRemain, {}};
  for (const auto& ord : kFaninOrders) {
    auto branches = MakeLayerNormSubMulBranches(scale_subtree);
    const utils::OpTypePattern* fanins[3] = {&branches.first, &branches.second,
                                             &residual_branch};
    residual_patterns.push_back(
        {"AddN",
         "output",
         NodeStatus::kReplace,
         {*fanins[ord[0]], *fanins[ord[1]], *fanins[ord[2]]}});
  }

  const auto try_patterns =
      [&](const std::vector<utils::OpTypePattern>& patterns,
          bool with_residual) -> bool {
    utils::SubGraphMatcher<MatchingDirection::kFollowInputs> matcher(
        &(ctx->graph_view));
    for (const auto& pattern : patterns) {
      matched_nodes_map->clear();
      remove_node_indices->clear();
      if (matcher.GetMatchedNodes(pattern, ctx->nodes_to_preserve, node_view,
                                  matched_nodes_map, remove_node_indices,
                                  false) &&
          validate_matched(ctx, *matched_nodes_map, with_residual)) {
        if (matched_residual_out != nullptr) {
          *matched_residual_out = with_residual;
        }
        return true;
      }
    }
    return false;
  };

  if (IsAddN(*node_def) && node_view->NumRegularFanins() == 3) {
    return try_patterns(residual_patterns, true);
  }
  if ((!IsAddN(*node_def) && node_def->op() != kAddV2 &&
       !IsAddWithNoBroadcast(*ctx, *node_def)) ||
      node_view->NumRegularFanins() != 2) {
    return false;
  }
  return try_patterns(two_input_patterns, false);
}

Status AddFusedLayerNorm(RemapperContext* ctx,
                         const std::map<string, int>& matched_nodes_map,
                         const std::set<int>& remove_node_indices,
                         std::vector<bool>* invalidated_nodes,
                         std::vector<bool>* nodes_to_delete,
                         bool with_residual) {
  auto* output_view = ctx->graph_view.GetNode(matched_nodes_map.at("output"));
  const NodeDef* output_node = output_view->node();
  auto* mean_mu_view = ctx->graph_view.GetNode(matched_nodes_map.at("mean_mu"));
  auto* scale_view = ctx->graph_view.GetNode(matched_nodes_map.at("scale"));
  auto* sub_view = ctx->graph_view.GetNode(matched_nodes_map.at("sub"));

  int res_port = -1;
  if (with_residual) {
    const int residual_ds = ResolveNodeDataSourceIndex(
        ctx->graph_view.GetNode(matched_nodes_map.at("residual_in")));
    for (int p = 0; p < 3; ++p) {
      if (ResolveNodeDataSourceIndex(
              output_view->GetRegularFanin(p).node_view()) == residual_ds) {
        res_port = p;
        break;
      }
    }
    if (res_port < 0) {
      return errors::Internal("FusedLayerNorm: residual fanin");
    }
  }

  std::vector<int32> axes;
  GetMeanReductionAxes(ctx, matched_nodes_map.at("mean_mu"), &axes);
  float epsilon = 0.f;
  ExtractEpsilonFromAddEps(ctx, matched_nodes_map.at("add_eps"), &epsilon);
  const DataType t = GetDataTypeFromAttr(*mean_mu_view->node(), "T");

  const string fused_name = with_residual
                                ? output_node->name() + "/FusedLayerNorm"
                                : output_node->name();

  NodeDef fused_ln;
  fused_ln.set_name(fused_name);
  fused_ln.set_op(kFusedLayerNorm);
  fused_ln.set_device(mean_mu_view->node()->device());
  fused_ln.add_input(mean_mu_view->node()->input(0));

  int gamma_in_port =
      scale_view->GetRegularFanin(0).node_view()->node()->op() == "Rsqrt" ? 1
                                                                          : 0;
  fused_ln.add_input(scale_view->node()->input(gamma_in_port));

  const int ds_beta = ResolveNodeDataSourceIndex(
      ctx->graph_view.GetNode(matched_nodes_map.at("beta")));
  const int ds_mul_mu = ResolveNodeDataSourceIndex(
      ctx->graph_view.GetNode(matched_nodes_map.at("mul_mu")));
  const int ds_s0 =
      ResolveNodeDataSourceIndex(sub_view->GetRegularFanin(0).node_view());
  const int ds_s1 =
      ResolveNodeDataSourceIndex(sub_view->GetRegularFanin(1).node_view());
  const int beta_in_port =
      (ds_s0 == ds_beta && ds_s1 == ds_mul_mu)
          ? 0
          : (ds_s0 == ds_mul_mu && ds_s1 == ds_beta ? 1 : -1);
  if (beta_in_port < 0) {
    return errors::Internal("FusedLayerNorm: beta port");
  }
  fused_ln.add_input(sub_view->node()->input(beta_in_port));

  AddNodeAttr("T", t, &fused_ln);
  AddNodeAttr("epsilon", epsilon, &fused_ln);
  AddNodeAttr("axes", axes, &fused_ln);

  zendnnl::error_handling::apilog_info(
      "Remapper: Fusing LayerNorm expanded subgraph (", output_node->name(),
      ") into ", kFusedLayerNorm, with_residual ? " + AddV2" : "");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_ln), &status);
  TF_RETURN_IF_ERROR(status);
  if (with_residual) {
    NodeDef add_res;
    add_res.set_name(output_node->name());
    add_res.set_op(kAddV2);
    add_res.set_device(output_node->device());
    add_res.add_input(fused_name);
    add_res.add_input(output_node->input(res_port));
    AddNodeAttr("T", t, &add_res);
    mutation->AddNode(std::move(add_res), &status);
    TF_RETURN_IF_ERROR(status);
  }

  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;
  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return OkStatus();
}

}  // namespace

Status RunRemapper(const char* device_name, const GrapplerItem& item,
                   const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  RemapperContext ctx(item, &multable_graph_def, &status,
                      /*level*/ RemapperLevel::BASIC);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  const int num_nodes = multable_graph_def.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  zendnnl::error_handling::apilog_info("Remapper: Running optimization on ",
                                       num_nodes, " nodes");

  // Infer statically first and only once.
  ctx.GetGraphProperties();

  bool is_visited = false;
  string last_op;
  for (int i = num_nodes - 1; i >= 0;) {
    NodeDef* node_def = (ctx.graph_view.GetNode(i))->node();

    const string& type_attr = "T";
    if (HasNodeAttr(*node_def, type_attr)) {
      const auto& attr = node_def->attr().at(type_attr);
      DataType dtype = attr.type();
      if ((dtype == DT_BFLOAT16) &&
          !tensorflow::port::TestCPUFeature(
              tensorflow::port::CPUFeature::AVX512F)) {
        return errors::FailedPrecondition(
            "Platform does not support AVX512 instruction set for BF16 Op!");
      }
    }

    // IMPORTANT: Always keep this dynamic check in the start.
    // Dynamic check node status:
    //   1. Do normal fusion check when current node is visited first time.
    //   2. Recheck this node only if it's new fused and never rechecked before.
    //   3. Iterate to next node after current node is visited and not fused, or
    //      already rechecked.
    if (is_visited) {
      if (invalidated_nodes[i] && last_op != node_def->op()) {
        // Recheck current node to find more possible fusion.
        zendnnl::error_handling::apilog_info("Remapper: Rechecking node ",
                                             node_def->name(), " for fusion");
        last_op = node_def->op();
      } else {
        // Iterate to next node and reset all flags.
        --i;
        is_visited = false;
        last_op = node_def->op();
        continue;
      }
    } else {
      last_op = node_def->op();
      is_visited = true;
    }

    // Check if node was deleted by one of the previous remaps.
    if (nodes_to_delete[i]) {
      continue;
    }
    zendnnl::error_handling::apilog_info("Remapper: Processing node ",
                                         node_def->name(), " (", node_def->op(),
                                         ")");
    {
      std::map<string, int> matched_nodes_map;
      std::set<int> remove_node_indices;

      // Horizontal fusion: GatherV2 x N -> [SafeCast] -> ConcatV2 =>
      // _ZenGroupEmbedding.
      GroupEmbedding group_embedding;
      if (FindGroupEmbedding(ctx, i, &group_embedding)) {
        TF_ABORT_IF_ERROR(AddGroupEmbeddingNode(
            &ctx, group_embedding, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Fuse safe_embedding_lookup_sparse subgraph.
      SafeEmbeddingLookupSparse safe_emb;
      if (FindSafeEmbeddingLookupSparse(ctx, i, &safe_emb)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found SafeEmbeddingLookupSparse (", safe_emb.combiner,
            ")");
        TF_ABORT_IF_ERROR(AddFusedSafeEmbeddingLookupSparse(
            &ctx, safe_emb, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Keras Dense layer fwd fusion.
      KerasDenseLayerFwd keras_dense_layer_fwd;
      if (FindKerasDenseLayerFwd(ctx, i, &keras_dense_layer_fwd)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found Keras Dense layer forward fusion");
        TF_ABORT_IF_ERROR(AddKerasDenseLayerFwd(
            &ctx, keras_dense_layer_fwd, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add+Activation into the _FusedConv2D.
      ContractionWithBiasAndAddActivation contract_with_bias_and_add_activation;
      if (FindContractionWithBiasAndAddActivation(
              ctx, i, &contract_with_bias_and_add_activation)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithBiasAndAddActivation");
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap MatMul+Relu into the _FusedMatMul.
      ContractionWithActivation contract_with_activation;
      if (FindContractionWithActivation(ctx, i, &contract_with_activation)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithActivation");
        AddFusedContractionNode(&ctx, contract_with_activation,
                                &invalidated_nodes, &nodes_to_delete);
        continue;
      }

      // Remap _FusedMatMul{MatMul + BiasAdd} + Sigmoid into the _FusedMatMul.
      ContractionWithActivation contract_with_sigmoid;
      if (FindContractionWithSigmoid(ctx, i, &contract_with_sigmoid)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithSigmoid");
        AddFusedMatMulSigmoidNode(&ctx, contract_with_sigmoid,
                                  &invalidated_nodes, &nodes_to_delete);
        continue;
      }

      // MatMul+BiasAdd + Keras decomposed inference BN -> _FusedMatMul.
      MatMulBiasKerasBnFoldMatch mm_bias_keras_bn;
      if (FindMatMulBiasKerasBnFold(ctx, i, &mm_bias_keras_bn)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found MatMulBiasKerasBnFold");
        TF_RETURN_IF_ERROR(AddMatMulBiasKerasBnFold(
            &ctx, mm_bias_keras_bn, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // MatMul+BiasAdd+Mul+Add|AddV2 -> _FusedMatMul (scale/shift folded into
      // Consts).
      MatMulBiasMulAddFoldMatch mm_bias_mul_add;
      if (FindMatMulBiasMulAddFold(ctx, i, &mm_bias_mul_add)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found MatMulBiasMulAddFold");
        TF_RETURN_IF_ERROR(AddMatMulBiasMulAddFold(
            &ctx, mm_bias_mul_add, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // _FusedMatMul + Mul + Add (after MatMul+BiasAdd -> _FusedMatMul, same
      // pass).
      FuseMatmulBNfoldMatch fuse_matmul_bn;
      if (FindFuseMatmulBNfold(ctx, i, &fuse_matmul_bn)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found FuseMatmulBNfold");
        TF_RETURN_IF_ERROR(AddFuseMatmulBNfold(
            &ctx, fuse_matmul_bn, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap _FusedMatMul{MatMul + BiasAdd} + Mish into _FusedMatMul (after BN
      // fold patterns so BN can rewrite weights/bias on BiasAdd-only fusion).
      ContractionWithActivation contract_with_mish;
      if (FindContractionWithMish(ctx, i, &contract_with_mish)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithMish");
        TF_RETURN_IF_ERROR(AddFusedMatMulMishNode(
            &ctx, contract_with_mish, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // _FusedMatMul -> Softplus -> Tanh -> Mul(x, tanh(...))  (decomposed
      // Mish).
      FusedMatMulMishDecomposedMatch mish_decomposed;
      if (FindFusedMatMulMishDecomposedPattern(ctx, i, &mish_decomposed)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found FusedMatMulMishDecomposed");
        TF_RETURN_IF_ERROR(AddFusedMatMulMishDecomposed(
            &ctx, mish_decomposed, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add into the _FusedConv2D.
      ContractionWithBiasAddAndAdd contract_with_bias_and_add;
      if (FindContractionWithBiasAddAndAdd(ctx, i,
                                           &contract_with_bias_and_add)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithBiasAddAndAdd");
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd into the
      // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}.
      ContractionWithBiasAdd contract_with_bias;
      if (FindContractionWithBias(ctx, i, &contract_with_bias)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithBias");
        TF_ABORT_IF_ERROR(AddFusedContractionNode(
            &ctx, contract_with_bias, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd+Activation into
      // the _Fused{Conv2D,DepthwiseConv2dNative,MatMul}.
      ContractionWithBiasAddAndActivation contract_with_bias_and_activation;
      if (FindContractionWithBiasAndActivation(
              ctx, i, &contract_with_bias_and_activation)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithBiasAndActivation");
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // TODO(plugin): Add the support for _ZenFusedBatchNormEx once we get
      // support from ZenDNN library. Remap FusedBatchNorm+<Activation> into
      // _FusedBatchNormEx.
      // FusedBatchNormEx fused_batch_norm_ex;
      // if (FindFusedBatchNormEx(ctx, i, &fused_batch_norm_ex)) {
      //   // Old ZenDNN logging removed;
      //   TF_ABORT_IF_ERROR(AddFusedBatchNormExNode(
      //       &ctx, fused_batch_norm_ex, &invalidated_nodes,
      //       &nodes_to_delete));
      // }

      // Remap BatchMatMul + Mul into the _FusedBatchMatMul.
      ContractionWithMul contract_with_mul;
      if (FindContractionWithMul(ctx, i, &contract_with_mul)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found ContractionWithMul");
        AddFusedContractionNode(&ctx, contract_with_mul, &invalidated_nodes,
                                &nodes_to_delete);
        continue;
      }

      // Remap MatMul + BiasAdd + gelu-subgraph.
      matched_nodes_map.clear();
      remove_node_indices.clear();
      bool is_gelu_approximate = false;
      bool expand_dims = false;
      if (FindMatMulBiasAddAndGelu(&ctx, i, &matched_nodes_map,
                                   &remove_node_indices, &is_gelu_approximate,
                                   &expand_dims)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found MatMulBiasAddAndGelu");
        TF_ABORT_IF_ERROR(AddFusedMatMulBiasAddAndGelu(
            &ctx, matched_nodes_map, remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, is_gelu_approximate, expand_dims));
        continue;
      }

      // Remap expanded LayerNorm (+ optional residual AddN) into
      // _FusedLayerNorm [+ AddV2].
      matched_nodes_map.clear();
      remove_node_indices.clear();
      bool fused_ln_residual = false;
      if (FindFusedLayerNorm(&ctx, i, &matched_nodes_map, &remove_node_indices,
                             &fused_ln_residual)) {
        zendnnl::error_handling::apilog_info(
            fused_ln_residual ? "Remapper: Found FusedLayerNormResidual"
                              : "Remapper: Found FusedLayerNorm");
        TF_RETURN_IF_ERROR(AddFusedLayerNorm(
            &ctx, matched_nodes_map, remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, fused_ln_residual));
        continue;
      }

      // Remap BatchMatMul + Mul + AddV2 into the _FusedBatchMatMul.
      matched_nodes_map.clear();
      remove_node_indices.clear();
      std::vector<string> input_node_names;
      input_node_names.clear();
      if (FindFusedBatchMatMul(&ctx, i, &matched_nodes_map,
                               &remove_node_indices, &input_node_names)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found FusedBatchMatMul");
        TF_RETURN_IF_ERROR(AddFusedBatchMatMul(
            &ctx, matched_nodes_map, remove_node_indices, input_node_names,
            &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap BatchMatMul + BiasAdd + Activation into the _FusedBatchMatMul.
      BatchMatMulWithBiasAddAndActivation batchmatmul_biasadd_activation;
      if (FindBatchMatMulBiasAddActivation(ctx, i,
                                           &batchmatmul_biasadd_activation)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found BatchMatMulBiasAddActivation");
        TF_RETURN_IF_ERROR(AddFusedBatchMatMulBiasAddActivation(
            &ctx, batchmatmul_biasadd_activation, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap Pad + (_Fused)Conv2D to (_Fused)Conv2D.
      PadWithContraction pad_conv;
      if (FindPadWithContraction(ctx, i, &pad_conv)) {
        zendnnl::error_handling::apilog_info(
            "Remapper: Found PadWithContraction");
        TF_ABORT_IF_ERROR(AddPadWithContractionNode(
            &ctx, pad_conv, &invalidated_nodes, &nodes_to_delete));
        continue;
      }
    }
  }

  // Remove invalidated nodes.
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx.graph_view.GetNode(i));
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());

  *optimized_graph = std::move(multable_graph_def);
  return OkStatus();
}

}  // namespace graph
}  // namespace amd_cpu_plugin
