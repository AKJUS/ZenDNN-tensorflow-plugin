/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright (c) 2021-2022 Intel Corporation

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

#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_common_utils.h"

#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {
namespace graph {

bool Is1D(const TensorShapeProto& proto) {
  // Note: Returns false when rank is unknown.
  if (proto.unknown_rank()) {
    return false;
  }
  // Returns false when dimension is unknown.
  for (const auto& dim : proto.dim()) {
    if (dim.size() < 0) {
      return false;
    }
  }

  auto tensor_shape = TensorShape(proto);

  for (int i = 0; i < tensor_shape.dims(); ++i) {
    zendnnVerbose(ZENDNN_FWKLOG, tensor_shape.dim_size(i));
  }
  return (tensor_shape.dims() == 1);
}

bool Is2D(const TensorShapeProto& proto) {
  // Note: Returns false when rank is unknown.
  if (proto.unknown_rank()) {
    return false;
  }
  // Returns false when dimension is unknown.
  for (const auto& dim : proto.dim()) {
    if (dim.size() < 0) {
      return false;
    }
  }

  auto tensor_shape = TensorShape(proto);
  return (tensor_shape.dims() == 2);
}

// Returns true if it is a scalar
bool IsScalar(const TensorShapeProto& proto) {
  // Note: Returns false when rank is unknown.
  if (proto.unknown_rank()) {
    return false;
  }
  // Returns false when dimension is unknown.
  for (const auto& dim : proto.dim()) {
    if (dim.size() < 0) {
      return false;
    }
  }
  return (TensorShape(proto).num_elements() == 1);
}

bool IsAnyBinary(const NodeDef& node) {
  return node.op() == "Add" || node.op() == "AddV2" || node.op() == "Mul" ||
         node.op() == "Sub";
}

}  // namespace graph
}  // namespace amd_cpu_plugin
