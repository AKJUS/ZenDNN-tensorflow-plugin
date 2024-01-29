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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_COMMON_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_COMMON_UTILS_H_

#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"

namespace amd_cpu_plugin {
namespace graph {

// Returns true if TensorShapeProto is 1-D tensor.
bool Is1D(const TensorShapeProto& proto);

// Returns true if TensorShapeProto is 2-D tensor.
bool Is2D(const TensorShapeProto& proto);

// Returns true if TensorShapeProto is a scalar.
bool IsScalar(const TensorShapeProto& proto);

// Returns true if an node is a binary op.
bool IsAnyBinary(const NodeDef& node);

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_COMMON_UTILS_H_
