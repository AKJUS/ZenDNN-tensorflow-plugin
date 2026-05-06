/*******************************************************************************
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************/

// TensorFlow C API headers.
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/ops/zendnn/shape_inference_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

// Routine for registering _ZenEinsum op.
// This op handles einsum operations by decomposing supported patterns
// into optimized ZenDNN primitives (primarily matmul).
void RegisterZenEinsum() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ZenEinsum");

  // Einsum can have variable number of inputs (typically 1-2 for supported
  // patterns).
  TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");

  // The einsum equation string (e.g., "ij,jk->ik" for matmul).
  TF_OpDefinitionBuilderAddAttr(op_builder, "equation: string");

  // Number of input tensors.
  TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");

  // Supported data types: float32 and bfloat16.
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16} = DT_FLOAT");

  // Zen-specific attributes (following existing pattern from other ops).
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int = 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int = 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool = false");

  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnl::error_handling::apilog_error("Failed to register _ZenEinsum: ",
                                          TF_Message(status));
  } else {
    zendnnl::error_handling::apilog_info("Registered op: _ZenEinsum");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenEinsumOps() { amd_cpu_plugin::RegisterZenEinsum(); }
