/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

static void RegisterFusedLayerNormOpDef(const char* op_name,
                                        bool add_zen_attrs) {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder(op_name);
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "gamma: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "beta: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float");
  TF_OpDefinitionBuilderAddAttr(op_builder, "axes: list(int)");
  if (add_zen_attrs) {
    TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
    TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
    TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
    TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
    TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  }
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);
  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnl::error_handling::apilog_error("Failed to register ", op_name, ": ",
                                          TF_Message(status));
  } else {
    zendnnl::error_handling::apilog_info("Registered op: ", op_name);
  }
  TF_DeleteStatus(status);
}

void RegisterFusedLayerNorm() {
  RegisterFusedLayerNormOpDef("_FusedLayerNorm", false);
}

void RegisterZenFusedLayerNorm() {
  RegisterFusedLayerNormOpDef("_ZenFusedLayerNorm", true);
}

}  // namespace amd_cpu_plugin

void RegisterZenFusedLayerNormOps() {
  amd_cpu_plugin::RegisterFusedLayerNorm();
  amd_cpu_plugin::RegisterZenFusedLayerNorm();
}
