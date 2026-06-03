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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_EINSUM_PATTERNS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_EINSUM_PATTERNS_H_

#include <string>
#include <unordered_set>

namespace amd_cpu_plugin {

// Enumeration of supported einsum patterns.
// Each pattern maps to a specific ZenDNN execution path.
enum class EinsumPatternType {
  kUnsupported = 0,

  // 2D Patterns
  kMatMul,  // ij,jk->ik - Standard 2D matrix multiplication

  // 3D Batch Patterns
  kBatchMatMulBroadcast,  // abc,bcd->abd - 3D batch matmul with broadcast on
                          // 'a'

  // Future patterns (Phase 2+)
  // kBatchMatMulTransA,   // bji,bjk->bik - Batch matmul with A transposed
  // kBatchMatMulTransB,   // bij,bkj->bik - Batch matmul with B transposed
  // kAttentionQK,         // bnqd,bnkd->bnqk - Attention Q @ K^T pattern
  // kAttentionSV,         // bnqk,bnkd->bnqd - Attention scores @ V pattern
};

// Structure to hold decomposition information for an einsum pattern.
// This tells the kernel how to execute the einsum using ZenDNN primitives.
struct EinsumDecomposition {
  EinsumPatternType pattern = EinsumPatternType::kUnsupported;
  bool transpose_a = false;
  bool transpose_b = false;
  int num_inputs = 2;  // Default value; overwritten by AnalyzeEinsumPattern()
                       // based on the matched einsum pattern
};

// Get the pattern type for an einsum equation.
// Returns kUnsupported if the equation is not optimizable.
inline EinsumPatternType GetEinsumPatternType(const std::string& equation) {
  // 2D matmul pattern
  if (equation == "ij,jk->ik") {
    return EinsumPatternType::kMatMul;
  }

  // 3D batch matmul with broadcast on first dimension
  // abc,bcd->abd: Input1[a,b,c] @ Input2[b,c,d] -> Output[a,b,d]
  // Used in MoE/expert projection layers
  if (equation == "abc,bcd->abd") {
    return EinsumPatternType::kBatchMatMulBroadcast;
  }

  // Future patterns can be added here:
  // if (equation == "bnqd,bnkd->bnqk") return EinsumPatternType::kAttentionQK;
  // ...

  return EinsumPatternType::kUnsupported;
}

// Analyze an einsum equation and return its decomposition.
// This tells the kernel what ZenDNN operations to use.
inline EinsumDecomposition AnalyzeEinsumPattern(const std::string& equation) {
  EinsumDecomposition result;
  result.pattern = EinsumPatternType::kUnsupported;
  result.transpose_a = false;
  result.transpose_b = false;
  result.num_inputs = 0;

  EinsumPatternType pattern_type = GetEinsumPatternType(equation);

  switch (pattern_type) {
    case EinsumPatternType::kMatMul:
      // ij,jk->ik: Standard matmul, no transposes needed
      result.pattern = EinsumPatternType::kMatMul;
      result.transpose_a = false;
      result.transpose_b = false;
      result.num_inputs = 2;
      break;

    case EinsumPatternType::kBatchMatMulBroadcast:
      // abc,bcd->abd: Batch matmul with broadcast on dimension 'a'
      // Input1[a,b,c] @ Input2[b,c,d] -> Output[a,b,d]
      // This requires special handling: reshape Input1 to [a*b,c,1]
      // and Input2 to [b,c,d], then batch matmul with broadcast
      result.pattern = EinsumPatternType::kBatchMatMulBroadcast;
      result.transpose_a = false;
      result.transpose_b = false;
      result.num_inputs = 2;
      break;

    default:
      // Unsupported - leave defaults
      break;
  }

  return result;
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_EINSUM_PATTERNS_H_
