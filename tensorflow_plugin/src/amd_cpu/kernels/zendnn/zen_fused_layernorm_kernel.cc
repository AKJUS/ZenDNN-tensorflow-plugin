/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <vector>

#include "lowoha_operators/normalization/lowoha_normalization.hpp"
#include "lowoha_operators/normalization/lowoha_normalization_common.hpp"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"

namespace amd_cpu_plugin {

namespace {

using zendnnl::common::data_type_t;
using zendnnl::common::status_t;
using zendnnl::lowoha::normalization::norm_algo_t;
using zendnnl::lowoha::normalization::norm_params;
using zendnnl::lowoha::normalization::norm_type_t;
using zendnnl::lowoha::normalization::normalization_direct;

bool ComputeLayerNormFlatDims(const TensorShape& shape,
                              gtl::ArraySlice<int32> axes, uint64_t* batch,
                              uint64_t* norm_size) {
  const int rank = shape.dims();
  std::vector<bool> reduced(rank, false);
  for (int32_t a : axes) {
    int idx = a < 0 ? rank + static_cast<int>(a) : static_cast<int>(a);
    if (idx < 0 || idx >= rank) {
      return false;
    }
    reduced[idx] = true;
  }
  uint64_t b = 1;
  uint64_t n = 1;
  for (int i = 0; i < rank; ++i) {
    const int64_t d = shape.dim_size(i);
    if (reduced[i]) {
      n *= static_cast<uint64_t>(d);
    } else {
      b *= static_cast<uint64_t>(d);
    }
  }
  *batch = b;
  *norm_size = n;
  return true;
}

template <typename T>
bool RunZenFusedLayerNormKernel(const Tensor& input, const Tensor& gamma,
                                const Tensor& beta, Tensor* output,
                                float epsilon, gtl::ArraySlice<int32> axes) {
  uint64_t batch = 0;
  uint64_t norm_size = 0;
  if (!ComputeLayerNormFlatDims(input.shape(), axes, &batch, &norm_size)) {
    return false;
  }
  if (static_cast<int64_t>(gamma.NumElements()) !=
          static_cast<int64_t>(norm_size) ||
      static_cast<int64_t>(beta.NumElements()) !=
          static_cast<int64_t>(norm_size)) {
    return false;
  }

  norm_params params;
  params.norm_type = norm_type_t::LAYER_NORM;
  params.batch = batch;
  params.norm_size = norm_size;
  params.num_channels = 0;
  params.epsilon = epsilon;
  params.use_scale = true;
  params.use_shift = true;
  params.algorithm = norm_algo_t::dynamic_dispatch;
  params.num_threads = 0;

  if (std::is_same<T, float>::value) {
    params.src_dt = data_type_t::f32;
    params.dst_dt = data_type_t::f32;
    params.gamma_dt = data_type_t::f32;
    params.beta_dt = data_type_t::f32;
  } else if (std::is_same<T, Eigen::bfloat16>::value) {
    params.src_dt = data_type_t::bf16;
    params.dst_dt = data_type_t::bf16;
    params.gamma_dt = data_type_t::bf16;
    params.beta_dt = data_type_t::bf16;
  } else {
    return false;
  }

  const T* in = input.flat<T>().data();
  T* out = output->flat<T>().data();
  const T* g = gamma.flat<T>().data();
  const T* b = beta.flat<T>().data();

  status_t st = normalization_direct(
      static_cast<const void*>(in), static_cast<void*>(out),
      static_cast<const void*>(g), static_cast<const void*>(b), nullptr,
      nullptr, nullptr, params);
  return st == status_t::success;
}

}  // namespace

template <typename T>
class ZenFusedLayerNormOp : public OpKernel {
 public:
  explicit ZenFusedLayerNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("axes", &axes_));
    OP_REQUIRES(context, !axes_.empty(),
                errors::InvalidArgument("axes must be non-empty"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& gamma = context->input(1);
    const Tensor& beta = context->input(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    OP_REQUIRES(context,
                RunZenFusedLayerNormKernel<T>(input, gamma, beta, output,
                                              epsilon_, axes_),
                errors::Internal("_ZenFusedLayerNorm execution failed"));
  }

 private:
  float epsilon_ = 1e-5f;
  std::vector<int32> axes_;
};

#define REGISTER_ZEN_FUSED_LAYERNORM_KERNEL(TYPE)                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenFusedLayerNorm").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenFusedLayerNormOp<TYPE>);

TF_CALL_float(REGISTER_ZEN_FUSED_LAYERNORM_KERNEL);
TF_CALL_bfloat16(REGISTER_ZEN_FUSED_LAYERNORM_KERNEL);
#undef REGISTER_ZEN_FUSED_LAYERNORM_KERNEL

}  // namespace amd_cpu_plugin
