#!/ usr / bin / env python
#coding = utf - 8

#******************************************************************************
#Copyright(c) 2026 Advanced Micro Devices, Inc.All rights reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#
#******************************************************************************

"""Expanded LayerNorm subgraph (Mean / SquaredDifference / Rsqrt / Mul /
Sub / AddV2).

This matches the Remapper pattern fused into _FusedLayerNorm /
_ZenFusedLayerNorm.
Compare ZenDNN vs oneDNN by running this script with
TF_ENABLE_ZENDNN_OPTS=1 vs TF_ENABLE_ONEDNN_OPTS=1 (see
fused_layernorm.sh).
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

tf.compat.v1.disable_eager_execution()

np.random.seed(0)

#Small static shapes : [batch, dim], normalize over last axis.
BATCH = 2
DIM = 8

input_arr = np.random.normal(0.0, 1.0, size=(BATCH, DIM)).astype(np.float32)
gamma_arr = np.random.normal(0.0, 1.0, size=(DIM,)).astype(np.float32)
beta_arr = np.random.normal(0.0, 1.0, size=(DIM,)).astype(np.float32)


def _print_middle_elements(arr, n=10, label=""):
  """Print `n` contiguous elements around the middle of the flattened tensor."""
  flat = np.asarray(arr).ravel()
  size = flat.size
  if size == 0:
    print("{} middle {}: (empty tensor)".format(label or "result", n))
    return
  take = min(n, size)
  start = max(0, (size - take) // 2)
  sl = flat[start : start + take]
  prefix = "{}: ".format(label) if label else ""
  print(
      "{}middle {} flat elements [index {}:{}] = {}".format(
          prefix, take, start, start + take, sl
      )
  )


def _expanded_layer_norm_graph():
  """Build the expanded LayerNorm pattern recognized by the ZenDNN remapper."""
  x = tf.compat.v1.placeholder(tf.float32, shape=[BATCH, DIM])
  gamma = tf.compat.v1.placeholder(tf.float32, shape=[DIM])
  beta = tf.compat.v1.placeholder(tf.float32, shape=[DIM])

#Reduction over the last dimension; axes must be a Const for remapper.
  axes = tf.constant([-1], dtype=tf.int32)

  mean_mu = tf.raw_ops.Mean(input=x, axis=axes, keep_dims=True)
  sqdiff = tf.raw_ops.SquaredDifference(x=x, y=mean_mu)
  mean_var = tf.raw_ops.Mean(input=sqdiff, axis=axes, keep_dims=True)
  eps = tf.constant(1e-5, dtype=tf.float32)
  add_eps = tf.raw_ops.AddV2(x=mean_var, y=eps)
  rsqrt = tf.raw_ops.Rsqrt(x=add_eps)
  scale = tf.raw_ops.Mul(x=gamma, y=rsqrt)
  mul_mu = tf.raw_ops.Mul(x=mean_mu, y=scale)
  mul_x = tf.raw_ops.Mul(x=x, y=scale)
  #Sub : first input must be beta(see FindFusedLayerNorm validation in remapper).
  sub = tf.raw_ops.Sub(x=beta, y=mul_mu)
  out = tf.raw_ops.AddV2(x=sub, y=mul_x)
  return x, gamma, beta, out


def model(graph_options=None):
  with tf.device("/CPU:0"):
    x, gamma, beta, out = _expanded_layer_norm_graph()
  sess = tf.compat.v1.Session(
      config=tf.compat.v1.ConfigProto(graph_options=graph_options))
  return sess.run(
      out,
      feed_dict={x: input_arr, gamma: gamma_arr, beta: beta_arr},
  )


bf16 = os.getenv("TF_ZENDNN_PLUGIN_BF16")
is_zendnn = os.getenv("TF_ENABLE_ZENDNN_OPTS")
if bf16 is not None and int(bf16) == 1 and int(is_zendnn) == 0:
  print("AMP BF16 ENABLED")
  graph_options = tf.compat.v1.GraphOptions(
      rewrite_options=rewriter_config_pb2.RewriterConfig(
          auto_mixed_precision_onednn_bfloat16=(
              rewriter_config_pb2.RewriterConfig.ON)
      )
  )
  result = model(graph_options)
else:
  result = model()

res_file = sys.argv[1]
np.save(res_file, result)

# Label FUSED_LAYERNORM_BACKEND (e.g. ZenDNN / oneDNN) when set by
# fused_layernorm.sh.
_print_middle_elements(
    result,
    n=10,
    label=os.environ.get("FUSED_LAYERNORM_BACKEND", "").strip(),
)
