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

"""Expanded LayerNorm plus residual: AddN(Sub, Mul(x,scale), x).

Remapper fuses the LayerNorm piece into _FusedLayerNorm and leaves
AddV2(output, x).

Validates numerics for every AddN fan-in ordering accepted by the plugin (six
permutations). Run tests/fused_layernorm_residual.sh for ZenDNN vs oneDNN
parity when the plugin is installed.
"""

import os
import sys

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

np.random.seed(1)

BATCH = 2
DIM = 8

input_arr = np.random.normal(0.0, 1.0, size=(BATCH, DIM)).astype(np.float32)
gamma_arr = np.random.normal(0.0, 1.0, size=(DIM,)).astype(np.float32)
beta_arr = np.random.normal(0.0, 1.0, size=(DIM,)).astype(np.float32)


def _expanded_layer_norm_residual_graph(addn_order):
  """Build LN subgraph then AddN with fanins ordered per addn_order.

  addn_order is a triple (i,j,k)
of indices into [sub, mul_x, raw_x].
  """
  x = tf.compat.v1.placeholder(tf.float32, shape=[BATCH, DIM])
  gamma = tf.compat.v1.placeholder(tf.float32, shape=[DIM])
  beta = tf.compat.v1.placeholder(tf.float32, shape=[DIM])

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
  sub = tf.raw_ops.Sub(x=beta, y=mul_mu)

  branches = [sub, mul_x, x]
  ordered = [branches[addn_order[0]], branches[addn_order[1]],
             branches[addn_order[2]]]
  out = tf.raw_ops.AddN(inputs=ordered)
  return x, gamma, beta, out


def _reference_numpy():
  x = input_arr
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
  scale_np = gamma_arr / np.sqrt(var + 1e-5)
  ln = beta_arr - mean * scale_np + x * scale_np
  return ln + x


def _run_session(addn_order, graph_options=None):
  with tf.device("/CPU:0"):
    x, gamma, beta, out = _expanded_layer_norm_residual_graph(addn_order)
  sess = tf.compat.v1.Session(
      config=tf.compat.v1.ConfigProto(graph_options=graph_options))
  return sess.run(
      out,
      feed_dict={x: input_arr, gamma: gamma_arr, beta: beta_arr},
  )


def _save_session_graph_pb(pb_path, addn_order=(0, 1, 2), graph_options=None):
  """Write the session graph as a binary GraphDef (.pb)."""
  tf.compat.v1.reset_default_graph()
  with tf.Graph().as_default():
    with tf.device("/CPU:0"):
      _expanded_layer_norm_residual_graph(addn_order)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(graph_options=graph_options))
    try:
      abs_path = os.path.abspath(pb_path)
      logdir = os.path.dirname(abs_path) or "."
      os.makedirs(logdir, exist_ok=True)
      name = os.path.basename(abs_path)
      tf.io.write_graph(
          sess.graph.as_graph_def(),
          logdir,
          name,
          as_text=False,
      )
    finally:
      sess.close()


def run_all_orders(graph_options=None):
  ref = _reference_numpy()
  perms = [
      (0, 1, 2),
      (0, 2, 1),
      (1, 0, 2),
      (1, 2, 0),
      (2, 0, 1),
      (2, 1, 0),
  ]
  for p in perms:
    got = _run_session(p, graph_options=graph_options)
    if not np.allclose(got, ref, rtol=1e-5, atol=1e-5):
      raise AssertionError(
          "AddN order {} mismatch vs numpy reference".format(p))
  return ref


def main():
  graph_options = None
  bf16 = os.getenv("TF_ZENDNN_PLUGIN_BF16")
  is_zendnn = os.getenv("TF_ENABLE_ZENDNN_OPTS")
  if bf16 is not None and int(bf16) == 1 and int(is_zendnn) == 0:
    from tensorflow.core.protobuf import rewriter_config_pb2
    graph_options = tf.compat.v1.GraphOptions(
        rewrite_options=rewriter_config_pb2.RewriterConfig(
            auto_mixed_precision_onednn_bfloat16=(
                rewriter_config_pb2.RewriterConfig.ON)
        )
    )

  result = run_all_orders(graph_options=graph_options)

  res_file = sys.argv[1]
  np.save(res_file, result)

  if len(sys.argv) >= 3:
    pb_file = sys.argv[2]
  else:
    pb_file = os.path.splitext(os.path.abspath(res_file))[0] + "_graph.pb"
  _save_session_graph_pb(
      pb_file, addn_order=(0, 1, 2), graph_options=graph_options)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    run_all_orders()
    print(
        "fused_layernorm_residual: all AddN permutations match numpy reference"
    )
  else:
    main()
