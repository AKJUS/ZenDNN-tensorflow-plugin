#!/usr/bin/env python
# coding=utf-8

# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************

"""NATIVE-bf16 LayerNorm in the catastrophic-cancellation regime.

Why this file exists (vs fused_layernorm.py):
  * fused_layernorm.py relies on AMP to go bf16, but AMP's allowlist is
    anchored on MatMul/Conv, so a standalone LayerNorm graph stays fp32
    ("No ops in allow_set, skipping AMP"). Here the graph is built DIRECTLY
    in bf16 (inputs cast to bf16 up front, every op bf16), so both backends
    truly run bf16 -- ZenDNN fuses to the fp32-internal kernel, oneDNN runs
    the decomposed bf16 ops.
  * Inputs use a LARGE offset + SMALL variance. Then x ~= mean, so the
    decomposed 'x*scale - mean*scale' subtracts two big, nearly-equal bf16
    numbers -> catastrophic cancellation. The fused kernel does (x-mean)
    first in fp32, so it avoids it. This mirrors the model's near-zero
    output regime where the accuracy test fails.

Usage:
  python fused_layernorm_bf16.py <out.npy>              # run backend
  python fused_layernorm_bf16.py <out.npy> --reference  # fp64 truth

Regime is tunable via env: LN_BATCH, LN_DIM, LN_OFFSET, LN_STD, LN_EPS.
"""

import os
import sys

import numpy as np

BATCH = int(os.getenv("LN_BATCH", "8"))
DIM = int(os.getenv("LN_DIM", "64"))
OFFSET = float(os.getenv("LN_OFFSET", "30.0"))   # large DC offset -> cancellation
STD = float(os.getenv("LN_STD", "0.5"))          # small variation around offset
EPS = float(os.getenv("LN_EPS", "1e-5"))


def make_inputs():
  """Deterministic inputs shared by every backend and the reference."""
  rng = np.random.default_rng(1234)
  x = (rng.normal(0.0, STD, size=(BATCH, DIM)) + OFFSET).astype(np.float32)
  gamma = rng.normal(1.0, 0.05, size=(DIM,)).astype(np.float32)
  beta = rng.normal(0.0, 0.02, size=(DIM,)).astype(np.float32)
  return x, gamma, beta


def to_bf16(x):
  """Round-to-nearest-even fp32 -> bf16 (stored in a float32 container)."""
  x = np.asarray(x, dtype=np.float32)
  u = x.view(np.uint32)
  rounding_bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
  u = (u + rounding_bias) & np.uint32(0xFFFF0000)
  return u.view(np.float32)


def run_reference(out_path):
  """True fp64 LayerNorm on the same bf16-rounded inputs the backends see."""
  x, gamma, beta = make_inputs()
  xb, gb, bb = to_bf16(x), to_bf16(gamma), to_bf16(beta)
  x64 = xb.astype(np.float64)
  mean = x64.mean(axis=-1, keepdims=True)
  var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)  # biased (÷N)
  inv_std = 1.0 / np.sqrt(var + np.float64(EPS))
  y = gb.astype(np.float64) * (x64 - mean) * inv_std + bb.astype(np.float64)
  np.save(out_path, y.astype(np.float32))
  print("[TRUE fp64 reference] saved {} (offset={}, std={}, N={})".format(
      out_path, OFFSET, STD, DIM))


def run_backend(out_path):
  """Build the expanded LayerNorm natively in bf16 and run it."""
  import ml_dtypes  # bundled with TF 2.16; gives numpy bfloat16
  import tensorflow as tf

  tf.compat.v1.disable_eager_execution()
  x_arr, gamma_arr, beta_arr = make_inputs()
  bf16_np = ml_dtypes.bfloat16
  # Feed real bf16 data so the placeholders' dtype (DT_BFLOAT16) is genuine.
  x_bf = x_arr.astype(bf16_np)
  g_bf = gamma_arr.astype(bf16_np)
  b_bf = beta_arr.astype(bf16_np)

  with tf.device("/CPU:0"):
    # Native bf16 placeholders: their "dtype" attr is DT_BFLOAT16, which the
    # fusion validation accepts (unlike a Cast, which exposes SrcT/DstT). No
    # AMP, no interior casts -> the remapper LayerNorm pattern matches and
    # fuses, and the input is genuinely bf16.
    x = tf.compat.v1.placeholder(tf.bfloat16, shape=[BATCH, DIM])
    gamma = tf.compat.v1.placeholder(tf.bfloat16, shape=[DIM])
    beta = tf.compat.v1.placeholder(tf.bfloat16, shape=[DIM])

    axes = tf.constant([-1], dtype=tf.int32)
    eps = tf.constant(EPS, dtype=tf.bfloat16)

    mean_mu = tf.raw_ops.Mean(input=x, axis=axes, keep_dims=True)
    sqdiff = tf.raw_ops.SquaredDifference(x=x, y=mean_mu)
    mean_var = tf.raw_ops.Mean(input=sqdiff, axis=axes, keep_dims=True)
    add_eps = tf.raw_ops.AddV2(x=mean_var, y=eps)
    rsqrt = tf.raw_ops.Rsqrt(x=add_eps)
    scale = tf.raw_ops.Mul(x=gamma, y=rsqrt)
    mul_mu = tf.raw_ops.Mul(x=mean_mu, y=scale)
    mul_x = tf.raw_ops.Mul(x=x, y=scale)
    sub = tf.raw_ops.Sub(x=beta, y=mul_mu)   # beta first (remapper requirement)
    out = tf.raw_ops.AddV2(x=sub, y=mul_x)

  sess = tf.compat.v1.Session()
  result = sess.run(out, feed_dict={x: x_bf, gamma: g_bf, beta: b_bf})
  # result is bf16; store as float32 for comparison tooling.
  np.save(out_path, np.asarray(result, dtype=np.float32))
  label = os.environ.get("FUSED_LAYERNORM_BACKEND", "").strip()
  print("{}: saved {} shape={} dtype(bf16 compute)".format(
      label or "backend", out_path, np.asarray(result).shape))


def compare_to_reference(ref_path, labeled_results):
  """Score each backend result against the fp64 reference (L1/L2/Linf).

  In the cancellation regime the fused fp32-internal kernel (ZenDNN) should be
  closer to the fp64 truth than the decomposed bf16 path (oneDNN).
  """
  ref = np.load(ref_path).astype(np.float64).ravel()
  print("Reference (truth): {}".format(ref_path))
  print("-" * 78)
  best_label, best_l1 = None, float("inf")
  for label, res_path in labeled_results:
    res = np.load(res_path).astype(np.float64).ravel()
    if res.shape != ref.shape:
      print("{:10s}: SHAPE MISMATCH {} vs ref {}".format(
          label, res.shape, ref.shape))
      continue
    d = np.abs(res - ref)
    l1 = float(d.mean())
    l2 = float(np.sqrt((d ** 2).mean()))
    linf = float(d.max())
    print("{:10s} vs truth:  L1={:.6e}  L2={:.6e}  Linf={:.6e}".format(
        label, l1, l2, linf))
    if l1 < best_l1:
      best_l1, best_label = l1, label
  print("-" * 78)
  print("CLOSEST TO TRUTH: {} (lowest L1 vs reference)".format(best_label))


def main():
  args = sys.argv[1:]
  if not args:
    print("usage: fused_layernorm_bf16.py <out.npy> [--reference]")
    print("       fused_layernorm_bf16.py --compare <ref.npy> "
          "<label> <res.npy> [<label> <res.npy> ...]")
    sys.exit(2)
  # Accuracy scoring: --compare <ref.npy> <label> <res.npy> [...]
  if args[0] == "--compare":
    ref_path = args[1]
    rest = args[2:]
    labeled = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    compare_to_reference(ref_path, labeled)
    return
  out_path = args[0]
  if "--reference" in args[1:]:
    run_reference(out_path)
  else:
    run_backend(out_path)


if __name__ == "__main__":
  main()
