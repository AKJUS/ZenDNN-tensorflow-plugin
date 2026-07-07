#!/usr/bin/env python3
# coding=utf-8

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

"""
Run from repo root so Zen + remapper load (the shell script sets this up):

  bash tests/matmul_bias_fbn_fold.sh

"""

from __future__ import annotations

import os

import numpy as np


def _bn_reference(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    moving_mean: np.ndarray,
    moving_var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
  """Same math as the TF graph (float64 accumulate)."""
  linear = x.astype(np.float64) @ W.astype(np.float64) + b.astype(np.float64)
  inv = 1.0 / np.sqrt(moving_var.astype(np.float64) + epsilon)
  out = (linear - moving_mean.astype(np.float64)) * inv
  out = out * gamma.astype(np.float64) + beta.astype(np.float64)
  return out.astype(np.float32)


def main() -> None:
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  # Match tests/matmul_bias_fbn_fold.sh; override with export ...=0 to silence.
  os.environ.setdefault("ZENDNNL_BNFOLD_TENSOR_DBG", "1")

  import tensorflow as tf  # noqa: PLC0415

  tf.compat.v1.disable_eager_execution()

  batch, k, n = 2, 4, 8
  epsilon = 1e-3
  rng = np.random.RandomState(0)
  W_np = rng.randn(k, n).astype(np.float32)
  b_np = rng.randn(n).astype(np.float32)
  moving_mean_np = rng.randn(n).astype(np.float32) * 0.1
  moving_var_np = np.abs(rng.randn(n).astype(np.float32)) * 0.5 + 0.01
  gamma_np = rng.randn(n).astype(np.float32) * 0.2 + 1.0
  beta_np = rng.randn(n).astype(np.float32) * 0.05
  x_np = rng.randn(batch, k).astype(np.float32)

  tf.compat.v1.reset_default_graph()
  with tf.device("/CPU:0"):
    x = tf.compat.v1.placeholder(tf.float32, shape=[batch, k])
    W = tf.constant(W_np)
    b = tf.constant(b_np)
    moving_mean_c = tf.constant(moving_mean_np)
    moving_var_c = tf.constant(moving_var_np)
    gamma_c = tf.constant(gamma_np)
    beta_c = tf.constant(beta_np)

    linear = tf.raw_ops.BiasAdd(value=tf.matmul(x, W), bias=b)
    inv = tf.math.rsqrt(moving_var_c + epsilon)
    centered = tf.raw_ops.Sub(x=linear, y=moving_mean_c)
    scaled = tf.raw_ops.Mul(x=centered, y=inv)
    scaled_gamma = tf.raw_ops.Mul(x=scaled, y=gamma_c)
    # Legacy ``Add`` matches many frozen graphs; ``AddV2`` is common on TF 2.x.
    out = tf.raw_ops.Add(x=scaled_gamma, y=beta_c)

  with tf.compat.v1.Session() as sess:
    got = sess.run(out, feed_dict={x: x_np})

  ref = _bn_reference(
      x_np,
      W_np,
      b_np,
      moving_mean_np,
      moving_var_np,
      gamma_np,
      beta_np,
      epsilon,
  )
  if not np.allclose(got, ref, rtol=1e-4, atol=1e-4):
    raise AssertionError(
        "Output mismatch vs NumPy BN reference (max_abs=%r)"
        % (np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))),)
    )
  print(
      "matmul_bias_fbn_fold OK "
      "(decomposed BN inference after MatMul+BiasAdd)"
  )


if __name__ == "__main__":
  main()
