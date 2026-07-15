#!/bin/bash

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

# Run from ZenDNN_TensorFlow_Plugin repo root:
#   bash tests/fused_layernorm_bf16.sh
#
# NATIVE bf16 LayerNorm in the large-offset / low-variance (cancellation)
# regime. Scores ZenDNN (fused, fp32-internal) and oneDNN (decomposed bf16)
# against a true fp64 reference. This is the regime where the fused kernel's
# single-round / no-cancellation design should clearly beat the decomposed
# path.
#
# Tune the regime with env vars, e.g.:
#   LN_OFFSET=100 LN_STD=0.25 LN_DIM=128 bash tests/fused_layernorm_bf16.sh

set -uo pipefail

PY=tests/fused_layernorm_bf16.py

# ---- fp64 reference (truth) ----
echo "== Generating fp64 reference =="
python "$PY" "truth_bf16.npy" --reference

# ---- ZenDNN (fused bf16 kernel) ----
export TF_ENABLE_ZENDNN_OPTS=1
export TF_ENABLE_ONEDNN_OPTS=0
export USE_ZENDNNL=1
export ZENDNNL_MATMUL_ALGO=1
export USE_ZENDNN_MATMUL_DIRECT=1
export FUSED_LAYERNORM_BACKEND=ZenDNN
echo "== Running ZenDNN (native bf16, fused) =="
python "$PY" "zendnn_bf16.npy"

# ---- oneDNN (decomposed bf16) ----
export TF_ENABLE_ZENDNN_OPTS=0
export TF_ENABLE_ONEDNN_OPTS=1
export FUSED_LAYERNORM_BACKEND=oneDNN
echo "== Running oneDNN (native bf16, decomposed) =="
python "$PY" "onednn_bf16.npy"
unset FUSED_LAYERNORM_BACKEND

# ---- Score both backends against the fp64 truth ----
echo
echo "===== Accuracy vs fp64 truth (native bf16, cancellation regime) ====="
python "$PY" --compare "truth_bf16.npy" \
    ZenDNN "zendnn_bf16.npy" oneDNN "onednn_bf16.npy"

rm -f truth_bf16.npy zendnn_bf16.npy onednn_bf16.npy
