#!/bin/bash

#*******************************************************************************
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
#*******************************************************************************

# Run from ZenDNN_TensorFlow_Plugin repo root:
#   bash tests/fused_layernorm_residual.sh

set -e

export TF_ENABLE_ONEDNN_OPTS=0
echo "TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"

export USE_ZENDNNL=1
export ZENDNN_PROFILE=1
export ZENDNNL_API_LOG_LEVEL=4

echo "--- ZenDNN ---"
export TF_ENABLE_ZENDNN_OPTS=1
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"
export FUSED_LAYERNORM_BACKEND=ZenDNN
python tests/fused_layernorm_residual.py "zendnn_residual.npy"

echo "--- oneDNN ---"
export TF_ENABLE_ZENDNN_OPTS=0
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"
export TF_ENABLE_ONEDNN_OPTS=1
echo "TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"
export FUSED_LAYERNORM_BACKEND=oneDNN
python tests/fused_layernorm_residual.py "onednn_residual.npy"
unset FUSED_LAYERNORM_BACKEND

python tests/utils.py "zendnn_residual.npy" "onednn_residual.npy"

rm -f zendnn_residual.npy onednn_residual.npy
