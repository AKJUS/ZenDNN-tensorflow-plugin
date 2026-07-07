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
#   bash tests/matmul_bias_fbn_fold.sh

set -e

export TF_ENABLE_ONEDNN_OPTS=0
export TF_ENABLE_ZENDNN_OPTS=1
export USE_ZENDNNL=1
export ZENDNN_DUMP_GRAPH=1
export ZENDNNL_API_LOG_LEVEL=4
# First 10 W/B floats in ApplyFuseMatmulBNfoldToTensors -> apilog
# (unset or 0 to disable)
export ZENDNNL_BNFOLD_TENSOR_DBG=1

python tests/matmul_bias_fbn_fold.py
