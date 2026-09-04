#!/usr/bin/env bash
# Build the CUDA extension(s). Set GDN_QSA_BUILD_CUDA=1 to compile csrc.
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
GDN_QSA_BUILD_CUDA=1 pip install -e . --no-build-isolation "$@"
echo "[gdn-qsa-sm80] build OK"
