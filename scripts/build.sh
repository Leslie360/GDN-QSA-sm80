#!/usr/bin/env bash
# Build the CUDA extension(s). Use the env python if set, else system python.
#
#   ENV=/path/to/toolchain bash scripts/build.sh          # env toolchain
#   GDN_QSA_BUILD_OPS=gdn_chunk bash scripts/build.sh       # single operator
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${ENV:-}" ] && [ -x "$ENV/bin/python" ]; then
    PY="${PY:-$ENV/bin/python}"
else
    PY="${PY:-python}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
"$PY" setup.py build_ext --inplace "$@"
echo "[gdn-qsa-sm80] build OK"
