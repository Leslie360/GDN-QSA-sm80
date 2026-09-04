#!/usr/bin/env bash
# Full correctness gate on a clean A800. Refuses to proceed if any test fails.
#
# Use the env python if set (e.g. ENV=/path/to/toolchain), else system python.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${ENV:-}" ] && [ -x "$ENV/bin/python" ]; then
    PY="${PY:-$ENV/bin/python}"
else
    PY="${PY:-python}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$PY" -m pytest tests/ -x "$@"
echo "[gdn-qsa-sm80] verify_all OK"
