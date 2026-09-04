#!/usr/bin/env bash
# Full correctness gate on a clean A800. Refuses to proceed if any test fails.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m pytest tests/ -x "$@"
echo "[gdn-qsa-sm80] verify_all OK"
