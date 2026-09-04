#!/usr/bin/env bash
# Full benchmark matrix on a clean A800; prints a markdown table for the README.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python benchmarks/verify.py        # correctness gate before timing
python -m benchmarks.run --all     # timing + table
echo "[gdn-qsa-sm80] bench_all OK"
