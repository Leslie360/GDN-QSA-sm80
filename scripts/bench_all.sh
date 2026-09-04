#!/usr/bin/env bash
# Full benchmark matrix on a clean A800; prints per-op tables for the README.
# Runs the correctness gate first (pytest), then each benchmark script.
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

echo "[gdn-qsa-sm80] correctness gate (pytest)..."
"$PY" -m pytest tests/ -q "$@"

echo "[gdn-qsa-sm80] gdn_chunk"
"$PY" benchmarks/bench_gdn_chunk.py

echo "[gdn-qsa-sm80] qsa_indexer"
"$PY" benchmarks/bench_qsa_indexer.py

echo "[gdn-qsa-sm80] output_gate"
"$PY" benchmarks/bench_output_gate.py

echo "[gdn-qsa-sm80] qsa_core"
"$PY" benchmarks/bench_qsa_core.py

echo "[gdn-qsa-sm80] bench_all OK"
