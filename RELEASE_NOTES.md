# Release Notes — gdn-qsa-sm80 v0.2.3

**Date**: 2026-09-07
**GPU**: NVIDIA A800-SXM4-80GB (SM80/Ampere)
**Build**: CUDA 13.0 / nvcc 13.0, torch 2.13.0+cu130 (Run 6; CUDA 12.4 / torch 2.6.0+cu124 toolchain also verified), bf16 tensor cores (`mma.sync.m16n8k16`)

From-scratch SM80 CUDA operators for the **Gated DeltaNet + QSA** attention
family (the public Qwen3.8-Flash-Next architecture). No Triton, no cuDNN; only
the vendored CUTLASS/CuTe headers under `third_party/` (BSD-3-Clause).

## Operators

| Operator | Function | Ships |
|---|---|---|
| `gdn_chunk` | Gated DeltaNet chunked linear attention (prepare / stage1-3, auto-dispatch serial / reset-fast-path / two-level scan) | bf16 |
| `qsa_indexer` | block-level MQA indexer (radix-select TopK) | fp32 |
| `qsa_core` | QSA sparse-core attention (scalar all-dtypes + TC pass2 `qsa_pass2_tc` + query-tile reuse `qsa_pass2_tc_reuse`) | bf16/fp16/fp32 |
| `output_gate` | RMSNormGated + out_proj (self-written + CUTLASS GEMM) | bf16 |

## Performance (S = sequence length, clean A800, median of N)

**qsa_core** — the headline: `qsa_pass2_tc_reuse` (query-tile K/V reuse,
auto-dispatched for `1024 ≤ S ≤ 8192`) vs the per-query TC pass2:

| S | scalar (ms) | TC pass2 v3 (ms) | TC pass2 reuse (ms) | reuse vs v3 | reuse vs scalar |
|---|---|---|---|---|---|
| 2048 | 14.63 | 3.85 | 1.52 | 2.54x | 9.64x |
| 8192 | 91.27 | 25.89 | 9.23 | **2.81x** | **9.89x** |

The reuse kernel groups 4 adjacent queries per CTA, shares each gathered
64-token K/V tile across them, fuses the softmax P/plsum into one barrier,
swizzles the Q/K/P smem layouts for single-LDS mma fragments, hoists all
loop-invariant fragment/token loads out of the per-query loops, and gathers
16 cols/thread with paired uint32 stores.  v0.2.3 adds three structural
changes on top: P is packed as token **pairs** — one u32 per `(row, token-PAIR)`
halves the P smem footprint and P-store instructions (11.54→9.40ms); the
rowsum is computed **entirely in-warp** via 5 `__shfl` reductions (the `plsum`
smem array is dropped); and the online-softmax state update is **fused into
the P phase** via an `sm_m`/`sm_l` ping-pong double buffer, eliminating the
separate phase-5 barrier (→9.23ms, 230 regs no spill).

**Other operators** (S=8192): `gdn_chunk` 0.895ms vs fla 1.197ms (1.34x);
`qsa_indexer` 1.974ms vs vectorized eager 3.371ms (1.71x, all lengths);
`output_gate` cutlass proj 0.678ms vs self-written 2.505ms (3.69x).

Every number is produced by `bash scripts/bench_all.sh` (correctness gate +
all four benchmarks) on a clean A800; full provenance in `docs/VALIDATION_LOG.md`,
methodology in `docs/BENCHMARK_METHODOLOGY.md`.

## Install

```bash
# SM80 (A800/Ampere) + CUDA 12.4 toolchain required
pip install -e . --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ -q     # 37/37 PASS
CUDA_VISIBLE_DEVICES=0 bash scripts/bench_all.sh        # all benchmark tables
```

## Correctness

- **37/37 tests PASS** across the four operators.
- Global relative L1 `<1e-2` vs pure-torch references for qsa_core
  (recent/random modes, all S) and gdn_chunk.
- `qsa_indexer` TopK `IOU > 0.999`; `output_gate` bit-comparable to eager.
- See `docs/CORRECTNESS_POLICY.md` for the bf16 error floor and metrics.

## Notes

- Source build only; SM80 (`-arch=sm_80`) required — no wheels.
- The reuse kernel is gated to `1024 ≤ S ≤ 8192` (its union-build overhead does
  not pay below S=1024); `qsa_pass2_tc` (v3) covers any S.
- License: Apache-2.0 (repo) + BSD-3-Clause (vendored CUTLASS headers).
