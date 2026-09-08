# Release Notes — gdn-qsa-sm80 v0.2.4

**Date**: 2026-09-08
**GPU**: NVIDIA A800-SXM4-80GB (SM80/Ampere)
**Build**: CUDA 13.0 / nvcc 13.0, torch 2.13.0+cu130 (cloud A800; CUDA 12.4 /
torch 2.6.0+cu124 toolchain also verified), bf16 tensor cores
(`mma.sync.m16n8k16`)

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

**v0.2.4 highlights** — `qsa_indexer` and `gdn_chunk`:

| Operator | S | before | after | baseline |
|---|---|---|---|---|
| `qsa_indexer` | 512 | 19.0x | **27.6x** | vs vectorized eager |
| `qsa_indexer` | 2048 | 3.39x | **6.3x** | vs vectorized eager |
| `qsa_indexer` | 8192 | 1.71x | **3.0x** | vs vectorized eager |
| `gdn_chunk` | 2048 | 1.45x | 1.18x* | vs fla |
| `gdn_chunk` | 4096 | 1.19x | 1.14x* | vs fla |
| `gdn_chunk` | 8192 | 1.34x | **1.27x*** | vs fla |
| `gdn_chunk` | 32768 | 1.62x | **1.31x*** (1.31–1.55 across sessions) | vs fla |

`*` gdn_chunk's fla baseline was upgraded to 0.5.2 (older fla baselines in this
repo's earlier runs were ~13% slower), so the new ratios are only comparable
within fla 0.5.2; the speedups hold against the current baseline.  gdn_chunk
peak memory **~halves** on the reset fast path (S=8192 582→356 MB, S=32768
2291→1385 MB).

What changed:

- **`qsa_indexer` topk: all-smem bitonic → warp-shuffle hybrid bitonic.**
  Consecutive pairs `(2t, 2t+1)` live in registers, every `j<=32` exchange is a
  `__shfl_xor` (zero barriers; `j==1` is a thread-local compare-swap), and only
  the `64/128/256` distances of the k=128/256/512 merges round-trip through
  shared memory — 6 barriers instead of 45.  topk S=8192 0.79→0.49 ms.
- **`qsa_indexer` score: 2×2 register-blocked long-sequence kernel.** 256
  threads, four output cells each (2 query rows × 2 block columns), staged block
  keys kept in shared as unpadded `float4` rows, both q and k staged via
  `cp.async` (`SCORE2_CQ=16`/`SCORE2_CB=64`, 64 B/cell).  Dispatched when
  `NB>=256 && S*NB>=512·1024`.  score S=8192 1.13→0.85 ms.
- **`gdn_chunk` workspace-free fused reset fast path.**  Per-group `gt` +
  last-chunk `B_g` recomputed in-CTA from raw k/v/g/beta (fused stage-1); each
  replay CTA recomputes its chunks' `kd/qd/kr/INV/Mqk` in-CTA (fused stage-3,
  bit-identical to prepare).  The ~216MB prepare workspace is only allocated by
  the exact-scan fallback → peak memory ~halves.  GC table re-swept per `S`.
- **`gdn_chunk` 8-warp mma + register-persistent replay state.**  The serial
  per-chunk replay state lives entirely in per-warp registers across the whole
  group (per-chunk serial mma latency scales as 1/kWarps; `state_acc` smem is
  only the `g>0` staging buffer); all 8 warps join the MMA phases.  S=8192
  1.01→0.95 ms.

**qsa_core** — unchanged from v0.2.3: `qsa_pass2_tc_reuse` (query-tile K/V
reuse, auto-dispatched for `1024 ≤ S ≤ 8192`):

| S | scalar (ms) | TC pass2 v3 (ms) | TC pass2 reuse (ms) | reuse vs v3 | reuse vs scalar |
|---|---|---|---|---|---|
| 2048 | 14.63 | 3.85 | 1.52 | 2.54x | 9.64x |
| 8192 | 91.27 | 25.89 | 9.23 | **2.81x** | **9.89x** |

**output_gate** (unchanged): cutlass proj S=8192 0.678ms vs self-written
2.505ms (3.69x).

Every number is produced by `bash scripts/bench_all.sh` (correctness gate +
all four benchmarks) on a clean A800; full provenance in
`docs/VALIDATION_LOG.md` (Run 7), methodology in `docs/BENCHMARK_METHODOLOGY.md`.

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
