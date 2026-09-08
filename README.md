# GDN-QSA-sm80

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-37%2F37-brightgreen)](docs/VALIDATION_LOG.md)

From-scratch **SM80 (A100/A800) CUDA/CUTE** kernels for the **GDN (Gated DeltaNet)**
and **QSA (query-key sparse attention)** attention operators.

Public implementations of this architecture family are mostly Triton kernels
(e.g. fla), while first-class CUDA kernel libraries (FlashMLA, FlashKDA, …)
target SM90+ — leaving the large installed base of A100/A800 (SM80) GPUs
without hand-written CUDA kernels for GDN/QSA. This repo fills that gap.

The kernels implement the compute core of modern open-weight GDN + QSA models
(chunked gated delta-rule linear attention, block-level sparse indexer, sparse
core attention, and the gated output projection) — written from scratch for
A800 (compute capability 8.0), with tensor-core `mma.sync` + `cp.async`
kernels and reproducible benchmarks against public baselines.

> **Status**: all four operators shipped — `gdn_chunk` (M1), `qsa_indexer` + `output_gate` (M2), `qsa_core` (M3).
>
> **Validation**: clean-A800 build / test / benchmark record in [`docs/VALIDATION_LOG.md`](docs/VALIDATION_LOG.md) (37/37 tests PASS).
>
> **v0.2.4**: see [`RELEASE_NOTES.md`](RELEASE_NOTES.md) — `qsa_indexer` 3.0×
> vs eager at S=8192, `gdn_chunk` 1.27× vs fla at S=8192 with peak memory
> halved; `qsa_core` reuse path unchanged at S=8192 9.23ms (2.81× TC pass2, 9.89× scalar).

## News

- **2026.09.08 · v0.2.4** — perf release: `qsa_indexer` warp-shuffle hybrid
  bitonic final sort (45 barriers → 6; topk S=8192 0.79→0.49ms) + a 2×2
  register-blocked long-sequence score kernel (score 1.13→0.85ms) → **3.0×**
  vs vectorized eager at S=8192 (27.6× at S=512, 6.3× at S=2048); `gdn_chunk`
  workspace-free fused reset (peak memory **~halved**: S=8192 582→356 MB) and
  8-warp mma register-persistent replay → **1.27×** vs fla 0.5.2 at S=8192.
- **2026.09.07 · v0.2.3** — perf release: `qsa_core` reuse kernel pair-packed P (one u32 per token pair, half the smem slots/stores), in-warp rowsum (5 `__shfl` reductions, `plsum` array dropped), and online-softmax state fused into the P phase via an `sm_m`/`sm_l` ping-pong double buffer — S=8192 11.50→**9.23ms**, now **2.81×** over the per-query TC pass-2 and **9.89×** over scalar.
- **2026.09.06 · v0.2.2** — perf release: `qsa_core` reuse kernel inner-loop fusion — row-max merged into QK (register scores) and rescale fused into P (drops the `sm_l` barrier) — S=8192 12.88→**11.50ms**, now **2.26×** over the per-query TC pass-2 and **7.93×** over scalar.
- **2026.09.06 · v0.2.1** — perf release: `qsa_core` reuse kernel round-2 optimizations (loop-invariant fragment/token hoists, mma-fragment smem swizzle, 16-col uint32 gathers) — S=8192 20.85→12.88ms, now **2.01×** over the per-query TC pass-2 and **7.07×** over scalar; added `RELEASE_NOTES.md`.
- **2026.09.05 · v0.2.0** — perf release: `qsa_indexer` radix-select TopK, beats vectorized eager at all lengths (1.71× at S=8192, 19× at S=512); `qsa_core` TC pass-2 v3 (3.52× vs scalar) + query-tile K/V reuse kernel (20.85 ms, 4.37× vs scalar at S=8192); `gdn_chunk` dynamic GC dispatch; tests 33→37.
- **2026.09.04 · v0.1.0** — initial public release: all four operators shipped, 33/33 tests PASS, clean-A800 validation log.

## Scope

- [x] M0: repo skeleton
- [x] M1: `gdn_chunk` — Gated DeltaNet chunked linear attention
- [x] M2: `qsa_indexer` + `output_gate`
- [x] M3: `qsa_core` — sparse attention core (scalar + TC pass2)

Deliberately **out of scope** for this repo:

- `fused_linear_ce` → experimental / training-side, not in the main API
- `flashmla-sm80` → a separate SM80 MLA decode project, to be released independently

## Highlights

- **gdn_chunk up to 1.44× vs fla 0.5.2** (1.27× at S=8192, peak memory ~halved); **qsa_indexer 3.0× vs vectorized eager at S=8192** (27.6× at S=512, 6.3× at S=2048); **qsa_core TC pass-2 3.52× vs scalar, reuse path 9.89×** — full reproduce commands in [Benchmarks](#benchmarks).
- **From-scratch SM80 CUDA/CUTE** kernels (not Triton wrappers).
- Tensor-core kernels via `mma.sync` + `cp.async`, tuned for A800.
- **Fair, reproducible benchmarks** vs public baselines (fla) — see [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md).
- Numeric correctness policy documented in [`docs/CORRECTNESS_POLICY.md`](docs/CORRECTNESS_POLICY.md).
- `qsa_core` ships a scalar path (all dtypes) **and** a tensor-core pass-2 (v3) path.

## Supported operators

| Operator | Component | Target | Dtypes | Notes |
|---|---|---|---|---|
| `gdn_chunk` | Gated DeltaNet (linear attention) | SM80 (A100/A800) | bf16 | serial / reset-fast-path / two-level scan, auto-dispatch by S |
| `qsa_indexer` | QSA indexer (MQA 4Q/1K) | SM80 | fp32 | two-stage path: 3.0× vs vectorized eager at S=8192, 27.6× at S=512 (see [Benchmarks](#benchmarks)) |
| `output_gate` | Gated residual output gate | SM80 | bf16 | RMSNormGated + CUTLASS GEMM |
| `qsa_core` | QSA sparse-block attention | SM80 | scalar: all / TC: bf16 | TC pass-2 requires D=256 |

Default benchmark shapes follow the public GDN+QSA architecture
(GDN `Hq=16/Hv=32/D=128`, QSA `24Q/2KV/D=256`, bf16). Kernels are specialized
for the published head dimensions and SM80, not locked to any single model.

## Repository layout

```
csrc/                CUDA/C++ sources, one dir per operator
  gdn_chunk/         Gated DeltaNet: prepare / stage1 / stage2 / stage3 + host wrapper
  qsa_indexer/       block-level MQA indexer (kernel + pybind)
  output_gate/       RMSNormGated + out_proj (self-written + CUTLASS GEMM)
  qsa_core/          sparse core attention (scalar) + TC pass-2 (v3)
gdn_qsa_sm80/        Python package: per-op functional API + torch references
  reference/         pure-torch correctness anchors
tests/               pytest, one file per operator (37 tests)
benchmarks/          per-op benchmark scripts (bench_<op>.py)
docs/                per-op notes, methodology, correctness policy, validation log
tools/               analysis / verification helpers (reuse analysis, proto checks)
third_party/         vendored CUTLASS/CuTe headers (BSD-3-Clause)
scripts/             build.sh / verify_all.sh / bench_all.sh
```

## Install

Source build only (no wheels provided):

```bash
export TORCH_CUDA_ARCH_LIST=8.0
pip install -e . --no-build-isolation
```

Requires: CUDA ≥ 12.0, `sm_80` target (A800), PyTorch with CUDA, and a C++17
compiler. The extension is compiled in-place; `gdn_qsa_sm80.*_cuda` `.so` files
appear under the package after a successful build.

To build only a subset of operators:

```bash
GDN_QSA_BUILD_OPS=gdn_chunk,qsa_core bash scripts/build.sh
# or legacy alias: GDN_QSA_BUILD_GDN_ONLY=1 bash scripts/build.sh
```

## Testing & benchmarks

Run on a clean A800 (`CUDA_VISIBLE_DEVICES=0`):

```bash
bash scripts/verify_all.sh   # correctness gate: pytest tests/ -x
bash scripts/bench_all.sh    # gate + all four benchmark tables
```

## Usage

### GDN — `gdn_chunk`

```python
from gdn_qsa_sm80 import gdn_chunk

B, S, Hk, Hv, D = 1, 8192, 16, 32, 128
q = torch.randn(B, S, Hk, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, S, Hk, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, S, Hv, D, dtype=torch.bfloat16, device="cuda")
g = -torch.rand(B, S, Hv, dtype=torch.bfloat16, device="cuda") * 2.0
beta = torch.rand(B, S, Hv, dtype=torch.bfloat16, device="cuda").sigmoid()

out, final_state = gdn_chunk(q, k, v, g, beta, output_final_state=True)
# out: [B, S, Hv, D]    final_state: [B, Hv, D, D]
```

`gdn_chunk` auto-dispatches serial / reset-fast-path / two-level scan by decay
strength and `S`; the reset fast path is workspace-free (peak memory ~halved).
See [`docs/GDN_CHUNK.md`](docs/GDN_CHUNK.md).

### QSA indexer — `qsa_indexer` / `qsa_indexer_topk_only`

```python
from gdn_qsa_sm80 import qsa_indexer, qsa_indexer_topk_only

# full API: also returns the dense [B,S,NB] block_scores matrix (debug/compat)
block_scores, block_indices, selected_scores = qsa_indexer(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
# q/raw_keys fp32, cos/sin [B,S,R] -> block_indices [B,S,KB] int32 (-1 pad)

# memory-light API: fused score+TopK, never materializes block_scores
block_indices, selected_scores = qsa_indexer_topk_only(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
```

See [`docs/QSA_INDEXER.md`](docs/QSA_INDEXER.md).

### QSA sparse-core attention — `qsa_sparse_core_attention` / `qsa_pass2_tc`

```python
from gdn_qsa_sm80 import qsa_sparse_core_attention, qsa_expand, qsa_pass2_tc

# scalar (all dtypes)
out = qsa_sparse_core_attention(q, k, v, block_idx, r)

# TC-accelerated pass2 (bf16, D=256)
sel_idx, sel_cnt = qsa_expand(block_idx, r)
out_tc = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)

# query-tile local K/V reuse pass2 — ~2.81x faster than qsa_pass2_tc at
# S=8192 (9.23ms vs 25.89ms): shares each gathered 64-token tile across 4
# adjacent queries, packs P as token pairs, computes the rowsum in-warp, and
# fuses the online-softmax state into the P phase via a ping-pong double
# buffer; auto-falls back to v3 for S>8192.
out_re = qsa_pass2_tc_reuse(q, k, v, sel_idx, sel_cnt, r)
```

See [`docs/QSA_CORE.md`](docs/QSA_CORE.md).

### output gate — `rmsnorm_gated` + `out_proj`

```python
from gdn_qsa_sm80 import rmsnorm_gated, out_proj_gemm_cutlass

gated = rmsnorm_gated(y, z, weight)                       # [N,128] bf16
out = out_proj_gemm_cutlass(gated.reshape(T, 4096), W)    # W [N,K] bf16
```

See [`docs/OUTPUT_GATE.md`](docs/OUTPUT_GATE.md).

## Benchmarks

All numbers from a clean A800, same input / same dtype / same GPU / warmup +
median. Baselines: fla for `gdn_chunk`; vectorized eager with identical math
for `qsa_indexer`. Full methodology:
[`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md).

### gdn_chunk (vs fla 0.5.2, bf16, Hk=16/Hv=32/D=128)

| S | ours (ms) | fla (ms) | speedup | peak mem (ours, MB) |
|---|---|---|---|---|
| 2048 | 0.423 | 0.500 | 1.18x | 102.2 |
| 4096 | 0.561 | 0.640 | 1.14x | 186.6 |
| 8192 | 0.950 | 1.205 | **1.27x** | 355.5 (582 before) |
| 32768 | 3.187 | 4.189 | 1.31x (1.31–1.55 across sessions) | 1385.2 (2291 before) |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_chunk.py`

`gdn_chunk` auto-dispatches serial / reset-fast-path / two-level scan by decay
strength and sequence length.  The reset fast path is **workspace-free**: the
per-group gt metric and last-chunk `B_g` are recomputed in-CTA from raw
k/v/g/beta (fused stage-1), and each replay CTA recomputes its chunks'
`kd/qd/kr/INV/Mqk` in-CTA from raw q/k/g/beta (fused stage-3, bit-identical
math to the prepare kernel), so the ~216MB prepare workspace is only
allocated by the exact-scan fallback — peak memory **~halves** on the reset
path (S=8192 582→356 MB, S=32768 2291→1385 MB).  The replay state is now
**register-persistent**: each warp owns its 16-col state blocks across the
whole group (per-chunk serial mma latency scales as 1/kWarps) and all 8 warps
join the MMA phases.  The superchunk group size is swept per `S` (the serial
per-group replay chain shortens with smaller groups while the cross-group scan
amortizes over larger ones): `GC=8` in the `S<=2048` reset band, `GC=16` for
`S<=4096`, `GC=32` for the `8192..16384` band, `GC=64` beyond (A800 sweep,
g=-rand*2.0).

### qsa_indexer (vs vectorized eager, fp32, Hq=4/D=128/R=64/r=4/KB=512)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.034 | ~0.93 | **27.6x** |
| 2048 | 0.148 | ~0.93 | **6.3x** |
| 8192 | 1.126 | 3.373 | **3.0x** |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_indexer.py`

The two-stage path (pool → encode → tiled score → per-query TopK) beats the
vectorized eager baseline at every length, including `S=8192` (`3.0x`; the
absolute `ours` ms drift ~±20% across sessions on shared A800 infra, so quote
the ratios — see `docs/BENCHMARK_METHODOLOGY.md`).  The wins come from
SM80-scalar work-splitting and sorting:

- **Coalesced preprocess kernels.** The pool-keys and encode kernels each run
  one warp per (batch·block / batch·query·head) row, so every lane owns one
  dims-contiguous `float4`; the RMSNorm sum is a 32-lane warp reduction and the
  partial RoPE pair is exchanged with a single `__shfl_xor`.  The old
  thread-per-row versions read rows strided `D` floats apart (~2–3% coalescing)
  and under-filled the GPU at short lengths — pool 117→9 µs, encode 349→27 µs
  at `S=8192`, and `S=512` dropped ~6× overall.
- **Score kernel: invisible-tile early exit + cp.async staging + float2 keys.**
  Tiles the causal mask fully hides now write `-inf` and return before staging
  (skips the global loads + barrier); the query tile is staged with `cp.async`
  so its copies overlap the block-key staging; a `SCORE_CQ=8` tile keeps 3
  blocks resident; key rows padded to `D+2` load 4 dims as 2 float2 reads
  (half the load-issue slots) with independent per-head accumulators.
- **TopK round-2 slab narrowing.** The `==pivot` slab of the radix select is
  narrowed one byte at a time (CUB `block_topk_air` style) instead of a full
  `O(eq_n log² eq_n)` bitonic sort; once it fits one warp it is sorted in
  registers with warp-shuffle bitonic.  The exact `K` winners are then ordered
  by a warp-shuffle **hybrid bitonic** network — consecutive pairs live in
  registers, every `j<=32` exchange is a `__shfl_xor`, and only the
  `64/128/256` distances of the k-merges round-trip through shared memory
  (6 barriers instead of the previous 45).  Candidates are packed to fixed
  positions (no single-counter `atomicAdd` contention), and `P`/`K_eff` come
  from a thread-0 pivot walk.
- **Score kernel: 2×2 register-blocked variant for long sequences.** 256
  threads but four output cells each (2 query rows × 2 block columns); the
  staged block keys stay in shared as unpadded `float4` rows and both q and k
  stage via `cp.async` (`SCORE2_CQ=16`/`SCORE2_CB=64`, 64 B/cell).  Dispatched
  when `NB>=256 && S*NB>=512·1024` so every k tile stays full; short problems
  keep the `SCORE_CQ=8` kernel.

A fused topK-only variant (`qsa_indexer_topk_only`) skips materializing the
dense `[S,NB]` score matrix entirely (lower peak memory, no 64MB write/read at
`S=8192`) but is per-query CTA-based, so it loses the cross-query block-key
reuse of the tiled score kernel and is not the fast path at long sequences —
the two-stage `qsa_indexer` is recommended for `S=8192`.

### output_gate (bf16, G=4096 → O=2560)

| T | gate (ms) | proj self (ms) | proj cutlass (ms) | cutlass speedup |
|---|---|---|---|---|
| 512 | 0.011 | 0.279 | 0.087 | 3.19x |
| 2048 | 0.036 | 0.730 | 0.215 | 3.40x |
| 8192 | 0.128 | 2.505 | 0.678 | 3.69x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_output_gate.py`

### qsa_core (scalar vs TC pass2, bf16, H=24/KVH=2/D=256/KB=512/r=4)

`qsa_pass2_tc` is the TC pass-2 (v3) kernel for any `S`; `qsa_pass2_tc_reuse`
adds query-tile local K/V reuse (auto-dispatched for `1024 ≤ S ≤ 8192`).

| S | scalar (ms) | TC pass2 v3 (ms) | TC pass2 reuse (ms) | v3 vs scalar | reuse vs v3 |
|---|---|---|---|---|---|
| 512 | 1.37 | 0.35 | 0.35* | 3.91x | 1.00x |
| 2048 | 14.63 | 3.85 | 1.52 | 3.80x | **2.54x** |
| 8192 | 91.27 | 25.89 | 9.23 | 3.52x | **2.81x** |

`*` S=512 still routes to v3 (union-build overhead; dispatch gate unchanged).
Reuse is within noise of v3 at S=512 (its union-build overhead doesn't pay at
short sequences), so the auto-dispatch falls back to v3 below S=1024.
The S=8192 reuse path is 9.89x over scalar.

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_core.py`

All tables are from the clean-A800 `bash scripts/bench_all.sh` run logged in
[`docs/VALIDATION_LOG.md`](docs/VALIDATION_LOG.md).

## Roadmap

- `qsa_indexer` now beats the vectorized eager baseline at all lengths, including
  `S=8192` (3.0x), via coalesced preprocess kernels, a narrowing radix-select
  TopK with a warp-shuffle hybrid bitonic final sort, cross-query block-key
  reuse, and a 2×2 register-blocked long-sequence score kernel. Further gains
  would come from a fused score+radix kernel (single pass, no dense `[S,NB]`
  materialization) and tensor-core score with fp32-emulation precision.
- `qsa_core` TC pass-2 is a 3.52x win over scalar (S=8192 25.89ms). A query-tile
  local K/V reuse kernel (`qsa_pass2_tc_reuse`) is shipped for 1024≤S≤8192: it
  groups four adjacent queries per CTA, builds the union of their selected token
  sets in smem, and shares each gathered 64-token K/V tile — S=8192 **9.23ms
  (2.81x over the per-query v3 path, 9.89x over scalar)**, S=2048 1.52ms (2.54x
  over v3). On top of the union-sharing it (a) fuses the softmax P/plsum into
  one barrier, (b) swizzles the Q/K/P smem layouts so each mma A/B fragment
  register (columns j and j+8) is one LDS.32 instead of two scattered LDS.32,
  (c) hoists the PV/QK fragment loads and the softmax token/ownership reads out
  of the per-query loops (the mma B operands and union_tok/qmap are
  qq-invariant), (d) gathers 16 cols/thread so swz pairs store as uint32,
  (e) packs P as token pairs — one u32 per `(row, token-PAIR)` halves the P smem
  footprint and P-store instructions, (f) computes the rowsum entirely in-warp
  via 5 `__shfl` reductions (the `plsum` smem array is dropped), and (g) fuses
  the online-softmax state update into the P phase via an `sm_m`/`sm_l`
  ping-pong double buffer (no separate phase-5 barrier).
  Round-2 fusion merges the softmax row-max into QK (register scores) and the
  rescale into P (`mnew` phase, no `sm_l` barrier).
  Block-reuse analysis (`tools/analyze_qsa_core_reuse.py`) showed ~90% of the
  gather L2 traffic is shared across adjacent queries.
- `fused_linear_ce` and `flashmla-sm80` intentionally live outside this repo
  (see Scope).

## Related work

- [fla](https://github.com/fla-org/flash-linear-attention) — public baseline for
  the GDN benchmark; kernel implementations here are independent.
- `flashmla-sm80` — an SM80 FlashMLA decode optimization, maintained separately.
- [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) — CuTe/CUTLASS headers
  vendored under `third_party/cutlass` (BSD-3-Clause).

## Acknowledgements

Built on lessons from hand-optimizing GDN/QSA-family kernels for SM80 (two-level
scan, tensor-core QK/PV, bank-conflict-free layouts). Thanks to the fla and
CUTLASS communities for public baselines and primitives.

## License

[Apache-2.0](LICENSE). `third_party/` retains its own licenses
(`third_party/cutlass/LICENSE`, BSD-3-Clause).
