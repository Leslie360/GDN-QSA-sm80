# Validation Log

Clean A800 (SM80) build / test / benchmark record for gdn-qsa-sm80. Every README
benchmark table row traces to a run below.

## Run 2 (2026-09-04) — qsa_indexer long-sequence optimization

| Field | Value |
|---|---|
| date | 2026-09-04 |
| machine / GPU | clean **NVIDIA A800-SXM4-80GB** (SM80), idle GPU |
| tree state | commits `cdd4946` (topk P2-window) + `6d57fce` (radix topk) + `b556c70` (SCORE_CQ=16) |
| build command | `ENV=... GDN_QSA_BUILD_OPS=qsa_indexer bash scripts/build.sh` (torch 2.6.0+cu124, nvcc 12.4) |
| test command | `CUDA_VISIBLE_DEVICES=0 <python> -m pytest tests/ -q` |
| correctness | **37/37 PASS** (4 indexer + 4 new fused/topk-only tests added) |

### What changed

- **Kernel D TopK: bitonic → one-round radix select.** Histogram the top 8 bits
  of a packed `(sortable score, reversed index)` uint64 key, find the pivot
  bucket, collect the ~`block_topk` winners, and bitonic-sort only those.
  CPU-validated in `tests/radix_topk_proto.py` (5000/5000 vs a full sort on
  normal/uniform/heavy-tie/constant data).  topk kernel 1.85ms → 1.23ms at S=8192.
- **Kernel C score: `SCORE_CQ` 8 → 16.** Each staged block-key tile is shared by
  16 query rows (512 threads, one thread per cell), halving per-score
  shared-memory key traffic.  score kernel 2.34ms → 1.71ms at S=8192.
  `SCORE_REG` register-blocking and `SCORE_CQ=32` were measured and rejected
  (occupancy loss).  score kernel got a 48KB+ smem opt-in.
- **New API `qsa_indexer_topk_only`**: fused per-query score+TopK, never
  materializes the dense `[B,S,NB]` block_scores (lower peak memory).  Correct
  (matches full API IOU>0.999, score err<1e-3) but not the S=8192 fast path.

### qsa_indexer vs vectorized eager (fp32, Hq=4/D=128/R=64/r=4/KB=512; median of 5)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.318 | 0.852 | 2.68x |
| 2048 | 0.528 | 0.860 | 1.63x |
| 8192 | 3.088 | 3.368 | **1.09x** |

`qsa_indexer` now beats eager at all lengths, including S=8192 (previously
0.62x / bandwidth-bound).  Fused `qsa_indexer_topk_only`: 512=0.313ms,
2048=0.730ms, 8192=6.22ms (per-query CTA loses cross-query key reuse; not the
long-sequence fast path).

---

## Run 3 (2026-09-05) — qsa_indexer coalescing + TopK narrowing

| Field | Value |
|---|---|
| date | 2026-09-05 |
| machine / GPU | clean **NVIDIA A800-SXM4-80GB** (SM80); a neighbour on GPU1 was at 100% util during some runs, numbers below are the consistent (GPU0) medians |
| tree state | commits `3dfef2f` (round-2 slab narrowing) + `df5311f` (encode vectorize) + `07a6838` (pool vectorize) |
| build command | `ENV=... CUDA_HOME=$ENV PATH=$ENV/bin:$PATH GDN_QSA_BUILD_OPS=qsa_indexer bash scripts/build.sh` (torch 2.6.0+cu124, nvcc 12.4) |
| correctness | **37/37 PASS** |

### What changed

- **TopK round-2: bitonic → byte-wise slab narrowing + warp-shuffle tail.**
  The `==b1` radix slab can be hundreds of elements (ReLU scores cluster in a
  few exponent buckets), but only its top-`need` is kept.  Narrow it one 8-bit
  byte at a time (CUB `block_topk_air` style): histogram the current byte of the
  candidate slab, keep `==pivot` as the next smaller slab, collect `>pivot` as
  final winners; once the slab fits one warp (≤32) it is warp-shuffle-sorted in
  registers (no barriers, no smem round trips).  topk kernel 1.26ms → 0.79ms.
- **encode kernel: thread-per-row → one warp per (b,q,h) row.**  Lane `dg`
  loads the dims-contiguous `float4` (fully coalesced vs ~3% before); RMSNorm
  is a 32-lane warp reduction; RoPE pair is a `__shfl_xor`.  encode 349 → 27 µs.
- **pool-keys kernel: thread-per-block → one warp per (b,blk).**  Same
  coalescing + warp-reduce treatment (also cuts the B*NB-thread underfill that
  dominated short sequences).  pool 117 → 9 µs.
- Host now requires D=128, R=64 (the production indexer shapes).

### qsa_indexer vs vectorized eager (fp32, Hq=4/D=128/R=64/r=4/KB=512)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.052 | 0.877 | 16.9x |
| 2048 | 0.263 | 0.877 | 3.33x |
| 8192 | 2.287 | 3.372 | 1.47x |

At S=8192 the remaining time is score kernel 1.47ms + topk 0.79ms (both
~roofline-bound for the scalar approach: the score kernel is smem/issue bound on
a 48KB tile, and the TopK final sort is a 45-stage bitonic of the ~K winners).
A tensor-core score with fp32-emulation or a fused score+radix kernel are the
documented next steps.

---

## Run 1 (2026-09-04)

| Field | Value |
|---|---|
| date | 2026-09-04 |
| machine / GPU | clean **NVIDIA A800-SXM4-80GB** (SM80), idle GPU used for measurement |
| tree state | commits `7653334` (M1), `d51295b` (M2+M3), `f12e89a` (review-fix) |
| build command | `TORCH_CUDA_ARCH_LIST=8.0 <python> setup.py build_ext --inplace` (torch 2.6.0+cu124, nvcc 12.4, python 3.10, CUDA_HOME set to the toolchain) |
| test command | `CUDA_VISIBLE_DEVICES=0 <python> -m pytest tests/ -v` |
| benchmark command | `CUDA_VISIBLE_DEVICES=0 <python> benchmarks/bench_<op>.py` (+ fla on PYTHONPATH for gdn_chunk baseline) |
| correctness | **33/33 PASS** (see below) |
| fla baseline | `chunk_gated_delta_rule` from [fla](https://github.com/fla-org/flash-linear-attention), pre-expanded 32-head Q/K |

### Build

All 5 extensions compiled and copied in-place:

```
gdn_qsa_sm80/_gdn_chunk.so
gdn_qsa_sm80/_qsa_indexer.so
gdn_qsa_sm80/_output_gate.so
gdn_qsa_sm80/_qsa_core.so
gdn_qsa_sm80/_qsa_pass2_tc.so
```

Renamed exports verified: `expand_blocks` present / `debug_*` gone.

### Correctness — 33/33 PASS (7.74 s)

```
test_gdn_chunk.py         7/7   (chunk + recurrent reference, auto-vs-serial, head_ratio bit-identical)
test_output_gate.py       8/8   (rmsnorm_gated, out_proj self GEMM x4, CUTLASS GEMM x2, e2e shape)
test_qsa_core.py         14/14  (scalar bf16/fp16/fp32 vs reference x9, GQA, TC pass2 vs scalar x4)
test_qsa_indexer.py       4/4   (score err<1e-3, TopK IOU>0.999, pad match, topk cap)
```

### Benchmarks (median, warmup + n=20..50; transcribed from `bash scripts/bench_all.sh`)

**gdn_chunk vs fla** (bf16, Hk=16/Hv=32/D=128):

| S | ours (ms) | fla (ms) | speedup | peak mem (MB) |
|---|---|---|---|---|
| 2048 | 0.467 | 0.671 | 1.44x | 150.5 |
| 4096 | 0.566 | 0.681 | 1.20x | 287.3 |
| 8192 | 1.013 | 1.199 | 1.18x | 573.6 |
| 32768 | 2.910 | 4.755 | 1.63x | 2291.1 |

**qsa_indexer vs vectorized eager** (fp32, Hq=4/D=128/R=64/r=4/KB=512):

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.247 | 0.882 | 3.57x |
| 2048 | 0.582 | 0.890 | 1.53x |
| 8192 | 5.453 | 3.383 | 0.62x (bandwidth-bound; see README note) |

**output_gate** (bf16, G=4096 → O=2560):

| T | gate (ms) | proj self (ms) | proj cutlass (ms) | cutlass speedup |
|---|---|---|---|---|
| 512 | 0.011 | 0.279 | 0.087 | 3.19x |
| 2048 | 0.036 | 0.730 | 0.215 | 3.40x |
| 8192 | 0.128 | 2.505 | 0.678 | 3.69x |

**qsa_core scalar vs TC pass2** (bf16, H=24/KVH=2/D=256/KB=512/r=4):

| S | scalar (ms) | TC pass2 (ms) | speedup |
|---|---|---|---|
| 512 | 1.39 | 0.85 | 1.63x |
| 2048 | 15.51 | 9.19 | 1.69x |
| 8192 | 91.17 | 62.36 | 1.46x |

### README table source

README benchmark tables were transcribed verbatim from the runs above.
Measurements on later clean A800 runs may differ by a few % (shared infra noise);
always re-run the bench scripts for authoritative current numbers.

### Repro

```bash
cd <repo>
export TORCH_CUDA_ARCH_LIST=8.0 CUDA_HOME=<toolchain> PATH=<toolchain>/bin:$PATH CUDA_VISIBLE_DEVICES=0
pip install -e . --no-build-isolation
python -m pytest tests/ -v
bash scripts/bench_all.sh             # correctness gate + all 4 benchmark tables
# or individually:
python benchmarks/bench_gdn_chunk.py  # + fla on PYTHONPATH for the baseline
python benchmarks/bench_qsa_indexer.py
python benchmarks/bench_output_gate.py
python benchmarks/bench_qsa_core.py
```
