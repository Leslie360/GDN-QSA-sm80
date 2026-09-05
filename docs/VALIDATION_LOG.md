# Validation Log

Clean A800 (SM80) build / test / benchmark record for gdn-qsa-sm80. Every README
benchmark table row traces to a run below.

## Run 4 (2026-09-05) — qsa_core pass2 + gdn_chunk dispatch

| Field | Value |
|---|---|
| date | 2026-09-05 |
| machine / GPU | clean **NVIDIA A800-SXM4-80GB** (SM80); a neighbour on GPU1 was at 100% util (numbers below are GPU0 medians) |
| tree state | commits `e9c7eda`+`6dc3460` (qsa_core pass2) + `1d2875a`+`90d2fe6`+`c4fe1cc` (gdn_chunk GC dispatch) |
| build command | `GDN_QSA_BUILD_OPS=... bash scripts/build.sh` (torch 2.6.0+cu124, nvcc 12.4) |
| correctness | **37/37 PASS** |

### qsa_core pass2 — 62.4 → 25.4 ms at S=8192

`clock64()` section instrumentation showed the online softmax was 73% of pass2
time, from two 32x cross-lane redundancies: the tile row-max loop made every
lane re-iterate all 16×8 cells, and the online rescale recomputed exp for all
16 rows per lane.  Fixed by distributing the cells across lanes (8-lane shuffle
reduction) and moving the running max/sum to one shared copy with per-lane
updates of only its own rows (gi, gi+8).  TC pass2 vs scalar:

| S | scalar (ms) | TC-pass2 (ms) | speedup |
|---|---|---|---|
| 512 | 1.39 | 0.37 | 3.79x |
| 2048 | 17.28 | 3.86 | 4.47x |
| 8192 | 91.36 | 25.38 | 3.60x |

Also tried and rejected: dual mma accumulators (neutral), N_TILE=32 for 2
CTAs/SM (27.1 ms — tile count doubles and only 4 warps do QK).

### gdn_chunk — dynamic GC dispatch

The superchunk group size is swept per S (A800, g=-rand*2.0): the serial
per-group replay chain shortens with smaller groups while the cross-group scan
amortizes over larger ones.  GC=32 for S in (4096, 16384], GC=64 elsewhere.

| S | ours (ms) | fla (ms) | speedup |
|---|---|---|---|
| 2048 | 0.464 | 0.675 | 1.45x |
| 4096 | 0.566 | 0.676 | 1.19x |
| 8192 | 0.895 | 1.197 | 1.34x |
| 32768 | 2.910 | 4.702 | 1.62x |

### Next steps (documented, not yet implemented)

- The pass2/replay kernels are at ~50% bf16 TC utilization (serial chunk chain);
  a software-pipelined rewrite (8-warp mma, register-persistent state) is the
  remaining big lever for qsa_core pass2 and gdn_chunk replay.
- gdn_chunk workspace fusion: the prepare+replay workspace round-trip (240MB
  write + read at S=8192) costs ~0.16ms; fusing prepare into the replay (with a
  stage1_reset that no longer depends on the workspace) would cut total memory
  traffic roughly in half.  A/B-verified the loads cost ~0.16ms.

---

## Run 5 (2026-09-05) — qsa_core query-tile reuse: P/plsum fusion + mma-fragment swizzle

| Field | Value |
|---|---|
| date | 2026-09-05 |
| machine / GPU | clean **NVIDIA A800-SXM4-80GB** (SM80); a neighbour on GPU1 was at 100% util (GPU0 medians below) |
| tree state | commits `fab936d` + `819872a` + `ae0137b` + `f816194` + `c886399` + `41445f5` + `6085bb7` + `4d9f4d2` + `18cb987` + `9e2d0df` (rowmax-into-QK) + `7cc1eae` (rescale-into-P) + docs |
| build command | `GDN_QSA_BUILD_OPS=qsa_pass2_tc bash scripts/build.sh` (torch 2.6.0+cu124, nvcc 12.4) |
| correctness | **37/37 PASS** (full suite) + reuse recent/random all-S PASS + dense-union S=8188/4092 PASS |

### qsa_core `qsa_pass2_tc_reuse` — query-tile local K/V reuse

`clock64()` segment instrumentation of the reuse kernel (S=8192 recent):
setup 3.4% / gather 9.3% / qk 21.3% / rowmax 12.4% / rescale 8.3% /
p_plsum 20.9% / pv 23.7% / finalize 0.8%.  Two changes closed most of the
non-gather overhead:

1. **P+plsum fused into one barrier.**  The P phase wrote bf16 p to smem and
   the plsum phase read it back across a barrier; the 8-lane shuffle reduce now
   runs on the register value (quantized `b2f(f2b(p))` to match PV's bf16
   operands exactly — relL1 unchanged at 1.6e-3).  `sm_l` is only read at
   finalize, so its update no longer needs its own barrier.
2. **mma-fragment smem swizzle.**  Each 32-bit mma A/B fragment register packs
   two bf16 (columns j and j+8) that were 8 columns apart in row-major smem →
   2x scattered LDS.32 at half byte-efficiency.  Swizzling each 16-element
   K-group as `swz16(j)=(j&7)*2+(j>>3)` puts (j, j+8) adjacent at (2j, 2j+1),
   so one LDS.32 loads the whole register.  Applied to Qsm/Ksm (d dimension,
   Q-stage + gather-K store split into 8 uint16 at swz positions) and P (token
   dimension).  Vsm is unchanged — its mma B pairs span two token rows and a
   transpose would blow the smem budget.  `nPitch` 68→72 (36 words/row) gives
   the PV A-fragment a full 32-bank spread; the redundant `[UMAX]` qmask copy
   was dropped (read `qmap[token]` live, `tok>=0` guarded) to stay under the
   166912B smem limit.

Release-bench row (same clean GPU0, `benchmarks/bench_qsa_core.py`, mean n=10):

| S | scalar (ms) | v3 `qsa_pass2_tc` (ms) | reuse `qsa_pass2_tc_reuse` (ms) | v3 vs scalar | reuse vs v3 | reuse vs scalar |
|---|---|---|---|---|---|---|
| 512 | 1.39 | 0.36 | 0.36* | 3.88x | 1.00x | 3.89x |
| 2048 | 16.38 | 3.85 | 2.09 | 4.25x | **1.85x** | 7.85x |
| 8192 | 91.12 | 25.91 | 11.50 | 3.52x | **2.26x** | **7.93x** |

`*` S=512 is within noise of v3 (reuse pays its union-build overhead only for
long sequences), so the `qsa_pass2_tc_reuse` auto-dispatch is gated to
`1024 ≤ S ≤ 8192` (commit `f816194`); below that it routes to v3.  relL1 vs
scalar 1.6e-3 (unchanged), 37/37 tests PASS.  (Rejected during the session:
pad-row-skip — consistently ~0.2ms slower; tmax-barrier merge — reading
`pmax[8]` per thread costs ~3000x the barrier saving.)

**Fixes landed after the initial Run 5 (adversarial release review):**
- `ae0137b` — union_tok out-of-bounds read on the last streaming tile when
  `S % 64 != 0` and the union is near-dense (e.g. S=8188).  Sized the union
  list `UMAX+N_TILE` and padded through the last tile boundary; verified with a
  synthetic dense-union case (S=8188/4092 reuse-vs-v3 relL1 ~1.1e-3, PASS).
- `f816194` — dispatch gate 1024≤S≤8192; reuse wired into `bench_qsa_core.py`.

**Post-release perf branch merged (2026-09-06): row-max into QK + rescale into P.**
`9e2d0df` merges the row-max into QK — the mma leaves the scaled scores in
registers, so pmax is computed right there (mask the thread's 2 score-columns,
4-lane shuffle) instead of re-reading Sc in a separate phase (removes one
barrier).  `7cc1eae` stores `mnew[r]=max(sm_m[r], tile-max)` in the tmax buffer
so the rescale AND the P phase read one value (the rescale→P barrier vanishes),
defers the sm_l/sm_m update to a single owner per row (reads its own pre-tile
sm_m/sm_l, no barrier before PV), and skips the pad-row P stores.  7 → 5
barriers per tile.  Combined: S=8192 12.88 → **11.50ms (2.26x v3, 7.93x
scalar)**; 37/37 PASS.

**Rejected on the perf branch (reverted — do not re-explore):**
- v2 "pair-ordered" mma-fragment swizzle (LDS.64 for a[0]/a[2]): 14.92ms — the
  ord16 layout collides on banks (gi*4 + {0,2,4,6} is 2-way) whereas the v1
  swizzle's gi*4+li covers all 32 banks.
- Q-only kPitchQ (144 then 272) + v2 swizzle: 11.71ms — 144 was an OOB bug
  (the swizzled row needs 256 positions); with 272 the LDS.64 gain is offset
  by the compiler/bank behaviour.  No net win.
- `corr==1` skip of the O_r rescale FMULs: 11.78ms — in recent mode each tile
  brings new tokens so the running max keeps growing; corr is rarely exactly 1
  and the branch/divergence costs more than the saved FMULs.
  Structural floor reached: p_plsum 34% / qk 25% / gather 18% (LDS + barriers
  + online-softmax bookkeeping).  A further speedup needs a structural rewrite
  (halve smem for 2 CTA/SM — the union/qmap buffers are a 25KB hard floor; or a
  kv-major tile-stationary kernel).

---

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
| tree state | commits `3dfef2f` (round-2 slab narrowing) + `df5311f` (encode vectorize) + `07a6838` (pool vectorize) + `253f0d8` (score invisible-tile exit) + `02daf87` (score float2+acc) + `7a418db` (topk pack-all) + `873b6e9` (SCORE_CQ 8) + `46c1670` (cp.async q) + `a5edc90` (topk pack+hist fuse) |
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
- **score kernel: invisible-tile early exit + float2 key reads.**  Tiles the
  causal mask fully hides write `-inf` and return before staging (skips the
  global loads + barrier that ~half the blocks wasted); key rows padded to
  `D+2` load 4 dims as 2 float2 reads (half the load-issue slots), and each
  head's dot uses independent accumulators.  score 1.47 → ~1.0 ms.
- **score: cp.async query staging + SCORE_CQ=8 tile.**  The query tile is
  staged with `cp.async` (committed, waited at the compute barrier) so its
  copies overlap the block-key staging; halving the query tile keeps 3 blocks
  resident.  Together these take score from ~1.2ms to ~1.0ms (S=8192).
- **topk: pack-all candidates + fused pack+hist.**  Dense compaction's single
  `atomicAdd` counter is replaced by fixed-position packing with a
  `NEG_INF_KEY` sentinel; `P`/`K_eff` come from a thread-0 pivot walk, and the
  pack+hist share one pass over the global scores.  topk ~0.81 → ~0.65 ms.
- Host now requires D=128, R=64 (the production indexer shapes).

### qsa_indexer vs vectorized eager (fp32, Hq=4/D=128/R=64/r=4/KB=512)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.046 | 0.875 | 19.0x |
| 2048 | 0.260 | 0.882 | 3.39x |
| 8192 | 1.974 | 3.371 | 1.71x |

At S=8192 the remaining time is score kernel ~1.0ms + topk ~0.65ms (both near
the scalar roofline: the score kernel is load-issue bound on a ~33KB tile, and
the TopK final sort is a 45-stage bitonic of the ~K winners).  A tensor-core
score with fp32-emulation, a fused score+radix kernel, or a non-bitonic final
sort are the documented next steps.

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
