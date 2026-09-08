# gdn_chunk — Gated DeltaNet chunked linear attention

From-scratch SM80 (A800) CUDA kernel for gated delta-rule (GDN) linear attention,
with a superchunk two-level scan and an auto dispatch between serial / reset
fast-path / two-level paths.

## Algorithm

Gated delta-rule recurrence over a per-head state matrix `S ∈ R^{D×D}`
(`D` = head dim), per token `i`:

```
g_i < 0                                (log-decay; state scales by exp(g_i))
S_i = exp(g_i) * S_{i-1} + beta_i * k_i^T @ (v_i - k_i @ S_{i-1})
o_i = q_i @ S_i
```

with `q_i, k_i ∈ R^{1×D}`, `v_i, o_i ∈ R^{1×D}`, `beta_i ∈ R`, and a per-row
gate `beta_i ∈ [0,1]` controlling the write. Equivalently, the rank-1 update
`(v_i - k_i S_{i-1})` is written with strength `beta_i` after decaying the
state.

> The CUDA kernel internally stores the state in a transposed MMA-friendly
> physical layout; the public `final_state` output follows the logical `[D, D]`
> convention (the host wrapper transposes on the way out). `q @ S` is the
> row-vector form of the attention readout.

Chunked form solves an intra-chunk lower-triangular system (via
`solve_triangular`) for `v_new`, then a sequential inter-chunk recurrence over
the carried state. See `gdn_qsa_sm80/reference/gdn_chunk_ref.py` for the exact
torch formulation used as the correctness anchor.

## Kernel structure

Two-level (superchunk) scan splits the sequence into `G = S / (CHUNK·GC)`
groups; within each group chunks are processed in parallel and group transitions
are combined by a small affine scan (Hillis-Steele or Blelloch), then a parallel
group replay produces `out` and `final_state`.

| stage | file | role |
|---|---|---|
| prepare | `gdn_kernel.cu` | workspace: `kd/qd/kr/gt/inv/mqk` (QK/V projections, decay, inverse) |
| stage 1 | `gdn_scan_stage1.cu` | per-group transfer `A_g/B_g` from workspace + raw v/beta |
| reset | `gdn_scan_stage1_reset.cu` | reset fast path: `B_g`-only + decay-bound metric |
| stage 2 | `gdn_scan_stage2.cu` / `_blelloch.cu` | affine prefix scan over groups |
| stage 3 | `gdn_kernel.cu` (replay) | parallel group replay → `out`, `final_state` |
| host | `gdn_ops.cu` | torch wrapper + auto dispatch |

## Auto dispatch (`forward_gdn_chunk_auto`)

- `S <= 512`: serial recurrence (sync/launch overhead not amortized by scan).
- `512 < S <= 2048`: reset fast path when `mean(-g) >= 0.55` (strong decay makes
  reset replay win over serial), else serial.
- `S > 2048`: two-level reset fast path with a fine per-`S` GC table
  (`GC=16` for `S<=4096`, `GC=32` for `S<=16384`, `GC=64` beyond — the fused
  replay wants ~256+ CTAs to fill 108 SMs at 2 CTA/SM).

The reset fast path is **workspace-free**: per-group `gt` + last-chunk `B_g`
are recomputed in-CTA from raw k/v/g/beta (fused stage-1) and each replay CTA
recomputes its chunks' `kd/qd/kr/INV/Mqk` in-CTA (fused stage-3, bit-identical
to prepare) — the ~216MB prepare workspace is only allocated by the exact-scan
fallback, so peak memory ~halves.  The replay state is register-persistent
(each warp owns its 16-col state blocks, per-chunk serial mma latency scales as
1/kWarps, all 8 warps in the MMA phases).

Correctness is never at risk: the two-level path internally falls back to the
exact scan when reset does not hold.

## Key correctness / precision notes

- Head ratio is fused into prepare (`Hk=16` → `Hv=32` expansion inside the
  kernel), eliminating a separate `repeat_interleave` — see `test_head_ratio_expansion`
  which asserts bit-identical output vs pre-expanded input.
- `final_state` is stored `[N,H,D,D]` contiguous in the *transposed* physical
  layout; the kernel transposes on the way out to match the logical state
  convention of the torch reference.
- bf16 state model: measured `rel ~9–11e-3` (out) / `~7–10e-3` (state) for
  `S=2048–32768`, bounded by decayed accumulation (see
  `docs/CORRECTNESS_POLICY.md`).

## Fair benchmark (A800 clean, bf16, Hk=16/Hv=32/D=128)

`python benchmarks/bench_gdn_chunk.py` — same input, same dtype, fla baseline
given pre-expanded 32-head Q/K (no repeat-trap), same GPU, warmup + median.
Baseline is **fla 0.5.2** (older fla baselines in this repo's earlier runs were
~13% slower, so ratios are only comparable within the same fla version).

| S | ours (ms) | fla (ms) | speedup | peak mem (ours, MB) |
|---|---|---|---|---|
| 2048 | 0.423 | 0.500 | 1.18x | 102.2 |
| 4096 | 0.561 | 0.640 | 1.14x | 186.6 |
| 8192 | 0.950 | 1.205 | **1.27x** | 355.5 (582 before) |
| 32768 | 3.187 | 4.189 | 1.31x (1.31–1.55 across sessions) | 1385.2 (2291 before) |

## Reproduce

```bash
pip install -e .                 # builds gdn_qsa_sm80._gdn_chunk (CUDA, sm_80)
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_gdn_chunk.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_chunk.py
```
