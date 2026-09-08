# qsa_indexer — QSA block-level MQA indexer

From-scratch SM80 CUDA kernel for the sparse-block-attention indexer: given
MQA query states and a shared raw-key stream, produce a per-query shortlist of
the most-attended compressed blocks for the sparse core attention.

## Algorithm

For each query token `i` and each compressed block `b`:

1. **Query**: RMSNorm over `D` → partial RoPE (first `R=64` dims) at token pos `i`.
2. **Block keys**: AvgPool raw keys in groups of `r=4` → RMSNorm → partial RoPE
   at block-start position `p_b = b*r`.
3. **Score**: `I[i,b] = (1/sqrt(D)) * sum_h ReLU(<q[i,h], kbar[b]>)` — block-causal
   (a block is visible only if it is fully before the query: `p_b + r - 1 <= i`).
4. **TopK**: one-round **radix select** + bitonic sort of only the ~`block_topk`
   winners keeps the highest-score blocks per query (a full `O(P log²P)` bitonic
   sort of all visible blocks is not needed).

## Kernel design

- **Kernel A (pool)**: raw keys → pooled / RMSNorm'd / partially-roped block
  keys `kbar[B,NB,D]`.
- **Kernel B (encode)**: queries → RMSNorm'd / roped `qenc[B,S,Hq,D]`.
- **Kernel C (score)**: tiled like a small GEMM — `SCORE_CQ=8` query rows share
  one staged 32-block key tile, `512` threads, one thread per (query, block)
  cell, scalar float4 dots.  For long sequences a 2×2 register-blocked variant
  (Kernel C2) takes over: 256 threads but four output cells each (2 query rows
  × 2 block columns), staged block keys kept in shared as unpadded `float4`
  rows, both q and k staged via `cp.async` (`SCORE2_CQ=16`/`SCORE2_CB=64`,
  64 B/cell).  Dispatched when `NB>=256 && S*NB >= 512·1024` so every k tile
  stays full; short problems keep Kernel C.  The cross-query block-key reuse is
  what makes long sequences fast.
- **Kernel D (topk)**: per query, histogram the top 8 bits of a packed
  `(sortable score, reversed index)` key → find the pivot bucket → collect
  candidates above it → sort only the `~block_topk` winners descending.  The
  `==pivot` slab is narrowed byte-wise (CUB `block_topk_air` style); once it
  fits one warp it is warp-shuffle-sorted in registers.  The final winner order
  is a warp-shuffle **hybrid bitonic** network — consecutive pairs `(2t, 2t+1)`
  live in registers, every `j<=32` exchange is a `__shfl_xor` (zero barriers;
  `j==1` is a thread-local compare-swap), and only the `64/128/256` distances of
  the `k=128/256/512` merges round-trip through shared memory (6 barriers total
  instead of 45).  The packed 64-bit key gives the reference's exact order
  (higher score first; equal score → lower block index first).  CPU-validated
  in `tests/radix_topk_proto.py` (5000/5000 vs a full sort).

## Interface

```python
from gdn_qsa_sm80 import qsa_indexer, qsa_indexer_topk_only

# full API: also returns the dense [B,S,NB] block_scores matrix
block_scores, block_indices, selected_scores = qsa_indexer(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
# q: [B,S,Hq,D] fp32, raw_keys: [B,S,D] fp32, cos/sin: [B,S,R] fp32
# -> block_scores [B,S,NB] (-inf invalid), block_indices [B,S,KB] int32 (-1 pad),
#    selected_scores [B,S,KB] (-inf pad)

# fused topK-only: never materializes block_scores (lower peak memory)
block_indices, selected_scores = qsa_indexer_topk_only(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
```

`R % 2 == 0`, `R <= D`, `R % 4 == 0`; `NB = S // r`; default public config
`Hq=4, D=128, R=64, r=4, block_topk=512`.

## Correctness

`tests/test_qsa_indexer.py` mirrors the kernel math in pure torch
(`gdn_qsa_sm80/reference/qsa_indexer_ref.py`): score err `<1e-3`, invalid `-inf`
exact, TopK index set IOU `>0.999` (tie-order tolerant), selected scores `<1e-3`.
The radix topk is additionally cross-checked against the full two-stage API
(`test_topk_only_matches_full_api`) and the radix algorithm against a full sort
on 5000 random inputs including heavy-tie / constant cases.

## Known performance characteristic

| S | qsa_indexer (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.034 | ~0.93 | **27.6x** |
| 2048 | 0.148 | ~0.93 | **6.3x** |
| 8192 | 1.126 | 3.373 | **3.0x** |

The two-stage path beats the vectorized eager baseline at every length,
including `S=8192`.  The win comes from (1) coalesced one-warp-per-row
pool/encode kernels (dims-contiguous `float4` loads + warp-reduce RMSNorm +
`__shfl_xor` RoPE), (2) a radix TopK whose `==pivot` slab is narrowed byte-wise
and warp-shuffle-sorted instead of a full bitonic sort, a warp-shuffle hybrid
bitonic final sort (6 barriers vs 45), and (3) cross-query block-key reuse in
the tiled score kernel with a 2×2 register-blocked long-sequence variant.
The absolute `ours` ms drift ~±20% across sessions on shared A800 infra, so
quote the ratios.  The fused `qsa_indexer_topk_only`
skips the dense score matrix (lower memory) but is per-query-CTA based and loses
the cross-query key reuse of the tiled score kernel, so it is *not* the fast
path at `S=8192` — use `qsa_indexer` there.  Bench:
`python benchmarks/bench_qsa_indexer.py`.

## Reproduce

```bash
pip install -e .                 # builds gdn_qsa_sm80._qsa_indexer
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_qsa_indexer.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_indexer.py
```
