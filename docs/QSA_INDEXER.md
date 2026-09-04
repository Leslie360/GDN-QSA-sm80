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
4. **TopK**: shared-memory bitonic merge sort keeps the `block_topk` highest-score
   blocks per query.

## Interface

```python
from gdn_qsa_sm80 import qsa_indexer

block_scores, block_indices, selected_scores = qsa_indexer(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
# q: [B,S,Hq,D] fp32, raw_keys: [B,S,D] fp32, cos/sin: [B,S,R] fp32
# -> block_scores [B,S,NB] (-inf invalid), block_indices [B,S,KB] int32 (-1 pad),
#    selected_scores [B,S,KB] (-inf pad)
```

`R % 2 == 0`, `R <= D`, `R % 4 == 0`; `NB = S // r`; default public config
`Hq=4, D=128, R=64, r=4, block_topk=512`.

## Correctness

`tests/test_qsa_indexer.py` mirrors the kernel math in pure torch
(`gdn_qsa_sm80/reference/qsa_indexer_ref.py`): score err `<1e-3`, invalid `-inf`
exact, TopK index set IOU `>0.999` (tie-order tolerant), selected scores `<1e-3`.

## Known performance characteristic

Short/medium sequences (`S <= 2048`) beat eager/cuBLAS baselines ~1.6–2.3x;
at `S=8192` the kernel is ~1.35x slower than cuBLAS (bandwidth-bound score
matrix). Bench: `python benchmarks/bench_qsa_indexer.py`.

## Reproduce

```bash
pip install -e .                 # builds gdn_qsa_sm80._qsa_indexer
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_qsa_indexer.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_indexer.py
```
