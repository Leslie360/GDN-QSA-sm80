# qsa_core — QSA sparse-core attention

From-scratch SM80 sparse attention over the indexer-selected blocks. Two paths:

- **scalar** (`qsa_sparse_core_attention`): full sparse-core attention, all dtypes.
- **TC pass2** (`qsa_expand` + `qsa_pass2_tc`): tensor-core pass2 (v3), bf16 D=256.

## Algorithm

1. **Pass 1 (expand)**: given per-query selected *block* indices, expand each
   block into its token positions, always append the tail tokens of the current
   incomplete block, → `sel_idx`/`sel_cnt`.
2. **Pass 2 (attend)**: flash-attn-style online-softmax attention over *only*
   the selected K/V tokens (GQA: `H` query heads share `KVH` KV heads), with a
   causal mask over the gather set. Scalar path handles bf16/fp16/fp32.

**TC pass2 (v3)**: one-pass flash QK+PV via `mma.sync` m16n8k16, vectorized
`uint4` K/V gather (8 bf16 per load — gather was the #1 bottleneck), `N_TILE=64`
all-warp QK, `kPitch=264` to eliminate 8-way bank conflicts, `__launch_bounds__(256,2)`
for 2 CTA/SM occupancy. bf16, `D=256` only. Reversed the TC vs scalar gap:
S=8192 ~91ms → ~62ms (~1.46x).

## Interface

```python
from gdn_qsa_sm80 import qsa_sparse_core_attention, qsa_expand, qsa_pass2_tc

# scalar (all dtypes)
out = qsa_sparse_core_attention(q, k, v, block_idx, r)     # q [B,S,H,D]

# TC path (bf16 D=256): expand then TC pass2
sel_idx, sel_cnt = qsa_expand(block_idx, r)
out = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)
```

`block_idx` `[B,S,KB]` int32 (`-1` = no selection), `r` = tokens/block, default
public config `H=24, KVH=2, D=256, KB=512, r=4`.

## Correctness

`tests/test_qsa_core.py`:
- scalar vs pure-torch reference (`gdn_qsa_sm80/reference/qsa_core_ref.py`),
  global relative L1 `<1e-2` (robust metric for bf16 attention), across bf16/fp16/fp32.
- TC pass2 vs scalar forward, `relL1 < 1e-2` (6 configs incl. no-GQA).

## Bench

`python benchmarks/bench_qsa_core.py` — scalar vs TC pass2 at
`S ∈ {512, 2048, 8192}` (public config `H=24/KVH=2/D=256/KB=512/r=4`).

## Reproduce

```bash
pip install -e .                 # builds _qsa_core + _qsa_pass2_tc
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_qsa_core.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_core.py
```
