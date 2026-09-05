# qsa_core — QSA sparse-core attention

From-scratch SM80 sparse attention over the indexer-selected blocks. Three paths:

- **scalar** (`qsa_sparse_core_attention`): full sparse-core attention, all dtypes.
- **TC pass2** (`qsa_expand` + `qsa_pass2_tc`): tensor-core pass2 (v3), bf16 D=256.
- **TC pass2 reuse** (`qsa_expand` + `qsa_pass2_tc_reuse`): query-tile local K/V
  reuse over v3 — fastest path for `1024 ≤ S ≤ 8192` (falls back to the v3
  kernel below S=1024, above S=8192, or when `S % 4 != 0`).

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
for 2 CTA/SM occupancy, and a distributed online softmax (cells spread across
lanes + 8-lane `__shfl_xor` reduction) that removed a 32x cross-lane
redundancy. bf16, `D=256` only. TC pass2 vs scalar: S=8192 **91.36ms → 25.38ms
(3.60x)**, S=2048 3.86ms (4.47x), S=512 0.37ms (3.79x).

**TC pass2 reuse**: groups `M_TILE=4` adjacent queries per CTA, builds the union
of their selected token sets in shared memory (a `bitmap`+`atomicOr` union, no
preprocessing kernel), then streams the union in 64-token tiles — each gathered
K/V tile is shared by all four queries (reuse analysis
`tools/analyze_qsa_core_reuse.py`: ~90% of the gather L2 traffic is shared across
adjacent queries). On top of the union-sharing it
(a) fuses the softmax P/plsum into a single barrier and
(b) swizzles the Q/K/P smem layouts (`swz16(j)=(j&7)*2+(j>>3)`) so each mma
A/B fragment register (columns j and j+8) loads with one LDS.32 instead of two
scattered LDS.32. `nPitch=72` gives the PV A-fragment a full 32-bank spread.
S=8192 **20.85ms (4.37x over scalar, 1.24x over v3)**, S=2048 3.34ms (4.67x).
Dispatched only when `1024 ≤ S ≤ 8192` and `S % 4 == 0`; below S=1024 the
union-build overhead does not pay off (S=512 is within noise of v3), so it
falls back to v3.

## Interface

```python
from gdn_qsa_sm80 import qsa_sparse_core_attention, qsa_expand, qsa_pass2_tc, qsa_pass2_tc_reuse

# scalar (all dtypes)
out = qsa_sparse_core_attention(q, k, v, block_idx, r)     # q [B,S,H,D]

# TC path (bf16 D=256): expand then TC pass2
sel_idx, sel_cnt = qsa_expand(block_idx, r)
out = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)           # v3, any S
out = qsa_pass2_tc_reuse(q, k, v, sel_idx, sel_cnt, r)     # fastest for S<=8192
```

`block_idx` `[B,S,KB]` int32 (`-1` = no selection), `r` = tokens/block, default
public config `H=24, KVH=2, D=256, KB=512, r=4`.

## Correctness

`tests/test_qsa_core.py`:
- scalar vs pure-torch reference (`gdn_qsa_sm80/reference/qsa_core_ref.py`),
  global relative L1 `<1e-2` (robust metric for bf16 attention), across bf16/fp16/fp32.
- TC pass2 vs scalar forward, `relL1 < 1e-2` (6 configs incl. no-GQA).
- reuse vs scalar forward, `relL1 < 1e-2` (`tools/verify_reuse.py`,
  recent/random modes, all S).

## Bench

`python benchmarks/bench_qsa_core.py` — scalar vs TC pass2 at
`S ∈ {512, 2048, 8192}` (public config `H=24/KVH=2/D=256/KB=512/r=4`).
The reuse path is benchmarked by `tools/verify_reuse.py`.

## Reproduce

```bash
pip install -e .                 # builds _qsa_core + _qsa_pass2_tc
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_qsa_core.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_core.py
```
