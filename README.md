# gdn-qsa-sm80

Standalone **SM80 (A800) CUDA operators** for the **Gated DeltaNet + Sparse Attention (QSA)** architecture family, from scratch.

Optimized for modern open-weight GDN + QSA models on SM80/A800.

> Status: all four operators shipped — `gdn_chunk` (M1), `qsa_indexer` + `output_gate` (M2), `qsa_core` (M3).

## Scope

- [x] M0: repo skeleton
- [x] M1: `gdn_chunk` — Gated DeltaNet chunked linear attention
- [x] M2: `qsa_indexer` + `output_gate`
- [x] M3: `qsa_core` — sparse attention core (scalar + TC pass2)

Deliberately **out of scope** for this repo:

- `fused_linear_ce` → experimental / training-side, not in the main API
- `flashmla-sm80` → maintained separately (FlashMLA SM80 decode port, zero-CUTLASS)

## Highlights

- **From-scratch SM80 CUDA/CUTE** kernels (not Triton wrappers).
- Tensor-core kernels via `mma.sync` + `cp.async`, tuned for A800.
- **Fair, reproducible benchmarks** vs public baselines (fla) — see [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md).
- Numeric correctness policy documented in [`docs/CORRECTNESS_POLICY.md`](docs/CORRECTNESS_POLICY.md).

## Supported operators

| Operator | Component | Status |
|---|---|---|
| `gdn_chunk` | Gated DeltaNet (linear attention) | ✅ shipped |
| `qsa_indexer` | QSA indexer (MQA 4Q/1K) | ✅ shipped |
| `output_gate` | Gated residual output gate | ✅ shipped |
| `qsa_core` | QSA sparse-block attention | ✅ shipped (scalar + TC pass2) |

Default benchmark shapes follow the public GDN+QSA architecture
(GDN `Hq=16/Hv=32/D=128`, QSA `24Q/2KV/D=256`, bf16). Kernels are specialized
for the published head dimensions and SM80, not locked to any single model.

## Install

```bash
pip install -e .          # python package (CUDA ext compiled per-operator as it lands)
```

Requires: CUDA ≥ 12.0, `sm_80` target, PyTorch with CUDA.

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

`gdn_chunk` auto-dispatches serial / reset-fast-path / two-level scan by `S`.
See [`docs/GDN_CHUNK.md`](docs/GDN_CHUNK.md).

### QSA indexer — `qsa_indexer`

```python
from gdn_qsa_sm80 import qsa_indexer

block_scores, block_indices, selected_scores = qsa_indexer(
    q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512)
# q/raw_keys fp32, cos/sin [B,S,R] -> block_indices [B,S,KB] int32 (-1 pad)
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

### gdn_chunk (vs fla, bf16, Hk=16/Hv=32/D=128)

| S | ours (ms) | fla (ms) | speedup |
|---|---|---|---|
| 2048 | 0.464 | 0.699 | 1.51x |
| 4096 | 0.567 | 0.697 | 1.23x |
| 8192 | 1.011 | 1.209 | 1.20x |
| 32768 | 2.907 | 4.772 | 1.64x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_chunk.py`

### qsa_indexer (vs vectorized eager, fp32, Hq=4/D=128/R=64/r=4/KB=512)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.304 | 0.883 | 2.90x |
| 2048 | 0.583 | 0.884 | 1.51x |
| 8192 | 5.456 | 3.362 | 0.62x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_indexer.py`

### output_gate (bf16, G=4096 → O=2560)

| T | gate (ms) | proj self (ms) | proj cutlass (ms) | cutlass speedup |
|---|---|---|---|---|
| 512 | 0.012 | 0.280 | 0.088 | 3.19x |
| 2048 | 0.036 | 0.730 | 0.215 | 3.40x |
| 8192 | 0.128 | 2.750 | 0.825 | 3.34x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_output_gate.py`

### qsa_core (scalar vs TC pass2, bf16, H=24/KVH=2/D=256/KB=512/r=4)

| S | scalar (ms) | TC pass2 (ms) | speedup |
|---|---|---|---|
| 512 | 1.39 | 0.85 | 1.63x |
| 2048 | 17.25 | 9.18 | 1.88x |
| 8192 | 91.17 | 62.32 | 1.46x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_core.py`

## Related work

- `flashmla-sm80`: an SM80 FlashMLA decode optimization, maintained separately.
- Baseline references: [fla](https://github.com/fla-org/flash-linear-attention).

## License

[Apache-2.0](LICENSE). `third_party/` retains its own licenses.
