# gdn-qsa-sm80

Standalone **SM80 (A800) CUDA operators** for the **Gated DeltaNet + Sparse Attention (QSA)** architecture family, from scratch.

Optimized for modern open-weight GDN + QSA models on SM80/A800.

> Status: `gdn_chunk` shipped (M1). Next: `qsa_indexer` + `output_gate` (M2) → `qsa_core` (M3).

## Scope

- [x] M0: repo skeleton
- [x] M1: `gdn_chunk` — Gated DeltaNet chunked linear attention
- [ ] M2: `qsa_indexer` + `output_gate`
- [ ] M3: `qsa_core` — sparse attention core

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
| `qsa_indexer` | QSA indexer (MQA 4Q/1K) | M2 |
| `output_gate` | Gated residual output gate | M2 |
| `qsa_core` | QSA sparse-block attention | M3 |

Default benchmark shapes follow the public GDN+QSA architecture
(GDN `Hq=16/Hv=32/D=128`, QSA `24Q/2KV/D=256`, bf16). Kernels are specialized
for the published head dimensions and SM80, not locked to any single model.

## Install

```bash
pip install -e .          # python package (CUDA ext compiled per-operator as it lands)
```

Requires: CUDA ≥ 12.0, `sm_80` target, PyTorch with CUDA.

## Usage

```python
import torch
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

## Benchmarks

Fair vs fla on clean A800, bf16, `Hk=16/Hv=32/D=128` (fla given pre-expanded
Q/K; same input, same GPU, warmup + median). Full methodology:
[`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md).

### gdn_chunk

| S | ours (ms) | fla (ms) | speedup |
|---|---|---|---|
| 2048 | 0.464 | 0.699 | 1.51x |
| 4096 | 0.567 | 0.697 | 1.23x |
| 8192 | 1.011 | 1.209 | 1.20x |
| 32768 | 2.907 | 4.772 | 1.64x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_chunk.py`

## Related work

- `flashmla-sm80`: an SM80 FlashMLA decode optimization, maintained separately.
- Baseline references: [fla](https://github.com/fla-org/flash-linear-attention).

## License

[Apache-2.0](LICENSE). `third_party/` retains its own licenses.
