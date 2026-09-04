# gdn-qsa-sm80

From-scratch **SM80 (A800) CUDA/CUTE** operators for the **Gated DeltaNet + Sparse Attention (QSA)** architecture family.

The kernels implement the compute core of modern open-weight GDN + QSA models
(chunked gated delta-rule linear attention, block-level sparse indexer, sparse
core attention, and the gated output projection) — written from scratch for
A800 (compute capability 8.0), with tensor-core `mma.sync` + `cp.async`
kernels and reproducible benchmarks against public baselines.

> **Status**: all four operators shipped — `gdn_chunk` (M1), `qsa_indexer` + `output_gate` (M2), `qsa_core` (M3).
>
> **Validation**: clean-A800 build / test / benchmark record in [`docs/VALIDATION_LOG.md`](docs/VALIDATION_LOG.md) (33/33 tests PASS).

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
- `qsa_core` ships a scalar path (all dtypes) **and** a tensor-core pass-2 (v3) path.

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

## Repository layout

```
csrc/                CUDA/C++ sources, one dir per operator
  gdn_chunk/         Gated DeltaNet: prepare / stage1 / stage2 / stage3 + host wrapper
  qsa_indexer/       block-level MQA indexer (kernel + pybind)
  output_gate/       RMSNormGated + out_proj (self-written + CUTLASS GEMM)
  qsa_core/          sparse core attention (scalar) + TC pass-2 (v3)
gdn_qsa_sm80/        Python package: per-op functional API + torch references
  reference/         pure-torch correctness anchors
tests/               pytest, one file per operator (33 tests)
benchmarks/          per-op benchmark scripts (bench_<op>.py)
docs/                per-op notes, methodology, correctness policy, validation log
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
| 2048 | 0.467 | 0.671 | 1.44x |
| 4096 | 0.566 | 0.681 | 1.20x |
| 8192 | 1.013 | 1.199 | 1.18x |
| 32768 | 2.910 | 4.755 | 1.63x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_gdn_chunk.py`

### qsa_indexer (vs vectorized eager, fp32, Hq=4/D=128/R=64/r=4/KB=512)

| S | ours (ms) | eager (ms) | speedup |
|---|---|---|---|
| 512 | 0.247 | 0.882 | 3.57x |
| 2048 | 0.582 | 0.890 | 1.53x |
| 8192 | 5.453 | 3.383 | 0.62x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_indexer.py`

> `qsa_indexer` is optimized for short/medium sequences; at `S=8192` the current
> CUDA path is bandwidth-bound (dense `[S,NB]` score matrix) and slower than the
> vectorized eager baseline. Tracked as future work — the kernel does not
> advertise a long-sequence win.

### output_gate (bf16, G=4096 → O=2560)

| T | gate (ms) | proj self (ms) | proj cutlass (ms) | cutlass speedup |
|---|---|---|---|---|
| 512 | 0.011 | 0.279 | 0.087 | 3.19x |
| 2048 | 0.036 | 0.730 | 0.215 | 3.40x |
| 8192 | 0.128 | 2.505 | 0.678 | 3.69x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_output_gate.py`

### qsa_core (scalar vs TC pass2, bf16, H=24/KVH=2/D=256/KB=512/r=4)

| S | scalar (ms) | TC pass2 (ms) | speedup |
|---|---|---|---|
| 512 | 1.39 | 0.85 | 1.63x |
| 2048 | 15.51 | 9.19 | 1.69x |
| 8192 | 91.17 | 62.36 | 1.46x |

Reproduce: `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_qsa_core.py`

All tables are from the clean-A800 `bash scripts/bench_all.sh` run logged in
[`docs/VALIDATION_LOG.md`](docs/VALIDATION_LOG.md).

## Roadmap

- `qsa_indexer` long-sequence (`S=8192+`) is bandwidth-bound and slower than a
  vectorized eager baseline; a split-K score path is planned.
- `qsa_core` TC pass-2 is a documented ~1.5x win over scalar; further gains are
  expected from a fused pass-1+pass-2 kernel.
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
