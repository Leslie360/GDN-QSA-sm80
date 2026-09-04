# output_gate — RMSNormGated + out_proj

SM80 kernels for the gated-residual output projection of the GDN+QSA block:
`rmsnorm_gated` (fused RMSNorm + gated SiLU) followed by the `out_proj` linear.

## Operators

### rmsnorm_gated

```
out[i] = RMSNorm(y)[i] * weight[i] * silu(z[i]),  per 128-wide row
```
Warp-per-row: one warp processes one 128-wide row via warp shuffles
(no `__syncthreads`), 8 rows per block.

### out_proj_gemm / out_proj_gemm_cutlass

```
C[M,N] = A[M,K] @ W[N,K]^T   (bf16 in, fp32 accumulate)
```
- `out_proj_gemm` — self-written SM80 tensor-core GEMM (`mma.sync` m16n8k16,
  128×64 tile, cp.async double-buffered). Zero CUTLASS dependency fallback.
- `out_proj_gemm_cutlass` — `cutlass::gemm::device::Gemm` 128×128×32, bf16,
  column-major layout trick so `W^T` is never physically transposed. Production
  path: ~3x faster than the self-written GEMM, on par with cuBLASLt.

## Interface

```python
from gdn_qsa_sm80 import rmsnorm_gated, out_proj_gemm, out_proj_gemm_cutlass

gated = rmsnorm_gated(y, z, weight)          # [N,128] bf16
out = out_proj_gemm_cutlass(gated.reshape(T, 4096), W)   # W [2560,4096] bf16
```

## Correctness

`tests/test_output_gate.py`: `rmsnorm_gated` vs fp32 manual reference (`rel<1e-2`);
both GEMMs vs `A.float() @ W.float().t()` (`rel<1e-2`) across
`(512/2048/8192)×(4096×2560)` and small shapes; end-to-end shape chain.

## Bench

`python benchmarks/bench_output_gate.py` — reports gate / self-GEMM / CUTLASS
times at `T ∈ {512, 2048, 8192}` (public GDN+QSA config `G=4096 → O=2560`).

## Reproduce

```bash
pip install -e .                 # builds gdn_qsa_sm80._output_gate
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_output_gate.py -x
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_output_gate.py
```
