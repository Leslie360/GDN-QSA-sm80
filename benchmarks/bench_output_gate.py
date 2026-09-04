"""Fair benchmark for output_gate: rmsnorm_gated + out_proj (self vs CUTLASS).

Reproduce on a clean A800:
    python benchmarks/bench_output_gate.py
"""

import time

import torch

from gdn_qsa_sm80 import out_proj_gemm, out_proj_gemm_cutlass, rmsnorm_gated


def _timed(fn, *a, n=30, warmup=5, **kw):
    for _ in range(warmup):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    G, O = 4096, 2560  # gated width, out width (public GDN+QSA config)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("T        gate(ms)  proj_self(ms)  proj_cutlass(ms)  cutlass_speedup")
    for T in (512, 2048, 8192):
        N = T * 32  # B*S*H rows
        y = torch.randn(N, 128, dtype=torch.bfloat16, device="cuda")
        z = torch.randn(N, 128, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
        A = torch.randn(T, G, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(O, G, dtype=torch.bfloat16, device="cuda")

        gate_ms = _timed(rmsnorm_gated, y, z, w)
        self_ms = _timed(out_proj_gemm, A, W)
        cut_ms = _timed(out_proj_gemm_cutlass, A, W)
        print(f"{T:>6}  {gate_ms:9.3f}  {self_ms:12.3f}  {cut_ms:14.3f}  {self_ms/cut_ms:7.2f}x")


if __name__ == "__main__":
    main()
