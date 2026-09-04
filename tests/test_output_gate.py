"""Correctness tests for output_gate (RMSNormGated + out_proj).

Reproduce:  python -m pytest tests/test_output_gate.py -x  (clean A800)
"""

import pytest
import torch

from gdn_qsa_sm80 import out_proj_gemm, out_proj_gemm_cutlass, rmsnorm_gated


def torch_ref_rmsnorm_gated(y, z, weight, eps=1e-6):
    yf = y.float()
    mean = yf.pow(2).mean(-1, keepdim=True)
    normed = yf * torch.rsqrt(mean + eps)
    return (normed * weight.float() * torch.nn.functional.silu(z.float())).to(y.dtype)


def test_rmsnorm_gated():
    torch.manual_seed(0)
    N = 512
    y = torch.randn(N, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    z = torch.randn(N, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    out = rmsnorm_gated(y, z, w)
    ref = torch_ref_rmsnorm_gated(y, z, w)
    rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    assert rel < 1e-2, f"rmsnorm_gated rel {rel:.2e}"


@pytest.mark.parametrize("M,K,N", [(512, 4096, 2560), (2048, 4096, 2560), (37, 4096, 2560), (100, 128, 64)])
def test_out_proj_gemm(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ W.float().t()).to(torch.bfloat16)
    out = out_proj_gemm(A, W)
    rel = (out.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
    assert rel < 1e-2, f"out_proj_gemm rel {rel:.2e}"


@pytest.mark.parametrize("M,K,N", [(512, 4096, 2560), (8192, 4096, 2560)])
def test_out_proj_gemm_cutlass(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ref = (A.float() @ W.float().t()).to(torch.bfloat16)
    out = out_proj_gemm_cutlass(A, W)
    rel = (out.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9)
    assert rel < 1e-2, f"out_proj_gemm_cutlass rel {rel:.2e}"


def test_e2e_shape_chain():
    """rmsnorm_gated [T*32,128] -> reshape [T,4096] -> out_proj [T,2560]."""
    T = 64
    y = torch.randn(T * 32, 128, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(T * 32, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    gated = rmsnorm_gated(y, z, w).reshape(T, 32 * 128)
    W = torch.randn(2560, 4096, dtype=torch.bfloat16, device="cuda")
    out = out_proj_gemm_cutlass(gated, W)
    assert out.shape == (T, 2560)
