"""Correctness tests for gdn_chunk (CUDA) vs pure-torch reference.

Reproduce:  python -m pytest tests/test_gdn_chunk.py -x  (on a clean A800)
"""

import pytest
import torch

from gdn_qsa_sm80 import gdn_chunk, gdn_chunk_reference


def _gen(S, gs=2.0, Hk=16, Hv=32, seed=0, device="cuda"):
    torch.manual_seed(seed)
    B = 1
    sc = 0.05
    q = (torch.randn(B, S, Hk, 128, dtype=torch.bfloat16, device=device) * sc)
    k = (torch.randn(B, S, Hk, 128, dtype=torch.bfloat16, device=device) * sc)
    v = (torch.randn(B, S, Hv, 128, dtype=torch.bfloat16, device=device) * sc)
    g = -torch.rand(B, S, Hv, dtype=torch.bfloat16, device=device) * gs
    beta = torch.rand(B, S, Hv, dtype=torch.bfloat16, device=device).sigmoid()
    return q, k, v, g, beta


@pytest.mark.parametrize("S,gs", [(2048, 2.0), (8192, 2.0), (32768, 2.0), (8192, 0.2)])
def test_vs_chunk_reference(S, gs):
    q, k, v, g, beta = _gen(S, gs)
    out, state = gdn_chunk(q, k, v, g, beta, output_final_state=True)
    # reference needs k heads == Hv (it multiplies beta[Hv] directly); expand
    q32, k32 = q.repeat_interleave(2, dim=2), k.repeat_interleave(2, dim=2)
    ref_out, ref_state = gdn_chunk_reference(q32, k32, v, g, beta, mode="chunk")
    o = out.float().cpu(); ro = ref_out.float().cpu()
    s = state.float().cpu(); rs = ref_state.float().cpu()
    orr = (o - ro).abs().max().item() / (ro.abs().max().item() + 1e-9)
    srr = (s - rs).abs().max().item() / (rs.abs().max().item() + 1e-9)
    assert orr < 2e-2, f"S={S} out rel={orr:.2e}"
    assert srr < 2e-2, f"S={S} state rel={srr:.2e}"


@pytest.mark.parametrize("S", [2048, 8192])
def test_auto_matches_serial(S):
    """auto-dispatch output must be consistent regardless of path (rel < 1e-2)."""
    q, k, v, g, beta = _gen(S, 2.0)
    out, _ = gdn_chunk(q, k, v, g, beta, output_final_state=True)
    # serial path reference: recurrent mode
    q32, k32 = q.repeat_interleave(2, dim=2), k.repeat_interleave(2, dim=2)
    ref_out, _ = gdn_chunk_reference(q32, k32, v, g, beta, mode="recurrent")
    o = out.float().cpu(); ro = ref_out.float().cpu()
    orr = (o - ro).abs().max().item() / (ro.abs().max().item() + 1e-9)
    assert orr < 2e-2, f"S={S} auto-vs-recurrent rel={orr:.2e}"


def test_head_ratio_expansion():
    """Hk=16 with Hv=32 must equal pre-expanded Hk=32 input."""
    q, k, v, g, beta = _gen(4096, 2.0, Hk=16, Hv=32)
    out16, _ = gdn_chunk(q, k, v, g, beta, output_final_state=True)
    q32 = q.repeat_interleave(2, dim=2)
    k32 = k.repeat_interleave(2, dim=2)
    out32, _ = gdn_chunk(q32, k32, v, g, beta, output_final_state=True)
    d = (out16.float().cpu() - out32.float().cpu()).abs().max().item()
    assert d == 0.0, f"head_ratio expansion not bit-identical, max|d|={d:.3e}"
