"""Correctness tests for qsa_core (scalar) and qsa_pass2_tc (TC v3).

Reproduce:  python -m pytest tests/test_qsa_core.py -x  (clean A800)
"""

import pytest
import torch

from gdn_qsa_sm80 import qsa_expand, qsa_pass2_tc, qsa_sparse_core_attention
from gdn_qsa_sm80.reference.qsa_core_ref import (
    expand_blocks_torch,
    make_block_idx,
    reference_sparse_attention,
)


def _gen(B, S, H, KVH, D, dtype, device="cuda"):
    torch.manual_seed(0)
    q = (torch.randn(B, S, H, D, dtype=dtype, device=device) * 0.5)
    k = (torch.randn(B, S, KVH, D, dtype=dtype, device=device) * 0.5)
    v = (torch.randn(B, S, KVH, D, dtype=dtype, device=device) * 0.5)
    return q, k, v


def rel_global(out, ref):
    num = (out.float() - ref.float()).abs().sum().item()
    den = ref.float().abs().sum().item() + 1e-9
    return num / den


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("B,S,H,KVH,D,KB,r", [
    (2, 256, 24, 2, 256, 4, 8),
    (2, 128, 24, 2, 256, 2, 8),
    (1, 64, 24, 2, 256, 3, 16),
])
def test_scalar_vs_reference(B, S, H, KVH, D, KB, r, dtype):
    q, k, v = _gen(B, S, H, KVH, D, dtype)
    block_idx = make_block_idx(B, S, KB, r, "cuda", mode="recent")
    out = qsa_sparse_core_attention(q, k, v, block_idx, r)
    ref = reference_sparse_attention(q, k, v, block_idx, r)
    assert rel_global(out, ref) < 1e-2, f"scalar rel {rel_global(out, ref):.2e}"


def test_gqa_repeat():
    """H==KVH (no GQA) must match via repeat_interleave in reference."""
    B, S, H, KVH, D, KB, r = 1, 64, 12, 12, 256, 2, 8
    q, k, v = _gen(B, S, H, KVH, D, torch.bfloat16)
    block_idx = make_block_idx(B, S, KB, r, "cuda", mode="recent")
    out = qsa_sparse_core_attention(q, k, v, block_idx, r)
    ref = reference_sparse_attention(q, k, v, block_idx, r)
    assert rel_global(out, ref) < 1e-2


@pytest.mark.parametrize("B,S,H,KVH,D,KB,r", [
    (1, 64, 12, 12, 256, 2, 8),   # no-gqa tiny
    (2, 256, 24, 2, 256, 4, 8),   # gqa
    (2, 128, 24, 2, 256, 4, 8),   # gqa shorter
    (1, 64, 24, 2, 256, 2, 16),   # gqa r=16
])
def test_pass2_tc_vs_scalar(B, S, H, KVH, D, KB, r):
    """TC pass2 (v3, bf16 D=256) must match scalar forward within relL1<1e-2."""
    q, k, v = _gen(B, S, H, KVH, D, torch.bfloat16)
    block_idx = make_block_idx(B, S, KB, r, "cuda", mode="recent")
    o_sc = qsa_sparse_core_attention(q, k, v, block_idx, r)
    sel_idx, sel_cnt = qsa_expand(block_idx, r)
    o_tc = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)
    assert rel_global(o_tc, o_sc) < 1e-2, f"tc vs scalar rel {rel_global(o_tc, o_sc):.2e}"
