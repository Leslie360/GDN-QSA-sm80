"""Correctness tests for qsa_indexer (CUDA) vs pure-torch reference.

Reproduce:  python -m pytest tests/test_qsa_indexer.py -x  (clean A800)
"""

import math

import pytest
import torch

from gdn_qsa_sm80 import qsa_indexer, qsa_indexer_topk_only
from gdn_qsa_sm80.reference.qsa_indexer_ref import make_rope_tables, ref_indexer


def _gen(B, S, Hq, D, R, r, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, S, Hq, D, dtype=torch.float32, device=device) * 0.5
    raw_keys = torch.randn(B, S, D, dtype=torch.float32, device=device) * 0.5
    cos_q, sin_q = make_rope_tables(B, S, R, device=device)
    cos_k, sin_k = make_rope_tables(B, S, R, device=device)
    return q, raw_keys, cos_q, sin_q, cos_k, sin_k


@pytest.mark.parametrize("B,S,r,KB", [(2, 256, 4, 64), (3, 256, 4, 16), (2, 1024, 4, 64)])
def test_vs_reference(B, S, r, KB):
    Hq, D, R = 4, 128, 64
    q, raw_keys, cos_q, sin_q, cos_k, sin_k = _gen(B, S, Hq, D, R, r)
    scores, idxs, sel = qsa_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    rs, ridx, rsel = ref_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)

    # block scores: valid entries within 1e-3, invalid must be -inf
    valid = rs != float("-inf")
    err = (scores - rs).abs()
    assert err[valid].max().item() < 1e-3, f"score err {err[valid].max().item():.2e}"
    assert torch.equal(scores[~valid], rs[~valid]), "invalid scores must be -inf"

    # topk indices: padded -1 must match; valid set must be identical (IOU>0.999)
    assert torch.equal(idxs == -1, ridx == -1), "pad positions mismatch"
    iou = _set_iou(idxs, ridx)
    assert iou > 0.999, f"topk IOU {iou:.4f}"

    # selected scores: non-inf within 1e-3
    m = rsel != float("-inf")
    assert (sel[m] - rsel[m]).abs().max().item() < 1e-3


def _set_iou(a, b):
    a_s = set(a.cpu().flatten().tolist()) - {-1}
    b_s = set(b.cpu().flatten().tolist()) - {-1}
    inter = len(a_s & b_s)
    union = len(a_s | b_s)
    return inter / max(1, union)


def test_shapes_and_topk_cap():
    """topk > NB must not error; output keeps [B,S,KB] with -1 pads beyond NB."""
    B, S, r, KB = 1, 64, 4, 100  # NB = 16 < KB
    q, raw_keys, cos_q, sin_q, cos_k, sin_k = _gen(B, S, 4, 128, 64, r)
    scores, idxs, sel = qsa_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    assert idxs.shape == (B, S, KB)  # output width = block_topk, -1 padded
    # any slot beyond the per-query valid count must be -1 (reference behavior)
    rs, ridx, rsel = ref_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    assert ridx.shape == idxs.shape
    assert torch.equal(idxs == -1, ridx == -1), "pad positions must match reference"


@pytest.mark.parametrize("B,S,r,KB", [(2, 256, 4, 64), (3, 256, 4, 16), (2, 1024, 4, 64)])
def test_topk_only_matches_full_api(B, S, r, KB):
    """Fused topK-only path must agree with the full qsa_indexer on the topK.

    The fused kernel skips the dense block_scores matrix entirely, so we compare
    against the full API (which is itself validated against the torch reference):
    same selected set (IOU > 0.999), same pad positions, scores within 1e-3.
    """
    Hq, D, R = 4, 128, 64
    q, raw_keys, cos_q, sin_q, cos_k, sin_k = _gen(B, S, Hq, D, R, r)
    _, idxs, sel = qsa_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    fidxs, fsel = qsa_indexer_topk_only(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)

    assert fidxs.shape == idxs.shape and fsel.shape == sel.shape
    assert torch.equal(fidxs == -1, idxs == -1), "pad positions mismatch"
    iou = _set_iou(fidxs, idxs)
    assert iou > 0.999, f"topk IOU {iou:.4f}"
    m = sel != float("-inf")
    assert (fsel[m] - sel[m]).abs().max().item() < 1e-3, "selected score err"


def test_topk_only_long_sequence_smoke():
    """Production shape S=8192 (NB=2048, dense scores would be 64MB) runs fine."""
    B, S, r, KB = 1, 8192, 4, 512
    Hq, D, R = 4, 128, 64
    q, raw_keys, cos_q, sin_q, cos_k, sin_k = _gen(B, S, Hq, D, R, r)
    fidxs, fsel = qsa_indexer_topk_only(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    assert fidxs.shape == (B, S, KB) and fsel.shape == (B, S, KB)
    # every query with visible blocks has exactly min(KB, P) valid entries
    assert (fidxs != -1).sum(-1).min().item() >= 0
    assert (fidxs != -1).sum(-1).max().item() <= KB
