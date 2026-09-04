"""Public Python API for the QSA sparse-core attention.

Wraps `gdn_qsa_sm80._qsa_core` (scalar, all dtypes) and
`gdn_qsa_sm80._qsa_pass2_tc` (TC-accelerated pass2, bf16 D=256).
"""

from __future__ import annotations

import torch

from . import _qsa_core, _qsa_pass2_tc

__all__ = ["qsa_sparse_core_attention", "qsa_expand", "qsa_pass2_tc"]


def qsa_sparse_core_attention(q, k, v, block_idx, block_size):
    """Full sparse-core attention over selected blocks (scalar path, all dtypes).

    Args:
        q: [B, S, H, D] bf16/fp16/fp32.
        k, v: [B, S, KVH, D] same dtype (GQA: H % KVH == 0).
        block_idx: [B, S, KB] int32 — selected block indices, -1 = no selection.
        block_size: tokens per compressed block (r).

    Returns:
        out: [B, S, H, D].
    """
    q, k, v, block_idx = (x.contiguous() for x in (q, k, v, block_idx))
    return _qsa_core.forward(q, k, v, block_idx, block_size)


def qsa_expand(block_idx, block_size):
    """Pass-1 expand only: selected-block indices -> selected token positions.

    Args:
        block_idx: [B, S, KB] int32.
        block_size: r.

    Returns:
        sel_idx: [B, S, NMAX] int32, sel_cnt: [B, S] int32.
    """
    block_idx = block_idx.contiguous()
    return _qsa_core.expand_blocks(block_idx, block_size)


def qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, block_size):
    """TC-accelerated pass2 (v3). bf16, D=256 only.

    Use with `qsa_expand` output: TC path = qsa_expand + qsa_pass2_tc.
    """
    q, k, v, sel_idx, sel_cnt = (x.contiguous() for x in (q, k, v, sel_idx, sel_cnt))
    return _qsa_pass2_tc.qsa_pass2_tc_v3(q, k, v, sel_idx, sel_cnt, block_size)
