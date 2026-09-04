"""Public Python API for the QSA block-level MQA indexer.

Wraps the compiled CUDA extension `gdn_qsa_sm80._qsa_indexer`.
"""

from __future__ import annotations

import torch

from . import _qsa_indexer

__all__ = ["qsa_indexer", "qsa_indexer_reference"]


def qsa_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512):
    """Block-level MQA indexer for sparse-block attention.

    Args:
        q: [B, S, Hq, D] float32 — query states (MQA, Hq query heads).
        raw_keys: [B, S, D] float32 — raw key states (1 shared key head, pre-compression).
        cos_q/sin_q: [B, S, R] float32 — rotary for query token positions.
        cos_k/sin_k: [B, S, R] float32 — rotary for block start positions.
        r: compression ratio (default 4).
        block_topk: KB = token_budget // r (default 512).

    Returns:
        block_scores: [B, S, NB] float32 — block-causal score matrix (-inf invalid).
        block_indices: [B, S, KB] int32 — selected block indices (-1 padded).
        selected_scores: [B, S, KB] float32 — scores of selected blocks (-inf padded).
    """
    q, raw_keys, cos_q, sin_q, cos_k, sin_k = (
        x.contiguous() for x in (q, raw_keys, cos_q, sin_q, cos_k, sin_k)
    )
    fn = getattr(_qsa_indexer, "qsa_indexer_forward", None) or getattr(_qsa_indexer, "forward")
    return fn(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, block_topk)


def qsa_indexer_reference(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r=4, block_topk=512):
    """Pure-torch reference for the indexer (correctness anchor).

    Mirrors the kernel math: RMSNorm -> partial RoPE -> block AvgPool -> ReLU
    block score -> causal visibility -> TopK. See tests for exact usage.
    """
    # Implemented in tests (kept here so the anchor lives next to the op).
    from .reference.qsa_indexer_ref import ref_indexer
    return ref_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, block_topk)
