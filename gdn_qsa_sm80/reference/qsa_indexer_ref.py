"""Pure-torch reference for the QSA block-level MQA indexer (correctness anchor)."""

import math

import torch


def make_rope_tables(B, S, R, theta=10000.0, device="cuda", dtype=torch.float32):
    """Rotary cos/sin for positions 0..S-1 with rope_dim=R."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, R, 2, dtype=dtype, device=device) / R))
    pos = torch.arange(S, dtype=dtype, device=device)
    freqs = torch.outer(pos, inv_freq)  # [S, R/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, R]
    cos = emb.cos()[None, :, :].expand(B, S, R).contiguous()
    sin = emb.sin()[None, :, :].expand(B, S, R).contiguous()
    return cos, sin


def ref_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, block_topk):
    """PyTorch reference mirroring the CUDA indexer kernel.

    q:        [B, S, Hq, D]
    raw_keys: [B, S, D]
    cos/sin:  [B, S, R]
    """
    B, S, Hq, D = q.shape
    NB = S // r
    eps = 1e-6

    # --- normalized + roped q ---
    R = cos_q.shape[-1]
    qn = q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    cos_qq = cos_q.unsqueeze(2)  # [B,S,1,R]
    sin_qq = sin_q.unsqueeze(2)
    q_rope = qn.clone()
    a = qn[..., : R // 2]
    b = qn[..., R // 2 : R]
    q_rope[..., : R // 2] = a * cos_qq[..., : R // 2] - b * sin_qq[..., : R // 2]
    q_rope[..., R // 2 : R] = b * cos_qq[..., R // 2 :] + a * sin_qq[..., R // 2 :]

    # --- pooled block keys: AvgPool r, RMSNorm, RoPE at block start ---
    pooled = raw_keys.view(B, NB, r, D).mean(dim=2)  # [B,NB,D]
    kb = pooled / torch.sqrt(pooled.pow(2).mean(-1, keepdim=True) + eps)
    block_starts = torch.arange(NB, device=q.device) * r
    k_rope = kb.clone()
    a = kb[..., : R // 2]
    b = kb[..., R // 2 : R]
    k_rope[..., : R // 2] = a * cos_k[:, block_starts, : R // 2] - b * sin_k[:, block_starts, : R // 2]
    k_rope[..., R // 2 : R] = b * cos_k[:, block_starts, R // 2 :] + a * sin_k[:, block_starts, R // 2 :]

    # --- scores: [B,S,NB], per (query,block) sum over heads ReLU(dot) ---
    per_head = torch.einsum('bshd,bnd->bshn', q_rope.float(), k_rope.float())  # [B,S,Hq,NB]
    scores = torch.relu(per_head).sum(dim=2) / math.sqrt(D)  # [B,S,NB]

    # --- causal masking: block fully visible iff p_b + r - 1 <= t ---
    p_b = torch.arange(NB, device=q.device) * r  # [NB]
    mask = (p_b + r - 1 <= torch.arange(S, device=q.device).unsqueeze(-1))  # [S,NB]
    scores = scores.masked_fill(~mask.unsqueeze(0), -float("inf"))

    # --- TopK per query over valid blocks ---
    KB = block_topk
    idxs = torch.full((B, S, KB), -1, dtype=torch.int32, device=q.device)
    sel = torch.full((B, S, KB), -float("inf"), dtype=torch.float32, device=q.device)
    for b in range(B):
        for t in range(S):
            n_valid = mask[t].sum().item()
            k = min(KB, int(n_valid))
            if k > 0:
                topv, topi = scores[b, t].topk(k)
                idxs[b, t, :k] = topi.to(torch.int32)
                sel[b, t, :k] = topv
    return scores, idxs, sel
