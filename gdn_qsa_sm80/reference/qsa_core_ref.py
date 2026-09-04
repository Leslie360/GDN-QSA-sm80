"""Pure-torch references for the QSA sparse-core attention (correctness anchor)."""

import torch


def expand_blocks_torch(block_idx, r):
    """Reference block expansion (matches kernel pass 1)."""
    B, S, KB = block_idx.shape
    NMAX = KB * r + r
    sel_idx = torch.full((B, S, NMAX), -1, dtype=torch.int64, device=block_idx.device)
    sel_cnt = torch.zeros(B, S, dtype=torch.int64, device=block_idx.device)
    for b in range(B):
        for s in range(S):
            n = 0
            for kk in range(KB):
                bk = block_idx[b, s, kk].item()
                if bk < 0:
                    continue
                start = bk * r
                for t in range(r):
                    if n < NMAX:
                        sel_idx[b, s, n] = start + t
                        n += 1
            tail_start = (s // r) * r
            for t in range(tail_start, s + 1):
                if n < NMAX:
                    sel_idx[b, s, n] = t
                    n += 1
            sel_cnt[b, s] = n
    return sel_idx, sel_cnt


def reference_sparse_attention(q, k, v, block_idx, r):
    """Torch reference: sparse core attention over selected KV with causal mask."""
    B, S, H, D = q.shape
    KVH = k.shape[2]
    G = H // KVH
    scale = D ** -0.5
    sel_idx, sel_cnt = expand_blocks_torch(block_idx, r)

    # repeat kv: k/v [B,S,KVH,D] -> [B,S,H,D]
    k_r = k.repeat_interleave(G, dim=2)
    v_r = v.repeat_interleave(G, dim=2)

    out = torch.zeros_like(q)
    for b in range(B):
        for s in range(S):
            n = sel_cnt[b, s].item()
            idx = sel_idx[b, s, :n].to(torch.long)
            q_cur = q[b, s]                      # [H, D]
            k_sel = k_r[b, idx, :]               # [n, H, D]
            v_sel = v_r[b, idx, :]               # [n, H, D]
            scores = torch.einsum("hd,nhd->hn", q_cur.float(), k_sel.float()) * scale  # [H,n]
            causal = idx.unsqueeze(0) <= s
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(q.dtype)  # [H,n]
            acc = torch.einsum("hn,nhd->hd", probs.float(), v_sel.float())
            out[b, s] = acc
    return out


def make_block_idx(B, S, KB, r, device, mode="recent"):
    """Deterministic block index tensor (recent or random) for testing."""
    block_idx = torch.full((B, S, KB), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for s in range(S):
            cur_blk = s // r
            cand = list(range(cur_blk))
            kk = min(KB, len(cand))
            if mode == "recent":
                picks = cand[-kk:] if kk else []
            else:
                picks = (torch.randperm(max(1, len(cand)))[:kk].tolist() if cand else [])
            for i, blk in enumerate(picks):
                block_idx[b, s, i] = blk
    return block_idx
