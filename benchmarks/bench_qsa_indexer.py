"""Fair benchmark for qsa_indexer vs a vectorized eager baseline.

The eager baseline computes the full indexer math (RMSNorm + partial RoPE +
block AvgPool + ReLU score + causal mask + TopK) with vectorized torch ops
(no Python loops), so it is the same work as the kernel.

Reproduce on a clean A800:
    python benchmarks/bench_qsa_indexer.py
"""

import math
import time

import torch

from gdn_qsa_sm80 import qsa_indexer, qsa_indexer_topk_only
from gdn_qsa_sm80.reference.qsa_indexer_ref import make_rope_tables


def _timed(fn, *a, n=20, warmup=3, **kw):
    for _ in range(warmup):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def _eager_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, block_topk):
    """Vectorized eager baseline mirroring the kernel math."""
    B, S, Hq, D = q.shape
    NB = S // r
    eps = 1e-6
    R = cos_q.shape[-1]

    qn = q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    cos_qq, sin_qq = cos_q.unsqueeze(2), sin_q.unsqueeze(2)
    a, b = qn[..., :R // 2], qn[..., R // 2:R]
    qr = qn.clone()
    qr[..., :R // 2] = a * cos_qq[..., :R // 2] - b * sin_qq[..., :R // 2]
    qr[..., R // 2:R] = b * cos_qq[..., R // 2:] + a * sin_qq[..., R // 2:]

    pooled = raw_keys.view(B, NB, r, D).mean(2)
    kb = pooled / torch.sqrt(pooled.pow(2).mean(-1, keepdim=True) + eps)
    bs = torch.arange(NB, device=q.device) * r
    a, b = kb[..., :R // 2], kb[..., R // 2:R]
    kr = kb.clone()
    kr[..., :R // 2] = a * cos_k[:, bs, :R // 2] - b * sin_k[:, bs, :R // 2]
    kr[..., R // 2:R] = b * cos_k[:, bs, R // 2:] + a * sin_k[:, bs, R // 2:]

    scores = torch.relu(torch.einsum('bshd,bnd->bshn', qr, kr)).sum(2) / math.sqrt(D)
    mask = (torch.arange(NB, device=q.device) * r + r - 1 <=
            torch.arange(S, device=q.device).unsqueeze(-1))
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # vectorized topk over valid blocks
    n_valid = mask.sum(-1)  # [S]
    k = torch.clamp(n_valid, max=block_topk)  # [S]
    maxk = int(k.max().item())
    topv, topi = scores.topk(maxk, dim=-1)  # [B,S,maxk]
    valid = torch.arange(maxk, device=q.device)[None, None, :] < k[None, :, None]  # [1,S,maxk]
    idxs = torch.full((B, S, block_topk), -1, dtype=torch.int32, device=q.device)
    sel = torch.full((B, S, block_topk), float("-inf"), dtype=torch.float32, device=q.device)
    idxs[..., :maxk] = torch.where(valid, topi.to(torch.int32), -1)
    sel[..., :maxk] = torch.where(valid, topv, float("-inf"))
    return scores, idxs, sel


def main():
    Hq, D, R, r, KB = 4, 128, 64, 4, 512
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("S        ours(ms)  fused(ms)  eager(ms)  vs-full   vs-eager(fused)")
    for S in (512, 2048, 8192):
        torch.manual_seed(0)
        q = torch.randn(1, S, Hq, D, dtype=torch.float32, device="cuda") * 0.5
        raw_keys = torch.randn(1, S, D, dtype=torch.float32, device="cuda") * 0.5
        cos_q, sin_q = make_rope_tables(1, S, R, device="cuda")
        cos_k, sin_k = make_rope_tables(1, S, R, device="cuda")

        ours = _timed(qsa_indexer, q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
        fused = _timed(qsa_indexer_topk_only, q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
        eager_ms = _timed(_eager_indexer, q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
        print(f"{S:>6}  {ours:8.3f}  {fused:8.3f}  {eager_ms:8.3f}  "
              f"{ours/fused:7.2f}x  {eager_ms/fused:10.2f}x")


if __name__ == "__main__":
    main()
