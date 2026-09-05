#!/usr/bin/env python
"""Phase-2 torch prototype: query-tile local K/V reuse for qsa_core pass2.

Verifies the union + per-query bitmap + streaming-tile attention math that the
CUDA reuse kernel will implement, BEFORE writing any CUDA.

Algorithm under test (matches the planned qsa_pass2_tc_reuse kernel):
  group M_TILE adjacent queries into one CTA;
    U = union of the queries' selected token sets S_q;
    stream U in 64-token tiles;
      per tile: gather K/V for U_tile ONCE;
      per query qq in tile: QK over U_tile masked by (token in S_q),
                            online-softmax update, PV accumulate.

Expected: output == reference_sparse_attention (relL1 < 1e-2).

Usage: CUDA_VISIBLE_DEVICES=0 <env-python> tools/proto_qsa_reuse.py [--S 8192] [--MT 8]
"""

import argparse
import importlib.util
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_ref(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "gdn_qsa_sm80", "reference", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_core_ref = _load_ref("qsa_core_ref")
expand_blocks_torch = _core_ref.expand_blocks_torch
make_block_idx = _core_ref.make_block_idx
reference_sparse_attention = _core_ref.reference_sparse_attention

N_TILE = 64  # token rows per streaming tile (matches CUDA kernel)


def reuse_tile_attention(q, k, v, block_idx, r, M_TILE):
    """Query-tile local-reuse attention, streamed like the planned CUDA kernel.

    q [B,S,H,D], k/v [B,S,KVH,D], block_idx [B,S,KB]. Returns out [B,S,H,D].
    """
    B, S, H, D = q.shape
    KVH = k.shape[2]
    G = H // KVH
    scale = D ** -0.5
    sel_idx, sel_cnt = expand_blocks_torch(block_idx, r)
    # GQA repeat
    k_r = k.repeat_interleave(G, dim=2)
    v_r = v.repeat_interleave(G, dim=2)

    out = torch.zeros_like(q)
    for b in range(B):
        # group adjacent queries
        for s0 in range(0, S, M_TILE):
            sqs = list(range(s0, min(s0 + M_TILE, S)))
            # union of selected token sets across the tile's queries
            sets = [set(sel_idx[b, s, : sel_cnt[b, s]].tolist()) for s in sqs]
            union = sorted(set().union(*sets))
            U = len(union)
            if U == 0:
                continue
            utab = {tok: i for i, tok in enumerate(union)}
            # per-query bitmap over union positions
            bm = torch.zeros(M_TILE, U, dtype=torch.bool, device=q.device)
            for j, s in enumerate(sqs):
                toks = sel_idx[b, s, : sel_cnt[b, s]]
                idx = torch.tensor([utab[int(t)] for t in toks], dtype=torch.long,
                                   device=q.device)
                bm[j, idx] = True

            for j, s in enumerate(sqs):
                # per-query online softmax state
                m = torch.full((H,), -torch.inf, device=q.device)
                l = torch.zeros(H, device=q.device)
                acc = torch.zeros(H, D, device=q.device)
                q_cur = q[b, s]                                   # [H,D]
                # stream the union in N_TILE-token tiles
                for t0 in range(0, U, N_TILE):
                    seg = torch.tensor(union[t0:t0 + N_TILE], dtype=torch.long,
                                       device=q.device)
                    nrows = seg.numel()
                    k_t = k_r[b, seg]                             # [n,H,D]
                    v_t = v_r[b, seg]                             # [n,H,D]
                    scores = torch.einsum("hd,nhd->hn", q_cur.float(),
                                          k_t.float()) * scale     # [H,n]
                    valid = bm[j, t0:t0 + N_TILE]                 # [n]
                    scores = scores.masked_fill(~valid, -torch.inf)
                    m_new = torch.maximum(m, scores.amax(dim=-1))  # [H]
                    corr = torch.exp(m - m_new)
                    acc = acc * corr.unsqueeze(-1)
                    p = torch.exp(scores - m_new.unsqueeze(-1))
                    p = p.masked_fill(~valid, 0.0)                 # drop -inf rows
                    l = l * corr + p.sum(dim=-1)
                    acc = acc + torch.einsum("hn,nhd->hd", p.float(), v_t.float())
                    m = m_new
                out[b, s] = acc / l.clamp_min(1e-30).unsqueeze(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--KB", type=int, default=512)
    ap.add_argument("--MT", type=int, default=8)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--KVH", type=int, default=2)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--mode", default="smooth",
                    choices=["recent", "random", "smooth"])
    args = ap.parse_args()

    torch.manual_seed(0)
    B = 1
    q = torch.randn(B, args.S, args.H, args.D, dtype=torch.bfloat16,
                    device="cuda") * 0.5
    k = torch.randn(B, args.S, args.KVH, args.D, dtype=torch.bfloat16,
                    device="cuda") * 0.5
    v = torch.randn(B, args.S, args.KVH, args.D, dtype=torch.bfloat16,
                    device="cuda") * 0.5

    if args.mode == "smooth":
        block_idx = _smooth_block_idx(B, args.S, args.KB, args.r)
    else:
        block_idx = make_block_idx(B, args.S, args.KB, args.r, "cuda",
                                   mode=args.mode)

    t0 = time.time()
    ref = reference_sparse_attention(q, k, v, block_idx, args.r)
    t_ref = time.time() - t0
    t0 = time.time()
    out = reuse_tile_attention(q, k, v, block_idx, args.r, args.MT)
    t_rt = time.time() - t0

    rel = (out.float() - ref.float()).abs().sum().item() / (
        ref.float().abs().sum().item() + 1e-9)
    print(f"S={args.S} MT={args.MT} mode={args.mode}: relL1 = {rel:.3e} "
          f"(ref {t_ref*1e3:.0f}ms / reuse {t_rt*1e3:.0f}ms)")
    ok = rel < 1e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _smooth_block_idx(B, S, KB, r):
    """Locality-mimicking indexer output: nearest KB blocks (recent-like)."""
    idx = torch.full((B, S, KB), -1, dtype=torch.int32, device="cuda")
    for s in range(S):
        cur = s // r
        k = min(KB, cur)
        if k:
            idx[0, s, :k] = torch.arange(cur - k, cur, device="cuda")
    return idx


if __name__ == "__main__":
    sys.exit(main())
