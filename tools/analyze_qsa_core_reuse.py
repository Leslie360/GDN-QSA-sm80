#!/usr/bin/env python
"""qsa_core pass2 sparse-gather reuse analysis (GPT decision Phase-1 / B1).

Quantifies how much K/V global traffic qsa_core can save by REUSING selected
blocks across adjacent queries (query-tile local reuse), WITHOUT changing any
kernel. Pure torch/numpy — block_idx [B,S,KB] is the only op-specific input.

Traffic model (matches qsa_pass2_tc_v3 gather):
    per query  gather_bytes = #sel_tokens * D * 2 (K,V) * 2B (bf16)
    per tile   gather_bytes = |union sel_tokens| * D * 2 * 2B
    saving(MT) = 1 - sum_tile|union| / sum_query|sel|
  (D, #KV, dtype cancel out of the ratio; block-level == token-level since
   every selected block contributes exactly r consecutive tokens; the expand
   "tail" (current partial block) is also ignored because it enters both the
   numerator and denominator and cancels to <0.25% — verified numerically.)

Known bias (see Phase-1 review): in `random` mode the first ~25% of queries
(where the candidate pool < KB, so "random" degenerates to "select all") inflate
the reported saving by 1.3-2.5pp; `recent` is unbiased. Neither affects the gate.

Decide gate (GPT B1): continue to a local-reuse prototype iff
    saving(M_TILE in {8,16}) > 30%.

Block-idx sources:
    --mode recent       : every query picks its nearest KB visible blocks (upper bound)
    --mode random       : every query picks KB random visible blocks (lower bound)
    --mode refindexer   : qsa_indexer reference scoring + topK on random q/raw_keys
    --mode refindexer:smooth : refindexer on locality-mimicking smooth inputs
                          (nearby keys similar => recent/neighbor blocks score high,
                           mimics a language-model latent distribution)

Usage:
    python analyze_qsa_core_reuse.py --S 8192 --mode recent
    python analyze_qsa_core_reuse.py --S 8192 --mode refindexer:smooth
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# public GDN+QSA sparse config (bench_qsa_core.py)
B, H, KVH, D, KB, R = 1, 24, 2, 256, 512, 64


# ---------------------------------------------------------------- block-idx gen
def recent_block_idx(S: int, r: int, KB: int) -> np.ndarray:
    """Every query picks its KB nearest fully-visible blocks (make_block_idx 'recent')."""
    idx = np.full((1, S, KB), -1, dtype=np.int32)
    for s in range(S):
        cur_blk = s // r
        k = min(KB, cur_blk)
        for i, blk in enumerate(range(cur_blk - k, cur_blk)):
            idx[0, s, i] = blk
    return idx


def random_block_idx(S: int, r: int, KB: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.full((1, S, KB), -1, dtype=np.int32)
    for s in range(S):
        cur_blk = s // r
        if cur_blk > 0:
            k = min(KB, cur_blk)
            idx[0, s, :k] = rng.choice(cur_blk, size=k, replace=False)
    return idx


def _load_ref():
    path = os.path.join(REPO, "gdn_qsa_sm80", "reference", "qsa_indexer_ref.py")
    spec = importlib.util.spec_from_file_location("qsa_indexer_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ref_indexer, mod.make_rope_tables


def refindexer_block_idx(S: int, r: int, KB: int, smooth: bool,
                         device: str) -> np.ndarray:
    ref_indexer, make_rope_tables = _load_ref()
    Hq, Dh, Rr = 4, 128, 64
    torch.manual_seed(0)
    s_idx = torch.arange(S, dtype=torch.float32, device=device) / S

    if smooth:
        # low-frequency per-dim sinusoids => nearby keys are similar (locality).
        freq = torch.randint(1, 12, (Dh,), device=device).float()
        phase = torch.rand(Dh, device=device) * 2.0 * math.pi
        key_sig = torch.sin(2.0 * math.pi * s_idx[:, None] * freq[None, :]
                            + phase[None, :])                          # [S,Dh]
        # q is a projection of the key at the SAME token => query correlates
        # with its own recent context (real LLM behaviour).
        q = key_sig.unsqueeze(0).repeat_interleave(Hq, dim=0) \
                 .permute(1, 0, 2).contiguous().unsqueeze(0)           # [1,S,Hq,Dh]
        q = q + torch.randn_like(q) * 0.15
        raw_keys = key_sig.unsqueeze(0) + torch.randn(1, S, Dh, device=device) * 0.05
    else:
        q = torch.randn(1, S, Hq, Dh, device=device) * 0.5
        raw_keys = torch.randn(1, S, Dh, device=device) * 0.5

    cos_q, sin_q = make_rope_tables(1, S, Rr, device=device)
    cos_k, sin_k = make_rope_tables(1, S, Rr, device=device)
    scores, idxs, sel = ref_indexer(q, raw_keys, cos_q, sin_q, cos_k, sin_k, r, KB)
    return idxs.cpu().numpy()


# ------------------------------------------------------------- reuse analysis
def analyze(block_idx: np.ndarray, r: int, tile_sizes) -> dict:
    B_, S_, KB_ = block_idx.shape
    sets = [set(int(x) for x in row if x >= 0)
            for row in block_idx.reshape(-1, KB_)]
    n_q = len(sets)

    reuse = Counter()
    for st in sets:
        reuse.update(st)

    # per-block reuse histogram
    vals = np.array(sorted(reuse.values()), dtype=np.float64)
    hist = {}
    if vals.size:
        hist.update(dict(
            n_blocks=vals.size,
            min=int(vals[0]), p25=int(np.percentile(vals, 25)),
            median=int(np.median(vals)), p75=int(np.percentile(vals, 75)),
            max=int(vals[-1]),
            frac_high = float(np.mean(vals > 1024)),   # reuse > half the queries
        ))

    # M_TILE scan: union/sum saving
    rows = []
    for MT in tile_sizes:
        sum_union = 0
        sum_sel = 0
        for t0 in range(0, n_q, MT):
            tile = sets[t0:t0 + MT]
            u = set().union(*tile) if tile else set()
            sum_union += len(u)
            sum_sel += sum(len(x) for x in tile)
        rows.append(dict(
            M_TILE=MT,
            sum_sel=sum_sel, sum_union=sum_union,
            saving=1.0 - sum_union / max(1, sum_sel),
            avg_union=sum_union / max(1, math.ceil(n_q / MT)),
        ))
    return dict(k_eff=sum(len(s) for s in sets) / n_q, reuse=hist, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--KB", type=int, default=KB)
    ap.add_argument("--mode", default="recent",
                    choices=["recent", "random", "refindexer", "refindexer:smooth"])
    ap.add_argument("--tiles", default="1,2,4,8,16,32,64")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tile_sizes = [int(x) for x in args.tiles.split(",")]

    t0 = time.time()
    if args.mode == "recent":
        idx = recent_block_idx(args.S, args.r, args.KB)
    elif args.mode == "random":
        idx = random_block_idx(args.S, args.r, args.KB)
    elif args.mode in ("refindexer", "refindexer:smooth"):
        idx = refindexer_block_idx(args.S, args.r, args.KB,
                                   smooth=args.mode.endswith("smooth"),
                                   device=args.device)
    gen_ms = (time.time() - t0) * 1e3

    res = analyze(idx, args.r, tile_sizes)
    print(f"=== qsa_core gather reuse — S={args.S} r={args.r} KB={args.KB} mode={args.mode} ===")
    print(f"block_idx gen: {gen_ms:.1f}ms | per-query valid blocks: {res['k_eff']:.1f}/{args.KB}")
    rh = res["reuse"]
    print(f"per-block reuse: n_blocks={rh.get('n_blocks')} "
          f"min={rh.get('min')} p25={rh.get('p25')} med={rh.get('median')} "
          f"p75={rh.get('p75')} max={rh.get('max')} frac>1024={rh.get('frac_high', 0):.3f}")
    print(f"{'M_TILE':>7} {'sum_sel':>12} {'sum_union':>12} {'saving%':>8} {'avg_union/tile':>14}")
    for row in res["rows"]:
        print(f"{row['M_TILE']:>7} {row['sum_sel']:>12} {row['sum_union']:>12} "
              f"{row['saving']*100:>7.1f}% {row['avg_union']:>14.1f}")
    gate = [r for r in res["rows"] if r["M_TILE"] in (8, 16)]
    if gate:
        best = max(r["saving"] for r in gate)
        print(f"\nGATE (GPT B1, M_TILE=8/16): saving = {best*100:.1f}% "
              f"-> {'CONTINUE to local-reuse prototype' if best > 0.30 else 'STOP tile-stationary'}")


if __name__ == "__main__":
    main()
