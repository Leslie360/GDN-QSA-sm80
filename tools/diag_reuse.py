#!/usr/bin/env python
"""Diagnose qsa_pass2_tc_reuse correctness: where does it diverge from v3?"""

import torch

from gdn_qsa_sm80 import qsa_expand, qsa_pass2_tc, qsa_pass2_tc_reuse
from gdn_qsa_sm80.reference.qsa_core_ref import make_block_idx


def main():
    H, KVH, D, KB, r = 24, 2, 256, 512, 4
    S = 512
    torch.manual_seed(0)
    q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
    block_idx = make_block_idx(1, S, KB, r, "cuda", mode="recent")
    sel_idx, sel_cnt = qsa_expand(block_idx, r)
    v3 = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)
    re = qsa_pass2_tc_reuse(q, k, v, sel_idx, sel_cnt, r)

    print("NaN reuse:", torch.isnan(re).any().item(), "NaN v3:", torch.isnan(v3).any().item())
    diff = (re.float() - v3.float()).abs()
    print("max|diff| per-s:", diff.amax(dim=(2, 3))[:16].cpu().tolist())
    # per-head max diff pattern
    print("max|diff| per (s,head) first 4 heads of s=100..103:")
    d = diff[0, 100:104]
    print(d.amax(dim=-1).cpu().tolist())
    # norm of outputs per query
    print("|v3| per-s first 8:", v3.float().norm(dim=(2, 3))[0, :8].cpu().tolist())
    print("|re| per-s first 8:", re.float().norm(dim=(2, 3))[0, :8].cpu().tolist())
    # which s are wrong (first 32)
    bad = (diff.amax(dim=(2, 3))[0] > 0.05)
    print("wrong s:", bad[:64].cpu().tolist())
    print("frac wrong:", bad.float().mean().item())
    # check a specific (s,h) row
    s, h = 100, 0
    print("v3[0,100,0,:8]", v3[0, s, h, :8].cpu().tolist())
    print("re[0,100,0,:8]", re[0, s, h, :8].cpu().tolist())


if __name__ == "__main__":
    main()
