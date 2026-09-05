#!/usr/bin/env python
"""Correctness + quick perf check for qsa_pass2_tc_reuse vs scalar/v3.

Reproduce (cloud): cd <repo> && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
    <env>/bin/python tools/verify_reuse.py
"""

import time

import torch

from gdn_qsa_sm80 import (
    qsa_expand,
    qsa_pass2_tc,
    qsa_pass2_tc_reuse,
    qsa_sparse_core_attention,
)
from gdn_qsa_sm80.reference.qsa_core_ref import make_block_idx


def rel_global(out, ref):
    num = (out.float() - ref.float()).abs().sum().item()
    den = ref.float().abs().sum().item() + 1e-9
    return num / den


def _timed(fn, *a, n=10, warmup=3):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    H, KVH, D, KB, r = 24, 2, 256, 512, 4
    ok = True
    for S in (512, 2048, 8192):
        for mode in ("recent", "random"):
            torch.manual_seed(0)
            q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
            k = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
            v = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
            block_idx = make_block_idx(1, S, KB, r, "cuda", mode=mode)
            sel_idx, sel_cnt = qsa_expand(block_idx, r)
            sc = qsa_sparse_core_attention(q, k, v, block_idx, r)
            re = qsa_pass2_tc_reuse(q, k, v, sel_idx, sel_cnt, r)
            v3 = qsa_pass2_tc(q, k, v, sel_idx, sel_cnt, r)
            r_re = rel_global(re, sc)
            r_v3 = rel_global(v3, sc)
            tag = "OK " if (r_re < 1e-2 and r_v3 < 1e-2) else "FAIL"
            ok &= (r_re < 1e-2 and r_v3 < 1e-2)
            print(f"{tag} S={S:5d} {mode:6s} reuse-vs-scalar {r_re:.2e}  v3-vs-scalar {r_v3:.2e}")
    print("--- perf (S=8192 recent) ---")
    S = 8192
    torch.manual_seed(0)
    q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
    v = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
    block_idx = make_block_idx(1, S, KB, r, "cuda", mode="recent")
    sel_idx, sel_cnt = qsa_expand(block_idx, r)
    t_v3 = _timed(qsa_pass2_tc, q, k, v, sel_idx, sel_cnt, r, n=20, warmup=5)
    t_re = _timed(qsa_pass2_tc_reuse, q, k, v, sel_idx, sel_cnt, r, n=20, warmup=5)
    print(f"S=8192 recent:  v3 {t_v3:7.3f}ms   reuse {t_re:7.3f}ms   reuse/v3 {t_re/t_v3:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
