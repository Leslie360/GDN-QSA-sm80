"""Fair benchmark for qsa_core: scalar sparse-core attention vs TC pass2 (v3).

Reproduce on a clean A800:
    python benchmarks/bench_qsa_core.py
"""

import time

import torch

from gdn_qsa_sm80 import qsa_expand, qsa_pass2_tc, qsa_sparse_core_attention
from gdn_qsa_sm80.reference.qsa_core_ref import make_block_idx


def _timed(fn, *a, n=10, warmup=3, **kw):
    for _ in range(warmup):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    H, KVH, D, KB, r = 24, 2, 256, 512, 4  # public GDN+QSA sparse config
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("S        scalar(ms)  TC-pass2(ms)  speedup(TC vs scalar)")
    for S in (512, 2048, 8192):
        torch.manual_seed(0)
        q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
        k = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
        v = torch.randn(1, S, KVH, D, dtype=torch.bfloat16, device="cuda") * 0.5
        block_idx = make_block_idx(1, S, KB, r, "cuda", mode="recent")
        sel_idx, sel_cnt = qsa_expand(block_idx, r)

        sc_ms = _timed(qsa_sparse_core_attention, q, k, v, block_idx, r)
        tc_ms = _timed(qsa_pass2_tc, q, k, v, sel_idx, sel_cnt, r)
        print(f"{S:>6}  {sc_ms:10.2f}  {tc_ms:12.2f}  {sc_ms/tc_ms:7.2f}x")


if __name__ == "__main__":
    main()
