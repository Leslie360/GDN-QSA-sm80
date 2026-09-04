"""Fair benchmark for gdn_chunk vs public fla baseline (if installed).

Reproduce on a clean A800:
    python benchmarks/bench_gdn_chunk.py
"""

import time

import torch

from gdn_qsa_sm80 import gdn_chunk

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdn
    _HAVE_FLA = True
except Exception:
    _HAVE_FLA = False


def _gen(S, gs=2.0, Hk=16, Hv=32, seed=0):
    torch.manual_seed(seed)
    sc = 0.05
    q = (torch.randn(1, S, Hk, 128, dtype=torch.bfloat16, device="cuda") * sc)
    k = (torch.randn(1, S, Hk, 128, dtype=torch.bfloat16, device="cuda") * sc)
    v = (torch.randn(1, S, Hv, 128, dtype=torch.bfloat16, device="cuda") * sc)
    g = -torch.rand(1, S, Hv, dtype=torch.bfloat16, device="cuda") * gs
    beta = torch.rand(1, S, Hv, dtype=torch.bfloat16, device="cuda").sigmoid()
    return q, k, v, g, beta


def _timed(fn, *a, n=30, warmup=5, **kw):
    for _ in range(warmup):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3  # ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"fla baseline: {'available' if _HAVE_FLA else 'NOT installed (bench ours only)'}")
    print()
    print("S        ours(ms)  fla(ms)   speedup   peak_mem(ours, MB)")
    print("------   --------  --------  --------   ------------------")
    for S in (2048, 4096, 8192, 32768):
        q, k, v, g, beta = _gen(S, 2.0)
        q32 = q.repeat_interleave(2, dim=2)
        k32 = k.repeat_interleave(2, dim=2)
        torch.cuda.reset_peak_memory_stats()
        ours = _timed(gdn_chunk, q, k, v, g, beta, True)
        mem = torch.cuda.max_memory_allocated() / 1e6
        if _HAVE_FLA:
            fla_ms = _timed(fla_gdn, q32, k32, v, g, beta, chunk_size=64, n=20)
            speed = fla_ms / ours
            print(f"{S:>6}  {ours:8.3f}  {fla_ms:8.3f}  {speed:7.2f}x  {mem:9.1f}")
        else:
            print(f"{S:>6}  {ours:8.3f}      n/a        n/a    {mem:9.1f}")


if __name__ == "__main__":
    main()
