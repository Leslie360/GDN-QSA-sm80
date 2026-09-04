"""CPU prototype for a 2-round radix-select topk (R1: replace bitonic in kernel D).

Validates the exact semantics kernel D must produce:
  - packed uint64: [63:32] sortable score (monotone fp32 key), [31:0] reversed idx
  - descending by packed key  =>  higher score first; equal score -> lower idx first
  - only the first P (valid) slots are considered; remaining padded -inf/-1

The kernel replacement plan:
  Pass A: histogram top 8 bits (bits 63..56 of packed) -> find pivot bucket b1,
          cnt_gt = #(> b1), need = K - cnt_gt.
  Pass B: collect >b1 into out (unordered) + ==b1 into tmp.
  Pass C: histogram bits 55..48 of tmp -> find b2; collect >b2 into out, ==b2 into tmp2.
  Pass D: sort tmp2 (tiny) desc, take top need2 -> append to out.  Out is now exactly
          the top-K set, but NOT globally sorted: a final sort of the ~K winners
          (bitonic on K, not P2) yields the ordered result.
  Final: bitonic sort the K winners -> block_indices / selected_scores (descending).

Why this beats the current bitonic: current sorts P2 (~1024) elements to pick 512;
here we sort only ~K (512) winners after an O(P) radix pass.
"""

import numpy as np
import torch


def to_ordered(f):
    """fp32 -> monotone uint32 key (matches FlashInfer ToOrdered + kernel D)."""
    b = f.view(np.uint32)
    return np.where(b & 0x80000000, ~b, b ^ 0x80000000).astype(np.uint64)


def pack(score, idx):
    hi = to_ordered(np.float32(score))
    lo = np.uint64(0xFFFFFFFF) - np.uint64(idx)
    return (hi << 32) | lo


def msd_radix_sort_desc(arr):
    """In-place MSD radix sort (8 bits/pass) on an integer array, DESCENDING.

    The CUDA kernel sorts the ~K collected winners with exactly this scheme:
    per pass, histogram the current byte, exclusive-prefix-scan the 256 buckets,
    and scatter each element to base_desc[d] = tot - prefix[d+1] (descending),
    keeping a running per-bucket counter for stability.  8 passes over the 64-bit
    key => the output is the exact packed-desc total order (score desc, then
    reversed-index desc == lowest block index first).  O(K*8) vs bitonic O(K log^2
    K) — this is the CUB-style multi-pass radix that replaces the winner sort.
    """
    arr = arr.astype(np.uint64).copy()
    n = len(arr)
    tmp = np.empty_like(arr)
    src, dst = arr, tmp
    for p in range(8):
        shift = 8 * p          # LSB -> MSB (LSD stable counting sort)
        d = (src >> shift).astype(np.int64) & 0xFF
        hist = np.bincount(d, minlength=256)
        prefix = np.zeros(257, dtype=np.int64)   # prefix[d] = #(digit < d); prefix[256]=n
        s = 0
        for b in range(256):
            prefix[b] = s
            s += int(hist[b])
        prefix[256] = s
        run = np.zeros(256, dtype=np.int64)
        for i in range(n):
            dd = int(d[i])
            pos = (n - int(prefix[dd + 1])) + int(run[dd])  # descending block base
            dst[pos] = src[i]
            run[dd] += 1
        src, dst = dst, src
    return src


def radix_select_topk(packed, K):
    """Return indices (0..P-1 into the compacted array) of the top-K, DESC by key.

    Single-round radix to collect the ~K winners, then MSD radix sort (instead of
    bitonic) to order them — this is the current kernel D structure with the
    winner sort swapped from bitonic to multi-pass radix.
    """
    n = len(packed)
    K = min(K, n)
    if K <= 0:
        return np.zeros(0, dtype=np.int64)
    d1 = (packed >> 56).astype(np.int64)
    hist1 = np.bincount(d1, minlength=256)
    cnt_gt = 0
    b1 = 0
    need = K
    for b in range(255, -1, -1):
        c = int(hist1[b])
        if cnt_gt + c >= K:
            b1 = b
            need = K - cnt_gt
            break
        cnt_gt += c

    gt = packed[d1 > b1]
    eq = packed[d1 == b1]
    eq_sorted = msd_radix_sort_desc(eq)[:need]
    out = msd_radix_sort_desc(np.concatenate([gt, eq_sorted]))[:K]
    lookup = {int(v): i for i, v in enumerate(packed)}
    return np.array([lookup[int(v)] for v in out], dtype=np.int64)


def reference_topk(packed, K):
    """Ground truth: exact packed-desc order.  Because the packed key encodes
    (sortable score << 32) | reversed_idx, descending is a TOTAL order with a
    deterministic tie-break (equal score -> lower idx first), which is exactly
    what kernel D must reproduce.  Returns the compact indices in desc order."""
    K = min(K, len(packed))
    if K <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(packed)[::-1][:K]
    return order.astype(np.int64)


def main():
    rng = np.random.default_rng(0)
    ok = 0
    trials = 0
    for trial in range(5000):
        n = int(rng.integers(1, 300))
        K = int(rng.integers(1, n + 1))
        # mix of normal, uniform, heavy-tie, and constant scores
        mode = trial % 5
        if mode == 0:
            scores = rng.normal(0, 1, n)
        elif mode == 1:
            scores = rng.uniform(-2, 2, n)
        elif mode == 2:  # many exact ties
            scores = rng.integers(0, 5, n).astype(float)
        elif mode == 3:  # constant + noise
            scores = np.full(n, 1.0) + rng.normal(0, 0.01, n)
        else:  # all identical (radix worst case)
            scores = np.full(n, 3.0)

        packed = pack(scores, np.arange(n))
        r_sel = radix_select_topk(packed, K)
        r_ref = reference_topk(packed, K)  # exact total order
        trials += 1
        # radix must reproduce the exact desc total order (indices), not just a set
        if not np.array_equal(r_sel, r_ref):
            ok += 1
            if ok <= 5:
                print(f"[FAIL] trial {trial}: n={n} K={K} mode={mode}")
                print("  radix:", r_sel.tolist())
                print("  ref  :", r_ref.tolist())
                print("  scores:", scores)
    print(f"trials={trials} failures={ok}")


if __name__ == "__main__":
    main()
