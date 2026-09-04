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


def radix_select_topk(packed, K):
    """Return indices (0..P-1 into the compacted array) of the top-K, DESC by key.

    Single-round radix + exact sort of the pivot bucket:
      1. histogram the top 8 bits (bits 63..56) of the packed key
      2. pivot b1 = first digit whose running count reaches K; cnt_gt = #(>b1),
         need = K - cnt_gt
      3. >b1 candidates go (unordered) to out; ==b1 candidates go to eq
      4. sort eq DESC (tiny for uniform data, large only in degenerate tie cases)
         and take its top `need` -> append to out
      5. out has exactly K winners, but >b1 were not mutually ordered: sort out DESC
         -> exact packed-desc total order.
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
    # sort eq desc and take its top `need`
    eq_sorted = np.sort(eq)[::-1][:need]
    out = np.concatenate([gt, eq_sorted])
    out = np.sort(out)[::-1]          # final exact order of the K winners
    lookup = {int(v): i for i, v in enumerate(packed)}
    return np.array([lookup[int(v)] for v in out[:K]], dtype=np.int64)


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
