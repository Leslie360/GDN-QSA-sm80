# Correctness Policy

> How we prove a kernel is correct before it ships. Tests must pass on clean A800,
> with numbers reproduced from the exact command in the test file header.

## Precision floors (bf16 kernels)

- **bf16 relative error floor**: ~2e-3 for elementwise, ~1e-2 for reductions over
  long sequences is expected. Do not claim "exact" for bf16 math; claim bounded error.
- Kernel output must be within `atol=5e-1 / rtol=5e-2` of a reference for
  attention/scan outputs at production shapes (matches the production acceptance
  thresholds used for this kernel family).
- **Measured floor for gdn_chunk (bf16, S=2048–32768, decay g<0)**: out `rel ~9–11e-3`,
  state `rel ~7–10e-3`, stable in S (bounded by decayed accumulation). Tests accept
  `rel < 2e-2` — a real kernel bug produces `rel >= 1e-1`+, so the margin is safe.

## Metrics and when to use them

| Metric | Definition | Use when |
|---|---|---|
| `max_abs` | `max|ours - ref|` | fp32/fp64 truth, small shapes, bit-adjacent checks |
| `max_rel` | `max(|d| / max(|ref|, eps))` | non-normalized magnitudes |
| `corr` | Pearson correlation | high-level agreement, coarse |
| `ok%` | fraction of elements within tol | broadcast large-tensor sanity |

- **fp64 truth**: use when the operation accumulates (scans, softmax, reductions) and
  bf16 reference is not trustworthy. Cheap for small/medium shapes.
- **bit-identical**: required only for *plumbing* — e.g. refactorings that must not
  change results (`max|diff| == 0`), or when the kernel is a drop-in for an existing
  golden. Never promised for bf16 arithmetic itself.

## Reference strategy

1. **Self-written torch/numpy reference** (`gdn_qsa_sm80/reference/`) — the primary anchor.
2. **Public-baseline cross-check** (e.g. fla) — validates against an independent impl,
   same host/dtype. Report `max_abs` vs the baseline explicitly.
3. **gradcheck** — required for any op exposed through `autograd.Function`.

## Gate before bench

`benchmarks/verify.py` runs the correctness suite for the target op and refuses to
time anything that does not pass. A kernel is not benchmarked unless it is correct.

## Test layout

- `tests/test_<op>.py` — one file per operator, `pytest`.
- `tests/test_util.py` — shared compare helpers (max_abs / max_rel / corr / ok%) +
  reference constructors.
- Head of each test file states the exact command to reproduce (env, shape, dtype).
