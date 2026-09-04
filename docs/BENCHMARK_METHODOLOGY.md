# Benchmark Methodology

> Reproducible performance numbers are the credibility of this repo. Every number
> in the README must trace back to this methodology and a runnable command.

## Hard rules

1. **Same input** — both kernels receive identical tensors (same dtype, shape, values).
2. **Same dtype** — bf16/fp16 must be identical across comparison; no hidden upcast for the baseline.
3. **Same head layout** — expanded heads / head ratios must match on both sides (a fair baseline must not do extra `repeat_interleave` work — see team's `repeat-trap` note).
4. **Same GPU** — comparisons happen on the *same physical device*, never cross-machine.
5. **Clean GPU** — measure on an idle device. A shared/busy GPU inflates numbers ~2x. For A800, use a dedicated clean device (GPU0 of the bench machine).
6. **Warmup + median** — warm up, then take the median (not min/max) over N runs.
7. **Report memory + speed** — both peak memory and latency/throughput.
8. **Exact command** — every table row has the exact reproduction command (script + args).

## Baseline discipline

- Public baselines only (e.g. [fla](https://github.com/fla-org/flash-linear-attention)) built from source, same host, same dtype.
- If a fair baseline is not available for some input, say so explicitly instead of claiming a win.
- Report `ours`, `baseline`, and `speedup` columns; never fold the baseline number into prose only.

## Measurement

- Use CUDA events (or torch profiler) for wall-clock of the kernel region only.
- `torch.cuda.synchronize()` before timing.
- Warmup ≥ 5 iters, measure ≥ 50 iters, report median.
- For memory: `torch.cuda.max_memory_allocated()` delta across the kernel region.

## Files

- `benchmarks/registry.py` — operator registration
- `benchmarks/run.py` — unified entry: `python -m benchmarks.run <op>`
- `benchmarks/verify.py` — correctness gate before any timing
- `scripts/bench_all.sh` — full matrix, writes a markdown table

## Output format

Every benchmark run prints a table row:

```
op | shape | dtype | ours(ms) | baseline(ms) | speedup | peak_mem_ours | peak_mem_base | command
```

Numbers in the README are copied verbatim from `scripts/bench_all.sh` output on a clean A800.
