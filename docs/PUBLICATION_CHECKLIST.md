# Publication Checklist

> Run before every public release / commit that touches the tree. Nothing internal
> leaks into the open repo.

## Scan for leaks

- [ ] **No absolute internal mount paths** — grep for `<internal-mount>`,
      team/personal directory names; repo must contain only repo-relative
      references.
- [ ] **No internal IPs** — grep for private-net octets, cloud-box hostnames.
- [ ] **No internal model codenames** — README references only the public
      GDN+QSA architecture family and public tech reports, never a private name.
- [ ] **No training-machine names** — no bench/train box identifiers anywhere.
- [ ] **No unpublished benchmark source** — every number links to a runnable script
      and this repo's methodology; nothing cites internal only.
- [ ] **No agent/debug voice** — no "I tried", "fixed by", "TODO experiment" notes
      copied from internal docs; no `debug*.py` leftovers in `csrc/`/`gdn_qsa_sm80/`.
- [ ] **No personal paths** (`/root`, `$HOME`, `/dev/shm/triton_cache` baked in).
- [ ] **No private dependencies** — imports resolve only to stdlib / public PyPI
      (`torch`, `triton`, `numpy`, `pytest`); nothing from internal repos.
- [ ] `setup.py` / `csrc` has no leftover CUTLASS path pointing outside
      `third_party/cutlass/`.

## Structure

- [ ] LICENSE present (Apache-2.0) and referenced from README.
- [ ] `third_party/` retains upstream LICENSE/NOTICE for vendored code.
- [ ] `.gitignore` excludes `build/`, `*.so`, `__pycache__`, `.ninja_*`.
- [ ] No compiled artifacts (`*.so`, `*.o`) tracked.

## Verification before commit

- [ ] `pip install -e .` clean on a fresh venv.
- [ ] `python -m pytest tests/ -x` all PASS on clean A800.
- [ ] `bash scripts/verify_all.sh` PASS; `bash scripts/bench_all.sh` produced the
      README table verbatim.
- [ ] `git diff` review — confirm only intended files.

## On release

- [ ] Tag with version; README table dates the measurements + states the exact GPU.
- [ ] AUTHORS / CONTRIBUTING optional but recommended.
