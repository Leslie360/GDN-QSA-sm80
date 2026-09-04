# third_party

## CUTLASS / CuTe (vendored headers)

Only the header-only portions of [CUTLASS](https://github.com/NVIDIA/cutlass)
(`include/cute` + `include/cutlass`, the subset needed by the kernels here)
are vendored under `cutlass/`.

- **License: BSD-3-Clause** (see `cutlass/LICENSE`, NVIDIA CUTLASS license).
  Upstream license/NOTICE retained verbatim — do not rename or relicense.
- Populated with the first CUDA operator (M1); grows only if a kernel needs
  more headers.
