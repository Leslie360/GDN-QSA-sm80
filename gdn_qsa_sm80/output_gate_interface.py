"""Public Python API for the output_gate operator (RMSNormGated + out_proj).

Wraps the compiled CUDA extension `gdn_qsa_sm80._output_gate`.
"""

from __future__ import annotations

import torch

from . import _output_gate

__all__ = ["rmsnorm_gated", "out_proj_gemm", "out_proj_gemm_cutlass"]


def rmsnorm_gated(y, z, weight):
    """out[i] = RMSNorm(y)[i] * weight[i] * silu(z[i]), per 128-wide row.

    Args:
        y: [N, 128] bf16 — input to normalize.
        z: [N, 128] bf16 — gate input.
        weight: [128] bf16 — per-dim scale.

    Returns:
        [N, 128] bf16.
    """
    y, z, weight = (x.contiguous() for x in (y, z, weight))
    return _output_gate.rmsnorm_gated(y, z, weight)


def out_proj_gemm(A, W):
    """C = A @ W^T (self-written SM80 tensor-core GEMM, bf16).

    Args:
        A: [M, K] bf16 row-major.
        W: [N, K] bf16 row-major (Linear weight layout [N, K]).

    Returns:
        [M, N] bf16.
    """
    A, W = (x.contiguous() for x in (A, W))
    return _output_gate.out_proj_gemm(A, W)


def out_proj_gemm_cutlass(A, W):
    """C = A @ W^T via CUTLASS device::Gemm (bf16, production path).

    Same shapes as `out_proj_gemm`; CUTLASS-based, typically ~3x faster than the
    self-written GEMM and on par with cuBLASLt.
    """
    A, W = (x.contiguous() for x in (A, W))
    return _output_gate.out_proj_gemm_cutlass(A, W)
