"""gdn-qsa-sm80: from-scratch SM80 (A800) CUDA operators for Gated DeltaNet + QSA."""

from .gdn_chunk_interface import (
    gdn_chunk,
    gdn_chunk_reference,
    gdn_chunk_twolevel,
)
from .output_gate_interface import (
    out_proj_gemm,
    out_proj_gemm_cutlass,
    rmsnorm_gated,
)
from .qsa_core_interface import (
    qsa_expand,
    qsa_pass2_tc,
    qsa_sparse_core_attention,
)
from .qsa_indexer_interface import (
    qsa_indexer,
    qsa_indexer_reference,
)

__version__ = "0.1.0"
__all__ = [
    # GDN
    "gdn_chunk",
    "gdn_chunk_twolevel",
    "gdn_chunk_reference",
    # QSA indexer
    "qsa_indexer",
    "qsa_indexer_reference",
    # QSA sparse-core attention
    "qsa_sparse_core_attention",
    "qsa_expand",
    "qsa_pass2_tc",
    # output gate
    "rmsnorm_gated",
    "out_proj_gemm",
    "out_proj_gemm_cutlass",
]
