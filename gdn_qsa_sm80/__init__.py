"""gdn-qsa-sm80: from-scratch SM80 (A800) CUDA operators for Gated DeltaNet + QSA."""

from .gdn_chunk_interface import (
    gdn_chunk,
    gdn_chunk_reference,
    gdn_chunk_twolevel,
)

__version__ = "0.1.0"
__all__ = ["gdn_chunk", "gdn_chunk_twolevel", "gdn_chunk_reference"]
