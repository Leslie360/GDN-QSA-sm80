import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_INCLUDE = os.path.join(HERE, "third_party", "cutlass", "include")

# Select which operators to build (all by default).
#   GDN_QSA_BUILD_OPS=gdn_chunk,qsa_indexer,output_gate,qsa_core
#   GDN_QSA_BUILD_GDN_ONLY=1   # legacy alias for building gdn_chunk alone
_build_ops = os.getenv("GDN_QSA_BUILD_OPS", "gdn_chunk,qsa_indexer,output_gate,qsa_core,qsa_pass2_tc")
if os.getenv("GDN_QSA_BUILD_GDN_ONLY"):
    _build_ops = "gdn_chunk"
_ops = {o.strip() for o in _build_ops.split(",") if o.strip()}

EXTRA_COMPILE_ARGS = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-lineinfo",
        "-std=c++17",
        "-arch=sm_80",
        f"-I{CUTLASS_INCLUDE}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--diag-suppress=1301",  # unused variable warnings in cute
    ],
}


def _cu(*names):
    return [os.path.join("csrc", "gdn_chunk", n) for n in names]


def _qsa_idx(*names):
    return [os.path.join("csrc", "qsa_indexer", n) for n in names]


def _og(*names):
    return [os.path.join("csrc", "output_gate", n) for n in names]


def _qc(*names):
    return [os.path.join("csrc", "qsa_core", n) for n in names]


ext_modules = []

# gdn_chunk — Gated DeltaNet chunked linear attention (cute)
if "gdn_chunk" in _ops:
    ext_modules.append(
        CUDAExtension(
            "gdn_qsa_sm80._gdn_chunk",
            _cu("gdn_kernel.cu", "gdn_scan_stage1.cu", "gdn_scan_stage1_reset.cu",
                "gdn_scan_stage2.cu", "gdn_scan_stage2_blelloch.cu",
                "gdn_scan_stage3_reset.cu", "gdn_ops.cu"),
            include_dirs=[CUTLASS_INCLUDE],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        )
    )

# qsa_indexer — QSA block-level MQA indexer (no cutlass)
if "qsa_indexer" in _ops:
    ext_modules.append(
        CUDAExtension(
            "gdn_qsa_sm80._qsa_indexer",
            _qsa_idx("qsa_indexer_kernel.cu", "bindings.cpp"),
            include_dirs=[os.path.join(HERE, "csrc", "qsa_indexer")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_80"],
            },
        )
    )

# output_gate — RMSNormGated + out_proj (self-written + CUTLASS GEMM)
if "output_gate" in _ops:
    ext_modules.append(
        CUDAExtension(
            "gdn_qsa_sm80._output_gate",
            _og("output_gate.cu", "output_gate_cutlass.cu"),
            include_dirs=[CUTLASS_INCLUDE],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "-gencode", "arch=compute_80,code=sm_80"],
            },
        )
    )

# qsa_core — scalar sparse-core attention (all dtypes) + bindings
if "qsa_core" in _ops:
    ext_modules.append(
        CUDAExtension(
            "gdn_qsa_sm80._qsa_core",
            _qc("qsa_core.cu", "qsa_core_bind.cpp"),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_80"],
            },
        )
    )

# qsa_pass2_tc — TC-accelerated pass2 (v3, bf16 D=256) + query-tile reuse
if "qsa_pass2_tc" in _ops:
    ext_modules.append(
        CUDAExtension(
            "gdn_qsa_sm80._qsa_pass2_tc",
            _qc("qsa_pass2_tc_v3.cu", "qsa_pass2_tc_reuse.cu"),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_80"],
            },
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
