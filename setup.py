import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_INCLUDE = os.path.join(HERE, "third_party", "cutlass", "include")

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

GDN_SOURCES = [
    os.path.join("csrc", "gdn_chunk", f) for f in [
        "gdn_kernel.cu",
        "gdn_scan_stage1.cu",
        "gdn_scan_stage1_reset.cu",
        "gdn_scan_stage2.cu",
        "gdn_scan_stage2_blelloch.cu",
        "gdn_scan_stage3_reset.cu",
        "gdn_ops.cu",
    ]
]

setup(
    ext_modules=[
        CUDAExtension(
            "gdn_qsa_sm80._gdn_chunk",
            GDN_SOURCES,
            include_dirs=[CUTLASS_INCLUDE],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
