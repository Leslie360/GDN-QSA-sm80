// output_gate CUTLASS out_proj GEMM baseline.
//
// out_proj_gemm_cutlass(A, W)
//   C[M,N] = A[M,K] * W[N,K]^T   (bf16, tensor-op fp32 accumulate)
//   A: [M,K] bf16 row-major, W: [N,K] bf16 row-major (Linear weight [N,K])
//   C: [M,N] bf16 row-major
//
// Implemented with cutlass::gemm::device::Gemm on SM80 (16x8x16 bf16 MMA).
// Uses CUTLASS device::Gemm rather than cuBLASLt/cuBLAS (large-K GEMMs show
// out-of-bounds reads on some cuBLASLt releases).
//
// Layout trick for W^T without a physical transpose:
//   CUTLASS Gemm computes  C[M,N] = A[M,K] * B[K,N].
//   We need B = W^T  ->  B[k][n] = W[n][k].
//   W is stored row-major [N,K]: W[n][k] lives at offset n*K + k.
//   We describe B as a column-major [K,N] matrix, whose element B[k][n] lives
//   at offset k + n*ldb. For B[k][n] to alias W[n][k] we need k + n*ldb = n*K + k,
//   i.e. ldb = K. So pass B with cutlass::layout::ColumnMajor and ldb = K,
//   pointer = W. No copy needed.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace output_gate_cutlass_detail {

using Element = cutlass::bfloat16_t;
using Accumulator = float;

// Threadblock tile: 128(M) x 128(N) x 32(K). Warp tile 64x64x32.
// Instruction: SM80 16x8x16 bf16 tensor-core MMA (OpClassTensorOp).
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using Gemm = cutlass::gemm::device::Gemm<
    Element,                       // A element
    cutlass::layout::RowMajor,     // A layout: [M,K] row-major
    Element,                       // B element
    cutlass::layout::ColumnMajor,  // B layout: [K,N] column-major == W row-major [N,K]
    Element,                       // C element
    cutlass::layout::RowMajor,     // C layout: [M,N] row-major
    Accumulator,                   // fp32 accumulate
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<Element, 1, Accumulator, Accumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;                            // 3-stage pipeline

torch::Tensor out_proj_gemm_cutlass(torch::Tensor A, torch::Tensor W)
{
    TORCH_CHECK(A.is_cuda() && W.is_cuda(), "A and W must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(W.scalar_type() == torch::kBFloat16, "W must be bf16");
    TORCH_CHECK(A.dim() == 2 && W.dim() == 2, "2D tensors required");
    TORCH_CHECK(A.is_contiguous() && W.is_contiguous(), "A and W must be contiguous");

    const at::cuda::OptionalCUDAGuard guard(A.device());

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    TORCH_CHECK((int)W.size(1) == K, "K mismatch between A and W");
    const int N = (int)W.size(0);

    auto C = torch::empty({M, N}, A.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // A [M,K] row-major, lda = K
    cutlass::gemm::GemmCoord problem_size{M, N, K};
    typename Gemm::Arguments args{
        problem_size,
        typename Gemm::TensorRefA(reinterpret_cast<Element const*>(A.data_ptr()),
                                  cutlass::layout::RowMajor(K)),             // A, lda = K
        typename Gemm::TensorRefB(reinterpret_cast<Element const*>(W.data_ptr()),
                                  cutlass::layout::ColumnMajor(K)),          // B column-major [K,N], ldb = K (=W row stride)
        typename Gemm::TensorRefC(reinterpret_cast<Element const*>(C.data_ptr()),
                                  cutlass::layout::RowMajor(N)),             // C, ldc = N
        typename Gemm::TensorRefD(reinterpret_cast<Element*>(C.data_ptr()),
                                  cutlass::layout::RowMajor(N)),             // D, ldc = N
        {1.0f, 0.0f}                                                         // alpha, beta
    };

    Gemm gemm;
    // Workspace-free (non-split-K) Gemm requires zero bytes.
    size_t workspace_size = gemm.get_workspace_size(args);
    TORCH_CHECK(workspace_size == 0, "unexpected workspace requirement: ", workspace_size);

    cutlass::Status status = gemm.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "out_proj_gemm_cutlass: can_implement failed");

    status = gemm.initialize(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "out_proj_gemm_cutlass: initialize failed");

    status = gemm();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "out_proj_gemm_cutlass: launch failed");

    return C;
}

} // namespace output_gate_cutlass_detail
