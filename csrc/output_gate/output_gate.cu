// output_gate: RMSNormGated + out_proj (SM80)
//
// Two SM80 kernels:
//   1) rmsnorm_gated: out[i] = RMSNorm(y)[i] * weight[i] * silu(z[i]), i in [0,128)
//      y,z : [N,128] bf16  ->  out [N,128] bf16  (N = B*S*32), eps=1e-6
//      Warp-per-row: one warp (32 threads) processes one 128-wide row using
//      warp shuffles for the reduction (no __syncthreads).
//   2) out_proj: GEMM  C = A * W^T   A:[M,K], W:[N,K] (Linear weight [N,K]) -> C:[M,N] bf16
//      Hand-written tiled SM80 GEMM using the 16x8x16 bf16 tensor-core MMA.
//      Block tile 128(M) x 64(N), 256 threads (8 warps, 2x4 warp grid), BK=32.
//      Each warp accumulates a 64x16 output fragment across the K loop.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include <cutlass/bfloat16.h>

using BF16 = cutlass::bfloat16_t;

namespace output_gate_detail {

constexpr int DIM = 128;      // per-row norm dim (head_v_dim = 128)
constexpr float EPS = 1e-6f;

__device__ __forceinline__ float b2f(BF16 x) {
    float r; asm("cvt.f32.bf16 %0, %1;\n" : "=f"(r) : "h"(x.storage)); return r;
}
__device__ __forceinline__ BF16 f2b(float f) {
    __nv_bfloat16 h = __float2bfloat16(f);
    BF16 r; r.storage = *reinterpret_cast<uint16_t*>(&h); return r;
}
// Pack two bf16 values into one 32-bit register (low = lo, high = hi).
__device__ __forceinline__ unsigned pack2(BF16 lo, BF16 hi) {
    return (unsigned)lo.storage | ((unsigned)hi.storage << 16);
}

// ---------------------------------------------------------------------------
// Kernel 1: RMSNormGated — warp-per-row.
// One warp (32 lanes) handles one 128-wide row: each lane loads 4 elements,
// warp-reduces the sum of squares via __shfl_xor_sync (no block syncs), then
// fuses the silu gate write. 8 rows per block (256 threads).
// ---------------------------------------------------------------------------
__global__ void rmsnorm_gated_kernel(
    BF16 const* __restrict__ y,
    BF16 const* __restrict__ z,
    BF16 const* __restrict__ weight,
    BF16* __restrict__ out,
    int N)
{
    constexpr int WARPS = 8;                 // threads / 32
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;

    const BF16* yrow = y + (size_t)row * DIM;
    const BF16* zrow = z + (size_t)row * DIM;
    BF16* orow = out + (size_t)row * DIM;

    // local partial sums over the 4 elements owned by this lane
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    float v0 = b2f(yrow[lane]);
    float v1 = b2f(yrow[lane + 32]);
    float v2 = b2f(yrow[lane + 64]);
    float v3 = b2f(yrow[lane + 96]);
    s0 = v0 * v0; s1 = v1 * v1; s2 = v2 * v2; s3 = v3 * v3;
    float local = s0 + s1 + s2 + s3;

    // warp reduction (5 steps)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(0xffffffffu, local, off);

    float rstd = rsqrtf(local / (float)DIM + EPS);

    float g0 = b2f(zrow[lane]);
    float g1 = b2f(zrow[lane + 32]);
    float g2 = b2f(zrow[lane + 64]);
    float g3 = b2f(zrow[lane + 96]);
    float w0 = b2f(weight[lane]);
    float w1 = b2f(weight[lane + 32]);
    float w2 = b2f(weight[lane + 64]);
    float w3 = b2f(weight[lane + 96]);

    float s0v = 1.0f / (1.0f + expf(-g0));
    float s1v = 1.0f / (1.0f + expf(-g1));
    float s2v = 1.0f / (1.0f + expf(-g2));
    float s3v = 1.0f / (1.0f + expf(-g3));

    orow[lane]       = f2b(v0 * rstd * w0 * (g0 * s0v));
    orow[lane + 32]  = f2b(v1 * rstd * w1 * (g1 * s1v));
    orow[lane + 64]  = f2b(v2 * rstd * w2 * (g2 * s2v));
    orow[lane + 96]  = f2b(v3 * rstd * w3 * (g3 * s3v));
}

// ---------------------------------------------------------------------------
// Kernel 2: out_proj GEMM  C[M,N] = A[M,K] * W[N,K]^T  (bf16, tensor cores)
//
// Block tile BM x BN, 256 threads = 8 warps arranged 2(M) x 4(N).
//   warpM = warp/4 (0,1)  ->  M region 64 rows  -> 4 sub-tiles of 16
//   warpN = warp%4 (0..3) ->  N region 16 cols  -> 2 sub-tiles of 8
// Per warp: 4*2 = 8 MMA sub-tiles, each a 16x8x16 bf16 MMA.
// BK = 32 columns of K per stage (2 inner MMA k-steps of 16).
// ---------------------------------------------------------------------------
// cp.async helpers (SM80): 16-byte async global->shared copy.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Load one BM x BK A-tile (and BN x BK B-tile) via 16-byte cp.async into buffer `buf`.
template <int BM, int BN, int BK>
__device__ __forceinline__ void load_tiles_async(
    BF16 const* __restrict__ A, BF16 const* __restrict__ W,
    BF16* __restrict__ sA, BF16* __restrict__ sB,
    int tid, int block_m, int block_n, int kt, int M, int N, int K)
{
    constexpr int THREADS = 256;
    const int kbase = kt * BK;
    constexpr int K8 = BK / 8;                       // 8-bf16 chunks per row = 4
    // A: BM*BK bf16 = BM*BK*2 bytes -> /16 = BM*BK/8 units
    constexpr int UNITS_A = BM * BK / 8;
    #pragma unroll
    for (int u = tid; u < UNITS_A; u += THREADS) {
        int m = u / K8;
        int kc = u % K8;
        int gm = block_m + m;
        int gk = kbase + kc * 8;
        if (gm < M && gk + 8 <= K) {
            cp_async16(&sA[m * BK + kc * 8], &A[(size_t)gm * K + gk]);
        } else {
            BF16* dst = &sA[m * BK + kc * 8];
            dst[0] = dst[1] = dst[2] = dst[3] = dst[4] = dst[5] = dst[6] = dst[7] = BF16(0.f);
        }
    }
    constexpr int UNITS_B = BN * BK / 8;
    #pragma unroll
    for (int u = tid; u < UNITS_B; u += THREADS) {
        int n = u / K8;
        int kc = u % K8;
        int gn = block_n + n;
        int gk = kbase + kc * 8;
        if (gn < N && gk + 8 <= K) {
            cp_async16(&sB[n * BK + kc * 8], &W[(size_t)gn * K + gk]);
        } else {
            BF16* dst = &sB[n * BK + kc * 8];
            dst[0] = dst[1] = dst[2] = dst[3] = dst[4] = dst[5] = dst[6] = dst[7] = BF16(0.f);
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN>
__global__ void out_proj_gemm_kernel(
    BF16 const* __restrict__ A,   // [M, K] row-major
    BF16 const* __restrict__ W,   // [N, K] row-major (Linear weight)
    BF16* __restrict__ C,         // [M, N_pad] row-major
    int M, int N, int K)
{
    constexpr int THREADS = 256;
    constexpr int MS = (BM / WM) / 16;   // M sub-tiles (16 rows) per warp
    constexpr int NS = (BN / WN) / 8;    // N sub-tiles (8 cols) per warp
    // Double-buffered shared tiles
    __shared__ BF16 sA[2][BM * BK];
    __shared__ BF16 sB[2][BN * BK];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int warpM = warp / WN;          // 0..WM-1
    int warpN = warp % WN;          // 0..WN-1

    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;
    const int n_k = (K + BK - 1) / BK;

    // Accumulators: MS(M-subtiles) x NS(N-subtiles) x 4 regs
    float acc[MS][NS][4];
    #pragma unroll
    for (int im = 0; im < MS; ++im)
        #pragma unroll
        for (int in = 0; in < NS; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[im][in][r] = 0.0f;

    int group = lane >> 2;         // 0..7
    int lane4 = lane & 3;          // 0..3

    // Preload kt=0 into buffer 0
    load_tiles_async<BM, BN, BK>(A, W, sA[0], sB[0], tid, block_m, block_n, 0, M, N, K);
    cp_async_commit();

    for (int kt = 0; kt < n_k; ++kt) {
        int buf = kt & 1;
        // Prefetch next K-tile into the other buffer
        if (kt + 1 < n_k) {
            load_tiles_async<BM, BN, BK>(A, W, sA[buf ^ 1], sB[buf ^ 1],
                                         tid, block_m, block_n, kt + 1, M, N, K);
            cp_async_commit();
        }
        // Wait until this stage's tile is ready (at most the prefetch may remain)
        if (kt + 1 < n_k) cp_async_wait<1>();
        else               cp_async_wait<0>();
        __syncthreads();

        // ---- MMA accumulation over the 2 inner k-steps ----
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            int koff = kk * 16;            // 0 or 16 within this BK chunk
            #pragma unroll
            for (int im = 0; im < MS; ++im) {
                int m0 = warpM * (BM / WM) + im * 16;
                // A fragment: 4 b32 regs, each packing 2 bf16 (low=col 2l, high=col 2l+1)
                unsigned a0 = pack2(sA[buf][(m0 + group) * BK + koff + 2 * lane4],
                                    sA[buf][(m0 + group) * BK + koff + 2 * lane4 + 1]);
                unsigned a1 = pack2(sA[buf][(m0 + group + 8) * BK + koff + 2 * lane4],
                                    sA[buf][(m0 + group + 8) * BK + koff + 2 * lane4 + 1]);
                unsigned a2 = pack2(sA[buf][(m0 + group) * BK + koff + 2 * lane4 + 8],
                                    sA[buf][(m0 + group) * BK + koff + 2 * lane4 + 9]);
                unsigned a3 = pack2(sA[buf][(m0 + group + 8) * BK + koff + 2 * lane4 + 8],
                                    sA[buf][(m0 + group + 8) * BK + koff + 2 * lane4 + 9]);
                #pragma unroll
                for (int in = 0; in < NS; ++in) {
                    int n0 = warpN * (BN / WN) + in * 8 + group;
                    // B fragment: 2 b32 regs (B[n0][k], k = K index)
                    unsigned b0 = pack2(sB[buf][n0 * BK + koff + 2 * lane4],
                                        sB[buf][n0 * BK + koff + 2 * lane4 + 1]);
                    unsigned b1 = pack2(sB[buf][n0 * BK + koff + 2 * lane4 + 8],
                                        sB[buf][n0 * BK + koff + 2 * lane4 + 9]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                        "{%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[im][in][0]), "+f"(acc[im][in][1]),
                          "+f"(acc[im][in][2]), "+f"(acc[im][in][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1));
                }
            }
        }
        __syncthreads();
    }

    // ---- write accumulator fragments to gmem C (padded buffer, always in-bounds) ----
    #pragma unroll
    for (int im = 0; im < MS; ++im) {
        int m0 = warpM * (BM / WM) + im * 16;
        #pragma unroll
        for (int in = 0; in < NS; ++in) {
            int cbase = block_n + warpN * (BN / WN) + in * 8;
            int rbase = block_m + m0;
            C[(size_t)(rbase + group) * N + cbase + 2 * lane4]     = f2b(acc[im][in][0]);
            C[(size_t)(rbase + group) * N + cbase + 2 * lane4 + 1] = f2b(acc[im][in][1]);
            C[(size_t)(rbase + group + 8) * N + cbase + 2 * lane4]     = f2b(acc[im][in][2]);
            C[(size_t)(rbase + group + 8) * N + cbase + 2 * lane4 + 1] = f2b(acc[im][in][3]);
        }
    }
}

} // namespace

torch::Tensor rmsnorm_gated(
    torch::Tensor y, torch::Tensor z, torch::Tensor weight)
{
    TORCH_CHECK(y.is_cuda() && z.is_cuda() && weight.is_cuda(), "cuda tensors required");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be bf16");
    TORCH_CHECK(y.dim() == 2, "y must be [N,128]");
    int N = y.size(0);
    TORCH_CHECK(y.size(1) == output_gate_detail::DIM, "last dim must be 128");
    TORCH_CHECK(z.sizes() == y.sizes(), "z must match y shape");

    auto out = torch::empty_like(y);
    constexpr int WARPS = 8, THREADS = 256;
    int grid = (N + WARPS - 1) / WARPS;
    auto stream = at::cuda::getCurrentCUDAStream();
    output_gate_detail::rmsnorm_gated_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<BF16*>(y.data_ptr()),
        reinterpret_cast<BF16*>(z.data_ptr()),
        reinterpret_cast<BF16*>(weight.data_ptr()),
        reinterpret_cast<BF16*>(out.data_ptr()),
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor out_proj_gemm(
    torch::Tensor A, torch::Tensor W)  // A:[M,K], W:[N,K]
{
    TORCH_CHECK(A.is_cuda() && W.is_cuda(), "cuda tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(A.dim() == 2 && W.dim() == 2, "2D tensors");
    int M = A.size(0), K = A.size(1);
    TORCH_CHECK(W.size(1) == K, "K mismatch");
    int N = W.size(0);

    // Pad M/N to multiples of the block tile so the kernel never straddles the
    // boundary; slice back to the logical shape afterwards.
    constexpr int BM = 128, BN = 128, WM = 2, WN = 4;
    const int M_pad = (M + BM - 1) / BM * BM;
    const int N_pad = (N + BN - 1) / BN * BN;
    auto C_pad = torch::empty({M_pad, N_pad}, A.options());
    dim3 grid(M_pad / BM, N_pad / BN);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    output_gate_detail::out_proj_gemm_kernel<BM, BN, 32, WM, WN><<<grid, block, 0, stream>>>(
        reinterpret_cast<BF16*>(A.data_ptr()),
        reinterpret_cast<BF16*>(W.data_ptr()),
        reinterpret_cast<BF16*>(C_pad.data_ptr()),
        M_pad, N_pad, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C_pad.narrow(0, 0, M).narrow(1, 0, N);
}

// CUTLASS out_proj baseline lives in output_gate_cutlass.cu (same extension).
namespace output_gate_cutlass_detail {
torch::Tensor out_proj_gemm_cutlass(torch::Tensor A, torch::Tensor W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_gated", &rmsnorm_gated, "RMSNormGated (y,z,weight)->out");
    m.def("out_proj_gemm", &out_proj_gemm, "out_proj GEMM A*W^T");
    m.def("out_proj_gemm_cutlass", &output_gate_cutlass_detail::out_proj_gemm_cutlass,
          "out_proj GEMM A*W^T via CUTLASS device::Gemm (bf16, row-major)");
}
