// gdn_scan_stage2.cu — Stage 2 of the gdn_chunk superchunk affine scan:
// Hillis-Steele prefix scan over per-group transfer matrices A_g/B_g.
//
// PACKED [A|B] formulation (Phase A, exact):
//   Each composition (A2,B2) o (A1,B1) = (A2@A1, A2@B1 + B2) is done as ONE
//   tensor-core GEMM on the packed operand [A1 | B1] (shape [D, 2D]):
//     [A_tmp | B_tmp] = A2 @ [A1 | B1]      # 128 x 256 x 128
//     epilogue: B_out = B_tmp + B2
//   vs the old two separate 128x128x128 GEMMs. A2 is read from smem once and
//   reused for all 16 output col-blocks (8 A-col + 8 B-col).
//
// One kernel launch per scan round (offset doubles each round); groups with
// g < offset are passed through. After ceil(log2(G)) rounds element g holds
// the transfer of groups 0..g.
//
// gmem layout unchanged (A_g/B_g separate [D,D] bf16 row-major, ping-pong).
// smem: A2 [D,D] + X=[A1|B1] [D,2D] + B2 [D,D] + small staging tile.
//
// Tensor cores: SM80 m16n8k16, 8 warps. warp w owns output rows [16w,16w+16).
// A-op via copy_A (correct on MMALayout); B-op via load_C + c_to_b (copy_B
// would silently TRANSPOSE — see gdn-copyb-transpose-trap).
//
// Precision matches the torch scan reference prefix_scan 'bf16' cfg:
//   A = bf16(A2@A1), B = bf16(A2@B1 + B2), fp32 accumulate.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>
#include <type_traits>
#include <utility>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/stride.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

using namespace cute;
using BF16 = cutlass::bfloat16_t;

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float r; asm("cvt.f32.bf16 %0, %1;\n" : "=f"(r) : "h"(x.storage)); return r;
}

// [D,D] K_INTER swizzle
template <int D>
using StateSmemLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
    make_shape(Int<D>{}, Int<D>{}),
    LayoutLeft{}));

// [D,2D] K_INTER swizzle for the packed [A|B] operand
template <int D>
using PackedSmemLayout = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
    make_shape(Int<D>{}, Int<2 * D>{}),
    LayoutLeft{}));

// per-warp [16,16] staging, stacked as [D,16] row-major (8 warps * 16 rows)
template <int D>
using StagingLayout = decltype(make_layout(
    make_shape(Int<D>{}, Int<16>{}), LayoutRight{}));

template <int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_scan_stage2_kernel(
    const BF16* __restrict__ srcA,   // [B*H*G, D, D] row-major bf16
    const BF16* __restrict__ srcB,
    BF16* __restrict__ dstA,
    BF16* __restrict__ dstB,
    int offset,   // Hillis-Steele offset for this round
    int G)        // num_groups
{
    using SLayout = StateSmemLayout<D>;
    using PLayout = PackedSmemLayout<D>;
    using StLayout = StagingLayout<D>;
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = NumThreads / kWarpSize;
    const int w = threadIdx.x / kWarpSize;   // 0..7
    const int lane = threadIdx.x % kWarpSize;
    const int tid = threadIdx.x;

    extern __shared__ __align__(128) unsigned char smem[];
    auto buf = make_smem_ptr(reinterpret_cast<BF16*>(smem));

    // smem: A2 [D,D] + X=[A1|B1] [D,2D] + B2 [D,D] + staging [D,16]
    // (each warp stores its [16,16] output tile at row-block w of St)
    auto A2_t = make_tensor(buf, SLayout{});
    auto X_t  = make_tensor(buf + D*D, PLayout{});
    auto B2_t = make_tensor(buf + D*D + D*(2*D), SLayout{});
    auto St_t = make_tensor(buf + D*D + D*(2*D) + D*D, StLayout{});

    const int bh = blockIdx.x;
    const int g  = blockIdx.y;
    const int64_t base  = int64_t(bh) * G + g;
    const int64_t base1 = int64_t(bh) * G + (g - offset);

    // Load own-group A2 = A[g] and B2 = B[g]
    {
        const BF16* srcA2 = srcA + base * int64_t(D*D);
        const BF16* srcB2 = srcB + base * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            A2_t(r, c) = srcA2[r*D + c];
            B2_t(r, c) = srcB2[r*D + c];
        }
    }
    if (g >= offset) {
        // Load [A1 | B1] into X_t: cols 0:D = A[g-offset], cols D:2D = B[g-offset]
        const BF16* srcA1 = srcA + base1 * int64_t(D*D);
        const BF16* srcB1 = srcB + base1 * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            X_t(r, c)     = srcA1[r*D + c];
            X_t(r, D + c) = srcB1[r*D + c];
        }
    }
    __syncthreads();

    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
        Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
    ThrMMA thr_mma = mma.get_slice(lane);

    auto smem_tiled_copy_A   = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(lane);
    auto smem_tiled_load_C   = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto smem_thr_load_C     = smem_tiled_load_C.get_slice(lane);
    auto smem_tiled_store_C  = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
    auto smem_thr_store_C    = smem_tiled_store_C.get_slice(lane);

    Tensor Sref = local_tile(X_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
    using AccFragT = decltype(thr_mma.make_fragment_C(thr_mma.partition_C(Sref)));
    using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(thr_mma.partition_C(Sref))));
    using AFragT = decltype(thr_mma.partition_fragment_A(Sref));
    using BFragT = decltype(thr_mma.partition_fragment_B(Sref));

    auto c_to_b = [&](SFragT const& c_frag) -> BFragT {
        BFragT b = thr_mma.partition_fragment_B(Sref);
        uint32_t reg[4];
        uint32_t const* uc = reinterpret_cast<uint32_t const*>(&c_frag(0));
        SM75_U32x1_MOVM_T::copy(uc[0], reg[0]);
        SM75_U32x1_MOVM_T::copy(uc[1], reg[1]);
        SM75_U32x1_MOVM_T::copy(uc[2], reg[2]);
        SM75_U32x1_MOVM_T::copy(uc[3], reg[3]);
        uint32_t* bd = reinterpret_cast<uint32_t*>(&b(0));
        bd[0] = reg[0]; bd[1] = reg[1]; bd[2] = reg[2]; bd[3] = reg[3];
        return b;
    };

    if (g < offset) {
        // pass-through: dstA[g] = srcA[g], dstB[g] = srcB[g]
        for (int i = tid; i < D*D; i += NumThreads) {
            dstA[base*D*D + i] = srcA[base*D*D + i];
            dstB[base*D*D + i] = srcB[base*D*D + i];
        }
        return;
    }

    // ---- Packed GEMM: [A_tmp | B_tmp] = A2 @ [A1 | B1] ----
    // warp w owns output rows [16w,16w+16). Output [D,2D] has 16 col-blocks:
    //   n in [0,8)   -> A part (A_out = A2@A1)
    //   n in [8,16)  -> B part (B_out = A2@B1 + B2)
    constexpr int NB = 2 * D / 16;       // 16 col-blocks
    constexpr int KB = D / 16;           // 8 k-blocks
    const int half = D / 16;             // 8 A-col blocks

    AFragT Af;
    auto Af_v = smem_thr_copy_A.retile_D(Af);

    for (int n = 0; n < NB; ++n) {
        AccFragT acc; clear(acc);
        #pragma unroll
        for (int k = 0; k < KB; ++k) {
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(A2_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, k))), Af_v);
            SFragT Cf;
            copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                local_tile(X_t, make_shape(Int<16>{}, Int<16>{}), make_coord(k, n))),
                smem_thr_load_C.retile_D(Cf));
            BFragT Bf = c_to_b(Cf);
            gemm(thr_mma, Af(_,_,Int<0>{}), Bf(_,_,Int<0>{}), acc);
        }
        // epilogue: B half adds own B2 = B[g] at (w, n - half)
        if (n >= half) {
            SFragT B2f;
            copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                local_tile(B2_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n - half))),
                smem_thr_load_C.retile_D(B2f));
            for (int i = 0; i < size(acc); ++i) acc(i) += bf16_to_f32(B2f(i));
        }
        // store via store_C into warp w's own [16,16] staging tile (row-block w)
        SFragT sOut; cute::transform(acc, sOut, [] __device__ (float x) { return BF16(x); });
        copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sOut),
             smem_thr_store_C.partition_D(
                 local_tile(St_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, 0))));
        __syncthreads();
        {
            int r0 = w * 16;
            int c0 = n * 16;
            BF16* dst = (n < half)
                ? dstA + base*int64_t(D*D) + r0*D + c0
                : dstB + base*int64_t(D*D) + r0*D + (c0 - half*16);
            for (int i = lane; i < 256; i += 32) {
                int rr = i >> 4;
                int cc = i & 15;
                dst[rr*D + cc] = St_t(r0 + rr, cc);
            }
        }
        __syncthreads();
    }
}

// ==================== C launcher (extern "C") ====================
#ifndef GDN_SCAN_D
#define GDN_SCAN_D 128
#define GDN_SCAN_NUM_THREADS 256
#endif

extern "C" void gdn_scan_stage2(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int offset,
    int G,
    int B, int H,
    cudaStream_t stream) {
    constexpr int D = GDN_SCAN_D;
    constexpr int NUM_THREADS = GDN_SCAN_NUM_THREADS;
    // A2 [D,D] + X [D,2D] + B2 [D,D] + St [D,16]
    const size_t smem = (D*D + D*2*D + D*D + D*16) * sizeof(cutlass::bfloat16_t);
    cudaFuncSetAttribute(gdn_scan_stage2_kernel<D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(B * H, G);
    dim3 block(NUM_THREADS);
    gdn_scan_stage2_kernel<D, NUM_THREADS><<<grid, block, smem, stream>>>(
        srcA, srcB, dstA, dstB, offset, G);
}
