// gdn_scan_stage2_blelloch.cu — Stage 2 Phase B of the gdn_chunk superchunk
// affine scan: Blelloch (work-efficient) EXCLUSIVE prefix scan over per-group
// transfer matrices A_g/B_g, replacing the Hillis-Steele scan in
// gdn_scan_stage2.cu. Cuts compositions from G*log2(G) to ~2*G.
//
// EXCLUSIVE vs inclusive: element g of the output holds the transfer of groups
// 0..g-1 (group g itself excluded); element 0 = identity (A=I, B=0). Stage-3
// uses this as the start state of group g.
//
// Two phases, each log2(G) kernel launches (one per tree level):
//   UP-SWEEP   (reduction)  x[i+s-1] = compose(x[i+s-1], x[i+s/2-1])
//               parent = compose(right_child, left_child)  (left half first)
//   DOWN-SWEEP (distribution) per parent (i, s=2*half):
//               prefix_left       = P          (parent's exclusive prefix)
//               prefix_right      = compose(L, P)   ("P first, then L")
//               x[i+half-1] = P ;  x[i+s-1] = compose(x[i+half-1], P)
//   where L = total of left child (x[i+half-1]), P = parent exclusive prefix
//   (x[i+s-1]). The root x[G-1] is set to identity before the down-sweep.
//
// Non-commutative ordering verified against the scan_torch reference prototype
// (fp32: matches serial + Hillis-Steele-shifted to ~1e-6; bf16: bf16-rounding).
//
// PACKED [A|B] formulation, same as gdn_scan_stage2.cu: each composition is ONE
// tensor-core GEMM A2 @ [A1|B1] (128 x 256 x 128), epilogue B_out = A2@B1 + B2.
//   up-sweep   A2/B2 = x[parent],   A1/B1 = x[left child]
//   down-sweep A2/B2 = x[left child], A1/B1 = x[parent prefix]
// B-op is loaded via load_C + c_to_b (copy_B would silently TRANSPOSE — see
// gdn-copyb-transpose-trap). smem/gmem layout identical to gdn_scan_stage2.cu.
//
// Precision matches the torch scan reference 'bf16' cfg: operands bf16, fp32 MMA
// accumulate, round to bf16 at phase boundaries.
//
// Tree layout: one array per sequence of G nodes, contiguous like the existing
// stage2 (index = bh*G + node). grid = (B*H, G/step); block (bh, j) handles
// parent node j*step + step - 1 of sequence bh.

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

// ===========================================================================
// UP-SWEEP level: x[parent] = compose(x[parent], x[left_child])
//   parent = j*step + step - 1, left_child = parent - half  (half = step/2)
// One block per (bh, parent); every block composes (no pass-through).
// ===========================================================================
template <int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_scan_stage2_blelloch_up_kernel(
    const BF16* __restrict__ srcA,   // [B*H*G, D, D] row-major bf16 (tree)
    const BF16* __restrict__ srcB,
    BF16* __restrict__ dstA,
    BF16* __restrict__ dstB,
    int step,    // 2 * half for this level
    int G)       // num_groups (power of two)
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
    auto A2_t = make_tensor(buf, SLayout{});
    auto X_t  = make_tensor(buf + D*D, PLayout{});
    auto B2_t = make_tensor(buf + D*D + D*(2*D), SLayout{});
    auto St_t = make_tensor(buf + D*D + D*(2*D) + D*D, StLayout{});

    const int bh = blockIdx.x;
    const int half = step >> 1;
    const int parent = blockIdx.y;                 // node id (all G nodes)
    const int left   = parent - half;              // only meaningful for parents
    const int64_t baseP = int64_t(bh) * G + parent;
    const int64_t baseL = int64_t(bh) * G + left;

    // Non-parent nodes pass through unchanged (ping-pong requires dst fully
    // written each level).
    if ((parent % step) != (step - 1)) {
        for (int i = tid; i < D*D; i += NumThreads) {
            dstA[baseP*D*D + i] = srcA[baseP*D*D + i];
            dstB[baseP*D*D + i] = srcB[baseP*D*D + i];
        }
        return;
    }

    // Load A2/B2 = x[parent] (right operand / left-multiplier)
    {
        const BF16* srcA2 = srcA + baseP * int64_t(D*D);
        const BF16* srcB2 = srcB + baseP * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            A2_t(r, c) = srcA2[r*D + c];
            B2_t(r, c) = srcB2[r*D + c];
        }
    }
    // Load [A1|B1] = x[left_child] into X_t: cols 0:D = A, cols D:2D = B
    {
        const BF16* srcA1 = srcA + baseL * int64_t(D*D);
        const BF16* srcB1 = srcB + baseL * int64_t(D*D);
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

    // ---- Packed GEMM: [A_tmp | B_tmp] = A2 @ [A1 | B1] ----
    //   n in [0,8)  -> A part (A_out = A2@A1)
    //   n in [8,16) -> B part (B_out = A2@B1 + B2)
    constexpr int NB = 2 * D / 16;       // 16 col-blocks
    constexpr int KB = D / 16;           // 8 k-blocks
    const int half_blk = D / 16;         // 8 A-col blocks

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
        // epilogue: B half adds own B2 = B[parent] at (w, n - half_blk)
        if (n >= half_blk) {
            SFragT B2f;
            copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                local_tile(B2_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n - half_blk))),
                smem_thr_load_C.retile_D(B2f));
            for (int i = 0; i < size(acc); ++i) acc(i) += bf16_to_f32(B2f(i));
        }
        SFragT sOut; cute::transform(acc, sOut, [] __device__ (float x) { return BF16(x); });
        copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sOut),
             smem_thr_store_C.partition_D(
                 local_tile(St_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, 0))));
        __syncthreads();
        {
            int r0 = w * 16;
            int c0 = n * 16;
            BF16* dst = (n < half_blk)
                ? dstA + baseP*int64_t(D*D) + r0*D + c0
                : dstB + baseP*int64_t(D*D) + r0*D + (c0 - half_blk*16);
            for (int i = lane; i < 256; i += 32) {
                int rr = i >> 4;
                int cc = i & 15;
                dst[rr*D + cc] = St_t(r0 + rr, cc);
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// DOWN-SWEEP level: distribute exclusive prefixes.
//   prefix_left  = P                       -> node[left]
//   prefix_right = compose(L, P) ("P then L") -> node[parent]
// where L = total of left child (x[left]), P = parent exclusive prefix (x[parent]).
// A2/B2 = x[left] (the L total), A1/B1 = x[parent] (the P prefix).
// ===========================================================================
template <int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_scan_stage2_blelloch_down_kernel(
    const BF16* __restrict__ srcA,
    const BF16* __restrict__ srcB,
    BF16* __restrict__ dstA,
    BF16* __restrict__ dstB,
    int step,    // 2 * half for this level
    int G)       // num_groups (power of two)
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

    auto A2_t = make_tensor(buf, SLayout{});
    auto X_t  = make_tensor(buf + D*D, PLayout{});
    auto B2_t = make_tensor(buf + D*D + D*(2*D), SLayout{});
    auto St_t = make_tensor(buf + D*D + D*(2*D) + D*D, StLayout{});

    const int bh = blockIdx.x;
    const int half = step >> 1;
    const int parent = blockIdx.y * step + step - 1;
    const int left   = parent - half;
    const int64_t baseP = int64_t(bh) * G + parent;
    const int64_t baseL = int64_t(bh) * G + left;

    // Load A2/B2 = x[left] = L total (right operand / left-multiplier)
    {
        const BF16* srcA2 = srcA + baseL * int64_t(D*D);
        const BF16* srcB2 = srcB + baseL * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            A2_t(r, c) = srcA2[r*D + c];
            B2_t(r, c) = srcB2[r*D + c];
        }
    }
    // Load [A1|B1] = x[parent] = P exclusive prefix into X_t
    {
        const BF16* srcA1 = srcA + baseP * int64_t(D*D);
        const BF16* srcB1 = srcB + baseP * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            X_t(r, c)     = srcA1[r*D + c];
            X_t(r, D + c) = srcB1[r*D + c];
        }
    }
    __syncthreads();

    // ---- prefix_left = P: copy X_t (P prefix) into node[left] ----
    {
        BF16* dstA1 = dstA + baseL * int64_t(D*D);
        BF16* dstB1 = dstB + baseL * int64_t(D*D);
        for (int i = tid; i < D*D; i += NumThreads) {
            int r = i / D, c = i - r*D;
            dstA1[r*D + c] = X_t(r, c);
            dstB1[r*D + c] = X_t(r, D + c);
        }
    }
    // (no syncthreads needed yet: X_t is read-only from now on; A2_t/B2_t for
    //  the GEMM are already staged in smem, independent of X_t / dst writes)

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

    // ---- Packed GEMM: [A_tmp | B_tmp] = A2 @ [A1 | B1]
    //      = L @ [P | ...]; result A = L@P, B = L@BP + BL.  Store -> node[parent]
    constexpr int NB = 2 * D / 16;
    constexpr int KB = D / 16;
    const int half_blk = D / 16;

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
        // epilogue: B half adds B2 = B[left] (the L total's B) at (w, n - half_blk)
        if (n >= half_blk) {
            SFragT B2f;
            copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                local_tile(B2_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n - half_blk))),
                smem_thr_load_C.retile_D(B2f));
            for (int i = 0; i < size(acc); ++i) acc(i) += bf16_to_f32(B2f(i));
        }
        SFragT sOut; cute::transform(acc, sOut, [] __device__ (float x) { return BF16(x); });
        copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sOut),
             smem_thr_store_C.partition_D(
                 local_tile(St_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, 0))));
        __syncthreads();
        {
            int r0 = w * 16;
            int c0 = n * 16;
            BF16* dst = (n < half_blk)
                ? dstA + baseP*int64_t(D*D) + r0*D + c0
                : dstB + baseP*int64_t(D*D) + r0*D + (c0 - half_blk*16);
            for (int i = lane; i < 256; i += 32) {
                int rr = i >> 4;
                int cc = i & 15;
                dst[rr*D + cc] = St_t(r0 + rr, cc);
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// Set root node x[G-1] of every sequence to identity (A=I, B=0).
// Runs once between the up-sweep and the down-sweep.
// ===========================================================================
template <int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_scan_stage2_blelloch_setroot_kernel(
    BF16* __restrict__ dstA,
    BF16* __restrict__ dstB,
    int G)
{
    const int bh = blockIdx.x;
    const int64_t base = int64_t(bh) * G + (G - 1);
    BF16* A = dstA + base * int64_t(D*D);
    BF16* B = dstB + base * int64_t(D*D);
    const int tid = threadIdx.x;
    for (int i = tid; i < D*D; i += NumThreads) {
        int r = i / D, c = i - r*D;
        A[r*D + c] = BF16((r == c) ? 1.0f : 0.0f);
        B[r*D + c] = BF16(0.0f);
    }
}

// ==================== C launchers (extern "C") ====================
#ifndef GDN_SCAN_D
#define GDN_SCAN_D 128
#define GDN_SCAN_NUM_THREADS 256
#endif

// Shared smem footprint for one packed-GEMM composition block.
static size_t stage2_blelloch_smem_bytes() {
    constexpr int D = GDN_SCAN_D;
    return (size_t)(D*D + D*2*D + D*D + D*16) * sizeof(cutlass::bfloat16_t);
}

// One UP-SWEEP level. src/dst may alias (ping-pong). step = 2^d (d = level).
extern "C" void gdn_scan_stage2_blelloch_up(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int step, int G, int B, int H,
    cudaStream_t stream) {
    constexpr int D = GDN_SCAN_D;
    constexpr int NUM_THREADS = GDN_SCAN_NUM_THREADS;
    const size_t smem = stage2_blelloch_smem_bytes();
    cudaFuncSetAttribute(gdn_scan_stage2_blelloch_up_kernel<D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(B * H, G);
    dim3 block(NUM_THREADS);
    gdn_scan_stage2_blelloch_up_kernel<D, NUM_THREADS><<<grid, block, smem, stream>>>(
        srcA, srcB, dstA, dstB, step, G);
}

// One DOWN-SWEEP level.
extern "C" void gdn_scan_stage2_blelloch_down(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int step, int G, int B, int H,
    cudaStream_t stream) {
    constexpr int D = GDN_SCAN_D;
    constexpr int NUM_THREADS = GDN_SCAN_NUM_THREADS;
    const size_t smem = stage2_blelloch_smem_bytes();
    cudaFuncSetAttribute(gdn_scan_stage2_blelloch_down_kernel<D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(B * H, G / step);
    dim3 block(NUM_THREADS);
    gdn_scan_stage2_blelloch_down_kernel<D, NUM_THREADS><<<grid, block, smem, stream>>>(
        srcA, srcB, dstA, dstB, step, G);
}

// Set root node (index G-1) of every sequence to identity.
extern "C" void gdn_scan_stage2_blelloch_setroot(
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int G, int B, int H,
    cudaStream_t stream) {
    constexpr int D = GDN_SCAN_D;
    constexpr int NUM_THREADS = GDN_SCAN_NUM_THREADS;
    dim3 grid(B * H);
    dim3 block(NUM_THREADS);
    gdn_scan_stage2_blelloch_setroot_kernel<D, NUM_THREADS><<<grid, block, 0, stream>>>(
        dstA, dstB, G);
}

// Full Blelloch EXCLUSIVE scan (all up + down levels).
// A_g/B_g are [B*H*G, D, D] bf16; after return element g of the OUTPUT buffers
// holds the exclusive prefix (transfer of groups 0..g-1; element 0 = identity).
// G must be a power of two. src and dst must be distinct buffers (the caller
// allocates a ping-pong partner, like the existing stage2_scan). The result is
// left in dst (matching the gdn_scan_stage2 single-round convention).
extern "C" void gdn_scan_stage2_blelloch(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int G, int B, int H,
    cudaStream_t stream) {
    // internal ping-pong: A/BI = current source buffer, A/BO = current sink.
    const cutlass::bfloat16_t* AI = srcA; const cutlass::bfloat16_t* BI = srcB;
    cutlass::bfloat16_t* AO = dstA;     cutlass::bfloat16_t* BO = dstB;

    // ---- UP-SWEEP: levels d = 0..nlevels-1, step = 2,4,...,G ----
    for (int step = 2; step <= G; step <<= 1) {
        gdn_scan_stage2_blelloch_up(AI, BI, AO, BO, step, G, B, H, stream);
        const cutlass::bfloat16_t* tA = AI; AI = AO; AO = const_cast<cutlass::bfloat16_t*>(tA);
        const cutlass::bfloat16_t* tB = BI; BI = BO; BO = const_cast<cutlass::bfloat16_t*>(tB);
    }
    // after the up-sweep, the result is in AI/BI (last swap moved it there).
    // ---- root -> identity (in-place on the up-sweep result buffer) ----
    gdn_scan_stage2_blelloch_setroot(const_cast<cutlass::bfloat16_t*>(AI),
                                     const_cast<cutlass::bfloat16_t*>(BI), G, B, H, stream);
    // ---- DOWN-SWEEP: levels d = nlevels-1..0, step = G,G/2,...,2 ----
    for (int step = G; step >= 2; step >>= 1) {
        gdn_scan_stage2_blelloch_down(AI, BI, AO, BO, step, G, B, H, stream);
        const cutlass::bfloat16_t* tA = AI; AI = AO; AO = const_cast<cutlass::bfloat16_t*>(tA);
        const cutlass::bfloat16_t* tB = BI; BI = BO; BO = const_cast<cutlass::bfloat16_t*>(tB);
    }
    // result is in AI/BI; ensure it lands in dst (the documented output buffer)
    if (AI != dstA) {
        const size_t n = (size_t)B * H * G * GDN_SCAN_D * GDN_SCAN_D;
        cudaMemcpyAsync(dstA, AI, n * sizeof(cutlass::bfloat16_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dstB, BI, n * sizeof(cutlass::bfloat16_t),
                        cudaMemcpyDeviceToDevice, stream);
    }
}
