// Gated DeltaNet chunked forward kernel (SM80 / A800).
//
// Two-kernel split:
//   Kernel 1 (prepare): per chunk compute k_decayed, q_decayed, k_inv,
//                       k_restored, g_total, L, INV=(I-L)^{-1}, Mqk.
//   Kernel 2 (recurrence): per chunk scan:
//                       U   = INV @ ((v - k_decayed @ S_prev) * beta)
//                       y   = q_decayed @ S_prev + Mqk @ U
//                       S   = S * exp(g_total) + k_restored^T @ U
//
// g (log-space decay <= 0) and beta (already sigmoid in (0,1)) are given
// directly. No dt/A_log gate and no beta sigmoid are applied. L does NOT
// contain beta; beta multiplies the error term in the recurrence phase.
//
// Inputs (host repeat_interleave of q/k onto v heads):
//   q, k, v : [B, S, H, D] contiguous (viewed as [T=B*S, H, D] row-major)
//   g, beta : [B, S, H] -> passed head-major [H, B*S] contiguous
// Outputs:
//   y           : [B, S, H, D]
//   final_state : [B, H, D, D]
//
// D = 128, CHUNK = 16.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>
#include <type_traits>
#include <utility>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

using namespace cute;

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

using BF16 = cutlass::bfloat16_t;
using FP16 = cutlass::half_t;

// ---------------- Cooperative copy helpers ----------------
template <int NumThreads, class SrcTensor, class DstTensor>
__device__ __forceinline__ void coop_copy_2d(
    SrcTensor const& src, DstTensor& dst, int tid
) {
    int R = int(cute::size<0>(src));
    int C = int(cute::size<1>(src));
    int N = R * C;
    for (int i = tid; i < N; i += NumThreads) {
        int r = i / C;
        int c = i - r * C;
        dst(r, c) = src(r, c);
    }
}

template <int NumThreads, class SrcTensor, class DstTensor>
__device__ __forceinline__ void coop_copy_1d(
    SrcTensor const& src, DstTensor& dst, int tid
) {
    int N = int(cute::size(src));
    for (int i = tid; i < N; i += NumThreads) {
        dst(i) = src(i);
    }
}

template <int NumThreads, class SrcTensor, class DstTensor>
__device__ __forceinline__ void coop_copy_2d_vec8(
    SrcTensor const& src, DstTensor& dst, int tid
) {
    int R = int(cute::size<0>(src));
    int C = int(cute::size<1>(src));
    int NV = C / 8;
    for (int i = tid; i < R * NV; i += NumThreads) {
        int r = i / NV;
        int c = (i - r * NV) * 8;
        *reinterpret_cast<uint4*>(&dst(r, c)) = *reinterpret_cast<uint4 const*>(&src(r, c));
    }
}

template <int NumThreads, class SrcTensor, class DstTensor>
__device__ __forceinline__ void coop_copy_1d_vec4(
    SrcTensor const& src, DstTensor& dst, int tid
) {
    int NV = int(cute::size(src)) / 4;
    for (int i = tid; i < NV; i += NumThreads) {
        *reinterpret_cast<uint4*>(&dst(i * 4)) = *reinterpret_cast<uint4 const*>(&src(i * 4));
    }
}

__device__ __forceinline__ void cp_async_16b_zfill(void* smem_dst, void const* gmem_src, bool pred) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_dst);
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_src), "r"(src_size));
}

// ---------------- MMA wrappers ----------------
template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16bf16_1warp(
    TensorA const& A, TensorB const& B, TensorC& C, int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    if (mma_tid >= int(size(mma))) return;
    auto sC_store_op = [] __device__ (float x) { return BF16(x); };
    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, AutoVectorizingCopy{});
}

template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16fp16_1warp(
    TensorA const& A, TensorB const& B, TensorC& C, int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    if (mma_tid >= int(size(mma))) return;
    auto sC_store_op = [] __device__ (float x) { return FP16(x); };
    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, AutoVectorizingCopy{});
}

// Neumann inverse fused 1 warp: INV = (I-L)^{-1} via L^2 + L^4 + L^8 series.
template <class TensorL, class TensorINV_fp16, class TensorINV_bf16>
CUTLASS_DEVICE void neumann_inv_fused_1warp(
    TensorL const& L_fp16, TensorINV_fp16 const& INV_fp16, TensorINV_bf16& INV_bf16_out, int tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    if (tid >= int(size(mma))) return;
    auto thr_mma = mma.get_slice(tid);
    auto smem_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, FP16>{}, mma);
    auto thr_copy_A = smem_copy_A.get_thread_slice(tid);

    Tensor tCrL = thr_mma.partition_fragment_A(L_fp16);
    {
        Tensor tmp = make_fragment_like<FP16>(tCrL);
        copy(smem_copy_A, thr_copy_A.partition_S(L_fp16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrL, cute::identity{});
    }

    Tensor tCrINV = thr_mma.partition_fragment_A(INV_fp16);
    {
        Tensor tmp = make_fragment_like<FP16>(tCrINV);
        copy(smem_copy_A, thr_copy_A.partition_S(INV_fp16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrINV, cute::identity{});
    }

    uint32_t* L_a = reinterpret_cast<uint32_t*>(&tCrL(0));
    uint32_t* INV_a = reinterpret_cast<uint32_t*>(&tCrINV(0));

    uint32_t Lpow_c[4], Lpow_b[4], INV_c[4], tmp_a[4], mm_c[4];

    auto clear_u32x4 = [](uint32_t* x) { x[0] = x[1] = x[2] = x[3] = 0; };
    auto add_fp16x2_u32x4 = [] (uint32_t* dst, uint32_t const* src) {
        union U32H2 { uint32_t u; __half2 h2; };
        U32H2 a{dst[0]}, b{src[0]}, a1{dst[1]}, b1{src[1]};
        U32H2 a2{dst[2]}, b2{src[2]}, a3{dst[3]}, b3{src[3]};
        a.h2 = __hadd2(a.h2, b.h2);
        a1.h2 = __hadd2(a1.h2, b1.h2);
        a2.h2 = __hadd2(a2.h2, b2.h2);
        a3.h2 = __hadd2(a3.h2, b3.h2);
        dst[0] = a.u; dst[1] = a1.u; dst[2] = a2.u; dst[3] = a3.u;
    };
    auto transpose_u32x4 = [](uint32_t const* src, uint32_t* dst) {
        SM75_U32x1_MOVM_T::copy(src[0], dst[0]);
        SM75_U32x1_MOVM_T::copy(src[1], dst[1]);
        SM75_U32x1_MOVM_T::copy(src[2], dst[2]);
        SM75_U32x1_MOVM_T::copy(src[3], dst[3]);
    };
    auto copy_u32x4 = [](uint32_t const* src, uint32_t* dst) {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    };
    auto mma_16x16 = [](uint32_t* d, uint32_t const* a, uint32_t const* b, uint32_t const* c) {
        SM80_16x8x16_F16F16F16F16_TN::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]);
        SM80_16x8x16_F16F16F16F16_TN::fma(d[2], d[3], a[0], a[1], a[2], a[3], b[2], b[3], c[2], c[3]);
    };

    transpose_u32x4(L_a, Lpow_b);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, L_a, Lpow_b, Lpow_c);

    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_a, INV_c);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, INV_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    copy_u32x4(Lpow_c, tmp_a);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, tmp_a, Lpow_b, Lpow_c);

    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_c, tmp_a);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, tmp_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    copy_u32x4(Lpow_c, tmp_a);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, tmp_a, Lpow_b, Lpow_c);

    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_c, tmp_a);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, tmp_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    Tensor tCsC_mma = thr_mma.partition_C(INV_fp16);
    Tensor tCrC = thr_mma.make_fragment_C(tCsC_mma);
    uint32_t* C_regs = reinterpret_cast<uint32_t*>(&tCrC(0));
    C_regs[0] = INV_c[0]; C_regs[1] = INV_c[1]; C_regs[2] = INV_c[2]; C_regs[3] = INV_c[3];

    Tensor tCrC_bf16 = make_fragment_like<BF16>(tCrC);
    cute::transform(tCrC, tCrC_bf16, [] __device__ (FP16 x) -> BF16 { return BF16(x); });

    auto smem_tiled_store = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
    auto smem_thr_store = smem_tiled_store.get_slice(tid);
    Tensor tCsC_st = smem_thr_store.partition_D(INV_bf16_out);
    Tensor tCrC_st_view = smem_thr_store.retile_S(tCrC_bf16);
    copy(smem_tiled_store, tCrC_st_view, tCsC_st);
}

// ---------------- Layouts ----------------
template <int D, int CHUNK = 16>
struct GDNLayouts {
    using QKLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using GLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using BetaSmemLayout = Layout<Shape<Int<40>>, Stride<Int<1>>>;
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutRight{}
    ));
    // Workspace tiles (k_decayed/q_decayed/k_restored, v, out) are moved
    // gmem<->smem through the SAME MMA swizzle layout that later reads them.
    // (K2: VOLayout = MMALayout.) Loading with plain row-major while
    // reading via MMA/transposed-MMA layouts silently misplaces every element.
    using VOLayout = MMALayout;
    using TransposedVOLayout = TransposedMMALayout;
    static constexpr int kChunk = CHUNK;
    static constexpr int kD = D;
};

// ---------------- Shared memory for prepare kernel ----------------
template <class Layouts>
struct GDNPrepareStorage {
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    union {
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> v;
            alignas(128) cute::ArrayEngine<float, cute::cosize_v<GLayout>> g;   // cumsum fp32 [CHUNK,D]
        };
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> L;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
        };
    };

    union {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g_bf16;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
    };
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
    alignas(16) cute::ArrayEngine<BF16, 16> beta;   // per-token beta for this chunk
};

// ==================== Kernel 1: Prepare ====================
template <int CHUNK, int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_prepare_kernel(
    const BF16* __restrict__ q_ptr, int q_row_stride,   // [T,H,D] row-major (T=B*S)
    const BF16* __restrict__ k_ptr, int k_row_stride,
    const BF16* __restrict__ v_ptr, int v_row_stride,
    const BF16* __restrict__ g_ptr, int g_row_stride,   // [H, T_total] head-major (stride=T_total)
    const BF16* __restrict__ beta_ptr, int beta_row_stride,  // [H, T_total] head-major (stride=T_total)
    BF16* __restrict__ ws_kd,
    BF16* __restrict__ ws_qd,
    BF16* __restrict__ ws_kr,
    float* __restrict__ ws_gt,
    BF16* __restrict__ ws_inv,
    BF16* __restrict__ ws_mqk,
    int ws_tile_elems,     // CHUNK*D
    int ws_tile_lm,        // CHUNK*CHUNK
    int ws_gt_elems,       // D
    float scale,
    int T_seq,             // tokens per seq (=S)
    int H,
    int chunks_per_seq,
    int head_ratio         // GQA: q/k have H/head_ratio heads; v/g/beta have H
) {
    using Layouts = GDNLayouts<D, CHUNK>;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using MMALayout = typename Layouts::MMALayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNPrepareStorage<Layouts>& shared_storage = *reinterpret_cast<GDNPrepareStorage<Layouts>*>(shared_mem);

    int local_chunk = blockIdx.x;
    int bh = blockIdx.y;          // seq*H + head
    int seq = bh / H;
    int head = bh % H;
    int t_offset = seq * T_seq + local_chunk * CHUNK;
    int ws_idx = bh * chunks_per_seq + local_chunk;
    int actual_len = min(CHUNK, T_seq - local_chunk * CHUNK);
    int tid = threadIdx.x;

    // ---- Load q/k/v/g tiles ----
    // GQA: q/k have H/head_ratio heads. v-head h shares q/k-head h/head_ratio,
    // so read q/k at (head/head_ratio)*D (row stride = Hk*D); v stays per-v-head.
    const int qk_head = head / head_ratio;
    auto g_q = make_tensor(q_ptr + int64_t(t_offset) * q_row_stride + qk_head * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(q_row_stride, Int<1>{})));
    auto g_k = make_tensor(k_ptr + int64_t(t_offset) * k_row_stride + qk_head * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(k_row_stride, Int<1>{})));
    auto g_v = make_tensor(v_ptr + int64_t(t_offset) * v_row_stride + head * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(v_row_stride, Int<1>{})));
    // g is [H, T_total] head-major (row stride = T_total between heads, stride 1 within).
    auto g_g = make_tensor(g_ptr + int64_t(head) * g_row_stride + t_offset,
        make_layout(make_shape(actual_len), make_stride(Int<1>{})));

    auto s_q = make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
    auto s_k = make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
    auto s_v = make_tensor(make_smem_ptr(shared_storage.v.begin()), QKLayout{});
    auto s_g_bf16 = make_tensor(make_smem_ptr(shared_storage.g_bf16.begin()), QKLayout{});

    coop_copy_2d_vec8<NumThreads>(g_q, s_q, tid);
    coop_copy_2d_vec8<NumThreads>(g_k, s_k, tid);
    coop_copy_2d_vec8<NumThreads>(g_v, s_v, tid);

    // g: scalar per token -> replicate across D columns (row stride D)
    for (int i = tid; i < CHUNK; i += NumThreads) {
        float val = (i < actual_len) ? bf16_to_f32(g_g(i)) : 0.0f;
        BF16 bv = BF16(val);
        uint16_t b = bv.storage;
        uint32_t pair = (uint32_t(b) << 16) | uint32_t(b);  // two bf16 copies
        uint4 v; v.x = pair; v.y = pair; v.z = pair; v.w = pair; // 8 copies = 16B
        #pragma unroll
        for (int c = 0; c < D; c += 8) {
            *reinterpret_cast<uint4*>(&s_g_bf16(i, c)) = v;
        }
    }
    // beta: scalar per token -> store in smem (head-major [H, T_total], stride=1 within seq)
    for (int i = tid; i < CHUNK; i += NumThreads) {
        float bval = (i < actual_len)
            ? bf16_to_f32(beta_ptr[int64_t(head) * beta_row_stride + t_offset + i])
            : 1.0f;   // tail rows: beta unused (k/v zero-filled); 1 is safe for L row scale
        shared_storage.beta.begin()[i] = BF16(bval);
    }
    // zero-fill tail rows of q/k/v/g
    for (int i = tid + actual_len * D; i < CHUNK * D; i += NumThreads) {
        shared_storage.q.begin()[i] = BF16();
        shared_storage.k.begin()[i] = BF16();
        shared_storage.v.begin()[i] = BF16();
        shared_storage.g_bf16.begin()[i] = BF16();
    }
    __syncthreads();

    // ---- Cumsum of g over chunk (fp32), broadcast across D cols ----
    // Each of the first 128 threads owns one column; rows are processed serially.
    {
        int col = tid;
        if (col < D) {
            BF16 const* g_src = shared_storage.g_bf16.begin();
            float* g_smem = shared_storage.g.begin();
            float sum = 0.0f;
            #pragma unroll
            for (int row = 0; row < CHUNK; ++row) {
                sum += bf16_to_f32(g_src[row * D + col]);
                g_smem[row * D + col] = sum;
            }
            shared_storage.g_total.begin()[col] = sum;
        }
    }
    __syncthreads();

    Tensor g_tile = make_tensor(make_smem_ptr(shared_storage.g.begin()), GLayout{});
    Tensor q_tile = make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
    Tensor k_tile = make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
    Tensor g_total = make_tensor(make_smem_ptr(shared_storage.g_total.begin()), GTotalLayout{});
    Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{});
    Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
    Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{});
    Tensor k_inv = make_tensor(make_smem_ptr(shared_storage.k_inv.begin()), MMALayout{});

    // g_total -> exp()  (GDN uses natural log-decay: state *= exp(g))
    if (tid < D) {
        float x = g_total(tid);
        g_total(tid) = expf(x);
    }
    __syncthreads();

    // ---- decay_apply: q_decayed, k_decayed, k_inv, k_restored ----
    if (tid < 256) {
        static_assert(D % 64 == 0);
        static_assert(CHUNK % 8 == 0);

        int lane = tid % 32;
        int warp_id = tid / 32;
        int g4 = lane / 4;
        int t = lane % 4;

        auto vec8_2d = make_shape(_1{}, _8{});
        auto vec8_1d = make_shape(_8{});
        auto thr2_2d = make_shape(_1{}, _2{});
        auto thr2_1d = make_shape(_2{});

        constexpr int N_M = CHUNK / 8;
        constexpr int N_N = D / 64;
        constexpr int N_TILES = N_M * N_N;

        float reg_g[N_TILES][2];
        BF16  reg_q[N_TILES][2];
        BF16  reg_k[N_TILES][2];
        float reg_gt[N_TILES][2];

        #pragma unroll
        for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
            #pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
                int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                int row = m_blk + ((warp_id + g4) % 8);
                int col_base = n_blk + g4 * 8;
                int col_tile = col_base / 8;

                Tensor tile_g  = local_tile(g_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_q  = local_tile(q_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_k  = local_tile(k_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_gt = local_tile(g_total, vec8_1d, make_coord(col_tile));

                Tensor s_g  = local_tile(tile_g,  thr2_2d, make_coord(0, t));
                Tensor s_q  = local_tile(tile_q,  thr2_2d, make_coord(0, t));
                Tensor s_k  = local_tile(tile_k,  thr2_2d, make_coord(0, t));
                Tensor s_gt = local_tile(tile_gt, thr2_1d, make_coord(t));

                Tensor r_g  = make_tensor_like<float>(s_g);
                Tensor r_q  = make_tensor_like<BF16>(s_q);
                Tensor r_k  = make_tensor_like<BF16>(s_k);
                Tensor r_gt = make_tensor_like<float>(s_gt);

                cute::copy(AutoVectorizingCopy{}, s_g, r_g);
                cute::copy(AutoVectorizingCopy{}, s_q, r_q);
                cute::copy(AutoVectorizingCopy{}, s_k, r_k);
                cute::copy(AutoVectorizingCopy{}, s_gt, r_gt);

                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    reg_g[tile_idx][v]  = r_g(0, v);
                    reg_q[tile_idx][v]  = r_q(0, v);
                    reg_k[tile_idx][v]  = r_k(0, v);
                    reg_gt[tile_idx][v] = r_gt(v);
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
            #pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
                int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                int row = m_blk + ((warp_id + g4) % 8);
                int col_base = n_blk + g4 * 8;
                int col_tile = col_base / 8;

                Tensor tile_qd = local_tile(q_decayed, vec8_2d, make_coord(row, col_tile));
                Tensor tile_kd = local_tile(k_decayed, vec8_2d, make_coord(row, col_tile));
                Tensor tile_kr = local_tile(k_restored, vec8_2d, make_coord(row, col_tile));
                Tensor tile_ki = local_tile(k_inv, vec8_2d, make_coord(row, col_tile));

                Tensor s_qd = local_tile(tile_qd, thr2_2d, make_coord(0, t));
                Tensor s_kd = local_tile(tile_kd, thr2_2d, make_coord(0, t));
                Tensor s_kr = local_tile(tile_kr, thr2_2d, make_coord(0, t));
                Tensor s_ki = local_tile(tile_ki, thr2_2d, make_coord(0, t));

                Tensor r_qd = make_tensor_like<BF16>(s_qd);
                Tensor r_kd = make_tensor_like<BF16>(s_kd);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    float gg = reg_g[tile_idx][v];
                    BF16 q = reg_q[tile_idx][v];
                    BF16 k = reg_k[tile_idx][v];
                    BF16 exp_cumsum = BF16(expf(gg));
                    r_qd(0, v) = q * exp_cumsum * BF16(scale);
                    r_kd(0, v) = k * exp_cumsum;
                }
                cute::copy(AutoVectorizingCopy{}, r_qd, s_qd);
                cute::copy(AutoVectorizingCopy{}, r_kd, s_kd);

                Tensor r_ki = make_tensor_like<BF16>(s_ki);
                Tensor r_kr = make_tensor_like<BF16>(s_kr);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    float gg = reg_g[tile_idx][v];
                    BF16 k = reg_k[tile_idx][v];
                    BF16 inv_cumsum = BF16(expf(-gg));
                    r_ki(0, v) = k * inv_cumsum;
                    r_kr(0, v) = k * inv_cumsum * BF16(reg_gt[tile_idx][v]);
                }
                cute::copy(AutoVectorizingCopy{}, r_ki, s_ki);
                cute::copy(AutoVectorizingCopy{}, r_kr, s_kr);
            }
        }
    }
    __syncthreads();

    // ---- L, Mqk via MMA ----
    // L = tril(k_decayed @ k_inv^T) stored as fp16 (for Neumann); Mqk as bf16.
    // For CHUNK=32 the matrices are 32x32 = a 2x2 grid of 16x16 MMA tiles;
    // use 4 warps for L and 4 warps for Mqk. (CHUNK=16 keeps the old 1-warp path.)
    Tensor L = make_tensor(make_smem_ptr(shared_storage.L.begin()), LMLayout{});
    Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{});
    Tensor L_fp16 = make_tensor(make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.L.begin())), LMLayout{});

    constexpr int L_TILES = (CHUNK / 16) * (CHUNK / 16);   // 1 or 4
    if (tid < L_TILES * 32) {
        int tile = tid / 32;                 // 0..3 for CHUNK=32
        int r = tile / (CHUNK / 16);
        int c = tile % (CHUNK / 16);
        auto A = local_tile(k_decayed, make_shape(Int<16>{}, Int<D>{}), make_coord(r, 0));
        auto B = local_tile(k_inv,     make_shape(Int<16>{}, Int<D>{}), make_coord(c, 0));
        auto C = local_tile(L_fp16,    make_shape(Int<16>{}, Int<16>{}), make_coord(r, c));
        mma_m16n16_bf16bf16fp16_1warp(A, B, C, tid % 32);
    } else if (tid < 2 * L_TILES * 32) {
        int tile = (tid - L_TILES * 32) / 32;
        int r = tile / (CHUNK / 16);
        int c = tile % (CHUNK / 16);
        auto A = local_tile(q_decayed, make_shape(Int<16>{}, Int<D>{}), make_coord(r, 0));
        auto B = local_tile(k_inv,     make_shape(Int<16>{}, Int<D>{}), make_coord(c, 0));
        auto C = local_tile(Mqk,       make_shape(Int<16>{}, Int<16>{}), make_coord(r, c));
        mma_m16n16_bf16bf16bf16_1warp(A, B, C, tid % 32);
    }
    __syncthreads();

    Tensor INV = make_tensor(make_smem_ptr(shared_storage.INV.begin()), LMLayout{});
    Tensor INV_fp16 = make_tensor(make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.INV.begin())), LMLayout{});

    // tril on L; strict-upper-zero Mqk; INV = I - L
    // Reference ut_system = (k*beta @ k^T) * pairwise  => L is row-scaled by beta_i.
    // We scale the strictly-lower L rows by the per-token beta (beta already sigmoid'd).
    // CHUNK<=16: 256 threads cover the whole CHUNKxCHUNK tile in one pass.
    // CHUNK=32:  1024 elements, so stride over NumThreads (4 elems/thread).
    for (int e = tid; e < CHUNK * CHUNK; e += NumThreads) {
        int i = e / CHUNK;
        int j = e % CHUNK;
        if (i <= j) {
            L_fp16(i, j) = FP16::bitcast(0);   // strictly lower kept
        }
        if (i < j) {
            Mqk(i, j) = BF16::bitcast(0);
        }
        FP16 x = L_fp16(i, j);
        if (i > j) {  // strictly lower: beta_i row scaling
            float xf = static_cast<float>(x);   // fp16 -> float
            float bi = bf16_to_f32(shared_storage.beta.begin()[i]);
            x = FP16(xf * bi);
            L_fp16(i, j) = x;   // keep the scaled value so Neumann sees L_beta
        }
        INV_fp16(i, j) = (i == j ? FP16(1.0f) - x : -x);
    }
    __syncthreads();

    // Neumann inverse. INV = (I - L)^{-1}; the 16x16 path uses one warp.
    // For CHUNK=32, block-Schur: with L = [[L00,0],[L10,L11]] (16x16 blocks),
    //   INV00 = (I+L00)^{-1}, INV11 = (I+L11)^{-1} via the 16x16 Neumann,
    //   INV10 = -INV11 @ L10 @ INV00  (scalar 16x16x16 loop, correctness-first).
    if constexpr (CHUNK == 16) {
        neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, tid);
    } else {
        // diagonal 16x16 blocks via the same 1-warp Neumann on local tiles
        {
            auto L00 = local_tile(L_fp16,  make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            auto I00 = local_tile(INV_fp16, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            auto O00 = local_tile(INV,     make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            auto L11 = local_tile(L_fp16,  make_shape(Int<16>{}, Int<16>{}), make_coord(1, 1));
            auto I11 = local_tile(INV_fp16, make_shape(Int<16>{}, Int<16>{}), make_coord(1, 1));
            auto O11 = local_tile(INV,     make_shape(Int<16>{}, Int<16>{}), make_coord(1, 1));
            if (tid < 32) {
                neumann_inv_fused_1warp(L00, I00, O00, tid);
            } else if (tid < 64) {
                neumann_inv_fused_1warp(L11, I11, O11, tid - 32);
            }
        }
        __syncthreads();
        // INV10[i,j] = -sum_k INV11[i,k] * (L10 @ INV00)[k,j]
        if (tid < 256) {
            int i = tid / 16, j = tid % 16;
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                float c = bf16_to_f32(INV(16 + i, 16 + k));   // INV11[i,k]
                float s = 0.0f;
                for (int m = 0; m < 16; ++m) {
                    s += static_cast<float>(L_fp16(16 + k, m)) * bf16_to_f32(INV(m, j));
                }
                acc -= c * s;
            }
            INV(16 + i, j) = BF16(acc);
        }
        __syncthreads();
    }

    // ---- Store to workspace ----
    auto g_ws_kd = make_tensor(ws_kd + int64_t(ws_idx) * ws_tile_elems,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_qd = make_tensor(ws_qd + int64_t(ws_idx) * ws_tile_elems,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_kr = make_tensor(ws_kr + int64_t(ws_idx) * ws_tile_elems,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_gt = make_tensor(ws_gt + int64_t(ws_idx) * ws_gt_elems,
        make_layout(make_shape(Int<D>{}), make_stride(Int<1>{})));
    auto g_ws_inv = make_tensor(ws_inv + int64_t(ws_idx) * ws_tile_lm,
        make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), make_stride(Int<CHUNK>{}, Int<1>{})));
    auto g_ws_mqk = make_tensor(ws_mqk + int64_t(ws_idx) * ws_tile_lm,
        make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), make_stride(Int<CHUNK>{}, Int<1>{})));

    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{}),
                                  g_ws_kd, tid);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{}),
                                  g_ws_qd, tid);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{}),
                                  g_ws_kr, tid);
    __syncthreads();
    coop_copy_1d_vec4<NumThreads>(make_tensor(make_smem_ptr(shared_storage.g_total.begin()), GTotalLayout{}),
                                  g_ws_gt, tid);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.INV.begin()), LMLayout{}),
                                  g_ws_inv, tid);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{}),
                                  g_ws_mqk, tid);
}

// ==================== Shared memory for recurrence kernel ====================
template <class Layouts>
struct GDNRecurrenceStorage {
    using VOLayout = typename Layouts::VOLayout;   // = MMALayout (same swizzle as readers)
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };

    struct OutputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> out;
    };

    // CHUNK>16: cross-warp staging for X = (v - kd@s)*beta and U = INV @ X.
    // Phase 3 (INV@U) couples all CHUNK rows (INV is block-triangular 32x32),
    // so the per-warp 16-row results must land in smem before the INV gemms.
    alignas(128) cute::ArrayEngine<BF16, (Layouts::kChunk > 16 ? Layouts::kChunk * Layouts::kD : 1)> u_stage;

    union {
        struct {
            InputStorage input[2];
            OutputStorage output[1];
        };
        alignas(128) char state_fp32_buf[1];
    };
};

// ==================== Shared memory for the column-split recurrence kernel ====================
// Only state/v/out are split to COLS_PER_SPLIT columns; the read-only workspace
// (kd/qd/kr/INV/Mqk/g_total) stays full-width (v1 keeps workspace loads full).
template <class Layouts, int COLS_PER_SPLIT>
struct GDNRecurrenceColsplitStorage {
    using MMALayout = typename Layouts::MMALayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    using VOLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<Layouts::kChunk>{}, Int<COLS_PER_SPLIT>{}),
        LayoutLeft{}));
    using StateSmemLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<COLS_PER_SPLIT>{}, Int<Layouts::kD>{}),
        LayoutLeft{}));
    using TransposedStateSmemLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<Layouts::kD>{}, Int<COLS_PER_SPLIT>{}),
        LayoutRight{}));

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayoutSplit>> state_acc;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayoutSplit>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };

    struct OutputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayoutSplit>> out;
    };

    union {
        struct {
            InputStorage input[2];
            OutputStorage output[1];
        };
        alignas(128) char state_fp32_buf[1];
    };
};

// ==================== Kernel 2: Recurrence ====================
template <int CHUNK, int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_recurrence_kernel(
    const BF16* __restrict__ v_ptr, int v_row_stride,    // [T,H,D] row-major
    const BF16* __restrict__ beta_ptr,                   // [H, T] head-major
    const BF16* __restrict__ ws_kd,
    const BF16* __restrict__ ws_qd,
    const BF16* __restrict__ ws_kr,
    const float* __restrict__ ws_gt,
    const BF16* __restrict__ ws_inv,
    const BF16* __restrict__ ws_mqk,
    BF16* __restrict__ out_raw_ptr,                       // [T,H,D] row-major
    const BF16* __restrict__ init_state,                  // [N,H,D,D] contiguous
    BF16* __restrict__ final_state,                       // [N,H,D,D]
    int ws_tile_elems,
    int ws_tile_lm,
    int ws_gt_elems,
    int T_total,             // B*S
    int H,
    int T_seq,               // S
    int chunks_per_seq,
    // ---- optional group-replay mode (superchunk Stage 3) ----
    int group_chunks,        // GROUP_CHUNKS when replaying a group; 0 = serial (all chunks)
    int num_groups,          // G; only meaningful when group_chunks > 0
    int prefix_exclusive     // 1: init_state[g] is exclusive prefix (Blelloch);
                             // 0: init_state[g-1] is inclusive prefix (Hillis-Steele)
) {
    using Layouts = GDNLayouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;   // = MMALayout, matches how ws tiles are read
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNRecurrenceStorage<Layouts>& shared_storage = *reinterpret_cast<GDNRecurrenceStorage<Layouts>*>(shared_mem);

    int seq = blockIdx.x;
    int head = blockIdx.y;
    const int g = blockIdx.z;          // group id (0 when serial)
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int kWarpSize = 32;
    // CHUNK=32 uses all 8 warps for MMA (2 row-blocks x 4 col-pairs);
    // CHUNK=16 keeps the original 4-warp MMA layout.
    constexpr int kComputeThreads = (CHUNK > 16) ? NumThreads : 128;
    bool is_mma = (warp_id < kComputeThreads / kWarpSize);

    int bos = seq * T_seq;
    int seq_len = T_seq;
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    int tile_base = seq * chunks_per_seq;
    const bool replay = (group_chunks > 0);
    const int t0 = replay ? g * group_chunks : 0;
    const int t1 = replay ? min(t0 + group_chunks, t_tiles) : t_tiles;

    // ---- Load initial state: zeros for serial / first group; otherwise the
    //      group's start state from prefix_B[g-1] (logical S row-major). ----
    {
        constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
        if (!replay || g == 0 || init_state == nullptr) {
            BF16* buf = shared_storage.state_acc.begin();
            for (int i = tid; i < kTotal; i += NumThreads) buf[i] = BF16(0);
        } else {
            // smem keeps S^T (Phase 6 writes s_acc_T; final store reads it back
            // via TransposedStateSmemLayout to get logical S). Load the start
            // state symmetrically through the transposed view.
            Tensor s_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
            const int64_t bh = int64_t(seq) * H + head;
            // start state of group g: inclusive prefix = transfer of 0..g-1 lives
            // at index g-1; exclusive prefix (Blelloch) at index g.
            const int start_idx = g - (prefix_exclusive ? 0 : 1);
            const BF16* src = init_state + (bh * num_groups + start_idx) * int64_t(D * D);
            for (int i = tid; i < D * D; i += NumThreads) {
                int r = i / D, c = i - r * D;
                s_T(r, c) = src[r * D + c];
            }
        }
        __syncthreads();
    }

    auto issue_loads = [&](int t, int stage) {
        int ws_idx = (seq * H + head) * chunks_per_seq + t;
        // v [CHUNK, D]
        {
            const BF16* v_base = v_ptr + int64_t(bos + t * CHUNK) * v_row_stride + head * D;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), VOLayout{});
            int v_rows = min(CHUNK, seq_len - t * CHUNK);
            constexpr int NV = D / 8;
            for (int i = tid; i < CHUNK * NV; i += NumThreads) {
                int r = i / NV;
                int c = (i - r * NV) * 8;
                cp_async_16b_zfill(&s_tile(r, c), v_base + r * v_row_stride + c, r < v_rows);
            }
        }
        // beta (32 elems, aligned 8)
        {
            int beta_linear = head * T_total + bos + t * CHUNK;
            int beta_aligned = beta_linear & ~7;
            const BF16* beta_base = beta_ptr + beta_aligned;
            BF16* s_beta = shared_storage.input[stage].beta.begin();
            int beta_rem = H * T_total - beta_aligned;
            for (int i = tid; i < 32; i += NumThreads) {
                s_beta[i] = (i < beta_rem) ? beta_base[i] : BF16(0);
            }
        }
        auto cp_ws_tile = [&](const BF16* ws_base, BF16* s_ptr, auto const& smem_layout, int rows, int cols) {
            Tensor s_tile = make_tensor(make_smem_ptr(s_ptr), smem_layout);
            int nv = cols / 8;
            for (int i = tid; i < rows * nv; i += NumThreads) {
                int r = i / nv;
                int c = (i - r * nv) * 8;
                cp_async_16b_zfill(&s_tile(r, c), ws_base + int64_t(ws_idx) * (rows * cols) + r * cols + c, true);
            }
        };
        cp_ws_tile(ws_kd, shared_storage.input[stage].k_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_qd, shared_storage.input[stage].q_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_kr, shared_storage.input[stage].k_restored.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_inv, shared_storage.input[stage].INV.begin(), LMLayout{}, CHUNK, CHUNK);
        cp_ws_tile(ws_mqk, shared_storage.input[stage].Mqk.begin(), LMLayout{}, CHUNK, CHUNK);
        // g_total (D floats)
        {
            const float* gt_base = ws_gt + int64_t(ws_idx) * ws_gt_elems;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), GTotalLayout{});
            for (int i = tid; i < D / 4; i += NumThreads) {
                cp_async_16b_zfill(&s_tile(i * 4), gt_base + i * 4, true);
            }
        }
    };

    if (t1 > t0) {
        issue_loads(t0, 0);
        cute::cp_async_fence();
    }

    for (int t = t0; t < t1; ++t) {
        const int stage = t & 1;
        if (t + 1 < t1) issue_loads(t + 1, (t + 1) & 1);
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
        __syncthreads();

        if (is_mma) {
            const int load_stage = stage;
            constexpr int out_stage = 0;

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head * T_total + bos + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});

            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int w_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[2], out_acc[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: k@s and q@s ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2 + 1, k))), tCrBi_view);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, k + 1))), tCrBi_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }

            // ======== Phase 2 ========
            asm volatile("bar.sync 8, 128;" ::: "memory");
            SFragT out_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            // beta already sigmoid'd (no sigmoid applied)
            BF16 beta0 = beta_tile(beta_smem_offset + group_id);
            BF16 beta1 = beta_tile(beta_smem_offset + group_id + 8);

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u ========
            SFragT u_bf16[2];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Mqk@U + add to out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u_arr[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // ======== Phase 5: store out ========
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: s_acc update ========
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[2][PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, warp_id * 2 + bi));
                    copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[bi][i]));
                }

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            ring_S_acc[bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c0)) * g0 + u_acc[bi](c0));
                            ring_S_acc[bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c1)) * g1 + u_acc[bi](c1));
                        }
                    }

                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id * 2 + bi));
                    copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[bi][slot]), smem_thr_store_C_T.partition_D(s_block));

                    if (m + PREFETCH < S_M_BLOCKS) {
                        Tensor s_next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, warp_id * 2 + bi));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[bi][slot]));
                    }
                }
            }
            }
        }
        __syncthreads();
        // ---- STORE output ----
        {
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.output[0].out.begin()), VOLayout{});
            if (actual_len < CHUNK) {
                int tail_elems = actual_len * D;
                for (int i = tid; i < tail_elems; i += NumThreads) {
                    int row = i / D;
                    int col = i - row * D;
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head * D;
                    out_raw_ptr[global_base + col] = s_out(row, col);
                }
            } else {
                for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                    int r = i / (D / 8);
                    int c = (i - r * (D / 8)) * 8;
                    int64_t global_base = (bos + t * CHUNK + r) * H * D + head * D;
                    *reinterpret_cast<uint4*>(&out_raw_ptr[global_base + c]) =
                        *reinterpret_cast<uint4 const*>(&s_out(r, c));
                }
            }
        }
        __syncthreads();
    }

    // ---- Store final state [N,H,D,D] contiguous: index seq*H+head ----
    // The smem state is maintained in the TRANSPOSED physical layout (Phase 6
    // writes s_acc_T; Phase 1 consumes it as the B-operand which needs S^T),
    // so reading it back row-major yields S^T. The authoritative 
    // reference (torch_chunk_gated_delta_rule / recurrent) returns the logical
    // state S, so we transpose on the way out. (fla uses a transposed
    // state convention too; the mismatch only shows up against the gdn_ref.)
    // In replay mode only the LAST group's CTA writes final_state (each CTA
    // owns the state after its own group); serial mode always writes it.
    if (final_state != nullptr && (!replay || g == num_groups - 1)) {
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
        int64_t base = int64_t(seq * H + head) * D * D;
        for (int i = tid; i < D * D; i += NumThreads) {
            int r = i / D;
            int c = i - r * D;
            final_state[base + r * D + c] = s_state(r, c);
        }
        __syncthreads();
    }
}

// ==================== Fused (workspace-free) replay path ====================
// Reset-fast-path replay with the per-chunk workspace (kd/qd/kr/INV/Mqk/gt)
// computed IN-CTA from raw q/k/g/beta instead of being prepared into ~216MB
// of global workspace by a separate kernel and read back here. The per-chunk
// prep below mirrors gdn_prepare_kernel's math op-for-op (bit-identical
// outputs), so replay results are unchanged.
//
// Pipeline per chunk (serialized prep, all 256 threads):
//   issue cp.async raw q/k/v/beta(t+1) -> fence/wait -> prep(t) in smem ->
//   recurrence(t) on warps 0-3 -> store out(t).
// Cooked tiles are single-buffered (prep of t+1 starts only after out(t) is
// stored); raw q/k and v/beta stay double-buffered.  Shared memory stays
// below the 2-CTA/SM budget of the workspace-based kernel.

// In-CTA prep of one chunk tile. Inputs: raw q/k in MMALayout smem (k_raw is
// overwritten in place by k_inv), per-token decays g_smem[CHUNK] (already
// zero-padded), beta_stage at [beta_off, beta_off+CHUNK). Outputs: kd/qd/kr
// (MMALayout), INV/Mqk (LMLayout), scalar group decay gt.
template <int CHUNK, int D, int NumThreads, class MmaL, class LML>
__device__ __forceinline__ void gdn_prep_tile(
    BF16* q_raw, BF16* k_raw,
    BF16* kd_p, BF16* qd_p, BF16* kr_p, BF16* INVp, BF16* Mqkp,
    FP16* L_f16, FP16* INV_f16,
    const BF16* g_smem,
    const BF16* beta_stage, int beta_off,
    float* gt_out, float scale, int tid)
{
    Tensor s_q   = make_tensor(make_smem_ptr(q_raw), MmaL{});
    Tensor s_k   = make_tensor(make_smem_ptr(k_raw), MmaL{});
    Tensor s_kd  = make_tensor(make_smem_ptr(kd_p),  MmaL{});
    Tensor s_qd  = make_tensor(make_smem_ptr(qd_p),  MmaL{});
    Tensor s_kr  = make_tensor(make_smem_ptr(kr_p),  MmaL{});
    Tensor s_INV = make_tensor(make_smem_ptr(INVp),  LML{});
    Tensor s_Mqk = make_tensor(make_smem_ptr(Mqkp),  LML{});
    Tensor L_fp16 = make_tensor(make_smem_ptr(L_f16), LML{});
    Tensor INV_fp16 = make_tensor(make_smem_ptr(INV_f16), LML{});

    // ---- per-token decay cumsum (fp32) ----
    // g is replicated across D in prepare; here each thread owns one row
    // (r = tid/(D/8)) and re-derives its prefix from the padded smem copy.
    const int r = tid / (D / 8);
    const int c0 = (tid % (D / 8)) * 8;
    float gtot = 0.f, gc = 0.f;
    #pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
        float gj = bf16_to_f32(g_smem[j]);
        gtot += gj;
        gc += (j <= r) ? gj : 0.f;
    }
    const float gt = expf(gtot);
    if (tid == 0) *gt_out = gt;

    // ---- decay_apply: kd/qd/ki/kr (same expressions as prepare) ----
    {
        BF16 exp_cs = BF16(expf(gc));
        BF16 inv_cs = BF16(expf(-gc));
        BF16 gt_bf = BF16(gt);
        BF16 scale_bf = BF16(scale);
        uint4 qv = *reinterpret_cast<uint4 const*>(&s_q(r, c0));
        uint4 kv = *reinterpret_cast<uint4 const*>(&s_k(r, c0));
        const BF16* qp = reinterpret_cast<const BF16*>(&qv);
        const BF16* kp = reinterpret_cast<const BF16*>(&kv);
        uint4 qdv, kdv, kiv, krv;
        BF16* qdp = reinterpret_cast<BF16*>(&qdv);
        BF16* kdp = reinterpret_cast<BF16*>(&kdv);
        BF16* kip = reinterpret_cast<BF16*>(&kiv);
        BF16* krp = reinterpret_cast<BF16*>(&krv);
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            BF16 q = qp[v], k = kp[v];
            qdp[v] = (q * exp_cs) * scale_bf;   // prepare: q * exp_cumsum * BF16(scale)
            kdp[v] = k * exp_cs;
            kip[v] = k * inv_cs;
            krp[v] = (k * inv_cs) * gt_bf;      // prepare: k * inv_cumsum * BF16(reg_gt)
        }
        *reinterpret_cast<uint4*>(&s_qd(r, c0)) = qdv;   // q_raw free after this
        *reinterpret_cast<uint4*>(&s_kd(r, c0)) = kdv;
        *reinterpret_cast<uint4*>(&s_k(r, c0))  = kiv;   // k_raw -> k_inv in place
        *reinterpret_cast<uint4*>(&s_kr(r, c0)) = krv;
    }
    __syncthreads();

    // ---- L = tril(kd @ ki^T) fp16, Mqk = tril(qd @ ki^T) bf16 (via MMA) ----
    if (tid < 32) {
        auto A = local_tile(s_kd, make_shape(Int<16>{}, Int<D>{}), make_coord(0, 0));
        auto B = local_tile(s_k,  make_shape(Int<16>{}, Int<D>{}), make_coord(0, 0));
        auto C = local_tile(L_fp16, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
        mma_m16n16_bf16bf16fp16_1warp(A, B, C, tid);
    } else if (tid < 64) {
        auto A = local_tile(s_qd, make_shape(Int<16>{}, Int<D>{}), make_coord(0, 0));
        auto B = local_tile(s_k,  make_shape(Int<16>{}, Int<D>{}), make_coord(0, 0));
        auto C = local_tile(s_Mqk, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
        mma_m16n16_bf16bf16bf16_1warp(A, B, C, tid - 32);
    }
    __syncthreads();

    // tril on L; strict-upper-zero Mqk; INV = I - L (beta row scale on L).
    for (int e = tid; e < CHUNK * CHUNK; e += NumThreads) {
        int i = e / CHUNK;
        int j = e % CHUNK;
        if (i <= j) {
            L_fp16(i, j) = FP16::bitcast(0);
        }
        if (i < j) {
            s_Mqk(i, j) = BF16::bitcast(0);
        }
        FP16 x = L_fp16(i, j);
        if (i > j) {
            float xf = static_cast<float>(x);
            float bi = bf16_to_f32(beta_stage[beta_off + i]);
            x = FP16(xf * bi);
            L_fp16(i, j) = x;
        }
        INV_fp16(i, j) = (i == j ? FP16(1.0f) - x : -x);
    }
    __syncthreads();

    // Neumann inverse INV = (I - L)^{-1} (single warp).
    if (tid < 32) {
        neumann_inv_fused_1warp(L_fp16, INV_fp16, s_INV, tid);
    }
    __syncthreads();
}

template <class Layouts>
struct GDNRecurrenceFusedStorage {
    using VOLayout = typename Layouts::VOLayout;   // = MMALayout
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;

    struct RawStage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q;  // raw q tile
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k;  // raw k -> k_inv
    };
    struct InStage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(8) cute::ArrayEngine<BF16, Layouts::kChunk> g_raw;   // zero-padded token decays
    };
    RawStage raw[2];
    InStage in[2];

    // cooked tiles of the CURRENT chunk (prep is serialized per chunk)
    struct Cooked {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    } cooked;
    float gt;   // scalar exp(total decay); ws g_total is uniform across D

    union {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> out;
        struct {
            alignas(128) FP16 L_fp16[16 * 16];
            alignas(128) FP16 INV_fp16[16 * 16];
        } lm;
    } u;
};

// Fused group replay (reset fast path only): identical recurrence math to
// gdn_recurrence_kernel replay mode, but the per-chunk workspace tiles are
// produced in-CTA by gdn_prep_tile instead of being read from global ws.
// grid = (B, H, num_groups); each CTA replays its group starting from the
// previous group's transfer matrix B_g[g-1] (shift semantics, prefix=0).
template <int CHUNK, int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads, 2) gdn_recurrence_fused_kernel(
    const BF16* __restrict__ q_ptr, const BF16* __restrict__ k_ptr,
    int qk_row_stride,
    const BF16* __restrict__ v_ptr, int v_row_stride,
    const BF16* __restrict__ g_ptr, int g_row_stride,   // [H, T_total]
    const BF16* __restrict__ beta_ptr,
    BF16* __restrict__ out_raw_ptr,                      // [T,H,D] row-major
    const BF16* __restrict__ prefix_B,                   // [B*H*G, D, D]
    BF16* __restrict__ final_state,                      // [B*H, D, D]
    float scale,
    int T_total, int H, int T_seq, int chunks_per_seq,
    int group_chunks, int num_groups, int head_ratio) {
    using Layouts = GDNLayouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using LMLayout = typename Layouts::LMLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNRecurrenceFusedStorage<Layouts>& shared_storage =
        *reinterpret_cast<GDNRecurrenceFusedStorage<Layouts>*>(shared_mem);

    int seq = blockIdx.x;
    int head = blockIdx.y;
    const int g = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int kComputeThreads = 128;   // warps 0-3 do the MMA phases
    bool is_mma = (warp_id < kComputeThreads / 32);

    int bos = seq * T_seq;
    int seq_len = T_seq;
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    const int t0 = g * group_chunks;
    const int t1 = min(t0 + group_chunks, t_tiles);
    const int qk_head = head / head_ratio;

    // ---- initial state: zeros for g == 0, else B_g[g-1] (shift path) ----
    {
        constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
        if (g == 0) {
            BF16* buf = shared_storage.state_acc.begin();
            for (int i = tid; i < kTotal; i += NumThreads) buf[i] = BF16(0);
        } else {
            Tensor s_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
            const int64_t bh = int64_t(seq) * H + head;
            const BF16* src = prefix_B + (bh * num_groups + (g - 1)) * int64_t(D * D);
            for (int i = tid; i < D * D; i += NumThreads) {
                int r = i / D, c = i - r * D;
                s_T(r, c) = src[r * D + c];
            }
        }
        __syncthreads();
    }

    auto issue_loads = [&](int t, int stage) {
        // raw q/k [CHUNK, D] (GQA source head), cp.async + zfill tail
        {
            const BF16* q_base = q_ptr + int64_t(bos + t * CHUNK) * qk_row_stride + qk_head * D;
            const BF16* k_base = k_ptr + int64_t(bos + t * CHUNK) * qk_row_stride + qk_head * D;
            Tensor s_q = make_tensor(make_smem_ptr(shared_storage.raw[stage].q.begin()), MMALayout{});
            Tensor s_k = make_tensor(make_smem_ptr(shared_storage.raw[stage].k.begin()), MMALayout{});
            int rows = min(CHUNK, seq_len - t * CHUNK);
            for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                int r = i / (D / 8);
                int c = (i - r * (D / 8)) * 8;
                cp_async_16b_zfill(&s_q(r, c), q_base + r * qk_row_stride + c, r < rows);
                cp_async_16b_zfill(&s_k(r, c), k_base + r * qk_row_stride + c, r < rows);
            }
        }
        // v [CHUNK, D]
        {
            const BF16* v_base = v_ptr + int64_t(bos + t * CHUNK) * v_row_stride + head * D;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.in[stage].v.begin()), VOLayout{});
            int v_rows = min(CHUNK, seq_len - t * CHUNK);
            for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                int r = i / (D / 8);
                int c = (i - r * (D / 8)) * 8;
                cp_async_16b_zfill(&s_tile(r, c), v_base + r * v_row_stride + c, r < v_rows);
            }
        }
        // beta (32 elems, aligned 8)
        {
            int beta_linear = head * T_total + bos + t * CHUNK;
            int beta_aligned = beta_linear & ~7;
            const BF16* beta_base = beta_ptr + beta_aligned;
            BF16* s_beta = shared_storage.in[stage].beta.begin();
            int beta_rem = H * T_total - beta_aligned;
            for (int i = tid; i < 32; i += NumThreads) {
                s_beta[i] = (i < beta_rem) ? beta_base[i] : BF16(0);
            }
        }
        // g (16 token decays, zero-padded tail; plain loads issued a full
        // chunk ahead so the global latency is hidden behind prep/recurrence)
        {
            int valid = seq_len - t * CHUNK;
            const BF16* g_base = g_ptr + int64_t(head) * g_row_stride + bos + t * CHUNK;
            BF16* s_g = shared_storage.in[stage].g_raw.begin();
            for (int i = tid; i < CHUNK; i += NumThreads) {
                BF16 gv = g_base[min(i, valid - 1)];
                s_g[i] = (i < valid) ? gv : BF16(0.f);
            }
        }
    };

    if (t1 > t0) {
        issue_loads(t0, 0);
        cute::cp_async_fence();
    }

    for (int t = t0; t < t1; ++t) {
        const int stage = (t - t0) & 1;
        if (t + 1 < t1) issue_loads(t + 1, stage ^ 1);
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
        __syncthreads();

        // ---- fused in-CTA workspace prep for chunk t ----
        {
            int beta_off = (head * T_total + bos + t * CHUNK) & 7;
            gdn_prep_tile<CHUNK, D, NumThreads, MMALayout, LMLayout>(
                shared_storage.raw[stage].q.begin(),
                shared_storage.raw[stage].k.begin(),
                shared_storage.cooked.k_decayed.begin(),
                shared_storage.cooked.q_decayed.begin(),
                shared_storage.cooked.k_restored.begin(),
                shared_storage.cooked.INV.begin(),
                shared_storage.cooked.Mqk.begin(),
                shared_storage.u.lm.L_fp16,
                shared_storage.u.lm.INV_fp16,
                shared_storage.in[stage].g_raw.begin(),
                shared_storage.in[stage].beta.begin(), beta_off,
                &shared_storage.gt, scale, tid);
        }

        if (is_mma) {
            constexpr int out_stage = 0;

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.in[stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.in[stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head * T_total + bos + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.u.out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.cooked.k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.cooked.q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.cooked.k_restored.begin()), MMALayout{});
            float gt = shared_storage.gt;
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.cooked.INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.cooked.Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});

            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.cooked.k_restored.begin()), typename Layouts::TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int lane_id = threadIdx.x % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[2], out_acc[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: k@s and q@s ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2 + 1, k))), tCrBi_view);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, k + 1))), tCrBi_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }

            // ======== Phase 2 ========
            asm volatile("bar.sync 8, 128;" ::: "memory");
            SFragT out_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = beta_tile(beta_smem_offset + group_id);
            BF16 beta1 = beta_tile(beta_smem_offset + group_id + 8);

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u ========
            SFragT u_bf16[2];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Mqk@U + add to out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u_arr[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // ======== Phase 5: store out ========
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: s_acc update ========
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[2][PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, warp_id * 2 + bi));
                    copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[bi][i]));
                }

                // ws g_total is uniform across D: both lanes use the scalar gt
                ring_g0[i] = gt;
                ring_g1[i] = gt;
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = gt;
                    ring_g1[slot] = gt;
                }

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            ring_S_acc[bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c0)) * g0 + u_acc[bi](c0));
                            ring_S_acc[bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c1)) * g1 + u_acc[bi](c1));
                        }
                    }

                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id * 2 + bi));
                    copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[bi][slot]), smem_thr_store_C_T.partition_D(s_block));

                    if (m + PREFETCH < S_M_BLOCKS) {
                        Tensor s_next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, warp_id * 2 + bi));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[bi][slot]));
                    }
                }
            }
            }
        }
        __syncthreads();
        // ---- STORE output ----
        {
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.u.out.begin()), VOLayout{});
            if (actual_len < CHUNK) {
                int tail_elems = actual_len * D;
                for (int i = tid; i < tail_elems; i += NumThreads) {
                    int row = i / D;
                    int col = i - row * D;
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head * D;
                    out_raw_ptr[global_base + col] = s_out(row, col);
                }
            } else {
                for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                    int r = i / (D / 8);
                    int c = (i - r * (D / 8)) * 8;
                    int64_t global_base = (bos + t * CHUNK + r) * H * D + head * D;
                    *reinterpret_cast<uint4*>(&out_raw_ptr[global_base + c]) =
                        *reinterpret_cast<uint4 const*>(&s_out(r, c));
                }
            }
        }
        __syncthreads();
    }

    // ---- Store final state [B*H, D, D] (transposed read, as in recurrence) ----
    if (g == num_groups - 1) {
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
        int64_t base = int64_t(seq * H + head) * D * D;
        for (int i = tid; i < D * D; i += NumThreads) {
            int r = i / D;
            int c = i - r * D;
            final_state[base + r * D + c] = s_state(r, c);
        }
        __syncthreads();
    }
}
// A self-contained CHUNK=32 recurrence. 8 warps (256 threads). Each warp w:
//   mb = w>>2   (16-row half of the chunk: 0 or 1)
//   nb = w&3    (state-column-block pair index 0..3)
//   handles state n-blocks {2*nb, 2*nb+1}.
// Cross-warp coupling (block-triangular INV / Mqk) is staged through the
// `u_stage` smem tile (CHUNK x D). This keeps the CHUNK=16 path untouched.
template <int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_recurrence_kernel32(
    const BF16* __restrict__ v_ptr, int v_row_stride,    // [T,H,D] row-major
    const BF16* __restrict__ beta_ptr,                   // [H, T] head-major
    const BF16* __restrict__ ws_kd,
    const BF16* __restrict__ ws_qd,
    const BF16* __restrict__ ws_kr,
    const float* __restrict__ ws_gt,
    const BF16* __restrict__ ws_inv,
    const BF16* __restrict__ ws_mqk,
    BF16* __restrict__ out_raw_ptr,                      // [T,H,D] row-major
    const BF16* __restrict__ init_state,                 // [N,H,D,D]
    BF16* __restrict__ final_state,                      // [N,H,D,D]
    int ws_tile_elems,
    int ws_tile_lm,
    int ws_gt_elems,
    int T_total,             // B*S
    int H,
    int T_seq,               // S
    int chunks_per_seq
) {
    constexpr int CHUNK = 32;
    using Layouts = GDNLayouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNRecurrenceStorage<Layouts>& shared_storage = *reinterpret_cast<GDNRecurrenceStorage<Layouts>*>(shared_mem);

    int seq = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = NumThreads;
    bool is_mma = (warp_id < kComputeThreads / kWarpSize);

    const int mb = warp_id >> 2;   // 0 or 1
    const int nb = warp_id & 3;    // 0..3
    const int nblk0 = 2 * nb;      // first state n-block
    const int nblk1 = 2 * nb + 1;

    int bos = seq * T_seq;
    int seq_len = T_seq;
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    int tile_base = seq * chunks_per_seq;

    // ---- Load initial state (zero) ----
    {
        BF16* buf = shared_storage.state_acc.begin();
        constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
        for (int i = tid; i < kTotal; i += NumThreads) {
            buf[i] = BF16(0);
        }
        __syncthreads();
    }

    auto issue_loads = [&](int t, int stage) {
        int ws_idx = (seq * H + head) * chunks_per_seq + t;
        // v [CHUNK, D]
        {
            const BF16* v_base = v_ptr + int64_t(bos + t * CHUNK) * v_row_stride + head * D;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), VOLayout{});
            int v_rows = min(CHUNK, seq_len - t * CHUNK);
            constexpr int NV = D / 8;
            for (int i = tid; i < CHUNK * NV; i += NumThreads) {
                int r = i / NV;
                int c = (i - r * NV) * 8;
                cp_async_16b_zfill(&s_tile(r, c), v_base + r * v_row_stride + c, r < v_rows);
            }
        }
        // beta (CHUNK tokens at CHUNK-aligned base; load CHUNK+8 to cover alignment)
        {
            int beta_linear = head * T_total + bos + t * CHUNK;
            int beta_aligned = beta_linear & ~7;
            const BF16* beta_base = beta_ptr + beta_aligned;
            BF16* s_beta = shared_storage.input[stage].beta.begin();
            int beta_rem = H * T_total - beta_aligned;
            for (int i = tid; i < CHUNK + 8; i += NumThreads) {
                s_beta[i] = (i < beta_rem) ? beta_base[i] : BF16(0);
            }
        }
        auto cp_ws_tile = [&](const BF16* ws_base, BF16* s_ptr, auto const& smem_layout, int rows, int cols) {
            Tensor s_tile = make_tensor(make_smem_ptr(s_ptr), smem_layout);
            int nv = cols / 8;
            for (int i = tid; i < rows * nv; i += NumThreads) {
                int r = i / nv;
                int c = (i - r * nv) * 8;
                cp_async_16b_zfill(&s_tile(r, c), ws_base + int64_t(ws_idx) * (rows * cols) + r * cols + c, true);
            }
        };
        cp_ws_tile(ws_kd, shared_storage.input[stage].k_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_qd, shared_storage.input[stage].q_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_kr, shared_storage.input[stage].k_restored.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(ws_inv, shared_storage.input[stage].INV.begin(), LMLayout{}, CHUNK, CHUNK);
        cp_ws_tile(ws_mqk, shared_storage.input[stage].Mqk.begin(), LMLayout{}, CHUNK, CHUNK);
        // g_total (D floats)
        {
            const float* gt_base = ws_gt + int64_t(ws_idx) * ws_gt_elems;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), GTotalLayout{});
            for (int i = tid; i < D / 4; i += NumThreads) {
                cp_async_16b_zfill(&s_tile(i * 4), gt_base + i * 4, true);
            }
        }
    };

    if (t_tiles > 0) {
        issue_loads(0, 0);
        cute::cp_async_fence();
    }

    for (int t = 0; t < t_tiles; ++t) {
        const int stage = t & 1;
        if (t + 1 < t_tiles) issue_loads(t + 1, (t + 1) & 1);
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
        __syncthreads();

        if (is_mma) {
            const int load_stage = stage;
            constexpr int out_stage = 0;

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), Layout<Shape<Int<CHUNK+8>>, Stride<Int<1>>>{});
            int beta_smem_offset = (head * T_total + bos + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});
            Tensor u_stage = make_tensor(make_smem_ptr(shared_storage.u_stage.begin()), MMALayout{});

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int w_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            using AccFragT = decltype(thr_mma.make_fragment_C(thr_mma.partition_C(C_ref)));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(thr_mma.partition_C(C_ref))));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[2], out_acc[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(thr_mma.partition_C(C_ref)); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(thr_mma.partition_C(C_ref)); clear(out_acc[i]); }

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);
            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);
            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            // ======== Phase 1: u_acc = kd[mb]@S, out_acc = qd[mb]@S (n-blocks nblk0,nblk1) ========
            constexpr int K_BLOCKS = D / 16;  // 8
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(nblk0, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(nblk1, k))), tCrBi_view);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(nblk0, k + 1))), tCrBi_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }

            // ======== Phase 2: X = (v - u) * beta; stage X into u_stage ========
            SFragT out_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16[2], u_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, nblk0 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }
            // beta for this warp's m-half (tokens 16*mb .. 16*mb+15)
            BF16 beta0 = beta_tile(beta_smem_offset + 16 * mb + group_id);
            BF16 beta1 = beta_tile(beta_smem_offset + 16 * mb + group_id + 8);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }
                // stage X = u_bf16 to u_stage[mb rows, nblk0+i cols]
                Tensor x_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, nblk0 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(u_bf16[i]), smem_thr_store_C.partition_D(x_block));
            }
            __syncthreads();

            // ======== Phase 3: U = INV @ X (block-triangular, staged) ========
            // X was staged into u_stage via store_C (C-layout). Read it back via
            // load_C (consistent) and convert C-fragment -> B-fragment in-register
            // (the proven CHUNK=16 Phase-3 transpose) before the INV gemms.
            auto c_to_b = [&](SFragT const& c_frag) {
                BFragT_u b = thr_mma.partition_fragment_B(B_ref);
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

            SFragT X0C[2], X1C[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor x0_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(0, nblk0 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(x0_block), smem_thr_load_C.retile_D(X0C[i]));
            }
            if (mb == 1) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    Tensor x1_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(1, nblk0 + i));
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(x1_block), smem_thr_load_C.retile_D(X1C[i]));
                }
            }
            __syncthreads();

            // compute U[mb rows, nblk0+i] and stage back into u_stage
            SFragT U_frag[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                clear(u_acc[i]);
                // diagonal: INV_mbmb @ X[mb]
                {
                    auto inv_blk = local_tile(INV, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, mb));
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(inv_blk), tCrAi_k_view);
                    cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                    BFragT_u xb = c_to_b((mb == 0) ? X0C[i] : X1C[i]);
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), xb(_,_,Int<0>{}), u_acc[i]);
                }
                if (mb == 1) {
                    // off-diagonal: INV10 @ X[0]
                    auto inv10 = local_tile(INV, make_shape(Int<16>{}, Int<16>{}), make_coord(1, 0));
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(inv10), tCrAi_q_view);
                    cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                    BFragT_u xb0 = c_to_b(X0C[i]);
                    gemm(thr_mma, tCrA_q(_,_,Int<0>{}), xb0(_,_,Int<0>{}), u_acc[i]);
                }
                cute::transform(u_acc[i], U_frag[i], [] __device__ (float x) { return BF16(x); });
                Tensor u_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, nblk0 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(U_frag[i]), smem_thr_store_C.partition_D(u_block));
            }
            __syncthreads();

            // ======== Phase 4: out += Mqk @ U ========
            // Mqk is LOWER block-triangular (Mqk01 = 0). So:
            //   out[0:16]  += Mqk00 @ U0
            //   out[16:32] += Mqk11 @ U1 + Mqk10 @ U0
            SFragT U0C[2], U1C[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor u0_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(0, nblk0 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(u0_block), smem_thr_load_C.retile_D(U0C[i]));
            }
            if (mb == 1) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    Tensor u1_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(1, nblk0 + i));
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(u1_block), smem_thr_load_C.retile_D(U1C[i]));
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                clear(out_acc[i]);
                // diagonal: Mqk_mbmb @ U_mb
                auto mqk_blk = local_tile(Mqk, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, mb));
                copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(mqk_blk), tCrAi_k_view);
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                BFragT_u ub = c_to_b((mb == 0) ? U0C[i] : U1C[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), ub(_,_,Int<0>{}), out_acc[i]);
                if (mb == 1) {
                    // off-diagonal: Mqk10 @ U0
                    auto mqk10 = local_tile(Mqk, make_shape(Int<16>{}, Int<16>{}), make_coord(1, 0));
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(mqk10), tCrAi_q_view);
                    cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                    BFragT_u ub0 = c_to_b(U0C[i]);
                    gemm(thr_mma, tCrA_q(_,_,Int<0>{}), ub0(_,_,Int<0>{}), out_acc[i]);
                }
                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // ======== Phase 5: store out ========
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(mb, nblk0 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: S[rb,n] = S[rb,n]*exp(g_total[rb]) + sum_c kr^T[rb,c]*U[c,n] ========
            // warp handles state row-block rb = warp_id (0..7), all 8 n-blocks.
            constexpr int S_N_BLOCKS = D / 16;  // 8
            const int rb = warp_id;             // 0..7

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);
            AFragT kr_A[2];
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(rb, c));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, kr_A[c], cute::identity{});
            }
            float g0 = g_total(rb * 16 + group_id);
            float g1 = g_total(rb * 16 + group_id + 8);

            #pragma unroll
            for (int nblk = 0; nblk < S_N_BLOCKS; ++nblk) {
                // B = U[c, nblk], c=0,1 (from u_stage, via load_C + C->B convert)
                SFragT uC[2];
                BFragT_u uB[2];
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    Tensor u_block = local_tile(u_stage, make_shape(Int<16>{}, Int<16>{}), make_coord(c, nblk));
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(u_block), smem_thr_load_C.retile_D(uC[c]));
                    uB[c] = c_to_b(uC[c]);
                }
                clear(u_acc[0]);
                gemm(thr_mma, kr_A[0](_,_,Int<0>{}), uB[0](_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, kr_A[1](_,_,Int<0>{}), uB[1](_,_,Int<0>{}), u_acc[0]);

                // load old S[rb, nblk], decay, add U contribution
                Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(rb, nblk));
                SFragT S_frag;
                copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(S_frag));
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        S_frag(c0) = BF16(bf16_to_f32(S_frag(c0)) * g0 + u_acc[0](c0));
                        S_frag(c1) = BF16(bf16_to_f32(S_frag(c1)) * g1 + u_acc[0](c1));
                    }
                }
                copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(S_frag), smem_thr_store_C_T.partition_D(s_block));
            }
        }
        __syncthreads();
        // ---- STORE output ----
        {
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.output[0].out.begin()), VOLayout{});
            if (actual_len < CHUNK) {
                int tail_elems = actual_len * D;
                for (int i = tid; i < tail_elems; i += NumThreads) {
                    int row = i / D;
                    int col = i - row * D;
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head * D;
                    out_raw_ptr[global_base + col] = s_out(row, col);
                }
            } else {
                for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                    int r = i / (D / 8);
                    int c = (i - r * (D / 8)) * 8;
                    int64_t global_base = (bos + t * CHUNK + r) * H * D + head * D;
                    *reinterpret_cast<uint4*>(&out_raw_ptr[global_base + c]) =
                        *reinterpret_cast<uint4 const*>(&s_out(r, c));
                }
            }
        }
        __syncthreads();
    }

    // ---- Store final state [N,H,D,D] contiguous ----
    {
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});
        int64_t base = int64_t(seq * H + head) * D * D;
        for (int i = tid; i < D * D; i += NumThreads) {
            int r = i / D;
            int c = i - r * D;
            final_state[base + r * D + c] = s_state(r, c);
        }
        __syncthreads();
    }
    (void)init_state;
}

// ==================== Kernel 2c: Recurrence, column-split (P1) ====================
// Grid = (B, H, SPLIT). Each CTA owns D/SPLIT state/output columns. The state
// update S[:, C] depends only on the same column group C, so splits are fully
// independent (numpy-verified bit-identical). kd/qd/kr/INV/Mqk/g_total are
// shared read-only inputs loaded full-width (v1 keeps workspace loads full);
// only state/v/out smem are split to reduce per-CTA smem for occupancy.
//
// SPLIT=4: COLS_PER_SPLIT=32, kNBlocks=2 MMA warps (warps 0,1), each owns one
// 16-column block. warp_id maps directly to the local 16-col block.
template <int CHUNK, int D, int NumThreads, int SPLIT>
__global__ void __launch_bounds__(NumThreads) gdn_recurrence_colsplit_kernel(
    const BF16* __restrict__ v_ptr, int v_row_stride,    // [T,H,D] row-major
    const BF16* __restrict__ beta_ptr,                   // [H, T] head-major
    const BF16* __restrict__ ws_kd,
    const BF16* __restrict__ ws_qd,
    const BF16* __restrict__ ws_kr,
    const float* __restrict__ ws_gt,
    const BF16* __restrict__ ws_inv,
    const BF16* __restrict__ ws_mqk,
    BF16* __restrict__ out_raw_ptr,                      // [T,H,D] row-major
    const BF16* __restrict__ init_state,                 // [N,H,D,D]
    BF16* __restrict__ final_state,                      // [N,H,D,D]
    int ws_tile_elems,
    int ws_tile_lm,
    int ws_gt_elems,
    int T_total,             // B*S
    int H,
    int T_seq,               // S
    int chunks_per_seq
) {
    static_assert(D % SPLIT == 0);
    constexpr int COLS_PER_SPLIT = D / SPLIT;
    constexpr int kNBlocks = COLS_PER_SPLIT / 16;   // 2 for SPLIT=4
    static_assert(COLS_PER_SPLIT % 16 == 0);

    using Layouts = GDNLayouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    // Split-width smem layouts: state is [N=COLS_PER_SPLIT, K=D] (S^T) when read
    // as the MMA B-operand; the transposed write view is [K=D, N=COLS_PER_SPLIT].
    using StateSmemLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<COLS_PER_SPLIT>{}, Int<D>{}),
        LayoutLeft{}));
    using TransposedStateSmemLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<COLS_PER_SPLIT>{}),
        LayoutRight{}));
    using VOLayoutSplit = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<COLS_PER_SPLIT>{}),
        LayoutLeft{}));

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNRecurrenceColsplitStorage<Layouts, COLS_PER_SPLIT>& shared_storage =
        *reinterpret_cast<GDNRecurrenceColsplitStorage<Layouts, COLS_PER_SPLIT>*>(shared_mem);

    int seq = blockIdx.x;
    int head = blockIdx.y;
    int split = blockIdx.z;
    int col0 = split * COLS_PER_SPLIT;     // global column offset
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int kWarpSize = 32;
    bool is_mma = (warp_id < kNBlocks);    // only kNBlocks MMA warps

    int bos = seq * T_seq;
    int seq_len = T_seq;
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    int tile_base = seq * chunks_per_seq;

    // ---- Load initial state (zero) ----
    {
        BF16* buf = shared_storage.state_acc.begin();
        constexpr int kTotal = cute::cosize_v<StateSmemLayoutSplit>;
        for (int i = tid; i < kTotal; i += NumThreads) {
            buf[i] = BF16(0);
        }
        __syncthreads();
    }

    auto issue_loads = [&](int t, int stage) {
        int ws_idx = (seq * H + head) * chunks_per_seq + t;
        // v [CHUNK, COLS_PER_SPLIT]: load only local columns [col0, col0+COLS_PER_SPLIT)
        {
            const BF16* v_base = v_ptr + int64_t(bos + t * CHUNK) * v_row_stride + head * D;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), VOLayoutSplit{});
            int v_rows = min(CHUNK, seq_len - t * CHUNK);
            constexpr int NV = COLS_PER_SPLIT / 8;
            for (int i = tid; i < CHUNK * NV; i += NumThreads) {
                int r = i / NV;
                int c = (i - r * NV) * 8;
                cp_async_16b_zfill(&s_tile(r, c), v_base + r * v_row_stride + col0 + c, r < v_rows);
            }
        }
        // beta (32 elems, aligned 8)
        {
            int beta_linear = head * T_total + bos + t * CHUNK;
            int beta_aligned = beta_linear & ~7;
            const BF16* beta_base = beta_ptr + beta_aligned;
            BF16* s_beta = shared_storage.input[stage].beta.begin();
            int beta_rem = H * T_total - beta_aligned;
            for (int i = tid; i < 32; i += NumThreads) {
                s_beta[i] = (i < beta_rem) ? beta_base[i] : BF16(0);
            }
        }
        auto cp_ws_tile = [&](const BF16* ws_base, BF16* s_ptr, auto const& smem_layout, int rows, int cols) {
            Tensor s_tile = make_tensor(make_smem_ptr(s_ptr), smem_layout);
            int nv = cols / 8;
            for (int i = tid; i < rows * nv; i += NumThreads) {
                int r = i / nv;
                int c = (i - r * nv) * 8;
                cp_async_16b_zfill(&s_tile(r, c), ws_base + int64_t(ws_idx) * (rows * cols) + r * cols + c, true);
            }
        };
        // Workspace loads stay FULL (shared read-only inputs); only v/out/state are split.
        cp_ws_tile(ws_kd, shared_storage.input[stage].k_decayed.begin(), MMALayout{}, CHUNK, D);
        cp_ws_tile(ws_qd, shared_storage.input[stage].q_decayed.begin(), MMALayout{}, CHUNK, D);
        cp_ws_tile(ws_kr, shared_storage.input[stage].k_restored.begin(), MMALayout{}, CHUNK, D);
        cp_ws_tile(ws_inv, shared_storage.input[stage].INV.begin(), LMLayout{}, CHUNK, CHUNK);
        cp_ws_tile(ws_mqk, shared_storage.input[stage].Mqk.begin(), LMLayout{}, CHUNK, CHUNK);
        // g_total (D floats)
        {
            const float* gt_base = ws_gt + int64_t(ws_idx) * ws_gt_elems;
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), GTotalLayout{});
            for (int i = tid; i < D / 4; i += NumThreads) {
                cp_async_16b_zfill(&s_tile(i * 4), gt_base + i * 4, true);
            }
        }
    };

    if (t_tiles > 0) {
        issue_loads(0, 0);
        cute::cp_async_fence();
    }

    for (int t = 0; t < t_tiles; ++t) {
        const int stage = t & 1;
        if (t + 1 < t_tiles) issue_loads(t + 1, (t + 1) & 1);
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
        __syncthreads();

        if (is_mma) {
            const int load_stage = stage;
            constexpr int out_stage = 0;

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayoutSplit{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head * T_total + bos + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayoutSplit{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayoutSplit{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayoutSplit{});

            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int w_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc);
            AccFragT out_acc = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc);

            // ======== Phase 1: k@s and q@s (single 16-col block per warp) ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;  // 8

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, k + 1))), tCrBi_view);
                }
            }

            // ======== Phase 2 ========
            // bar.sync count must be a compile-time immediate = number of MMA threads.
            asm volatile("bar.sync 8, %0;" :: "n"(kNBlocks * 32) : "memory");
            SFragT out_bf16;
            cute::transform(out_acc, out_bf16, [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16;
            {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            // beta already sigmoid'd (no sigmoid applied)
            BF16 beta0 = beta_tile(beta_smem_offset + group_id);
            BF16 beta1 = beta_tile(beta_smem_offset + group_id + 8);

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u ========
            SFragT u_bf16;
            uint32_t u_b_regs[4];

            cute::transform(u_acc, u_bf16, [] __device__ (float x) { return BF16(x); });

            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    auto c0 = make_coord(make_coord(a, 0), 0, d);
                    auto c1 = make_coord(make_coord(a, 1), 0, d);
                    u_bf16(c0) = (v_bf16(c0) - u_bf16(c0)) * beta0;
                    u_bf16(c1) = (v_bf16(c1) - u_bf16(c1)) * beta1;
                }
            }

            uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));
            SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
            SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
            SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
            SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

            auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
            uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
            b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
            b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

            clear(u_acc);
            gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc);

            cute::transform(u_acc, u_bf16, [] __device__ (float x) { return BF16(x); });

            // ======== Phase 4: Mqk@U + add to out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u;
            {
                uint32_t* u_c2 = reinterpret_cast<uint32_t*>(&u_bf16(0));
                SM75_U32x1_MOVM_T::copy(u_c2[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c2[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c2[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c2[3], u_b_regs[3]);

                tCrB_u = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst2 = reinterpret_cast<uint32_t*>(&tCrB_u(0));
                b_dst2[0] = u_b_regs[0]; b_dst2[1] = u_b_regs[1];
                b_dst2[2] = u_b_regs[2]; b_dst2[3] = u_b_regs[3];
            }

            clear(out_acc);
            gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u(_,_,Int<0>{}), out_acc);

            SFragT gemm_bf16;
            cute::transform(out_acc, gemm_bf16, [] __device__ (float x) { return BF16(x); });
            cute::transform(out_bf16, gemm_bf16, out_bf16, [] __device__ (BF16 c, BF16 a) { return c + a; });

            // ======== Phase 5: store out (local columns) ========
            {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: s_acc update (single 16-col block per warp) ========
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;  // 8

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, warp_id));
                copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[i]));

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                clear(u_acc);
                gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u(_,_,Int<0>{}), u_acc);

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        ring_S_acc[slot](c0) = BF16(bf16_to_f32(ring_S_acc[slot](c0)) * g0 + u_acc(c0));
                        ring_S_acc[slot](c1) = BF16(bf16_to_f32(ring_S_acc[slot](c1)) * g1 + u_acc(c1));
                    }
                }

                Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id));
                copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[slot]), smem_thr_store_C_T.partition_D(s_block));

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor s_next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, warp_id));
                    copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[slot]));
                }
            }
            }
        }
        __syncthreads();
        // ---- STORE output (local columns + col0) ----
        {
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.output[0].out.begin()), VOLayoutSplit{});
            if (actual_len < CHUNK) {
                int tail_elems = actual_len * COLS_PER_SPLIT;
                for (int i = tid; i < tail_elems; i += NumThreads) {
                    int row = i / COLS_PER_SPLIT;
                    int col = i - row * COLS_PER_SPLIT;
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head * D;
                    out_raw_ptr[global_base + col0 + col] = s_out(row, col);
                }
            } else {
                for (int i = tid; i < CHUNK * (COLS_PER_SPLIT / 8); i += NumThreads) {
                    int r = i / (COLS_PER_SPLIT / 8);
                    int c = (i - r * (COLS_PER_SPLIT / 8)) * 8;
                    int64_t global_base = (bos + t * CHUNK + r) * H * D + head * D;
                    *reinterpret_cast<uint4*>(&out_raw_ptr[global_base + col0 + c]) =
                        *reinterpret_cast<uint4 const*>(&s_out(r, c));
                }
            }
        }
        __syncthreads();
    }

    // ---- Store final state [N,H,D,D] contiguous (local columns + col0) ----
    {
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayoutSplit{});
        int64_t base = int64_t(seq * H + head) * D * D;
        for (int i = tid; i < D * COLS_PER_SPLIT; i += NumThreads) {
            int r = i / COLS_PER_SPLIT;
            int c = i - r * COLS_PER_SPLIT;
            final_state[base + r * D + col0 + c] = s_state(r, c);
        }
        __syncthreads();
    }
    (void)init_state;
    (void)tile_base;
}

// ==================== C launcher (extern "C") ====================
// Host entry used by the torch wrapper (gdn_ops.cu). Lives here so it can
// instantiate/launch the __global__ templates within this cute-only TU,
// avoiding cross-TU relocatable-device-code linking.

#ifndef GDN_CHUNK
#define GDN_CHUNK 16
#define GDN_D 128
#define GDN_NUM_THREADS 256
#endif

namespace {
size_t prepare_smem_bytes_impl() {
    // Use sizeof of the actual storage struct so alignas(128) padding is included.
    return sizeof(GDNPrepareStorage<GDNLayouts<GDN_D, GDN_CHUNK>>);
}
size_t recurrence_smem_bytes_impl() {
    // Use sizeof of the actual storage struct so alignas(128) padding is included.
    return sizeof(GDNRecurrenceStorage<GDNLayouts<GDN_D, GDN_CHUNK>>);
}
}

extern "C" void gdn_chunk_forward(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* g,
    const cutlass::bfloat16_t* beta,
    cutlass::bfloat16_t* ws_kd, cutlass::bfloat16_t* ws_qd,
    cutlass::bfloat16_t* ws_kr, float* ws_gt,
    cutlass::bfloat16_t* ws_inv, cutlass::bfloat16_t* ws_mqk,
    cutlass::bfloat16_t* out, cutlass::bfloat16_t* final_state,
    int q_row_stride, int g_row_stride,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    float scale, int T_seq, int H, int B, int chunks_per_seq,
    cudaStream_t stream) {
    const int T_total = B * T_seq;

    // The recurrence kernel uses >48KB of dynamic shared memory on sm_80.
    // Opt into the larger dynamic-smem allocation before launching.
    const size_t rec_smem = recurrence_smem_bytes_impl();
    const size_t prep_smem = prepare_smem_bytes_impl();
    cudaFuncSetAttribute(gdn_recurrence_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)rec_smem);
    cudaFuncSetAttribute(gdn_prepare_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);

    // Kernel 1: prepare
    {
        dim3 grid(chunks_per_seq, B * H);
        dim3 block(GDN_NUM_THREADS);
        gdn_prepare_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>
            <<<grid, block, prep_smem, stream>>>(
            q, q_row_stride, k, q_row_stride, v, q_row_stride,
            g, g_row_stride, beta, g_row_stride,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            scale, T_seq, H, chunks_per_seq, 1);
    }
    // Kernel 2: recurrence
    {
        dim3 grid(B, H, 1);
        dim3 block(GDN_NUM_THREADS);
        gdn_recurrence_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>
            <<<grid, block, rec_smem, stream>>>(
            v, q_row_stride, beta,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            out, nullptr, final_state,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            T_total, H, T_seq, chunks_per_seq,
            0, 1, 0);   // serial mode: all chunks, no group replay
    }
}

// Stage-3 launcher: parallel group replay (superchunk two-level scan).
// grid = (B, H, num_groups). Each CTA replays its GROUP_CHUNKS chunk range
// starting from the group start state in prefix_B (produced by Stage-2 scan).
// final_state written by the last group only (as in the torch scan reference replay).
extern "C" void gdn_chunk_replay(
    const cutlass::bfloat16_t* v, int v_row_stride,
    const cutlass::bfloat16_t* beta,
    const cutlass::bfloat16_t* ws_kd,
    const cutlass::bfloat16_t* ws_qd,
    const cutlass::bfloat16_t* ws_kr,
    const float* ws_gt,
    const cutlass::bfloat16_t* ws_inv,
    const cutlass::bfloat16_t* ws_mqk,
    cutlass::bfloat16_t* out,
    const cutlass::bfloat16_t* prefix_B,   // [B*H*G, D, D] bf16 row-major
    cutlass::bfloat16_t* final_state,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    int T_seq, int H, int B, int chunks_per_seq,
    int group_chunks, int prefix_exclusive, cudaStream_t stream) {
    const int T_total = B * T_seq;
    const int num_groups = (chunks_per_seq + group_chunks - 1) / group_chunks;
    const size_t rec_smem = recurrence_smem_bytes_impl();
    cudaFuncSetAttribute(gdn_recurrence_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)rec_smem);
    dim3 grid(B, H, num_groups);
    dim3 block(GDN_NUM_THREADS);
    gdn_recurrence_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>
        <<<grid, block, rec_smem, stream>>>(
        v, v_row_stride, beta,
        ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
        out, prefix_B, final_state,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        T_total, H, T_seq, chunks_per_seq,
        group_chunks, num_groups, prefix_exclusive);
}

// Fused (workspace-free) group replay launcher: reset fast path only. Each
// CTA recomputes its chunks' kd/qd/kr/INV/Mqk in-CTA from raw q/k/g/beta;
// start state is the previous group's transfer matrix B_g[g-1] (shift path).
extern "C" void gdn_chunk_replay_fused(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    int qk_row_stride,
    const cutlass::bfloat16_t* v, int v_row_stride,
    const cutlass::bfloat16_t* g, int g_row_stride,
    const cutlass::bfloat16_t* beta,
    cutlass::bfloat16_t* out,
    const cutlass::bfloat16_t* prefix_B,   // [B*H*G, D, D] bf16 row-major
    cutlass::bfloat16_t* final_state,
    float scale,
    int T_seq, int H, int B, int chunks_per_seq,
    int group_chunks, int head_ratio, cudaStream_t stream) {
    const int T_total = B * T_seq;
    const int num_groups = (chunks_per_seq + group_chunks - 1) / group_chunks;
    const size_t smem = sizeof(GDNRecurrenceFusedStorage<GDNLayouts<GDN_D, GDN_CHUNK>>);
    cudaFuncSetAttribute(gdn_recurrence_fused_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(B, H, num_groups);
    dim3 block(GDN_NUM_THREADS);
    gdn_recurrence_fused_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>
        <<<grid, block, smem, stream>>>(
        q, k, qk_row_stride, v, v_row_stride, g, g_row_stride, beta,
        out, prefix_B, final_state, scale,
        T_total, H, T_seq, chunks_per_seq,
        group_chunks, num_groups, head_ratio);
}

// CHUNK=32 specialization launcher (long sequences). Prepare uses the same
// template (block-Schur path), recurrence uses gdn_recurrence_kernel32.
extern "C" void gdn_chunk_forward32(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* g,
    const cutlass::bfloat16_t* beta,
    cutlass::bfloat16_t* ws_kd, cutlass::bfloat16_t* ws_qd,
    cutlass::bfloat16_t* ws_kr, float* ws_gt,
    cutlass::bfloat16_t* ws_inv, cutlass::bfloat16_t* ws_mqk,
    cutlass::bfloat16_t* out, cutlass::bfloat16_t* final_state,
    int q_row_stride, int g_row_stride,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    float scale, int T_seq, int H, int B, int chunks_per_seq,
    cudaStream_t stream) {
    const int T_total = B * T_seq;
    constexpr int CHUNK = 32;
    constexpr int D = 128;
    constexpr int NUM_THREADS = 256;

    const size_t rec_smem = sizeof(GDNRecurrenceStorage<GDNLayouts<D, CHUNK>>);
    const size_t prep_smem = sizeof(GDNPrepareStorage<GDNLayouts<D, CHUNK>>);
    cudaFuncSetAttribute(gdn_recurrence_kernel32<D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)rec_smem);
    cudaFuncSetAttribute(gdn_prepare_kernel<CHUNK, D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);

    // Kernel 1: prepare (CHUNK=32 block-Schur path)
    {
        dim3 grid(chunks_per_seq, B * H);
        dim3 block(NUM_THREADS);
        gdn_prepare_kernel<CHUNK, D, NUM_THREADS>
            <<<grid, block, prep_smem, stream>>>(
            q, q_row_stride, k, q_row_stride, v, q_row_stride,
            g, g_row_stride, beta, g_row_stride,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            scale, T_seq, H, chunks_per_seq, 1);
    }
    // Kernel 2: recurrence (CHUNK=32)
    {
        dim3 grid(B, H);
        dim3 block(NUM_THREADS);
        gdn_recurrence_kernel32<D, NUM_THREADS>
            <<<grid, block, rec_smem, stream>>>(
            v, q_row_stride, beta,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            out, nullptr, final_state,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            T_total, H, T_seq, chunks_per_seq);
    }
}

extern "C" void gdn_chunk_prepare_only(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* g,
    const cutlass::bfloat16_t* beta, int beta_row_stride,
    cutlass::bfloat16_t* ws_kd, cutlass::bfloat16_t* ws_qd,
    cutlass::bfloat16_t* ws_kr, float* ws_gt,
    cutlass::bfloat16_t* ws_inv, cutlass::bfloat16_t* ws_mqk,
    int qk_row_stride, int v_row_stride, int g_row_stride,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    float scale, int T_seq, int H, int B, int chunks_per_seq,
    int head_ratio, cudaStream_t stream) {
    const size_t prep_smem = prepare_smem_bytes_impl();
    cudaFuncSetAttribute(gdn_prepare_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);
    dim3 grid(chunks_per_seq, B * H);
    dim3 block(GDN_NUM_THREADS);
    gdn_prepare_kernel<GDN_CHUNK, GDN_D, GDN_NUM_THREADS>
        <<<grid, block, prep_smem, stream>>>(
        q, qk_row_stride, k, qk_row_stride, v, v_row_stride,
        g, g_row_stride, beta, g_row_stride,
        ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, T_seq, H, chunks_per_seq, head_ratio);
}

// Column-split launcher (P1): grid = (B, H, SPLIT). CHUNK=16, D=128,
// NumThreads=256, SPLIT=4. Each CTA owns D/SPLIT state columns. Uses the same
// prepare kernel (workspace identical) + the new colsplit recurrence kernel.
extern "C" void gdn_chunk_forward_colsplit(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* g,
    const cutlass::bfloat16_t* beta,
    cutlass::bfloat16_t* ws_kd, cutlass::bfloat16_t* ws_qd,
    cutlass::bfloat16_t* ws_kr, float* ws_gt,
    cutlass::bfloat16_t* ws_inv, cutlass::bfloat16_t* ws_mqk,
    cutlass::bfloat16_t* out, cutlass::bfloat16_t* final_state,
    int q_row_stride, int g_row_stride,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    float scale, int T_seq, int H, int B, int chunks_per_seq,
    int split, cudaStream_t stream) {
    const int T_total = B * T_seq;
    constexpr int CHUNK = 16;
    constexpr int D = 128;
    constexpr int NUM_THREADS = 256;

    const size_t prep_smem = sizeof(GDNPrepareStorage<GDNLayouts<D, CHUNK>>);
    cudaFuncSetAttribute(gdn_prepare_kernel<CHUNK, D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);

    // Kernel 1: prepare (identical workspace as the no-split CHUNK=16 path)
    {
        dim3 grid(chunks_per_seq, B * H);
        dim3 block(NUM_THREADS);
        gdn_prepare_kernel<CHUNK, D, NUM_THREADS>
            <<<grid, block, prep_smem, stream>>>(
            q, q_row_stride, k, q_row_stride, v, q_row_stride,
            g, g_row_stride, beta, g_row_stride,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            scale, T_seq, H, chunks_per_seq, 1);
    }
    // Kernel 2: recurrence (column-split) -- dispatch on split
    auto launch = [&](auto split_c) {
        constexpr int SPLIT = decltype(split_c)::value;
        constexpr int COLS_PER_SPLIT = D / SPLIT;
        const size_t rec_smem =
            sizeof(GDNRecurrenceColsplitStorage<GDNLayouts<D, CHUNK>, COLS_PER_SPLIT>);
        cudaFuncSetAttribute(gdn_recurrence_colsplit_kernel<CHUNK, D, NUM_THREADS, SPLIT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)rec_smem);
        dim3 grid(B, H, SPLIT);
        dim3 block(NUM_THREADS);
        gdn_recurrence_colsplit_kernel<CHUNK, D, NUM_THREADS, SPLIT>
            <<<grid, block, rec_smem, stream>>>(
            v, q_row_stride, beta,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            out, nullptr, final_state,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            T_total, H, T_seq, chunks_per_seq);
    };
    if (split == 8) {
        launch(std::integral_constant<int, 8>{});
    } else {
        launch(std::integral_constant<int, 4>{});
    }
}

// CHUNK=32 prepare-only (validates the block-Schur workspace in isolation).
extern "C" void gdn_chunk_prepare_only32(
    const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* g,
    const cutlass::bfloat16_t* beta, int beta_row_stride,
    cutlass::bfloat16_t* ws_kd, cutlass::bfloat16_t* ws_qd,
    cutlass::bfloat16_t* ws_kr, float* ws_gt,
    cutlass::bfloat16_t* ws_inv, cutlass::bfloat16_t* ws_mqk,
    int qk_row_stride, int v_row_stride, int g_row_stride,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    float scale, int T_seq, int H, int B, int chunks_per_seq,
    int head_ratio, cudaStream_t stream) {
    constexpr int CHUNK = 32;
    constexpr int D = 128;
    constexpr int NUM_THREADS = 256;
    const size_t prep_smem = sizeof(GDNPrepareStorage<GDNLayouts<D, CHUNK>>);
    cudaFuncSetAttribute(gdn_prepare_kernel<CHUNK, D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);
    dim3 grid(chunks_per_seq, B * H);
    dim3 block(NUM_THREADS);
    gdn_prepare_kernel<CHUNK, D, NUM_THREADS>
        <<<grid, block, prep_smem, stream>>>(
        q, qk_row_stride, k, qk_row_stride, v, v_row_stride,
        g, g_row_stride, beta, g_row_stride,
        ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, T_seq, H, chunks_per_seq, head_ratio);
}
