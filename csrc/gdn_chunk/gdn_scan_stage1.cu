// gdn_scan_stage1.cu — Stage 1 of the gdn_chunk superchunk affine scan.
//
// One CTA per (seq, head, group). Each CTA reads the per-chunk workspace tiles
// for its GROUP_CHUNKS chunks and produces the group transfer (A_g, B_g), both
// [128,128] bf16, via the LOW-RANK combine (O(D^2*C)):
//
//   per chunk t (D=128, C=16), from workspace + raw v/beta:
//     bkd = kd * beta            # [C,D] row-scaled by per-token beta
//     bkv = v  * beta            # [C,D]
//     P   = INV @ bkd            # [C,D]
//     R   = INV @ bkv            # [C,D]
//     K   = kr^T                 # [D,C]
//     A_t = D_t - K @ P          # D_t = diag(gt), row scale over state rows
//     B_t = K @ R
//   group prefix (start A=I, B=0), LOW-RANK combine:
//     A_new = D_t*A_old - K@(P@A_old)
//     B_new = D_t*B_old - K@(P@B_old) + B_t
//   => A_g, B_g [128,128] each; stored to a global buffer.
//
// Precision: A_g/B_g stored bf16, fp32 MMA accumulate, round to bf16 at the
// end of each chunk combine (matches the torch scan reference 'bf16' config).
//
// Workspace layout (VERIFIED row-major by the torch scan reference + the prepare
// kernel): ws_kd/ws_kr are logical [CHUNK,D] bf16 row-major tiles, ws_inv
// [16,16], ws_gt [D] fp32; tile index = bh*chunks_per_seq + t, bh=seq*H+head.
// v and beta are the raw inputs (not in the workspace) and are read row-major
// / head-major exactly like the recurrence kernel.
//
// Tensor cores: SM80 bf16 mma.sync via cute (SM80_16x8x16_F32BF16BF16F32_TN),
// Tile 16x16, one warp per 16x16 output tile. 8 warps.
//
// This is Stage 1 ONLY: no scan kernel, no replay kernel.

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
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

// ---------------- Layouts (same swizzle scheme as gdn_kernel.cu) ----------------
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
};

// ---------------- Shared memory for the stage-1 kernel ----------------
template <int CHUNK, int D, class Layouts>
struct GDNStage1Storage {
    using MMALayout = typename Layouts::MMALayout;
    using LMLayout = typename Layouts::LMLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using GTotalLayout = typename Layouts::GTotalLayout;

    // per-chunk inputs (MMALayout swizzled [C,D])
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> bkd;  // kd*beta
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> bkv;  // v*beta
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> kr;   // k_restored [C,D]
    // computed [C,D] intermediates
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> P;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> R;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> tmpA;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> tmpB;
    // group state [D,D]
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> A;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> B;
    // INV [C,C] LMLayout
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
    // gt fp32 [D]
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> gt;
    // beta bf16 [C]
    alignas(16) cute::ArrayEngine<BF16, CHUNK> beta;
};

// ==================== Kernel: Stage 1 group transfer ====================
template <int CHUNK, int D, int NumThreads>
__global__ void __launch_bounds__(NumThreads) gdn_scan_stage1_kernel(
    const BF16* __restrict__ ws_kd,
    const BF16* __restrict__ ws_kr,
    const float* __restrict__ ws_gt,
    const BF16* __restrict__ ws_inv,
    const BF16* __restrict__ v_ptr, int v_row_stride,   // [T,H,D] row-major
    const BF16* __restrict__ beta_ptr, int beta_row_stride, // [H, T] head-major
    BF16* __restrict__ A_g,
    BF16* __restrict__ B_g,
    float* __restrict__ diag_out,   // optional [8] diagnostic buffer (may be null)
    float* __restrict__ diag_full,  // optional [4*C*D + 2*D*D] f32 diagnostic dump (may be null)
    int ws_tile_elems,   // CHUNK*D
    int ws_tile_lm,      // CHUNK*CHUNK
    int ws_gt_elems,     // D
    int T_seq,           // S
    int H,
    int chunks_per_seq,
    int GROUP_CHUNKS) {

    using Layouts = GDNLayouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using LMLayout = typename Layouts::LMLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using GTotalLayout = typename Layouts::GTotalLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    GDNStage1Storage<CHUNK, D, Layouts>& ss =
        *reinterpret_cast<GDNStage1Storage<CHUNK, D, Layouts>*>(shared_mem);

    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int group = blockIdx.z;
    const int tid = threadIdx.x;
    const int w = tid / 32;         // warp 0..7
    const int lane = tid % 32;
    const int group_id = (lane / 4) % 8;
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = NumThreads / kWarpSize;
    const int bh = seq * H + head;
    const int group_base = group * GROUP_CHUNKS;

    // Reference cute tensors over smem
    auto kd_t = make_tensor(make_smem_ptr(ss.bkd.begin()), MMALayout{});
    auto kv_t = make_tensor(make_smem_ptr(ss.bkv.begin()), MMALayout{});
    auto kr_t = make_tensor(make_smem_ptr(ss.kr.begin()), MMALayout{});
    auto krT_t = make_tensor(make_smem_ptr(ss.kr.begin()), TransposedMMALayout{}); // K = kr^T
    auto P_t = make_tensor(make_smem_ptr(ss.P.begin()), MMALayout{});
    auto R_t = make_tensor(make_smem_ptr(ss.R.begin()), MMALayout{});
    auto tA_t = make_tensor(make_smem_ptr(ss.tmpA.begin()), MMALayout{});
    auto tB_t = make_tensor(make_smem_ptr(ss.tmpB.begin()), MMALayout{});
    auto INV_t = make_tensor(make_smem_ptr(ss.INV.begin()), LMLayout{});
    auto gt_t = make_tensor(make_smem_ptr(ss.gt.begin()), GTotalLayout{});
    auto A_t = make_tensor(make_smem_ptr(ss.A.begin()), StateSmemLayout{});
    auto B_t = make_tensor(make_smem_ptr(ss.B.begin()), StateSmemLayout{});
    auto A_T = make_tensor(make_smem_ptr(ss.A.begin()), TransposedStateSmemLayout{});
    auto B_T = make_tensor(make_smem_ptr(ss.B.begin()), TransposedStateSmemLayout{});

    // ---- Initialize group state: A = I, B = 0 (logical [D,D]) ----
    // CRITICAL: write through the swizzled tensor views. Writing the linear
    // buffer with (r==c) places the identity at LINEAR indices, but A_t reads
    // through StateSmemLayout (K_INTER swizzle), so the diagonal must be
    // placed at StateSmemLayout(r,r), not at r*D+r.
    {
        constexpr int N = D * D;
        for (int i = tid; i < N; i += NumThreads) {
            int r = i / D;
            int c = i - r * D;
            A_t(r, c) = (r == c) ? BF16(1.0f) : BF16(0.0f);
            B_t(r, c) = BF16(0.0f);
        }
        __syncthreads();
    }

    // mma objects
    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    ThrMMA thr_mma = mma.get_slice(lane);

    auto smem_tiled_copy_A   = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(lane);
    auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane);
    auto smem_tiled_copy_B   = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto smem_thr_copy_B     = smem_tiled_copy_B.get_thread_slice(lane);
    auto smem_tiled_store_C   = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
    auto smem_thr_store_C     = smem_tiled_store_C.get_slice(lane);
    auto smem_tiled_load_C    = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto smem_thr_load_C      = smem_tiled_load_C.get_slice(lane);
    auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane);
    auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
    auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane);

    // Reference 16x16 tiles for fragment partitioning
    Tensor A16 = local_tile(kd_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
    Tensor B16 = local_tile(A_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

    using AccFragT = decltype(thr_mma.make_fragment_C(thr_mma.partition_C(A16)));
    using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(thr_mma.partition_C(A16))));
    using AFragT = decltype(thr_mma.partition_fragment_A(A16));
    using BFragT = decltype(thr_mma.partition_fragment_B(B16));

    // PROVEN recurrence pattern (gdn_kernel.cu Phase 3): tiles written by
    // store_C are read back as MMA B-operands via load_C + an in-register
    // C->B fragment conversion (identity reg copy). Using copy_B directly
    // TRANSPOSES the tile (verified by identity-oracle probe), because the
    // MMA's B-operand smem layout is the transpose of the store_C C-layout.
    // Tiles stored UNtransposed (P/R/tmpA/tmpB) must go through load_C+c_to_b;
    // the STATE (stored transposed via store_C_T into A_T) is correctly read
    // back with copy_B (double transpose cancels).
    auto c_to_b = [&](SFragT const& c_frag) -> BFragT {
        BFragT b = thr_mma.partition_fragment_B(B16);
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

    // ---- Per-chunk combine loop ----
    for (int lc = 0; lc < GROUP_CHUNKS; ++lc) {
        const int t = group_base + lc;
        if (t >= chunks_per_seq) break;   // partial trailing group (not hit in the validated S)
        const int ws_idx = bh * chunks_per_seq + t;

        // ---- Load per-chunk inputs: kd, kr, v -> smem (row-major gmem -> MMALayout smem) ----
        {
            const BF16* kd_base = ws_kd + int64_t(ws_idx) * ws_tile_elems;
            const BF16* kr_base = ws_kr + int64_t(ws_idx) * ws_tile_elems;
            const BF16* inv_base = ws_inv + int64_t(ws_idx) * ws_tile_lm;
            const float* gt_base = ws_gt + int64_t(ws_idx) * ws_gt_elems;
            const BF16* v_base = v_ptr + int64_t(seq * T_seq + t * CHUNK) * v_row_stride + head * D;
            const BF16* beta_base = beta_ptr + int64_t(head) * beta_row_stride + seq * T_seq + t * CHUNK;

            // kd, kr: [C,D] row-major -> MMALayout
            for (int i = tid; i < CHUNK * (D / 8); i += NumThreads) {
                int r = i / (D / 8);
                int c = (i - r * (D / 8)) * 8;
                *reinterpret_cast<uint4*>(&kd_t(r, c)) =
                    *reinterpret_cast<uint4 const*>(kd_base + r * D + c);
                *reinterpret_cast<uint4*>(&kr_t(r, c)) =
                    *reinterpret_cast<uint4 const*>(kr_base + r * D + c);
                *reinterpret_cast<uint4*>(&kv_t(r, c)) =
                    *reinterpret_cast<uint4 const*>(v_base + r * v_row_stride + c);
            }
            // INV [C,C] -> LMLayout
            for (int i = tid; i < CHUNK * (CHUNK / 8); i += NumThreads) {
                int r = i / (CHUNK / 8);
                int c = (i - r * (CHUNK / 8)) * 8;
                *reinterpret_cast<uint4*>(&INV_t(r, c)) =
                    *reinterpret_cast<uint4 const*>(inv_base + r * CHUNK + c);
            }
            // gt [D] fp32
            for (int i = tid; i < D / 4; i += NumThreads) {
                *reinterpret_cast<float4*>(&gt_t(i * 4)) =
                    *reinterpret_cast<float4 const*>(gt_base + i * 4);
            }
            // beta [C] bf16
            for (int i = tid; i < CHUNK; i += NumThreads) {
                ss.beta.begin()[i] = beta_base[i];
            }
        }
        __syncthreads();

        // ---- Compute bkd = kd*beta, bkv = v*beta (elementwise over [C,D]) ----
        {
            for (int i = tid; i < CHUNK * D; i += NumThreads) {
                int r = i / D;
                int c = i - r * D;
                float be = bf16_to_f32(ss.beta.begin()[r]);
                kd_t(r, c) = BF16(bf16_to_f32(kd_t(r, c)) * be);
                kv_t(r, c) = BF16(bf16_to_f32(kv_t(r, c)) * be);
            }
        }
        __syncthreads();

        // ---- Phase B: P = INV @ bkd, R = INV @ bkv  ([16,16]@[16,16] per warp col-block) ----
        {
            AFragT Af, AfT_;
            auto Af_v = smem_thr_copy_A.retile_D(Af);

            // A-op = INV (16x16 full)
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV_t), Af_v);
            cute::transform(Af, AfT_, cute::identity{});

            // B-op = bkd col-block w (via load_C + c_to_b, NOT copy_B)
            {
                SFragT Cfb;
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                    local_tile(kd_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))),
                    smem_thr_load_C.retile_D(Cfb));
                BFragT Bfb = c_to_b(Cfb);
                AccFragT accP; clear(accP);
                gemm(thr_mma, AfT_(_,_,Int<0>{}), Bfb(_,_,Int<0>{}), accP);
                SFragT sP; cute::transform(accP, sP, [] __device__ (float x) { return BF16(x); });
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sP),
                     smem_thr_store_C.partition_D(local_tile(P_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))));
            }

            // B-op = bkv col-block w (via load_C + c_to_b, NOT copy_B)
            {
                SFragT Cfb;
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                    local_tile(kv_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))),
                    smem_thr_load_C.retile_D(Cfb));
                BFragT Bfb = c_to_b(Cfb);
                AccFragT accR; clear(accR);
                gemm(thr_mma, AfT_(_,_,Int<0>{}), Bfb(_,_,Int<0>{}), accR);
                SFragT sR; cute::transform(accR, sR, [] __device__ (float x) { return BF16(x); });
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sR),
                     smem_thr_store_C.partition_D(local_tile(R_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))));
            }
        }
        __syncthreads();
        if (lc == 0 && seq == 0 && head == 0 && group == 0 && tid == 0 && diag_out != nullptr) {
            float mp = 0, mr = 0, mv = 0, mi = 0, mk = 0, mkd = 0;
            for (int e = 0; e < CHUNK * D; ++e) {
                mp = fmaxf(mp, fabsf(bf16_to_f32(P_t(e / D, e % D))));
                mr = fmaxf(mr, fabsf(bf16_to_f32(R_t(e / D, e % D))));
                mv = fmaxf(mv, fabsf(bf16_to_f32(kv_t(e / D, e % D))));
                mk = fmaxf(mk, fabsf(bf16_to_f32(kd_t(e / D, e % D))));
            }
            for (int e = 0; e < CHUNK * CHUNK; ++e) {
                mi = fmaxf(mi, fabsf(bf16_to_f32(INV_t(e / CHUNK, e % CHUNK))));
            }
            diag_out[0] = mp; diag_out[1] = mr; diag_out[2] = mv;
            diag_out[5] = mi; diag_out[6] = mk; diag_out[7] = 0;
        }
        // Full dump: P [C,D], R [C,D] (logical row-major) — every chunk of group 0
        if (seq == 0 && head == 0 && group == 0 && diag_full != nullptr) {
            const int64_t slot = int64_t(lc) * (4 * CHUNK * D + 2 * D * D);
            for (int i = tid; i < CHUNK * D; i += NumThreads) {
                diag_full[slot + 0 * CHUNK * D + i] = bf16_to_f32(P_t(i / D, i % D));
                diag_full[slot + 1 * CHUNK * D + i] = bf16_to_f32(R_t(i / D, i % D));
            }
            __syncthreads();
        }

        // ---- Phase C: tmpA = P @ A_old, tmpB = P @ B_old  ([16,128]@[128,16] per warp col-block w) ----
        {
            AFragT Af, AfT_;
            BFragT Bf, BfT_;
            auto Af_v = smem_thr_copy_A.retile_D(Af);
            auto Bf_v = smem_thr_copy_B.retile_D(Bf);

            AccFragT accA; clear(accA);
            AccFragT accB; clear(accB);
            #pragma unroll
            for (int k = 0; k < D / 16; ++k) {
                copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                    local_tile(P_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k))), Af_v);
                cute::transform(Af, AfT_, cute::identity{});

                // NOTE: the state is stored TRANSPOSED (Phase D writes A_T via
                // store_C_T), and copy_B reads it back transposed; the double
                // transpose cancels only if the contraction index k is the
                // tile's SECOND (col-block) coord — matching the proven
                // recurrence Phase 1 (s_acc[warp_id*2, k]). Using (k, w) puts
                // k on the first coord -> per-16x16-block transpose of A_old,
                // which identity init masked. Fixed to (w, k).
                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(A_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, k))), Bf_v);
                cute::transform(Bf, BfT_, cute::identity{});
                gemm(thr_mma, AfT_(_,_,Int<0>{}), BfT_(_,_,Int<0>{}), accA);

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(B_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, k))), Bf_v);
                cute::transform(Bf, BfT_, cute::identity{});
                gemm(thr_mma, AfT_(_,_,Int<0>{}), BfT_(_,_,Int<0>{}), accB);
            }
            SFragT sA; cute::transform(accA, sA, [] __device__ (float x) { return BF16(x); });
            SFragT sB; cute::transform(accB, sB, [] __device__ (float x) { return BF16(x); });
            copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sA),
                 smem_thr_store_C.partition_D(local_tile(tA_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))));
            copy(smem_tiled_store_C, smem_thr_store_C.retile_S(sB),
                 smem_thr_store_C.partition_D(local_tile(tB_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, w))));
        }
        __syncthreads();

        // Full dump: tmpA [C,D], tmpB [C,D]
        if (seq == 0 && head == 0 && group == 0 && diag_full != nullptr) {
            const int64_t slot = int64_t(lc) * (4 * CHUNK * D + 2 * D * D);
            for (int i = tid; i < CHUNK * D; i += NumThreads) {
                diag_full[slot + 2 * CHUNK * D + i] = bf16_to_f32(tA_t(i / D, i % D));
                diag_full[slot + 3 * CHUNK * D + i] = bf16_to_f32(tB_t(i / D, i % D));
            }
            __syncthreads();
        }

        // ---- Phase D: A = D_t*A_old - K@tmpA ; B = D_t*B_old - K@tmpB + K@R ----
        // warp w handles STATE ROW-BLOCK w (rows 16w..16w+15), all 8 col-tiles n.
        {
            // A-op K[w] = kr^T row-block w (fixed across n)
            AFragT Af, AfT_;
            auto Af_v = smem_thr_copy_A_T.retile_D(Af);
            copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(
                local_tile(krT_t, make_shape(Int<16>{}, Int<16>{}), make_coord(w, 0))), Af_v);
            cute::transform(Af, AfT_, cute::identity{});

            // PROVEN recurrence pattern (gdn_kernel.cu Phase 3): tiles written by
            // store_C are read back as MMA B-operands via load_C + an in-register
            // C->B fragment conversion (identity reg copy). Using copy_B directly
            // TRANSPOSES the tile (verified by identity-oracle probe), because
            // copy_B's LDSM assumes the K_INTER-swizzle is in the B-operand's
            // transposed orientation. P/R/tmpA/tmpB are stored untransposed, so
            // copy_B would silently transpose them -> wrong K@(P@A), K@R, ...
            auto c_to_b = [&](SFragT const& c_frag) -> BFragT {
                BFragT b = thr_mma.partition_fragment_B(B16);
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

            float g0 = gt_t(w * 16 + group_id);
            float g1 = gt_t(w * 16 + group_id + 8);

            #pragma unroll
            for (int n = 0; n < D / 16; ++n) {
                AccFragT accA; clear(accA);   // K @ tmpA
                AccFragT accBt; clear(accBt); // K @ R   (B_t)
                AccFragT accB2; clear(accB2); // K @ tmpB

                {
                    SFragT CfA;
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                        local_tile(tA_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, n))),
                        smem_thr_load_C.retile_D(CfA));
                    BFragT bA = c_to_b(CfA);
                    gemm(thr_mma, AfT_(_,_,Int<0>{}), bA(_,_,Int<0>{}), accA);
                }

                {
                    SFragT CfR;
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                        local_tile(R_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, n))),
                        smem_thr_load_C.retile_D(CfR));
                    BFragT bR = c_to_b(CfR);
                    gemm(thr_mma, AfT_(_,_,Int<0>{}), bR(_,_,Int<0>{}), accBt);
                }

                {
                    SFragT CfB;
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(
                        local_tile(tB_t, make_shape(Int<16>{}, Int<16>{}), make_coord(0, n))),
                        smem_thr_load_C.retile_D(CfB));
                    BFragT bB = c_to_b(CfB);
                    gemm(thr_mma, AfT_(_,_,Int<0>{}), bB(_,_,Int<0>{}), accB2);
                }

                // read A_old / B_old row-block w, col-tile n
                SFragT Aold, Bold;
                copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(
                    local_tile(A_T, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n))),
                    smem_thr_load_C_T.retile_D(Aold));
                copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(
                    local_tile(B_T, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n))),
                    smem_thr_load_C_T.retile_D(Bold));

                SFragT Anew, Bnew;
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        Anew(c0) = BF16(bf16_to_f32(Aold(c0)) * g0 - accA(c0));
                        Anew(c1) = BF16(bf16_to_f32(Aold(c1)) * g1 - accA(c1));
                        Bnew(c0) = BF16(bf16_to_f32(Bold(c0)) * g0 - accB2(c0) + accBt(c0));
                        Bnew(c1) = BF16(bf16_to_f32(Bold(c1)) * g1 - accB2(c1) + accBt(c1));
                    }
                }
                copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(Anew),
                     smem_thr_store_C_T.partition_D(local_tile(A_T, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n))));
                copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(Bnew),
                     smem_thr_store_C_T.partition_D(local_tile(B_T, make_shape(Int<16>{}, Int<16>{}), make_coord(w, n))));
            }
        }
        // global check after chunk 0 Phase D: max|A| and max|B| over smem state
        __syncthreads();
        if (lc == 0 && seq == 0 && head == 0 && group == 0 && tid == 0 && diag_out != nullptr) {
            float mA = 0, mB = 0;
            for (int e = 0; e < D * D; ++e) {
                mA = fmaxf(mA, fabsf(bf16_to_f32(A_T(e / D, e % D))));
                mB = fmaxf(mB, fabsf(bf16_to_f32(B_T(e / D, e % D))));
            }
            diag_out[3] = mA; diag_out[4] = mB;
        }
        // Full dump: A [D,D], B [D,D] (logical row-major via A_T/B_T)
        if (seq == 0 && head == 0 && group == 0 && diag_full != nullptr) {
            const int64_t slot = int64_t(lc) * (4 * CHUNK * D + 2 * D * D);
            for (int i = tid; i < D * D; i += NumThreads) {
                diag_full[slot + 4 * CHUNK * D + i] = bf16_to_f32(A_T(i / D, i % D));
                diag_full[slot + 4 * CHUNK * D + D * D + i] = bf16_to_f32(B_T(i / D, i % D));
            }
        }
    }

    // ---- Store A_g / B_g: [B*H*num_groups, D, D] logical row-major ----
    {
        int64_t base = (int64_t(bh) * ((chunks_per_seq + GROUP_CHUNKS - 1) / GROUP_CHUNKS) + group) * D * D;
        for (int i = tid; i < D * D; i += NumThreads) {
            int r = i / D;
            int c = i - r * D;
            A_g[base + r * D + c] = A_T(r, c);
            B_g[base + r * D + c] = B_T(r, c);
        }
    }
}

// ==================== C launcher (extern "C") ====================
#ifndef GDN_SCAN_CHUNK
#define GDN_SCAN_CHUNK 16
#define GDN_SCAN_D 128
#define GDN_SCAN_NUM_THREADS 256
#endif

extern "C" void gdn_scan_stage1(
    const cutlass::bfloat16_t* ws_kd,
    const cutlass::bfloat16_t* ws_kr,
    const float* ws_gt,
    const cutlass::bfloat16_t* ws_inv,
    const cutlass::bfloat16_t* v_ptr, int v_row_stride,
    const cutlass::bfloat16_t* beta_ptr, int beta_row_stride,
    cutlass::bfloat16_t* A_g,
    cutlass::bfloat16_t* B_g,
    float* diag_out,
    float* diag_full,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    int T_seq, int H, int B, int chunks_per_seq,
    int GROUP_CHUNKS, cudaStream_t stream) {
    constexpr int CHUNK = GDN_SCAN_CHUNK;
    constexpr int D = GDN_SCAN_D;
    constexpr int NUM_THREADS = GDN_SCAN_NUM_THREADS;

    using Layouts = GDNLayouts<D, CHUNK>;
    const size_t smem = sizeof(GDNStage1Storage<CHUNK, D, Layouts>);
    cudaFuncSetAttribute(gdn_scan_stage1_kernel<CHUNK, D, NUM_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    int num_groups = (chunks_per_seq + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
    dim3 grid(B, H, num_groups);
    dim3 block(NUM_THREADS);
    gdn_scan_stage1_kernel<CHUNK, D, NUM_THREADS>
        <<<grid, block, smem, stream>>>(
        ws_kd, ws_kr, ws_gt, ws_inv,
        v_ptr, v_row_stride, beta_ptr, beta_row_stride,
        A_g, B_g, diag_out, diag_full,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        T_seq, H, chunks_per_seq, GROUP_CHUNKS);
}
