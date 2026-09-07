// QSA lightweight indexer (SM80)
//
// Implements the sparse-block attention block-level indexer:
//   - MQA: 4 query heads, 1 shared key head, head_dim = 128
//   - block compression: raw key sequence AvgPool'd in groups of r=4 -> pooled block key
//   - RMSNorm on q and pooled keys (head_dim)
//   - partial RoPE: only the first rope_dim (=64) of the 128 dims is rotated.
//       * q uses its own token position
//       * block key uses the block START position p_b = b*r
//   - block-causal scoring:  I[i,b] = (1/sqrt(head_dim)) * sum_h ReLU(<q[i,h], kbar[b]>)
//       only when block b is fully visible (p_b + r - 1 <= i), else -inf
//   - TopK: per query select the block_topk highest-scoring valid blocks
//
// Performance design (vs. original correctness-first version):
//   The original fused everything into one thread-per-(b,q,blk) kernel that
//   recomputed the pooled block key S times and the q encoding NB times per
//   query, plus an O(NB^2) per-query insertion/selection TopK that only worked
//   for n_blocks <= 512.  This version:
//     A. precomputes pooled+norm+roped block keys  kbar[B,NB,D]  once
//     B. precomputes norm+roped queries            qenc[B,S,Hq,D] once
//     C. computes the score matrix with float4 vectorized dot products
//     D. TopK via a shared-memory bitonic merge sort (correct for ANY n_blocks)
//
// Inputs (all contiguous):
//   q        : [B, S, index_n_heads, head_dim]  float (projected q)
//   raw_keys : [B, S, head_dim]                 float (1 shared key head, pre-compression)
//   cos      : [B, S, rope_dim]                 float
//   sin      : [B, S, rope_dim]                 float
// Outputs:
//   block_scores : [B, S, n_blocks]             float (I matrix; -inf where invalid)
//   block_indices: [B, S, block_topk]           int32 (selected block indices, -1 padded)
//   block_scores_out: [B, S, block_topk]        float (scores of selected blocks, -inf padded)
//
// Assumptions: S % r == 0; n_blocks = S / r; rope_dim divides head_dim and rope_dim <= head_dim.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>

#include "qsa_indexer.h"

#include <algorithm>
#include <vector>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT(x)  TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")

namespace {

__device__ __forceinline__ float rsqrtf_(float x) {
    return rsqrtf(x);
}

// ---------------------------------------------------------------------------
// Kernel A: compress raw keys into pooled, RMSNorm'd, partially-roped block keys.
//   kbar[B, NB, D]  where NB = S / r.
// One WARP per (batch, block): lane dg owns the float4 at dims [4dg, 4dg+4),
// which coalesces the r-token loads across the warp (the old thread-per-block
// version read rows strided r*D floats apart -> ~2% coalescing, and with only
// B*NB threads the GPU was nearly idle at S=8192).  Pool = mean over r tokens,
// RMSNorm via a 32-lane warp reduction, partial RoPE (first R dims) at block
// start p_b = blk * r with a __shfl_xor pair exchange.  D=128, R=64 enforced
// by the host (production indexer shapes).
// ---------------------------------------------------------------------------
__global__ void indexer_pool_keys_kernel(
    const float* __restrict__ raw_keys,   // [B,S,D]
    const float* __restrict__ cos_k,      // [B,S,R]
    const float* __restrict__ sin_k,      // [B,S,R]
    float* __restrict__ kbar,             // [B,NB,D]
    int B, int S, int D, int R, int r, int NB)
{
    const int D4 = D / 4;                 // 32 for D=128
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NB * D4;
    if (idx >= total) return;

    int dg = idx % D4;                    // dim group (lane)
    int bb = idx / D4;                    // b*NB + blk
    int blk = bb % NB;
    int b   = bb / NB;
    int p_b = blk * r;

    const float4* k4 = reinterpret_cast<const float4*>(raw_keys + (size_t)b * S * D);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int u = 0; u < r; ++u) {
        float4 v = k4[(size_t)(p_b + u) * D4 + dg];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float inv = 1.0f / (float)r;
    float4 pooled = make_float4(acc.x * inv, acc.y * inv,
                                acc.z * inv, acc.w * inv);

    // RMSNorm over D (warp reduce over the 32 dim groups)
    float ssum = pooled.x * pooled.x + pooled.y * pooled.y
               + pooled.z * pooled.z + pooled.w * pooled.w;
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 16);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 8);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 4);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 2);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 1);
    float rms_inv = rsqrtf_(ssum / (float)D + 1e-6f);
    float4 out = make_float4(pooled.x * rms_inv, pooled.y * rms_inv,
                             pooled.z * rms_inv, pooled.w * rms_inv);

    // partial RoPE at block start p_b (R=64: pairs (d, d+32), partner 8 lanes off)
    if (dg < 16) {
        float4 partner;
        partner.x = __shfl_xor_sync(0xFFFFFFFFu, out.x, 8);
        partner.y = __shfl_xor_sync(0xFFFFFFFFu, out.y, 8);
        partner.z = __shfl_xor_sync(0xFFFFFFFFu, out.z, 8);
        partner.w = __shfl_xor_sync(0xFFFFFFFFu, out.w, 8);
        int m = dg & 7;                   // cos/sin pair index
        const float4* c4 = reinterpret_cast<const float4*>(cos_k)
                         + ((size_t)b * S + p_b) * (R / 4) + m;
        const float4* s4 = reinterpret_cast<const float4*>(sin_k)
                         + ((size_t)b * S + p_b) * (R / 4) + m;
        float4 co = c4[0], si = s4[0];
        if (dg < 8) {
            // first half: new[d]       = x[d]*c - x[d+32]*s
            out.x = out.x * co.x - partner.x * si.x;
            out.y = out.y * co.y - partner.y * si.y;
            out.z = out.z * co.z - partner.z * si.z;
            out.w = out.w * co.w - partner.w * si.w;
        } else {
            // second half: new[d+32]   = x[d]*s + x[d+32]*c
            out.x = partner.x * si.x + out.x * co.x;
            out.y = partner.y * si.y + out.y * co.y;
            out.z = partner.z * si.z + out.z * co.z;
            out.w = partner.w * si.w + out.w * co.w;
        }
    }

    float4* o4 = reinterpret_cast<float4*>(kbar + (size_t)(b * NB + blk) * D);
    o4[dg] = out;
}

// ---------------------------------------------------------------------------
// Kernel B: encode each query: RMSNorm over D + partial RoPE at token position.
//   qenc[B, S, Hq, D]
//
// One WARP per (batch, query, head); lane dg owns the float4 at dims
// [4*dg, 4*dg+4).  The old thread-per-row version read each thread's 128-float
// row with 32 threads strided D floats apart, so every global load touched 32
// separate 128B cache lines (3% coalescing) and the kernel ran ~17x off its
// memory roofline at S=8192.  Reading one dims-contiguous float4 per lane makes
// the warp's 32 loads land in consecutive lines.  The RMSNorm sum is a 32-lane
// warp reduction (no barriers); the partial RoPE pairs dims (d, d+R/2) whose
// float4s sit 8 lanes apart, exchanged with one __shfl_xor.
// Requires D == 128 and D % 4 == 0, R % 8 == 0 (the production indexer shapes).
// ---------------------------------------------------------------------------
__global__ void indexer_encode_q_kernel(
    const float* __restrict__ q,          // [B,S,Hq,D]
    const float* __restrict__ cos_q,      // [B,S,R]
    const float* __restrict__ sin_q,      // [B,S,R]
    float* __restrict__ qenc,             // [B,S,Hq,D]
    int B, int S, int Hq, int D, int R)
{
    const int D4 = D / 4;                 // 32 float4 per row for D=128
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * Hq * D4;
    if (idx >= total) return;

    int dg = idx % D4;                    // dim group (lane)
    int ih = idx / D4;                    // (b*S + t) * Hq + h
    int h = ih % Hq;
    int t = (ih / Hq) % S;
    int b = ih / (Hq * S);

    const float4* q4 = reinterpret_cast<const float4*>(q)
                     + ((size_t)b * S + t) * Hq * D4 + (size_t)h * D4;
    float4 v = q4[dg];

    // RMSNorm: warp-reduce the sum of squares over all D dims.
    // D == 128 (host-enforced), so D4 == 32 and the shuffle distances are fixed.
    float ssum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 16);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 8);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 4);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 2);
    ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, 1);
    float rms_inv = rsqrtf_(ssum / (float)D + 1e-6f);
    float4 out = make_float4(v.x * rms_inv, v.y * rms_inv,
                             v.z * rms_inv, v.w * rms_inv);

    // partial RoPE over the first R = 64 dims: pairs (d, d + 32) for d in [0,32).
    // Dims 4*dg..4*dg+3 are in the rope region iff dg < 16; the partner float4
    // (dims +32) is 8 lanes away.  Lanes < 8 write the first half, the rest the
    // second half, using the cos/sin pair index (dg & 7).
    if (4 * dg < R) {
        float4 partner;                   // float4 has no shuffle overload
        partner.x = __shfl_xor_sync(0xFFFFFFFFu, out.x, 8);
        partner.y = __shfl_xor_sync(0xFFFFFFFFu, out.y, 8);
        partner.z = __shfl_xor_sync(0xFFFFFFFFu, out.z, 8);
        partner.w = __shfl_xor_sync(0xFFFFFFFFu, out.w, 8);
        int m = dg & 7;                   // cos/sin index for this dim group
        const float4* c4 = reinterpret_cast<const float4*>(cos_q)
                         + ((size_t)b * S + t) * (R / 4) + m;
        const float4* s4 = reinterpret_cast<const float4*>(sin_q)
                         + ((size_t)b * S + t) * (R / 4) + m;
        float4 co = c4[0], si = s4[0];
        if (dg < 8) {
            // first half: new[d]       = x[d]*c - x[d+R/2]*s
            out.x = out.x * co.x - partner.x * si.x;
            out.y = out.y * co.y - partner.y * si.y;
            out.z = out.z * co.z - partner.z * si.z;
            out.w = out.w * co.w - partner.w * si.w;
        } else {
            // second half: new[d+R/2]  = x[d]*s + x[d+R/2]*c
            out.x = partner.x * si.x + out.x * co.x;
            out.y = partner.y * si.y + out.y * co.y;
            out.z = partner.z * si.z + out.z * co.z;
            out.w = partner.w * si.w + out.w * co.w;
        }
    }

    float4* o4 = reinterpret_cast<float4*>(qenc)
               + ((size_t)b * S + t) * Hq * D4 + (size_t)h * D4;
    o4[dg] = out;
}

// ---------------------------------------------------------------------------
// Kernel C: score matrix I[B, S, NB], tiled like a small GEMM.
//
// Grid: (S/SCORE_CQ) x (NB/SCORE_CB) x B blocks.  Each block computes a
// [SCORE_CQ x SCORE_CB] tile of the score matrix: it stages the encoded
// queries for SCORE_CQ rows and the pooled block keys for SCORE_CB columns
// into shared memory, then one thread per (row, col) computes the dot product
// sum over heads with ReLU.  Both operands are reused SCORE_CB/SCORE_CQ times
// from shared memory, so the kernel is compute / shared-bandwidth bound rather
// than L2-bandwidth bound, and the many blocks give high memory-level
// parallelism.
//
// Tile sizes: SCORE_CQ queries x SCORE_CB blocks, one thread per output cell.
// The block-key tile is stored as a PADDED float array with row stride D+1 so
// that a warp (whose 32 lanes are the 32 block columns of one query row, at a
// fixed dim d) hits all 32 distinct shared-memory banks instead of a single
// bank -> eliminates the catastrophic 32-way bank conflict that a row stride
// of D (a multiple of 32) would otherwise cause.
// ---------------------------------------------------------------------------
#define SCORE_CQ 8
#define SCORE_CB 32

// ---------------------------------------------------------------------------
// Kernel C2: score matrix I[B, S, NB] — 2x2 register-blocked variant of
// Kernel C for long sequences (dispatched when NB >= 256, i.e. S >= 1024 at
// r=4).  Profiled instruction accounting on A800 showed the per-cell inner
// loop is dominated by dependent shared-load latency under queueing: each
// output cell streams a full 128-dim k row through the LSU.  This kernel
// keeps 256 threads per block but gives every thread FOUR output cells
// (2 query rows x 2 block columns), so:
//   - per-cell shared loads drop from 6 to 2.5 per dim-group (each staged
//     float4 of q/k is reused for 4 cells instead of 1),
//   - the staged block keys sit in shared as unpadded float4 rows (row
//     stride D4+1 float4 keeps the per-lane LDS.128 conflict-free) staged
//     DIRECTLY with cp.async — the old padded-float transpose (LDG + 4x
//     STS through registers) disappears from the critical path,
//   - the [16 x 64] tile halves staging redundancy vs [8 x 32] (64 B/cell).
// Thread mapping: warp = one query-row pair x all 32 column pairs, so the
// 8 q loads per dim-group stay warp-uniform broadcasts and the 2 k loads
// hit distinct bank quads (col c maps to quad c%8; pairs (c, c+32)).
// fp32 math is unchanged (FMA per dim, ReLU per head, rsqrt(D) scale).
// ---------------------------------------------------------------------------
#define SCORE2_CQ 16
#define SCORE2_CB 64

__global__ void __launch_bounds__(SCORE2_CQ* SCORE2_CB / 4)
    indexer_score_blocked_kernel(
    const float4* __restrict__ qenc,      // [B,S,Hq,D]  (D/4 float4 per head)
    const float4* __restrict__ kbar,      // [B,NB,D]
    float* __restrict__ block_scores,     // [B,S,NB]
    int B, int S, int Hq, int D, int r, int NB)
{
    const int D4 = D / 4;
    const int HqD4 = Hq * D4;
    const int KST = D4 + 1;               // k row stride in float4 (bank pad)

    extern __shared__ char smem_raw[];
    float4* sh_q = reinterpret_cast<float4*>(smem_raw);                        // SCORE2_CQ * HqD4 float4
    float4* sh_k = reinterpret_cast<float4*>(smem_raw + (size_t)SCORE2_CQ * HqD4 * sizeof(float4)); // SCORE2_CB * KST float4

    int qtile = blockIdx.x;
    int btile = blockIdx.y;
    int b     = blockIdx.z;
    int q0 = qtile * SCORE2_CQ;
    int b0 = btile * SCORE2_CB;

    // Same causal-tile early exit as Kernel C.
    if (b0 * r + (r - 1) > q0 + SCORE2_CQ - 1) {
        for (int i = threadIdx.x; i < SCORE2_CQ * SCORE2_CB; i += blockDim.x) {
            int lq = i / SCORE2_CB, lc = i % SCORE2_CB;
            int q = q0 + lq, blk = b0 + lc;
            if (q < S && blk < NB) block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] = -CUDART_INF_F;
        }
        return;
    }

    // Stage q and k tiles; both are plain row-major float4 copies, so both
    // go through cp.async (no register round-trip, no scalar STS).
    {
        const float4* src = qenc + ((size_t)b * S + q0) * HqD4;
        for (int i = threadIdx.x; i < SCORE2_CQ * HqD4; i += blockDim.x) {
            int lq = i / HqD4;
            if (q0 + lq < S) __pipeline_memcpy_async(&sh_q[i], src + i, sizeof(float4));
        }
    }
    {
        const float4* src = kbar + ((size_t)b * NB + b0) * D4;
        for (int i = threadIdx.x; i < SCORE2_CB * D4; i += blockDim.x) {
            int lb = i / D4;
            if (b0 + lb < NB)
                __pipeline_memcpy_async(&sh_k[lb * KST + (i % D4)], src + i, sizeof(float4));
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int tid = threadIdx.x;          // 0 .. 255
    int cp = tid % 32;              // column pair: cols cp and cp+32
    int rp = tid / 32;              // row pair:    rows 2*rp and 2*rp+1

    const float4* k1 = sh_k + (size_t)cp * KST;
    const float4* k2 = sh_k + (size_t)(cp + 32) * KST;

    if (Hq == 4) {
        // One accumulator per (row, head): 16 independent FMA chains — the
        // acc[row][col][head]; one accumulator per (row, col, head) keeps
        // 16 independent FMA chains in flight — the 4-cell body already
        // provides the ILP that required 4 accumulators per head in the
        // single-cell kernel.
        float acc[2][2][4] = {};
        const float4* qp[2][4];
        #pragma unroll
        for (int row = 0; row < 2; ++row)
            #pragma unroll
            for (int h = 0; h < 4; ++h)
                qp[row][h] = sh_q + (size_t)((2 * rp + row) * 4 + h) * D4;
        #pragma unroll 8
        for (int d = 0; d < D4; ++d) {
            float4 x1 = k1[d], x2 = k2[d];
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                #pragma unroll
                for (int h = 0; h < 4; ++h) {
                    float4 u = qp[row][h][d];
                    acc[row][0][h] += u.x * x1.x; acc[row][0][h] += u.y * x1.y;
                    acc[row][0][h] += u.z * x1.z; acc[row][0][h] += u.w * x1.w;
                    acc[row][1][h] += u.x * x2.x; acc[row][1][h] += u.y * x2.y;
                    acc[row][1][h] += u.z * x2.z; acc[row][1][h] += u.w * x2.w;
                }
            }
        }
        // Row A -> cols cp and cp+32, then row B.
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            int q = q0 + 2 * rp + row;
            #pragma unroll
            for (int col = 0; col < 2; ++col) {
                int blk = b0 + cp + col * 32;
                if (q < S && blk < NB) {
                    float score = -CUDART_INF_F;
                    if (blk * r + (r - 1) <= q) {
                        float s = 0.0f;
                        #pragma unroll
                        for (int h = 0; h < 4; ++h) {
                            float dh = acc[row][col][h];
                            if (dh > 0.0f) s += dh;   // ReLU per head
                        }
                        score = s * rsqrtf_((float)D);
                    }
                    block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] = score;
                }
            }
        }
    } else {
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            int q = q0 + 2 * rp + row;
            #pragma unroll
            for (int col = 0; col < 2; ++col) {
                int blk = b0 + cp + col * 32;
                if (!(blk * r + (r - 1) <= q) || q >= S || blk >= NB) continue;
                const float4* kk = col == 0 ? k1 : k2;
                float acc = 0.0f;
                for (int h = 0; h < Hq; ++h) {
                    const float4* qh = sh_q + (size_t)((2 * rp + row) * Hq + h) * D4;
                    float dot = 0.0f;
                    #pragma unroll 4
                    for (int d = 0; d < D4; ++d) {
                        float4 a = qh[d];
                        float4 kx = kk[d];
                        dot += a.x * kx.x + a.y * kx.y + a.z * kx.z + a.w * kx.w;
                    }
                    if (dot > 0.0f) acc += dot;   // ReLU
                }
                block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] =
                    acc * rsqrtf_((float)D);
            }
        }
    }
}

__global__ void indexer_score_kernel(
    const float4* __restrict__ qenc,      // [B,S,Hq,D]  (D/4 float4 per head)
    const float4* __restrict__ kbar,      // [B,NB,D]
    float* __restrict__ block_scores,     // [B,S,NB]
    int B, int S, int Hq, int D, int r, int NB)
{
    const int D4 = D / 4;
    const int HqD4 = Hq * D4;

    extern __shared__ char smem_raw[];
    float4* sh_q = reinterpret_cast<float4*>(smem_raw);            // SCORE_CQ * HqD4 float4
    float*  sh_kf = reinterpret_cast<float*>(smem_raw + (size_t)SCORE_CQ * HqD4 * sizeof(float4)); // SCORE_CB * (D+2) floats

    int qtile = blockIdx.x;
    int btile = blockIdx.y;
    int b     = blockIdx.z;
    int q0 = qtile * SCORE_CQ;
    int b0 = btile * SCORE_CB;

    // The causal mask hides every cell of a tile whose earliest block is not yet
    // visible to its latest query (b0*r + r-1 > q0 + SCORE_CQ - 1).  Early query
    // tiles are almost entirely invisible, so bail out here instead of staging
    // the query/key tiles (global loads + barrier) just to write -inf.
    if (b0 * r + (r - 1) > q0 + SCORE_CQ - 1) {
        for (int i = threadIdx.x; i < SCORE_CQ * SCORE_CB; i += blockDim.x) {
            int lq = i / SCORE_CB, lc = i % SCORE_CB;
            int q = q0 + lq, blk = b0 + lc;
            if (q < S && blk < NB) block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] = -CUDART_INF_F;
        }
        return;
    }

    // Stage the query tile [q0, q0+CQ) x [Hq, D] (float4) with cp.async so the
    // copies proceed while the block-key tile (padded transpose, not cp.async-
    // able) is staged; __pipeline_wait_prior(0) lands right before the compute
    // barrier so the two stages overlap.
    {
        const float4* src = qenc + ((size_t)b * S + q0) * HqD4;
        for (int i = threadIdx.x; i < SCORE_CQ * HqD4; i += blockDim.x) {
            int lq = i / HqD4;
            if (q0 + lq < S) __pipeline_memcpy_async(&sh_q[i], src + i, sizeof(float4));
        }
        __pipeline_commit();
    }
    // Stage the block-key tile [b0, b0+CB) x [D] as padded floats (row stride D+2).
    {
        const float4* src = kbar + ((size_t)b * NB + b0) * D4;
        for (int i = threadIdx.x; i < SCORE_CB * D4; i += blockDim.x) {
            int lb = i / D4;
            if (b0 + lb < NB) {
                float4 v = src[i];
                float* row = sh_kf + (size_t)lb * (D + 2) + (size_t)(i % D4) * 4;
                row[0] = v.x; row[1] = v.y; row[2] = v.z; row[3] = v.w;
            }
        }
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    int tid = threadIdx.x;          // 0 .. SCORE_CQ*SCORE_CB-1
    int c = tid % SCORE_CB;         // block within tile
    int rq = tid / SCORE_CB;        // query within tile
    int q = q0 + rq;
    int blk = b0 + c;

    bool inb = (q < S) && (blk < NB);
    // visibility: p_b + r - 1 <= q with p_b = blk*r
    bool visible = (blk * r + (r - 1) <= q);

    float score = -CUDART_INF_F;
    if (visible && inb) {
        const float2* kk = reinterpret_cast<const float2*>(sh_kf + (size_t)c * (D + 2));
        if (Hq == 4) {
            // Loop exchange (Hq=4 fast path): each block-key element is read from
            // shared memory ONCE per dim and reused across the 4 query heads —
            // the h-outer form re-reads kk[] Hq times (4x redundant smem reads).
            // Per-head accumulators are needed because ReLU is applied per head.
            // Four accumulators per head fully break the serial FMA dependency
            // chain (each += feeds exactly one dim-product per d); the 4-acc
            // version was latency-limited at ~500 cyc/warp-d.
            float d0a = 0, d0b = 0, d0c = 0, d0d = 0;
            float d1a = 0, d1b = 0, d1c = 0, d1d = 0;
            float d2a = 0, d2b = 0, d2c = 0, d2d = 0;
            float d3a = 0, d3b = 0, d3c = 0, d3d = 0;
            const float4* q0 = sh_q + (size_t)(rq * 4 + 0) * D4;
            const float4* q1 = sh_q + (size_t)(rq * 4 + 1) * D4;
            const float4* q2 = sh_q + (size_t)(rq * 4 + 2) * D4;
            const float4* q3 = sh_q + (size_t)(rq * 4 + 3) * D4;
            #pragma unroll 8
            for (int d = 0; d < D4; ++d) {
                // k rows are padded to D+2 so the 4 dims load as 2 float2 reads
                // (half the load-issue slots; the 2-way bank spread is hidden by
                // the 2-cycle LDS.64 = same smem cycles as 4 scalar LDS.32).
                float2 k01 = kk[d * 2 + 0];
                float2 k23 = kk[d * 2 + 1];
                float k0 = k01.x, k1 = k01.y, k2 = k23.x, k3 = k23.y;
                float4 a0 = q0[d], a1 = q1[d], a2 = q2[d], a3 = q3[d];
                d0a += a0.x * k0;  d0b += a0.y * k1;
                d0c += a0.z * k2;  d0d += a0.w * k3;
                d1a += a1.x * k0;  d1b += a1.y * k1;
                d1c += a1.z * k2;  d1d += a1.w * k3;
                d2a += a2.x * k0;  d2b += a2.y * k1;
                d2c += a2.z * k2;  d2d += a2.w * k3;
                d3a += a3.x * k0;  d3b += a3.y * k1;
                d3c += a3.z * k2;  d3d += a3.w * k3;
            }
            float acc = 0.0f;
            float d0 = (d0a + d0b) + (d0c + d0d);
            float d1 = (d1a + d1b) + (d1c + d1d);
            float d2 = (d2a + d2b) + (d2c + d2d);
            float d3 = (d3a + d3b) + (d3c + d3d);
            if (d0 > 0.0f) acc += d0;   // ReLU per head
            if (d1 > 0.0f) acc += d1;
            if (d2 > 0.0f) acc += d2;
            if (d3 > 0.0f) acc += d3;
            score = acc * rsqrtf_((float)D);
        } else {
            float acc = 0.0f;
            for (int h = 0; h < Hq; ++h) {
                const float4* qh = sh_q + (size_t)(rq * Hq + h) * D4;
                float dot = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D4; ++d) {
                    float4 a = qh[d];
                    float2 k01 = kk[d * 2 + 0], k23 = kk[d * 2 + 1];
                    dot += a.x * k01.x + a.y * k01.y
                         + a.z * k23.x + a.w * k23.y;
                }
                if (dot > 0.0f) acc += dot;   // ReLU
            }
            score = acc * rsqrtf_((float)D);
        }
    }
    if (inb) block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] = score;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Kernel D: per-query TopK via a TWO-ROUND RADIX select + bitonic sort of
// EXACTLY K winners (replaces a full O(P log^2 P) bitonic sort of all P2
// candidates, and avoids padding a ~K+epsilon winner set up to P2 = 2*K).
//
// Grid: one thread block per (batch, query).
// Dynamic shared memory: 2 * P_pad * sizeof(uint64) + 256 * sizeof(int).
//
// Each candidate is one PACKED uint64
//     [63:32] sortable score   (bit-inverted monotone fp32 key)
//     [31:0]  reversed index   (0xFFFFFFFF - block_idx)
// so the descending order by key is the exact reference order (higher score
// wins; equal score -> lower block index wins).  P < KB is never an issue: we
// select min(KB, P).
//
// Radix plan (CPU-validated in tests/radix_topk_proto.py, 5000/5000 vs a full
// sort on normal/uniform/heavy-tie/constant data):
//   1. histogram the top 8 bits (bits 63..56) of the valid packed keys
//   2. walk digits high->low accumulating counts; pivot b1 = first digit whose
//      running count reaches K_eff; cnt_gt = #(>b1); need = K_eff - cnt_gt
//   3. collect >b1 -> s_out[0..cnt_gt) and ==b1 -> s_eq (s_pack region)
//   4. round 2: sort s_eq descending (it is only hist[b1] ~ P/256 elements for
//      uniform data) and take its top `need` -> append after cnt_gt.  Now the
//      winner set is EXACTLY K_eff, so the final bitonic runs on P2s=next_pow2(K)
//      instead of next_pow2(~K+epsilon) which was often 2x K.
//   5. bitonic sort s_out[0..K_eff) descending -> exact reference order.
// ---------------------------------------------------------------------------
// Packed key for a -inf score: sortable(-inf) = 0x007FFFFF (the only float that
// maps there), so valid keys can never collide with this sentinel and the
// histogram/collect scans test `pk != NEG_INF_KEY` to skip invalid candidates.
#define NEG_INF_KEY ((uint64_t)0x007FFFFFu << 32)

// Warp-shuffle hybrid bitonic sort of P2S elements (P2S = power of 2 <= 512),
// templated so every network stage unrolls to straight-line code.  Thread t
// owns the consecutive pair (2t, 2t+1) in registers; exchange stages with
// distance j <= 32 stay inside the warp (__shfl_xor, zero barriers) and
// j == 1 is a thread-local compare-swap.  Only the distances 64/128/256 of
// the k = 128/256/512 merges round-trip through shared memory (bufA/bufB),
// one __syncthreads per round-trip.  Ping-pong buffers mean no extra barrier
// separates consecutive rounds: round m's write buffer was round m-2's read
// buffer and round m-1's write-barrier already drained round m-2's reads.
// Caller passes the pair pre-loaded (slots >= K_eff must be 0, which sorts
// below every valid packed key); the sorted pair is returned the same way.
// Same comparator as the old all-smem network: strict descending by
// (sortable score, reversed index).
template <int P2S>
__device__ __forceinline__ void topk_reg_bitonic_sort(
    uint64_t& v0, uint64_t& v1, int t, uint64_t* bufA, uint64_t* bufB)
{
    const int p = 2 * t;
    #pragma unroll
    for (int k = 2; k <= P2S; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const bool up = ((p & k) == 0);
            if (j == 1) {
                // Thread-local pair (p, p+1); this thread holds the lower
                // position, so it keeps hi iff up: with c = (v0 < v1),
                // v0' = (c == up) ? v1 : v0 and vice versa.
                const bool c = (v0 < v1);
                const bool e = (c == up);
                uint64_t nv0 = e ? v1 : v0;
                v1 = e ? v0 : v1;
                v0 = nv0;
            } else if (j <= 32) {
                // Intra-warp exchange with partner thread t ^ (j>>1).  The
                // lower position keeps hi iff up and the upper iff !up, so
                // this thread takes the partner's value exactly when
                // (lower == up): v' = ((v < partner) == sw) ? partner : v.
                uint64_t p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, j >> 1);
                uint64_t p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, j >> 1);
                const bool sw = (((p & j) == 0) == up);
                v0 = ((v0 < p0) == sw) ? p0 : v0;
                v1 = ((v1 < p1) == sw) ? p1 : v1;
            } else {
                // cross-warp exchange: one barrier per round-trip
                if (p < P2S) {
                    bufA[p]     = v0;
                    bufA[p + 1] = v1;
                }
                __syncthreads();
                uint64_t p0 = 0, p1 = 0;
                if (p < P2S) {
                    p0 = bufA[p ^ j];
                    p1 = bufA[(p + 1) ^ j];
                }
                const bool sw = (((p & j) == 0) == up);
                v0 = ((v0 < p0) == sw) ? p0 : v0;
                v1 = ((v1 < p1) == sw) ? p1 : v1;
                uint64_t* tmp = bufA; bufA = bufB; bufB = tmp;
            }
        }
    }
}

__global__ void indexer_topk_kernel(
    const float* __restrict__ block_scores,   // [B,S,NB]
    int* __restrict__ block_indices,          // [B,S,KB]
    float* __restrict__ selected_scores,      // [B,S,KB]
    int B, int S, int NB, int KB, int P_pad)
{
    extern __shared__ char smem_raw[];
    uint64_t* s_pack = reinterpret_cast<uint64_t*>(smem_raw);   // P_pad (reused as s_eq in round 2)
    uint64_t* s_out  = reinterpret_cast<uint64_t*>(smem_raw + (size_t)P_pad * sizeof(uint64_t)); // P_pad
    int* s_hist = reinterpret_cast<int*>(smem_raw + (size_t)2 * P_pad * sizeof(uint64_t));       // 256

    int bq = blockIdx.x;
    if (bq >= B * S) return;
    int q = bq % S;
    int b = bq / S;

    const float* scores = block_scores + (size_t)b * S * NB + (size_t)q * NB;

    // Pack ALL NB candidates to FIXED positions (no atomics) AND histogram their
    // top byte in the same pass over the global scores: valid keys packed as
    // (sortable score, reversed index), invalid as NEG_INF_KEY (the only float
    // mapping to sortable 0x007FFFFF, so it never collides and the collect scan
    // skips it).  The old dense compaction used a single contended atomicAdd,
    // and this fusion drops one full s_pack pass + barrier.
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < NB; i += blockDim.x) {
        float s = scores[i];
        uint32_t sb = __float_as_uint(s);
        uint64_t pk;
        if (s != -CUDART_INF_F) {
            uint32_t sortable = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            pk = ((uint64_t)sortable << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)i);
            atomicAdd(&s_hist[(int)((pk >> 56) & 0xFFu)], 1);
        } else {
            pk = NEG_INF_KEY;
        }
        s_pack[i] = pk;
    }
    __syncthreads();

    // ---- find pivot (thread 0): P = total valid, K_eff = min(KB, P), b1 ----
    __shared__ int K_eff, sb1, scnt_gt, sneed;
    if (threadIdx.x == 0) {
        int total = 0;
        for (int b = 255; b >= 0; --b) total += s_hist[b];
        int k = std::min(KB, total);
        int cnt = 0, need = k, b1 = 0;
        for (int b = 255; b >= 0; --b) {
            int c = s_hist[b];
            if (cnt + c >= k) { b1 = b; need = k - cnt; break; }
            cnt += c;
        }
        K_eff = k; sb1 = b1; scnt_gt = cnt; sneed = need;
    }
    __syncthreads();

    if (K_eff > 0) {
        // ---- collect: >b1 -> s_out head; ==b1 -> s_eq (in-place into s_pack) ----
        __shared__ int sgt_cnt, seq_cnt;
        if (threadIdx.x == 0) { sgt_cnt = 0; seq_cnt = 0; }
        __syncthreads();
        for (int i = threadIdx.x; i < NB; i += blockDim.x) {
            uint64_t pk = s_pack[i];
            if (pk == NEG_INF_KEY) continue;
            int d = (int)((pk >> 56) & 0xFFu);
            if (d > sb1) {
                s_out[atomicAdd(&sgt_cnt, 1)] = pk;
            } else if (d == sb1) {
                s_pack[atomicAdd(&seq_cnt, 1)] = pk;   // s_eq in-place
            }
        }
        __syncthreads();

        // ---- round 2: narrow the ==b1 slab, then warp-shuffle-sort it ----
        //
        // The ==b1 slab can be large (ReLU scores cluster in a few exponent
        // buckets, so hist[b1] ~ hundreds) but we only need its top-`rem`
        // elements.  Sorting the whole slab is O(eq_n log^2 eq_n) bitonic work;
        // instead, narrow it one 8-bit byte at a time (CUB block_topk_air
        // style): each round histograms the current byte of the CANDIDATE
        // slab, collects everything above the pivot bucket as final winners,
        // and keeps only the ==pivot elements as the next, smaller slab.
        // Real ReLU scores spread over ~256 sub-buckets in the second byte, so
        // one round shrinks the slab from ~400 to a handful; once the slab fits
        // one warp (<=32) it is sorted entirely in registers with warp-shuffle
        // bitonic (no barriers, no shared-memory round trips).  Since the
        // reversed index fills the low 32 bits of the packed key, equal scores
        // tie-break by lowest block index, matching the reference.  The final
        // bitonic below orders all K_eff winners anyway.
        //
        // The shrinking slab ping-pongs between the low and high halves of
        // s_pack so compaction never reads and writes the same slot.
        int n = seq_cnt;               // current slab size
        int src = 0;                   // current slab base in s_pack
        int dst = P_pad >> 1;          // compaction target base
        int rem = sneed;               // still need `rem` winners from the slab
        if (n > P_pad / 2) {
            // Degenerate: slab too large for the half-and-half scheme.  This
            // can only happen with extreme tie clustering; fall back to the
            // exact full-slab bitonic for this query.
            n = seq_cnt;
            int P2e = 1;
            while (P2e < n) P2e <<= 1;
            for (int i = n + threadIdx.x; i < P2e; i += blockDim.x) {
                s_pack[i] = ((uint64_t)0x007FFFFFu << 32) | (uint64_t)0;   // -inf pad
            }
            __syncthreads();
            for (int k = 2; k <= P2e; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = threadIdx.x; i < P2e; i += blockDim.x) {
                        int l = i ^ j;
                        if (l > i) {
                            bool up = ((i & k) == 0);
                            uint64_t a = s_pack[i];
                            uint64_t b = s_pack[l];
                            if ((up && a < b) || (!up && b < a)) {
                                s_pack[i] = b;
                                s_pack[l] = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            for (int j = threadIdx.x; j < sneed; j += blockDim.x) {
                s_out[scnt_gt + j] = s_pack[j];
            }
            __syncthreads();
        } else {
            __shared__ int s_owin;   // winners appended after cnt_gt
            if (threadIdx.x == 0) s_owin = 0;
            __syncthreads();
            int byte = 48;
            while (n > 32 && byte >= 0 && rem > 0) {
                for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
                __syncthreads();
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    atomicAdd(&s_hist[(int)((s_pack[src + i] >> byte) & 0xFFu)], 1);
                }
                __syncthreads();
                __shared__ int sb2, sneed2;
                if (threadIdx.x == 0) {
                    int cnt = 0;
                    int b2 = 0;
                    for (int b = 255; b >= 0; --b) {
                        int c = s_hist[b];
                        if (cnt + c >= rem) { b2 = b; sneed2 = rem - cnt; break; }
                        cnt += c;
                    }
                    sb2 = b2;
                }
                __syncthreads();
                __shared__ int s_keep;   // compaction counter into `dst`
                if (threadIdx.x == 0) s_keep = 0;
                __syncthreads();
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    uint64_t pk = s_pack[src + i];
                    int d = (int)((pk >> byte) & 0xFFu);
                    if (d > sb2) {
                        s_out[scnt_gt + atomicAdd(&s_owin, 1)] = pk;   // final winner
                    } else if (d == sb2) {
                        s_pack[dst + atomicAdd(&s_keep, 1)] = pk;      // next slab
                    }
                }
                __syncthreads();
                n = s_keep;
                int tmp = src; src = dst; dst = tmp;
                rem = sneed2;
                byte -= 8;
            }
            // The slab now fits one warp: warp-shuffle bitonic (descending) on
            // the full packed key; lanes 0..rem-1 are the final winners.
            if (rem > 0) {
                uint64_t v = (threadIdx.x < n) ? s_pack[src + threadIdx.x]
                                              : ((uint64_t)0x007FFFFFu << 32);
                if (threadIdx.x < 32) {
                    for (int k = 2; k <= 32; k <<= 1) {
                        for (int j = k >> 1; j >= 1; j >>= 1) {
                            uint64_t o = __shfl_xor_sync(0xFFFFFFFFu, v, j);
                            bool up = ((threadIdx.x & k) == 0);
                            uint64_t lo = (v < o) ? v : o;
                            uint64_t hi = (v < o) ? o : v;
                            bool lower = ((threadIdx.x & j) == 0);
                            v = lower ? (up ? hi : lo) : (up ? lo : hi);
                        }
                    }
                    if (threadIdx.x < rem) {
                        s_out[scnt_gt + s_owin + threadIdx.x] = v;   // desc order kept
                    }
                }
                __syncthreads();
            }
        }

        // ---- final sort (descending) of the EXACTLY K_eff winners ----
        int P2s = 1;
        while (P2s < K_eff) P2s <<= 1;
        if (P2s <= 512 && KB <= 512) {
            // Warp-shuffle hybrid bitonic (see topk_reg_bitonic_sort): the
            // 6-barrier register/shuffle network replaces the 45-barrier
            // all-shared-memory bitonic.  Slots >= K_eff are virtual pad
            // keys (0 sorts below every valid packed key, and
            // s_out[K_eff, P2s) is uninitialized) carried in registers only
            // -- positions >= K_eff are rewritten as -1/-inf at the output
            // anyway, so the smem padding pass is dropped too, and the top
            // K_eff is written straight from registers (no trailing
            // barrier).  Templated on P2s so all network stages unroll.
            const int t = threadIdx.x;
            const int p = 2 * t;
            uint64_t v0 = (p < K_eff) ? s_out[p] : (uint64_t)0;
            uint64_t v1 = (p + 1 < K_eff) ? s_out[p + 1] : (uint64_t)0;
            switch (P2s) {
                case 512: topk_reg_bitonic_sort<512>(v0, v1, t, s_pack, s_out); break;
                case 256: topk_reg_bitonic_sort<256>(v0, v1, t, s_pack, s_out); break;
                case 128: topk_reg_bitonic_sort<128>(v0, v1, t, s_pack, s_out); break;
                case 64:  topk_reg_bitonic_sort<64>(v0, v1, t, s_pack, s_out);  break;
                case 32:  topk_reg_bitonic_sort<32>(v0, v1, t, s_pack, s_out);  break;
                case 16:  topk_reg_bitonic_sort<16>(v0, v1, t, s_pack, s_out);  break;
                case 8:   topk_reg_bitonic_sort<8>(v0, v1, t, s_pack, s_out);   break;
                case 4:   topk_reg_bitonic_sort<4>(v0, v1, t, s_pack, s_out);   break;
                case 2:   topk_reg_bitonic_sort<2>(v0, v1, t, s_pack, s_out);   break;
                default: break;   // P2s == 1: single winner, already in v0
            }
            // Write the top K_eff straight from registers: thread t owns
            // positions (2t, 2t+1) of the descending order.
            int*   inds = block_indices   + (size_t)bq * KB;
            float* sels = selected_scores + (size_t)bq * KB;
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int pos = p + s;
                if (pos < KB) {
                    if (pos < K_eff) {
                        uint64_t pk = s ? v1 : v0;
                        uint32_t sb = (uint32_t)(pk >> 32);
                        uint32_t fb = (sb & 0x80000000u) ? (sb & 0x7FFFFFFFu) : ~sb;
                        uint32_t ridx = (uint32_t)(pk & 0xFFFFFFFFu);
                        inds[pos] = (int)(0xFFFFFFFFu - ridx);
                        sels[pos] = __uint_as_float(fb);
                    } else {
                        inds[pos] = -1;
                        sels[pos] = -CUDART_INF_F;
                    }
                }
            }
        } else {
            // Large winner sets (K_eff or KB > 512): keep the exact original
            // all-shared-memory network.
            for (int i = K_eff + threadIdx.x; i < P2s; i += blockDim.x) {
                s_out[i] = ((uint64_t)0x007FFFFFu << 32) | (uint64_t)0;   // -inf pad
            }
            __syncthreads();
            for (int k = 2; k <= P2s; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = threadIdx.x; i < P2s; i += blockDim.x) {
                        int l = i ^ j;
                        if (l > i) {
                            bool up = ((i & k) == 0);
                            uint64_t a = s_out[i];
                            uint64_t b = s_out[l];
                            if ((up && a < b) || (!up && b < a)) {
                                s_out[i] = b;
                                s_out[l] = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

    if (K_eff == 0 || K_eff > 512 || KB > 512) {
        // ---- write top K_eff from the descending head of s_out ----
        int*   inds = block_indices   + (size_t)bq * KB;
        float* sels = selected_scores + (size_t)bq * KB;
        for (int j = threadIdx.x; j < KB; j += blockDim.x) {
            if (j < K_eff) {
                uint64_t pk = s_out[j];
                uint32_t sb = (uint32_t)(pk >> 32);
                uint32_t fb = (sb & 0x80000000u) ? (sb & 0x7FFFFFFFu) : ~sb;
                uint32_t ridx = (uint32_t)(pk & 0xFFFFFFFFu);
                inds[j] = (int)(0xFFFFFFFFu - ridx);
                sels[j] = __uint_as_float(fb);
            } else {
                inds[j] = -1;
                sels[j] = -CUDART_INF_F;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel E: FUSED score + TopK (topK-only production fast path).
//
// One CTA per (batch, query) — the same grid as the TopK kernel — but the block
// scores are computed inline and written straight into the shared-memory TopK
// buffer.  The dense [B,S,NB] block_scores matrix is NEVER materialized, which
// removes the 64MB write + 64MB read of the two-stage pipeline at S=8192.
//
//   grid : B*S blocks (one query each)
//   smem : s_pack[P_pad] (uint64) + sh_q[Hq*D] + sh_k[IDXER_BLOCK_N * D]
//
// Score math is identical to kernel C (scalar float4 dot, fp32-exact).  Unlike
// the first fused version (which re-read every block key from L2, stalling on
// the load latency chain), kbar is staged into shared memory in IDXER_BLOCK_N
// tiles so the dot products run against ~30-cycle smem loads.
//
// The TopK sort stores each candidate as one PACKED uint64:
//     [31:0]  sortable score  (bit-inverted monotone fp32 key)
//     [63:32] reversed index  (0xFFFFFFFF - block_idx)
// Ascending-by-key reproduces exactly kernel D's bitonic semantics: higher
// score sorts higher; among equal scores the lower block index sorts higher
// (so it is selected first).  A single 64-bit LDS/STS halves the shared-memory
// traffic of the two-array (score + index) version.
// ---------------------------------------------------------------------------
#define IDXER_BLOCK_N 32

__global__ void indexer_fused_score_topk_kernel(
    const float4* __restrict__ qenc,          // [B,S,Hq,D]  (D/4 float4 per head)
    const float4* __restrict__ kbar,          // [B,NB,D]
    int*    __restrict__ block_indices,       // [B,S,KB]
    float*  __restrict__ selected_scores,     // [B,S,KB]
    int B, int S, int Hq, int D, int r, int NB, int KB, int P_pad)
{
    extern __shared__ char smem_raw[];
    uint64_t* s_pack = reinterpret_cast<uint64_t*>(smem_raw);                   // P_pad uint64
    size_t q_off = (((size_t)P_pad * sizeof(uint64_t) + 15) & ~(size_t)15);     // 16B-aligned
    float* sh_q = reinterpret_cast<float*>(smem_raw + q_off);                   // Hq*D floats
    size_t k_off = ((q_off + (size_t)Hq * D * sizeof(float) + 15) & ~(size_t)15);
    float* sh_k = reinterpret_cast<float*>(smem_raw + k_off);                   // IDXER_BLOCK_N*D floats

    const int D4 = D / 4;

    int bq = blockIdx.x;
    if (bq >= B * S) return;
    int q = bq % S;
    int b = bq / S;

    // Number of fully-visible blocks for this query: blocks blk with
    // blk*r + (r-1) <= q  =>  blk in [0, P).  Same rule as kernel C.
    int P = (q >= r - 1) ? (q - (r - 1)) / r + 1 : 0;
    if (P > NB) P = NB;

    // Stage this query's encoded heads [Hq, D] into shared memory (float4).
    const float4* qsrc = qenc + (size_t)bq * (Hq * D4);
    {
        float4* sh4 = reinterpret_cast<float4*>(sh_q);
        for (int i = threadIdx.x; i < Hq * D4; i += blockDim.x) sh4[i] = qsrc[i];
    }
    __syncthreads();

    // Score visible blocks [0, P) in tiles.  Each tile stages IDXER_BLOCK_N block
    // keys into shared memory (all 256 threads cooperate on the coalesced
    // float4 copy); then all 256 threads compute the tile's scores with a
    // 4-thread-per-block split: thread (blk, h) does the fp32 dot for ONE query
    // head over the full D dims (32 float4 FMAs), applies its ReLU, and a 2-step
    // shuffle-reduce across the 4 head lanes yields score[blk].  This keeps the
    // whole CTA busy (no idle threads, unlike thread-per-block scoring) while
    // keeping the shared-memory reads broadcast / conflict-free.
    // The tile row stride is D4+1 (one float4 of padding): with D=128 the raw
    // D4=32 stride puts every row at the same bank group (512B apart = 128 banks
    // = a full bank cycle), causing a 32-way conflict; stride 33 is coprime with
    // the 32-lane bank set and gives a conflict-free float4 layout.
    const int KD4 = D4 + 1;
    const float4* kb = kbar + (size_t)b * NB * D4;
    for (int b0 = 0; b0 < P; b0 += IDXER_BLOCK_N) {
        int nblk = std::min(IDXER_BLOCK_N, P - b0);
        {
            float4* shk4 = reinterpret_cast<float4*>(sh_k);
            for (int i = threadIdx.x; i < IDXER_BLOCK_N * D4; i += blockDim.x) {
                int lb = i / D4;
                if (lb < nblk) shk4[(size_t)lb * KD4 + (i % D4)] = kb[(size_t)(b0 + lb) * D4 + (i % D4)];
            }
        }
        __syncthreads();
        const float4* shk4 = reinterpret_cast<const float4*>(sh_k);
        {
            int blk = threadIdx.x >> 2;   // thread group = (blk, head)
            int h   = threadIdx.x & 3;
            float relu = 0.0f;
            if (blk < nblk) {
                const float4* qh = reinterpret_cast<const float4*>(sh_q) + (size_t)h * D4;
                const float4* krow = shk4 + (size_t)blk * KD4;
                float dot = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D4; ++d) {
                    float4 a = qh[d];
                    float4 kk = krow[d];
                    dot += a.x * kk.x + a.y * kk.y + a.z * kk.z + a.w * kk.w;
                }
                relu = dot > 0.0f ? dot : 0.0f;   // ReLU per head
            }
            // Reduce across the 4 head lanes of this block.
            relu += __shfl_xor_sync(0xFFFFFFFFu, relu, 1);
            relu += __shfl_xor_sync(0xFFFFFFFFu, relu, 2);
            if (blk < nblk && (threadIdx.x & 3) == 0) {
                float score = relu * rsqrtf_((float)D);
                uint32_t sb = __float_as_uint(score);
                uint32_t sortable = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                s_pack[b0 + blk] = ((uint64_t)sortable << 32)
                                 | (uint64_t)(0xFFFFFFFFu - (uint32_t)(b0 + blk));
            }
        }
        __syncthreads();   // protect sh_k before the next tile overwrites it
    }

    // Pad [P, P_pad) with -inf / -1 (sorts to the ascending front, never selected).
    // -inf: sortable key = ~0xFF800000 = 0x007FFFFF ; reversed(-1) = 0.
    for (int i = P + threadIdx.x; i < P_pad; i += blockDim.x) {
        s_pack[i] = ((uint64_t)0x007FFFFFu << 32) | (uint64_t)0;
    }
    __syncthreads();

    // Bitonic merge sort (descending) over the full P2 valid window.  A
    // truncated window-only variant was tried but is NOT a valid topK selector:
    // in the large-j cross passes, winners must bubble in from the outside half
    // through exactly those comparisons the window truncation skips (verified by
    // a hand-run counter-example).  So we keep the full network, which is
    // provably correct and already cheaper than the two-stage topk because the
    // packed uint64 key halves the shared-memory traffic (one 8B compare+swap
    // vs. two 4B arrays).
    int P2 = 1;
    while (P2 < P) P2 <<= 1;

    for (int k = 2; k <= P2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < P2; i += blockDim.x) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & k) == 0);
                    uint64_t a = s_pack[i];
                    uint64_t b = s_pack[l];
                    if ((up && a < b) || (!up && b < a)) {
                        s_pack[i] = b;
                        s_pack[l] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write the top KB from the descending window head [0, k).
    int k = std::min(KB, P);
    int*   inds = block_indices   + (size_t)bq * KB;
    float* sels = selected_scores + (size_t)bq * KB;
    for (int j = threadIdx.x; j < KB; j += blockDim.x) {
        if (j < k) {
            uint64_t pk = s_pack[j];
            uint32_t sb = (uint32_t)(pk >> 32);
            uint32_t fb = (sb & 0x80000000u) ? (sb & 0x7FFFFFFFu) : ~sb;
            uint32_t ridx = (uint32_t)(pk & 0xFFFFFFFFu);
            inds[j] = (int)(0xFFFFFFFFu - ridx);
            sels[j] = __uint_as_float(fb);
        } else {
            inds[j] = -1;
            sels[j] = -CUDART_INF_F;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> qsa_indexer_forward(
    torch::Tensor q,          // [B,S,Hq,D]
    torch::Tensor raw_keys,   // [B,S,D]
    torch::Tensor cos_q,      // [B,S,R]
    torch::Tensor sin_q,      // [B,S,R]
    torch::Tensor cos_k,      // [B,S,R]
    torch::Tensor sin_k,      // [B,S,R]
    int64_t r,                // compress ratio
    int64_t block_topk)       // KB
{
    CHECK_CUDA(q); CHECK_CONTIG(q); CHECK_FLOAT(q);
    CHECK_CUDA(raw_keys); CHECK_CONTIG(raw_keys); CHECK_FLOAT(raw_keys);
    CHECK_CUDA(cos_q); CHECK_CONTIG(cos_q); CHECK_FLOAT(cos_q);
    CHECK_CUDA(sin_q); CHECK_CONTIG(sin_q); CHECK_FLOAT(sin_q);
    CHECK_CUDA(cos_k); CHECK_CONTIG(cos_k); CHECK_FLOAT(cos_k);
    CHECK_CUDA(sin_k); CHECK_CONTIG(sin_k); CHECK_FLOAT(sin_k);

    int B  = q.size(0);
    int S  = q.size(1);
    int Hq = q.size(2);
    int D  = q.size(3);
    int R  = cos_q.size(2);
    TORCH_CHECK(S % r == 0, "S must be divisible by compress ratio");
    TORCH_CHECK(D == 128, "indexer encode/score kernels are specialized for D=128");
    int NB = S / (int)r;
    TORCH_CHECK(raw_keys.size(0) == B && raw_keys.size(1) == S && raw_keys.size(2) == D);
    TORCH_CHECK(cos_q.size(0) == B && cos_q.size(1) == S);
    TORCH_CHECK(R == 64, "indexer encode kernel is specialized for R=64");

    auto opts = q.options();
    auto block_scores = torch::empty({B, S, NB}, opts);
    auto block_indices = torch::empty({B, S, (int)block_topk}, at::TensorOptions().dtype(at::kInt).device(q.device()));
    auto selected_scores = torch::empty({B, S, (int)block_topk}, opts);

    auto kbar = torch::empty({B, NB, D}, opts);
    auto qenc = torch::empty({B, S, Hq, D}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

#define CHECK_LAUNCH(name) { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) TORCH_CHECK(false, name, " launch failed: ", cudaGetErrorString(e)); }

    // Kernel A: pool + norm + rope block keys
    {
        int total = B * NB * (D / 4);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        indexer_pool_keys_kernel<<<blocks, threads, 0, stream>>>(
            raw_keys.data_ptr<float>(),
            cos_k.data_ptr<float>(), sin_k.data_ptr<float>(),
            kbar.data_ptr<float>(),
            B, S, D, R, (int)r, NB);
    CHECK_LAUNCH("pool_keys");
    }

    // Kernel B: encode queries (one warp per (b,s,h), dims-contiguous loads)
    {
        int total = B * S * Hq * (D / 4);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        indexer_encode_q_kernel<<<blocks, threads, 0, stream>>>(
            q.data_ptr<float>(),
            cos_q.data_ptr<float>(), sin_q.data_ptr<float>(),
            qenc.data_ptr<float>(),
            B, S, Hq, D, R);
    CHECK_LAUNCH("encode_q");
    }

    // Kernel C: score matrix (tiled).  Scalar float4-dot kernel (fp32-exact).
    // A tensor-core (tf32 mma) variant was prototyped but measured slower than
    // the scalar kernel at production shapes (m16n8k8 tf32 load+convert cost
    // dominated the mma), so the scalar path is kept for all shapes.
    {
        int D4 = D / 4;
        int HqD4 = Hq * D4;
        // Long sequences dispatch to the 2x2 register-blocked Kernel C2
        // (NB >= 256 keeps every k tile full; the S*NB gate keeps small
        // problems on Kernel C, whose smaller blocks fill the GPU better).
        bool use_blocked = (NB >= 256) && ((int64_t)S * NB >= (int64_t)512 * 1024);
        if (use_blocked) {
            int smem2 = SCORE2_CQ * HqD4 * (int)sizeof(float4)
                      + SCORE2_CB * (D4 + 1) * (int)sizeof(float4);
            cudaFuncSetAttribute(indexer_score_blocked_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem2);
            dim3 grid2((S + SCORE2_CQ - 1) / SCORE2_CQ,
                       (NB + SCORE2_CB - 1) / SCORE2_CB,
                       B);
            indexer_score_blocked_kernel<<<grid2, SCORE2_CQ * SCORE2_CB / 4, smem2, stream>>>(
                reinterpret_cast<const float4*>(qenc.data_ptr<float>()),
                reinterpret_cast<const float4*>(kbar.data_ptr<float>()),
                block_scores.data_ptr<float>(),
                B, S, Hq, D, (int)r, NB);
        } else {
        int smem_bytes = SCORE_CQ * HqD4 * (int)sizeof(float4)
                       + SCORE_CB * (D + 2) * (int)sizeof(float);
        // Larger query tiles can push past the 48KB default; opt in (A800 max 164KB).
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(indexer_score_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes);
        }
        dim3 grid((S + SCORE_CQ - 1) / SCORE_CQ,
                  (NB + SCORE_CB - 1) / SCORE_CB,
                  B);
        indexer_score_kernel<<<grid, SCORE_CQ * SCORE_CB, smem_bytes, stream>>>(
            reinterpret_cast<const float4*>(qenc.data_ptr<float>()),
            reinterpret_cast<const float4*>(kbar.data_ptr<float>()),
            block_scores.data_ptr<float>(),
            B, S, Hq, D, (int)r, NB);
        }
    CHECK_LAUNCH("score");
    }

    // Kernel D: per-query TopK (bitonic)
    {
        int P_pad = 1;
        while (P_pad < NB) P_pad <<= 1;
        // Radix kernel D layout: s_pack[P_pad] uint64 + s_out[P_pad] uint64 + s_hist[256] int.
        int smem_bytes = (int)(2 * (size_t)P_pad * sizeof(uint64_t) + 256 * sizeof(int));
        int bq_total = B * S;
        int threads = 256;
        // SM80 default dynamic shared memory limit is 48KB per block. For
        // n_blocks up to 4096 (P_pad=4096 -> 32KB) we stay within it; if the
        // model grows beyond that, opt in via cudaFuncSetAttribute.
        TORCH_CHECK(smem_bytes <= 48 * 1024,
                    "topk shared memory ", smem_bytes,
                    " bytes exceeds 48KB; n_blocks too large");
        indexer_topk_kernel<<<bq_total, threads, smem_bytes, stream>>>(
            block_scores.data_ptr<float>(),
            block_indices.data_ptr<int>(),
            selected_scores.data_ptr<float>(),
            B, S, NB, (int)block_topk, P_pad);
    CHECK_LAUNCH("topk");
    }

    return {block_scores, block_indices, selected_scores};
}

// ---------------------------------------------------------------------------
// Host launcher — topK-only fused fast path.
//
// Same math as qsa_indexer_forward (pool keys -> encode queries -> block-causal
// score -> TopK), but the score matrix is never materialized: the fused kernel
// scores each query's visible blocks straight into its shared-memory TopK
// buffer.  Returns only {block_indices, selected_scores} — the dense
// [B,S,NB] block_scores are dropped by contract, which is what makes the long-
// sequence case viable (no 64MB write + 64MB read at S=8192).
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> qsa_indexer_topk_only_forward(
    torch::Tensor q,          // [B,S,Hq,D]
    torch::Tensor raw_keys,   // [B,S,D]
    torch::Tensor cos_q,      // [B,S,R]
    torch::Tensor sin_q,      // [B,S,R]
    torch::Tensor cos_k,      // [B,S,R]
    torch::Tensor sin_k,      // [B,S,R]
    int64_t r,                // compress ratio
    int64_t block_topk)       // KB
{
    CHECK_CUDA(q); CHECK_CONTIG(q); CHECK_FLOAT(q);
    CHECK_CUDA(raw_keys); CHECK_CONTIG(raw_keys); CHECK_FLOAT(raw_keys);
    CHECK_CUDA(cos_q); CHECK_CONTIG(cos_q); CHECK_FLOAT(cos_q);
    CHECK_CUDA(sin_q); CHECK_CONTIG(sin_q); CHECK_FLOAT(sin_q);
    CHECK_CUDA(cos_k); CHECK_CONTIG(cos_k); CHECK_FLOAT(cos_k);
    CHECK_CUDA(sin_k); CHECK_CONTIG(sin_k); CHECK_FLOAT(sin_k);

    int B  = q.size(0);
    int S  = q.size(1);
    int Hq = q.size(2);
    int D  = q.size(3);
    int R  = cos_q.size(2);
    TORCH_CHECK(S % r == 0, "S must be divisible by compress ratio");
    TORCH_CHECK(D == 128, "indexer encode/score kernels are specialized for D=128");
    int NB = S / (int)r;
    TORCH_CHECK(raw_keys.size(0) == B && raw_keys.size(1) == S && raw_keys.size(2) == D);
    TORCH_CHECK(cos_q.size(0) == B && cos_q.size(1) == S);
    TORCH_CHECK(R == 64, "indexer encode kernel is specialized for R=64");

    auto opts = q.options();
    auto block_indices = torch::empty({B, S, (int)block_topk}, at::TensorOptions().dtype(at::kInt).device(q.device()));
    auto selected_scores = torch::empty({B, S, (int)block_topk}, opts);

    auto kbar = torch::empty({B, NB, D}, opts);
    auto qenc = torch::empty({B, S, Hq, D}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Kernel A: pool + norm + rope block keys
    {
        int total = B * NB * (D / 4);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        indexer_pool_keys_kernel<<<blocks, threads, 0, stream>>>(
            raw_keys.data_ptr<float>(),
            cos_k.data_ptr<float>(), sin_k.data_ptr<float>(),
            kbar.data_ptr<float>(),
            B, S, D, R, (int)r, NB);
    CHECK_LAUNCH("pool_keys");
    }

    // Kernel B: encode queries (one warp per (b,s,h), dims-contiguous loads)
    {
        int total = B * S * Hq * (D / 4);
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        indexer_encode_q_kernel<<<blocks, threads, 0, stream>>>(
            q.data_ptr<float>(),
            cos_q.data_ptr<float>(), sin_q.data_ptr<float>(),
            qenc.data_ptr<float>(),
            B, S, Hq, D, R);
    CHECK_LAUNCH("encode_q");
    }

    // Kernel E: fused score + TopK (no dense score matrix)
    {
        int P_pad = 1;
        while (P_pad < NB) P_pad <<= 1;
        // Must match the kernel's smem layout exactly: packed uint64 buffer
        // (16B-rounded), q-staging block, then the block-key tile with its
        // bank-conflict-avoiding D4+1 float4 row stride.
        const int KD4 = D / 4 + 1;
        size_t smem_bytes = (((size_t)P_pad * sizeof(uint64_t) + 15) & ~(size_t)15);
        smem_bytes = ((smem_bytes + (size_t)Hq * D * (int)sizeof(float) + 15) & ~(size_t)15);
        smem_bytes += (size_t)IDXER_BLOCK_N * KD4 * (int)sizeof(float4);
        // With a 64-block key tile the per-block dynamic smem exceeds the 48KB
        // default on SM80; opt in to the larger carve-out (A800 supports up to
        // 164KB/block) when needed.
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(indexer_fused_score_topk_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem_bytes);
        }
        indexer_fused_score_topk_kernel<<<B * S, 256, smem_bytes, stream>>>(
            reinterpret_cast<const float4*>(qenc.data_ptr<float>()),
            reinterpret_cast<const float4*>(kbar.data_ptr<float>()),
            block_indices.data_ptr<int>(),
            selected_scores.data_ptr<float>(),
            B, S, Hq, D, (int)r, NB, (int)block_topk, P_pad);
    CHECK_LAUNCH("fused_score_topk");
    }

    return {block_indices, selected_scores};
}
