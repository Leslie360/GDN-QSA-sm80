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
// One thread per (batch, block).  Pool = mean over r tokens, RMSNorm over D,
// partial RoPE (first R dims) at block-start position p_b = blk * r.
// ---------------------------------------------------------------------------
__global__ void indexer_pool_keys_kernel(
    const float* __restrict__ raw_keys,   // [B,S,D]
    const float* __restrict__ cos_k,      // [B,S,R]
    const float* __restrict__ sin_k,      // [B,S,R]
    float* __restrict__ kbar,             // [B,NB,D]
    int B, int S, int D, int R, int r, int NB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NB;
    if (idx >= total) return;

    int blk = idx % NB;
    int b   = idx / NB;
    int p_b = blk * r;

    const float* kb = raw_keys + (size_t)b * S * D;

    // AvgPool over r tokens -> pooled block key
    float local[128];   // D <= 128
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int u = 0; u < r; ++u) acc += kb[(size_t)(p_b + u) * D + d];
        local[d] = acc * (1.0f / (float)r);
    }

    // RMSNorm over D
    float mean_sq = 0.0f;
    for (int d = 0; d < D; ++d) mean_sq += local[d] * local[d];
    mean_sq /= (float)D;
    float rms_inv = rsqrtf_(mean_sq + 1e-6f);
    for (int d = 0; d < D; ++d) local[d] *= rms_inv;

    // partial RoPE at block start p_b
    const float* cbp = cos_k + (size_t)b * S * R + (size_t)p_b * R;
    const float* sbp = sin_k + (size_t)b * S * R + (size_t)p_b * R;
    for (int d = 0; d < R / 2; ++d) {
        float x0 = local[d];
        float x1 = local[d + R / 2];
        float co = cbp[d];
        float si = sbp[d];
        local[d]         = x0 * co - x1 * si;
        local[d + R / 2] = x0 * si + x1 * co;
    }

    float* out = kbar + (size_t)b * NB * D + (size_t)blk * D;
    for (int d = 0; d < D; ++d) out[d] = local[d];
}

// ---------------------------------------------------------------------------
// Kernel B: encode each query: RMSNorm over D + partial RoPE at token position.
//   qenc[B, S, Hq, D]
// One thread per (batch, query, head).
// ---------------------------------------------------------------------------
__global__ void indexer_encode_q_kernel(
    const float* __restrict__ q,          // [B,S,Hq,D]
    const float* __restrict__ cos_q,      // [B,S,R]
    const float* __restrict__ sin_q,      // [B,S,R]
    float* __restrict__ qenc,             // [B,S,Hq,D]
    int B, int S, int Hq, int D, int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * Hq;
    if (idx >= total) return;

    int h = idx % Hq;
    int t = (idx / Hq) % S;
    int b = idx / (Hq * S);

    const float* qh = q + ((size_t)b * S + t) * Hq * D + (size_t)h * D;

    // RMSNorm
    float mean_sq = 0.0f;
    for (int d = 0; d < D; ++d) mean_sq += qh[d] * qh[d];
    mean_sq /= (float)D;
    float rms_inv = rsqrtf_(mean_sq + 1e-6f);

    float* qe = qenc + ((size_t)b * S + t) * Hq * D + (size_t)h * D;
    for (int d = 0; d < D; ++d) qe[d] = qh[d] * rms_inv;

    // partial RoPE at token position t
    const float* ctp = cos_q + (size_t)b * S * R + (size_t)t * R;
    const float* stp = sin_q + (size_t)b * S * R + (size_t)t * R;
    for (int d = 0; d < R / 2; ++d) {
        float x0 = qe[d];
        float x1 = qe[d + R / 2];
        float co = ctp[d];
        float si = stp[d];
        qe[d]         = x0 * co - x1 * si;
        qe[d + R / 2] = x0 * si + x1 * co;
    }
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
    float*  sh_kf = reinterpret_cast<float*>(smem_raw + (size_t)SCORE_CQ * HqD4 * sizeof(float4)); // SCORE_CB * (D+1) floats

    int qtile = blockIdx.x;
    int btile = blockIdx.y;
    int b     = blockIdx.z;
    int q0 = qtile * SCORE_CQ;
    int b0 = btile * SCORE_CB;

    // Stage the query tile [q0, q0+CQ) x [Hq, D] (float4).
    {
        const float4* src = qenc + ((size_t)b * S + q0) * HqD4;
        for (int i = threadIdx.x; i < SCORE_CQ * HqD4; i += blockDim.x) {
            int lq = i / HqD4;
            if (q0 + lq < S) sh_q[i] = src[i];
        }
    }
    // Stage the block-key tile [b0, b0+CB) x [D] as padded floats (row stride D+1).
    {
        const float4* src = kbar + ((size_t)b * NB + b0) * D4;
        for (int i = threadIdx.x; i < SCORE_CB * D4; i += blockDim.x) {
            int lb = i / D4;
            if (b0 + lb < NB) {
                float4 v = src[i];
                float* row = sh_kf + (size_t)lb * (D + 1) + (size_t)(i % D4) * 4;
                row[0] = v.x; row[1] = v.y; row[2] = v.z; row[3] = v.w;
            }
        }
    }
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
        const float* kk = sh_kf + (size_t)c * (D + 1);
        float acc = 0.0f;
        for (int h = 0; h < Hq; ++h) {
            const float4* qh = sh_q + (size_t)(rq * Hq + h) * D4;
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D4; ++d) {
                float4 a = qh[d];
                dot += a.x * kk[d * 4 + 0] + a.y * kk[d * 4 + 1]
                     + a.z * kk[d * 4 + 2] + a.w * kk[d * 4 + 3];
            }
            if (dot > 0.0f) acc += dot;   // ReLU
        }
        score = acc * rsqrtf_((float)D);
    }
    if (inb) block_scores[(size_t)b * S * NB + (size_t)q * NB + blk] = score;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Kernel D: per-query TopK via a one-round RADIX select + bitonic sort of the
// ~K winners (replaces a full O(P log^2 P) bitonic sort of all P2 candidates).
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
//   3. collect >b1 -> s_out[0..cnt_gt) (unordered) and ==b1 -> s_out[cnt_gt..)
//      (shared-memory atomic counters); total written = cnt_gt + hist[b1] >= K
//   4. bitonic sort s_out[0..tot) descending (tot ~= K for uniform data, never
//      more than P2) and take the head K_eff.  This is the exact order.
// ---------------------------------------------------------------------------
__global__ void indexer_topk_kernel(
    const float* __restrict__ block_scores,   // [B,S,NB]
    int* __restrict__ block_indices,          // [B,S,KB]
    float* __restrict__ selected_scores,      // [B,S,KB]
    int B, int S, int NB, int KB, int P_pad)
{
    extern __shared__ char smem_raw[];
    uint64_t* s_pack = reinterpret_cast<uint64_t*>(smem_raw);   // P_pad
    uint64_t* s_out  = reinterpret_cast<uint64_t*>(smem_raw + (size_t)P_pad * sizeof(uint64_t)); // P_pad
    int* s_hist = reinterpret_cast<int*>(smem_raw + (size_t)2 * P_pad * sizeof(uint64_t));       // 256

    int bq = blockIdx.x;
    if (bq >= B * S) return;
    int q = bq % S;
    int b = bq / S;

    const float* scores = block_scores + (size_t)b * S * NB + (size_t)q * NB;

    // Compact valid (non -inf) candidates into shared memory, packed.
    __shared__ int scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < NB; i += blockDim.x) {
        float s = scores[i];
        if (s != -CUDART_INF_F) {
            int pos = atomicAdd(&scount, 1);
            uint32_t sb = __float_as_uint(s);
            uint32_t sortable = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            s_pack[pos] = ((uint64_t)sortable << 32)
                        | (uint64_t)(0xFFFFFFFFu - (uint32_t)i);
        }
    }
    __syncthreads();
    int P = scount;
    int K_eff = std::min(KB, P);

    if (K_eff > 0) {
        // ---- round 1: histogram top 8 bits of the valid keys ----
        for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < P; i += blockDim.x) {
            atomicAdd(&s_hist[(int)((s_pack[i] >> 56) & 0xFFu)], 1);
        }
        __syncthreads();

        // ---- find pivot: walk digits high->low, accumulate >b1 count ----
        __shared__ int sb1, scnt_gt, sneed;
        if (threadIdx.x == 0) {
            int cnt = 0, need = K_eff, b1 = 0;
            for (int b = 255; b >= 0; --b) {
                int c = s_hist[b];
                if (cnt + c >= K_eff) { b1 = b; need = K_eff - cnt; break; }
                cnt += c;
            }
            sb1 = b1; scnt_gt = cnt; sneed = need;
        }
        __syncthreads();

        // ---- collect: >b1 to head, ==b1 after it ----
        __shared__ int sgt_cnt, seq_cnt;
        if (threadIdx.x == 0) { sgt_cnt = 0; seq_cnt = 0; }
        __syncthreads();
        for (int i = threadIdx.x; i < P; i += blockDim.x) {
            uint64_t pk = s_pack[i];
            int d = (int)((pk >> 56) & 0xFFu);
            if (d > sb1) {
                s_out[atomicAdd(&sgt_cnt, 1)] = pk;
            } else if (d == sb1) {
                s_out[scnt_gt + atomicAdd(&seq_cnt, 1)] = pk;
            }
        }
        __syncthreads();
        int tot = scnt_gt + seq_cnt;   // >= K_eff by pivot construction

        // ---- bitonic sort (descending) of the ~K winners ----
        int P2s = 1;
        while (P2s < tot) P2s <<= 1;
        for (int i = tot + threadIdx.x; i < P2s; i += blockDim.x) {
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
    int NB = S / (int)r;
    TORCH_CHECK(raw_keys.size(0) == B && raw_keys.size(1) == S && raw_keys.size(2) == D);
    TORCH_CHECK(cos_q.size(0) == B && cos_q.size(1) == S);
    TORCH_CHECK(R % 2 == 0 && R <= D && R % 4 == 0);

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
        int total = B * NB;
        int threads = 128;
        int blocks = (total + threads - 1) / threads;
        indexer_pool_keys_kernel<<<blocks, threads, 0, stream>>>(
            raw_keys.data_ptr<float>(),
            cos_k.data_ptr<float>(), sin_k.data_ptr<float>(),
            kbar.data_ptr<float>(),
            B, S, D, R, (int)r, NB);
    CHECK_LAUNCH("pool_keys");
    }

    // Kernel B: encode queries
    {
        int total = B * S * Hq;
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
        int smem_bytes = SCORE_CQ * HqD4 * (int)sizeof(float4)
                       + SCORE_CB * (D + 1) * (int)sizeof(float);
        TORCH_CHECK(smem_bytes <= 48 * 1024, "score kernel shared memory too large");
        dim3 grid((S + SCORE_CQ - 1) / SCORE_CQ,
                  (NB + SCORE_CB - 1) / SCORE_CB,
                  B);
        indexer_score_kernel<<<grid, SCORE_CQ * SCORE_CB, smem_bytes, stream>>>(
            reinterpret_cast<const float4*>(qenc.data_ptr<float>()),
            reinterpret_cast<const float4*>(kbar.data_ptr<float>()),
            block_scores.data_ptr<float>(),
            B, S, Hq, D, (int)r, NB);
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
    int NB = S / (int)r;
    TORCH_CHECK(raw_keys.size(0) == B && raw_keys.size(1) == S && raw_keys.size(2) == D);
    TORCH_CHECK(cos_q.size(0) == B && cos_q.size(1) == S);
    TORCH_CHECK(R % 2 == 0 && R <= D && R % 4 == 0);

    auto opts = q.options();
    auto block_indices = torch::empty({B, S, (int)block_topk}, at::TensorOptions().dtype(at::kInt).device(q.device()));
    auto selected_scores = torch::empty({B, S, (int)block_topk}, opts);

    auto kbar = torch::empty({B, NB, D}, opts);
    auto qenc = torch::empty({B, S, Hq, D}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Kernel A: pool + norm + rope block keys
    {
        int total = B * NB;
        int threads = 128;
        int blocks = (total + threads - 1) / threads;
        indexer_pool_keys_kernel<<<blocks, threads, 0, stream>>>(
            raw_keys.data_ptr<float>(),
            cos_k.data_ptr<float>(), sin_k.data_ptr<float>(),
            kbar.data_ptr<float>(),
            B, S, D, R, (int)r, NB);
    CHECK_LAUNCH("pool_keys");
    }

    // Kernel B: encode queries
    {
        int total = B * S * Hq;
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
