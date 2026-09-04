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
// Kernel D: per-query TopK via a shared-memory bitonic merge sort (ascending),
// then take the largest KB.  Works for ANY n_blocks (no fixed 512-slot buffer).
// Grid: one thread block per (batch, query).
// Dynamic shared memory: P_pad * (sizeof(float) + sizeof(int)) bytes, where
// P_pad is the next power of two >= NB.
// ---------------------------------------------------------------------------
__global__ void indexer_topk_kernel(
    const float* __restrict__ block_scores,   // [B,S,NB]
    int* __restrict__ block_indices,          // [B,S,KB]
    float* __restrict__ selected_scores,      // [B,S,KB]
    int B, int S, int NB, int KB, int P_pad)
{
    extern __shared__ char smem_raw[];
    float* s_score = reinterpret_cast<float*>(smem_raw);
    int*   s_idx   = reinterpret_cast<int*>(smem_raw + (size_t)P_pad * sizeof(float));

    int bq = blockIdx.x;
    if (bq >= B * S) return;
    int q = bq % S;
    int b = bq / S;

    const float* scores = block_scores + (size_t)b * S * NB + (size_t)q * NB;

    // Compact valid (non -inf) candidates into shared memory.
    __shared__ int scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < NB; i += blockDim.x) {
        float s = scores[i];
        if (s != -CUDART_INF_F) {
            int pos = atomicAdd(&scount, 1);
            s_score[pos] = s;
            s_idx[pos] = i;
        }
    }
    __syncthreads();
    int P = scount;

    // Pad the remaining P_pad - P slots with -inf (they sort to the ascending
    // front, i.e. are never selected as top-K).
    for (int i = P + threadIdx.x; i < P_pad; i += blockDim.x) {
        s_score[i] = -CUDART_INF_F;
        s_idx[i] = -1;
    }
    __syncthreads();

    // Sort only the P2 = next_pow2(P) window that actually holds valid entries.
    // P (visible blocks for this query) is usually far below NB, so sorting the
    // full P_pad is wasted work: e.g. query q sees ~q/r blocks, average ~NB/2.
    // The remaining [P, P2) slots are -inf and sort to the front (never selected).
    int P2 = 1;
    while (P2 < P) P2 <<= 1;

    // Bitonic merge sort (ascending) over the P2 window.  To reproduce the
    // reference's TopK tie breaking (lowest block index among equal scores wins),
    // equal scores are ordered so that the lower index is treated as "greater"
    // and sorts toward the selected (high) end.
    for (int k = 2; k <= P2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < P2; i += blockDim.x) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & k) == 0);
                    float si = s_score[i];
                    float sl = s_score[l];
                    int   ii = s_idx[i];
                    int   il = s_idx[l];
                    // i "greater than" l: higher score, or equal score + lower index.
                    bool i_gt_l = (si > sl) || (si == sl && ii < il);
                    bool l_gt_i = (sl > si) || (sl == si && il < ii);
                    if ((up && i_gt_l) || (!up && l_gt_i)) {
                        s_score[i] = sl;
                        s_score[l] = si;
                        s_idx[i] = il;
                        s_idx[l] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    int k = std::min(KB, P);
    int*   inds = block_indices   + (size_t)bq * KB;
    float* sels = selected_scores + (size_t)bq * KB;
    for (int j = threadIdx.x; j < KB; j += blockDim.x) {
        if (j < k) {
            inds[j] = s_idx[P2 - 1 - j];
            sels[j] = s_score[P2 - 1 - j];
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
        int smem_bytes = P_pad * (sizeof(float) + sizeof(int));
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
