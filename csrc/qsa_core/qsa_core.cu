// QSA sparse core attention (SM80).
//
// Implements the sparse core attention of sparse-block attention:
//   Given per-query selected *block* indices, expand each block into its
//   constituent token positions, always add the tail tokens of the current
//   (incomplete) block that holds the query, then run a causal softmax
//   attention over ONLY those selected key/value tokens (flash-attn style
//   online softmax, but over a sparse gather set instead of the full seq).
//
// Inputs (all already RoPE-applied on the 64 rotary dims, bf16):
//   q        [B, S, H,   D]   query states      (H = num_attention_heads)
//   k        [B, S, KVH, D]   key states        (KVH = num_key_value_heads)
//   v        [B, S, KVH, D]   value states
//   block_idx[B, S, KB]       int32 selected block indices (indexer output),
//                             -1 means "no selection" (padded slot).
//   block_size = r            tokens per compressed block
// Output:
//   out      [B, S, H, D]     sparse attended output (before the model-side
//                             sigmoid output gate, which lives outside this op).
//
// GQA: H query heads share KVH KV heads (group = H/KVH).
// scale = head_dim^-0.5.
//
// Implementation:
//   Pass 1 (qsa_expand_kernel): block_idx -> expanded selected token list
//     sel_idx[B,S,NMAX] (+ counts sel_cnt[B,S]); NMAX = KB*r + r.
//     S_i = Expand(B_i) U { tail of current incomplete block }.
//   Pass 2 (qsa_sparse_attn_kernel): one warp per (b,s,h); online-softmax
//     sparse attention over sel_idx. head_dim D = 32 * DB dims per lane.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace qsa_core {

// ---------------------------------------------------------------------------
// dtype conversion helpers
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// ---------------------------------------------------------------------------
// Pass 1: expand selected block indices into token positions.
//   One WARP per (b,s) (the old one-thread-per-(b,s) version ran the whole GPU
//   with only B*S threads -> badly latency bound). Lanes sweep the KB selected
//   blocks in groups of 32; a ballot prefix-sum gives each valid block its
//   output ordinal so -1 padded slots are skipped correctly. When r%4==0 each
//   block's r tokens are emitted as coalesced int4 stores.
// ---------------------------------------------------------------------------
__global__ void qsa_expand_kernel(
    const int* __restrict__ block_idx,  // [B, S, KB]
    int* __restrict__ sel_idx,          // [B, S, NMAX]
    int* __restrict__ sel_cnt,          // [B, S]
    int B, int S, int KB, int r, int NMAX) {
    int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    long long total = (long long)B * S;
    if (wid >= total) return;
    long long bs = wid;
    int s = (int)(bs % S);

    const int* bi = block_idx + bs * KB;
    int* si = sel_idx + bs * NMAX;

    const bool vec4 = (r % 4) == 0;   // implies NMAX = r*(KB+1) % 4 == 0
    int base = 0;                     // ordinals of valid blocks so far
    for (int kk0 = 0; kk0 < KB; kk0 += 32) {
        int kk = kk0 + lane;
        int bk = (kk < KB) ? bi[kk] : -1;
        bool valid = (bk >= 0);
        unsigned mask = __ballot_sync(0xffffffffu, valid);
        int pref = __popc(mask & ((1u << lane) - 1));
        if (valid) {
            int ordinal = base + pref;
            int start = bk * r;
            int* dst = si + ordinal * r;
            if (vec4) {
                for (int t4 = 0; t4 < r / 4; t4++) {
                    int p = t4 * 4;
                    if (ordinal * r + p + 3 < NMAX) {
                        int4 v;
                        v.x = start + p; v.y = v.x + 1; v.z = v.x + 2; v.w = v.x + 3;
                        *((int4*)(dst + p)) = v;
                    }
                }
            } else {
                for (int t = 0; t < r; t++) {
                    if (ordinal * r + t < NMAX) dst[t] = start + t;
                }
            }
        }
        base += __popc(mask);
    }
    const int V = base;                       // number of valid blocks
    const int tail_start = (s / r) * r;       // first token of block holding s
    int n = V * r + (s - tail_start + 1);
    if (n > NMAX) n = NMAX;
    // tail tokens occupy [V*r, n)
    for (int p = V * r + lane; p < n; p += 32) si[p] = tail_start + (p - V * r);
    // pad the rest with -1
    for (int p = n + lane; p < NMAX; p += 32) si[p] = -1;
    if (lane == 0) sel_cnt[bs] = n;
}

// ---------------------------------------------------------------------------
// Pass 2: sparse paged softmax attention. One warp per (b,s,h).
// head_dim D = 32 * DB.
// ---------------------------------------------------------------------------
template <typename T, int DB>
__global__ void qsa_sparse_attn_kernel(
    const T* __restrict__ q,       // [B, S, H,   D]
    const T* __restrict__ k,       // [B, S, KVH, D]
    const T* __restrict__ v,       // [B, S, KVH, D]
    const int* __restrict__ sel_idx,  // [B, S, NMAX]
    const int* __restrict__ sel_cnt,  // [B, S]
    T* __restrict__ out,           // [B, S, H, D]
    int B, int S, int H, int KVH, int NMAX, float scale) {
    int warp_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    long long total_warps = (long long)B * S * H;
    if (warp_id >= total_warps) return;

    int h = warp_id % H;
    int tmp = warp_id / H;
    int s = tmp % S;
    int b = tmp / S;
    int g = h / (H / KVH);   // KV head group for this query head

    const int n = sel_cnt[(long long)b * S + s];
    const int* sidx = sel_idx + ((long long)b * S + s) * NMAX;

    // q for this thread's DB dims: q[b, s, h, lane*DB + d]
    const T* qptr = q + ((long long)(b * S + s) * H + h) * (32 * DB);
    // k/v base for this kv-head: k[b, :, g, :]
    const T* kbase = k + ((long long)b * S * KVH + g) * (32 * DB);
    const T* vbase = v + ((long long)b * S * KVH + g) * (32 * DB);

    float qv[DB];
#pragma unroll
    for (int d = 0; d < DB; d++) qv[d] = to_float(qptr[lane * DB + d]);

    float outv[DB];
#pragma unroll
    for (int d = 0; d < DB; d++) outv[d] = 0.f;

    float maxv = -1e30f;
    float sumexp = 0.f;

    for (int m = 0; m < n; m++) {
        int j = sidx[m];
        if (j < 0) break;
        if (j > s) continue;  // causal mask: only tokens <= s are visible

        // k/v layout is [B,S,KVH,D]: consecutive tokens are KVH*D apart.
        const T* kptr = kbase + (long long)j * KVH * (32 * DB);
        float pdot = 0.f;
#pragma unroll
        for (int d = 0; d < DB; d++) pdot += qv[d] * to_float(kptr[lane * DB + d]);
        // warp reduce the partial dot -> full score, broadcast
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) pdot += __shfl_xor_sync(0xffffffffu, pdot, off);
        float sc = scale * pdot;

        // online softmax update
        float nsc = fmaxf(sc, maxv);
        float p = __expf(sc - nsc);
        float corr = __expf(maxv - nsc);
        sumexp = sumexp * corr + p;
        const T* vptr = vbase + (long long)j * KVH * (32 * DB);
#pragma unroll
        for (int d = 0; d < DB; d++) outv[d] = outv[d] * corr + p * to_float(vptr[lane * DB + d]);
        maxv = nsc;
    }

    float inv = 1.f / fmaxf(sumexp, 1e-30f);
#pragma unroll
    for (int d = 0; d < DB; d++) outv[d] *= inv;

    T* optr = out + ((long long)(b * S + s) * H + h) * (32 * DB);
#pragma unroll
    for (int d = 0; d < DB; d++) optr[lane * DB + d] = from_float<T>(outv[d]);
}

// Cooperative gather of one chunk's kv head g from global into smem buffer.
template <typename T, int DB, int CHUNK>
__device__ __forceinline__ void load_chunk_smem(
    T* buf_k, T* buf_v, int b, int s, int g, int S, int KVH,
    const int* __restrict__ sidx, int base, int cnt,
    const T* __restrict__ k, const T* __restrict__ v,
    int tid, int nthreads) {
    const int elems_per_chunk = CHUNK * (32 * DB);
    for (int e = tid; e < elems_per_chunk; e += nthreads) {
        int m = e / (32 * DB);
        int dim = e % (32 * DB);
        if (m < cnt) {
            int j = sidx[base + m];
            if (j >= 0) {
                long long koff = ((long long)b * S + j) * KVH * (32 * DB) + g * (32 * DB) + dim;
                buf_k[e] = k[koff];
                buf_v[e] = v[koff];
            } else {
                buf_k[e] = from_float<T>(0.f);
                buf_v[e] = from_float<T>(0.f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2 (v2): block-per-(b,s,kvgroup), optional split-K over the token slice.
//   Each block owns ONE kv head and the H/KVH query heads that share it (GQA).
//   KV is read from global exactly once per kv head (zero redundancy), staged
//   into smem per chunk (coalesced), reused by all heads in the block.
//   - Grid = B*S*KVH*SPLIT blocks (split-K gives more blocks -> better latency
//     hiding at small S where memory traffic is low and the op is compute bound).
//   - Each warp owns HPP query heads with persistent online-softmax
//     accumulators carried across chunks (single KV pass over its slice).
//   - When SPLIT > 1, each block writes a partial (max,sumexp,out) that a
//     follow-up kernel rescales and combines.
//   - HPP = heads per warp (template). nwarps = blockDim/32.
// ---------------------------------------------------------------------------
template <typename T, int DB, int HPP, int CHUNK, int SPLIT>
__global__ void qsa_sparse_attn_kernel_v2(
    const T* __restrict__ q,       // [B, S, H,   D]
    const T* __restrict__ k,       // [B, S, KVH, D]
    const T* __restrict__ v,       // [B, S, KVH, D]
    const int* __restrict__ sel_idx,  // [B, S, NMAX]
    const int* __restrict__ sel_cnt,  // [B, S]
    T* __restrict__ out,           // [B, S, H, D] (SPLIT==1) or partial scratch
    float* __restrict__ part_max,  // [B, S, H, SPLIT] (SPLIT>1)
    float* __restrict__ part_sum,  // [B, S, H, SPLIT]
    int B, int S, int H, int KVH, int NMAX, float scale) {
    // blockIdx.x = ((b*S + s)*KVH + g)*SPLIT + p
    const int p = blockIdx.x % SPLIT;
    const int gbs = blockIdx.x / SPLIT;
    const int g = gbs % KVH;
    const int bs = gbs / KVH;
    if (bs >= B * S) return;
    const int b = bs / S;
    const int s = bs % S;
    const int n = sel_cnt[bs];
    const int* sidx = sel_idx + (long long)bs * NMAX;

    const int H_block = H / KVH;          // query heads sharing kv head g

    // token slice for this split block
    const int slice_start = (n * p) / SPLIT;
    const int slice_end   = (n * (p + 1)) / SPLIT;
    const int n_slice = slice_end - slice_start;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int nwarps = nthreads >> 5;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // smem: [CHUNK][32*DB] for k and for v (single kv head)
    extern __shared__ char smem_raw[];
    T* sk = reinterpret_cast<T*>(smem_raw);
    T* sv = sk + CHUNK * (32 * DB);

    // per-warp persistent accumulators for its HPP query heads
    int   hh_idx[HPP];
    float qv[HPP][DB];
    float outv[HPP][DB];
    float maxv[HPP];
    float sumexp[HPP];
    bool  active[HPP];

#pragma unroll
    for (int w = 0; w < HPP; w++) {
        int hl = warp + w * nwarps;       // local head index in [0, H_block)
        active[w] = (hl < H_block);
        if (active[w]) {
            int hh = g * H_block + hl;    // global query head index
            hh_idx[w] = hh;
            const T* qptr = q + ((long long)bs * H + hh) * (32 * DB);
#pragma unroll
            for (int d = 0; d < DB; d++) qv[w][d] = to_float(qptr[lane * DB + d]);
#pragma unroll
            for (int d = 0; d < DB; d++) outv[w][d] = 0.f;
            maxv[w] = -1e30f;
            sumexp[w] = 0.f;
        }
    }

    const int nchunks = (n_slice + CHUNK - 1) / CHUNK;
    for (int c = 0; c < nchunks; c++) {
        const int base = slice_start + c * CHUNK;
        const int cnt = min(CHUNK, n_slice - c * CHUNK);

        // cooperative coalesced gather of this chunk's kv head g into smem
        load_chunk_smem<T, DB, CHUNK>(sk, sv, b, s, g, S, KVH, sidx, base, cnt,
                                      k, v, tid, nthreads);
        __syncthreads();

        // each warp processes its HPP heads over this chunk
#pragma unroll
        for (int w = 0; w < HPP; w++) {
            if (!active[w]) continue;
            for (int m = 0; m < cnt; m++) {
                int j = sidx[base + m];
                bool vis = (j >= 0) && (j <= s);
                const T* kptr = sk + m * (32 * DB);
                float pdot = 0.f;
#pragma unroll
                for (int d = 0; d < DB; d++) pdot += qv[w][d] * to_float(kptr[lane * DB + d]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    pdot += __shfl_xor_sync(0xffffffffu, pdot, off);
                float sc = scale * pdot;
                if (vis) {
                    float nsc = fmaxf(sc, maxv[w]);
                    float p = __expf(sc - nsc);
                    float corr = __expf(maxv[w] - nsc);
                    sumexp[w] = sumexp[w] * corr + p;
                    const T* vptr = sv + m * (32 * DB);
#pragma unroll
                    for (int d = 0; d < DB; d++)
                        outv[w][d] = outv[w][d] * corr + p * to_float(vptr[lane * DB + d]);
                    maxv[w] = nsc;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int w = 0; w < HPP; w++) {
        if (!active[w]) continue;
        int hh = hh_idx[w];
        if (SPLIT == 1) {
            float inv = 1.f / fmaxf(sumexp[w], 1e-30f);
            T* optr = out + ((long long)bs * H + hh) * (32 * DB);
#pragma unroll
            for (int d = 0; d < DB; d++) optr[lane * DB + d] = from_float<T>(outv[w][d] * inv);
        } else {
            // partial_out[b,s,h,*,p], part_max/sum[b,s,h,p]
            long long base = ((long long)bs * H + hh) * SPLIT;
            part_max[base + p] = maxv[w];
            part_sum[base + p] = sumexp[w];
            T* poptr = reinterpret_cast<T*>(out) + ((long long)bs * H + hh) * (32 * DB) * SPLIT;
#pragma unroll
            for (int d = 0; d < DB; d++) poptr[(lane * DB + d) * SPLIT + p] = from_float<T>(outv[w][d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Combine kernel for SPLIT>1: rescale partials across the SPLIT blocks.
//   out[b,s,h,d] = sum_p( out_p[h,d] * exp(max_p[h] - M) ) / sum_p( sumexp_p[h] * exp(max_p[h] - M) )
//   where M = max over p of max_p[h].
//   Layout: partial_out[B,S,H,D,SPLIT], part_max/sum[B,S,H,SPLIT].
//   Grid: one warp per (b,s,h); all 32 lanes cover the D=32*DB dims.
// ---------------------------------------------------------------------------
template <typename T, int DB, int SPLIT>
__global__ void qsa_combine_kernel(
    const T* __restrict__ partial_out,  // [B,S,H,D,SPLIT] unnormalized per-split outs
    const float* __restrict__ part_max, // [B,S,H,SPLIT]
    const float* __restrict__ part_sum,
    T* __restrict__ out,               // [B,S,H,D]
    int B, int S, int H) {
    int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    long long total = (long long)B * S * H;
    if (wid >= total) return;
    int h = wid % H;
    long long bs = wid / H;
    long long hb = bs * H + h;

    float M = -1e30f;
#pragma unroll
    for (int p = 0; p < SPLIT; p++) M = fmaxf(M, part_max[hb * SPLIT + p]);
    float denom = 0.f;
#pragma unroll
    for (int p = 0; p < SPLIT; p++) denom += part_sum[hb * SPLIT + p] * __expf(part_max[hb * SPLIT + p] - M);
    const float inv = 1.f / fmaxf(denom, 1e-30f);

    const T* psrc = partial_out + hb * (32 * DB) * SPLIT;
    T* optr = out + hb * (32 * DB);
#pragma unroll
    for (int d = 0; d < DB; d++) {
        int dim = lane * DB + d;
        float acc = 0.f;
#pragma unroll
        for (int p = 0; p < SPLIT; p++) {
            float w = __expf(part_max[hb * SPLIT + p] - M) * inv;
            acc += w * to_float(psrc[dim * SPLIT + p]);
        }
        optr[dim] = from_float<T>(acc);
    }
}

// ---------------------------------------------------------------------------
// Host launcher (full operator: expand + sparse attention)
// ---------------------------------------------------------------------------
template <typename T>
void launch_qsa_core(const T* q, const T* k, const T* v, const int* block_idx,
                     T* out, int* sel_idx_buf, int* sel_cnt_buf,
                     int B, int S, int H, int KVH, int KB, int r, int D,
                     cudaStream_t stream) {
    const int NMAX = KB * r + r;

    // Pass 1: expand (warp per (b,s))
    {
        const int warps_per_block = 8;
        int nthreads = warps_per_block * 32;
        long long total_warps = (long long)B * S;
        int nblocks = (int)((total_warps + warps_per_block - 1) / warps_per_block);
        qsa_expand_kernel<<<nblocks, nthreads, 0, stream>>>(
            block_idx, sel_idx_buf, sel_cnt_buf, B, S, KB, r, NMAX);
    }

    // Pass 2: sparse attention (block per (b,s,kvgroup,split), KV staged in smem)
    {
        const int nwarps = 8;   // 256 threads / 32
        const int nthreads = nwarps * 32;
        const int H_block = H / KVH;
        // split-K over the token slice: boost block count so small-S (B=1) still
        // fills the GPU. Target ~>= 4-6 blocks per SM worth of grid parallelism.
        const int SM_COUNT = 108;
        long long base_blocks = (long long)B * S * KVH;
        int SPLIT = 1;
        if (base_blocks > 0) {
            long long target = (long long)SM_COUNT * 3;
            while ((base_blocks * SPLIT) < target && SPLIT < 4) SPLIT *= 2;
        }
        int nblocks = (int)(base_blocks * SPLIT);
        float scale = 1.0f / sqrtf((float)D);
        const int HPP = (H_block + nwarps - 1) / nwarps;

        // CHUNK chosen so single-buffered smem stays modest for good occupancy
        int CHUNK = 32;
        size_t smem_bytes = (size_t)CHUNK * (32 * (D / 32)) * 2 * sizeof(T);  // k + v
        while (smem_bytes > 40 * 1024 && CHUNK > 4) {
            CHUNK /= 2;
            smem_bytes = (size_t)CHUNK * (32 * (D / 32)) * 2 * sizeof(T);
        }
        const int smemB = (int)smem_bytes;

        // scratch for split-K partials (only when SPLIT>1)
        T* partial_out = out;  // when SPLIT==1, v2 writes final directly to out
        float* part_max = nullptr;
        float* part_sum = nullptr;
        if (SPLIT > 1) {
            cudaMalloc(&partial_out, (size_t)B * S * H * D * SPLIT * sizeof(T));
            cudaMalloc(&part_max, (size_t)B * S * H * SPLIT * sizeof(float));
            cudaMalloc(&part_sum, (size_t)B * S * H * SPLIT * sizeof(float));
        }

        // helper that instantiates the v2 kernel for a concrete (DB,HPP,CHUNK,SPLIT)
        #define LAUNCH_V2(db, hpp, chunk, split) do { \
            auto kern = qsa_sparse_attn_kernel_v2<T, db, hpp, chunk, split>; \
            static bool cfg_##db##_##hpp##_##chunk##_##split = false; \
            if (!cfg_##db##_##hpp##_##chunk##_##split) { \
                cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemB); \
                cfg_##db##_##hpp##_##chunk##_##split = true; \
            } \
            kern<<<nblocks, nthreads, smemB, stream>>>( \
                q, k, v, sel_idx_buf, sel_cnt_buf, partial_out, part_max, part_sum, \
                B, S, H, KVH, NMAX, scale); \
        } while (0)

        #define DISPATCH_SPLIT(db, chunk, split) \
            if (HPP == 1) LAUNCH_V2(db, 1, chunk, split); \
            else if (HPP == 2) LAUNCH_V2(db, 2, chunk, split); \
            else if (HPP == 3) LAUNCH_V2(db, 3, chunk, split); \
            else LAUNCH_V2(db, 4, chunk, split)

        #define DISPATCH_DB(db) \
            if (SPLIT == 1) { \
                if (CHUNK == 8) DISPATCH_SPLIT(db, 8, 1); \
                else if (CHUNK == 16) DISPATCH_SPLIT(db, 16, 1); \
                else if (CHUNK == 64) DISPATCH_SPLIT(db, 64, 1); \
                else DISPATCH_SPLIT(db, 32, 1); \
            } else if (SPLIT == 2) { \
                if (CHUNK == 8) DISPATCH_SPLIT(db, 8, 2); \
                else if (CHUNK == 16) DISPATCH_SPLIT(db, 16, 2); \
                else if (CHUNK == 64) DISPATCH_SPLIT(db, 64, 2); \
                else DISPATCH_SPLIT(db, 32, 2); \
            } else if (SPLIT == 4) { \
                if (CHUNK == 8) DISPATCH_SPLIT(db, 8, 4); \
                else if (CHUNK == 16) DISPATCH_SPLIT(db, 16, 4); \
                else if (CHUNK == 64) DISPATCH_SPLIT(db, 64, 4); \
                else DISPATCH_SPLIT(db, 32, 4); \
            } else { \
                if (CHUNK == 8) DISPATCH_SPLIT(db, 8, 8); \
                else if (CHUNK == 16) DISPATCH_SPLIT(db, 16, 8); \
                else if (CHUNK == 64) DISPATCH_SPLIT(db, 64, 8); \
                else DISPATCH_SPLIT(db, 32, 8); \
            }

        const int DB = D / 32;
        if (DB == 8) DISPATCH_DB(8)
        else if (DB == 4) DISPATCH_DB(4)
        else if (DB == 2) DISPATCH_DB(2)
        else if (DB == 1) DISPATCH_DB(1)
        else DISPATCH_DB(16)

        #undef DISPATCH_SPLIT
        #undef DISPATCH_DB
        #undef LAUNCH_V2

        // combine partials across splits
        if (SPLIT > 1) {
            long long total_w = (long long)B * S * H;
            int wblocks = (int)((total_w + 7) / 8);  // 8 warps/block
            #define LAUNCH_COMBINE(db, split) qsa_combine_kernel<T, db, split><<<wblocks, 256, 0, stream>>>( \
                partial_out, part_max, part_sum, out, B, S, H)
            if (DB == 8) {
                if (SPLIT == 2) LAUNCH_COMBINE(8, 2);
                else if (SPLIT == 4) LAUNCH_COMBINE(8, 4);
                else LAUNCH_COMBINE(8, 8);
            } else if (DB == 4) {
                if (SPLIT == 2) LAUNCH_COMBINE(4, 2);
                else if (SPLIT == 4) LAUNCH_COMBINE(4, 4);
                else LAUNCH_COMBINE(4, 8);
            } else if (DB == 2) {
                if (SPLIT == 2) LAUNCH_COMBINE(2, 2);
                else if (SPLIT == 4) LAUNCH_COMBINE(2, 4);
                else LAUNCH_COMBINE(2, 8);
            } else if (DB == 1) {
                if (SPLIT == 2) LAUNCH_COMBINE(1, 2);
                else if (SPLIT == 4) LAUNCH_COMBINE(1, 4);
                else LAUNCH_COMBINE(1, 8);
            } else {
                if (SPLIT == 2) LAUNCH_COMBINE(16, 2);
                else if (SPLIT == 4) LAUNCH_COMBINE(16, 4);
                else LAUNCH_COMBINE(16, 8);
            }
            #undef LAUNCH_COMBINE
            cudaFree(partial_out);
            cudaFree(part_max);
            cudaFree(part_sum);
        }
    }
}

// Run pass 1 (expand) only, returning sel_idx and sel_cnt.
void launch_expand(const int* block_idx, int* sel_idx, int* sel_cnt,
                         int B, int S, int KB, int r, cudaStream_t stream) {
    const int NMAX = KB * r + r;
    const int warps_per_block = 8;
    int nthreads = warps_per_block * 32;
    long long total_warps = (long long)B * S;
    int nblocks = (int)((total_warps + warps_per_block - 1) / warps_per_block);
    qsa_expand_kernel<<<nblocks, nthreads, 0, stream>>>(
        block_idx, sel_idx, sel_cnt, B, S, KB, r, NMAX);
}

// explicit instantiations
template void launch_qsa_core<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const int*,
    __nv_bfloat16*, int*, int*, int, int, int, int, int, int, int, cudaStream_t);
template void launch_qsa_core<__half>(
    const __half*, const __half*, const __half*, const int*,
    __half*, int*, int*, int, int, int, int, int, int, int, cudaStream_t);
template void launch_qsa_core<float>(
    const float*, const float*, const float*, const int*,
    float*, int*, int*, int, int, int, int, int, int, int, cudaStream_t);

}  // namespace qsa_core
