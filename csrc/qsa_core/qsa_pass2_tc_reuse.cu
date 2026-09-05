// qsa_pass2_tc_reuse.cu — query-tile local K/V reuse for qsa_pass2_tc.
//
// Phase-2 (GPT decision B2): pass2 is query-stationary — each CTA gathers its
// own ~KB*r selected K/V rows from global, and adjacent queries re-gather the
// same blocks (block reuse median ~1400+ queries; ~90% of the gather traffic
// is L2-resident, see tools/analyze_qsa_core_reuse.py).  This kernel groups
// M_TILE=4 adjacent queries into one CTA, builds the UNION of their selected
// token sets in shared memory, then streams the union in 64-token tiles:
//   per tile:  gather K/V ONCE, share it across all M_TILE queries.
// Each query keeps its own online-softmax state (sm_m/sm_l/O_acc in smem) and
// its own validity mask (token in own select set AND causal tok <= s).
//
// Expected: gather L2 traffic ~ x4 lower => S=8192 ~25.9ms -> ~20ms.
// Correctness: relL1 < 1e-2 vs qsa_sparse_core_attention (validated by the
// torch prototype tools/proto_qsa_reuse.py in recent/random/smooth modes).
//
// Dispatch (host): reuse kernel only when S <= 8192 and S % M_TILE == 0 —
// larger S overflows the S-byte qmap shared-memory budget and falls back to
// the per-query v3 kernel (identical output, both paths tested).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace qsa_pass2_tc_v3 {
namespace qsa_pass2_tc_reuse {

__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    uint16_t l = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t h = *reinterpret_cast<uint16_t*>(&hi);
    return (uint32_t(h) << 16) | uint32_t(l);
}
__device__ __forceinline__ void mma16n8k16(float (&d)[4], uint32_t const (&a)[4], uint32_t const (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint4 ldg8(const __nv_bfloat16* __restrict__ src, int e8) {
    return *reinterpret_cast<const uint4*>(src + e8);
}

constexpr int M16 = 16;
constexpr int D = 256;
constexpr int N_TILE = 64;      // union token rows per streaming tile
constexpr int kPitch = D + 8;   // 264 bf16/row: kill 8-way bank conflicts
constexpr int nPitch = N_TILE + 4;  // 68
constexpr int K16 = 16;
constexpr int M_TILE = 4;       // adjacent queries per CTA (O in registers)
constexpr int U_MAX_CAP = M_TILE * 2052;  // conservative union cap (MT*NMAX)

// grid = (B*S/M_TILE) * KVH.  Host enforces S <= 8192, S % M_TILE == 0, and
// passes UMAX = min(U_MAX_CAP, S) (union tokens are int16: safe for S <= 32767).
__global__ void __launch_bounds__(256, 1) qsa_pass2_tc_reuse_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int*    __restrict__ sel_idx,
    const int*    __restrict__ sel_cnt,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int H, int KVH, int NMAX, float scale, int UMAX) {
    const int HBLK = H / KVH;   // 12 real query rows per head-group
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gi   = lane / 4;
    const int li   = lane % 4;

    const int cta  = blockIdx.x;
    const int gg   = cta % KVH;
    const int qg   = cta / KVH;
    const int bs_base = qg * M_TILE;      // first query (flattened b*S+s)
    if (bs_base >= B * S) return;
    const int b  = bs_base / S;
    const int s0 = bs_base % S;


    // ---- shared memory ----
    extern __shared__ char smem_raw[];
    char* sp = smem_raw;
    __nv_bfloat16* Qsm = reinterpret_cast<__nv_bfloat16*>(sp);           // [M_TILE][16][264]
    sp += M_TILE * M16 * kPitch * sizeof(__nv_bfloat16);
    __nv_bfloat16* Ksm = reinterpret_cast<__nv_bfloat16*>(sp);           // [64][264]
    sp += N_TILE * kPitch * sizeof(__nv_bfloat16);
    __nv_bfloat16* Vsm = reinterpret_cast<__nv_bfloat16*>(sp);           // [64][264]
    sp += N_TILE * kPitch * sizeof(__nv_bfloat16);
    float* Sc  = reinterpret_cast<float*>(sp);                            // [M_TILE][16][68] f32
    sp += M_TILE * M16 * nPitch * sizeof(float);
    __nv_bfloat16* P = reinterpret_cast<__nv_bfloat16*>(sp);             // [M_TILE][16][68]
    sp += M_TILE * M16 * nPitch * sizeof(__nv_bfloat16);
    float* part = reinterpret_cast<float*>(sp);                           // per-query reductions
    float* pmax  = part;                                                  // [M_TILE][16][8]
    float* plsum = part + M_TILE * M16 * 8;                               // [M_TILE][16][8]
    float* tmax  = part + M_TILE * M16 * 16;                              // [M_TILE][16]
    float* tlsum = tmax + M_TILE * M16;
    sp += (2 * M_TILE * M16 * 8 + 2 * M_TILE * M16) * sizeof(float);
    float* sm_m = reinterpret_cast<float*>(sp);                           // [M_TILE][16]
    sp += M_TILE * M16 * sizeof(float);
    float* sm_l = reinterpret_cast<float*>(sp);                           // [M_TILE][16]
    sp += M_TILE * M16 * sizeof(float);
    // NOTE: O accumulator lives in REGISTERS (O_r[2][8] per thread) — putting
    // it in smem requires a (warp, gi, li) key (gi=lane/4 repeats across the 8
    // warps), which is what the racecheck failures were about.
    unsigned int* bitmap = reinterpret_cast<unsigned int*>(sp);           // [S/32] int, atomicOr
    sp += ((S + 31) / 32) * sizeof(unsigned int);
    unsigned char* qmap = reinterpret_cast<unsigned char*>(sp);           // [S]
    sp += S * sizeof(unsigned char);
    int* u_cnt = reinterpret_cast<int*>(sp);                              // union length
    sp += sizeof(int);
    short* union_tok = reinterpret_cast<short*>(sp);                      // [UMAX] int16
    sp += (size_t)UMAX * sizeof(short);
    unsigned char* qmask = reinterpret_cast<unsigned char*>(sp);          // [UMAX]

    // ---- stage Q: M_TILE queries x 16 rows (12 real + 4 pad) ----
    for (int e = tid; e < M_TILE * M16 * D; e += 256) {
        int qq  = e / (M16 * D);
        int rr  = (e % (M16 * D)) / D;
        int d   = e % D;
        __nv_bfloat16 val = f2b(0.f);
        if (rr < HBLK) {
            int h = gg * HBLK + rr;
            val = q[(((long long)b * S + s0 + qq) * H + h) * D + d];
        }
        Qsm[qq * M16 * kPitch + rr * kPitch + d] = val;
    }
    for (int e = tid; e < (S + 31) / 32; e += 256) bitmap[e] = 0;
    for (int e = tid; e < S; e += 256) qmap[e] = 0;
    __syncthreads();

    // ---- build qmap (token -> owning query bitmask) + bitmap (token in union) ----
    for (int qq = 0; qq < M_TILE; qq++) {
        const int s = s0 + qq;
        const int* sidx = sel_idx + ((long long)b * S + s) * NMAX;
        const int n = sel_cnt[(long long)b * S + s];
        for (int e = tid; e < n; e += 256) {
            int tok = sidx[e];
            if (tok >= 0 && tok < S) {
                qmap[tok] |= (unsigned char)(1u << qq);
                atomicOr(&bitmap[tok >> 5], 1u << (tok & 31));   // non-atomic |= loses bits
            }
        }
        __syncthreads();
    }

    // ---- union token list (order irrelevant; streamed in 64-token tiles) ----
    if (tid == 0) *u_cnt = 0;
    __syncthreads();
    for (int tok = tid; tok < S; tok += 256) {
        if (bitmap[tok >> 5] & (1u << (tok & 31))) {
            int pos = atomicAdd(u_cnt, 1);
            if (pos < UMAX) { union_tok[pos] = (short)tok; qmask[pos] = qmap[tok]; }
        }
    }
    __syncthreads();
    const int U = *u_cnt;
    for (int e = tid + U; e < UMAX; e += 256) union_tok[e] = -1;
    __syncthreads();

    // ---- online-softmax state per query (O in registers, 16 f32 per query) ----
    for (int e = tid; e < M_TILE * M16; e += 256) { sm_m[e] = -1e30f; sm_l[e] = 0.f; }
    float O_r[M_TILE][2][8];
#pragma unroll
    for (int qq = 0; qq < M_TILE; qq++)
#pragma unroll
        for (int rh = 0; rh < 2; rh++)
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][rh][d] = 0.f;
    __syncthreads();
    const int n_utiles = (U + N_TILE - 1) / N_TILE;
    for (int t = 0; t < n_utiles; t++) {
        const int tbase = t * N_TILE;

        // ---- 1) vectorized gather K/V: union_tok[t*64 + row] -> Ksm/Vsm ----
        for (int it = 0; it < (N_TILE * D / 8) / 256; it++) {
            int vidx = it * 256 + tid;
            int m  = vidx >> 5;        // union-token row within the tile
            int cb = vidx & 31;        // col-block (d = cb*8 .. cb*8+8)
            int tok = union_tok[tbase + m];
            int e8 = m * kPitch + cb * 8;
            if (tok >= 0) {
                long long off = ((long long)b * S + tok) * KVH * D + gg * D + cb * 8;
                *reinterpret_cast<uint4*>(&Ksm[e8]) = ldg8(k, (int)off);
                *reinterpret_cast<uint4*>(&Vsm[e8]) = ldg8(v, (int)off);
            } else {
                *reinterpret_cast<uint4*>(&Ksm[e8]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&Vsm[e8]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- 2) QK: all queries share one barrier ----
        for (int qq = 0; qq < M_TILE; qq++) {
            if (warp < 8) {
                float s0v[4] = {0.f, 0.f, 0.f, 0.f};
                float s1v[4] = {0.f, 0.f, 0.f, 0.f};
                const int colw = warp * 8;
                const __nv_bfloat16* Qq = Qsm + qq * M16 * kPitch;
                float* Scq = Sc + qq * M16 * nPitch;
#pragma unroll
                for (int kb = 0; kb < D / K16; kb++) {
                    const int ko = kb * K16;
                    uint32_t a[4];
                    a[0] = pack2(Qq[gi * kPitch + ko + li],         Qq[gi * kPitch + ko + li + 8]);
                    a[1] = pack2(Qq[(gi + 8) * kPitch + ko + li],   Qq[(gi + 8) * kPitch + ko + li + 8]);
                    a[2] = pack2(Qq[gi * kPitch + ko + li + 4],     Qq[gi * kPitch + ko + li + 12]);
                    a[3] = pack2(Qq[(gi + 8) * kPitch + ko + li + 4], Qq[(gi + 8) * kPitch + ko + li + 12]);
                    uint32_t bb[2];
                    bb[0] = pack2(Ksm[(colw + gi) * kPitch + ko + li],     Ksm[(colw + gi) * kPitch + ko + li + 8]);
                    bb[1] = pack2(Ksm[(colw + gi) * kPitch + ko + li + 4], Ksm[(colw + gi) * kPitch + ko + li + 12]);
                    if (kb & 1) mma16n8k16(s1v, a, bb); else mma16n8k16(s0v, a, bb);
                }
                s0v[0] += s1v[0]; s0v[1] += s1v[1]; s0v[2] += s1v[2]; s0v[3] += s1v[3];
                s0v[0] *= scale; s0v[1] *= scale; s0v[2] *= scale; s0v[3] *= scale;
                Scq[gi * nPitch + colw + 2 * li]         = s0v[0];
                Scq[gi * nPitch + colw + 2 * li + 1]     = s0v[1];
                Scq[(gi + 8) * nPitch + colw + 2 * li]   = s0v[2];
                Scq[(gi + 8) * nPitch + colw + 2 * li + 1] = s0v[3];
            }
        }
        __syncthreads();

        // ---- 3) row-max + tmax: all queries, shared barriers ----
        for (int qq = 0; qq < M_TILE; qq++) {
            const int s_qq = s0 + qq;
            const float* Scq = Sc + qq * M16 * nPitch;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                int e = it * 32 + lane;
                int r = e / 8; int c = e % 8; int m = warp * 8 + c;
                int tok = union_tok[tbase + m];
                bool ok = (m < N_TILE) && (tok >= 0) && (tok <= s_qq) && (qmask[tbase + m] & (1u << qq));
                float val = ok ? Scq[r * nPitch + m] : -1e30f;
                val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 1));
                val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 2));
                val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 4));
                if (c == 0) pmax[qq * (M16 * 8) + r * 8 + warp] = val;
            }
        }
        __syncthreads();
        if ((tid & 15) == 0) {                       // one thread per row
            const int r = tid >> 4;
            for (int qq = 0; qq < M_TILE; qq++) {
                float mm = -1e30f;
                for (int x = 0; x < 8; x++) mm = fmaxf(mm, pmax[qq * (M16 * 8) + r * 8 + x]);
                tmax[qq * M16 + r] = mm;
            }
        }
        __syncthreads();

        // ---- 4) online rescale: all queries, registers ----
        for (int qq = 0; qq < M_TILE; qq++) {
            const int r0 = gi, r1 = gi + 8;
            const int rep0 = (r0 & 7) * 4, rep1 = (r1 & 7) * 4;   // warp-0 lane (r%8)*4 owns row r
            float mnew = fmaxf(sm_m[qq * M16 + r0], tmax[qq * M16 + r0]);
            float corr = __expf(sm_m[qq * M16 + r0] - mnew);
            if (tid == rep0) { sm_m[qq * M16 + r0] = mnew; sm_l[qq * M16 + r0] *= corr; }
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][0][d] *= corr;
            mnew = fmaxf(sm_m[qq * M16 + r1], tmax[qq * M16 + r1]);
            corr = __expf(sm_m[qq * M16 + r1] - mnew);
            if (tid == rep1) { sm_m[qq * M16 + r1] = mnew; sm_l[qq * M16 + r1] *= corr; }
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][1][d] *= corr;
        }
        __syncthreads();

        // ---- 5) P + plsum + sm_l: all queries, shared barriers ----
        for (int qq = 0; qq < M_TILE; qq++) {
            const int s_qq = s0 + qq;
            const float* Scq = Sc + qq * M16 * nPitch;
            __nv_bfloat16* Pq = P + qq * M16 * nPitch;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                int e = it * 32 + lane;
                int r = e / 8; int c = e % 8; int m = warp * 8 + c;
                int tok = union_tok[tbase + m];
                bool ok = (m < N_TILE) && (tok >= 0) && (tok <= s_qq) && (qmask[tbase + m] & (1u << qq));
                float p = ok ? __expf(Scq[r * nPitch + m] - sm_m[qq * M16 + r]) : 0.f;
                if (m < N_TILE) Pq[r * nPitch + m] = f2b(p);
            }
        }
        __syncthreads();
        for (int qq = 0; qq < M_TILE; qq++) {
            const __nv_bfloat16* Pq = P + qq * M16 * nPitch;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                int e = it * 32 + lane;
                int r = e / 8; int c = e % 8; int m = warp * 8 + c;
                float pv = (m < N_TILE) ? b2f(Pq[r * nPitch + m]) : 0.f;
                pv += __shfl_xor_sync(0xffffffffu, pv, 1);
                pv += __shfl_xor_sync(0xffffffffu, pv, 2);
                pv += __shfl_xor_sync(0xffffffffu, pv, 4);
                if (c == 0) plsum[qq * (M16 * 8) + r * 8 + warp] = pv;
            }
        }
        __syncthreads();
        for (int qq = 0; qq < M_TILE; qq++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                int r = gi + rh * 8;
                if (tid == (r & 7) * 4) {
                    float lt = 0.f;
                    for (int x = 0; x < 8; x++) lt += plsum[qq * (M16 * 8) + r * 8 + x];
                    sm_l[qq * M16 + r] += lt;
                }
            }
        __syncthreads();

        // ---- 6) TC PV: all queries, one barrier ----
        for (int qq = 0; qq < M_TILE; qq++) {
            const __nv_bfloat16* Pq = P + qq * M16 * nPitch;
            for (int oc = 0; oc < 4; oc++) {
                float o0v[4] = {0.f, 0.f, 0.f, 0.f};
                float o1v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kt = 0; kt < N_TILE / K16; kt++) {
                    const int ko = kt * K16;
                    const int dg = warp * 32 + oc * 8 + gi;
                    uint32_t a[4];
                    a[0] = pack2(Pq[gi * nPitch + ko + li],       Pq[gi * nPitch + ko + li + 8]);
                    a[1] = pack2(Pq[(gi + 8) * nPitch + ko + li], Pq[(gi + 8) * nPitch + ko + li + 8]);
                    a[2] = pack2(Pq[gi * nPitch + ko + li + 4],   Pq[gi * nPitch + ko + li + 12]);
                    a[3] = pack2(Pq[(gi + 8) * nPitch + ko + li + 4], Pq[(gi + 8) * nPitch + ko + li + 12]);
                    uint32_t bb[2];
                    bb[0] = pack2(Vsm[(ko + li) * kPitch + dg],      Vsm[(ko + li + 8) * kPitch + dg]);
                    bb[1] = pack2(Vsm[(ko + li + 4) * kPitch + dg],  Vsm[(ko + li + 12) * kPitch + dg]);
                    if (kt & 1) mma16n8k16(o1v, a, bb); else mma16n8k16(o0v, a, bb);
                }
                o0v[0] += o1v[0]; o0v[1] += o1v[1]; o0v[2] += o1v[2]; o0v[3] += o1v[3];
                O_r[qq][0][oc * 2 + 0] += o0v[0];
                O_r[qq][0][oc * 2 + 1] += o0v[1];
                O_r[qq][1][oc * 2 + 0] += o0v[2];
                O_r[qq][1][oc * 2 + 1] += o0v[3];
            }
        }
        __syncthreads();
    }

    // ---- finalize: thread (gi,li) owns rows gi/gi+8 of each query ----
    for (int qq = 0; qq < M_TILE; qq++) {
        const int s = s0 + qq;
#pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            const int r = gi + rh * 8;
            if (r < HBLK) {
                float inv = 1.f / fmaxf(sm_l[qq * M16 + r], 1e-30f);
                int h = gg * HBLK + r;
                __nv_bfloat16* o = out + ((long long)b * S + s) * H * D + h * D + warp * 32;
#pragma unroll
                for (int oc = 0; oc < 4; oc++) {
                    o[oc * 8 + 2 * li]     = f2b(O_r[qq][rh][oc * 2 + 0] * inv);
                    o[oc * 8 + 2 * li + 1] = f2b(O_r[qq][rh][oc * 2 + 1] * inv);
                }
            }
        }
    }
}

void launch_qsa_pass2_tc_reuse(const __nv_bfloat16* q, const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const int* sel_idx,
                               const int* sel_cnt, __nv_bfloat16* out,
                               int B, int S, int H, int KVH, int NMAX,
                               cudaStream_t stream) {
    const float scale = 1.0f / sqrtf((float)D);
    const int UMAX = U_MAX_CAP < S ? U_MAX_CAP : S;   // union token budget
    size_t smem_bytes = (size_t)((M_TILE * M16 + 2 * N_TILE) * kPitch * sizeof(__nv_bfloat16)
                                 + M_TILE * M16 * nPitch * sizeof(float)
                                 + M_TILE * M16 * nPitch * sizeof(__nv_bfloat16)
                                 + (2 * M_TILE * M16 * 8 + 2 * M_TILE * M16) * sizeof(float)
                                 + 2 * M_TILE * M16 * sizeof(float)
                                 + ((S + 31) / 32) * sizeof(unsigned int)
                                 + S
                                 + sizeof(int)
                                 + (size_t)UMAX * sizeof(short)
                                 + UMAX);
    long long total_groups = (long long)B * S / M_TILE;
    int nblocks = (int)(total_groups * KVH);
    auto kern = qsa_pass2_tc_reuse_kernel;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    kern<<<nblocks, 256, smem_bytes, stream>>>(
        q, k, v, sel_idx, sel_cnt, out, B, S, H, KVH, NMAX, scale, UMAX);
}

}  // namespace qsa_pass2_tc_reuse
}  // namespace qsa_pass2_tc_v3
