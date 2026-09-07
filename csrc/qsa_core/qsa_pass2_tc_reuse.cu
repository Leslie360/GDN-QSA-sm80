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
// mma-fragment swizzle: within each 16-element K-group store natural columns
// j and j+8 (the two bf16 packed into one mma A/B register) CONTIGUOUSLY, so a
// single LDS.32 loads the whole register pair.  swz16(0..7)=0,2,..14, 8..15=1,3,..15.
__device__ __forceinline__ int swz16(int j) { return (j & 7) * 2 + (j >> 3); }
__device__ __forceinline__ int swz_d(int d) { return (d & ~15) + swz16(d & 15); }   // D=256 d
__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

constexpr int M16 = 16;
constexpr int D = 256;
constexpr int N_TILE = 64;      // union token rows per streaming tile
constexpr int kPitch = D + 8;   // 264 bf16/row: kill 8-way bank conflicts
constexpr int nPitch = N_TILE + 8;  // 72 bf16/row for P, 72 f32/row for Sc:
// 36 words/row -> 36%32=4 gives the PV A-fragment ld32 a full 32-bank spread
// (nPitch=68 -> 34%32=2 banks only 0..17, 2-way conflicts).
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
    float* sm_m = reinterpret_cast<float*>(sp);                           // [2][M_TILE][16] ping-pong
    sp += 2 * M_TILE * M16 * sizeof(float);
    float* sm_l = reinterpret_cast<float*>(sp);                           // [2][M_TILE][16] ping-pong
    sp += 2 * M_TILE * M16 * sizeof(float);
    // NOTE: O accumulator lives in REGISTERS (O_r[2][8] per thread) — putting
    // it in smem requires a (warp, gi, li) key (gi=lane/4 repeats across the 8
    // warps), which is what the racecheck failures were about.
    unsigned int* bitmap = reinterpret_cast<unsigned int*>(sp);           // [S/32] int, atomicOr
    sp += ((S + 31) / 32) * sizeof(unsigned int);
    unsigned char* qmap = reinterpret_cast<unsigned char*>(sp);           // [S]
    sp += S * sizeof(unsigned char);
    int* u_cnt = reinterpret_cast<int*>(sp);                              // union length
    sp += sizeof(int);
    // [UMAX + N_TILE] int16: +N_TILE padding so the last streaming tile (which
    // reads all 64 slots of a partial tile, tbase_last+63 possibly > UMAX when
    // S%64 != 0) stays in-bounds.  (query-ownership is read live from qmap[token];
    // no [UMAX] qmask copy — saves 8KB smem, keeps us under 166912B with nPitch=72.)
    short* union_tok = reinterpret_cast<short*>(sp);
    sp += (size_t)(UMAX + N_TILE) * sizeof(short);

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
        Qsm[qq * M16 * kPitch + rr * kPitch + swz_d(d)] = val;
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
            if (pos < UMAX) union_tok[pos] = (short)tok;
        }
    }
    __syncthreads();
    const int U = *u_cnt;
    for (int e = tid + U; e < UMAX + N_TILE; e += 256) union_tok[e] = -1;
    __syncthreads();

    // ---- online-softmax state per query (O in registers, 16 f32 per query) ----
    for (int e = tid; e < 2 * M_TILE * M16; e += 256) { sm_m[e] = -1e30f; sm_l[e] = 0.f; }
    float O_r[M_TILE][2][8];
#pragma unroll
    for (int qq = 0; qq < M_TILE; qq++)
#pragma unroll
        for (int rh = 0; rh < 2; rh++)
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][rh][d] = 0.f;
    __syncthreads();
    const int n_utiles = (U + N_TILE - 1) / N_TILE;
    int buf = 0;   // sm_m/sm_l ping-pong parity (flipped after each tile)
    for (int t = 0; t < n_utiles; t++) {
        const int tbase = t * N_TILE;
        const float* sm_m_c = sm_m + buf * (M_TILE * M16);
        const float* sm_l_c = sm_l + buf * (M_TILE * M16);
        float* sm_m_n = sm_m + (buf ^ 1) * (M_TILE * M16);
        float* sm_l_n = sm_l + (buf ^ 1) * (M_TILE * M16);

        // ---- 1) vectorized gather K/V: union_tok[t*64 + row] -> Ksm/Vsm ----
        // Ksm is d-swizzled (swz_d) so QK's mma A/B fragments can LDS.32 whole
        // register pairs; Vsm keeps the natural row-major layout (its mma B
        // pairs span two token rows and cannot be made contiguous).
        // Each thread gathers 16 cols (2x uint4): the swz pair (col j, j+8) is
        // then both in hand, so K stores as 8 uint32 (was 16 uint16 after the
        // split-store) — store-issue halved.
        for (int it = 0; it < (N_TILE * D / 16) / 256; it++) {
            int vidx = it * 256 + tid;
            int m  = vidx >> 4;        // union-token row within the tile
            int cb = vidx & 15;        // 16-col block
            int tok = union_tok[tbase + m];
            const int base = cb * 16;
            int e8 = m * kPitch + base;
            uint16_t* kp16 = reinterpret_cast<uint16_t*>(&Ksm[m * kPitch]);
            if (tok >= 0) {
                long long off = ((long long)b * S + tok) * KVH * D + gg * D + base;
                uint4 kv0 = ldg8(k, (int)off);
                uint4 kv1 = ldg8(k, (int)off + 8);
                uint4 vv0 = ldg8(v, (int)off);
                uint4 vv1 = ldg8(v, (int)off + 8);
                *reinterpret_cast<uint4*>(&Vsm[e8]) = vv0;
                *reinterpret_cast<uint4*>(&Vsm[e8 + 8]) = vv1;
                uint16_t* k0 = reinterpret_cast<uint16_t*>(&kv0);
                uint16_t* k1 = reinterpret_cast<uint16_t*>(&kv1);
#pragma unroll
                for (int j = 0; j < 8; j++) {   // pair (col base+j, base+8+j) -> (2j, 2j+1)
                    uint32_t pair = (uint32_t(k1[j]) << 16) | uint32_t(k0[j]);
                    *reinterpret_cast<uint32_t*>(&kp16[base + 2 * j]) = pair;
                }
            } else {
                *reinterpret_cast<uint4*>(&Vsm[e8]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&Vsm[e8 + 8]) = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    *reinterpret_cast<uint32_t*>(&kp16[base + 2 * j]) = 0;
            }
        }
        __syncthreads();

        // ---- 2) QK: all queries share one barrier ----
        // The Ksm B operand is identical across the 4 qq iterations — hoist it
        // (like the PV hoist).  The ROW-MAX is computed here directly from the
        // register scores (4-lane shuffle, no Sc round-trip), eliminating the
        // separate row-max phase + its barrier.
        if (warp < 8) {
            const int colw = warp * 8;
            const int m0 = colw + 2 * li;            // this thread's score columns
            const int m1 = m0 + 1;
            const int tok0 = union_tok[tbase + m0];  // qq-invariant token/ownership
            const int tok1 = union_tok[tbase + m1];
            const unsigned char qm0 = (tok0 >= 0) ? qmap[tok0] : 0;
            const unsigned char qm1 = (tok1 >= 0) ? qmap[tok1] : 0;
            uint32_t Kf[D / K16][2];
#pragma unroll
            for (int kb = 0; kb < D / K16; kb++) {
                const int ko = kb * K16;
                Kf[kb][0] = ld32(&Ksm[(colw + gi) * kPitch + ko + 2 * li]);
                Kf[kb][1] = ld32(&Ksm[(colw + gi) * kPitch + ko + 2 * li + 8]);
            }
            for (int qq = 0; qq < M_TILE; qq++) {
                float s0v[4] = {0.f, 0.f, 0.f, 0.f};
                float s1v[4] = {0.f, 0.f, 0.f, 0.f};
                const __nv_bfloat16* Qq = Qsm + qq * M16 * kPitch;
                float* Scq = Sc + qq * M16 * nPitch;
#pragma unroll
                for (int kb = 0; kb < D / K16; kb++) {
                    const int ko = kb * K16;
                    uint32_t a[4];
                    a[0] = ld32(&Qq[gi * kPitch + ko + 2 * li]);
                    a[1] = ld32(&Qq[(gi + 8) * kPitch + ko + 2 * li]);
                    a[2] = ld32(&Qq[gi * kPitch + ko + 2 * li + 8]);
                    a[3] = ld32(&Qq[(gi + 8) * kPitch + ko + 2 * li + 8]);
                    uint32_t bb[2] = {Kf[kb][0], Kf[kb][1]};
                    if (kb & 1) mma16n8k16(s1v, a, bb); else mma16n8k16(s0v, a, bb);
                }
                s0v[0] += s1v[0]; s0v[1] += s1v[1]; s0v[2] += s1v[2]; s0v[3] += s1v[3];
                // row-max for rows gi, gi+8 from the register scores: mask invalid
                // columns, local max of the thread's 2 scores, 4-lane (li) reduce.
                const int s_qq = s0 + qq;
                const bool v0 = (tok0 >= 0) && (tok0 <= s_qq) && (qm0 & (1u << qq));
                const bool v1 = (tok1 >= 0) && (tok1 <= s_qq) && (qm1 & (1u << qq));
                float mx = fmaxf(v0 ? s0v[0] : -1e30f, v1 ? s0v[1] : -1e30f);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (li == 0) pmax[qq * (M16 * 8) + gi * 8 + warp] = mx * scale;
                float mx2 = fmaxf(v0 ? s0v[2] : -1e30f, v1 ? s0v[3] : -1e30f);
                mx2 = fmaxf(mx2, __shfl_xor_sync(0xffffffffu, mx2, 1));
                mx2 = fmaxf(mx2, __shfl_xor_sync(0xffffffffu, mx2, 2));
                if (li == 0) pmax[qq * (M16 * 8) + (gi + 8) * 8 + warp] = mx2 * scale;
                // store scaled scores for the softmax/P phase
                s0v[0] *= scale; s0v[1] *= scale; s0v[2] *= scale; s0v[3] *= scale;
                Scq[gi * nPitch + colw + 2 * li]         = s0v[0];
                Scq[gi * nPitch + colw + 2 * li + 1]     = s0v[1];
                Scq[(gi + 8) * nPitch + colw + 2 * li]   = s0v[2];
                Scq[(gi + 8) * nPitch + colw + 2 * li + 1] = s0v[3];
            }
        }
        __syncthreads();

        // ---- 3) running max: mnew[r] = max(sm_m[r], tile row max); one thread
        //      per row (16 threads), 8 pmax reads batched so the max tree overlaps
        //      load latency.  Storing mnew (not the raw max) lets BOTH the rescale
        //      and the P phase read a single value, so phases 4+5 fuse.
        if ((tid & 15) == 0) {                       // one thread per row
            const int r = tid >> 4;
            for (int qq = 0; qq < M_TILE; qq++) {
                const float* pm8 = pmax + qq * (M16 * 8) + r * 8;
                const float p0 = pm8[0], p1 = pm8[1], p2 = pm8[2], p3 = pm8[3];
                const float p4 = pm8[4], p5 = pm8[5], p6 = pm8[6], p7 = pm8[7];
                const float mm = fmaxf(fmaxf(fmaxf(p0, p1), fmaxf(p2, p3)),
                                       fmaxf(fmaxf(p4, p5), fmaxf(p6, p7)));
                tmax[qq * M16 + r] = fmaxf(sm_m_c[qq * M16 + r], mm);   // now holds mnew
            }
        }
        __syncthreads();

        // ---- 4) fused online-rescale + P + rowsum (ONE barrier).  P items are
        // (row, token-PAIR): thread tid handles rows {warp, warp+8} x pair lane
        // -> tokens (16g+m, +8) with g=lane/8, m=lane%8, packed into ONE u32 at
        // swizzled slot 16g+2m (slot 2a <-> token a, 2a+1 <-> token a+8).  Each
        // row's 32 pairs sit in ONE warp -> the rowsum reduces with 5 in-warp
        // shuffles, NO plsum smem round trip.  Pad rows skip the P store.
        const int pg    = lane >> 3;               // 16-token group 0..3
        const int pm    = lane & 7;                // pair slot within group
        const int plo   = pg * 16 + pm;            // lo token 0..63
        const int phi   = plo + 8;
        const int ptok0 = union_tok[tbase + plo];
        const int ptok1 = union_tok[tbase + phi];
        const unsigned char pqm0 = (ptok0 >= 0) ? qmap[ptok0] : 0;
        const unsigned char pqm1 = (ptok1 >= 0) ? qmap[ptok1] : 0;
        float lsum0[M_TILE], lsum1[M_TILE];
#pragma unroll
        for (int qq = 0; qq < M_TILE; qq++) {
            // rescale O_r (sm_m still pre-tile) — own rows gi, gi+8
            const int r0 = gi, r1 = gi + 8;
            float corr0 = __expf(sm_m_c[qq * M16 + r0] - tmax[qq * M16 + r0]);
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][0][d] *= corr0;
            float corr1 = __expf(sm_m_c[qq * M16 + r1] - tmax[qq * M16 + r1]);
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[qq][1][d] *= corr1;
            // P (two packed pair stores, rows warp / warp+8) + in-register rowsum
            const int s_qq = s0 + qq;
            const float* Scq = Sc + qq * M16 * nPitch;
            __nv_bfloat16* Pq = P + qq * M16 * nPitch;
            const float mn0 = tmax[qq * M16 + warp];
            const float mn1 = tmax[qq * M16 + warp + 8];
            const bool ok0 = (ptok0 >= 0) && (ptok0 <= s_qq) && (pqm0 & (1u << qq));
            const bool ok1 = (ptok1 >= 0) && (ptok1 <= s_qq) && (pqm1 & (1u << qq));
            const float p0 = ok0 ? __expf(Scq[warp * nPitch + plo] - mn0) : 0.f;
            const float p1 = ok1 ? __expf(Scq[warp * nPitch + phi] - mn0) : 0.f;
            const float q0 = ok0 ? __expf(Scq[(warp + 8) * nPitch + plo] - mn1) : 0.f;
            const float q1 = ok1 ? __expf(Scq[(warp + 8) * nPitch + phi] - mn1) : 0.f;
            if (warp < HBLK)
                *reinterpret_cast<uint32_t*>(&Pq[warp * nPitch + pg * 16 + 2 * pm]) = pack2(f2b(p0), f2b(p1));
            if (warp + 8 < HBLK)
                *reinterpret_cast<uint32_t*>(&Pq[(warp + 8) * nPitch + pg * 16 + 2 * pm]) = pack2(f2b(q0), f2b(q1));
            float rs0 = b2f(f2b(p0)) + b2f(f2b(p1));   // quantized like PV operands
            float rs1 = b2f(f2b(q0)) + b2f(f2b(q1));
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 4);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 8);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 16);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 4);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 8);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 16);
            lsum0[qq] = rs0;
            lsum1[qq] = rs1;
        }
        // ---- 5) sm_l/sm_m update, fused INTO phase 4 (before its barrier): lanes
        //      0/1 of each warp own rows {w, w+8} and write the PING-PONG next
        //      buffer while everyone else still reads the current one — no
        //      race, no separate phase, no serial-chain exposure.
        if (lane < 2) {
            const int r = warp + lane * 8;   // lane0 -> row w, lane1 -> row w+8
            float mn[M_TILE], smo[M_TILE], slo[M_TILE];
#pragma unroll
            for (int qq = 0; qq < M_TILE; qq++) {
                mn[qq]  = tmax[qq * M16 + r];
                smo[qq] = sm_m_c[qq * M16 + r];
                slo[qq] = sm_l_c[qq * M16 + r];
            }
#pragma unroll
            for (int qq = 0; qq < M_TILE; qq++) {
                const float corr = __expf(smo[qq] - mn[qq]);
                sm_l_n[qq * M16 + r] = slo[qq] * corr + (lane == 0 ? lsum0[qq] : lsum1[qq]);
                sm_m_n[qq * M16 + r] = mn[qq];
            }
        }
        __syncthreads();

        // ---- 6) TC PV: all queries, one barrier ----
        // Fragment operands are loop-invariant in a way the compiler cannot
        // always CSE under register pressure: P (A operand) is identical across
        // the 4 oc iterations, and V (B operand) is identical across the 4 qq
        // iterations.  Hoist both explicitly — 512 -> 128 LDS per thread/tile
        // (A: 4kt x 4 x 4qq = 64, B: 4oc x 4kt x 2 = 32).
        uint32_t Vf[4][4][2];
        for (int oc = 0; oc < 4; oc++) {
            const int dg = warp * 32 + oc * 8 + gi;
#pragma unroll
            for (int kt = 0; kt < N_TILE / K16; kt++) {
                const int ko = kt * K16;
                Vf[oc][kt][0] = pack2(Vsm[(ko + li) * kPitch + dg],      Vsm[(ko + li + 8) * kPitch + dg]);
                Vf[oc][kt][1] = pack2(Vsm[(ko + li + 4) * kPitch + dg],  Vsm[(ko + li + 12) * kPitch + dg]);
            }
        }
        for (int qq = 0; qq < M_TILE; qq++) {
            const __nv_bfloat16* Pq = P + qq * M16 * nPitch;
            uint32_t Af[4][4];
#pragma unroll
            for (int kt = 0; kt < N_TILE / K16; kt++) {
                const int ko = kt * K16;
                // token-swizzled P: mma A register pair (token j, j+8) at
                // (ko+2j, ko+2j+1) — one LDS.32 loads both halves.
                Af[kt][0] = ld32(&Pq[gi * nPitch + ko + 2 * li]);
                Af[kt][1] = ld32(&Pq[(gi + 8) * nPitch + ko + 2 * li]);
                Af[kt][2] = ld32(&Pq[gi * nPitch + ko + 2 * li + 8]);
                Af[kt][3] = ld32(&Pq[(gi + 8) * nPitch + ko + 2 * li + 8]);
            }
            for (int oc = 0; oc < 4; oc++) {
                float o0v[4] = {0.f, 0.f, 0.f, 0.f};
                float o1v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kt = 0; kt < N_TILE / K16; kt++) {
                    uint32_t a[4] = {Af[kt][0], Af[kt][1], Af[kt][2], Af[kt][3]};
                    uint32_t bb[2] = {Vf[oc][kt][0], Vf[oc][kt][1]};
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
        buf ^= 1;
    }

    // ---- finalize: thread (gi,li) owns rows gi/gi+8 of each query ----
    const float* sm_l_f = sm_l + buf * (M_TILE * M16);
    for (int qq = 0; qq < M_TILE; qq++) {
        const int s = s0 + qq;
#pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            const int r = gi + rh * 8;
            if (r < HBLK) {
                float inv = 1.f / fmaxf(sm_l_f[qq * M16 + r], 1e-30f);
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
                                 + (size_t)(UMAX + N_TILE) * sizeof(short));
    long long total_groups = (long long)B * S / M_TILE;
    int nblocks = (int)(total_groups * KVH);
    auto kern = qsa_pass2_tc_reuse_kernel;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    kern<<<nblocks, 256, smem_bytes, stream>>>(
        q, k, v, sel_idx, sel_cnt, out, B, S, H, KVH, NMAX, scale, UMAX);
}

}  // namespace qsa_pass2_tc_reuse
}  // namespace qsa_pass2_tc_v3
