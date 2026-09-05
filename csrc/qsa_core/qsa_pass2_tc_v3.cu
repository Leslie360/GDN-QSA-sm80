// qsa_pass2_tc_v3.cu — optimized qsa_pass2_tc.
// Profiling showed GATHER (K/V global gather) = 57.5% of time, the #1 bottleneck
// (barriers only 0.4%). v2 attacks the gather:
//   * VECTORIZED K/V gather: each thread loads 8 bf16 (16B uint4) at once instead
//     of 128 scalar 2B loads per tile -> 8x fewer load instructions, better MLP.
//   * 2 CTA/SM occupancy via __launch_bounds__(256,2) + reduced smem (N_TILE=32).
//     With N_TILE=32, QK uses 4 warps (each owns 8 cols); the other 4 warps idle
//     on QK but ALL 8 still share the cheaper gather. Occupancy 2 hides the
//     gather latency of one CTA behind compute of the other.
//   * Reduced barriers: removed the unnecessary sync between online-rescale and
//     P-quantize (m_r is a per-thread register, needs no cross-warp barrier).
// Algorithm identical (one-pass flash QK+PV, online softmax) -> correctness
// preserved at the 1e-3 relL1 level.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace qsa_pass2_tc_v3 {

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

constexpr int M16 = 16;
constexpr int D = 256;
constexpr int N_TILE = 64;   // 8 warps all do QK (each 8 cols)
constexpr int kPitch = D + 8;  // 264 bf16/row for Qsm/Ksm/Vsm: kill 8-way bank conflicts
constexpr int nPitch = N_TILE + 4;  // 68 bf16/row for P; Sc uses 36 f32/row
constexpr int K16 = 16;

// gather 8 consecutive bf16 (16B) as a uint4 from src element-offset e8 (multiple of 8).
__device__ __forceinline__ uint4 ldg8(const __nv_bfloat16* __restrict__ src, int e8) {
    return *reinterpret_cast<const uint4*>(src + e8);
}
__device__ __forceinline__ void stg8(__nv_bfloat16* __restrict__ dst, int e8, uint4 v) {
    *reinterpret_cast<uint4*>(dst + e8) = v;
}

__global__ void __launch_bounds__(256, 2) qsa_pass2_tc_v3_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int*    __restrict__ sel_idx,
    const int*    __restrict__ sel_cnt,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int H, int KVH, int NMAX, float scale) {
    const int HBLK = H / KVH;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gi   = lane / 4;
    const int li   = lane % 4;

    const int cta = blockIdx.x;
    const int gg  = cta % KVH;
    const int bs  = cta / KVH;
    if (bs >= B * S) return;
    const int s = bs % S;
    const int b = bs / S;

    const int n = sel_cnt[bs];
    const int* sidx = sel_idx + (long long)bs * NMAX;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qsm = reinterpret_cast<__nv_bfloat16*>(smem_raw);              // [16][264]
    __nv_bfloat16* Ksm = Qsm + M16 * kPitch;                                      // [64][264]
    __nv_bfloat16* Vsm = Ksm + N_TILE * kPitch;                                   // [64][264]
    float*  Sc  = reinterpret_cast<float*>(Vsm + N_TILE * kPitch);                // [16][68] f32
    __nv_bfloat16* P = reinterpret_cast<__nv_bfloat16*>(Sc + M16 * nPitch);       // [16][68] bf16
    bool*   valid = reinterpret_cast<bool*>(P + M16 * nPitch);                    // [64]
    float*  part = reinterpret_cast<float*>(valid + N_TILE);                      // [16][8]+[16][8]+[16]+[16]
    float*  pmax = part;
    float*  plsum = part + M16 * 8;
    float*  tmax = part + M16 * 16;
    float*  tlsum = tmax + M16;
    // Shared running per-row softmax max/sum.  Every row's max/sum is the same
    // for all threads, so one shared copy is correct and lets each lane update
    // only its OWN rows (gi, gi+8) instead of redundantly recomputing all 16
    // (32x waste) with the old per-thread local arrays.
    float*  sm_m = tlsum + M16;
    float*  sm_l = sm_m + M16;

    // ---- cooperative load Q: 12 real rows + 4 zero pad rows ----
    for (int e = tid; e < M16 * D; e += 256) {
        int r = e / D, d = e % D;
        __nv_bfloat16 val = f2b(0.f);
        if (r < HBLK) {
            int h = gg * HBLK + r;
            val = q[((long long)bs * H + h) * D + d];
        }
        Qsm[r * kPitch + d] = val;   // Qsm rows padded to kPitch=264
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < M16; r++) { sm_m[r] = -1e30f; sm_l[r] = 0.f; }
    float O_r[2][8];
#pragma unroll
    for (int rh = 0; rh < 2; rh++)
#pragma unroll
        for (int d = 0; d < 8; d++) O_r[rh][d] = 0.f;

    const int n_tiles = (n + N_TILE - 1) / N_TILE;
    for (int t = 0; t < n_tiles; t++) {
        const int base = t * N_TILE;

        // ---- 1) vectorized gather K/V: 32 rows x 256d = 32x32 uint4 = 1024 vecs ----
        // Each thread handles vec index tid, tid+256, ..., (4 vecs). 32 threads
        // cover one row (32 uint4 = 256 d), coalesced 512B per row.
        for (int it = 0; it < (N_TILE * D / 8) / 256; it++) {
            int vidx = it * 256 + tid;
            int m = vidx >> 5;     // row = vidx/32
            int cb = vidx & 31;    // col-block = vidx%32  (d = cb*8 .. cb*8+8)
            int j = (base + m < n) ? sidx[base + m] : -1;
            int e8 = m * kPitch + cb * 8;   // element offset into Ksm/Vsm row m (padded)
            if (j >= 0) {
                long long off = ((long long)b * S + j) * KVH * D + gg * D + cb * 8;
                *reinterpret_cast<uint4*>(&Ksm[e8]) = ldg8(k, (int)off);
                *reinterpret_cast<uint4*>(&Vsm[e8]) = ldg8(v, (int)off);
            } else {
                *reinterpret_cast<uint4*>(&Ksm[e8]) = make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(&Vsm[e8]) = make_uint4(0,0,0,0);
            }
        }
        if (tid < N_TILE) {
            int j = (base + tid < n) ? sidx[base + tid] : -1;
            valid[tid] = (j >= 0) && (j <= s);
        }
        __syncthreads();

        // ---- 2) QK mma: warp w owns N-column [w*8, w*8+8).  Two accumulators
        //      interleave the 16 serial mma.sync chains (each waits ~2 dozen
        //      cycles on the previous accumulator), hiding the latency. ----
        if (warp < 8) {
            float s0[4] = {0.f, 0.f, 0.f, 0.f};
            float s1[4] = {0.f, 0.f, 0.f, 0.f};
            const int colw = warp * 8;
#pragma unroll
            for (int kb = 0; kb < D / K16; kb++) {
                const int ko = kb * K16;
                uint32_t a[4];
                a[0] = pack2(Qsm[gi * kPitch + ko + li],      Qsm[gi * kPitch + ko + li + 8]);
                a[1] = pack2(Qsm[(gi + 8) * kPitch + ko + li], Qsm[(gi + 8) * kPitch + ko + li + 8]);
                a[2] = pack2(Qsm[gi * kPitch + ko + li + 4],  Qsm[gi * kPitch + ko + li + 12]);
                a[3] = pack2(Qsm[(gi + 8) * kPitch + ko + li + 4], Qsm[(gi + 8) * kPitch + ko + li + 12]);
                uint32_t bb[2];
                bb[0] = pack2(Ksm[(colw + gi) * kPitch + ko + li],     Ksm[(colw + gi) * kPitch + ko + li + 8]);
                bb[1] = pack2(Ksm[(colw + gi) * kPitch + ko + li + 4], Ksm[(colw + gi) * kPitch + ko + li + 12]);
                if (kb & 1) mma16n8k16(s1, a, bb); else mma16n8k16(s0, a, bb);
            }
            s0[0] += s1[0]; s0[1] += s1[1]; s0[2] += s1[2]; s0[3] += s1[3];
            s0[0] *= scale; s0[1] *= scale; s0[2] *= scale; s0[3] *= scale;
            Sc[gi * nPitch + colw + 2 * li]     = s0[0];
            Sc[gi * nPitch + colw + 2 * li + 1] = s0[1];
            Sc[(gi + 8) * nPitch + colw + 2 * li]     = s0[2];
            Sc[(gi + 8) * nPitch + colw + 2 * li + 1] = s0[3];
        }
        __syncthreads();

        // ---- 3) tile row-max over valid cols (all warps read Sc), butterfly ----
        // Distributed: lane l owns 4 (r, m) cells (same mapping as the P pass),
        // one 8-lane shuffle reduction collapses each row's 8 columns, and lane
        // c==0 of each row-group writes pmax.  The old version made every lane
        // redundantly re-iterate all 16x8 cells (32x waste).
#pragma unroll
        for (int it = 0; it < 4; it++) {
            int e = it * 32 + lane;
            int r = e / 8; int c = e % 8; int m = warp * 8 + c;
            float val = (m < N_TILE && valid[m]) ? Sc[r * nPitch + m] : -1e30f;
            val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 1));
            val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 2));
            val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, 4));
            if (c == 0) pmax[r * 8 + warp] = val;   // rows it*4+0..3, lane c==0
        }
        __syncthreads();
        for (int r = tid / 16; r < M16; r += 16) {
            float mm = -1e30f;
            for (int x = 0; x < 8; x++) mm = fmaxf(mm, pmax[r * 8 + x]);
            tmax[r] = mm;
        }
        __syncthreads();

        // ---- 4) online rescale O/l by corr, fold new max ----
        // Each lane only touches its OWN rows (gi, gi+8) in the shared running
        // max/sum; the 4 lanes sharing a row write the identical value, then a
        // barrier makes every row's m/l visible to the P pass below.
        {
            float mnew = fmaxf(sm_m[gi], tmax[gi]);
            float corr = __expf(sm_m[gi] - mnew);
            sm_m[gi] = mnew; sm_l[gi] *= corr;
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[0][d] *= corr;
            mnew = fmaxf(sm_m[gi + 8], tmax[gi + 8]);
            corr = __expf(sm_m[gi + 8] - mnew);
            sm_m[gi + 8] = mnew; sm_l[gi + 8] *= corr;
#pragma unroll
            for (int d = 0; d < 8; d++) O_r[1][d] *= corr;
            __syncthreads();
        }

        // ---- 5) P = exp(S - mnew) quantized to bf16; partial tile-sumexp ----
        // 16 rows x 8 cols per warp = 128 elems / 32 lanes = 4 per lane.
        // m_r is a register, Sc shared (already synced). P write must precede the
        // cross-warp plsum butterfly -> one barrier between P write and plsum.
#pragma unroll
        for (int it = 0; it < 4; it++) {
            int e = it * 32 + lane;
            int r = e / 8; int c = e % 8; int m = warp * 8 + c;
            float p = (m < N_TILE && valid[m]) ? __expf(Sc[r * nPitch + m] - sm_m[r]) : 0.f;
            // BUGFIX: warp 4-7 has m in [32,64) which is OOB for P[16][N_TILE=32].
            // Without the guard, P[r*32 + m] writes into row r+1 / the part buffers
            // (pmax/plsum), corrupting the softmax reduction for ALL GQA shapes.
            if (m < N_TILE) P[r * nPitch + m] = f2b(p);
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 4; it++) {
            int e = it * 32 + lane;
            int r = e / 8; int c = e % 8; int m = warp * 8 + c;
            float pv = (m < N_TILE) ? b2f(P[r * nPitch + m]) : 0.f;
            pv += __shfl_xor_sync(0xffffffffu, pv, 1);
            pv += __shfl_xor_sync(0xffffffffu, pv, 2);
            pv += __shfl_xor_sync(0xffffffffu, pv, 4);
            if (c == 0) plsum[r * 8 + warp] = pv;
        }
        __syncthreads();
        // each lane folds only its own rows' partial sums into sm_l
#pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            int r = gi + rh * 8;
            float lt = 0.f;
            for (int x = 0; x < 8; x++) lt += plsum[r * 8 + x];
            sm_l[r] += lt;
        }
        __syncthreads();

        // ---- 6) TC PV: all 8 warps split D into 8 slices of 32; read full P ----
        for (int oc = 0; oc < 4; oc++) {
            float o0[4] = {0.f, 0.f, 0.f, 0.f};
            float o1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kt = 0; kt < N_TILE / K16; kt++) {
                const int ko = kt * K16;
                const int dg = warp * 32 + oc * 8 + gi;
                uint32_t a[4];
                a[0] = pack2(P[gi * nPitch + ko + li],      P[gi * nPitch + ko + li + 8]);
                a[1] = pack2(P[(gi + 8) * nPitch + ko + li], P[(gi + 8) * nPitch + ko + li + 8]);
                a[2] = pack2(P[gi * nPitch + ko + li + 4],  P[gi * nPitch + ko + li + 12]);
                a[3] = pack2(P[(gi + 8) * nPitch + ko + li + 4], P[(gi + 8) * nPitch + ko + li + 12]);
                uint32_t bb[2];
                bb[0] = pack2(Vsm[(ko + li) * kPitch + dg],      Vsm[(ko + li + 8) * kPitch + dg]);
                bb[1] = pack2(Vsm[(ko + li + 4) * kPitch + dg],  Vsm[(ko + li + 12) * kPitch + dg]);
                if (kt & 1) mma16n8k16(o1, a, bb); else mma16n8k16(o0, a, bb);
            }
            o0[0] += o1[0]; o0[1] += o1[1]; o0[2] += o1[2]; o0[3] += o1[3];
            O_r[0][oc * 2 + 0] += o0[0];
            O_r[0][oc * 2 + 1] += o0[1];
            O_r[1][oc * 2 + 0] += o0[2];
            O_r[1][oc * 2 + 1] += o0[3];
        }
        __syncthreads();
    }

    // ---- finalize: thread (gi,li) owns rows gi/gi+8 and dims oc*8+2li/+1 ----
#pragma unroll
    for (int rh = 0; rh < 2; rh++) {
        const int r = gi + rh * 8;
        if (r < HBLK) {
            float inv = 1.f / fmaxf(sm_l[r], 1e-30f);
            int h = gg * HBLK + r;
            __nv_bfloat16* o = out + ((long long)bs * H + h) * D + warp * 32;
#pragma unroll
            for (int oc = 0; oc < 4; oc++) {
                o[oc * 8 + 2 * li]     = f2b(O_r[rh][oc * 2 + 0] * inv);
                o[oc * 8 + 2 * li + 1] = f2b(O_r[rh][oc * 2 + 1] * inv);
            }
        }
    }
}

torch::Tensor qsa_pass2_tc_v3(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              torch::Tensor sel_idx, torch::Tensor sel_cnt,
                              int64_t block_size) {
    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "TC path requires bf16");
    const at::cuda::OptionalCUDAGuard guard(q.device());

    auto qc = q.contiguous(); auto kc = k.contiguous(); auto vc = v.contiguous();
    auto sic = sel_idx.contiguous(); auto scc = sel_cnt.contiguous();

    int B = qc.size(0), S = qc.size(1), H = qc.size(2), Dd = qc.size(3);
    int KVH = kc.size(2); int NMAX = sic.size(2);
    TORCH_CHECK(Dd == D, "TC pass2 requires D=256");
    auto out = torch::empty_like(qc);
    float scale = 1.0f / sqrtf((float)D);

    size_t smem_bytes = (size_t)((M16 + 2 * N_TILE) * kPitch * sizeof(__nv_bfloat16)
                                 + M16 * nPitch * sizeof(float)
                                 + M16 * nPitch * sizeof(__nv_bfloat16)
                                 + N_TILE * sizeof(bool)
                                 + 2 * M16 * 8 * sizeof(float) + 2 * M16 * sizeof(float));
    long long total_ctas = (long long)B * S * KVH;
    int nblocks = (int)total_ctas;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto kern = qsa_pass2_tc_v3_kernel;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    kern<<<nblocks, 256, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr()),
        sic.data_ptr<int>(), scc.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, S, H, KVH, NMAX, scale);
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qsa_pass2_tc_v3", &qsa_pass2_tc_v3::qsa_pass2_tc_v3, "v2 optimized pass2");
}
