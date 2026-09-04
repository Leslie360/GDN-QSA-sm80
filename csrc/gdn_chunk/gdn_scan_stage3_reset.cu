// gdn_scan_stage3_reset.cu — decay-reset detection for Stage-2 dispatch.
//
// Phase C: if max_abs(A_g) < eps for (almost) all groups, the group transfer
// is numerically dead (A_g ≈ 0 in bf16 after decay), so the prefix scan can
// be skipped and each group's start state is just B_{g-1} (Stage-3 already
// reads prefix_B[g-1]; passing B_g unchanged implements the shift).
//
// This kernel computes, per (bh,g), the max absolute value of A_g into
// `out_max` [B*H*G] f32, plus a global count of groups exceeding `eps`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cutlass/bfloat16.h>

using BF16 = cutlass::bfloat16_t;

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float r; asm("cvt.f32.bf16 %0, %1;\n" : "=f"(r) : "h"(x.storage)); return r;
}

__global__ void gdn_reset_check_kernel(
    const BF16* __restrict__ A_g,   // [B*H*G, D, D] row-major bf16
    float* __restrict__ out_max,    // [B*H*G] f32 per-group max|A_g|
    float* __restrict__ out_count,  // [1] f32: #groups with max|A_g| >= eps
    float eps,
    int total_groups)
{
    // One block per group; the block cooperates over the [128,128] tile
    // (grid-stride), then block-reduces the per-thread maxima.
    const int gidx = blockIdx.x;
    if (gidx >= total_groups) return;
    const BF16* p = A_g + int64_t(gidx) * 128 * 128;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    float m = 0.f;
    for (int i = tid; i < 128 * 128; i += nthr)
        m = fmaxf(m, fabsf(bf16_to_f32(p[i])));
    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    __shared__ float sred[32];
    if (tid % 32 == 0) sred[tid / 32] = m;
    __syncthreads();
    if (tid < 8) {
        m = sred[tid];
        for (int off = 4; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0x000000ffu, m, off));
        if (tid == 0) {
            out_max[gidx] = m;
            if (m >= eps) atomicAdd(out_count, 1.f);
        }
    }
}

extern "C" void gdn_reset_check(
    const cutlass::bfloat16_t* A_g,
    float* out_max,
    float* out_count,
    float eps,
    int total_groups,
    cudaStream_t stream) {
    constexpr int BLOCK = 256;
    gdn_reset_check_kernel<<<total_groups, BLOCK, 0, stream>>>(
        A_g, out_max, out_count, eps, total_groups);
}
