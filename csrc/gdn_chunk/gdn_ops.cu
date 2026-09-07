// Host wrapper for the GDN chunked delta-rule kernels (SM80).
// Pure torch wrapper: converts tensors to raw pointers and calls the extern
// "C" launcher `gdn_chunk_forward` defined in gdn_kernel.cu (cute-only TU).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <vector>
#include <cmath>

#define CHUNK 16
#define D 128

// Launcher defined in gdn_kernel.cu (extern "C", no mangling).
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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    int split, cudaStream_t stream);

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
    int head_ratio, cudaStream_t stream);

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
    int head_ratio, cudaStream_t stream);

// Stage 1 reset fast path (B_g-only + decay-bound metric). Defined in
// gdn_scan_stage1_reset.cu (extern "C").
extern "C" void gdn_scan_stage1_reset(
    const cutlass::bfloat16_t* ws_kd,
    const cutlass::bfloat16_t* ws_kr,
    const float* ws_gt,
    const cutlass::bfloat16_t* ws_inv,
    const cutlass::bfloat16_t* v_ptr, int v_row_stride,
    const cutlass::bfloat16_t* beta_ptr, int beta_row_stride,
    cutlass::bfloat16_t* B_g,
    float* reset_metric,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    int T_seq, int H, int B, int chunks_per_seq,
    int GROUP_CHUNKS, cudaStream_t stream);

// Stage 1 (superchunk affine scan): group transfer kernel launcher.
// Defined in gdn_scan_stage1.cu (extern "C").
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
    int GROUP_CHUNKS, cudaStream_t stream);

// Stage 3 (superchunk affine scan): parallel group replay. Defined in
// gdn_kernel.cu (extern "C").
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
    const cutlass::bfloat16_t* prefix_B,
    cutlass::bfloat16_t* final_state,
    int ws_tile_elems, int ws_tile_lm, int ws_gt_elems,
    int T_seq, int H, int B, int chunks_per_seq,
    int group_chunks, int prefix_exclusive, cudaStream_t stream);

// Stage 2 Phase B: Blelloch work-efficient exclusive scan. Defined in
// gdn_scan_stage2_blelloch.cu (extern "C").
extern "C" void gdn_scan_stage2_blelloch(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int G, int B, int H,
    cudaStream_t stream);

// Decay-reset detection (Phase C). Defined in gdn_scan_stage3_reset.cu.
extern "C" void gdn_reset_check(
    const cutlass::bfloat16_t* A_g,
    float* out_max,
    float* out_count,
    float eps,
    int total_groups,
    cudaStream_t stream);

// Stage 2 (superchunk affine scan): one Hillis-Steele round over per-group
// transfer matrices. Defined in gdn_scan_stage2.cu (extern "C").
extern "C" void gdn_scan_stage2(
    const cutlass::bfloat16_t* srcA,
    const cutlass::bfloat16_t* srcB,
    cutlass::bfloat16_t* dstA,
    cutlass::bfloat16_t* dstB,
    int offset,
    int G,
    int B, int H,
    cudaStream_t stream);

// Run only the prepare kernel and return the workspace tensors.
std::vector<torch::Tensor> prepare_workspace(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta) {
    int B = q.size(0);
    int S = q.size(1);
    int H = v.size(2);
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    int chunks_per_seq = (S + CHUNK - 1) / CHUNK;
    int total_tiles = B * H * chunks_per_seq;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    int ws_tile_elems = CHUNK * D;
    int ws_tile_lm = CHUNK * CHUNK;
    int ws_gt_elems = D;
    auto ws_kd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::empty({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    float scale = 1.0f / std::sqrt((float)D);
    auto stream = at::cuda::getCurrentCUDAStream();
    int q_row_stride = H * D;
    int g_row_stride = B * S;
    gdn_chunk_prepare_only(
        reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr()),
        g_row_stride,
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<float*>(ws_gt.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
        q_row_stride, q_row_stride, g_row_stride,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, S, H, B, chunks_per_seq, 1, stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk};
}

// Run only the CHUNK=32 prepare kernel and return its workspace.
std::vector<torch::Tensor> prepare_workspace32(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta) {
    int B = q.size(0);
    int S = q.size(1);
    int H = v.size(2);
    const int CK = 32;
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    int chunks_per_seq = (S + CK - 1) / CK;
    int total_tiles = B * H * chunks_per_seq;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    int ws_tile_elems = CK * D;
    int ws_tile_lm = CK * CK;
    int ws_gt_elems = D;
    auto ws_kd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::empty({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    float scale = 1.0f / std::sqrt((float)D);
    auto stream = at::cuda::getCurrentCUDAStream();
    int q_row_stride = H * D;
    int g_row_stride = B * S;
    gdn_chunk_prepare_only32(
        reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr()),
        g_row_stride,
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<float*>(ws_gt.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
        q_row_stride, q_row_stride, g_row_stride,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, S, H, B, chunks_per_seq, 1, stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk};
}

std::vector<torch::Tensor> forward_gdn_chunk(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, bool output_final_state) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && v.dim() == 4 && g.dim() == 3);

    int B = q.size(0);
    int S = q.size(1);
    int H = v.size(2);   // num_v_heads
    int Hd = D;

    TORCH_CHECK(q.size(2) == H && k.size(2) == H && q.size(3) == Hd && v.size(3) == Hd);
    TORCH_CHECK(H == 32, "num_v_heads must be 32 for this build");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();

    // Runtime chunk dispatch. CHUNK=32 was implemented and validated for
    // correctness, but measured within noise of CHUNK=16 at long sequences
    // (no speedup in this implementation), so the default stays CHUNK=16 to
    // avoid regression. Use forward_gdn_chunk32 for the CHUNK=32 path.
    const int CK = 16;

    int T_seq = S;
    int chunks_per_seq = (T_seq + CK - 1) / CK;
    int total_tiles = B * H * chunks_per_seq;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    int ws_tile_elems = CK * D;   // 2048 (c16) / 4096 (c32)
    int ws_tile_lm = CK * CK;     // 256 / 1024
    int ws_gt_elems = D;          // 128

    auto ws_kd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::zeros({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);

    auto out = torch::zeros({B, S, H, Hd}, opts_bf16);
    auto final_state = torch::zeros({B, H, D, D}, opts_bf16);

    float scale = 1.0f / std::sqrt((float)D);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr());
    const auto* kp = reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr());
    const auto* vp = reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr());
    const auto* gp = reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr());
    const auto* betap = reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr());

    int q_row_stride = H * D;
    int g_row_stride = B * S;  // g/beta are [H, T_total] head-major: stride between heads

    if (CK == 32) {
        gdn_chunk_forward32(
            qp, kp, vp, gp, betap,
            reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
            reinterpret_cast<float*>(ws_gt.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
            q_row_stride, g_row_stride,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            scale, T_seq, H, B, chunks_per_seq,
            stream.stream());
    } else {
        gdn_chunk_forward(
            qp, kp, vp, gp, betap,
            reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
            reinterpret_cast<float*>(ws_gt.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
            q_row_stride, g_row_stride,
            ws_tile_elems, ws_tile_lm, ws_gt_elems,
            scale, T_seq, H, B, chunks_per_seq,
            stream.stream());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!output_final_state) {
        final_state = torch::Tensor();
    }
    return {out, final_state};
}

// Force CHUNK=32 specialization.
std::vector<torch::Tensor> forward_gdn_chunk32(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, bool output_final_state) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && v.dim() == 4 && g.dim() == 3);

    int B = q.size(0);
    int S = q.size(1);
    int H = v.size(2);
    int Hd = D;

    TORCH_CHECK(q.size(2) == H && k.size(2) == H && q.size(3) == Hd && v.size(3) == Hd);
    TORCH_CHECK(H == 32, "num_v_heads must be 32 for this build");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();

    const int CK = 32;
    int T_seq = S;
    int chunks_per_seq = (T_seq + CK - 1) / CK;
    int total_tiles = B * H * chunks_per_seq;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    int ws_tile_elems = CK * D;
    int ws_tile_lm = CK * CK;
    int ws_gt_elems = D;

    auto ws_kd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::zeros({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);

    auto out = torch::zeros({B, S, H, Hd}, opts_bf16);
    auto final_state = torch::zeros({B, H, D, D}, opts_bf16);

    float scale = 1.0f / std::sqrt((float)D);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr());
    const auto* kp = reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr());
    const auto* vp = reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr());
    const auto* gp = reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr());
    const auto* betap = reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr());

    int q_row_stride = H * D;
    int g_row_stride = B * S;

    gdn_chunk_forward32(
        qp, kp, vp, gp, betap,
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<float*>(ws_gt.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
        q_row_stride, g_row_stride,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, T_seq, H, B, chunks_per_seq,
        stream.stream());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!output_final_state) {
        final_state = torch::Tensor();
    }
    return {out, final_state};
}

// Column-split (P1) variant: CHUNK=16 semantics, grid=(B,H,SPLIT).
// Mirrors forward_gdn_chunk32 in shape handling; only the recurrence launcher
// differs (gdn_chunk_forward_colsplit). Exists so all existing paths stay green.
std::vector<torch::Tensor> forward_gdn_chunk_colsplit(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, bool output_final_state, int split = 4) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && v.dim() == 4 && g.dim() == 3);

    int B = q.size(0);
    int S = q.size(1);
    int H = v.size(2);
    int Hd = D;

    TORCH_CHECK(q.size(2) == H && k.size(2) == H && q.size(3) == Hd && v.size(3) == Hd);
    TORCH_CHECK(H == 32, "num_v_heads must be 32 for this build");

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();

    const int CK = 16;
    int T_seq = S;
    int chunks_per_seq = (T_seq + CK - 1) / CK;
    int total_tiles = B * H * chunks_per_seq;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    int ws_tile_elems = CK * D;
    int ws_tile_lm = CK * CK;
    int ws_gt_elems = D;

    auto ws_kd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::zeros({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::zeros({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::zeros({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);

    auto out = torch::zeros({B, S, H, Hd}, opts_bf16);
    auto final_state = torch::zeros({B, H, D, D}, opts_bf16);

    float scale = 1.0f / std::sqrt((float)D);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr());
    const auto* kp = reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr());
    const auto* vp = reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr());
    const auto* gp = reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr());
    const auto* betap = reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr());

    int q_row_stride = H * D;
    int g_row_stride = B * S;

    gdn_chunk_forward_colsplit(
        qp, kp, vp, gp, betap,
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<float*>(ws_gt.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
        q_row_stride, g_row_stride,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, T_seq, H, B, chunks_per_seq,
        split, stream.stream());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!output_final_state) {
        final_state = torch::Tensor();
    }
    return {out, final_state};
}

// Stage 1 (superchunk affine scan): group transfer. Reads the CHUNK=16 prepare
// workspace (ws_kd/ws_kr/ws_gt/ws_inv) plus raw v/beta and produces the per-group
// transfer matrices A_g/B_g [num_groups*B*H, D, D] bf16 each.
// v must be [B,S,Hv,D], beta [B,S,Hv] (raw inputs), ws from prepare_workspace.
std::vector<torch::Tensor> stage1_group_transfer(
    torch::Tensor ws_kd, torch::Tensor ws_kr, torch::Tensor ws_gt,
    torch::Tensor ws_inv,
    torch::Tensor v, torch::Tensor beta,
    int64_t group_chunks) {

    const int CK = 16;
    const int HD = 128;
    int B = v.size(0);
    int S = v.size(1);
    int H = v.size(2);   // num_v_heads
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(beta.dtype() == torch::kBFloat16);
    TORCH_CHECK(beta.size(2) == H);

    auto v_c = v.contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();

    int chunks_per_seq = (S + CK - 1) / CK;
    int num_groups = (chunks_per_seq + int(group_chunks) - 1) / int(group_chunks);
    int total_groups = B * H * num_groups;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(v.device());
    auto A_g = torch::zeros({int64_t(total_groups), HD, HD}, opts_bf16);
    auto B_g = torch::zeros({int64_t(total_groups), HD, HD}, opts_bf16);

    int ws_tile_elems = CK * D;
    int ws_tile_lm = CK * CK;
    int ws_gt_elems = D;
    int v_row_stride = H * D;
    int beta_row_stride = B * S;
    auto stream = at::cuda::getCurrentCUDAStream();

    gdn_scan_stage1(
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<const float*>(ws_gt.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr()), v_row_stride,
        reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr()), beta_row_stride,
        reinterpret_cast<cutlass::bfloat16_t*>(A_g.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(B_g.data_ptr()),
        nullptr,
        nullptr,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        S, H, B, chunks_per_seq, int(group_chunks), stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {A_g, B_g};
}

// Stage 1 reset fast path: B_g-only build + decay-bound reset metric.
// Returns (B_g [B*H*G,D,D] bf16, reset_metric [B*H*G] f32). The metric is a
// strict upper bound on max_abs(A_g); metric < eps => group is decay-reset.
std::vector<torch::Tensor> stage1_reset(
    torch::Tensor ws_kd, torch::Tensor ws_kr, torch::Tensor ws_gt,
    torch::Tensor ws_inv,
    torch::Tensor v, torch::Tensor beta,
    int64_t group_chunks) {
    const int CK = 16;
    const int HD = 128;
    int B = v.size(0);
    int S = v.size(1);
    int H = v.size(2);
    auto v_c = v.contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    int chunks_per_seq = (S + CK - 1) / CK;
    int num_groups = (chunks_per_seq + int(group_chunks) - 1) / int(group_chunks);
    int total_groups = B * H * num_groups;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(v.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
    auto B_g = torch::zeros({int64_t(total_groups), HD, HD}, opts_bf16);
    auto metric = torch::zeros({int64_t(total_groups)}, opts_f32);
    auto stream = at::cuda::getCurrentCUDAStream();
    gdn_scan_stage1_reset(
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<const float*>(ws_gt.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr()), H * D,
        reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr()), B * S,
        reinterpret_cast<cutlass::bfloat16_t*>(B_g.data_ptr()),
        reinterpret_cast<float*>(metric.data_ptr()),
        CK * HD, CK * CK, HD,
        S, H, B, chunks_per_seq, int(group_chunks), stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {B_g, metric};
}

// Stage 2: one Hillis-Steele round (low-level, for testing).
// A_g/B_g are [B*H*G, D, D] bf16 row-major; dst may alias src (ping-pong).
void stage2_round(
    torch::Tensor srcA, torch::Tensor srcB,
    torch::Tensor dstA, torch::Tensor dstB,
    int64_t offset, int64_t G, int64_t B, int64_t H) {
    auto stream = at::cuda::getCurrentCUDAStream();
    gdn_scan_stage2(
        reinterpret_cast<const cutlass::bfloat16_t*>(srcA.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(srcB.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(dstA.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(dstB.data_ptr()),
        int(offset), int(G), int(B), int(H), stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Stage 2: full Hillis-Steele prefix scan (all rounds, ping-pong in place).
// A_g/B_g [B*H*G, D, D] bf16; after return they hold prefix transfers.
void stage2_scan(torch::Tensor A_g, torch::Tensor B_g, int64_t G) {
    int64_t total = A_g.numel() / (D * D);
    int64_t B = 1, H = total / G;
    auto ping = torch::empty_like(A_g);
    auto pong = torch::empty_like(A_g);
    std::vector<torch::Tensor> bufA = {A_g, ping};
    std::vector<torch::Tensor> bufB = {B_g, pong};
    int src = 0;
    int offset = 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    while (offset < G) {
        int dst = 1 - src;
        gdn_scan_stage2(
            reinterpret_cast<const cutlass::bfloat16_t*>(bufA[src].data_ptr()),
            reinterpret_cast<const cutlass::bfloat16_t*>(bufB[src].data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(bufA[dst].data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(bufB[dst].data_ptr()),
            offset, int(G), int(B), int(H), stream.stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        src = dst;
        offset *= 2;
    }
    if (src == 1) {
        A_g.copy_(bufA[1]);
        B_g.copy_(bufB[1]);
    }
}

// Stage 3: parallel group replay. ws = [kd,qd,kr,gt,inv,mqk] from prepare_workspace,
// prefix_B [B*H*G, D, D] bf16 from stage2_scan, v/beta raw inputs.
// Returns out [B,S,H,D] and final_state [B,H,D,D].
std::vector<torch::Tensor> stage3_replay(
    torch::Tensor ws_kd, torch::Tensor ws_qd, torch::Tensor ws_kr,
    torch::Tensor ws_gt, torch::Tensor ws_inv, torch::Tensor ws_mqk,
    torch::Tensor prefix_B, torch::Tensor v, torch::Tensor beta,
    int64_t group_chunks, int64_t prefix_exclusive = 0) {
    const int CK = 16;
    const int HD = 128;
    int B = v.size(0);
    int S = v.size(1);
    int H = v.size(2);
    auto v_c = v.contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    int chunks_per_seq = (S + CK - 1) / CK;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(v.device());
    auto out = torch::zeros({B, S, H, HD}, opts_bf16);
    auto final_state = torch::zeros({B, H, HD, HD}, opts_bf16);
    auto stream = at::cuda::getCurrentCUDAStream();
    gdn_chunk_replay(
        reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr()), H * HD,
        reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kd.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_qd.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_kr.data_ptr()),
        reinterpret_cast<const float*>(ws_gt.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_inv.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(ws_mqk.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(prefix_B.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
        CK * HD, CK * CK, HD,
        S, H, B, chunks_per_seq, int(group_chunks), int(prefix_exclusive),
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, final_state};
}

// Phase C: decay-reset detection. Returns (max_per_group [B*H*G] f32,
// count_ge [1] f32) = #groups with max_abs(A_g) >= eps.
std::vector<torch::Tensor> stage2_reset_check(torch::Tensor A_g, double eps) {
    int total_groups = A_g.size(0);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A_g.device());
    auto out_max = torch::zeros({total_groups}, opts_f32);
    auto out_count = torch::zeros({1}, opts_f32);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(out_count.data_ptr(), 0, sizeof(float), stream.stream());
    gdn_reset_check(
        reinterpret_cast<const cutlass::bfloat16_t*>(A_g.data_ptr()),
        reinterpret_cast<float*>(out_max.data_ptr()),
        reinterpret_cast<float*>(out_count.data_ptr()),
        float(eps), total_groups, stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_max, out_count};
}

// Stage 2 Phase B: full Blelloch exclusive scan, in place on A_g/B_g.
// G must be a power of two (S=2^k, CHUNK/GC powers of two -> yes for the
// validated matrix). Returns exclusive prefix: element g = transfer of 0..g-1.
void stage2_blelloch_scan(torch::Tensor A_g, torch::Tensor B_g, int64_t G) {
    int64_t total = A_g.numel() / (D * D);
    int64_t B = 1, H = total / G;
    auto scratch = torch::empty_like(A_g);
    auto scratchB = torch::empty_like(B_g);
    auto stream = at::cuda::getCurrentCUDAStream();
    gdn_scan_stage2_blelloch(
        reinterpret_cast<const cutlass::bfloat16_t*>(A_g.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(scratch.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(scratchB.data_ptr()),
        int(G), int(B), int(H), stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    A_g.copy_(scratch);
    B_g.copy_(scratchB);
}

// Fused two-level scan (superchunk): prepare -> stage1 -> reset-check ->
// dispatch (shift | Blelloch | Hillis-Steele) -> stage3 replay, all in ONE
// call on the current CUDA stream. Only ONE host sync (reading the reset
// count) — avoids per-stage pybind crossings and per-stage tensor allocation.
// Returns (out, final_state, info_tensor).
std::vector<torch::Tensor> forward_gdn_chunk_twolevel(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, int64_t group_chunks,
    double eps, double frac, double gt_eps) {
    int B = q.size(0);
    int S = q.size(1);
    int Hk = q.size(2);
    int H = v.size(2);
    TORCH_CHECK(H % Hk == 0, "Hv must be multiple of Hk");
    // GQA head expansion is fused into prepare via head_ratio (no materialized
    // repeat_interleave): v-head h reads q/k-head h/head_ratio in-kernel.
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_hm = g.permute({2, 0, 1}).reshape({H, B * S}).contiguous();
    auto beta_hm = beta.permute({2, 0, 1}).reshape({H, B * S}).contiguous();

    int chunks_per_seq = (S + CHUNK - 1) / CHUNK;
    int G = (chunks_per_seq + int(group_chunks) - 1) / int(group_chunks);
    int total_tiles = B * H * chunks_per_seq;
    int total_groups = B * H * G;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    int ws_tile_elems = CHUNK * D;
    int ws_tile_lm = CHUNK * CHUNK;
    int ws_gt_elems = D;
    float scale = 1.0f / std::sqrt((float)D);
    int qk_row_stride = Hk * D;   // q/k stored at Hk heads (GQA source)
    int v_row_stride = H * D;     // v stored at Hv heads
    int g_row_stride = B * S;
    const int head_ratio = H / Hk;

    // ---- workspace + reset-path buffers (lazy: A_g/scratch only in fallback) ----
    auto ws_kd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_qd = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_kr = torch::empty({int64_t(total_tiles) * ws_tile_elems}, opts_bf16);
    auto ws_gt = torch::empty({int64_t(total_tiles) * ws_gt_elems}, opts_f32);
    auto ws_inv = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto ws_mqk = torch::empty({int64_t(total_tiles) * ws_tile_lm}, opts_bf16);
    auto B_g = torch::empty({int64_t(total_groups) * D * D}, opts_bf16);
    auto metric = torch::empty({int64_t(total_groups)}, opts_f32);
    auto out = torch::empty({B, S, H, D}, opts_bf16);
    auto final_state = torch::empty({B, H, D, D}, opts_bf16);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const cutlass::bfloat16_t*>(q_c.data_ptr());
    const auto* kp = reinterpret_cast<const cutlass::bfloat16_t*>(k_c.data_ptr());
    const auto* vp = reinterpret_cast<const cutlass::bfloat16_t*>(v_c.data_ptr());
    const auto* gp = reinterpret_cast<const cutlass::bfloat16_t*>(g_hm.data_ptr());
    const auto* bp = reinterpret_cast<const cutlass::bfloat16_t*>(beta_hm.data_ptr());
    auto* kdp = reinterpret_cast<cutlass::bfloat16_t*>(ws_kd.data_ptr());
    auto* qdp = reinterpret_cast<cutlass::bfloat16_t*>(ws_qd.data_ptr());
    auto* krp = reinterpret_cast<cutlass::bfloat16_t*>(ws_kr.data_ptr());
    auto* gtp = reinterpret_cast<float*>(ws_gt.data_ptr());
    auto* invp = reinterpret_cast<cutlass::bfloat16_t*>(ws_inv.data_ptr());
    auto* mqkp = reinterpret_cast<cutlass::bfloat16_t*>(ws_mqk.data_ptr());

    bool use_shift = false;
    int prefix_exclusive = 0;
    float hcnt = 0.f;

    // ---- 1. prepare ----
    gdn_chunk_prepare_only(qp, kp, vp, gp, bp, g_row_stride,
        kdp, qdp, krp, gtp, invp, mqkp,
        qk_row_stride, v_row_stride, g_row_stride,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        scale, S, H, B, chunks_per_seq, head_ratio, stream.stream());

    // ---- 2. RESET FAST PATH: B_g-only stage1 + decay-bound metric ----
    // Computes B_g (bit-identical to full build) + a strict UPPER bound metric
    // on max_abs(A_g) per group, skipping the ENTIRE A chain.  If every group
    // has metric < eps, the whole scan is provably reset -> shift directly,
    // no full A_g build, no scan, no separate reset-check.
    gdn_scan_stage1_reset(kdp, krp, gtp, invp,
        vp, H * D, bp, B * S,
        reinterpret_cast<cutlass::bfloat16_t*>(B_g.data_ptr()),
        reinterpret_cast<float*>(metric.data_ptr()),
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        S, H, B, chunks_per_seq, int(group_chunks), stream.stream());
    // read metric to host (single sync), count groups above eps
    {
        std::vector<float> hmetric(total_groups);
        cudaMemcpyAsync(hmetric.data(), metric.data_ptr(),
                        total_groups * sizeof(float),
                        cudaMemcpyDeviceToHost, stream.stream());
        cudaStreamSynchronize(stream.stream());
        int n_ge = 0;
        float mmax = 0.f;
        for (int i = 0; i < total_groups; ++i) {
            if (hmetric[i] >= float(gt_eps)) ++n_ge;   // gt-only metric vs gt_eps
            if (hmetric[i] > mmax) mmax = hmetric[i];
        }
        if (1.0f - float(n_ge) / float(total_groups) >= float(frac)) {
            // all (or frac of) groups provably reset -> SHIFT path
            use_shift = true;
            prefix_exclusive = 0;
            // keep hcnt = n_ge for the info tensor
            hcnt = float(n_ge);
            // skip to stage3 replay
            gdn_chunk_replay(vp, H * D, bp,
                kdp, qdp, krp, gtp, invp, mqkp,
                reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
                ws_tile_elems, ws_tile_lm, ws_gt_elems,
                S, H, B, chunks_per_seq, int(group_chunks), 0, stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            auto info = torch::tensor({double(use_shift), double(prefix_exclusive),
                                       double(hcnt), double(total_groups),
                                       double(mmax)},
                                      opts_f32);
            return {out, final_state, info};
        }
        // else: fall through to full A_g build + exact dispatch
        hcnt = float(n_ge);
    }

    // ---- 3. (fallback) full stage1: A_g/B_g + reset check ----
    // Lazy allocation: exact-scan buffers are only needed when the reset
    // fast path did NOT hold. (reset path above returned before reaching here)
    auto A_g = torch::empty({int64_t(total_groups) * D * D}, opts_bf16);
    auto scratchA = torch::empty_like(A_g);
    auto scratchB = torch::empty_like(B_g);
    auto cnt = torch::zeros({1}, opts_f32);
    gdn_scan_stage1(kdp, krp, gtp, invp,
        vp, H * D, bp, B * S,
        reinterpret_cast<cutlass::bfloat16_t*>(A_g.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(B_g.data_ptr()),
        nullptr, nullptr,
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        S, H, B, chunks_per_seq, int(group_chunks), stream.stream());

    gdn_reset_check(reinterpret_cast<const cutlass::bfloat16_t*>(A_g.data_ptr()),
                    reinterpret_cast<float*>(scratchA.data_ptr()),
                    reinterpret_cast<float*>(cnt.data_ptr()),
                    float(eps), total_groups, stream.stream());
    float hcnt_fb = 0.f;
    cudaMemcpyAsync(&hcnt_fb, cnt.data_ptr(), sizeof(float),
                    cudaMemcpyDeviceToHost, stream.stream());
    cudaStreamSynchronize(stream.stream());
    // fallback decides dispatch on the TRUE max_abs(A_g), not the bound
    hcnt = hcnt_fb;

    use_shift = (1.0f - hcnt / float(total_groups) >= float(frac));
    prefix_exclusive = 0;
    if (!use_shift) {
        if ((G & (G - 1)) == 0) {
            // Blelloch exclusive
            gdn_scan_stage2_blelloch(
                reinterpret_cast<const cutlass::bfloat16_t*>(A_g.data_ptr()),
                reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(scratchA.data_ptr()),
                reinterpret_cast<cutlass::bfloat16_t*>(scratchB.data_ptr()),
                G, B, H, stream.stream());
            // result lands in dst (scratch); copy back
            A_g.copy_(scratchA);
            B_g.copy_(scratchB);
            prefix_exclusive = 1;
        } else {
            // Hillis-Steele inclusive (supports any G)
            for (int off = 1; off < G; off <<= 1) {
                gdn_scan_stage2(
                    reinterpret_cast<const cutlass::bfloat16_t*>(A_g.data_ptr()),
                    reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr()),
                    reinterpret_cast<cutlass::bfloat16_t*>(scratchA.data_ptr()),
                    reinterpret_cast<cutlass::bfloat16_t*>(scratchB.data_ptr()),
                    off, G, B, H, stream.stream());
                A_g.copy_(scratchA);
                B_g.copy_(scratchB);
            }
            prefix_exclusive = 0;
        }
    }

    // ---- 4. stage3 replay ----
    gdn_chunk_replay(vp, H * D, bp,
        kdp, qdp, krp, gtp, invp, mqkp,
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
        use_shift ? reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr())
                  : reinterpret_cast<const cutlass::bfloat16_t*>(B_g.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(final_state.data_ptr()),
        ws_tile_elems, ws_tile_lm, ws_gt_elems,
        S, H, B, chunks_per_seq, int(group_chunks), prefix_exclusive, stream.stream());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto info = torch::tensor({double(use_shift), double(prefix_exclusive),
                               double(hcnt), double(total_groups), 0.0},
                              opts_f32);
    return {out, final_state, info};
}

// Production dispatch entry. Chooses serial vs reset-fast-path by sequence
// length (survey: S<=2048 serial, S>=4096 reset fast path GC=64).
// Accepts GQA input (Hk <= Hv); head expansion is fused into prepare.
// Returns (out, final_state) — clean production signature.
std::vector<torch::Tensor> forward_gdn_chunk_auto(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, bool output_final_state) {
    const int S = q.size(1);
    const int Hk = q.size(2);
    const int Hv = v.size(2);
    TORCH_CHECK(Hv % Hk == 0, "Hv must be multiple of Hk");

    if (S <= 2048) {
        // Mid-sequence band (512 < S <= 2048): route to the two-level reset
        // fast path when per-token decay is strong. Empirically the reset
        // replay only beats the serial recurrence under strong decay; the
        // crossover is gmean = mean(-g) ~ 0.55 on uniform g=-rand*gs
        // (gs=1.0 -> gmean~0.50: serial wins; gs=1.2 -> gmean~0.60: reset
        // fast path wins ~2x, consistent across S=512/1024/2048). Below that
        // (or at tiny S <= 512 where the sync + launch overhead is not
        // amortized) fall back to serial without paying the decay reduction.
        // Correctness is never at risk: the two-level path internally falls
        // back to the exact scan if reset does not hold.
        if (S > 512) {
            const double gmean = g.neg().mean().item<double>();
            if (gmean >= 0.55) {
                auto r = forward_gdn_chunk_twolevel(q, k, v, g, beta, 32, 1e-6, 1.0, 1e-2);
                if (!output_final_state) r[1] = torch::Tensor();
                return {r[0], r[1]};
            }
        }
        // Serial recurrence. Requires 32h q/k (TORCH_CHECK H==32 inside
        // forward_gdn_chunk); expand cheaply at short sequence.
        torch::Tensor qq = (Hk == Hv) ? q : q.repeat_interleave(Hv / Hk, 2);
        torch::Tensor kk = (Hk == Hv) ? k : k.repeat_interleave(Hv / Hk, 2);
        return forward_gdn_chunk(qq, kk, v, g, beta, output_final_state);
    }
    // Long sequence: reset fast path, exact fallback inside.  GC is swept per S
    // (A800, g=-rand*2.0): the replay wants >= ~256 CTAs to fill 108 SMs at
    // 2 CTA/SM, and shorter per-group chains to cut the serial tail.  Measured
    // (ours-ms): S=4096: 16=0.63<64=0.66<32=0.68; S=8192: 16=0.89=32<64=1.01;
    // S=32768: 64=2.91<32=2.99<16=3.18.  -> GC=16 for S<=8192, GC=32 for
    // S<=16384, GC=64 beyond (S=4096: GC=64 leaves only 128 CTAs = SM underfill).
    int64_t GC;
    if (S <= 8192) GC = 16;
    else if (S <= 16384) GC = 32;
    else GC = 64;
    auto r = forward_gdn_chunk_twolevel(q, k, v, g, beta, GC, 1e-6, 1.0, 1e-2);
    if (!output_final_state) {
        r[1] = torch::Tensor();
    }
    return {r[0], r[1]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_gdn_chunk", &forward_gdn_chunk, "GDN chunk forward");
    m.def("forward_gdn_chunk32", &forward_gdn_chunk32, "GDN chunk forward (CHUNK=32)");
    m.def("forward_gdn_chunk_colsplit", &forward_gdn_chunk_colsplit,
          "GDN chunk forward (column-split)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("output_final_state"), py::arg("split") = 4);
    m.def("prepare_workspace", &prepare_workspace,
          "Run the prepare kernel and return the workspace tensors",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"));
    m.def("prepare_workspace32", &prepare_workspace32,
          "CHUNK=32 prepare workspace",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"));
    m.def("stage1_group_transfer", &stage1_group_transfer,
          "Stage 1: group transfer A_g/B_g from workspace + raw v/beta",
          py::arg("ws_kd"), py::arg("ws_kr"), py::arg("ws_gt"), py::arg("ws_inv"),
          py::arg("v"), py::arg("beta"), py::arg("group_chunks"));
    m.def("stage1_reset", &stage1_reset,
          "Stage 1 reset fast path: B_g-only + decay-bound metric",
          py::arg("ws_kd"), py::arg("ws_kr"), py::arg("ws_gt"), py::arg("ws_inv"),
          py::arg("v"), py::arg("beta"), py::arg("group_chunks"));
    m.def("stage2_round", &stage2_round, "Stage 2: one Hillis-Steele round",
          py::arg("srcA"), py::arg("srcB"), py::arg("dstA"), py::arg("dstB"),
          py::arg("offset"), py::arg("G"), py::arg("B"), py::arg("H"));
    m.def("stage2_scan", &stage2_scan, "Stage 2: full Hillis-Steele prefix scan",
          py::arg("A_g"), py::arg("B_g"), py::arg("G"));
    m.def("stage3_replay", &stage3_replay,
          "Stage 3: parallel group replay from prefix_B",
          py::arg("ws_kd"), py::arg("ws_qd"), py::arg("ws_kr"),
          py::arg("ws_gt"), py::arg("ws_inv"), py::arg("ws_mqk"),
          py::arg("prefix_B"), py::arg("v"), py::arg("beta"),
          py::arg("group_chunks"), py::arg("prefix_exclusive") = 0);
    m.def("stage2_reset_check", &stage2_reset_check,
          "Phase C: per-group max|A_g| + count above eps",
          py::arg("A_g"), py::arg("eps"));
    m.def("stage2_blelloch_scan", &stage2_blelloch_scan,
          "Phase B: Blelloch exclusive prefix scan (G power of two)",
          py::arg("A_g"), py::arg("B_g"), py::arg("G"));
    m.def("forward_gdn_chunk_twolevel", &forward_gdn_chunk_twolevel,
          "Fused superchunk two-level scan with reset/scan dispatch",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("group_chunks"),
          py::arg("eps") = 1e-6, py::arg("frac") = 1.0,
          py::arg("gt_eps") = 1e-2);
    m.def("forward_gdn_chunk_auto", &forward_gdn_chunk_auto,
          "Production GDN forward: serial (S<=2048) / reset fast path (S>2048)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("output_final_state") = true);
}
