// pybind / torch extension binding for the QSA sparse core attention op.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

namespace qsa_core {
template <typename T>
void launch_qsa_core(const T* q, const T* k, const T* v, const int* block_idx,
                     T* out, int* sel_idx_buf, int* sel_cnt_buf,
                     int B, int S, int H, int KVH, int KB, int r, int D,
                     cudaStream_t stream);
void launch_expand_debug(const int* block_idx, int* sel_idx, int* sel_cnt,
                         int B, int S, int KB, int r, cudaStream_t stream);
}

// q [B,S,H,D], k [B,S,KVH,D], v [B,S,KVH,D], block_idx [B,S,KB] (int32)
// -> out [B,S,H,D]
torch::Tensor qsa_sparse_core_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor block_idx, int64_t block_size) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && block_idx.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf ||
                    q.scalar_type() == at::kFloat,
                "q must be bf16, fp16, or fp32");
    TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(),
                "q/k/v must have the same dtype");

    const at::cuda::OptionalCUDAGuard guard(q.device());

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto bi_c = block_idx.contiguous();

    int B = q_c.size(0);
    int S = q_c.size(1);
    int H = q_c.size(2);
    int D = q_c.size(3);
    int KVH = k_c.size(2);
    int KB = bi_c.size(2);
    int r = (int)block_size;

    TORCH_CHECK(k_c.size(0) == B && k_c.size(1) == S && v_c.size(0) == B &&
                    v_c.size(1) == S,
                "shape mismatch");
    TORCH_CHECK(bi_c.size(0) == B && bi_c.size(1) == S, "block_idx shape mismatch");
    TORCH_CHECK(D % 32 == 0, "head_dim must be divisible by 32");
    TORCH_CHECK(H % KVH == 0, "H must be divisible by KVH");

    const int NMAX = KB * r + r;
    auto sel_idx = torch::empty({B, S, NMAX}, block_idx.options().dtype(at::kInt));
    auto sel_cnt = torch::empty({B, S}, block_idx.options().dtype(at::kInt));
    auto out = torch::empty_like(q_c);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (q_c.scalar_type() == at::kBFloat16) {
        qsa_core::launch_qsa_core<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_c.data_ptr()),
            bi_c.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            sel_idx.data_ptr<int>(), sel_cnt.data_ptr<int>(),
            B, S, H, KVH, KB, r, D, stream);
    } else if (q_c.scalar_type() == at::kHalf) {
        qsa_core::launch_qsa_core<__half>(
            reinterpret_cast<const __half*>(q_c.data_ptr()),
            reinterpret_cast<const __half*>(k_c.data_ptr()),
            reinterpret_cast<const __half*>(v_c.data_ptr()),
            bi_c.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr()),
            sel_idx.data_ptr<int>(), sel_cnt.data_ptr<int>(),
            B, S, H, KVH, KB, r, D, stream);
    } else {
        qsa_core::launch_qsa_core<float>(
            reinterpret_cast<const float*>(q_c.data_ptr()),
            reinterpret_cast<const float*>(k_c.data_ptr()),
            reinterpret_cast<const float*>(v_c.data_ptr()),
            bi_c.data_ptr<int>(),
            reinterpret_cast<float*>(out.data_ptr()),
            sel_idx.data_ptr<int>(), sel_cnt.data_ptr<int>(),
            B, S, H, KVH, KB, r, D, stream);
    }
    return out;
}

// Debug helper: run pass 1 (expand) only, return sel_idx and sel_cnt.
std::vector<torch::Tensor> debug_expand(torch::Tensor block_idx, int64_t block_size) {
    const at::cuda::OptionalCUDAGuard guard(block_idx.device());
    auto bi_c = block_idx.contiguous();
    int B = bi_c.size(0), S = bi_c.size(1), KB = bi_c.size(2), r = (int)block_size;
    const int NMAX = KB * r + r;
    auto sel_idx = torch::empty({B, S, NMAX}, bi_c.options().dtype(at::kInt));
    auto sel_cnt = torch::empty({B, S}, bi_c.options().dtype(at::kInt));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qsa_core::launch_expand_debug(bi_c.data_ptr<int>(), sel_idx.data_ptr<int>(),
                                  sel_cnt.data_ptr<int>(), B, S, KB, r, stream);
    return {sel_idx, sel_cnt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &qsa_sparse_core_attention,
          "QSA sparse core attention (sparse paged softmax over selected KV)");
    m.def("debug_expand", &debug_expand, "pass 1 expand debug");
}
