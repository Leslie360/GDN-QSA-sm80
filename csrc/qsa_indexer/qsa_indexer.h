#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> qsa_indexer_forward(
    torch::Tensor q, torch::Tensor raw_keys,
    torch::Tensor cos_q, torch::Tensor sin_q,
    torch::Tensor cos_k, torch::Tensor sin_k,
    int64_t r, int64_t block_topk);

// topK-only fused fast path: no dense [B,S,NB] block_scores materialized.
std::vector<torch::Tensor> qsa_indexer_topk_only_forward(
    torch::Tensor q, torch::Tensor raw_keys,
    torch::Tensor cos_q, torch::Tensor sin_q,
    torch::Tensor cos_k, torch::Tensor sin_k,
    int64_t r, int64_t block_topk);
