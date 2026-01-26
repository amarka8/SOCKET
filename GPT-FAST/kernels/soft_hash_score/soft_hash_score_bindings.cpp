#include <torch/extension.h>

torch::Tensor soft_hash_score_cuda(
    torch::Tensor q_probs,      // [B,H,L,R] half
    torch::Tensor key_buckets,  // [B,H,L,T] int16
    torch::Tensor v_norm,       // [B,H,T] half
    torch::Tensor allowed       // [B,H,T] bool
);

torch::Tensor soft_hash_score(
    torch::Tensor q_probs,
    torch::Tensor key_buckets,
    torch::Tensor v_norm,
    torch::Tensor allowed
) {
    TORCH_CHECK(q_probs.is_cuda(), "q_probs must be CUDA");
    TORCH_CHECK(key_buckets.is_cuda(), "key_buckets must be CUDA");
    TORCH_CHECK(v_norm.is_cuda(), "v_norm must be CUDA");
    TORCH_CHECK(allowed.is_cuda(), "allowed must be CUDA");

    TORCH_CHECK(q_probs.scalar_type() == at::kHalf, "q_probs must be fp16 (half)");
    TORCH_CHECK(key_buckets.scalar_type() == at::kShort, "key_buckets must be int16");
    TORCH_CHECK(v_norm.scalar_type() == at::kHalf, "v_norm must be fp16 (half)");
    TORCH_CHECK(allowed.scalar_type() == at::kBool, "allowed must be bool");

    TORCH_CHECK(q_probs.is_contiguous(), "q_probs must be contiguous");
    TORCH_CHECK(key_buckets.is_contiguous(), "key_buckets must be contiguous");
    TORCH_CHECK(v_norm.is_contiguous(), "v_norm must be contiguous");
    TORCH_CHECK(allowed.is_contiguous(), "allowed must be contiguous");

    return soft_hash_score_cuda(q_probs, key_buckets, v_norm, allowed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_hash_score", &soft_hash_score, "Soft-hash score (CUDA)");
}