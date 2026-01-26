#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

template<int L>
__global__ void soft_hash_collision_kernel_vnorm(
    const __half* __restrict__ q_probs,       // [B,H,L,R]
    const int16_t* __restrict__ key_buckets,  // [B,H,L,T]
    const __half* __restrict__ v_norm,        // [B,H,T]
    const bool* __restrict__ allowed,         // [B,H,T]
    float* __restrict__ out,                  // [B,H,T]
    int B, int H, int R, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (t >= T) return;

    int bh = b * H + h;
    int out_idx = bh * T + t;

    if (!allowed[out_idx]) {
        out[out_idx] = -CUDART_INF_F;
        return;
    }

    float sum = 0.f;
    #pragma unroll
    for (int l = 0; l < L; ++l) {
        int kb_idx = (bh * L + l) * T + t;
        int r = (int)key_buckets[kb_idx];
        // If you are not 100% sure r is in-range, clamp:
        // r = max(0, min(r, R - 1));
        int q_idx = (bh * L + l) * R + r;
        sum += __half2float(q_probs[q_idx]);
    }

    float vn = __half2float(v_norm[out_idx]);
    out[out_idx] = sum * vn;
}


static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

torch::Tensor soft_hash_score_cuda(
    torch::Tensor q_probs,      // [B,H,L,R] half
    torch::Tensor key_buckets,  // [B,H,L,T] int16
    torch::Tensor v_norm,       // [B,H,T] half
    torch::Tensor allowed       // [B,H,T] bool
) {
    const auto B = (int)q_probs.size(0);
    const auto H = (int)q_probs.size(1);
    const auto L = (int)q_probs.size(2);
    const auto R = (int)q_probs.size(3);
    const auto T = (int)key_buckets.size(3);

    auto out = torch::empty({B, H, T}, torch::TensorOptions().dtype(torch::kFloat32).device(q_probs.device()));

    const int threads = 64;
    dim3 block(threads);
    dim3 grid(ceil_div(T, threads), H, B);

    const __half* q_ptr = (const __half*)q_probs.data_ptr<at::Half>();
    const int16_t* k_ptr = (const int16_t*)key_buckets.data_ptr<int16_t>();
    const __half* v_ptr = (const __half*)v_norm.data_ptr<at::Half>();
    const bool* a_ptr = (const bool*)allowed.data_ptr<bool>();
    float* o_ptr = (float*)out.data_ptr<float>();

    // Dispatch on compile-time L for unrolling.
    switch (L) {
        case 50:
            soft_hash_collision_kernel_vnorm<50><<<grid, block>>>(
                q_ptr, k_ptr, v_ptr, a_ptr, o_ptr, B, H, R, T);
            break;
        case 32:
            soft_hash_collision_kernel_vnorm<32><<<grid, block>>>(
                q_ptr, k_ptr, v_ptr, a_ptr, o_ptr, B, H, R, T);
            break;
        case 64:
            soft_hash_collision_kernel_vnorm<64><<<grid, block>>>(
                q_ptr, k_ptr, v_ptr, a_ptr, o_ptr, B, H, R, T);
            break;
        default:
            TORCH_CHECK(false, "Unsupported L for soft_hash_score_cuda: ", L,
                        ". Add a template instantiation in the switch.");
    }

    // Optional: check for launch errors in debug builds
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return out;
}