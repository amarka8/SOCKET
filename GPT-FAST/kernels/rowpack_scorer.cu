// ---------------------------------------------------------------------------
// HEAD-PACKED soft-hash scorer (socket::soft_hash_score_packed).
//
// Same score as socket::soft_hash_score -- for query head h = kv*rep + j:
//     out[b,h,t] = fp16( (sum_l fp32(q_probs[b,h,l, buckets[b,kv,l,t]])) * fp32(||v_t||) )
// accumulated in fp32 in ascending l, -inf past seq_len -- but q_probs arrives PACKED as
// [B,Hkv,L,R,4] with the GQA group's four query heads interleaved in the last dimension, so
// one aligned 8-byte load serves the whole group where the unpacked kernel issues four
// independent 2-byte gathers into four separate rows. At large R the unpacked gathers touch
// many more L2 sectors per warp and the per-group row working set falls out of L1; packing
// removes both effects. The caller selects this kernel only for R large enough to benefit.
//
// The transport is the only change: operand bits, fp32 add order, the final multiply, the
// fp16 rounding of the result and the -inf tail are identical to the Triton scorer, so the
// two kernels are bit-equal.
//
// Caller invariant (same as the Triton scorer): every bucket column in [0,T) holds a value
// in [0,R), including zero-filled padding columns.
//
// Launched on the current stream (CUDA-graph capturable) and error-checked.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template <typename T16>
__device__ __forceinline__ float to_f32(T16 v);
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_f32<__half>(__half v) {
    return __half2float(v);
}

template <typename TQ, typename TV>
__global__ void rowpack_scorer_kernel(
    const TQ* __restrict__ qp,              // [B, HKV, L, R, 4] packed
    const int16_t* __restrict__ kb,         // [B, HKV, L, T] int16
    const TV* __restrict__ vn,              // [B, HKV, T]
    const int* __restrict__ seqlen,         // device scalar
    __half* __restrict__ out,               // [B, HKV*4, T] fp16
    int L, int R, int T, int HKV)
{
    const int kv = blockIdx.y;
    const int b = blockIdx.z;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int16_t* kbr = kb + ((size_t)b * HKV + kv) * L * T;
    const TQ* qpr = qp + ((size_t)b * HKV + kv) * L * R * 4;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 4
    for (int l = 0; l < L; ++l) {
        int bk = (int)kbr[(size_t)l * T + t];
        unsigned long long v8 = *reinterpret_cast<const unsigned long long*>(
            qpr + ((size_t)l * R + bk) * 4);
        const TQ* x = reinterpret_cast<const TQ*>(&v8);
        a0 += to_f32<TQ>(x[0]);
        a1 += to_f32<TQ>(x[1]);
        a2 += to_f32<TQ>(x[2]);
        a3 += to_f32<TQ>(x[3]);
    }
    const float v = to_f32<TV>(vn[((size_t)b * HKV + kv) * T + t]);
    const bool keep = t < *seqlen;
    const __half ninf = __ushort_as_half(0xFC00u);   // fp16 -inf
    __half* ob = out + (((size_t)b * HKV + kv) * 4) * T + t;
    ob[0]             = keep ? __float2half_rn(a0 * v) : ninf;
    ob[(size_t)T]     = keep ? __float2half_rn(a1 * v) : ninf;
    ob[2 * (size_t)T] = keep ? __float2half_rn(a2 * v) : ninf;
    ob[3 * (size_t)T] = keep ? __float2half_rn(a3 * v) : ninf;
}

void soft_hash_score_packed(torch::Tensor qp_pack, torch::Tensor kb, torch::Tensor vn,
                            torch::Tensor seqlen, torch::Tensor out) {
    TORCH_CHECK(qp_pack.is_cuda() && kb.is_cuda() && vn.is_cuda() && out.is_cuda());
    TORCH_CHECK(qp_pack.is_contiguous() && kb.is_contiguous() && vn.is_contiguous()
                && out.is_contiguous(), "contiguous required");
    TORCH_CHECK(qp_pack.dim() == 5 && qp_pack.size(4) == 4, "qp_pack must be [B,HKV,L,R,4]");
    TORCH_CHECK(kb.scalar_type() == torch::kInt16, "buckets must be int16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be fp16");
    const int B = kb.size(0), HKV = kb.size(1), L = kb.size(2), T = kb.size(3);
    const int R = qp_pack.size(3);
    TORCH_CHECK(qp_pack.size(0) == B && qp_pack.size(1) == HKV && qp_pack.size(2) == L);
    TORCH_CHECK(out.size(1) == HKV * 4, "out must carry HKV*4 query heads");
    const int block = 512;
    dim3 grid((T + block - 1) / block, HKV, B);
    auto st = at::cuda::getCurrentCUDAStream();
    const bool q_bf = qp_pack.scalar_type() == torch::kBFloat16;
    const bool q_fp = qp_pack.scalar_type() == torch::kHalf;
    const bool v_bf = vn.scalar_type() == torch::kBFloat16;
    const bool v_fp = vn.scalar_type() == torch::kHalf;
    TORCH_CHECK(q_bf || q_fp, "packed scorer supports bf16/fp16 q_probs only");
    TORCH_CHECK(v_bf || v_fp, "packed scorer supports bf16/fp16 v_norm only");
#define RP_LAUNCH(TQ, TV)                                                        \
    rowpack_scorer_kernel<TQ, TV><<<grid, block, 0, st>>>(                        \
        reinterpret_cast<const TQ*>(qp_pack.data_ptr()), kb.data_ptr<int16_t>(),  \
        reinterpret_cast<const TV*>(vn.data_ptr()), seqlen.data_ptr<int>(),       \
        reinterpret_cast<__half*>(out.data_ptr()), L, R, T, HKV)
    if (q_bf && v_bf)      RP_LAUNCH(__nv_bfloat16, __nv_bfloat16);
    else if (q_bf && v_fp) RP_LAUNCH(__nv_bfloat16, __half);
    else if (q_fp && v_bf) RP_LAUNCH(__half, __nv_bfloat16);
    else                   RP_LAUNCH(__half, __half);
#undef RP_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_hash_score_packed", &soft_hash_score_packed,
          "head-packed GQA soft-hash scorer, bit-equal to socket::soft_hash_score");
}
