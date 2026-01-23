// start: soft_hash_collision_kernel_3
/*
    Tiled-over-T_k kernel: one thread per t in a tile for fixed (b,h).
    This improves coalescing for key_buckets and allowed_ext when T_k is large.

    Configuration:
    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid((T_k + threads - 1) / threads, H, B);

    Launch:
    soft_hash_collision_kernel_3<<<grid, block>>>(
        q.data_ptr<float>(),
        key_buckets.data_ptr<int16_t>(),
        allowed_ext.data_ptr<bool>(),
        v_hist.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(L),
        static_cast<int>(R),
        static_cast<int>(T_k)
    );
*/
__global__ void soft_hash_collision_kernel_3(
    const float* __restrict__ q_probs,        // [B,H,1,L,R] contiguous
    const int16_t* __restrict__ key_buckets,  // [B,H,L,T_k] contiguous
    const bool* __restrict__ allowed_ext,     // [B,H,1,T_k] contiguous
    const float* __restrict__ v_hist,        // [B,H,1,T_k] contiguous
    float* __restrict__ out,                  // [B,H,1,T_k] contiguous
    int B, int H, int L, int R, int T_k) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    if (t >= T_k || h >= H || b >= B) return;

    int bh = b * H + h;
    int al_idx = bh * T_k + t;
    if (!allowed_ext[al_idx]) {
        out[al_idx] = -INFINITY;
        return;
    }

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        int kb_idx = (bh * L + l) * T_k + t;
        int r = static_cast<int>(key_buckets[kb_idx]);
        int q_idx = (bh * L + l) * R + r;
        sum += q_probs[q_idx];
    }

    out[al_idx] = sum * v_hist[al_idx];
}
// end: soft_hash_collision_kernel_3
