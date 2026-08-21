// start: soft_hash_collision_kernel_3
/*
    Tiled-over-T_k kernel: one thread per t in a tile for fixed (b,h).
    This improves coalescing for key_buckets and allowed_ext when T_k is large.

    STREAM. The launch below MUST name at::cuda::getCurrentCUDAStream(). A bare
    kernel<<<grid, block>>>(...) goes to the LEGACY DEFAULT STREAM, and a legacy-default-stream
    launch is NOT captured as a node when torch.compile(mode="reduce-overhead") records the
    decode step into a CUDA graph -- so at every graph REPLAY the kernel would not run at all
    and `out` would stay at the torch::zeros the wrapper allocated, i.e. every heavy token
    would be selected from an all-zero score array. Also error-check the launch, because a
    silent failure here looks exactly like a very fast kernel.

    Configuration:
    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid((T_k + threads - 1) / threads, H, B);

    Launch:
    soft_hash_collision_kernel_3<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        key_buckets.data_ptr<int16_t>(),
        allowed_ext.data_ptr<bool>(),
        v_hist.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(Hkv),
        static_cast<int>(L),
        static_cast<int>(R),
        static_cast<int>(T_k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
*/
__global__ void soft_hash_collision_kernel_3(
    const float* __restrict__ q_probs,        // [B,H,1,L,R] contiguous (per query head)
    const int16_t* __restrict__ key_buckets,  // [B,Hkv,L,T_k] contiguous (per kv head)
    const bool* __restrict__ allowed_ext,     // [B,H,1,T_k] contiguous (per query head)
    const float* __restrict__ v_hist,        // [B,Hkv,1,T_k] contiguous (per kv head)
    float* __restrict__ out,                  // [B,H,1,T_k] contiguous (per query head)
    int B, int H, int Hkv, int L, int R, int T_k) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    if (t >= T_k || h >= H || b >= B) return;

    // PER-KV-HEAD: q_probs and allowed/out are per-QUERY-head (H heads); key_buckets and
    // v_hist are per-KV-head (Hkv heads). rep = H / Hkv (GQA group size). The kv head for
    // query head h is h / rep, mirroring the backend's cur_head // gqa_group_size. This is
    // bit-identical to the old repeat_interleave(rep,dim=1) path because that op maps
    // out-head h -> in-head h/rep (pure copy, no arithmetic).
    int rep = H / Hkv;            // exact: caller asserts H % Hkv == 0
    int kv = h / rep;
    int bh = b * H + h;           // per-query-head linear index (q_probs, allowed, out)
    int bh_kv = b * Hkv + kv;     // per-kv-head linear index (key_buckets, v_hist)
    int al_idx = bh * T_k + t;      // out / allowed: per-query-head
    int v_idx = bh_kv * T_k + t;    // v_hist: per-kv-head
    if (!allowed_ext[al_idx]) {
        out[al_idx] = -INFINITY;
        return;
    }

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        int kb_idx = (bh_kv * L + l) * T_k + t;   // key_buckets: per-kv-head
        int r = static_cast<int>(key_buckets[kb_idx]);
        int q_idx = (bh * L + l) * R + r;          // q_probs: per-query-head
        sum += q_probs[q_idx];
    }

    out[al_idx] = sum * v_hist[v_idx];
}
// end: soft_hash_collision_kernel_3
