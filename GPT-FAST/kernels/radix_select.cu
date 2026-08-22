// ---------------------------------------------------------------------------
// Histogram / radix THRESHOLD selection: an exact top-M index select.
//
// Replaces aten::topk(scores, M) -- mbtopk: four full radix passes plus gatherTopK and a
// tail of cleanup ops over the fp32 score array -- with a 3-digit MSB-first radix select
// that streams the score array and materialises nothing.
//
// EXACTNESS (GATE A'): the emitted set is
//     {t : score_t > theta}  U  {exactly (M - #above) arbitrary t with score_t == theta}
// where theta is the EXACT M-th largest score.  So the selected SCORE MULTISET equals
// torch.topk's.  (Index sets cannot be required to match: topk's own tie-break is
// arbitrary and the " hello"xN prompt has 1174-1572 tokens tied at the threshold.)
//
// ORDER KEY.  Floats are compared through the standard monotone uint32 transform
//     key(f) = (bits(f) & 0x80000000) ? ~bits(f) : (bits(f) | 0x80000000)
// which is strictly increasing in f over ALL floats including +-inf.  The scorer writes
// -inf into unfilled cache columns (t >= seq_len); those map to the smallest keys and can
// only be selected if fewer than M valid entries exist -- exactly topk's behaviour.
// We select the M SMALLEST values of  d = 0xFFFFFFFF - key  (largest score first).
//
// DIGITS.  d is split MSB-first into 11 / 11 / 10 bits (2048 / 2048 / 1024 bins):
//     digit1 = d >> 21 ; digit2 = (d - (b1<<21)) >> 10 ; digit3 = d - (b1<<21) - (b2<<10)
// After digit3 every surviving candidate has an IDENTICAL d, hence an identical score, so
// three digits resolve the threshold EXACTLY for any input distribution -- no adaptive
// shifts, no min/max reductions, no data-dependent iteration count.  All bin bounds are
// static, so a pass only needs the previously chosen digits (kept in a device-side ctrl
// block).  Nothing syncs to the host and the launch structure is fixed, so the pipeline is
// CUDA-graph capturable.
//
// WHY STATIC AND NOT ADAPTIVE: the histogram passes are bound by SHARED-MEMORY ATOMIC
// THROUGHPUT, and the atomic count is
//   (#elements / 32) * (avg distinct bins per warp).
// Spreading the values over more bins -- which is what an adaptive min/max-relative shift
// does -- therefore makes the FIRST pass more expensive, not cheaper. Coarse-then-fine is
// the right shape: digit1 lands most values in a handful of bins, digit2 splits those, and
// digit3 only ever sees the survivors of one digit2 bin.
//
// PASSES (each a coalesced streaming read of `scores`):
//   P1 digit1 histogram                           | S1 scan -> b1, c1, need1
//   P2 emit digit1<b1 ; digit2 hist of digit1==b1 | S2 scan -> b2, c2, need2
//   P3 emit digit2<b2 ; digit3 hist of digit2==b2 | S3 scan -> b3, need3
//   P4 emit digit3<b3 ; then exactly need3 of digit3==b3
// Early exit: when a boundary bin's population equals the remaining need, all of it is
// selected and the later passes are skipped by a device-side flag (uniform branch -> the
// blocks retire immediately).
//
// Histogram accumulation and output append are WARP-AGGREGATED (__match_any_sync /
// __activemask): the " hello"xN prompt has only 36-56 distinct scores over 143362 tokens
// (and NaN-poisoned layers exactly one), so un-aggregated same-address atomics would
// serialise 4.6M times per layer.
//
// Every launch uses at::cuda::getCurrentCUDAStream() and is error-checked -- mandatory, not
// stylistic: a launch on the legacy default stream is not captured as a CUDA-graph node and
// so does not run at replay, and an unchecked failed launch is indistinguishable from a very
// fast one.
// ---------------------------------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

#define NBINS 2048
#define SH1 21
#define SH2 10
// Memory-level parallelism: these are streaming reads, so what matters is bytes in flight =
// blocks*threads*UNROLL*4. Buy that parallelism with threads and ILP rather than with more
// blocks: the private-histogram flush costs NB * B*H * 2048 * 4 bytes.
#define UNROLL 4

// per-thread count planes for the deterministic emit; sized for the largest supported THR
#define TCNT_STRIDE 1024

#define CTRL_N 16
#define C_B1 0
#define C_C1 1
#define C_NEED1 2
#define C_B2 3
#define C_C2 4
#define C_NEED2 5
#define C_B3 6
#define C_NEED3 7
#define C_DONE 8       /* digit at which the selection was resolved: 1, 2 or 3 */
#define C_TIENEED 9
#define C_OUTCNT 10
#define C_TIECNT 11
// The RESOLVED threshold, in d-space, written by whichever scan resolves the selection.
// Selected set = { d < A }  U  { exactly TIENEED of  A <= d < A+W }.  |{d < A}| == M-TIENEED.
//   resolved at digit1: A = b1<<21,                      W = 1<<21
//   resolved at digit2: A = (b1<<21)+(b2<<10),            W = 1<<10
//   resolved at digit3: A = (b1<<21)+(b2<<10)+b3,         W = 1
// Expressing all three cases in the same (A, W) form is what lets ONE final pass do all the
// emitting: the digit-2 and digit-3 histogram passes then carry no output path at all, which
// is worth 12 us/layer (their warp_append round-trips were on the critical path).
#define C_A 12
#define C_W 13

__device__ __forceinline__ unsigned f2key(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ unsigned f2d(float f) { return 0xFFFFFFFFu - f2key(f); }

// HMODE selects the histogram accumulation strategy. 1 = PRODUCTION.
//   0 warp-aggregated (__match_any_sync) shared atomicAdd
//   1 plain per-lane shared atomicAdd                       <-- production
//   2 aggregation arithmetic with a plain shared STORE (diagnostic; wrong histogram)
//   3 no histogram at all (diagnostic; isolates the streaming read)
//
// This is the opposite of the textbook advice. When the scores concentrate in few distinct
// bins -- maximal same-address conflict -- plain per-lane atomicAdd still beats warp
// aggregation, because __match_any_sync (MATCH.ANY.U32) is the expensive instruction rather
// than the atomic: replacing only the atomic with a store (HMODE 2) is no cheaper. Shared-
// atomic conflict replays are inexpensive on Hopper. Replicating the shared histogram to cut
// cross-warp contention is a small loss at every replication factor, which confirms the
// atomics were never the bottleneck.
template <int HMODE>
__device__ __forceinline__ void hist_add(unsigned* s_cnt, int bin) {
    if (HMODE == 3) return;
    if (HMODE == 1) { atomicAdd(&s_cnt[bin], 1u); return; }
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, (unsigned)bin);
    unsigned lane = threadIdx.x & 31u;
    if ((peers & ((1u << lane) - 1u)) == 0u) {
        if (HMODE == 2) s_cnt[bin] = (unsigned)__popc(peers);
        else            atomicAdd(&s_cnt[bin], (unsigned)__popc(peers));
    }
}

// One global atomicAdd per warp; returns this lane's reserved slot.
__device__ __forceinline__ unsigned warp_append(unsigned* ctr) {
    unsigned mask = __activemask();
    unsigned lane = threadIdx.x & 31u;
    int leader = __ffs(mask) - 1;
    unsigned rank = __popc(mask & ((1u << lane) - 1u));
    unsigned base = 0u;
    if ((int)lane == leader) base = atomicAdd(ctr, (unsigned)__popc(mask));
    base = __shfl_sync(mask, base, leader);
    return base + rank;
}

// CALIBRATION ONLY: the streaming read + key transform with no histogram, so the
// microbenchmark can separate "how fast can this grid read the array" from the atomics.
__global__ void rs_readonly(const float* __restrict__ scores, int T, int NB,
                            unsigned* __restrict__ h1) {
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x;
    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    const int stride = blockDim.x;
    unsigned acc = 0u;
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * stride < t1; t += UNROLL * stride) {
        float v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = sc[t + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc += f2d(v[u]) >> SH1;
    }
    for (; t < t1; t += stride) acc += f2d(sc[t]) >> SH1;
    if (acc == 0xDEADBEEFu) h1[((long)blk * BH + bh) * NBINS] = acc;   // never taken
}

// ---------------------------------------------------------------------------
// P1, select-only form: digit-1 histogram over an existing score array.
// grid (B*H, NB) -- blockIdx.x is (b,h) so consecutive blocks share a kv head, which keeps
// the int16 bucket stream (fused form) L2-resident.
// ---------------------------------------------------------------------------
template <int HMODE>
__global__ __launch_bounds__(1024) void rs_hist1(const float* __restrict__ scores, int T, int NB,
                         unsigned* __restrict__ h1) {
    __shared__ unsigned s_cnt[NBINS];
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x;
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) s_cnt[i] = 0u;
    __syncthreads();

    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    const int stride = blockDim.x;
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * stride < t1; t += UNROLL * stride) {
        float v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = sc[t + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) hist_add<HMODE>(s_cnt, (int)(f2d(v[u]) >> SH1));
    }
    for (; t < t1; t += stride) hist_add<HMODE>(s_cnt, (int)(f2d(sc[t]) >> SH1));
    __syncthreads();
    unsigned* o = h1 + ((long)blk * BH + bh) * NBINS;
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) o[i] = s_cnt[i];
}

// ---------------------------------------------------------------------------
// SCAN. One block per (b,h): reduce the NB private histograms, prefix-scan the 2048 bins
// ascending, locate the boundary bin. Warp-shuffle scan (2 __syncthreads) rather than a
// shared Hillis-Steele (16 __syncthreads): with only B*H blocks there is no other work on
// the SM to hide sync latency behind, so the barrier count dominates.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned warp_incl_scan(unsigned v) {
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned n = __shfl_up_sync(0xFFFFFFFFu, v, off);
        if ((threadIdx.x & 31u) >= (unsigned)off) v += n;
    }
    return v;
}

// Reduce NB private histograms into s_cnt, then find the smallest bin b with
// cum(bins <= b) >= need. Returns b in *pb (-1 if the total is < need) and cum(bins < b) in
// *pc, valid on thread 0.
__device__ __forceinline__ void reduce_scan_find(const unsigned* __restrict__ h, int NB,
                                                 int BH, int bh, unsigned need,
                                                 unsigned* s_cnt, unsigned* s_warp,
                                                 unsigned* s_incl, unsigned* s_tot,
                                                 int* s_found, int* pb, unsigned* pc) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int nw = nthr >> 5, w = tid >> 5, lane = tid & 31;
    const int per = NBINS / nthr;
    for (int i = tid; i < NBINS; i += nthr) {
        unsigned c = 0u;
#pragma unroll 4
        for (int k = 0; k < NB; ++k) c += h[((long)k * BH + bh) * NBINS + i];
        s_cnt[i] = c;
    }
    if (tid == 0) *s_found = -1;
    __syncthreads();

    unsigned local = 0u;
    for (int i = 0; i < per; ++i) local += s_cnt[tid * per + i];
    unsigned incl = warp_incl_scan(local);
    if (lane == 31) s_warp[w] = incl;
    __syncthreads();
    if (tid < 32) {                      // exclusive scan of the per-warp totals, in warp 0
        unsigned v = (tid < nw) ? s_warp[tid] : 0u;
        unsigned sc = warp_incl_scan(v);
        s_warp[tid] = sc - v;
        if (tid == nw - 1) *s_tot = sc;
    }
    __syncthreads();
    incl += s_warp[w];
    s_incl[tid] = incl;                  // BLOCK-inclusive scan -> O(1) boundary base
    unsigned off = incl - local;
    if (off < need && incl >= need) *s_found = tid;
    __syncthreads();

    if (tid != 0) return;
    int f = *s_found;
    if (f < 0) { *pb = -1; *pc = *s_tot; return; }
    unsigned base = s_incl[f];
    for (int i = 0; i < per; ++i) base -= s_cnt[f * per + i];
    int b = f * per + per - 1;
    unsigned c = base;
    for (int i = 0; i < per; ++i) {
        int bin = f * per + i;
        if (c + s_cnt[bin] >= need) { b = bin; break; }
        c += s_cnt[bin];
    }
    *pb = b; *pc = c;
}

__global__ void rs_scan1(const unsigned* __restrict__ h1, int NB, int BH, int M, int Mout,
                         unsigned* __restrict__ ctrl, int* __restrict__ out) {
    __shared__ unsigned s_cnt[NBINS];
    __shared__ unsigned s_warp[32];
    __shared__ unsigned s_incl[1024];
    __shared__ unsigned s_tot;
    __shared__ int s_found;
    const int bh = blockIdx.x;
    for (int i = threadIdx.x; i < Mout; i += blockDim.x) out[(long)bh * Mout + i] = -1;
    int b1 = -1; unsigned c1 = 0u;
    reduce_scan_find(h1, NB, BH, bh, (unsigned)M, s_cnt, s_warp, s_incl, &s_tot, &s_found, &b1, &c1);
    if (threadIdx.x == 0) {
        unsigned* C = ctrl + (long)bh * CTRL_N;
        C[C_OUTCNT] = 0u; C[C_TIECNT] = 0u;
        if (b1 < 0) {                       // fewer than M valid entries: take them all
            C[C_B1] = (unsigned)NBINS; C[C_C1] = c1; C[C_NEED1] = 0u;
            C[C_DONE] = 1u; C[C_TIENEED] = (unsigned)M;
            C[C_A] = 0xFFFFFFFFu; C[C_W] = 1u;
        } else {
            unsigned need1 = (unsigned)M - c1;
            C[C_B1] = (unsigned)b1; C[C_C1] = c1; C[C_NEED1] = need1;
            if (s_cnt[b1] == need1) {          // the whole boundary bin is needed
                C[C_DONE] = 1u; C[C_TIENEED] = need1;
                C[C_A] = (unsigned)b1 << SH1; C[C_W] = 1u << SH1;
            } else {
                C[C_DONE] = 0u; C[C_TIENEED] = 0u;
            }
        }
    }
}

__global__ void rs_scan2(const unsigned* __restrict__ h2, int NB, int BH,
                         unsigned* __restrict__ ctrl) {
    __shared__ unsigned s_cnt[NBINS];
    __shared__ unsigned s_warp[32];
    __shared__ unsigned s_incl[1024];
    __shared__ unsigned s_tot;
    __shared__ int s_found;
    const int bh = blockIdx.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    if (C[C_DONE]) return;
    const unsigned need1 = C[C_NEED1];
    int b2 = -1; unsigned c2 = 0u;
    reduce_scan_find(h2, NB, BH, bh, need1, s_cnt, s_warp, s_incl, &s_tot, &s_found, &b2, &c2);
    if (threadIdx.x == 0) {
        const unsigned d1base = C[C_B1] << SH1;
        if (b2 < 0) {   // unreachable: the digit-2 histogram counts exactly cnt[b1] >= need1
            C[C_B2] = 0u; C[C_C2] = 0u; C[C_NEED2] = need1;
            C[C_DONE] = 2u; C[C_TIENEED] = need1;
            C[C_A] = d1base; C[C_W] = 1u << SH2;
        } else {
            unsigned need2 = need1 - c2;
            C[C_B2] = (unsigned)b2; C[C_C2] = c2; C[C_NEED2] = need2;
            if (s_cnt[b2] == need2) {
                C[C_DONE] = 2u; C[C_TIENEED] = need2;
                C[C_A] = d1base + ((unsigned)b2 << SH2); C[C_W] = 1u << SH2;
            }
        }
    }
}

__global__ void rs_scan3(const unsigned* __restrict__ h3, int NB, int BH,
                         unsigned* __restrict__ ctrl) {
    __shared__ unsigned s_cnt[NBINS];
    __shared__ unsigned s_warp[32];
    __shared__ unsigned s_incl[1024];
    __shared__ unsigned s_tot;
    __shared__ int s_found;
    const int bh = blockIdx.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    if (C[C_DONE]) return;
    const unsigned need2 = C[C_NEED2];
    int b3 = -1; unsigned c3 = 0u;
    reduce_scan_find(h3, NB, BH, bh, need2, s_cnt, s_warp, s_incl, &s_tot, &s_found, &b3, &c3);
    if (threadIdx.x == 0) {
        unsigned bb3 = (b3 < 0) ? 0u : (unsigned)b3;
        C[C_B3] = bb3;
        C[C_NEED3] = (b3 < 0) ? need2 : (need2 - c3);
        C[C_DONE] = 3u;
        C[C_TIENEED] = C[C_NEED3];
        // digit3 has unit resolution, so A is the EXACT threshold key and the tie window is 1
        C[C_A] = (C[C_B1] << SH1) + (C[C_B2] << SH2) + bb3;
        C[C_W] = 1u;
    }
}

// ---------------------------------------------------------------------------
// EMIT/HISTOGRAM passes. `tie_open` is refreshed ONCE PER UNROLLED GROUP, not per element:
// it is a volatile (L2) load of the shared tie counter, and on inputs where every token is
// a tie (the NaN-poisoned 140K layers put all 143362 tokens in one bin) a per-element
// refresh cost ~12 us/pass in pure L2 traffic.
// ---------------------------------------------------------------------------
template <int HMODE>
__global__ __launch_bounds__(1024) void rs_pass2(const float* __restrict__ scores, int T, int NB, int Mout,
                         unsigned* __restrict__ ctrl, int* __restrict__ out,
                         unsigned* __restrict__ h2) {
    __shared__ unsigned s_cnt[NBINS];
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    if (C[C_DONE]) return;                        // resolved at digit 1
    const unsigned b1 = C[C_B1];
    const unsigned d1base = b1 << SH1;
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) s_cnt[i] = 0u;
    __syncthreads();

    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    const int stride = blockDim.x;
#define P2_ONE(dd)                                                           \
    do {                                                                     \
        if (((dd) >> SH1) == b1) hist_add<HMODE>(s_cnt, (int)(((dd) - d1base) >> SH2)); \
    } while (0)
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * stride < t1; t += UNROLL * stride) {
        unsigned d[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) d[u] = f2d(sc[t + u * stride]);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) P2_ONE(d[u]);
    }
    for (; t < t1; t += stride) P2_ONE(f2d(sc[t]));
#undef P2_ONE
    __syncthreads();
    unsigned* o = h2 + ((long)blk * BH + bh) * NBINS;
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) o[i] = s_cnt[i];
}

template <int HMODE>
__global__ __launch_bounds__(1024) void rs_pass3(const float* __restrict__ scores, int T, int NB, int Mout,
                         unsigned* __restrict__ ctrl, int* __restrict__ out,
                         unsigned* __restrict__ h3) {
    __shared__ unsigned s_cnt[NBINS];
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    if (C[C_DONE]) return;                        // resolved at digit 1 or 2
    const unsigned b1 = C[C_B1], b2 = C[C_B2];
    const unsigned d1base = b1 << SH1;
    const unsigned d2base = d1base + (b2 << SH2);
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) s_cnt[i] = 0u;
    __syncthreads();

    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    const int stride = blockDim.x;
#define P3_ONE(dd)                                                             \
    do {                                                                       \
        if (((dd) >> SH1) == b1 && (((dd) - d1base) >> SH2) == b2)               \
            hist_add<HMODE>(s_cnt, (int)((dd) - d2base));                       \
    } while (0)
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * stride < t1; t += UNROLL * stride) {
        unsigned d[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) d[u] = f2d(sc[t + u * stride]);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) P3_ONE(d[u]);
    }
    for (; t < t1; t += stride) P3_ONE(f2d(sc[t]));
#undef P3_ONE
    __syncthreads();
    unsigned* o = h3 + ((long)blk * BH + bh) * NBINS;
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) o[i] = s_cnt[i];
}
// ===========================================================================
// EMIT. Two interchangeable implementations, selected at runtime by `det`:
//
// det=0 (FAST, 1 pass): stage the "above" group in shared memory and reserve with ONE global
//   atomicAdd per block; emit ties with a bounded global atomicAdd. NON-DETERMINISTIC: which
//   of the tokens TIED AT THE THRESHOLD are kept depends on atomic arrival order, so two runs
//   on the same input can pick a different (equally exact) tie subset. The selected SCORE
//   MULTISET is identical either way -- but the greedy token stream is then not reproducible,
//   which also makes the "compiled == eager" regression gate unusable.
//
// det=1 (DETERMINISTIC, 2 passes + 1 tiny kernel): count per thread, exclusive-prefix over
//   (block, thread), then emit at a slot that is a pure function of (block, thread,
//   iteration). Ties are the first TIENEED in that same fixed order. Reproducible run to run
//   and identical eager vs CUDA-graph replay. Costs one extra streaming pass over `scores`.
// ===========================================================================
__global__ __launch_bounds__(1024) void rs_pass4(
        const float* __restrict__ scores, int T, int NB, int Mout, int SBUF,
        unsigned* __restrict__ ctrl, int* __restrict__ out) {
    extern __shared__ int s_buf[];
    __shared__ unsigned s_n;
    __shared__ unsigned s_base;
    const int bh = blockIdx.x, blk = blockIdx.y;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    const unsigned A = C[C_A], W = C[C_W], tieneed = C[C_TIENEED];
    if (threadIdx.x == 0) s_n = 0u;
    __syncthreads();

    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    const int stride = blockDim.x;
    int* obase = out + (long)bh * Mout;
    volatile unsigned* tiectr = (volatile unsigned*)&C[C_TIECNT];

#define P4_ONE(tt, dd, topen)                                                 \
    do {                                                                      \
        if ((dd) < A) {                                                       \
            unsigned k = atomicAdd(&s_n, 1u);                                 \
            if (k < (unsigned)SBUF) s_buf[k] = (tt);                           \
            else {                                                             \
                unsigned pos = warp_append(&C[C_OUTCNT]);                      \
                if (pos < (unsigned)Mout) obase[pos] = (tt);                    \
            }                                                                  \
        } else if ((topen) && (dd) - A < W) {                                  \
            unsigned k = warp_append(&C[C_TIECNT]);                            \
            if (k < tieneed) obase[Mout - 1 - k] = (tt);                        \
        }                                                                      \
    } while (0)
    int t = t0 + threadIdx.x;
    bool topen = (tieneed > 0u);
    unsigned tc = 0u;
    for (; t + (UNROLL - 1) * stride < t1; t += UNROLL * stride) {
        // Probe the tie counter FIRST so its L2 latency overlaps the score loads, and consume
        // it only AFTER this group's appends: making `topen` depend on a volatile load in the
        // same iteration put a ~400-cycle round trip on the loop's critical path (+9 us/pass).
        // One group of staleness costs at most one extra group of bounded appends; the
        // `k < tieneed` guard is what makes the count exact.
        if (topen) tc = *tiectr;
        unsigned d[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) d[u] = f2d(sc[t + u * stride]);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) P4_ONE(t + u * stride, d[u], topen);
        if (topen) topen = (tc < tieneed);
    }
    for (; t < t1; t += stride) P4_ONE(t, f2d(sc[t]), topen && (*tiectr < tieneed));
#undef P4_ONE

    __syncthreads();
    unsigned n = min(s_n, (unsigned)SBUF);
    if (threadIdx.x == 0) s_base = atomicAdd(&C[C_OUTCNT], n);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned pos = s_base + i;
        if (pos < (unsigned)Mout) obase[pos] = s_buf[i];
    }
}

// ---- deterministic emit, phase A: per-thread and per-block counts -----------------------
__global__ __launch_bounds__(1024) void rs_count4(const float* __restrict__ scores, int T,
                                                  int NB, unsigned* __restrict__ ctrl,
                                                  unsigned* __restrict__ tcnt,
                                                  unsigned* __restrict__ bcnt) {
    __shared__ unsigned s_red[64];
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x, nthr = blockDim.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    const unsigned A = C[C_A], W = C[C_W];
    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    unsigned na = 0u, nt = 0u;
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * nthr < t1; t += UNROLL * nthr) {
        unsigned d[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) d[u] = f2d(sc[t + u * nthr]);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) { if (d[u] < A) ++na; else if (d[u] - A < W) ++nt; }
    }
    for (; t < t1; t += nthr) {
        unsigned d = f2d(sc[t]);
        if (d < A) ++na; else if (d - A < W) ++nt;
    }
    unsigned* tb = tcnt + ((long)blk * BH + bh) * 2 * TCNT_STRIDE;
    tb[threadIdx.x] = na;
    tb[TCNT_STRIDE + threadIdx.x] = nt;

    const int nw = nthr >> 5, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned sa = na, st = nt;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        sa += __shfl_down_sync(0xFFFFFFFFu, sa, o);
        st += __shfl_down_sync(0xFFFFFFFFu, st, o);
    }
    if (lane == 0) { s_red[w] = sa; s_red[32 + w] = st; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned ta = 0u, tt = 0u;
        for (int i = 0; i < nw; ++i) { ta += s_red[i]; tt += s_red[32 + i]; }
        bcnt[((long)blk * BH + bh) * 2 + 0] = ta;
        bcnt[((long)blk * BH + bh) * 2 + 1] = tt;
    }
}

// ---- deterministic emit, phase B: exclusive prefix over the NB blocks -------------------
__global__ void rs_prefix4(int NB, int BH, unsigned* __restrict__ bcnt) {
    const int bh = blockIdx.x;
    if (threadIdx.x) return;
    unsigned ca = 0u, ct = 0u;
    for (int k = 0; k < NB; ++k) {
        long o = ((long)k * BH + bh) * 2;
        unsigned a = bcnt[o + 0], tt = bcnt[o + 1];
        bcnt[o + 0] = ca; bcnt[o + 1] = ct;
        ca += a; ct += tt;
    }
}

// ---- deterministic emit, phase C -------------------------------------------------------
__global__ __launch_bounds__(1024) void rs_emit4(const float* __restrict__ scores, int T,
                                                 int NB, int Mout, unsigned* __restrict__ ctrl,
                                                 const unsigned* __restrict__ tcnt,
                                                 const unsigned* __restrict__ bcnt,
                                                 int* __restrict__ out) {
    __shared__ unsigned s_wa[32];
    __shared__ unsigned s_wt[32];
    const int bh = blockIdx.x, blk = blockIdx.y, BH = gridDim.x, nthr = blockDim.x;
    unsigned* C = ctrl + (long)bh * CTRL_N;
    const unsigned A = C[C_A], W = C[C_W], tieneed = C[C_TIENEED];
    const unsigned* tb = tcnt + ((long)blk * BH + bh) * 2 * TCNT_STRIDE;
    const int nw = nthr >> 5, w = threadIdx.x >> 5, lane = threadIdx.x & 31;

    unsigned va = tb[threadIdx.x], vt = tb[TCNT_STRIDE + threadIdx.x];
    unsigned ia = warp_incl_scan(va), it = warp_incl_scan(vt);
    if (lane == 31) { s_wa[w] = ia; s_wt[w] = it; }
    __syncthreads();
    if (threadIdx.x < 32) {
        unsigned x = (threadIdx.x < nw) ? s_wa[threadIdx.x] : 0u;
        unsigned y = (threadIdx.x < nw) ? s_wt[threadIdx.x] : 0u;
        s_wa[threadIdx.x] = warp_incl_scan(x) - x;
        s_wt[threadIdx.x] = warp_incl_scan(y) - y;
    }
    __syncthreads();
    unsigned ka = bcnt[((long)blk * BH + bh) * 2 + 0] + (ia - va) + s_wa[w];
    unsigned kt = bcnt[((long)blk * BH + bh) * 2 + 1] + (it - vt) + s_wt[w];

    const float* sc = scores + (long)bh * (long)T;
    const int per = (T + NB - 1) / NB;
    const int t0 = blk * per, t1 = min(T, t0 + per);
    int* obase = out + (long)bh * Mout;
#define E4_ONE(tt, dd)                                                        \
    do {                                                                      \
        if ((dd) < A) { if (ka < (unsigned)Mout) obase[ka] = (tt); ++ka; }       \
        else if ((dd) - A < W) {                                               \
            if (kt < tieneed) obase[Mout - 1 - kt] = (tt);                      \
            ++kt;                                                               \
        }                                                                       \
    } while (0)
    int t = t0 + threadIdx.x;
    for (; t + (UNROLL - 1) * nthr < t1; t += UNROLL * nthr) {
        unsigned d[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) d[u] = f2d(sc[t + u * nthr]);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) E4_ONE(t + u * nthr, d[u]);
    }
    for (; t < t1; t += nthr) E4_ONE(t, f2d(sc[t]));
#undef E4_ONE
}

// ---------------------------------------------------------------------------
// HOST SIDE
// ---------------------------------------------------------------------------
static inline int64_t ws_ints(int64_t NB, int64_t BH) {
    // 3 digit histograms + ctrl + (deterministic emit) 2 per-thread count planes and
    // 2 per-block counters
    return 3 * NB * BH * NBINS + BH * CTRL_N + 2 * NB * BH * TCNT_STRIDE + 2 * NB * BH;
}
int64_t rs_workspace_ints(int64_t NB, int64_t BH) { return ws_ints(NB, BH); }

struct WS { unsigned *h1, *h2, *h3, *ctrl, *tcnt, *bcnt; };

static WS split_ws(const torch::Tensor& ws, int64_t NB, int64_t BH) {
    unsigned* p = reinterpret_cast<unsigned*>(ws.data_ptr<int32_t>());
    int64_t blk = NB * BH * NBINS;
    WS w; w.h1 = p; w.h2 = p + blk; w.h3 = p + 2 * blk; w.ctrl = p + 3 * blk;
    w.tcnt = w.ctrl + BH * CTRL_N;
    w.bcnt = w.tcnt + 2 * NB * BH * TCNT_STRIDE;
    return w;
}

// stages bitmask: 1=hist1 2=scan1 4=pass2 8=scan2 16=pass3 32=scan3 64=pass4 128=readonly
static void run_select(const float* scores, int T, int BH, int M, int Mout, int NB,
                       int THR, int STHR, WS w, int* out, cudaStream_t st, int stages,
                       int hmode, int det) {
    dim3 g((unsigned)BH, (unsigned)NB, 1), blk((unsigned)THR, 1, 1);
    dim3 gs((unsigned)BH, 1, 1), blks((unsigned)STHR, 1, 1);
#define HDISPATCH(K, ...)                                            \
    do {                                                             \
        if (hmode == 0)      K<0><<<g, blk, 0, st>>>(__VA_ARGS__);     \
        else if (hmode == 1) K<1><<<g, blk, 0, st>>>(__VA_ARGS__);     \
        else if (hmode == 2) K<2><<<g, blk, 0, st>>>(__VA_ARGS__);     \
        else                 K<3><<<g, blk, 0, st>>>(__VA_ARGS__);     \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                               \
    } while (0)
    if (stages & 128) {
        rs_readonly<<<g, blk, 0, st>>>(scores, T, NB, w.h1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stages & 1) HDISPATCH(rs_hist1, scores, T, NB, w.h1);
    if (stages & 2) {
        rs_scan1<<<gs, blks, 0, st>>>(w.h1, NB, BH, M, Mout, w.ctrl, out);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stages & 4) HDISPATCH(rs_pass2, scores, T, NB, Mout, w.ctrl, out, w.h2);
    if (stages & 8) {
        rs_scan2<<<gs, blks, 0, st>>>(w.h2, NB, BH, w.ctrl);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stages & 16) HDISPATCH(rs_pass3, scores, T, NB, Mout, w.ctrl, out, w.h3);
    if (stages & 32) {
        rs_scan3<<<gs, blks, 0, st>>>(w.h3, NB, BH, w.ctrl);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stages & 64) {
        if (det) {
            rs_count4<<<g, blk, 0, st>>>(scores, T, NB, w.ctrl, w.tcnt, w.bcnt);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            rs_prefix4<<<gs, dim3(32, 1, 1), 0, st>>>(NB, BH, w.bcnt);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            rs_emit4<<<g, blk, 0, st>>>(scores, T, NB, Mout, w.ctrl, w.tcnt, w.bcnt, out);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            // shared staging buffer for the "above" group: one block can never see more than
            // M of them, so SBUF = min(M, 8192) makes overflow impossible in every shipped
            // config (M <= 4104), and the overflow path in P4_ONE keeps it correct regardless.
            int SBUF = M < 8192 ? M : 8192;
            if (SBUF < 1) SBUF = 1;
            unsigned shb4 = (unsigned)SBUF * 4u;
            if (shb4 > 48u * 1024u)
                cudaFuncSetAttribute(rs_pass4, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     101376);
            rs_pass4<<<g, blk, shb4, st>>>(scores, T, NB, Mout, SBUF, w.ctrl, out);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
#undef HDISPATCH
}

static void common_checks(const torch::Tensor& out, const torch::Tensor& ws,
                          int64_t NB, int64_t BH, int64_t STHR, int64_t THR) {
    TORCH_CHECK(out.is_cuda() && ws.is_cuda(), "cuda tensors required");
    TORCH_CHECK(out.scalar_type() == torch::kInt && ws.scalar_type() == torch::kInt, "int32");
    TORCH_CHECK(out.is_contiguous() && ws.is_contiguous(), "contiguous required");
    TORCH_CHECK(THR >= 32 && THR <= TCNT_STRIDE && (THR % 32) == 0,
                "threads must be 32..1024 and a multiple of 32");
    TORCH_CHECK(STHR >= 32 && STHR <= 1024 && (NBINS % STHR) == 0 &&
                ((STHR & (STHR - 1)) == 0), "scan_threads must be a power of two dividing 2048");
    TORCH_CHECK(ws.numel() >= ws_ints(NB, BH), "workspace too small");
}

// SELECT-ONLY: exact top-M indices of scores[B,H,T] -> out[B,H,Mout] int32 (-1 padded).
void radix_select(torch::Tensor scores, torch::Tensor out, torch::Tensor ws,
                  int64_t M, int64_t NB, int64_t THR, int64_t STHR, int64_t stages,
                  int64_t hmode, int64_t det) {
    TORCH_CHECK(scores.is_cuda() && scores.is_contiguous(), "scores cuda+contiguous");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat, "scores must be fp32");
    TORCH_CHECK(scores.dim() == 3 && out.dim() == 3, "scores [B,H,T], out [B,H,M]");
    int BH = (int)(scores.size(0) * scores.size(1));
    common_checks(out, ws, NB, BH, STHR, THR);
    TORCH_CHECK(M <= out.size(2), "M must be <= out.size(2)");
    WS w = split_ws(ws, NB, BH);
    run_select(scores.data_ptr<float>(), (int)scores.size(2), BH, (int)M, (int)out.size(2),
               (int)NB, (int)THR, (int)STHR, w, out.data_ptr<int32_t>(),
               at::cuda::getCurrentCUDAStream(), (int)stages, (int)hmode, (int)det);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radix_select", &radix_select, "exact top-M via 3-digit MSB-first radix select",
          pybind11::arg("scores"), pybind11::arg("out"), pybind11::arg("ws"),
          pybind11::arg("M"), pybind11::arg("NB"), pybind11::arg("THR"),
          pybind11::arg("STHR"), pybind11::arg("stages") = 127,
          pybind11::arg("hmode") = 1, pybind11::arg("det") = 0);
    m.def("workspace_ints", &rs_workspace_ints, "workspace size in int32 elements");
}
