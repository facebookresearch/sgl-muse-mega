#pragma once
// The sampled step (-DMK_DFLASH_SAMPLE=1): DFlash2 at temperature > 0 as SGLang's DFlash2 worker runs it.
//
// The draft head walks the selector lattice by sampling: slot e draws its candidate from softmax(score row / T) over
// the row's 16 candidates (T clamped at 1e-5) with one uniform per slot, and keeps that row's probabilities q_e
// (SGLang's _selector_walk_kernel). The verify gathers the fp32 softcapped logits of every row on every rank (the
// verify graph's lm_head and softcap), and sample_accept decides the step from them as SGLang's chain sampling does
// (dspark_accept's accept_sampling and chain_speculative_sampling_triton):
//   target rows   P_r = softmax(logits_r / T) restricted to the top-k (when top_k < vocab) and then to the top-p set,
//                 renormalized (build_speculative_verify_target_probs: top-k first, top-p within it)
//   accept        block verification (Algorithm 2 of arXiv:2403.10444, SGLang's --speculative-use-block-verification)
//                 or token-wise rejection, with P_(e-1)(d_e) / q_(e-1)(d_e) for each draft d_e
//   bonus         a sample of the accepted row's residual max(s P_r - q_r, 0) (s: the accepted prefix's probability;
//                 P_r alone when every draft is accepted), by its CDF in vocabulary order
// Every rank holds the same logits and draws the same uniforms (Philox4x32-10 keyed by the request's seed, counted by
// the step), and every reduction runs in a fixed order or in integers, so every rank decides the same tokens: nothing
// is broadcast. A row's weights w = exp(logits / T - max) (one fast exp per element, the temperature as a reciprocal:
// within an ulp or two of SGLang's fp32 division and exp) are held in place; the top-k threshold and the top-p pivot
// are exact radix selections over their order image (counts, and masses in 2^-40 fixed point), and equal weights at
// the top-k boundary are all kept.
//
// sample_accept: one cluster of kCtas CTAs per live row holds the row's scaled logits in shared memory (a column slice
// each) and writes the row's statistics (max, normalizer, keep threshold), its candidates' probabilities and its
// per-chunk masses; the last CTA to finish decides the accept, samples the bonus and runs verify_accept's commit and
// feeds (verify_accept.cu, accept_post).
#include <cstdint>

#include "dflash/common.cuh"

namespace mk::dflash::sample {

inline constexpr uint32_t kCtas = 8;      // CTAs per row: one cluster
inline constexpr uint32_t kBins = 2048;   // radix digits of 11, 11 and 10 bits
inline constexpr uint32_t kChunk = 1024;  // the bonus's CDF walk: chunk masses, then one chunk's elements
inline constexpr uint32_t kRows = 16;
inline constexpr float kMassScale = 0x1p40f;  // fixed point of the top-p masses (each <= 1: 2^18 of them fit 64 bits)
inline constexpr float kDraftMinTemp = 1e-5f;

// the sampling words (int64 [kSampWords], set at a request's start)
inline constexpr uint32_t kSampSeed = 0;    // the request's 64-bit seed
inline constexpr uint32_t kSampStream = 1;  // the run's start position: counts the uniforms apart across restarts
inline constexpr uint32_t kSampTopK = 2;    // top_k (>= vocab: no top-k)
inline constexpr uint32_t kSampTemp = 3;    // fp32 temperature bits | fp32 top_p bits << 32
inline constexpr uint32_t kSampWords = 4;
inline constexpr uint32_t kPurposeWalk = 0, kPurposeAccept = 1;
inline constexpr uint32_t kFinalUniform = 15;  // the accept's coins are uniforms 0 .. gamma - 1, the bonus's this one

// per-row statistics (fp32 words): the max of the scaled row, the kept set's inverse normalizer, the keep key's bits
inline constexpr uint32_t kStatMax = 0, kStatInvZ = 1, kStatKey = 2, kStatWords = 3;

struct Params {
    float temperature;
    float top_p;
    uint32_t top_k;
};

__device__ __forceinline__ Params params(const int64_t* samp) {
    const uint64_t tp = static_cast<uint64_t>(samp[kSampTemp]);
    return {__uint_as_float(static_cast<uint32_t>(tp)), __uint_as_float(static_cast<uint32_t>(tp >> 32)),
            static_cast<uint32_t>(min(samp[kSampTopK], static_cast<int64_t>(UINT32_MAX)))};
}

// Philox4x32-10 (Salmon et al., SC'11); host-callable for the CPU tests
__host__ __device__ __forceinline__ uint4 philox(uint4 c, uint2 k) {
#pragma unroll
    for (uint32_t r = 0; r < 10; ++r) {
        const uint64_t p0 = static_cast<uint64_t>(0xD2511F53u) * c.x, p1 = static_cast<uint64_t>(0xCD9E8D57u) * c.z;
        c = make_uint4(static_cast<uint32_t>(p1 >> 32) ^ c.y ^ k.x, static_cast<uint32_t>(p1),
                       static_cast<uint32_t>(p0 >> 32) ^ c.w ^ k.y, static_cast<uint32_t>(p0));
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}

// uniform i in [0, 1) of a step's purpose: 24 random bits of output word i % 4 of counter {iter, stream, purpose, i /
// 4}
__host__ __device__ __forceinline__ float uniform(const int64_t* samp, uint64_t iter, uint32_t purpose, uint32_t i) {
    const uint64_t seed = static_cast<uint64_t>(samp[kSampSeed]);
    const uint4 r =
        philox(make_uint4(static_cast<uint32_t>(iter), static_cast<uint32_t>(samp[kSampStream]), purpose, i / 4),
               make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)));
    const uint32_t w[4] = {r.x, r.y, r.z, r.w};
    return static_cast<float>(w[i % 4] >> 8) * 0x1p-24f;
}

// ---- the draft head's sampled walk (head.cu, MK_DFLASH_SAMPLE) ----

struct WalkArgs {
    const int64_t* samp;
    float* q;       // [gamma][16] each slot's probabilities over its candidates
    int32_t* path;  // [gamma] the candidate index each slot drew
};

// Slot e reads the score row of slot e - 1's draw (row 0 for slot 0: every row of slot 0 has the anchor as its
// predecessor); lanes 0..15 hold the row's 16 candidates (lanes 16..31 the same: width-16 shuffles). ids[0] = the
// anchor, ids[1 + e] = slot e's draft. s_scores / s_cand: smem for [gamma][16][16] floats and [gamma][16] ids.
__device__ __forceinline__ void walk(const WalkArgs& w, const int32_t* cand, const float* scores, const int64_t* words,
                                     uint32_t gamma, float* s_scores, int32_t* s_cand, int64_t* ids) {
    const uint32_t t = threadIdx.x;
    for (uint32_t i = t; i < gamma * kTopK * kTopK; i += kThreads)
        s_scores[i] = __ldcg(scores + i);
    if (t < gamma * kTopK) s_cand[t] = __ldcg(cand + t);
    __syncthreads();
    if (t < 32) {
        const uint32_t c = t % kTopK;
        const uint64_t iter = static_cast<uint64_t>(words[kWordIter]);
        const float temp = fmaxf(params(w.samp).temperature, kDraftMinTemp);
        if (t == 0) ids[0] = words[kWordAnchor];
        uint32_t prev = 0;
        for (uint32_t e = 0; e < gamma; ++e) {
            const float x = __fdiv_rn(s_scores[(e * kTopK + prev) * kTopK + c], temp);
            float m = x;
#pragma unroll
            for (uint32_t o = 8; o >= 1; o >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o, kTopK));
            const float ex = expf(__fsub_rn(x, m));
            float sum = ex;
#pragma unroll
            for (uint32_t o = 8; o >= 1; o >>= 1)
                sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o, kTopK));
            const float pr = __fdiv_rn(ex, sum);
            float cum = pr;
#pragma unroll
            for (uint32_t o = 1; o < kTopK; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, cum, o, kTopK);
                if (c >= o) cum = __fadd_rn(cum, y);
            }
            const float u = uniform(w.samp, iter, kPurposeWalk, e);
            const uint32_t idx = min(__popc(__ballot_sync(0xffffffffu, u >= cum) & 0xFFFFu), kTopK - 1);
            if (t < kTopK) w.q[e * kTopK + c] = pr;
            if (t == 0) {
                w.path[e] = static_cast<int32_t>(idx);
                ids[1 + e] = s_cand[e * kTopK + idx];
            }
            prev = idx;
        }
    }
    __syncthreads();
}

// ---- sample_accept ----

struct Args {
    const float* logits;  // [16][vocab] the softcapped fp32 logits (every rank's identical copy)
    const int64_t* samp;
    const int32_t* cand;  // [gamma][16] the head's candidates
    const float* q;       // [gamma][16] the walk's probabilities
    const int32_t* path;  // [gamma] the walk's draws
    float* stats;         // [16][kStatWords]
    float* pc;            // [16][16] P_r of row r's candidates (r < gamma)
    float* chunk;         // [16][kCtas x chunks per CTA] P_r's mass per chunk
    uint32_t vocab;
    uint32_t slice;  // columns per CTA: vocab / kCtas rounded up to 4
    uint32_t gamma;  // live rows - 1
    uint32_t block;  // 1: block verification, 0: token-wise
};

__host__ __device__ __forceinline__ uint32_t slice_of(uint32_t vocab) {
    return ((vocab + kCtas - 1) / kCtas + 3) / 4 * 4;
}

__host__ __device__ __forceinline__ uint32_t chunks_per_cta(uint32_t slice) {
    return (slice + kChunk - 1) / kChunk;
}

// the order image of an fp32 value (larger value, larger key)
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// barrier.cluster with the compiler's memory order: this CTA's smem stores land before it, the peers' reads after it
__device__ __forceinline__ void cluster_barrier() {
    asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

// The word at `local` in every CTA of the cluster, in rank order: the kCtas loads issue back to back, then wait once.
__device__ __forceinline__ void cluster_load(const uint32_t* local, uint32_t (&v)[kCtas]) {
    const uint32_t a = ::ptx::to_shared(local);
#pragma unroll
    for (uint32_t r = 0; r < kCtas; ++r)
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v[r]) : "r"(::ptx::mapa_shared_cluster(a, r)));
}

__device__ __forceinline__ void cluster_load(const uint64_t* local, uint64_t (&v)[kCtas]) {
    const uint32_t a = ::ptx::to_shared(local);
#pragma unroll
    for (uint32_t r = 0; r < kCtas; ++r)
        asm volatile("ld.shared::cluster.u64 %0, [%1];" : "=l"(v[r]) : "r"(::ptx::mapa_shared_cluster(a, r)));
}

// the CTA's sum / max in every thread, in a fixed order (s_red: kThreads / 32 floats)
__device__ __forceinline__ float block_sum(float v, float* s_red) {
#pragma unroll
    for (uint32_t o = 16; o >= 1; o >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x % 32 == 0) s_red[threadIdx.x / 32] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < kThreads / 32; ++w)
        s = __fadd_rn(s, s_red[w]);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float v, float* s_red) {
#pragma unroll
    for (uint32_t o = 16; o >= 1; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x % 32 == 0) s_red[threadIdx.x / 32] = v;
    __syncthreads();
    float s = -INFINITY;
#pragma unroll
    for (uint32_t w = 0; w < kThreads / 32; ++w)
        s = fmaxf(s, s_red[w]);
    __syncthreads();
    return s;
}

// The cluster's value of a CTA value (identical in every thread), peers combined in rank order: the CTA's value goes
// to its own smem slot (one slot per use: a peer may still read an earlier one), then every CTA reads the 8.
__device__ __forceinline__ float cluster_sum(float v, uint32_t* slot) {
    if (threadIdx.x == 0) *slot = __float_as_uint(v);
    cluster_barrier();
    uint32_t p[kCtas];
    cluster_load(slot, p);
    float s = 0.f;
#pragma unroll
    for (uint32_t r = 0; r < kCtas; ++r)
        s = __fadd_rn(s, __uint_as_float(p[r]));
    return s;
}

__device__ __forceinline__ float cluster_max(float v, uint32_t* slot) {
    if (threadIdx.x == 0) *slot = __float_as_uint(v);
    cluster_barrier();
    uint32_t p[kCtas];
    cluster_load(slot, p);
    float s = -INFINITY;
#pragma unroll
    for (uint32_t r = 0; r < kCtas; ++r)
        s = fmaxf(s, __uint_as_float(p[r]));
    return s;
}

struct RadixSmem {
    uint64_t mass[2][kBins];   // this CTA's bins (top-p: fixed-point masses), double-buffered by pass parity
    uint32_t count[2][kBins];  // .. (top-k: counts)
    uint64_t tot[kBins];       // the cluster's bins of the current pass (this CTA's copy)
    uint64_t wsum[kThreads / 32];
    uint64_t above;
    uint32_t digit;
};

// One radix pass over the slice: every element of the group (key & mask == prefix, key >= floor) adds one (counts) or
// its fixed-point mass exp(y - m) to its digit's bin, once per bin and warp (a 64-bit shared add is a CAS loop, and the
// first digit puts most of a row in a few bins); the cluster's bins merge into tot. A pass reuses the buffer of the
// pass two before, which every peer finished reading before the pass in between merged.
template <bool kMass>
__device__ __forceinline__ void radix_merge(const float* w, uint32_t n, uint32_t floor_key, uint32_t prefix,
                                            uint32_t mask, uint32_t shift, uint32_t nb, uint32_t pass, RadixSmem& rs) {
    const uint32_t t = threadIdx.x, lane = t % 32;
    uint64_t* hm = rs.mass[pass & 1];
    uint32_t* hc = rs.count[pass & 1];
    if (pass >= 2) {
        for (uint32_t i = t; i < kBins; i += kThreads) {
            if constexpr (kMass) {
                hm[i] = 0;
            } else {
                hc[i] = 0;
            }
        }
        __syncthreads();
    }
    constexpr uint32_t kUnroll = 4;  // smem loads in flight per thread
    for (uint32_t i0 = 0; i0 < n; i0 += kUnroll * kThreads) {
        float v[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            const uint32_t i = i0 + u * kThreads + t;
            v[u] = i < n ? w[i] : 0.f;
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            uint32_t bin = UINT32_MAX;
            uint64_t add = 0;
            const uint32_t key = order_key(v[u]);
            if (i0 + u * kThreads + t < n && key >= floor_key && (key & mask) == prefix) {
                bin = (key >> shift) & (nb - 1);
                if constexpr (kMass) add = __float2ull_rz(__fmul_rn(v[u], kMassScale));
            }
            if (!__any_sync(0xffffffffu, bin != UINT32_MAX)) continue;
            // the lanes of one bin: the lowest adds the group's count or mass
            const uint32_t peers = __match_any_sync(0xffffffffu, bin);
            if (bin != UINT32_MAX) {
                const bool leader = lane == static_cast<uint32_t>(__ffs(peers) - 1);
                if constexpr (kMass) {
                    uint64_t sum = 0;
                    for (uint32_t g = peers; g != 0; g &= g - 1)
                        sum += __shfl_sync(peers, add, __ffs(g) - 1);
                    if (leader) atomicAdd(reinterpret_cast<unsigned long long*>(&hm[bin]), sum);
                } else {
                    if (leader) atomicAdd(&hc[bin], static_cast<uint32_t>(__popc(peers)));
                }
            }
        }
    }
    __syncthreads();
    cluster_barrier();
    for (uint32_t b = t; b < nb; b += kThreads) {
        uint64_t sum = 0;
        if constexpr (kMass) {
            uint64_t p[kCtas];
            cluster_load(&hm[b], p);
#pragma unroll
            for (uint32_t r = 0; r < kCtas; ++r)
                sum += p[r];
        } else {
            uint32_t p[kCtas];
            cluster_load(&hc[b], p);
#pragma unroll
            for (uint32_t r = 0; r < kCtas; ++r)
                sum += p[r];
        }
        rs.tot[b] = sum;
    }
    __syncthreads();
}

// The digit whose bin holds the need-th unit from the top of tot (the bins of higher digits hold less than need, those
// down to the digit at least need); rs.digit and rs.above (the units above its bin). Integers: no order question.
__device__ __forceinline__ void radix_pick(uint32_t nb, uint64_t need, RadixSmem& rs) {
    const uint32_t t = threadIdx.x, per = nb / kThreads, lane = t % 32, warp = t / 32;
    uint64_t mine = 0;
    for (uint32_t b = 0; b < per; ++b)
        mine += rs.tot[t * per + b];
    // the units in the bins of higher threads: a suffix scan (warp, then the warps above)
    uint64_t incl = mine;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint64_t v = __shfl_down_sync(0xffffffffu, incl, o);
        if (lane + o < 32) incl += v;
    }
    if (lane == 0) rs.wsum[warp] = incl;
    __syncthreads();
    uint64_t above = incl - mine;
    for (uint32_t w = warp + 1; w < kThreads / 32; ++w)
        above += rs.wsum[w];
    for (int32_t b = static_cast<int32_t>(per) - 1; b >= 0; --b) {
        const uint64_t c = rs.tot[t * per + b];
        if (above < need && above + c >= need) {
            rs.digit = t * per + b;
            rs.above = above;
        }
        above += c;
    }
    __syncthreads();
}

// The order key of the need-th unit from the top among the elements whose key has `prefix` under `mask` (and is >=
// floor), refined one digit per pass (counts, or kMass fixed-point masses): the whole key in digits of 11, 11 and 10
// bits, or (kMantissa) one exponent's mantissa in three bytes (the top byte's high bit is the exponent's, fixed). A
// digit has at least kThreads bins (radix_pick).
template <bool kMass, bool kMantissa>
__device__ __forceinline__ uint32_t radix_refine(const float* w, uint32_t n, uint32_t floor_key, uint32_t prefix,
                                                 uint32_t mask, uint64_t need, uint32_t& pass, RadixSmem& rs) {
    constexpr uint32_t shift[3] = {kMantissa ? 16u : 21u, kMantissa ? 8u : 10u, 0u};
    constexpr uint32_t nb[3] = {kMantissa ? 256u : 2048u, kMantissa ? 256u : 2048u, kMantissa ? 256u : 1024u};
    for (uint32_t d = 0; d < 3; ++d, ++pass) {
        radix_merge<kMass>(w, n, floor_key, prefix, mask, shift[d], nb[d], pass, rs);
        radix_pick(nb[d], need, rs);
        need -= rs.above;
        prefix |= rs.digit << shift[d];
        mask |= (nb[d] - 1) << shift[d];
        __syncthreads();  // rs.digit / rs.above are rewritten by the next pick
    }
    return prefix;
}

// Exponent buckets of a weight in [0, 1]: 0 holds 1.0, k holds [2^-k, 2^(1-k)), kBuckets - 1 everything below. A
// bucket below the last is one exponent: the order keys under kBucketMask, refined over the mantissa.
inline constexpr uint32_t kBuckets = 32;
inline constexpr uint32_t kBucketMask = 0xFF800000u;

__device__ __forceinline__ uint32_t bucket_of(float w) {
    const uint32_t e = (__float_as_uint(w) >> 23) & 0xFFu;
    return min(127u - min(e, 127u), kBuckets - 1);
}

__device__ __forceinline__ uint32_t bucket_prefix(uint32_t b) {
    return (0x100u + 127u - b) << 23;
}

// The cluster's sum of a per-thread u64 (fixed order is moot: integers), every thread; slot: a smem word per use as in
// cluster_sum (two uses apart may share one).
__device__ __forceinline__ uint64_t cluster_sum_u64(uint64_t v, uint64_t* s_warp, uint64_t* slot) {
#pragma unroll
    for (uint32_t o = 16; o >= 1; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) s_warp[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t c = 0;
        for (uint32_t k = 0; k < kThreads / 32; ++k)
            c += s_warp[k];
        *slot = c;
    }
    cluster_barrier();
    uint64_t p[kCtas];
    cluster_load(slot, p);
    uint64_t sum = 0;
#pragma unroll
    for (uint32_t r = 0; r < kCtas; ++r)
        sum += p[r];
    return sum;
}

// The CTA's weights >= floor in the bucket of `prefix`, appended to `list` (warp-aggregated; their order is moot: the
// radix passes count and add integers). Returns their count, which may exceed cap (the caller then scans all of w).
__device__ __forceinline__ uint32_t gather_bucket(const float* w, uint32_t n, uint32_t floor_key, uint32_t prefix,
                                                  float* list, uint32_t cap, uint32_t* s_count) {
    const uint32_t t = threadIdx.x, lane = t % 32;
    if (t == 0) *s_count = 0;
    __syncthreads();
    for (uint32_t i0 = 0; i0 < n; i0 += kThreads) {
        const uint32_t i = i0 + t;
        const float v = i < n ? w[i] : 0.f;
        const uint32_t key = order_key(v);
        const bool in = i < n && key >= floor_key && (key & kBucketMask) == prefix;
        const uint32_t ballot = __ballot_sync(0xffffffffu, in);
        if (ballot == 0) continue;
        uint32_t base = 0;
        if (lane == 0) base = atomicAdd(s_count, static_cast<uint32_t>(__popc(ballot)));
        base = __shfl_sync(0xffffffffu, base, 0);
        const uint32_t pos = base + __popc(ballot & ((1u << lane) - 1));
        if (in && pos < cap) list[pos] = v;
    }
    __syncthreads();
    return *s_count;
}

struct SelectSmem {
    uint32_t warp_cnt[kThreads / 32][kBuckets];
    uint32_t cnt[kBuckets];  // this CTA's per bucket
    uint32_t tot[kBuckets];  // the cluster's
    uint64_t warp_sum[kThreads / 32];
    uint64_t slot[2];
    uint32_t bucket;
    uint64_t above;
    uint32_t members;
};

// top-k: the order key of the k-th largest weight. One pass counts each exponent bucket (8-bit counters packed four to
// a register: a thread holds at most 255 elements), the bucket holding the k-th is refined over its mantissa (only its
// members take part); the last bucket refines over the whole key.
__device__ __forceinline__ uint32_t select_top_k(const float* w, uint32_t n, uint32_t k, uint32_t& pass, RadixSmem& rs,
                                                 SelectSmem& ss) {
    const uint32_t t = threadIdx.x, lane = t % 32, warp = t / 32;
    // every counter word takes an add (0 or its byte's one) per element: a conditional add would index the array
    uint32_t pk[kBuckets / 4] = {};
    for (uint32_t i = t; i < n; i += kThreads) {
        const uint32_t b = bucket_of(w[i]), one = 1u << (8 * (b & 3));
#pragma unroll
        for (uint32_t q = 0; q < kBuckets / 4; ++q)
            pk[q] += (b >> 2) == q ? one : 0u;
    }
#pragma unroll
    for (uint32_t b = 0; b < kBuckets; ++b) {
        const uint32_t c = __reduce_add_sync(0xffffffffu, (pk[b >> 2] >> (8 * (b & 3))) & 0xFFu);
        if (lane == 0) ss.warp_cnt[warp][b] = c;
    }
    __syncthreads();
    if (t < kBuckets) {
        uint32_t c = 0;
        for (uint32_t k2 = 0; k2 < kThreads / 32; ++k2)
            c += ss.warp_cnt[k2][t];
        ss.cnt[t] = c;
    }
    cluster_barrier();
    if (t < kBuckets) {
        uint32_t p[kCtas];
        cluster_load(&ss.cnt[t], p);
        uint32_t c = 0;
#pragma unroll
        for (uint32_t r = 0; r < kCtas; ++r)
            c += p[r];
        ss.tot[t] = c;
    }
    __syncthreads();
    if (t == 0) {
        uint64_t above = 0;
        uint32_t bk = kBuckets - 1;
        for (uint32_t b = 0; b < kBuckets; ++b) {
            const uint32_t c = ss.tot[b];
            if (above + c >= k) {
                bk = b;
                break;
            }
            above += c;
        }
        ss.bucket = bk;
        ss.above = bk == kBuckets - 1 ? 0 : above;
    }
    __syncthreads();
    const uint32_t bk = ss.bucket;
    const uint64_t need = k - ss.above;
    __syncthreads();
    if (bk == kBuckets - 1) return radix_refine<false, false>(w, n, 0u, 0u, 0u, need, pass, rs);
    // the bucket's members gathered once (into the mass bins, unused by counts), so the passes scan only them
    float* const list = reinterpret_cast<float*>(rs.mass);
    constexpr uint32_t kCap = sizeof(rs.mass) / sizeof(float);
    const uint32_t m = gather_bucket(w, n, 0u, bucket_prefix(bk), list, kCap, &ss.members);
    return m <= kCap ? radix_refine<false, true>(list, m, 0u, bucket_prefix(bk), kBucketMask, need, pass, rs)
                     : radix_refine<false, true>(w, n, 0u, bucket_prefix(bk), kBucketMask, need, pass, rs);
}

// The fixed-point mass of the weights >= floor in buckets <= b, over the cluster.
__device__ __forceinline__ uint64_t mass_upto(const float* w, uint32_t n, uint32_t floor_key, uint32_t b,
                                              SelectSmem& ss, uint32_t& use) {
    uint64_t s = 0;
#pragma unroll 4
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float v = w[i];
        if (order_key(v) >= floor_key && bucket_of(v) <= b) s += __float2ull_rz(__fmul_rn(v, kMassScale));
    }
    return cluster_sum_u64(s, ss.warp_sum, &ss.slot[use++ & 1]);
}

// top-p within the kept weights (>= floor): the pivot where the mass from the top first reaches top_p of their mass
// (SGLang keeps v while the mass above v is below top_p: that element is the last kept). The bucket where the mass
// crosses comes from a bisection of the bucket masses, then the refinement over its mantissa.
__device__ __forceinline__ uint32_t select_top_p(const float* w, uint32_t n, uint32_t floor_key, float top_p,
                                                 uint32_t& pass, RadixSmem& rs, SelectSmem& ss) {
    uint32_t use = 0;
    const uint64_t total = mass_upto(w, n, floor_key, kBuckets - 1, ss, use);
    uint64_t need = static_cast<uint64_t>(ceil(static_cast<double>(top_p) * static_cast<double>(total)));
    need = need < 1 ? 1 : (need > total ? total : need);
    int32_t lo = -1, hi = kBuckets - 1;  // mass(lo) < need <= mass(hi)
    uint64_t above = 0;
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) / 2;
        const uint64_t m = mass_upto(w, n, floor_key, mid, ss, use);
        if (m >= need) {
            hi = mid;
        } else {
            lo = mid;
            above = m;
        }
    }
    if (hi == kBuckets - 1) return radix_refine<true, false>(w, n, floor_key, 0u, 0u, need, pass, rs);
    // the bucket's members gathered once (into the count bins: every peer read them before the last cluster reduction)
    float* const list = reinterpret_cast<float*>(rs.count);
    constexpr uint32_t kCap = sizeof(rs.count) / sizeof(float);
    const uint32_t m = gather_bucket(w, n, floor_key, bucket_prefix(hi), list, kCap, &ss.members);
    return m <= kCap
               ? radix_refine<true, true>(list, m, floor_key, bucket_prefix(hi), kBucketMask, need - above, pass, rs)
               : radix_refine<true, true>(w, n, floor_key, bucket_prefix(hi), kBucketMask, need - above, pass, rs);
}

// a logit's weight exp(x / T - max): the expression the row phase and the bonus's scan both evaluate
__device__ __forceinline__ float weight(float x, float inv_t, float m) {
    return __expf(__fsub_rn(__fmul_rn(x, inv_t), m));
}

// The row phase: the cluster of row blockIdx.x / kCtas, this CTA's slice of its columns. smem: the slice's weights,
// then RadixSmem.
__device__ __forceinline__ void rows(const Args& a) {
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    float* const w = reinterpret_cast<float*>(smem_buf);
    RadixSmem& rs = *reinterpret_cast<RadixSmem*>(smem_buf + (a.slice * 4 + 15) / 16 * 16);
    __shared__ float s_red[kThreads / 32];
    __shared__ uint32_t s_slot[3];  // the cluster reductions' slots: max, Z_k, Z
    __shared__ SelectSmem ss;
    const uint32_t t = threadIdx.x, lane = t % 32, warp = t / 32, row = blockIdx.x / kCtas;
    const uint32_t j = ::ptx::cluster_ctarank();
    const uint32_t c0 = j * a.slice, n = min(a.slice, a.vocab - c0);
    for (uint32_t i = t; i < 2 * kBins; i += kThreads) {
        rs.mass[i / kBins][i % kBins] = 0;
        rs.count[i / kBins][i % kBins] = 0;
    }
    ::ptx::griddepcontrol_wait();  // the logits are the softcap's, the candidates and q the head's
    const Params p = params(a.samp);
    const float inv_t = __frcp_rn(p.temperature);
    const float* src = a.logits + static_cast<uint64_t>(row) * a.vocab + c0;
    float mx = -INFINITY;
    constexpr uint32_t kLoads = 8;  // float4 loads in flight per thread
    for (uint32_t i0 = 4 * t; i0 < n; i0 += 4 * kThreads * kLoads) {
        float4 x[kLoads];
#pragma unroll
        for (uint32_t k = 0; k < kLoads; ++k) {
            const uint32_t i = i0 + 4 * kThreads * k;
            if (i < n) x[k] = __ldcg(reinterpret_cast<const float4*>(src + i));
        }
#pragma unroll
        for (uint32_t k = 0; k < kLoads; ++k) {
            const uint32_t i = i0 + 4 * kThreads * k;
            if (i < n) {
                *reinterpret_cast<float4*>(w + i) = x[k];
                mx = fmaxf(fmaxf(mx, fmaxf(x[k].x, x[k].y)), fmaxf(x[k].z, x[k].w));
            }
        }
    }
    // the row's max over the scaled logits (T > 0: the scaled max is the max scaled), then its weights in place
    const float m = __fmul_rn(cluster_max(block_max(mx, s_red), &s_slot[0]), inv_t);
    for (uint32_t i = t; i < n; i += kThreads)
        w[i] = weight(w[i], inv_t, m);
    __syncthreads();
    uint32_t pass = 0;
    // top-k: keep the elements at or above the top_k-th largest
    const uint32_t tau = p.top_k < a.vocab ? select_top_k(w, n, p.top_k, pass, rs, ss) : 0u;
    float zk = 0.f;
    for (uint32_t i = t; i < n; i += kThreads)
        if (order_key(w[i]) >= tau) zk = __fadd_rn(zk, w[i]);
    zk = cluster_sum(block_sum(zk, s_red), &s_slot[1]);
    // top-p within the top-k: keep up to the pivot where the mass from the top first reaches top_p
    uint32_t keep = tau;
    float z = zk;
    if (p.top_p < 1.f) {
        keep = max(tau, select_top_p(w, n, tau, p.top_p, pass, rs, ss));
        float zs = 0.f;
        for (uint32_t i = t; i < n; i += kThreads)
            if (order_key(w[i]) >= keep) zs = __fadd_rn(zs, w[i]);
        z = cluster_sum(block_sum(zs, s_red), &s_slot[2]);
    }
    const float inv_z = __frcp_rn(z);
    // the chunk masses of the slice (warp w: chunks w, w + 8, ..; lane l: elements l, l + 32, ..), then the
    // candidates' probabilities and the row's statistics
    const uint32_t cpc = chunks_per_cta(a.slice);
    for (uint32_t cc = warp; cc < cpc; cc += kThreads / 32) {
        const uint32_t lo = cc * kChunk, hi = min(lo + kChunk, n);
        float sum = 0.f;
        for (uint32_t i = lo + lane; i < hi; i += 32)
            if (order_key(w[i]) >= keep) sum = __fadd_rn(sum, __fmul_rn(w[i], inv_z));
#pragma unroll
        for (uint32_t o = 16; o >= 1; o >>= 1)
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
        if (lane == 0) a.chunk[(row * kCtas + j) * cpc + cc] = sum;
    }
    if (row < a.gamma && t < kTopK) {
        const int64_t id = a.cand[row * kTopK + t];
        if (id >= c0 && id < c0 + n) {
            const float v = w[id - c0];
            a.pc[row * kTopK + t] = order_key(v) >= keep ? __fmul_rn(v, inv_z) : 0.f;
        }
    }
    if (j == 0 && t == 0) {
        a.stats[row * kStatWords + kStatMax] = m;
        a.stats[row * kStatWords + kStatInvZ] = inv_z;
        a.stats[row * kStatWords + kStatKey] = __uint_as_float(keep);
    }
    cluster_barrier();  // no CTA leaves while a peer may still read its smem
}

// ---- the decision (the last CTA, every thread) ----

struct DecideSmem {
    float pc[kRows * kTopK];
    float q[kRows * kTopK];
    int32_t cand[kRows * kTopK];
    int32_t path[kRows];
    float u[kRows];
    float tot[kRows];
    float stat[kRows * kStatWords];
    float wsum[kThreads / 32];
    float target, scale, base;
    uint32_t acc, all, chunk, pick, last;
};

// sum over v of max(s P_r(v) - q_r(v), 0): s times the mass off the candidates plus the candidates' terms
__device__ __forceinline__ float residual(const DecideSmem& d, uint32_t r, float s) {
    float resid = 0.f, pmass = 0.f;
    for (uint32_t c = 0; c < kTopK; ++c) {
        const float pv = d.pc[r * kTopK + c];
        resid = __fadd_rn(resid, fmaxf(__fsub_rn(__fmul_rn(s, pv), d.q[r * kTopK + c]), 0.f));
        pmass = __fadd_rn(pmass, pv);
    }
    return __fadd_rn(__fmul_rn(s, fmaxf(__fsub_rn(d.tot[r], pmass), 0.f)), resid);
}

// The block's inclusive prefix of v in thread order (fixed order).
__device__ __forceinline__ float block_scan(float v, float* wsum) {
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const float x = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v = __fadd_rn(v, x);
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    float base = 0.f;
    for (uint32_t w = 0; w < warp; ++w)
        base = __fadd_rn(base, wsum[w]);
    __syncthreads();
    return __fadd_rn(base, v);
}

// The accept length and the bonus token of the step (iter: the step word ITER), from the rows' statistics.
__device__ __forceinline__ void decide(const Args& a, uint64_t iter, float* s_chunk, uint32_t& acc_out,
                                       int64_t& bonus_out) {
    __shared__ DecideSmem d;
    const uint32_t t = threadIdx.x, lane = t % 32, warp = t / 32, g = a.gamma, cpc = chunks_per_cta(a.slice);
    const uint32_t nch = kCtas * cpc;
    if (t < g * kTopK) {
        d.pc[t] = __ldcg(a.pc + t);
        d.q[t] = __ldcg(a.q + t);
        d.cand[t] = __ldcg(a.cand + t);
    }
    if (t < g) d.path[t] = __ldcg(a.path + t);
    if (t < kRows) d.u[t] = uniform(a.samp, iter, kPurposeAccept, t);
    // every live row's statistics and chunk masses ahead of the decision that picks the row
    if (t < (g + 1) * kStatWords) d.stat[t] = __ldcg(a.stats + t);
#pragma unroll 4
    for (uint32_t i = t; i < (g + 1) * nch; i += kThreads)
        s_chunk[i] = __ldcg(a.chunk + i);
    __syncthreads();
    for (uint32_t r = warp; r <= g; r += kThreads / 32) {  // each live row's kept mass, chunks in a fixed order
        float s = 0.f;
        for (uint32_t c = lane; c < nch; c += 32)
            s = __fadd_rn(s, s_chunk[r * nch + c]);
#pragma unroll
        for (uint32_t o = 16; o >= 1; o >>= 1)
            s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
        if (lane == 0) d.tot[r] = s;
    }
    __syncthreads();
    if (t == 0) {
        // the accepted prefix's probability, step by step: min(pp * P / q, 1), 0 where q is no probability
        float pps[kRows];
        float pp = 1.f;
        for (uint32_t e = 1; e <= g; ++e) {
            const uint32_t i = (e - 1) * kTopK + d.path[e - 1];
            const float pv = d.pc[i], qv = d.q[i];
            pp = (qv > 0.f && qv <= 1.f) ? fminf(__fdiv_rn(__fmul_rn(pp, pv), qv), 1.f) : 0.f;
            pps[e] = pp;
        }
        uint32_t acc = 0;
        float scale = 1.f, norm = -1.f;
        if (a.block) {
            // the longest prefix first; a prefix passes its probability test, then the residual's
            for (uint32_t e = g; e >= 1 && acc == 0; --e) {
                const float coin = d.u[e - 1];
                pp = pps[e];
                if (coin < pp) {
                    float ap = pp, rm = -1.f;
                    if (e < g && pp > 0.f && pp < 1.f) {
                        rm = residual(d, e, pp);
                        ap = __fdiv_rn(rm, __fsub_rn(__fadd_rn(rm, 1.f), pp));
                    }
                    if (coin < ap) {
                        acc = e;
                        scale = pp;
                        norm = rm;
                    }
                }
            }
        } else {
            for (uint32_t e = 1; e <= g; ++e) {
                const uint32_t i = (e - 1) * kTopK + d.path[e - 1];
                const float pv = d.pc[i], qv = d.q[i];
                if (!(qv > 0.f && qv <= 1.f && __fmul_rn(d.u[e - 1], qv) < pv)) break;
                acc = e;
            }
        }
        const bool all = acc == g;
        if (norm < 0.f) norm = all ? d.tot[acc] : residual(d, acc, scale);
        d.acc = acc;
        d.all = all;
        d.scale = scale;
        d.target = __fmul_rn(d.u[kFinalUniform], norm);
        d.chunk = UINT32_MAX;
        d.pick = UINT32_MAX;
        d.last = 0;
    }
    __syncthreads();
    const uint32_t r = d.acc;
    const bool all = d.all;
    const float s = d.scale, target = d.target;
    const float* stat = d.stat + r * kStatWords;
    const float m = stat[kStatMax], inv_z = stat[kStatInvZ];
    const uint32_t keep = __float_as_uint(stat[kStatKey]);
    // the chunk masses of the residual (row r's candidates moved from s P to max(s P - q, 0)), the first chunk whose
    // inclusive mass passes the target
    float mk = 0.f;
    if (t < nch) {
        mk = s_chunk[r * nch + t];
        if (!all) {
            float pin = 0.f, rin = 0.f;
            for (uint32_t c = 0; c < kTopK; ++c) {
                const uint32_t id = static_cast<uint32_t>(d.cand[r * kTopK + c]);
                if ((id / a.slice) * cpc + (id % a.slice) / kChunk == t) {
                    const float pv = d.pc[r * kTopK + c];
                    pin = __fadd_rn(pin, pv);
                    rin = __fadd_rn(rin, fmaxf(__fsub_rn(__fmul_rn(s, pv), d.q[r * kTopK + c]), 0.f));
                }
            }
            mk = __fadd_rn(__fmul_rn(s, fmaxf(__fsub_rn(mk, pin), 0.f)), rin);
        }
    }
    const float incl = block_scan(mk, d.wsum);
    if (t < nch && incl > target) atomicMin(&d.chunk, t);
    __syncthreads();
    if (d.chunk == UINT32_MAX) {  // the CDF never passes the target (the residual vanishes): SGLang's last id
        acc_out = r;
        bonus_out = a.vocab - 1;
        return;
    }
    const uint32_t k = d.chunk;
    if (t == k) d.base = __fsub_rn(incl, mk);
    __syncthreads();
    // the chunk's elements in vocabulary order: thread t holds 4 consecutive ones
    const uint32_t jj = k / cpc, lo = jj * a.slice + (k % cpc) * kChunk;
    const uint32_t hi = min(min(lo + kChunk, jj * a.slice + a.slice), a.vocab);
    const float* lrow = a.logits + static_cast<uint64_t>(r) * a.vocab;
    const float inv_t = __frcp_rn(params(a.samp).temperature);
    float val[4], cum = 0.f;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t v = lo + 4 * t + e;
        float x = 0.f;
        if (v < hi) {
            const float wv = weight(__ldcg(lrow + v), inv_t, m);
            if (order_key(wv) >= keep) {
                const float pv = __fmul_rn(wv, inv_z);
                x = all ? pv : __fmul_rn(s, pv);
                if (!all)
                    for (uint32_t c = 0; c < kTopK; ++c)
                        if (static_cast<uint32_t>(d.cand[r * kTopK + c]) == v)
                            x = fmaxf(__fsub_rn(__fmul_rn(s, pv), d.q[r * kTopK + c]), 0.f);
            }
        }
        val[e] = x;
        cum = __fadd_rn(cum, x);
    }
    const float excl = __fsub_rn(block_scan(cum, d.wsum), cum);
    float run = __fadd_rn(d.base, excl);
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        run = __fadd_rn(run, val[e]);
        if (val[e] > 0.f) {
            if (run > target) atomicMin(&d.pick, lo + 4 * t + e);
            atomicMax(&d.last, lo + 4 * t + e);
        }
    }
    __syncthreads();
    acc_out = r;
    bonus_out = d.pick != UINT32_MAX ? d.pick : d.last;  // a rounding gap between the chunk's sum and its scan
}

}  // namespace mk::dflash::sample
