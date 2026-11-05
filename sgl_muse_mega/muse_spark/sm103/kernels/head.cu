// The verify graph's head and tail kernels: the window feeds, the fused entry, layer 0's qkv, the gathered lm_head and
// the softcap. One image per build: -DMK_IMAGE=window_feeds|entry_fused|qkv_head|lm_head|softcap instantiates that
// image's kernel (below); the host library includes this file under MK_HOST for the argument structs.
#include <cstdint>
#include <cstdio>

#include <cuda_bf16.h>

#include "framework.cuh"

// The window feeds' argument struct, shared with the host packer.

namespace mk {

struct WindowFeedsArgs {
    const int64_t* positions;          // [>= m]
    const float* cos_sin_cache;        // [P][d]; null = no rope rows this window
    const int64_t* out_cache_loc;      // [>= m]
    const int64_t* swa_out_cache_loc;  // [>= m]; null = no sliding-window pool
    const int64_t* seq_lens;           // [1]: the window's one request
    const int64_t* req_pool_indices;   // [1]
    const int32_t* req_to_token;       // [R][w]
    const int64_t* full_to_swa;        // [swa_map_len]; null = no sliding-window pool
    float* cos_sin;                    // [m][d]
    int32_t* loc_full;                 // [m]
    int32_t* loc_swa;                  // [m]
    int32_t* kv_len;                   // [1]
    int64_t* pt_full;                  // [w] int64 byte offsets (mk_attn's page table)
    int64_t* pt_swa;                   // [w]
    uint32_t m, d, w;
    uint32_t kv_row_bytes;
    int64_t r2t_stride;    // req_to_token's row stride (elements)
    int64_t r2t_rows;      // R: the request slot the window names must be one of them
    int64_t cos_sin_rows;  // P: every position must index a rope row
    int64_t swa_map_len;
};

}  // namespace mk

// The window feeds, the verify graph's first kernel.
// launch: grid ceil(w / 256), block 256, no smem, PDL
// The verify window's attention feeds in one launch: the per-forward tensors the qkv / rowops and attention kernels
// read (the rope rows, the int32 cache-write locations and kv_len, the page-table row and its full -> swa
// translation, as byte offsets). Every output is an integer gather, a truncating cast or an integer product:
//   cos_sin[t][c] = cos_sin_cache[positions[t]][c]                        (fp32)
//   loc_full[t]   = int32(out_cache_loc[t]);  loc_swa[t] = int32(swa_out_cache_loc[t])
//   kv_len[0]     = int32(seq_lens[0] + m)
//   pt_full[j]    = row[j] * kv_row_bytes              (row = req_to_token[req_pool_indices[0]]; int64 products)
//   pt_swa[j]     = full_to_swa[row[j]] * kv_row_bytes (a negative index wraps: -1 = the sentinel)
// The pool is bf16, whose pages are exactly their rows, so a slot's byte offset is slot x row bytes at any page size
// (a negative slot stays a negative sentinel).
// Block 0 owns the token-indexed outputs; every block grid-strides the page-table row. One request per launch, at
// most 16 rows.

namespace mk {

// the pool's byte offset of slot s (a negative slot stays a negative sentinel)
static __device__ __forceinline__ int64_t window_feeds_slot_off(const WindowFeedsArgs& a, const int64_t s) {
    return s * static_cast<int64_t>(a.kv_row_bytes);
}

// One request's feeds (``a`` = that request's view): block 0 the token-indexed outputs, every block the page-table row.
template <uint32_t kThreads>
static __device__ __forceinline__ void window_feeds_request(const WindowFeedsArgs& a) {
    const uint32_t tid = threadIdx.x;
    if (blockIdx.x == 0) {
        if (a.cos_sin_cache != nullptr) {
            const uint32_t md = a.m * a.d;
            for (uint32_t i = tid; i < md; i += kThreads) {
                const uint32_t t = i / a.d, c = i - t * a.d;
                const int64_t pos = a.positions[t];
                if (pos < 0 || pos >= a.cos_sin_rows) {
                    __trap();
                }
                a.cos_sin[i] = a.cos_sin_cache[pos * static_cast<int64_t>(a.d) + c];
            }
        }
        for (uint32_t t = tid; t < a.m; t += kThreads) {
            a.loc_full[t] = static_cast<int32_t>(a.out_cache_loc[t]);
            if (a.swa_out_cache_loc != nullptr) a.loc_swa[t] = static_cast<int32_t>(a.swa_out_cache_loc[t]);
        }
        if (tid == 0) a.kv_len[0] = static_cast<int32_t>(a.seq_lens[0] + static_cast<int64_t>(a.m));
    }
    const int64_t req = a.req_pool_indices[0];
    if (req < 0 || req >= a.r2t_rows) {
        if (threadIdx.x == 0 && blockIdx.x == 0) __trap();
    }
    const int32_t* row = a.req_to_token + req * a.r2t_stride;
    const uint32_t stride = gridDim.x * kThreads;
    for (uint32_t j = blockIdx.x * kThreads + tid; j < a.w; j += stride) {
        const int32_t r = row[j];
        a.pt_full[j] = window_feeds_slot_off(a, r);
        if (a.full_to_swa != nullptr) {
            int64_t idx = r;
            if (idx < 0) idx += a.swa_map_len;  // a negative index wraps (the appended -1 sentinel row)
            if (idx < 0 || idx >= a.swa_map_len) {
                __trap();
            }
            a.pt_swa[j] = window_feeds_slot_off(a, a.full_to_swa[idx]);
        }
    }
}

template <uint32_t kThreads>
__global__ static __launch_bounds__(kThreads) void window_feeds(const WindowFeedsArgs a) {
    // PDL: wait, then trigger (the dependent's own griddepcontrol.wait still covers this grid's stores)
    ptx::griddepcontrol_wait();
    if (threadIdx.x == 0) ptx::griddepcontrol_launch_dependents();
    window_feeds_request<kThreads>(a);
}

}  // namespace mk

// The verify step's entry in ONE launch, one CTA per window row t:
//   e           = embed[id_t]                                   the vocab shard that owns id_t, read in place
//   x           = bf16(e * rsqrt(mean(e^2) + eps_embed))         the weightless embed RMSNorm
//   residual[t] = fp32(x)                                        layer 0's residual stream
//   normed[t]   = bf16(x * rsqrt(mean(x^2) + eps_pre) * pre_w)   layer 0's pre-attention norm (the qkv head's x)
// It replaces four launches: a masked shard lookup on every rank, a one-shot all-reduce of those rows, the embed norm
// and the pre-norm. The vocab shards are disjoint, so that all-reduce adds one nonzero row to seven zero rows: it is a
// gather of the owner's row. With every shard in symmetric memory each rank loads the owner's row over NVLink itself.
// The source is immutable weights, so no flag, epoch or buffer-reuse protocol exists, and the result equals the
// all-reduce's (up to the sign of an exact zero). The reduction tree is fixed (256 threads x 4
// chunks, per-thread fma chain, warp butterfly, in-order sum of the warp partials), so x matches it bitwise.

namespace mk {

inline constexpr uint32_t kEntryRanks = cfg::kRanks;  // the tensor-parallel world: one vocab shard per rank

struct EntryArgs {
    const int64_t* ids;               // [rows]
    const uint4* embed[kEntryRanks];  // every rank's vocab shard [shard_rows, hidden] bf16
    int64_t shard_rows;
    const nv_bfloat16* pre_w;  // [hidden]
    float* residual;           // [rows, hidden]
    nv_bfloat16* normed;       // [rows, hidden]
    float eps_embed;
    float eps_pre;
};

template <uint32_t kThreads>
__device__ __forceinline__ float entry_row_sum(float acc, float* warp_sum) {
    constexpr uint32_t kWarps = kThreads / 32;
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (threadIdx.x % 32 == 0) warp_sum[threadIdx.x / 32] = acc;
    __syncthreads();
    // every thread sums the partials in the same order: one rstd per row, no broadcast round trip
    float sum = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w)
        sum += warp_sum[w];
    return sum;
}

template <uint32_t kThreads, uint32_t kU4PerThread, uint32_t kRanks>
__global__ __launch_bounds__(kThreads, 1) void entry_fused(const __grid_constant__ EntryArgs a) {
    constexpr uint32_t kRowU4 = kThreads * kU4PerThread;
    constexpr float kHidden = float(kRowU4 * 8);
    __shared__ float warp_sum[2][kThreads / 32];
    // PDL: the ids may come from the predecessor and the outputs may still be read by it; trigger at once so the qkv
    // head's prologue overlaps this launch
    // (its own wait covers these stores)
    ptx::griddepcontrol_wait();
    if (threadIdx.x == 0) ptx::griddepcontrol_launch_dependents();

    const int64_t id = a.ids[blockIdx.x];
    const bool hit = id >= 0 && id < a.shard_rows * kRanks;
    const int64_t owner = hit ? id / a.shard_rows : 0;
    const uint4* src = hit ? a.embed[owner] + (id - owner * a.shard_rows) * kRowU4 : nullptr;
    uint4 frag[kU4PerThread];
    float acc = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kU4PerThread; ++i) {
        frag[i] = hit ? src[threadIdx.x + i * kThreads] : make_uint4(0, 0, 0, 0);
        const auto e = reinterpret_cast<const nv_bfloat162*>(&frag[i]);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(e[j]);
            acc = fmaf(f.x, f.x, acc);
            acc = fmaf(f.y, f.y, acc);
        }
    }
    const float rstd0 = rsqrtf(entry_row_sum<kThreads>(acc, warp_sum[0]) / kHidden + a.eps_embed);

    float x[kU4PerThread][8];
    acc = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kU4PerThread; ++i) {
        const auto e = reinterpret_cast<const nv_bfloat162*>(&frag[i]);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(e[j]);
            const float2 r = __bfloat1622float2(__float22bfloat162_rn(make_float2(f.x * rstd0, f.y * rstd0)));
            x[i][2 * j] = r.x;
            x[i][2 * j + 1] = r.y;
            acc = fmaf(r.x, r.x, acc);
            acc = fmaf(r.y, r.y, acc);
        }
        auto* out =
            reinterpret_cast<float4*>(a.residual + (int64_t(blockIdx.x) * kRowU4 + threadIdx.x + i * kThreads) * 8);
        out[0] = make_float4(x[i][0], x[i][1], x[i][2], x[i][3]);
        out[1] = make_float4(x[i][4], x[i][5], x[i][6], x[i][7]);
    }
    const float rstd1 = rsqrtf(entry_row_sum<kThreads>(acc, warp_sum[1]) / kHidden + a.eps_pre);

    const auto* w = reinterpret_cast<const uint4*>(a.pre_w);
#pragma unroll
    for (uint32_t i = 0; i < kU4PerThread; ++i) {
        const uint4 wv = w[threadIdx.x + i * kThreads];
        const auto wb = reinterpret_cast<const nv_bfloat162*>(&wv);
        uint4 o;
        auto ob = reinterpret_cast<nv_bfloat162*>(&o);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const float2 wf = __bfloat1622float2(wb[j]);
            ob[j] = __float22bfloat162_rn(make_float2(x[i][2 * j] * rstd1 * wf.x, x[i][2 * j + 1] * rstd1 * wf.y));
        }
        reinterpret_cast<uint4*>(a.normed)[int64_t(blockIdx.x) * kRowU4 + threadIdx.x + i * kThreads] = o;
    }
}

}  // namespace mk

// The logits softcap over the gathered rows, the verify graph's last kernel.
// launch: grid ceil(n / 8 / 256), block 256, PDL

namespace mk {

// The logits tail in one pass: the gathered fp32 logits into the fp32 buffer with the scale and the tanh softcap
// applied, out = cap * tanh((x * scale) / cap) (SGLang's order: the logit scale, then the softcap). Grid-stride over
// 32 B (8 fp32) chunks; each reads and writes two float4.
template <uint32_t kNumThreads>
__device__ __forceinline__ void softcap_body(const float4* x, float4* out, uint32_t num_c8, float scale, float cap) {
    const uint32_t stride = gridDim.x * kNumThreads;
    const auto sc = [&](float4 v) {
        const auto f = [&](float e) { return __fmul_rn(tanhf(__fdiv_rn(__fmul_rn(e, scale), cap)), cap); };
        return make_float4(f(v.x), f(v.y), f(v.z), f(v.w));
    };
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < num_c8; i += stride) {
        // the gathered row is written by peers during this grid's life: coherent loads, never the nc path
        const float4 a = x[2 * i];
        const float4 b = x[2 * i + 1];
        out[2 * i] = sc(a);
        out[2 * i + 1] = sc(b);
    }
}

// The arrival wait's budget: a rank that never arrives traps with a named print instead of hanging the step. Eager
// warm-ups skew ranks by seconds (a rank's first module load), so the bound is a minute, not a step.
inline constexpr uint64_t kGatherSpinBudgetNs = 60000000000;

// The gathered input: x is this rank's gathered row, filled by every rank's lm_head gather epilogue, which adds
// `per_launch` arrivals to *arrive per launch (a monotonic s32, compared wrap-safe). state = [generation, done]:
// thread 0 waits for (generation + 1) x per_launch; the grid's last CTA to finish advances the generation.
template <uint32_t kNumThreads>
__global__ static __launch_bounds__(kNumThreads, 1) void softcap_gathered(const float4* x, float4* out, uint32_t num_c8,
                                                                          float scale, float cap, const int* arrive,
                                                                          uint32_t* state, uint32_t per_launch) {
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
    __shared__ uint32_t generation;
    if (threadIdx.x == 0) {
        generation = *reinterpret_cast<volatile uint32_t*>(state);
        const uint32_t target = (generation + 1) * per_launch;
        uint64_t t0 = 0;
        for (uint32_t polls = 0;; ++polls) {
            const int seen = ptx::ld_acq_sys_s32(arrive);
            if (static_cast<int32_t>(static_cast<uint32_t>(seen) - target) >= 0) break;
            if ((polls & 1023u) == 0) {
                const uint64_t now = ptx::globaltimer_ns();
                if (t0 == 0) {
                    t0 = now;
                } else if (now - t0 > kGatherSpinBudgetNs) {
                    printf("softcap: lm_head row arrivals %d of target %u after %llu ns (generation %u)\n", seen,
                           target, static_cast<unsigned long long>(now - t0), generation);
                    ptx::trap();
                }
            }
        }
    }
    __syncthreads();
    softcap_body<kNumThreads>(x, out, num_c8, scale, cap);
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(state + 1, 1u) == gridDim.x - 1) {
        state[1] = 0;
        state[0] = generation + 1;
    }
}

}  // namespace mk

#define MK_HEAD_IMAGE_window_feeds 1
#define MK_HEAD_IMAGE_entry_fused 2
#define MK_HEAD_IMAGE_qkv_head 3
#define MK_HEAD_IMAGE_lm_head 4
#define MK_HEAD_IMAGE_softcap 5
#define MK_HEAD_IMAGE_CAT_(x) MK_HEAD_IMAGE_##x
#define MK_HEAD_IMAGE_CAT(x) MK_HEAD_IMAGE_CAT_(x)
#define MK_HEAD_IMAGE MK_HEAD_IMAGE_CAT(MK_IMAGE)
#if defined(MK_HOST)
#elif MK_HEAD_IMAGE == MK_HEAD_IMAGE_window_feeds
[[maybe_unused]] static constexpr auto kKernel = &mk::window_feeds<256>;
#elif MK_HEAD_IMAGE == MK_HEAD_IMAGE_entry_fused
// The fused entry: 256 threads x 4 16-B chunks cover the 8192-wide row; one vocab shard per rank.
static const void* const kEntryFused = reinterpret_cast<const void*>(&mk::entry_fused<256, 4, mk::kEntryRanks>);
#elif MK_HEAD_IMAGE == MK_HEAD_IMAGE_qkv_head
// Layer 0's qkv projection with the rowops epilogue (the next-layer phase's qkv GEMM, launched alone): 2-CTA clusters,
// grid 2 * (2304 / 64), block 256, smem dense_smem_layout(16, 4, 20, 2).size, PDL.
using QkvHeadEpi = mk::TileEpiQkvRowops<mk::cfg::kQHeads, mk::cfg::kKvHeads, mk::cfg::kWindowM>;
// clang-format off
[[maybe_unused]] static constexpr auto kKernel =
    &mk::dense_gemm<mk::cfg::kWindowM, mk::cfg::kStages, mk::cfg::kPack, mk::cfg::kEvictFirst, mk::cfg::kQkvCk, QkvHeadEpi>;
// clang-format on
#elif MK_HEAD_IMAGE == MK_HEAD_IMAGE_lm_head
// The lm_head GEMM whose epilogue writes every rank's gathered logits row (persistent grid, block 256,
// smem dense_persist_smem_layout(16, 4, 20).size, PDL).
// clang-format off
[[maybe_unused]] static constexpr auto kKernel =
    &mk::dense_persist_gather<mk::cfg::kWindowM, mk::cfg::kStages, mk::cfg::kPack, mk::cfg::kEvictFirst>;
// clang-format on
#elif MK_HEAD_IMAGE == MK_HEAD_IMAGE_softcap
[[maybe_unused]] static constexpr auto kKernel = &mk::softcap_gathered<256>;
#else
#error "head.cu: -DMK_IMAGE=window_feeds|entry_fused|qkv_head|lm_head|softcap"
#endif
