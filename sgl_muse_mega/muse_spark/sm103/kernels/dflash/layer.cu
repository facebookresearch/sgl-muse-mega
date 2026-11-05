// A draft layer's launches, one image per build (-DMK_IMAGE=qkv|layer|tail -DMK_DFLASH_B=8|16): qkv, the layer's
// qkv on the dense executor; layer and tail, the draft layer in one launch with the attention phase (tail: the last
// layer, its rows ending at the final norm). The host libraries include this file under MK_HOST: with no image for the
// attention's geometry, with a layer image for its parameter block.
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "dflash/common.cuh"

// The DFlash draft attention at head_dim 128 and its split-KV combine. A rank's kv heads (dflash/common.cuh: 1 at TP8,
// 2 at TP4) each serve 8 query heads: a group of 128 rows (row = token x 8 + head) per kv head.
//
// The window. A draft layer attends non-causally over a sliding window: the query at position p sees the keys
// [p - 2047, p + 2047] of the block's KV [0, P + B) (FA's bottom-right alignment of the ENCODER_ONLY window
// (2047, 2047)). With B <= 16 the right edge never binds, so block row t (position P + t) sees
// [max(0, P + t - 2047), P + B). The keys live in a 4096-row bf16 ring per layer (slot = position & 4095): it holds the
// window and the block for every P, and needs no page table. A ring row holds the rank's kv heads side by side.
//
// The attention. The window's 64-aligned key range [lo_w, P + B), lo_w = (P - 2047) & ~63, is at most 34 blocks of 64
// keys, so with 36 attention CTAs per group every CTA takes at most one block of one group (CTA j: group j / 36, block
// j % 36): one QK, one softmax with no online rescale, one PV. The block's K and V rows are TMA'd straight into the
// SWIZZLE_128B operand buffers (bf16 as stored, no conversion), and so are the group's q rows (one 3D box [B tokens][8
// heads][64 dims] per half: per-thread row loads would cost the L1 one line per lane per load). S = Q K^T is [128 rows
// x 64 keys] with row = token x 8 + head and A = Q in TMEM (bf16 as stored: the softmax scale rides the exp2's FFMA, so
// Q is never rounded again); P is bf16, written into S's own TMEM columns, and O = P V lands in TMEM. Each CTA
// publishes its partial row (O as bf16 pairs, then (m, l) fp32 with m in log2 units) to ws[j]. At B = 8 the group's
// rows 64-127 are dead: zero Q, never published, never combined.
//
// The combine. Row r's partials (its group's) merge in one fixed order (eighths of the partials, then the eighths in
// order), and x = O / l is stored bf16 at x_attn[token][head x 128 + d].
//
// Placement in a draft-layer launch. The attention runs on the LAST kAttnCtas CTAs of the grid (whole width-4
// clusters); each adds 1 to the attention word once its partial is out, so the word reaches kAttnCtas x (launch
// ticket + 1) only when every partial of the launch is published, and the queue's combine spins on that target. The
// attention holds 256 TMEM columns from its prologue to its end and never relinquishes the allocator permit, so a later
// phase of the same CTA may allocate again.

namespace mk::dflash {

constexpr uint32_t kD = kHd, kXRow = kQHeads * kD;  // x_attn / q row: [head][d] bf16
constexpr uint32_t kWindow = 2047;                  // keys on each side of a query (HF sliding_window 2048 - 1)
constexpr uint32_t kRing = 4096;                    // draft KV ring rows; 64-aligned blocks never straddle its wrap
constexpr uint32_t kBlk = 64;                       // keys per block
constexpr uint32_t kGroupCtas = 36;                 // attention CTAs per kv-head group: 9 width-4 clusters
constexpr uint32_t kAttnCtas = kGroupCtas * kKvHeads;
// one partial row in fp32 words: O [128 rows][64 words of bf16 pairs], then (m, l) [128][2]
constexpr uint32_t kWsRow = 128 * (kD / 2 + 2), kWsMl = 128 * (kD / 2);
constexpr uint32_t kWsImageRows = kWsRow / 64;  // the partial as 64-word rows of the store map: 128 O + 4 (m, l)
constexpr float kMSent = -1e30f;                // the running max of a row with no visible key (finite: exp2 -> 0)
constexpr float kScaleLog2 = 0.08838834764831845f * 1.4426950408889634f;  // head_dim^-0.5 x log2(e)

template <uint32_t kB>
struct Shape {
    static_assert(kB == 8 or kB == 16, "the draft block: the anchor + 7 or 15 drafts");
    static constexpr uint32_t kLive = kGqa * kB;  // live rows of the 128-row tile
    // len - lo_w <= B + 2047 + 63 keys
    static constexpr uint32_t kMaxBlocks = (kB + kWindow + (kBlk - 1) + kBlk - 1) / kBlk;
    static_assert(kMaxBlocks <= kGroupCtas, "one block per attention CTA");
    // rows per combine CTA: two, or as many as keep the combine on 64 CTAs (16 clusters, in the queue's first round)
    static constexpr uint32_t kCombineRows = kLive * kKvHeads > 128 ? kLive* kKvHeads / 64 : 2;
    static constexpr uint32_t kCombineCtas = kLive * kKvHeads / kCombineRows;
};

// a q head's group (its kv head) and attention CTA j's group
__device__ __forceinline__ uint32_t group_of_head(const uint32_t head) {
    if constexpr (kKvHeads == 1)
        return 0;
    else
        return head / kGqa;
}
__device__ __forceinline__ uint32_t group_of_cta(const uint32_t j) {
    if constexpr (kKvHeads == 1)
        return 0;
    else
        return j / kGroupCtas;
}

// the window's block range at kv_len len = P + B: [lo_w, lo_w + 64 x blocks) covers [row 0's first key, len)
struct Window {
    uint32_t lo_w, blocks;
};
template <uint32_t kB>
__host__ __device__ constexpr Window window_of(const uint32_t len) {
    const uint32_t lo = len > kB + kWindow ? len - kB - kWindow : 0u;
    const uint32_t lo_w = lo & ~(kBlk - 1);
    return {lo_w, (len - lo_w + kBlk - 1) / kBlk};
}

struct DraftAttnArgs {
    const int32_t* kv_len;  // P + B
    float* ws;              // [kAttnCtas][kWsRow] partials (group-major)
    uint32_t* word;         // the attention word (zeroed once, never reset): read by the draft layer's combine
};

namespace attn_layout {
// the carve (offsets from a 1024 B aligned base): barriers, the TMEM address, the two WGs' row-max exchange, WG1's
// row sums, the (m, l) publish stage, the K and V blocks (two 64-column SWIZZLE_128B chunks each), then the q rows'
// two halves ([128 rows][128 B] SWIZZLE_128B each, row = token x 8 + head)
constexpr uint32_t kOffBars = 0, kOffTmemWord = 128, kOffXmax = 1024, kOffXsum = 2048, kOffMl = 2560;
constexpr uint32_t kChunk = kBlk * 128;
constexpr uint32_t kOffK = 4096, kOffV = kOffK + 2 * kChunk;
constexpr uint32_t kOffQ = kOffV + 2 * kChunk, kQBox = 128 * 128;
constexpr uint32_t kSmem = kOffQ + 2 * kQBox;
// the publish stage on the dead K / V blocks: box h = O dims [64h, 64h + 64) as bf16 pairs, 128 B per row
constexpr uint32_t kStageBox = 128 * 128;
static_assert(kOffK + 2 * kStageBox == kOffQ and kOffK % 1024 == 0, "the stage boxes fill the K and V blocks");
static_assert(kOffMl + 128 * 8 <= kOffK, "the header");
// kBQt: the q rows' TMA
enum Bar : uint32_t { kBK, kBV, kBQ, kBS, kBP, kBPv, kBTmem, kBQt, kNumBars };
static_assert(kOffBars + kNumBars * 8 <= kOffTmemWord, "the barriers");
// TMEM columns: Q (bf16 pairs) | S, with P (bf16 pairs) written over each WG's first 16 columns | O
constexpr uint32_t kTQ = 0, kTS = 64, kTO = 128, kTmemCols = 256;
// roles: warps 0-7 the two softmax WGs (thread = row = TMEM lane; WG h: keys [32h, 32h + 32), O dims [64h, + 64))
constexpr uint32_t kSoftmaxThreads = 256, kFeedWarp = 8, kIssuerWarp = 9, kAllocWarp = 10, kInitWarp = 11;
}  // namespace attn_layout

// bounded waits: a stuck wait traps instead of hanging the stream (a poll count, not dflash/common.cuh's timed wait:
// the composed layer image is scheduled around this one)
static __device__ __forceinline__ void bar_wait(uint64_t* bar, const uint32_t ph) {
    uint64_t polls = 0;
    while (not ::ptx::mbar_try_wait_parity(bar, ph))
        ::ptx::device_assert(++polls < (1ull << 26));
}
static __device__ __forceinline__ void word_spin(const uint32_t* word, const uint32_t target) {
    uint64_t polls = 0;  // wrap-safe: the word is u32, never reset
    while (static_cast<int32_t>(::ptx::ld_acq_fenced(word) - target) < 0)
        ::ptx::device_assert(++polls < (1ull << 26));
}

// a warp's arrive (one lane) past its own tcgen05 accesses
static __device__ __forceinline__ void warp_arrive(uint64_t* bar) {
    ::ptx::tcgen05_fence_before_thread_sync();
    __syncwarp();
    if (::ptx::elect_one_sync()) ::ptx::mbar_arrive(bar);
}

static __device__ __forceinline__ uint32_t bf16x2(const float lo, const float hi) {
    const nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// The attention on CTA j of its kAttnCtas, all 384 threads of the CTA (each warp returns when its role is done).
// kRelease: the CTA adds 1 to the attention word once its partial is out (the draft layer's attention). kGate: the q
// rows and the block's K / V rows are written by this same launch, so after the grid wait the CTA also waits for `gate`
// to reach arrivals x (*ticket + 1) (then fences the async proxy: both are read through TMA). tmap_q: the q rows as
// [B tokens][the rank's qkv heads][128] bf16 (host/runtime.cu q_map). smem: the 1024 B aligned carve
// (attn_layout::kSmem bytes).
template <uint32_t kB, bool kRelease, bool kGate = false>
__device__ __forceinline__ void attn_phase(const DraftAttnArgs& a, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v,
                                           const CUtensorMap* tmap_ws, const CUtensorMap* tmap_q, uint8_t* const smem,
                                           const uint32_t j, const uint32_t* gate = nullptr,
                                           const uint32_t* ticket = nullptr, const uint32_t arrivals = 0) {
    using namespace attn_layout;
    auto bars = reinterpret_cast<uint64_t*>(smem + kOffBars);
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // static prologue (no input read before the grid wait)
    if (warp == kAllocWarp) {
        ::ptx::tcgen05_alloc(::ptx::to_shared(smem + kOffTmemWord), kTmemCols);
    } else if (warp == kInitWarp and lane == 0) {
#pragma unroll
        for (uint32_t i = 0; i < kNumBars; ++i)
            ::ptx::mbar_init(&bars[i], (i == kBQ or i == kBP or i == kBTmem) ? 8u : 1u);
        ::ptx::fence_mbarrier_init_release_cluster();
    } else if (warp == kFeedWarp and lane == 0) {
        // the maps are launch parameters: their fetch rides the predecessor's tail
        ::ptx::prefetch_tensormap(tmap_q);
        ::ptx::prefetch_tensormap(tmap_k);
        ::ptx::prefetch_tensormap(tmap_v);
        ::ptx::prefetch_tensormap(tmap_ws);
    }
    __syncthreads();
    ::ptx::tcgen05_fence_after_thread_sync();
    const uint32_t tbase = *reinterpret_cast<const uint32_t*>(smem + kOffTmemWord);
    ::ptx::griddepcontrol_wait();  // q rows, the block's fresh K / V rows and kv_len are earlier launches' outputs
    if constexpr (kGate) {
        if (tid == 0) word_spin(gate, (*ticket + 1) * arrivals);
        __syncthreads();
        ::ptx::fence_async_global();  // the gated rows are read through TMA too
    }

    const uint32_t len = static_cast<uint32_t>(__ldg(a.kv_len));
    const Window win = window_of<kB>(len);
    // this CTA's (group, block): the group's kv head columns of the ring rows, its 8 q heads of the q rows
    const uint32_t group = group_of_cta(j), jb = j - group * kGroupCtas;
    const bool has = jb < win.blocks;
    const uint32_t kvp0 = win.lo_w + jb * kBlk;

    if (warp == kFeedWarp) {
        if (has and ::ptx::elect_one_sync()) {
            // the group's q rows (B tokens x 8 heads, both halves; token rows past B land zero)
            constexpr uint32_t kQBytes = kB * kGqa * 128;
            ::ptx::mbar_arrive_expect_tx(&bars[kBQt], 2 * kQBytes);
#pragma unroll
            for (uint32_t hh = 0; hh < 2; ++hh)
                ::ptx::cp_async_bulk_tensor_3d_load_hint(
                    ::ptx::to_shared(smem + kOffQ + hh * kQBox), tmap_q, static_cast<int32_t>(hh * 64),
                    static_cast<int32_t>(group * kGqa), 0, &bars[kBQt], ::ptx::kEvictNormal);
            const int32_t row = static_cast<int32_t>(kvp0 & (kRing - 1));
            const int32_t col = static_cast<int32_t>(group * kD);
            ::ptx::mbar_arrive_expect_tx(&bars[kBK], 2 * kChunk);
            ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(smem + kOffK), tmap_k, col, row, &bars[kBK],
                                                     ::ptx::kEvictFirst);
            ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(smem + kOffK + kChunk), tmap_k, col + 64, row,
                                                     &bars[kBK], ::ptx::kEvictFirst);
            ::ptx::mbar_arrive_expect_tx(&bars[kBV], 2 * kChunk);
            ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(smem + kOffV), tmap_v, col, row, &bars[kBV],
                                                     ::ptx::kEvictFirst);
            ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(smem + kOffV + kChunk), tmap_v, col + 64, row,
                                                     &bars[kBV], ::ptx::kEvictFirst);
        }
        return;
    }
    if (warp == kAllocWarp) {
        bar_wait(&bars[kBTmem], 0);
        ::ptx::tcgen05_fence_after_thread_sync();
        ::ptx::tcgen05_dealloc(tbase, kTmemCols);
        return;
    }
    if (warp == kInitWarp) return;
    if (warp == kIssuerWarp) {
        if (has) {
            constexpr uint32_t kM = 128;  // the group's rows (token x 8 + head); B = 8 leaves rows 64-127 dead
            constexpr uint32_t idesc_qk = ::ptx::mma_inst_desc_f16(kM, kBlk, ::ptx::F16Type::BF16, ::ptx::F16Type::BF16,
                                                                   ::ptx::DType::F32, ::ptx::Major::K, ::ptx::Major::K);
            constexpr uint32_t idesc_pv =
                ::ptx::mma_inst_desc_f16(kM, kD, ::ptx::F16Type::BF16, ::ptx::F16Type::BF16, ::ptx::DType::F32,
                                         ::ptx::Major::K, ::ptx::Major::MN);
            bar_wait(&bars[kBQ], 0);
            bar_wait(&bars[kBK], 0);
            ::ptx::tcgen05_fence_after_thread_sync();
            const uint64_t kd0 = ::ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(::ptx::to_shared(smem + kOffK));
#pragma unroll
            for (uint32_t kk = 0; kk < kD / 16; ++kk) {
                // 16 dims = 32 B inside the 128 B swizzle row (+2 in the 16 B address field); dims 64+ in chunk 1
                const uint64_t kd = kd0 + (kk >> 2) * (kChunk >> 4) + (kk & 3) * 2u;
                if (::ptx::elect_one_sync())
                    ::ptx::tcgen05_mma_f16_atmem(tbase + kTS, tbase + kTQ + kk * 8, kd, idesc_qk, kk > 0 ? 1u : 0u);
            }
            if (::ptx::elect_one_sync()) ::ptx::tcgen05_commit_arrive(&bars[kBS]);
            bar_wait(&bars[kBP], 0);
            bar_wait(&bars[kBV], 0);
            ::ptx::tcgen05_fence_after_thread_sync();
            // V is the MN-major B operand: 16 keys = 16 rows x 128 B per K step; dims 64+ one chunk (LBO) away
            const uint64_t vd0 =
                ::ptx::mma_smem_desc_mn_major<uint16_t, kBlk, 128, 128>(::ptx::to_shared(smem + kOffV));
#pragma unroll
            for (uint32_t kk = 0; kk < kBlk / 16; ++kk) {
                // keys [16kk, + 16) are WG (kk / 2)'s P columns [8 (kk % 2), + 8)
                const uint32_t pcol = tbase + kTS + 32 * (kk >> 1) + 8 * (kk & 1);
                if (::ptx::elect_one_sync())
                    ::ptx::tcgen05_mma_f16_atmem(tbase + kTO, pcol, vd0 + kk * 128u, idesc_pv, kk > 0 ? 1u : 0u);
            }
            if (::ptx::elect_one_sync()) ::ptx::tcgen05_commit_arrive(&bars[kBPv]);
        }
        return;
    }

    // ---- warps 0-7: the two softmax WGs -------------------------------------------------------------------------
    const uint32_t r = tid & 127, h = tid >> 7, tok = r >> 3, head = r & 7;
    auto xmax = reinterpret_cast<float*>(smem + kOffXmax);
    auto xsum = reinterpret_cast<float*>(smem + kOffXsum);
    float m = kMSent, lsum = 0.0f;
    if (has) {
        // Q -> TMEM: WG h stages dims [64h, + 64) of row r (32 words), zero past the live rows
        uint4 qv[8];
        bar_wait(&bars[kBQt], 0);
        if (tok < kB) {
            // row r of half h: 16-B chunk i at chunk position i ^ (r & 7) (SWIZZLE_128B): the 8 lanes of a phase read
            // 8 distinct chunk positions
            const uint8_t* const row = smem + kOffQ + h * kQBox + r * 128;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                qv[i] = *reinterpret_cast<const uint4*>(row + ((i ^ (r & 7)) << 4));
        } else {
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                qv[i] = make_uint4(0u, 0u, 0u, 0u);
        }
        ::ptx::tcgen05_st_32x32b_x32(tbase + kTQ + 32 * h, reinterpret_cast<const uint32_t*>(qv));
        ::ptx::tcgen05_wait_st();
        warp_arrive(&bars[kBQ]);

        // row r sees [lo_r, len): a block wholly inside every live row's range takes the fused ld.red max
        const uint32_t posq = len - kB + tok;
        const uint32_t lo_r = posq > kWindow ? posq - kWindow : 0u;
        const bool mfree = kvp0 + kBlk <= len and kvp0 + kWindow + 1 >= len;
        const uint32_t scol = tbase + kTS + 32 * h;
        bar_wait(&bars[kBS], 0);
        ::ptx::tcgen05_fence_after_thread_sync();
        uint32_t lv[32];
        float mh;
        if (mfree) {
            ::ptx::tcgen05_ld_red_max_32x32b_x32(scol, lv, mh);
            ::ptx::tcgen05_wait_ld();
        } else {
            // this WG's 32 keys at base: valid ones as a bit mask; masked logits become -inf (exp2 -> 0)
            const uint32_t base = kvp0 + 32 * h;
            const uint32_t cmax = len > base ? (len - base < 32u ? len - base : 32u) : 0u;
            const uint32_t cmin = lo_r > base ? (lo_r - base < 32u ? lo_r - base : 32u) : 0u;
            const uint32_t valid =
                (cmax >= 32 ? 0xffffffffu : (1u << cmax) - 1u) & (cmin >= 32 ? 0u : ~((1u << cmin) - 1u));
            ::ptx::tcgen05_ld_32x32b_x32(scol, lv);
            ::ptx::tcgen05_wait_ld();
            mh = -__int_as_float(0x7f800000);
#pragma unroll
            for (uint32_t c = 0; c < 32; ++c) {
                if (not((valid >> c) & 1u)) lv[c] = 0xff800000u;
                mh = fmaxf(mh, __uint_as_float(lv[c]));
            }
        }
        // the halves meet: the row's max in log2 units (max commutes with the positive scale exactly)
        xmax[h * 128 + r] = mh;
        ::ptx::sync_aligned(block::kItem, kSoftmaxThreads);
        m = fmaxf(kMSent, fmaxf(xmax[r], xmax[128 + r]) * kScaleLog2);
        uint32_t pw[16];
#pragma unroll
        for (uint32_t cp = 0; cp < 16; ++cp) {
            const float e0 = ::ptx::ex2_approx_ftz_f32(fmaf(__uint_as_float(lv[2 * cp]), kScaleLog2, -m));
            const float e1 = ::ptx::ex2_approx_ftz_f32(fmaf(__uint_as_float(lv[2 * cp + 1]), kScaleLog2, -m));
            lsum += e0 + e1;
            pw[cp] = bf16x2(e0, e1);
        }
        // P over the first 16 of this WG's S columns (read above by this thread only)
        ::ptx::tcgen05_st_32x32b_x16(scol, pw);
        ::ptx::tcgen05_wait_st();
        warp_arrive(&bars[kBP]);

        // O dims [64h, + 64) of row r as bf16 pairs into stage box h (SWIZZLE_128B: 16 B chunk i at i ^ (r & 7))
        bar_wait(&bars[kBPv], 0);
        ::ptx::tcgen05_fence_after_thread_sync();
        if (r < Shape<kB>::kLive) {  // warp-uniform: kLive is 64 or 128
            uint32_t pk[32];
#pragma unroll
            for (uint32_t p = 0; p < 2; ++p) {
                uint32_t ov[32];
                ::ptx::tcgen05_ld_32x32b_x32(tbase + kTO + 64 * h + 32 * p, ov);
                ::ptx::tcgen05_wait_ld();
#pragma unroll
                for (uint32_t c = 0; c < 16; ++c)
                    pk[16 * p + c] = bf16x2(__uint_as_float(ov[2 * c]), __uint_as_float(ov[2 * c + 1]));
            }
            uint8_t* const box = smem + kOffK + h * kStageBox + r * 128;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
                *reinterpret_cast<uint4*>(box + ((i ^ (r & 7)) << 4)) =
                    make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
        }
        if (h == 1) xsum[r] = lsum;
    }
    warp_arrive(&bars[kBTmem]);
    ::ptx::sync_aligned(block::kItem, kSoftmaxThreads);  // WG1's row sums and the stage boxes written
    if (has and h == 0 and r < Shape<kB>::kLive)
        reinterpret_cast<float2*>(smem + kOffMl)[r] = make_float2(m, lsum + xsum[r]);
    ::ptx::fence_async_smem();  // the stage's generic writes ahead of the bulk engine's read
    ::ptx::sync_aligned(block::kItem, kSoftmaxThreads);
    if (tid == 0) {
        if (has) {
            // the partial's stores complete and ordered ahead of the word's release
            const int32_t y = static_cast<int32_t>(j * kWsImageRows);
            ::ptx::cp_async_bulk_tensor_2d_store(::ptx::to_shared(smem + kOffK), tmap_ws, 0, y);
            ::ptx::cp_async_bulk_tensor_2d_store(::ptx::to_shared(smem + kOffK + kStageBox), tmap_ws, 32, y);
            ::ptx::cp_async_bulk_store(a.ws + static_cast<uint64_t>(j) * kWsRow + kWsMl,
                                       ::ptx::to_shared(smem + kOffMl), 8 * Shape<kB>::kLive);
            ::ptx::tma_store_commit();
            ::ptx::cp_async_bulk_wait_group<0>();
            ::ptx::fence_async_global();
        }
        if constexpr (kRelease) ::ptx::red_add_rel_b32(a.word, 1u);
        ::ptx::griddepcontrol_launch_dependents();
    }
}

// ---- the attention's split-KV combine ----------------------------------------------------------------------------
namespace attn_combine {
constexpr uint32_t kEighths = 8, kLaneWords = 6;             // (m, l, acc[4]) per lane per eighth
constexpr uint32_t kXchFloats = kEighths * 32 * kLaneWords;  // the exchange: 6 KB per row
constexpr uint32_t kFlight = 8;                              // every partial of an eighth in flight at once
static_assert(kFlight * kEighths >= kGroupCtas, "one round trip per eighth");
constexpr uint32_t kSmem = 2 * kXchFloats * 4;  // two rows at a time

// one lane's four dims (lane x 4 .. + 3) of the running row, and the row's (m, l)
struct Row {
    float m, l, acc[4];
};

// one partial's (m, l) word and its four dims into the running row
static __device__ __forceinline__ void step(Row& r, const uint2 ml, const uint2 o) {
    const float mk = __uint_as_float(ml.x), lk = __uint_as_float(ml.y);
    const float mn = fmaxf(r.m, mk);
    const float a = ::ptx::ex2_approx_ftz_f32(r.m - mn), bsc = ::ptx::ex2_approx_ftz_f32(mk - mn);
    const float2 o01 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&o.x));
    const float2 o23 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&o.y));
    r.acc[0] = r.acc[0] * a + o01.x * bsc;
    r.acc[1] = r.acc[1] * a + o01.y * bsc;
    r.acc[2] = r.acc[2] * a + o23.x * bsc;
    r.acc[3] = r.acc[3] * a + o23.y * bsc;
    r.l = r.l * a + lk * bsc;
    r.m = mn;
}

static __device__ __forceinline__ void put(float* const xch, const uint32_t w, const uint32_t lane, const Row& r) {
    float* const p = xch + (w * 32 + lane) * kLaneWords;
    p[0] = r.m;
    p[1] = r.l;
    p[2] = r.acc[0];
    p[3] = r.acc[1];
    p[4] = r.acc[2];
    p[5] = r.acc[3];
}

// Phase 1: warp w of eight takes eighth w (partials [w x eighth, min((w + 1) x eighth, nact)), eighth = ceil(nact / 8),
// ascending from a fresh row) of rows row0 AND row1, both rows' loads in flight together, into xch0 / xch1.
static __device__ __forceinline__ void combine_row_pair(const float* const __restrict__ ws, const uint32_t nact,
                                                        const uint32_t row0, const uint32_t row1, const uint32_t w,
                                                        const uint32_t lane, float* const xch0, float* const xch1) {
    const uint32_t eighth = (nact + kEighths - 1) / kEighths;
    const uint32_t kq0 = w * eighth, kq1 = kq0 + eighth < nact ? kq0 + eighth : nact;
    Row r0 = {kMSent, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f}}, r1 = r0;
    uint2 ml0[kFlight], o0[kFlight], ml1[kFlight], o1[kFlight];
#pragma unroll
    for (uint32_t k = 0; k < kFlight; ++k) {
        if (kq0 + k < kq1) {
            const float* const wk = ws + static_cast<uint64_t>(kq0 + k) * kWsRow;
            ml0[k] = ::ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(wk + kWsMl + row0 * 2));
            o0[k] = ::ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(wk + row0 * (kD / 2) + lane * 2));
            ml1[k] = ::ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(wk + kWsMl + row1 * 2));
            o1[k] = ::ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(wk + row1 * (kD / 2) + lane * 2));
        }
    }
#pragma unroll
    for (uint32_t k = 0; k < kFlight; ++k) {
        if (kq0 + k < kq1) {
            step(r0, ml0[k], o0[k]);
            step(r1, ml1[k], o1[k]);
        }
    }
    put(xch0, w, lane, r0);
    put(xch1, w, lane, r1);
}

// Phase 2 (after the barrier over the eight warps): the eighths in order 0..7 from a fresh row, then x = acc / l as
// the row's bf16 x 4 word for dims lane x 4 .. + 3 (every row sees its own position: l > 0)
static __device__ __forceinline__ uint2 combine_final(const float* const __restrict__ xch, const uint32_t lane) {
    Row r = {kMSent, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
    for (uint32_t e = 0; e < kEighths; ++e) {
        const float* const q = xch + (e * 32 + lane) * kLaneWords;
        const float mn = fmaxf(r.m, q[0]);
        const float a = ::ptx::ex2_approx_ftz_f32(r.m - mn), bsc = ::ptx::ex2_approx_ftz_f32(q[0] - mn);
        r.acc[0] = r.acc[0] * a + q[2] * bsc;
        r.acc[1] = r.acc[1] * a + q[3] * bsc;
        r.acc[2] = r.acc[2] * a + q[4] * bsc;
        r.acc[3] = r.acc[3] * a + q[5] * bsc;
        r.l = r.l * a + q[1] * bsc;
        r.m = mn;
    }
    return make_uint2(bf16x2(r.acc[0] / r.l, r.acc[1] / r.l), bf16x2(r.acc[2] / r.l, r.acc[3] / r.l));
}

// Combine CTA c's R = Shape<B>::kCombineRows rows on warps 0-7: tokens R t .. R t + R - 1 of head c / (B / R),
// t = c % (B / R), so a head's rows land on B / R consecutive CTAs; two rows at a time through the exchange (xch:
// kSmem bytes). warp: the eight merging warps' index 0..7 (a composing launch runs the combine on its item warps
// 4..11).
template <uint32_t kB>
static __device__ __forceinline__ void combine_cta(const float* const __restrict__ ws, const uint32_t nact,
                                                   const uint32_t c, nv_bfloat16* const x_attn, float* const xch,
                                                   const uint32_t warp = threadIdx.x >> 5) {
    constexpr uint32_t R = Shape<kB>::kCombineRows;
    static_assert(R % 2 == 0, "rows in pairs");
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t head = c / (kB / R), tok0 = R * (c % (kB / R));
    const uint32_t f = group_of_head(head), fh = head - f * kGqa;  // the group and the head's row offset in it
    const float* const wsf = ws + static_cast<uint64_t>(f) * kGroupCtas * kWsRow;
    const uint32_t row0 = tok0 * kGqa + fh;
#pragma unroll
    for (uint32_t i = 0; i < R; i += 2) {
        if (i > 0) ::ptx::sync_aligned(block::kItem, 256);  // the previous pair's finals have read the exchange
        combine_row_pair(wsf, nact, row0 + i * kGqa, row0 + (i + 1) * kGqa, warp, lane, xch, xch + kXchFloats);
        ::ptx::sync_aligned(block::kItem, 256);
        if (warp == 0 or warp == 4) {
            const uint32_t t = tok0 + i + (warp >> 2);
            const uint2 x = combine_final(xch + (warp >> 2) * kXchFloats, lane);
            *reinterpret_cast<uint2*>(x_attn + static_cast<uint64_t>(t) * kXRow + head * kD + lane * 4) = x;
        }
    }
}
}  // namespace attn_combine

}  // namespace mk::dflash

// The draft layer's row pieces and protocol waits (runs them): bounded spins that name a
// hang, the lamport consume, the gathered-coefficient poll, the conv finish of a drained GEMM tile, the residual add +
// RMSNorm row and the conv prepare (a row, or a prepare unit's tile). Every elementwise step is one IEEE fp32 op (no
// fused multiply-add; sqrt and divide correctly rounded), in the order each comment states.

namespace mk::dflash::layer {

// ---- bounded spins ----
// A cross-CTA wait that outlives the budget records {code, block, thread, observed, target} in host-pinned words
// (the host reads them after the trap has killed the context) and traps, so a
// protocol bug names its wait instead of hanging the job. Warm-ups run in lockstep behind a host barrier: the ranks'
// skew is far below the budget. (This wait, not dflash/common.cuh's: the composed layer image is scheduled around it,
// and the shared one reschedules it.)
inline constexpr uint64_t kSpinBudgetNs = 30000000000ull;
enum Hang : uint32_t { kHangXGate = 1, kHangRowGate = 2, kHangLamport = 3, kHangCoef = 4 };

struct Budget {
    uint64_t t0 = 0;
    uint32_t polls = 0;
    __device__ __forceinline__ bool expired() {
        if ((++polls & 1023u) != 0) return false;
        const uint64_t now = ::ptx::globaltimer_ns();
        if (t0 == 0) t0 = now;
        return now - t0 > kSpinBudgetNs;
    }
};

__device__ __noinline__ void hang(uint64_t* diag, uint32_t code, uint64_t observed, uint64_t target) {
    if (diag != nullptr) {
        volatile uint64_t* d = diag;
        d[1] = blockIdx.x;
        d[2] = threadIdx.x;
        d[3] = observed;
        d[4] = target;
        __threadfence_system();
        d[0] = code;
        __threadfence_system();
    }
    ::ptx::trap();
}

// The ticketed count wait (framework.cuh ticket_wait, bounded): the wrap-safe acquire spin, then the
// async-proxy fence so the caller's TMA reads see the published rows.
__device__ __forceinline__ void count_wait(const uint32_t* word, const uint32_t target, uint64_t* diag,
                                           const uint32_t code) {
    Budget b;
    uint32_t v;
    while (static_cast<int32_t>((v = ptx::ld_acq_fenced(word)) - target) < 0)
        if (b.expired()) hang(diag, code, v, target);
    ::ptx::fence_async_global();
}

// A gathered packet's single read (framework.cuh ag_poll_take_u4, bounded): poll until no half carries the
// empty mark, then re-mark it empty for its next use. Exactly one thread of the launch reads each packet.
__device__ __forceinline__ uint4 poll_take(const uint4* p, uint64_t* diag) {
    Budget b;
    uint4 v;
    for (;;) {
        v = ptx::ld_relaxed_sys_u4(p);
        if (!rowops::ag_packet_has_empty(v)) break;
        if (b.expired()) hang(diag, kHangCoef, reinterpret_cast<uint64_t>(p), 0);
    }
    // four word stores of one all-ones register: a 16-B store needs a register quad, which the tail image's
    // prepare row cannot spare without a spill (the packet's next writer is a later launch either way)
    uint32_t* const w = reinterpret_cast<uint32_t*>(const_cast<uint4*>(p));
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        ::ptx::st_global_b32(w + i, ~0u);
    return v;
}

// The lamport consume of one 16-B packet (framework.cuh ar_lamport_consume_packet8, bounded): every rank's packet
// polled until none carries a +0 marker, then summed in fp32 in rank order and rounded to bf16.
template <uint32_t kRanks>
__device__ __forceinline__ uint4 lamport_consume(const nv_bfloat16* const (&rows)[kRanks], const uint32_t col,
                                                 uint64_t* diag) {
    uint4 pv[kRanks];
    Budget b;
    for (;;) {
        bool marked = false;
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            pv[r] = ptx::ld_relaxed_sys_u4(reinterpret_cast<const uint4*>(rows[r] + col));
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            marked |= rowops::ar_packet8_has_pos_zero(pv[r]);
        if (!marked) break;
        if (b.expired()) hang(diag, kHangLamport, col, 0);
    }
    alignas(16) nv_bfloat162 xb[4];
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        float2 acc = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(&pv[0])[e]);
#pragma unroll
        for (uint32_t r = 1; r < kRanks; ++r) {
            const float2 f = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(&pv[r])[e]);
            acc.x = __fadd_rn(acc.x, f.x);
            acc.y = __fadd_rn(acc.y, f.y);
        }
        xb[e] = __float22bfloat162_rn(acc);
    }
    return *reinterpret_cast<const uint4*>(&xb[0]);
}

// ---- the conv finish of a drained tile (order 'pre_ar') ----
// The thread holds W row 16 quad + lane of the tile (one channel, group 4 j + quad of finish tile j) for every block
// row; coef[n] = row n's packet j [tap 0 of groups 4 j .. 4 j + 3 | tap 1 of the same]. out[n] = (b0 + d0[n]) x p[n] +
// (b1 + d1[n]) x p[n - 1], row 0 without tap 1, rounded once to bf16 into column `out` of the token-major tile.
template <uint32_t kB, uint32_t kM>
__device__ __forceinline__ void finish_tile(const uint32_t (&v)[kM], const uint4* coef, const uint32_t quad,
                                            const float b0, const float b1, nv_bfloat16* out) {
    static_assert(kB <= kM, "the block rows are accumulator columns");
#pragma unroll
    for (uint32_t n = 0; n < kB; ++n) {
        const nv_bfloat16* c = reinterpret_cast<const nv_bfloat16*>(coef + n);
        float o = __fmul_rn(__fadd_rn(b0, __bfloat162float(c[quad])), __int_as_float(v[n]));
        if (n > 0) o = __fadd_rn(o, __fmul_rn(__fadd_rn(b1, __bfloat162float(c[4 + quad])), __int_as_float(v[n - 1])));
        out[n * 64] = __float2bfloat16_rn(o);
    }
}

// ---- the sublayer's all-reduce consume + residual add + RMSNorm, one block row on the width-4 cluster ----
// CTA c's item warp w is virtual warp vw = 8 c + w and owns channels (32 vw + lane) x 8 .. + 7: s = o + r in fp32,
// r' = bf16(s), the sum of squares left to right, the lane butterfly, `round(part)` (the cluster exchange of the 32 vw
// partials, returning their butterfly), rstd = 1 / sqrt(sum x 2^-13 + eps), h = bf16((s x rstd) x w), then
// `on_h(h, col)` with the thread's 8 values. r_in may alias r_out (each thread reads its slice before it writes it).
struct NoH {
    __device__ __forceinline__ void operator()(const nv_bfloat16 (&)[8], const uint32_t) const {}
};
template <uint32_t kRanks, class Round, class OnH = NoH>
__device__ __forceinline__ void ar_norm_row(const uint32_t vw, const uint32_t lane,
                                            const nv_bfloat16* const (&planes)[kRanks], const nv_bfloat16* r_in,
                                            nv_bfloat16* r_out, const nv_bfloat16* w, nv_bfloat16* h, const float eps,
                                            uint64_t* diag, Round&& round, OnH&& on_h = NoH{}) {
    const uint32_t col = (vw * 32 + lane) * 8;
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w + col));
    const uint4 rv = ptx::ld_row_u4_cg(reinterpret_cast<const uint4*>(r_in + col));
    const uint4 ov = lamport_consume<kRanks>(planes, col, diag);
    const auto ob = reinterpret_cast<const nv_bfloat16*>(&ov);
    const auto rb = reinterpret_cast<const nv_bfloat16*>(&rv);
    const auto wb = reinterpret_cast<const nv_bfloat16*>(&wv);
    float s[8];
    float acc = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        s[e] = __fadd_rn(__bfloat162float(ob[e]), __bfloat162float(rb[e]));
        acc = __fadd_rn(acc, __fmul_rn(s[e], s[e]));
    }
    alignas(16) nv_bfloat16 rn[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        rn[e] = __float2bfloat16_rn(s[e]);
    *reinterpret_cast<uint4*>(r_out + col) = *reinterpret_cast<const uint4*>(rn);
    const float total = round(rowops::arnorm_vw_butterfly(acc));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(total, 1.0f / 8192.0f), eps)));
    alignas(16) nv_bfloat16 hn[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        hn[e] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(s[e], rstd), __bfloat162float(wb[e])));
    *reinterpret_cast<uint4*>(h + col) = *reinterpret_cast<const uint4*>(hn);
    on_h(hn, col);
}

// ---- the conv prepare, one block row on one CTA's item warps ----
// Item-warp thread t owns channels 32 t .. 32 t + 31 (groups 2 t and 2 t + 1). The even thread of each lane pair is the
// single reader of the row's coefficient packet t / 2 (groups 4 (t / 2) ..) and hands it to its odd neighbour.
// x[n] = (tap0 + d0[n]) x h[n] + (tap1 + d1[n]) x h[n - 1] (tap0 / tap1: the side-0 base rows), row 0 without tap 1,
// rounded once to bf16.
__device__ __forceinline__ void prep_row(const uint32_t t, const uint32_t lane, const uint32_t n, const uint4* coef_row,
                                         const nv_bfloat16* h_row, const nv_bfloat16* h_prev, const nv_bfloat16* tap0,
                                         const nv_bfloat16* tap1, nv_bfloat16* x_row, uint64_t* diag) {
    constexpr uint32_t kCh = 32;
    const uint32_t c0 = kCh * t;
    uint4 hv[4], hp[4] = {}, bv0[4], bv1[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        bv0[i] = __ldg(reinterpret_cast<const uint4*>(tap0 + c0) + i);
        bv1[i] = __ldg(reinterpret_cast<const uint4*>(tap1 + c0) + i);
        hv[i] = ptx::ld_row_u4_cg(reinterpret_cast<const uint4*>(h_row + c0) + i);
        if (n > 0) hp[i] = ptx::ld_row_u4_cg(reinterpret_cast<const uint4*>(h_prev + c0) + i);
    }
    uint4 pk = make_uint4(0u, 0u, 0u, 0u);
    if ((t & 1u) == 0) pk = poll_take(coef_row + t / 2, diag);
    const uint32_t src = lane & ~1u;
    pk.x = __shfl_sync(0xffffffffu, pk.x, src);
    pk.y = __shfl_sync(0xffffffffu, pk.y, src);
    pk.z = __shfl_sync(0xffffffffu, pk.z, src);
    pk.w = __shfl_sync(0xffffffffu, pk.w, src);
    // this thread's two groups sit in one 32-bit word per tap: groups (4 j, 4 j + 1) or (4 j + 2, 4 j + 3)
    const bool hi = (t & 1u) != 0;
    const uint32_t w0 = hi ? pk.y : pk.x, w1 = hi ? pk.w : pk.z;
    const float2 d0 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&w0));
    const float2 d1 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&w1));
    const auto hb = reinterpret_cast<const nv_bfloat16*>(hv);
    const auto pb = reinterpret_cast<const nv_bfloat16*>(hp);
    const auto b0 = reinterpret_cast<const nv_bfloat16*>(bv0);
    const auto b1 = reinterpret_cast<const nv_bfloat16*>(bv1);
    alignas(16) nv_bfloat16 out[kCh];
#pragma unroll
    for (uint32_t e = 0; e < kCh; ++e) {
        const float c0 = e < 16 ? d0.x : d0.y, c1 = e < 16 ? d1.x : d1.y;
        float o = __fmul_rn(__fadd_rn(__bfloat162float(b0[e]), c0), __bfloat162float(hb[e]));
        if (n > 0) o = __fadd_rn(o, __fmul_rn(__fadd_rn(__bfloat162float(b1[e]), c1), __bfloat162float(pb[e])));
        out[e] = __float2bfloat16_rn(o);
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
        reinterpret_cast<uint4*>(x_row + c0)[i] = reinterpret_cast<const uint4*>(out)[i];
}

// ---- the conv prepare in a prepare claim's epilogue ----
// After the K-quarter reduction the tile holds block rows 4 q .. 4 q + 3 (CTA q) of W block blk's 64 coefficients:
// groups 32 blk .. 32 blk + 31, group 32 blk + j's tap 0 at column 8 (j / 4) + j % 4 and its tap 1 four columns on.
// Epilogue thread
// (quad, lane j) prepares block row n = 4 q + quad over group 32 blk + j's 16 channels with prep_row's arithmetic, from
// its h rows (hv = h[n], hp = h[n - 1]) and the two side-0 base rows' channels (bv0 / bv1), into x_row at the group's
// first channel.
__device__ __forceinline__ void prep_tile(const uint4 (&hv)[2], const uint4 (&hp)[2], const nv_bfloat16* tile,
                                          const uint32_t n, const uint32_t j, const uint4 (&bv0)[2],
                                          const uint4 (&bv1)[2], nv_bfloat16* x_row) {
    const nv_bfloat16* ct = tile + 8 * (j / 4) + j % 4;
    const float d0 = __bfloat162float(ct[0]), d1 = __bfloat162float(ct[4]);
    const auto hb = reinterpret_cast<const nv_bfloat16*>(hv);
    const auto pb = reinterpret_cast<const nv_bfloat16*>(hp);
    const auto b0 = reinterpret_cast<const nv_bfloat16*>(bv0);
    const auto b1 = reinterpret_cast<const nv_bfloat16*>(bv1);
    alignas(16) nv_bfloat16 out[16];
#pragma unroll
    for (uint32_t e = 0; e < 16; ++e) {
        float o = __fmul_rn(__fadd_rn(__bfloat162float(b0[e]), d0), __bfloat162float(hb[e]));
        if (n > 0) o = __fadd_rn(o, __fmul_rn(__fadd_rn(__bfloat162float(b1[e]), d1), __bfloat162float(pb[e])));
        out[e] = __float2bfloat16_rn(o);
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
        reinterpret_cast<uint4*>(x_row)[i] = reinterpret_cast<const uint4*>(out)[i];
}

}  // namespace mk::dflash::layer

#if defined(MK_IMAGE)

// The image a build selects: qkv, or a draft-layer program (a draft layer; the last layer, its rows ending at the final
// norm), every one carrying the attention. Read by the images and by the layer images' host library. The axes
// stay macros: each selects a field group of the parameter block (Params), the layout the host binds.
#define MK_DFLASH_IMG_layer 1
#define MK_DFLASH_IMG_tail 2
#define MK_DFLASH_IMG_qkv 3
#define MK_DFLASH_IMG_CAT_(x) MK_DFLASH_IMG_##x
#define MK_DFLASH_IMG_CAT(x) MK_DFLASH_IMG_CAT_(x)
#define MK_DFLASH_IMG MK_DFLASH_IMG_CAT(MK_IMAGE)
#define MK_DFLASH_ATTN 1
#if MK_DFLASH_IMG == MK_DFLASH_IMG_layer
#define MK_DFLASH_PROG kProgLayer
#elif MK_DFLASH_IMG == MK_DFLASH_IMG_tail
#define MK_DFLASH_PROG kProgTail
#elif MK_DFLASH_IMG != MK_DFLASH_IMG_qkv
#error "dflash/layer.cu: -DMK_IMAGE=qkv|layer|tail"
#endif

#if MK_DFLASH_IMG == MK_DFLASH_IMG_qkv

// A draft layer's qkv on the dense executor (framework.cuh), one image per draft block (-DMK_DFLASH_B=8|16,
// kM = 2B rows): a 4-CTA K split per 64-row W block + the pair fixup's rowops (grid 80, cluster 4). It L2-prefetches
// its whole K slice ahead of the ring, as the target's layer-0 qkv head. host/runtime.cu binds it.

namespace mk::dflash {

template <uint32_t kB>
__launch_bounds__(256, 1) __global__
    void draft_qkv(const __grid_constant__ DenseArgs args, const __grid_constant__ CUtensorMap tmap_w,
                   const __grid_constant__ CUtensorMap tmap_x, const __grid_constant__ CUtensorMap tmap_out,
                   const __grid_constant__ DflashQkvArgs epi) {
    using Q = QkvPins<kB>;
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    dense_body<Q::kM, Q::kStages, Q::kPack, true, 4, DflashQkvPair<kB>, true, true>(args, tmap_w, tmap_x, tmap_out, epi,
                                                                                    smem_buf, blockIdx.x);
}

}  // namespace mk::dflash

#ifndef MK_HOST
[[maybe_unused]] static constexpr auto kKernel = &mk::dflash::draft_qkv<MK_DFLASH_B>;
#endif

#else

// The DFlash draft layer's dense unit queue on the post-attention phase's executor contract (postattn.cuh):
// one persistent launch of 36 width-4 clusters walks a static list of units, claim k of cluster c being entry k x 36 +
// c. A CTA runs its claims in queue order and a unit waits only on EARLIER entries (or on the previous launch, behind
// the grid wait), so the persistent grid never waits on an unclaimed unit.
//
// Units (per rank, B block rows; the full list is the draft layer, a prefix of it the last layer):
//   tile    a 64-row W block per CTA, full K (o_proj K = q heads x 128, down K = kI, gate_up K 8192)
//   ck4     a 64-row W block per cluster (a 4-way cluster K split): CTA q streams K quarter q, each CTA reduces its
//           token quarter in the fixed order ((p0 + p1) + p2) + p3 (the conv coefficient projections, qkv)
//   prepare a conv's side 0 as ck4 units over all 16 W blocks of the side (16 clusters): the CTA that reduces a block
//           prepares that block's 512 channels of the next GEMM's x for its token quarter (prep_tile), so the side-0
//           coefficients never leave the CTA: no gather and no prepare row between the all-reduce rows and the GEMM
//   ar row  one block row of a sublayer (an all-reduce row): lamport consume + residual add + RMSNorm, the row's
//           quarters on the cluster
//   prep    one block row of the conv prepare per CTA
// and the epilogues: fin (the conv finish on the fp32 tile, then the lamport push into the site's planes), gather
// (the coefficient-projection tile into the coefficient rows of the node's 4 ranks), SwiGLU on the gate_up tile
// (dflash/common.cuh) and the qkv's hd128 rowops behind the ck4 reduction, the pair fixup (dflash/common.cuh).
//
// Roles (the dense executor's map): warp 0 streams every claimed GEMM unit's W stages back to back and never waits on
// an activation gate; warp 1 takes the grid wait and each unit's x gate, then its x stages; warp 2 issues the MMAs on
// two alternating TMEM accumulators; warp 3 allocates TMEM; warps 4..7 drain and run the GEMM epilogues; warps 4..11
// run the row units. The ring: kSlots slots of kSlotBytes (four 16-row bf16 stages of 8 KB W + one x k-block), one
// (full, empty) barrier pair per slot; a claim packs stage_pack(m) stages into each slot (its W stages first, then
// their x), so a 32-row claim takes 3 per slot and the ring keeps 20 16-row stages at either B.
//
// Protocols. Every count word gains a power of two per launch and is never reset; a consumer waits for
// arrivals x (ticket + 1) with its CTA's launch ticket (framework.cuh). The lamport planes of a site are
// [2 phases][kRanks][B][8192] bf16; push and consume take ONE phase, (ticket + phase_offset) & 1, with phase_offset
// the draft layer's index, so consecutive draft layers alternate planes and a restore never races the next layer's
// push. Every coefficient packet has exactly one reader per use, which re-marks it empty. The ck4 reduction stage is
// double-buffered, use u in buffer u & 1 at parity (u >> 1) & 1: a peer can send use u + 2 only after it reduced use
// u + 1, which needs this CTA's use-(u + 1) send, which follows this CTA's reduction of use u. Plane reuse: a reader
// restores each packet in its launch; its next launch pushes the other phase only behind its grid wait (that launch
// complete), and a pusher rewrites a phase only after consuming the reader's push of the phase in between, so the
// restore precedes the rewrite through kernel completion and that observation, not a release / acquire pair.
//
// The attention images (layer / layer0 / tail): ONE launch per draft layer. The attention (attn_phase) runs first on
// the last kAttnClusters virtual clusters (9 per kv head), each attention CTA adding 1 to the attention word once its
// partial is out; those clusters then join the queue at round 1 (claim k of attention cluster c is round k + 1).
// Virtual cluster v runs on physical cluster (v + kAttnClusters) % 36: the hardware lands a launch's clusters in
// blockIdx order as the previous launch's CTAs exit, so the attention takes the first-dispatched SMs. The queue leads
// with the combine: Shape<B>::kCombineCtas CTAs on the first kCombineEntries clusters combine kCombineRows attention
// rows each behind the word, store x_attn and count it, and o_proj's x gate waits for the count. The attention's
// barriers are retired (and its TMEM freed) before the queue's carve overlays its bytes. Rounds >= 1 of layer / tail
// visit the combine's clusters first, then the attention's, then the rest, so no cluster streams a second o_proj tile
// behind the combine.
// The first draft layer (image layer0) also leads its queue with layer 0's own qkv (the entry launch wrote its x): the
// attention waits for that unit's count, not only the grid.

namespace mk::dflash::layer {

// the image's axis: the attention ahead of the queue
inline constexpr bool kAttnImage = MK_DFLASH_ATTN != 0;
// the programs (Program below): a draft layer, the last layer (its rows end at the final norm)
enum ProgId : uint32_t { kProgLayer = 2, kProgTail = 3 };

inline constexpr uint32_t kB = MK_DFLASH_B;  // block rows (the draft tokens + the anchor)
static_assert(kB == 8 || kB == 16, "B = 8 (7 drafts) or 16 (15 drafts)");
inline constexpr uint32_t kH = cfg::kHidden, kRanks = cfg::kRanks, kW = cfg::kW, kGrid = cfg::kGrid,
                          kClusters = kGrid / kW;
inline constexpr uint32_t kI = kInter;      // the draft's MLP intermediate per rank
inline constexpr uint32_t kCoefRow = 1024;  // one side's packed coefficients per block row (2 taps x 512 groups)
inline constexpr uint32_t BN = 64;          // W rows per block
// GEMM unit slots of Params: the draft layer's eight, and on an attention image a ninth (the first layer's own qkv)
inline constexpr uint32_t kGemms = kAttnImage ? 9 : 8;
// the x tile's rows: the MMA N of the widest unit (16: B rows padded to the m16 tile; 2 B where the qkv's context rows
// ride: a draft layer's qkv carries the block and its context rows, B = 16)
inline constexpr uint32_t kXM = kAttnImage && MK_DFLASH_PROG != kProgTail && kB == 16 ? 32 : 16;
static_assert(kXM == 16 || kXM == 32, "x k-blocks of 16 or 32 rows");
inline constexpr uint32_t kStageW = BN * 128;
// a claim's x bytes per stage: m rows x 128 B per k-block
__host__ __device__ constexpr uint32_t x_stage_bytes(const uint32_t m) {
    return m * 128;
}
// the slot: four 16-row stages (5 slots)
inline constexpr uint32_t kSlotBytes = 4 * (kStageW + x_stage_bytes(16));
inline constexpr uint32_t kSlots = 5;  // (the carve check below)
// a claim's stages per slot (40 KB slots: 4 for 16 rows, 3 for 32 rows)
__host__ __device__ constexpr uint32_t stage_pack(const uint32_t m) {
    return kSlotBytes / (kStageW + x_stage_bytes(m));
}
static_assert(stage_pack(16) == 4 && (kXM == 16 || stage_pack(32) == 3));
// the packing a claim walks (a 16-row x tile image's units all have 16 rows: constants there)
__device__ __forceinline__ uint32_t pack_of(const uint32_t m) {
    return kXM == 16 ? stage_pack(16) : stage_pack(m);
}
__device__ __forceinline__ uint32_t xbytes_of(const uint32_t m) {
    return kXM == 16 ? x_stage_bytes(16) : x_stage_bytes(m);
}
inline constexpr uint32_t kRingBytes = kSlots * kSlotBytes;
inline constexpr uint32_t kOutOff = kRingBytes;  // [kXM tokens][64] bf16
inline constexpr uint32_t kOutBytes = kXM * BN * 2;
inline constexpr uint32_t kCmbOff = kOutOff + kOutBytes;  // 2 x [64 rows][4 sources x kM / 4 tokens] fp32
inline constexpr uint32_t kCmbBytes = BN * dense_cmb_stride(kXM) * 4;
// an attention image's combine exchange aliases the ck4 reduction stages: a CTA combines its rows before any ck4 claim,
// and a peer's first ck4 partial needs o_proj's all-reduce, so every combine's x_attn, out first
inline constexpr uint32_t kCombineXch = kAttnImage ? attn_combine::kSmem : 0;
inline constexpr uint32_t kCmbRegion = 2 * kCmbBytes > kCombineXch ? 2 * kCmbBytes : kCombineXch;
inline constexpr uint32_t kCoefOff = kCmbOff + kCmbRegion;  // the finish tile's [kB] coefficient packets
inline constexpr uint32_t kTmemOff = kCoefOff + kB * 16;
inline constexpr uint32_t kBarOff = kTmemOff + 16;
// full[kSlots] empty[kSlots] tmem_full[2] tmem_empty[2] cmb_full[2] xchg[2] ticket
inline constexpr uint32_t kBars = 2 * kSlots + 9;
inline constexpr uint32_t kSmemBytes = (kBarOff + kBars * 8 + 1023) / 1024 * 1024;
// the static smem (cstage + s_ticket below, 260 B) rounds up to the dynamic carve's 1024-B alignment
inline constexpr uint32_t kStaticSmem = 1024;
static_assert(kSmemBytes + kStaticSmem <= cfg::kSmemCeil, "the queue's carve fits the 1-CTA/SM ceiling");
inline constexpr uint32_t kAccCols = 32, kTmemCols = 64;  // two accumulators of up to 32 token columns
inline constexpr uint32_t kSlot = kB * kH;                // one rank's lamport slot (bf16 elements)
// arrivals per launch (powers of two: framework.cuh); a count of n arrivers rounds up to a power of two by
// letting its first arrivers add two
inline constexpr uint32_t kRowArrivals = kW * kB;  // an all-reduce row's four CTAs
inline constexpr uint32_t kActTiles = 2 * kI / BN, kActArrivals = handoff::pow2_at_least(kActTiles);
static_assert(kActTiles <= kActArrivals && kActArrivals <= 2 * kActTiles, "the first tiles add two");

enum Kind : uint32_t { kTileFin, kTileSwiglu, kCk4Gather, kCk4Qkv, kCk4Prep, kArRow, kPrepRow, kNone, kCombineRow };
// a unit: its kind, its operand slot (Params.gemm / ar / prep), its epilogue slot (fin / gat), its entries, its MMA N
struct Unit {
    uint32_t kind, arg, epi, entries, m;
};
struct Claim {
    uint32_t kind, arg, epi, m, i;  // i: the entry's index within its unit
};

struct GemmArgs {
    uint32_t stages;         // K / 64 (a ck4 unit: a multiple of 4)
    uint32_t block0;         // the unit's first W block in its map
    const uint32_t* gate;    // x is ready at gate_arrivals x (ticket + 1); nullptr: x is the previous launch's
    uint32_t gate_arrivals;  // per launch
};
// FinArgs.tap, GatherArgs.n_dst and GatherArgs.tap are fixed at bind (null, 4, null) and kept only because removing
// them reschedules the attention images
struct FinArgs {
    const uint64_t* bases;     // [kRanks + 1] the site's plane bases (device addresses), then its multicast address
    const nv_bfloat16* coef;   // the finish's side-1 coefficient rows [kB][kCoefRow] (this rank's, gathered)
    const nv_bfloat16* base1;  // [2 taps][kH]: the conv's side-1 base
    float* tap;                // null
};
struct GatherArgs {
    uint64_t dst[4];      // the side's coefficient rows [kB][kCoefRow] in each of the node's 4 ranks (device addresses)
    uint32_t n_dst;       // 4
    uint32_t dst_block0;  // the unit's first 64-coefficient block in the row
    nv_bfloat16* tap;     // null
};
struct ArArgs {
    const uint64_t* bases;  // the site's plane bases (FinArgs.bases of the same site)
    const nv_bfloat16* r_in;
    nv_bfloat16* r_out;  // may alias r_in
    const nv_bfloat16* w;
    nv_bfloat16* h;
    float eps;
    uint32_t* ready;           // +kRowArrivals per launch
    const uint32_t* rin_gate;  // r_in written in this launch: ready at rin_arrivals x (ticket + 1); else nullptr
    uint32_t rin_arrivals;
};
struct PrepArgs {
    nv_bfloat16* coef;         // the side-0 coefficient rows [kB][kCoefRow] (this rank's, gathered)
    const nv_bfloat16* h;      // [kB][kH]
    const nv_bfloat16* base0;  // [2 taps][kH]: the conv's side-0 base
    nv_bfloat16* x;            // [>= kB][kH]: the next GEMM's x rows
    const uint32_t* h_gate;    // h written in this launch: ready at h_arrivals x (ticket + 1); else nullptr
    uint32_t h_arrivals;
    uint32_t* ready;  // +kB per launch (one per prep row)
};
struct Params {
    CUtensorMap maps[2 * kGemms];  // gemm g: W at 2 g, x at 2 g + 1
    GemmArgs gemm[kGemms];
    FinArgs fin[2];     // the two sites (attention, MLP)
    GatherArgs gat[4];  // mlp_conv_proj side 0 / 1, attn_conv_proj side 0 / 1
    ArArgs ar[2];
    PrepArgs prep[2];
    DflashSwiGlu<16>::Args swiglu;  // the gate_up tile's SwiGLU -> act [kB][kI]
    uint32_t* act_ready;            // +kActArrivals per launch
    DflashQkvArgs qkv;              // the next layer's qkv rows, its q heads and the draft KV ring
    uint32_t* launch_ctr;           // [kGrid]
    uint32_t rank;                  // TP rank
    uint32_t phase_offset;          // the draft layer's index (the planes' phase offset)
    uint64_t* diag;                 // host-pinned hang record (the bounded spins above), or nullptr
    uint8_t* xq;  // the last layer: the draft head's e4m3 rows of y (rows 1 .. B - 1), quantized by its final norm
#if MK_DFLASH_ATTN
    DraftAttnArgs attn;        // this layer's q rows, kv_len, the partials, the attention word
    nv_bfloat16* x_attn;       // the combine's output: o_proj's x (gemm 0)
    uint32_t* x_ready;         // +kCombineCtas per launch: o_proj's x gate
    CUtensorMap attn_maps[4];  // the K ring, the V ring, the partials, the q rows
    DflashQkvArgs qkv0;        // the first layer's own qkv (gemm 8; its rows are this launch's attention's q / K / V)
    uint32_t* q_ready;         // +kQkv0Arrivals per launch: the attention's gate on qkv0
#endif
};

// ---- the programs ----
inline constexpr uint32_t kConvProjEntries = kCoefRow / BN / 4;  // a rank's blocks of one side (its node quarter)
inline constexpr uint32_t kPrepEntries = kCoefRow / BN;       // a prepare unit: the side's 16 blocks, one per cluster
inline constexpr uint32_t kPrepArrivals = kW * kPrepEntries;  // its CTAs add 1 each per launch
template <uint32_t kProg>
struct Program;
// the draft layer's queue: o_proj + attention finish -> rows -> mlp_conv_proj side 0 (a prepare unit) -> mlp_conv_proj
// side 1 -> gate_up -> down + MLP finish -> rows (the next layer's input norm) -> attn_conv_proj side 0 (a prepare
// unit) -> qkv -> attn_conv_proj side 1 (read by the next layer's o_proj finish); the last layer ends at the MLP rows
// (the final norm)
template <bool kTail>
struct ProgramQueue {
    static constexpr uint32_t kUnits = kTail ? 7 : 10;
    __host__ __device__ static constexpr Unit unit(uint32_t u) {
        constexpr uint32_t kK = kConvProjEntries;
        constexpr Unit t[] = {{kTileFin, 0, 0, kH / BN / kW, 16},
                              {kArRow, 0, 0, kB, 0},
                              {kCk4Prep, 1, 0, kPrepEntries, 16},
                              {kCk4Gather, 2, 1, kK, 16},
                              {kTileSwiglu, 3, 0, 2 * kI / BN / kW, 16},
                              {kTileFin, 4, 1, kH / BN / kW, 16},
                              {kArRow, 1, 0, kB, 0},
                              {kCk4Prep, 5, 1, kPrepEntries, 16},
                              {kCk4Qkv, 6, 0, kQkvN / BN, 2 * kB},
                              {kCk4Gather, 7, 3, kK, 16}};
        return t[u];
    }
};
// the attention images: the attention (the last kAttnClusters clusters), then the combine ahead of the layer's queue
// (the combine's CTAs as kCombineEntries cluster entries: combine CTA kW x entry + rank)
inline constexpr uint32_t kAttnClusters = kAttnCtas / kW, kQueueClusters = kClusters - kAttnClusters;
inline constexpr uint32_t kCombineEntries = Shape<kB>::kCombineCtas / kW;
static_assert(Shape<kB>::kCombineCtas % kW == 0 && kCombineEntries <= kQueueClusters,
              "the combine fills the first round");
template <bool kTail>
struct ProgramLayer {
    static constexpr bool kAttn = true, kOs = true;
    static constexpr uint32_t kUnits = ProgramQueue<kTail>::kUnits + 1;
    __host__ __device__ static constexpr Unit unit(uint32_t u) {
        return u == 0 ? Unit{kCombineRow, 0, 0, kCombineEntries, 0} : ProgramQueue<kTail>::unit(u - 1);
    }
};
template <>
struct Program<kProgLayer> : ProgramLayer<false> {};
template <>
struct Program<kProgTail> : ProgramLayer<true> {};
// a first draft layer's own qkv (GEMM slot 8, qkv args 1) ahead of the combine (the gate image layer0's program); the
// attention gates on its count
inline constexpr uint32_t kQkv0Entries = kQkvN / BN, kQkv0Ctas = kW * kQkv0Entries;
inline constexpr uint32_t kQkv0Arrivals =
    handoff::pow2_at_least(kQkv0Ctas);  // the first kQkv0Arrivals - kQkv0Ctas CTAs add two
static_assert(kQkv0Ctas <= kQkv0Arrivals && kQkv0Arrivals <= 2 * kQkv0Ctas, "the first CTAs add two");

// an attention image's virtual cluster of physical cluster c (the attention on the first-dispatched clusters)
template <class P>
__host__ __device__ constexpr uint32_t virtual_cluster(const uint32_t c) {
    if constexpr (requires { P::kAttn; })
        return (c + kQueueClusters) % kClusters;
    else
        return c;
}
// the position of virtual cluster c in rounds >= 1: P::kOs visits the combine's clusters, then the attention's, then
// the rest (a permutation of the clusters)
template <class P>
__host__ __device__ constexpr uint32_t round_pos(const uint32_t c) {
    if constexpr (requires { P::kOs; })
        return c < kCombineEntries ? c : c >= kQueueClusters ? kCombineEntries + c - kQueueClusters : c + kAttnClusters;
    else
        return c;
}

template <class P>
__host__ __device__ constexpr uint32_t queue_len() {
    uint32_t n = 0;
    for (uint32_t u = 0; u < P::kUnits; ++u)
        n += P::unit(u).entries;
    return n;
}
template <class P>
__host__ __device__ constexpr bool program_ok() {
    for (uint32_t u = 0; u < P::kUnits; ++u) {
        const Unit x = P::unit(u);
        if (x.kind < kArRow && (x.m % 16 != 0 || x.m > kXM || x.m > kAccCols)) return false;
        if (x.kind == kPrepRow && kB % kW != 0) return false;
    }
    return true;
}

template <class P>
__device__ __forceinline__ Claim decode(uint32_t e) {
#pragma unroll
    for (uint32_t u = 0; u < P::kUnits; ++u) {
        const Unit x = P::unit(u);
        if (e < x.entries) return {x.kind, x.arg, x.epi, x.m, e};
        e -= x.entries;
    }
    return {kNone, 0, 0, 0, 0};
}

// the ring cursor one warp carries across units
struct RingCursor {
    uint32_t slot = 0, phase = 0;
    __device__ __forceinline__ void advance() {
        slot = slot == kSlots - 1 ? 0u : slot + 1;
        phase ^= slot == 0;
    }
};

inline constexpr uint32_t kEpiBar = block::kEpi, kItemBar = block::kItem, kEpiThreads = 128;

// Prm is the image's parameter block (Params): a template parameter, so the branches an image discards may name the
// fields only the attention images declare
template <class P, class Prm>
__device__ __forceinline__ void queue_body(const Prm& p, uint8_t* smem) {
    static_assert(program_ok<P>(), "every GEMM unit's rows fit the x tile and an accumulator");
    constexpr uint32_t kLen = queue_len<P>();
    const uint32_t smem_w = ptx::to_shared(smem);
    auto out_tile = reinterpret_cast<nv_bfloat16*>(smem + kOutOff);
    auto cmb = reinterpret_cast<float*>(smem + kCmbOff);
    auto coef_s = reinterpret_cast<uint4*>(smem + kCoefOff);
    auto tmem_word = reinterpret_cast<uint32_t*>(smem + kTmemOff);
    uint64_t* const bars = reinterpret_cast<uint64_t*>(smem + kBarOff);
    uint64_t* const full = bars;
    uint64_t* const empty = bars + kSlots;
    uint64_t* const tmem_full = bars + 2 * kSlots;
    uint64_t* const tmem_empty = tmem_full + 2;
    uint64_t* const cmb_full = tmem_empty + 2;
    uint64_t* const xchg = cmb_full + 2;
    uint64_t* const ticket_bar = xchg + 2;
    // the arrow rows' 32 vw partials (st.async from the cluster's CTAs), double-buffered by row: a peer's partial of
    // row k + 2 can land only after this CTA sent its row-(k + 1) partials, each warp after it read row k's
    __shared__ float cstage[2][32];
    __shared__ uint32_t s_ticket;

    const uint32_t warp = ptx::warp_idx_sync(), lane = ptx::lane_idx();
    const uint32_t rank = ptx::cluster_ctarank(), cluster = virtual_cluster<P>(blockIdx.x / kW);
    const auto take = [&](const uint32_t k) {
        uint32_t e = k * kClusters + cluster;
        if constexpr (requires { P::kAttn; }) {
            // the attention's clusters skip round 0 (they run the attention meanwhile); round r >= 1 is entries
            // [kQueueClusters + (r - 1) kClusters, + kClusters)
            const uint32_t round = cluster >= kQueueClusters ? k + 1 : k;
            e = round == 0 ? cluster : kQueueClusters + (round - 1) * kClusters + round_pos<P>(cluster);
        }
        return decode<P>(e < kLen ? e : 0xFFFFFFFFu);
    };
    const auto is_gemm = [](const uint32_t kind) { return kind < kArRow; };
    const auto is_ck4 = [](const uint32_t kind) { return kind == kCk4Gather || kind == kCk4Qkv || kind == kCk4Prep; };
    // a GEMM claim's W block and K loop on this CTA
    const auto geom = [&](const Claim& c, uint32_t& block, uint32_t& k0, uint32_t& iters) {
        const GemmArgs& g = p.gemm[c.arg];
        if (is_ck4(c.kind)) {
            block = g.block0 + c.i;
            iters = g.stages / kW;
            k0 = rank * iters;
        } else {
            block = g.block0 + kW * c.i + rank;
            iters = g.stages;
            k0 = 0;
        }
    };

    // ---- init: barriers (warp 1), TMEM (warp 3) ----
    if (warp == 1 && ptx::elect_one_sync()) {
        if (gridDim.x != kGrid) __trap();  // the static schedule is bound to this grid
        for (uint32_t i = 0; i < kSlots; ++i) {
            ptx::mbar_init(&full[i], 1);
            ptx::mbar_init(&empty[i], 1);
        }
        for (uint32_t b = 0; b < 2; ++b) {
            ptx::mbar_init(&tmem_full[b], 1);
            ptx::mbar_init(&tmem_empty[b], 1);
            ptx::mbar_init(&cmb_full[b], 1);
        }
        for (uint32_t b = 0; b < 2; ++b)
            ptx::mbar_init(&xchg[b], 8);  // the item warps' 8 warps arrive; the 3 peers' 24 partials complete tx
        ptx::mbar_init(ticket_bar, 1);
        ptx::fence_mbarrier_init_release_cluster();
    }
    if (warp == 3) ptx::tcgen05_alloc(ptx::to_shared(tmem_word), kTmemCols);
    ptx::cluster_sync();  // peer barriers armed before any st.async lands
    const uint32_t taddr = tmem_word[0];

    // ---- warp 0: the W stream of every claimed GEMM unit, back to back (weights are never a predecessor's output)
    if (warp == 0) {
        RingCursor w;
        for (uint32_t k = 0;; ++k) {
            const Claim c = take(k);
            if (c.kind == kNone) break;
            if (!is_gemm(c.kind)) continue;
            uint32_t block, k0, iters;
            geom(c, block, k0, iters);
            const CUtensorMap* tm = &p.maps[2 * c.arg];
            const uint32_t pk = pack_of(c.m);
#pragma unroll 1
            for (uint32_t s = 0; s < iters; s += pk) {
                const uint32_t n = iters - s < pk ? iters - s : pk;
                ptx::mbar_wait_parity_poll(&empty[w.slot], w.phase ^ 1);
                if (ptx::elect_one_sync()) {
                    // the slot's x lands as one box of pk k-blocks (warp 1): k-blocks past the claim load too
                    ptx::mbar_arrive_expect_tx(&full[w.slot], n * kStageW + pk * xbytes_of(c.m));
#pragma unroll 1
                    for (uint32_t d = 0; d < n; ++d)
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + w.slot * kSlotBytes + d * kStageW, tm,
                                                               int32_t((k0 + s + d) * 64u), int32_t(block * BN),
                                                               &full[w.slot], ptx::kEvictFirst);
                }
                __syncwarp();
                w.advance();
            }
        }
    }

    // ---- warp 1: the grid wait, the launch ticket, each unit's x gate and x stages ----
    if (warp == 1) {
        ptx::griddepcontrol_wait();
        // every read of the previous launch's outputs follows a wait like this one, so the dependent launch may
        // start its weight streams now
        if (ptx::elect_one_sync()) ptx::griddepcontrol_launch_dependents();
        const uint32_t ticket = p.launch_ctr[blockIdx.x];
        if (ptx::elect_one_sync()) {
            s_ticket = ticket;
            ptx::mbar_arrive(ticket_bar);
        }
        RingCursor w;
        for (uint32_t k = 0;; ++k) {
            const Claim c = take(k);
            if (c.kind == kNone) break;
            if (!is_gemm(c.kind)) continue;
            const GemmArgs& g = p.gemm[c.arg];
            // every lane takes the gate, so whichever lane issues the TMA has acquired and fenced the async proxy
            if (g.gate != nullptr) count_wait(g.gate, g.gate_arrivals * (ticket + 1), p.diag, kHangXGate);
            uint32_t block, k0, iters;
            geom(c, block, k0, iters);
            const CUtensorMap* tm = &p.maps[2 * c.arg + 1];
            const uint32_t pk = pack_of(c.m);
#pragma unroll 1
            for (uint32_t s = 0; s < iters; s += pk) {
                const uint32_t n = iters - s < pk ? iters - s : pk;
                ptx::mbar_wait_parity_poll(&empty[w.slot], w.phase ^ 1);
                // one 3D box per slot: [pk k-blocks][m rows][64] (the x map's dims: 64 columns, rows, k-blocks)
                if (ptx::elect_one_sync())
                    ptx::cp_async_bulk_tensor_3d_load_hint(smem_w + w.slot * kSlotBytes + pk * kStageW, tm, 0, 0,
                                                           int32_t(k0 + s), &full[w.slot], ptx::kEvictNormal);
                __syncwarp();
                w.advance();
            }
        }
        // this launch's ticket consumed: the next launch of this binding reads it + 1
        if (ptx::elect_one_sync()) p.launch_ctr[blockIdx.x] = ticket + 1;
    }

    // ---- warp 2: the MMAs, unit g on accumulator g & 1 behind that accumulator's previous drain ----
    if (warp == 2) {
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        // x tiles sit at slot offsets (stage_pack x kStageW + d x x_stage_bytes): every offset a multiple of 1 KB, so
        // the SWIZZLE_128B descriptor of the ring base advances by offset / 16
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        RingCursor w;
        uint32_t g = 0;
        for (uint32_t k = 0;; ++k) {
            const Claim c = take(k);
            if (c.kind == kNone) break;
            if (!is_gemm(c.kind)) continue;
            uint32_t block, k0, iters;
            geom(c, block, k0, iters);
            const uint32_t idesc = c.m == 16 ? ptx::mma_inst_desc_f16(BN, 16) : ptx::mma_inst_desc_f16(BN, 32);
            const uint32_t pk = pack_of(c.m), xb = xbytes_of(c.m);
            const uint32_t b = g & 1u;
            if (g >= 2) ptx::mbar_wait_parity_poll(&tmem_empty[b], ((g >> 1) - 1) & 1u);
            ptx::tcgen05_fence_after_thread_sync();
#pragma unroll 1
            for (uint32_t s = 0; s < iters; s += pk) {
                const uint32_t n = iters - s < pk ? iters - s : pk;
                ptx::mbar_wait_parity_poll(&full[w.slot], w.phase);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one_sync()) {
#pragma unroll 1
                    for (uint32_t d = 0; d < n; ++d) {
#pragma unroll
                        for (uint32_t kk = 0; kk < 4; ++kk) {
                            const uint64_t da = w_desc0 + (w.slot * kSlotBytes + d * kStageW) / 16 + kk * 2;
                            const uint64_t db = x_desc0 + (w.slot * kSlotBytes + pk * kStageW + d * xb) / 16 + kk * 2;
                            ptx::tcgen05_mma_f16(taddr + kAccCols * b, da, db, idesc, (s | d | kk) != 0 ? 1u : 0u);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&empty[w.slot]);
                    if (s + pk >= iters) ptx::tcgen05_commit_arrive(&tmem_full[b]);
                }
                __syncwarp();
                w.advance();
            }
            ++g;
        }
        // the last commit on each slot's empty barrier is never retired by a refill: drain them before the exit
        for (uint32_t s = 0; s < kSlots; ++s)
            ptx::mbar_wait_parity_poll(&empty[s], s < w.slot ? w.phase : w.phase ^ 1u);
    }

    // ---- warps 4..11: the GEMM epilogues (warps 4..7) and the row units (all eight) ----
    if (warp >= 4) {
        const uint32_t quad = warp - 4, item_warp = warp - 4, tid = quad * 32 + lane;
        const bool epi = warp < 8;
        ptx::mbar_wait_parity_poll(ticket_bar, 0);
        const uint32_t ticket = s_ticket;
        const uint32_t phase = (ticket + p.phase_offset) & 1u;
        // the item warps' first row reads the previous launch's outputs
        ptx::griddepcontrol_wait();
        uint32_t g = 0, ck4_uses = 0, rows = 0;
        // the accumulator into registers (lane < 16 of quad q: W row 16 q + lane, every token column), released
        const auto drain = [&]<uint32_t kM>(uint32_t (&v)[kM]) {
            const uint32_t b = g & 1u;
            ptx::mbar_wait_parity_poll(&tmem_full[b], (g >> 1) & 1u);
            ptx::tcgen05_fence_after_thread_sync();
#pragma unroll
            for (uint32_t j = 0; j < kM; j += 16)
                ptx::tcgen05_ld_32x32b_x16(taddr + kAccCols * b + j, v[j], v[j + 1], v[j + 2], v[j + 3], v[j + 4],
                                           v[j + 5], v[j + 6], v[j + 7], v[j + 8], v[j + 9], v[j + 10], v[j + 11],
                                           v[j + 12], v[j + 13], v[j + 14], v[j + 15]);
            ptx::tcgen05_wait_ld();
            ptx::sync_aligned(kEpiBar, kEpiThreads);
            if (warp == 4 && ptx::elect_one_sync()) ptx::mbar_arrive(&tmem_empty[b]);
            ++g;
        };
        // the ck4 reduction: ship each peer its token quarter's partial, reduce this CTA's quarter over the four in
        // rank order into the out tile (bf16)
        const auto reduce_part = [&]<uint32_t kM>(const uint32_t (&v)[kM]) {
            constexpr uint32_t kQ = kM / 4, kStride = dense_cmb_stride(kM);
            static_assert(kQ % 4 == 0, "a quarter's tokens ship as 16-B packets of four");
            const uint32_t buf = ck4_uses & 1u, par = (ck4_uses >> 1) & 1u;
            float* const stage = cmb + buf * (kCmbBytes / 4);
            if (lane < 16) {
                const uint32_t row = quad * 16 + lane;
                const uint32_t bar_l = ptx::to_shared(&cmb_full[buf]);
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) {
                    if (q == rank) continue;
                    const uint32_t dst = ptx::mapa_shared_cluster(ptx::to_shared(stage + row * kStride + rank * kQ), q);
                    const uint32_t bar = ptx::mapa_shared_cluster(bar_l, q);
#pragma unroll
                    for (uint32_t j = 0; j < kQ; j += 4)
                        ptx::st_async_cluster_v4(
                            dst + j * 4,
                            make_uint4(v[q * kQ + j], v[q * kQ + j + 1], v[q * kQ + j + 2], v[q * kQ + j + 3]), bar);
                }
            }
            ptx::mbar_wait_parity_acquire_cluster(&cmb_full[buf], par);
            if (lane < 16) {
                const uint32_t row = quad * 16 + lane;
                const auto reduce_q = [&](auto qc) {
                    constexpr uint32_t Q = decltype(qc)::value;
#pragma unroll
                    for (uint32_t j = 0; j < kQ; ++j) {
                        float part[4];
#pragma unroll
                        for (uint32_t src = 0; src < 4; ++src)
                            part[src] = src == Q ? __int_as_float(v[Q * kQ + j]) : stage[row * kStride + src * kQ + j];
                        out_tile[(Q * kQ + j) * BN + row] =
                            __float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(part[0], part[1]), part[2]), part[3]));
                    }
                };
                if (rank == 0)
                    reduce_q(std::integral_constant<uint32_t, 0>{});
                else if (rank == 1)
                    reduce_q(std::integral_constant<uint32_t, 1>{});
                else if (rank == 2)
                    reduce_q(std::integral_constant<uint32_t, 2>{});
                else
                    reduce_q(std::integral_constant<uint32_t, 3>{});
            }
            ++ck4_uses;
            ptx::sync_aligned(kEpiBar, kEpiThreads);
        };
        // the stage's expected bytes for this use, armed ahead of the drain (a landing ahead of the arm is legal:
        // the barrier's count of one holds the phase open)
        const auto arm_reduce = [&](const uint32_t m) {
            if (warp == 4 && ptx::elect_one_sync()) ptx::mbar_arrive_expect_tx(&cmb_full[ck4_uses & 1u], 3 * BN * m);
        };
        for (uint32_t k = 0;; ++k) {
            const Claim c = take(k);
            if (c.kind == kNone) break;
            if (c.kind == kTileFin) {
                if (epi) {
                    const FinArgs& f = p.fin[c.epi];
                    uint32_t block, k0, iters;
                    geom(c, block, k0, iters);
                    const uint32_t w_outer = block * BN, ch = w_outer + quad * 16 + (lane & 15u);
                    // the tile's coefficient packets, one reader each, ahead of the accumulator
                    if (tid < kB)
                        coef_s[tid] =
                            poll_take(reinterpret_cast<const uint4*>(f.coef + tid * kCoefRow) + block, p.diag);
                    const float b0 = __bfloat162float(f.base1[ch]), b1 = __bfloat162float(f.base1[kH + ch]);
                    uint32_t v[16];
                    // its barrier also publishes coef_s
                    drain(v);
                    if (lane < 16) {
                        if (f.tap != nullptr)
#pragma unroll
                            for (uint32_t n = 0; n < kB; ++n)
                                f.tap[n * kH + ch] = __int_as_float(v[n]);
                        finish_tile<kB, 16>(v, coef_s, quad, b0, b1, out_tile + quad * 16 + lane);
                    }
                    ptx::sync_aligned(kEpiBar, kEpiThreads);
                    // the lamport push: each packet marked (+0 -> -0) and stored once to the site's multicast
                    // address, into slot [phase][this rank] of every rank's planes
                    const uint64_t slot0 = f.bases[kRanks] + (uint64_t(phase) * kRanks + p.rank) * kSlot * 2;
                    if (tid < kB * 8) {
                        const uint32_t t = tid >> 3, pk = tid & 7u;
                        uint4 m = *reinterpret_cast<const uint4*>(out_tile + t * BN + pk * 8);
                        rowops::ar_lamport_mark_packet8(m);
                        ptx::multimem_st_relaxed_sys_v4_bf16x2(
                            reinterpret_cast<uint32_t*>(slot0 + (uint64_t(t) * kH + w_outer + pk * 8) * 2), m);
                    }
                    ptx::sync_aligned(kEpiBar, kEpiThreads);  // the out tile is free
                }
            } else if (c.kind == kTileSwiglu) {
                if (epi) {
                    uint32_t block, k0, iters;
                    geom(c, block, k0, iters);
                    uint32_t v[16];
                    drain(v);
                    // the hook reads the bf16 tile (the dense executor's tile-hook contract: after the out-tile sync)
                    if (lane < 16)
#pragma unroll
                        for (uint32_t n = 0; n < 16; ++n)
                            out_tile[n * BN + quad * 16 + lane] = __float2bfloat16_rn(__int_as_float(v[n]));
                    ptx::sync_aligned(kEpiBar, kEpiThreads);
                    DflashSwiGlu<16>::pre_store(p.swiglu, block * BN, out_tile, quad, lane);
                    // act is the down projection's TMA input: each writer's async-proxy fence, then one release
                    ::ptx::fence_async_global();
                    ptx::sync_aligned(kEpiBar, kEpiThreads);
                    if (tid == 0) ptx::red_add_rel_b32(p.act_ready, block < kActArrivals - kActTiles ? 2u : 1u);
                }
            } else if (c.kind == kCk4Gather) {
                if (epi) {
                    const GatherArgs& a = p.gat[c.epi];
                    arm_reduce(16);
                    uint32_t v[16];
                    drain(v);
                    reduce_part(v);
                    // the quarter's tokens (4 per CTA) x 8 packets, each to the node's 4 ranks' rows (system scope:
                    // the readers poll their own memory)
                    constexpr uint32_t kQ = 4;
                    const uint32_t pk = tid & 31u, d = tid >> 5, t = rank * kQ + pk / 8;
                    const uint32_t col = (a.dst_block0 + c.i) * BN + (pk & 7u) * 8;
                    if (t < kB) {
                        const uint4 val = *reinterpret_cast<const uint4*>(out_tile + t * BN + (pk & 7u) * 8);
                        if (d < a.n_dst)
                            ptx::st_relaxed_sys_u4(
                                reinterpret_cast<uint4*>(a.dst[d] + (uint64_t(t) * kCoefRow + col) * 2), val);
                        if (d == 0 && a.tap != nullptr) *reinterpret_cast<uint4*>(a.tap + t * kCoefRow + col) = val;
                    }
                    ptx::sync_aligned(kEpiBar, kEpiThreads);  // the out tile is free
                }
            } else if (c.kind == kCk4Prep) {
                if (epi) {
                    const GemmArgs& g = p.gemm[c.arg];
                    const PrepArgs& a = p.prep[c.epi];
                    const uint32_t n = rank * 4 + quad, c0 = ((g.block0 + c.i) * 32 + lane) * 16;
                    arm_reduce(16);
                    // the h rows (this launch's all-reduce rows, the GEMM's own x: each warp acquires their count,
                    // which the GEMM's x gate also waits on) and the side-0 base taps load ahead of the drain
                    if (lane == 0) count_wait(g.gate, g.gate_arrivals * (ticket + 1), p.diag, kHangRowGate);
                    __syncwarp();
                    uint4 hv[2] = {}, hp[2] = {}, bv0[2], bv1[2];
#pragma unroll
                    for (uint32_t i = 0; i < 2; ++i) {
                        bv0[i] = __ldg(reinterpret_cast<const uint4*>(a.base0 + c0) + i);
                        bv1[i] = __ldg(reinterpret_cast<const uint4*>(a.base0 + kH + c0) + i);
                    }
                    if (n < kB)
#pragma unroll
                        for (uint32_t i = 0; i < 2; ++i) {
                            hv[i] = ptx::ld_row_u4_cg(reinterpret_cast<const uint4*>(a.h + n * kH + c0) + i);
                            if (n > 0)
                                hp[i] = ptx::ld_row_u4_cg(reinterpret_cast<const uint4*>(a.h + (n - 1) * kH + c0) + i);
                        }
                    uint32_t v[16];
                    drain(v);
                    reduce_part(v);
                    if (n < kB) prep_tile(hv, hp, out_tile + n * BN, n, lane, bv0, bv1, a.x + n * kH + c0);
                    // x is the next GEMM's TMA input: the writers' async-proxy fence, the barrier, one release per CTA
                    ::ptx::fence_async_global();
                    ptx::sync_aligned(kEpiBar, kEpiThreads);
                    if (tid == 0) ptx::red_add_rel_b32(a.ready, 1u);
                }
            } else if (c.kind == kCk4Qkv) {
                if (epi) {
                    uint32_t block, k0, iters;
                    geom(c, block, k0, iters);
                    constexpr uint32_t kM = 2 * kB, kQ = kM / 4;
                    const DflashQkvArgs* qa_of = &p.qkv;  // an attention image's first layer: its own qkv (qkv args 1)
                    if constexpr (kAttnImage) qa_of = c.epi == 1 ? &p.qkv0 : &p.qkv;
                    const DflashQkvArgs& qa = *qa_of;
                    // the fixup's row operands load ahead of the drain
                    const auto ops = DflashQkvPair<kB>::template ck4_pair_ops<kM>(qa, block * BN, rank, warp, lane);
                    arm_reduce(kM);
                    uint32_t v[kM];
                    drain(v);
                    reduce_part(v);
                    // the quarter's GEMM rows by generic stores, then the pair fixup: the second CTA of each
                    // (head, token quarter) pair reads its partner's half back from L2 behind the pair's ticket
                    if (tid < kQ * 8) {
                        const uint32_t t = rank * kQ + tid / 8, j = tid % 8;
                        *reinterpret_cast<uint4*>(qa.out + uint64_t(t) * qa.out_stride + block * BN + j * 8) =
                            *reinterpret_cast<const uint4*>(out_tile + t * BN + j * 8);
                    }
                    DflashQkvPair<kB>::template ck4_pair_fixup_ops<kM>(qa, ops, block * BN, rank, warp, lane, out_tile,
                                                                       tid == 0);
                    if constexpr (kAttnImage)
                        if (c.epi == 1) {
                            // the attention reads these rows through TMA (K / V) and coherent ld.global.cg loads (q):
                            // every writer fences the async proxy, then one release per CTA (every CTA, so the second
                            // of each pair has stored by the time the count is complete)
                            ::ptx::fence_async_global();
                            ptx::sync_aligned(kEpiBar, kEpiThreads);
                            const uint32_t cta = kW * c.i + rank;
                            if (tid == 0) ptx::red_add_rel_b32(p.q_ready, cta < kQkv0Arrivals - kQkv0Ctas ? 2u : 1u);
                        }
                }
            } else if (c.kind == kArRow) {
                const ArArgs& a = p.ar[c.arg];
                const uint32_t row = c.i, vw = rank * 8 + item_warp;
                if (a.rin_gate != nullptr) {
                    if (lane == 0) count_wait(a.rin_gate, a.rin_arrivals * (ticket + 1), p.diag, kHangRowGate);
                    __syncwarp();
                }
                const nv_bfloat16* const mine = reinterpret_cast<const nv_bfloat16*>(a.bases[p.rank]);
                const nv_bfloat16* planes[kRanks];
#pragma unroll
                for (uint32_t r = 0; r < kRanks; ++r)
                    planes[r] = mine + (uint64_t(phase) * kRanks + r) * kSlot + row * kH;
                // the last layer's final norm also quantizes y rows 1 .. B - 1 for the draft head (its x: amax over
                // the row, 448 / amax, e4m3 round to nearest even, saturating)
                const auto xq_rows = [&](const nv_bfloat16(&hn)[8], const uint32_t col) {
                    if constexpr (MK_DFLASH_PROG == kProgTail) {
                        if (c.arg != 1 || row == 0) return;
                        float m = 0.0f;
#pragma unroll
                        for (uint32_t e = 0; e < 8; ++e)
                            m = fmaxf(m, fabsf(__bfloat162float(hn[e])));
#pragma unroll
                        for (uint32_t o = 16; o >= 1; o >>= 1)
                            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                        // the row's amax over the cluster: the 32 vw maxima exchanged as the norm's partials are
                        const uint32_t buf = rows & 1u;
                        if (lane == 0) {
                            cstage[buf][vw] = m;
                            const uint32_t src = ptx::to_shared(&cstage[buf][vw]), sbar = ptx::to_shared(&xchg[buf]);
#pragma unroll 1
                            for (uint32_t q = 1; q < kW; ++q) {
                                const uint32_t peer = (rank + q) % kW;
                                ptx::st_async_cluster_b32(ptx::mapa_shared_cluster(src, peer), __float_as_uint(m),
                                                          ptx::mapa_shared_cluster(sbar, peer));
                            }
                            if (item_warp == 0)
                                ptx::mbar_arrive_expect_tx(&xchg[buf], (kW - 1) * 8 * 4);
                            else
                                ptx::mbar_arrive(&xchg[buf]);
                        }
                        ptx::mbar_wait_parity_acquire_cluster_poll(&xchg[buf], (rows >> 1) & 1u);
                        ++rows;
                        float amax = cstage[buf][lane];
#pragma unroll
                        for (uint32_t o = 16; o >= 1; o >>= 1)
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                        const float inv = amax > 0.f ? __fdiv_rn(448.f, amax) : 1.f;
                        uint32_t q8[2];
#pragma unroll
                        for (uint32_t q = 0; q < 4; ++q) {
                            const float2 f = make_float2(__fmul_rn(__bfloat162float(hn[2 * q]), inv),
                                                         __fmul_rn(__bfloat162float(hn[2 * q + 1]), inv));
                            const uint32_t pq = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
                            if (q & 1)
                                q8[q / 2] |= pq << 16;
                            else
                                q8[q / 2] = pq;
                        }
                        *reinterpret_cast<uint2*>(p.xq + uint64_t(row - 1) * kH + col) = make_uint2(q8[0], q8[1]);
                    }
                };
                ar_norm_row<kRanks>(
                    vw, lane, planes, a.r_in + row * kH, a.r_out + row * kH, a.w, a.h + row * kH, a.eps, p.diag,
                    [&](const float part) {
                        // the exchange as a scatter (postattn.cuh): lane 0 of each warp writes its partial
                        // into every CTA's cstage[vw] (st.async to the peers, tx-counted on their barrier), so the
                        // 32 partials land in vw order on every CTA and the butterfly is the same everywhere
                        const uint32_t buf = rows & 1u;
                        if (lane == 0) {
                            cstage[buf][vw] = part;
                            const uint32_t src = ptx::to_shared(&cstage[buf][vw]), sbar = ptx::to_shared(&xchg[buf]);
#pragma unroll 1
                            for (uint32_t q = 1; q < kW; ++q) {
                                const uint32_t peer = (rank + q) % kW;
                                ptx::st_async_cluster_b32(ptx::mapa_shared_cluster(src, peer), __float_as_uint(part),
                                                          ptx::mapa_shared_cluster(sbar, peer));
                            }
                            if (item_warp == 0)
                                ptx::mbar_arrive_expect_tx(&xchg[buf], (kW - 1) * 8 * 4);
                            else
                                ptx::mbar_arrive(&xchg[buf]);
                        }
                        ptx::mbar_wait_parity_acquire_cluster_poll(&xchg[buf], (rows >> 1) & 1u);
                        ++rows;
                        return rowops::arnorm_vw_butterfly(cstage[buf][lane]);
                    },
                    xq_rows);
                // publish first, restore after: the readers need h and r, not the cleared planes (a plane is
                // pushed again only by a later launch, behind its grid wait)
                ::ptx::fence_async_global();
                ptx::sync_aligned(kItemBar, 256);
                if (item_warp == 0 && lane == 0) ptx::red_add_rel_b32(a.ready, 1u);
                nv_bfloat16* restore[kRanks];
#pragma unroll
                for (uint32_t r = 0; r < kRanks; ++r)
                    restore[r] = const_cast<nv_bfloat16*>(planes[r]);
                rowops::ar_lamport_restore_packet8<kRanks>(restore, (vw * 32 + lane) * 8);
            } else if (c.kind == kCombineRow) {
                if constexpr (kAttnImage) {
                    // the attention combine: its rows behind the attention word (the attention CTAs add 1 each per
                    // launch), x_attn out as o_proj's TMA input, then one release per CTA
                    // kv_len is an earlier iteration's word: read ahead of the spin
                    const uint32_t nact = window_of<kB>(uint32_t(__ldg(p.attn.kv_len))).blocks;
                    if (item_warp == 0 && lane == 0) word_spin(p.attn.word, (ticket + 1) * kAttnCtas);
                    ptx::sync_aligned(kItemBar, 256);
                    attn_combine::combine_cta<kB>(p.attn.ws, nact, kW * c.i + rank, p.x_attn,
                                                  reinterpret_cast<float*>(smem + kCmbOff), item_warp);
                    ::ptx::fence_async_global();
                    ptx::sync_aligned(kItemBar, 256);
                    if (item_warp == 0 && lane == 0) ptx::red_add_rel_b32(p.x_ready, 1u);
                }
            } else if (c.kind == kPrepRow) {
                const PrepArgs& a = p.prep[c.arg];
                const uint32_t row = kW * c.i + rank;
                if (a.h_gate != nullptr) {
                    if (lane == 0) count_wait(a.h_gate, a.h_arrivals * (ticket + 1), p.diag, kHangRowGate);
                    __syncwarp();
                }
                prep_row(item_warp * 32 + lane, lane, row, reinterpret_cast<const uint4*>(a.coef + row * kCoefRow),
                         a.h + row * kH, a.h + (row > 0 ? row - 1 : 0) * kH, a.base0, a.base0 + kH, a.x + row * kH,
                         p.diag);
                // x is the next GEMM's TMA input: the writers' async-proxy fence, the item barrier, one release
                ::ptx::fence_async_global();
                ptx::sync_aligned(kItemBar, 256);
                if (item_warp == 0 && lane == 0) ptx::red_add_rel_b32(a.ready, 1u);
            }
            __syncwarp();
        }
        if (warp == 4) ptx::tcgen05_dealloc(taddr, kTmemCols);
    }
    // the peers' DSMEM traffic (cstage, the attention combine stages) completes before any CTA exits
    ptx::cluster_sync();
}

// An attention image: the attention on the last kAttnCtas virtual CTAs (whole clusters), then every CTA's queue. The
// attention's barriers are retired before the queue's ring overlays their bytes (its TMEM is freed inside the phase).
template <class P, class Prm>
__device__ __forceinline__ void attn_layer_body(const Prm& p, uint8_t* smem) {
    static_assert(requires { P::kAttn; } && kAttnCtas == kAttnClusters * kW);
    static_assert(attn_layout::kSmem <= kSmemBytes);
    constexpr uint32_t kFirst = kGrid - kAttnCtas;
    constexpr bool kGate = requires { P::kHeadQkv; };
    // the attention's gate entries (the first layer's own qkv) are the queue clusters' claims of rounds 0 and 1: an
    // attention cluster's first claim (round 1, after the queue clusters') must not be one of them (TP4's 40 entries
    // are)
    static_assert(!kGate || kQkv0Entries <= 2 * kQueueClusters,
                  "the attention never waits on an attention cluster's claim");
    const uint32_t vblock = virtual_cluster<P>(blockIdx.x / kW) * kW + blockIdx.x % kW;
    if (vblock >= kFirst) {
        // the gate's target: qkv0's arrivals x (this CTA's launch ticket + 1), read after the attention's grid wait
        attn_phase<kB, true, kGate>(p.attn, &p.attn_maps[0], &p.attn_maps[1], &p.attn_maps[2], &p.attn_maps[3], smem,
                                    vblock - kFirst, kGate ? p.q_ready : nullptr, p.launch_ctr + blockIdx.x,
                                    kQkv0Arrivals);
        __syncthreads();  // every wait on the attention's barriers retired
        if (threadIdx.x < attn_layout::kNumBars)
            ptx::mbar_inval(reinterpret_cast<uint64_t*>(smem + attn_layout::kOffBars) + threadIdx.x);
    }
    queue_body<P>(p, smem);
}

}  // namespace mk::dflash::layer

namespace mk::dflash {

// The draft layer kernel (one per program): grid 144 = 36 width-4 clusters, 384 threads, layer::kSmemBytes of dynamic
// smem, PDL; an attention image runs the attention ahead of the queue.
template <uint32_t kProg>
__global__ void __launch_bounds__(384, 1) draft_layer(const __grid_constant__ layer::Params p) {
    extern __shared__ __align__(1024) uint8_t smem[];
    if constexpr (layer::kAttnImage)
        layer::attn_layer_body<layer::Program<kProg>>(p, smem);
    else
        layer::queue_body<layer::Program<kProg>>(p, smem);
}

}  // namespace mk::dflash

#ifndef MK_HOST
[[maybe_unused]] static constexpr auto kLayerKernel = &mk::dflash::draft_layer<mk::dflash::layer::MK_DFLASH_PROG>;
#endif

#endif
#endif
