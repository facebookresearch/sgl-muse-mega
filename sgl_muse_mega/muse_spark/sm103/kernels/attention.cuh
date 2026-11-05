#pragma once
// The attention phase of a layer image: split-KV decode attention over the window and its combine (gate-norm) into
// x_attn. Its constants, its arguments, the shared machinery, the global attention (tcgen05), the mma.sync attention
// loop of the sliding-window layers and the combine, and the global layers' attention policy.

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "ptx.cuh"
#include "framework.cuh"

// The attention's constants. The window's rows and heads are the build's (framework.cuh); the attention type
// (kImage.global_attn) sets the attention variant, its carve, its exp2 variant and its window. The query rows per
// sequence are the window's: 16, or 8 for an 8-row window. The feeds' kv_len is the end of their 16 rows (P + 16) in
// every binding: an 8-row attention's window ends 8 rows earlier, on the 16-row split grid and blocks, so its rows
// merge the same chunks in the same order as the 16-row window's first 8 rows. -DMK_DFLASH_TREE=1: the window is a
// DFlash token tree (dflash/tree.cuh): row r sits at position P + depth(r) and sees the window keys of its ancestor
// mask, both read from the tree table that follows the feeds' kv_len word.
#ifndef MK_DFLASH_TREE
#define MK_DFLASH_TREE 0
#endif
namespace mk::attn_cfg {
constexpr uint32_t kM = cfg::kWindowM;  // the attention's rows
constexpr uint32_t kFeedRows = 16;      // the feeds' 16 rows: kv_len = P + 16
static_assert(kM == kFeedRows || kM == 8, "the attention's rows: 16, or 8 for an 8-row window");
constexpr bool kTree = MK_DFLASH_TREE != 0;
static_assert(!kTree || kM == kFeedRows, "a tree window has the 16 feed rows");
// the tree table after kv_len: attn_kv_len[kTreeDepthWord + r] = row r's depth, [kTreeMaskWord + r] its ancestor mask
// (bit j: window row j is the row or an ancestor); dflash/tree.cuh kDepth / kMask, one word further
constexpr uint32_t kTreeDepthWord = 1;
constexpr uint32_t kTreeMaskWord = 17;
constexpr uint32_t kNumQHeads = cfg::kQHeads;    // query heads per rank (head_dim 64)
constexpr uint32_t kNumKvHeads = cfg::kKvHeads;  // KV heads per rank
// the global attention's carve bytes: its 256 B header (1 KB aligned), four K / V buffer pairs of 128 x 128 B, the
// merge words' 1 KB and the 16 KB Q staging (the mma.sync attention loop's ring lives in the post-attention ring's
// slots)
constexpr uint32_t kAttnSmemBytes = 149760;
constexpr bool kExpFtz = kImage.global_attn;  // softmax exp2 as the bare ex2.approx.ftz (softmax_exp2)
constexpr uint32_t kAttnWindow =
    kImage.global_attn ? 1073741824 : 2047;  // 2047 rows on swa; past any context on global
// arena byte offsets of the attention's activations: x_attn [16][q heads x 64] bf16, qkv_out [16][qkv row] bf16 (the
// row [q | k | v | o-gate]: 2 (q + kv heads) x 64), then the split partials (attn_ws)
constexpr uint32_t kOff_x_attn = 0;
constexpr uint32_t kOff_qkv_out = kFeedRows * kNumQHeads * 64 * 2;
constexpr uint32_t kOff_attn_ws = kOff_qkv_out + kFeedRows * 2 * (kNumQHeads + kNumKvHeads) * 64 * 2;
}  // namespace mk::attn_cfg

// The attention head phase's arguments and its counter words.

namespace mk {

// The head phase's arguments. The activation tensors resolve through the arena at the cfg header's offsets
// (attn_cfg::kOff_*); the counters are monotonic, their targets scaled by the per-SM epoch.
struct AttnArgs {
    uint8_t* arena;
    uint32_t* fctr;      // the attention counters (the map below)
    uint64_t* sm_epoch;  // [kNumSMs]
    uint64_t* prof_buf;  // unused; kept for the schedule (README.md)
    // bf16 KV pool: byte bases over [slots, kv_heads*64] rows
    uint8_t* k_cache;
    uint8_t* v_cache;
    // [nseq, attn_page_stride] position-ordered byte offsets into the pool and the per-sequence lengths, read on
    // the device every launch: only the pointers are baked into a captured graph, so the host rewrites the
    // contents between replays
    const int64_t* attn_page_table;
    const int32_t* attn_kv_len;
    uint32_t attn_page_stride;
    // unused (the window is attn_cfg::kAttnWindow); kept for the schedule (README.md)
    uint32_t attn_window;
};

// The counter map (fctr indices). Each ready counter owns a full 128 B line: the counters are acquire-polled by
// hundreds of warps and published concurrently, and co-resident counters on one line starve late pollers. Words 0..607
// are unused: the map's positions are fixed in the images' immediates.
constexpr uint32_t kCtrAttnOut = 608;
// the per-split ready words: the kCtrAttnPartsWords words above kCtrAttnParts shared over the units; split (u, j)
// owns word kCtrAttnParts + u x stride + j, stride = kCtrAttnPartsWords / units (64 words = 2 lines at four units, so a
// unit's combine poller never shares a line with another unit's writers)
constexpr uint32_t kCtrAttnParts = 672;
constexpr uint32_t kCtrAttnPartsWords = 256;

}  // namespace mk

// The attention's shared pieces: the MMA / TMEM / exp2 helpers, the barrier map, the unit decomposition and the
// per-rank head geometry.

// the sliding-window attention's ring depth in 16-row blocks
namespace mk::attn_cfg {
constexpr uint32_t kSwaDepth = 2;
}  // namespace mk::attn_cfg

namespace mk {
namespace attn {

// the qkv row of a unit row r (q head x kM + token) of sequence seq
static __device__ __forceinline__ uint32_t sk_row(const uint32_t seq, const uint32_t r) {
    return seq * attn_cfg::kM + r % attn_cfg::kM;
}

// exp2 for the softmax: with kExpFtz the bare ex2.approx.ftz (one MUFU.EX2); exp2f() adds a denormal fixup (a compare
// and two predicated multiplies per element) on the latency-bound fma -> ex2 chain. Values differ only for
// arguments below -126 (flushed to 0).
static __device__ __forceinline__ float softmax_exp2(const float& x) {
    if constexpr (attn_cfg::kExpFtz) {
        return ptx::ex2_approx_ftz_f32(x);
    } else {
        return exp2f(x);
    }
}

// the attention's MMAs, elected inside so every call site stays warp-wide
static __device__ __forceinline__ void attn_mma_ss(const uint64_t desc_a, const uint64_t desc_b, const uint32_t tmem_c,
                                                   const uint32_t scale_c, const uint32_t idesc) {
    if (::ptx::elect_one_sync()) ::ptx::tcgen05_mma_f16(tmem_c, desc_a, desc_b, idesc, scale_c);
}

static __device__ __forceinline__ void attn_mma_ts(const uint32_t tmem_a, const uint64_t desc_b, const uint32_t tmem_c,
                                                   const uint32_t scale_c, const uint32_t idesc) {
    if (::ptx::elect_one_sync()) ::ptx::tcgen05_mma_f16_atmem(tmem_c, tmem_a, desc_b, idesc, scale_c);
}

// an elected tcgen05 commit (the callers are warp-wide)
static __device__ __forceinline__ void umma_arrive(uint64_t* bar) {
    if (::ptx::elect_one_sync()) ::ptx::tcgen05_commit_arrive(bar);
}

static __device__ __forceinline__ constexpr uint32_t min_u32(const uint32_t& a, const uint32_t& b) {
    return a < b ? a : b;
}

// the global attention's 32-column TMEM row ops (lane = the executing thread's TMEM lane; col = a column of the CTA's
// whole-pool allocation, base 0)
static __device__ __forceinline__ void tmem_ld32(const uint32_t& col, uint32_t* r) {
    ::ptx::tcgen05_ld_32x32b_x32(col, r);
}

static __device__ __forceinline__ void tmem_st32(const uint32_t& col, const uint32_t* r) {
    ::ptx::tcgen05_st_32x32b_x32(col, r);
}

// the global attention's geometry: 128-row KV blocks (Q in smem), four K / V staging buffers, two feed warps (one
// parity stream each)
inline constexpr uint32_t kGlobalTcN = 128, kGlobalKvDepth = 4, kGlobalFeed = 2;

// the attention's barrier count: the global attention's 17 + 4 per KV buffer pair past the first two, else 10
constexpr uint32_t kNumAttnBars = kImage.global_attn ? 17 + 4 * (kGlobalKvDepth - 2) : 10;

// The attention's barriers: [0, 1] S ready per stream | [2, 3] K full | [4, 5] V full | [6, 7] K empty (armed by
// the QK commit) | [8, 9] V empty (armed by the PV commit); the global attention adds [10, 11] PV done per stream |
// [12, 13] S free | [14, 15] P ready | [16] Q ready, and [17, ...) the K / V full / empty barriers of buffers
// 2..kGlobalKvDepth-1.
static __device__ __forceinline__ void attn_init_bars(uint64_t* bars) {
#pragma unroll
    for (uint32_t i = 0; i < kNumAttnBars; ++i) {
        // the global attention's warpgroup barriers (12-16) take 128 arrivals
        uint32_t cnt = kImage.global_attn && i >= 12 && i < 17 ? 128u : 1u;
        // the S-free / P-ready arrives are elected one per warp (4 per warpgroup); Q ready keeps 128
        if (kImage.global_attn && i >= 12 && i < 16) cnt = 4u;
        ::ptx::mbar_init(&bars[i], cnt);
    }
}

// the unit decomposition, one source for the split loop, the combine set and the combine wait: every launch walks
// all (m / kM) x kNumKvHeads units (m rows in sequences of kM)
constexpr uint32_t attn_units(const uint32_t& m) {
    return (m / attn_cfg::kM) * attn_cfg::kNumKvHeads;
}
// a unit's rows: the kv head's 8 query heads x kM tokens (GQA 8:1)
inline constexpr uint32_t kUnitRows = attn_cfg::kNumQHeads / attn_cfg::kNumKvHeads * attn_cfg::kM;

// the split count per unit: one split per CTA of the grid
constexpr uint32_t attn_splits(const uint32_t& num_sms, const uint32_t& m) {
    return num_sms / attn_units(m);
}

// the box predicate: whether a global attention's chunk can hold one full 128-row TMA box of KV rows (the window bounds
// the chunk)
constexpr bool attn_box_live() {
    constexpr uint32_t kSplits = attn_splits(cfg::kGrid, attn_cfg::kM);
    return (attn_cfg::kAttnWindow + 16 + kSplits - 1) / kSplits >= kGlobalTcN;
}

// combine row groups per unit (the world's): 32 = kUnitRows / 32 rows per combine CTA (4, or 2 at kM 8), one warp per
// row; 16 = 8 rows (4) per combine CTA
inline constexpr uint32_t kAttnGroups = cfg::kWorld.combine_groups;
// the combine CTA count = the kCtrAttnOut arrivals per launch; every spin site derives its target here
constexpr uint32_t attn_combines(const uint32_t& num_sms, const uint32_t& m) {
    return attn_units(m) * kAttnGroups;
}

// the per-rank head geometry (head_dim 64, GQA 8:1: the attention unit is one kv head's 8-query group): the qkv
// row [q | k | v | og], the o-gate column, the x_attn row, the bf16 KV pool row, and the per-split ready-word
// stride (AttnArgs: the 256 words above kCtrAttnParts shared over the units)
static_assert(attn_cfg::kNumQHeads == 8 * attn_cfg::kNumKvHeads, "GQA 8:1");
constexpr uint32_t kQkvRow = 2 * (attn_cfg::kNumQHeads + attn_cfg::kNumKvHeads) * 64;
constexpr uint32_t kOgCol = (attn_cfg::kNumQHeads + 2 * attn_cfg::kNumKvHeads) * 64;
constexpr uint32_t kXAttnRow = attn_cfg::kNumQHeads * 64;
constexpr uint32_t kKvRowBytes = attn_cfg::kNumKvHeads * 128;
constexpr uint32_t kAttnPartStride = kCtrAttnPartsWords / attn_units(attn_cfg::kM);
// a split's attn_ws partial: O [kUnitRows, 64] + (m, l) [kUnitRows] fp32
constexpr uint32_t kWsRow = kUnitRows * 66;
// the merge-neutral sentinel (not -inf: exp2f(s - s) must be finite for the branchless combine; |logits| << 1e30, so
// max() and exp2f are unchanged)
constexpr float kMSent = -1e30f;
// the epilogue warps' named barrier
constexpr uint32_t kAttnBarId = 2;

// A tree window's key test (the linear window's: pos <= posq): key position pos of [.., end) within the window of the
// row at posq; the window's own keys (pos >= base, its first slot) by the row's ancestor mask, every earlier key is its
// past.
static __device__ __forceinline__ bool tree_key_visible(const uint32_t pos, const uint32_t end, const uint32_t posq,
                                                        const uint32_t win, const uint32_t base, const uint32_t tmask) {
    return pos < end && (pos < base || ((tmask >> (pos - base)) & 1u)) && pos + win >= posq;
}

}  // namespace attn

// The global attention: the attention phase of the global layers (tcgen05; S, P and O in TMEM).

namespace attn {

// One wide S drain to registers (the fused ld.red row max on interior blocks), register-resident exp / pack, one
// wide P store to TMEM; O accumulates in TMEM across blocks (PV scale_c = 1) with a vote-gated rescale when the row
// max moved, and drains once per split. Warps: the two below the issuer feed K / V by TMA (one parity stream each),
// the warp below the softmax band issues every QK and PV, and the eight epilogue warps are two softmax warpgroups on
// alternate 128-row blocks (both cover TMEM lanes 0-127). S is freed at the drain, so QK(b + 2) issues during
// softmax(b). The two streams' partials merge through TMEM and publish one (m, l, O) row. kEpiBase = the first
// softmax thread; every role id derives from it.
template <uint32_t kEpiBase>
static __device__ __forceinline__ void attn_global(const AttnArgs& args, uint8_t* smem_buffer, uint8_t* region,
                                                   const uint32_t& sm_idx, const uint32_t& warp_idx,
                                                   const uint32_t& lane_idx, const CUtensorMap* tmap_k,
                                                   const CUtensorMap* tmap_v, const CUtensorMap* tmap_kb,
                                                   const CUtensorMap* tmap_vb) {
    namespace fcfg = attn_cfg;
    constexpr uint32_t kM = fcfg::kM;
    constexpr uint32_t kSplits = attn_splits(cfg::kGrid, kM);
    // the window (2047 on swa, a sentinel past any context on global)
    constexpr uint32_t kWindow = fcfg::kAttnWindow;
    constexpr uint32_t win = kWindow;
    // 128-row KV blocks: one S-ready round trip, vote, (m, l) update and P window per 128 rows; Q sits in smem
    // behind a K-major A descriptor so the TMEM plan fits 512 columns
    constexpr uint32_t kTcN = kGlobalTcN;
    // the KV staging depth in buffers: buffer = block mod kDep; buffers 0 / 1 use barriers [2..9], deeper ones
    // are appended past 17
    constexpr uint32_t kDep = kGlobalKvDepth;
    // the TMEM plan (512 columns): P0 | P1 | S0 | S1 | O0 | O1 (Q in smem). Each half table is a two-word
    // select, not an array: runtime-indexed constexpr arrays land in a local frame read on every block.
    struct Half2 {
        uint32_t v0, v1;
        __host__ __device__ constexpr uint32_t operator[](uint32_t i) const { return i ? v1 : v0; }
    };
    constexpr Half2 kTP{0u, 64u};
    constexpr Half2 kTS{128u, 256u};
    constexpr Half2 kTO{384u, 448u};
    static_assert((kTO[1] + 64) <= 512, "B TMEM plan");
    constexpr uint32_t kHdr = 256;
    static_assert(fcfg::kAttnSmemBytes >= kHdr + 1024 + 2 * kDep * kTcN * 128 + 1024 + 16384, "global attention carve");

    const uint32_t u = sm_idx / kSplits, j = sm_idx % kSplits;
    const uint32_t seq = u / fcfg::kNumKvHeads, kvh = u % fcfg::kNumKvHeads;
    // the chunks cut the 16-row window (flen = kv_len = P + 16); the attention attends up to its own end, len
    const uint32_t flen = static_cast<uint32_t>(__ldg(args.attn_kv_len + seq));
    const uint32_t len = flen - (fcfg::kFeedRows - kM);
    const uint32_t lo = flen > 16 + win ? flen - 16 - win : 0u;
    const uint32_t chunk0 = (flen - lo + kSplits - 1) / kSplits;
    const uint32_t ca = (chunk0 + kTcN - 1) / kTcN * kTcN;
    const uint32_t chunk = (chunk0 >= kTcN) ? ca : chunk0;
    const uint32_t begin = lo + j * chunk;
    const uint32_t end = min_u32(begin + chunk, len);
    const int64_t* ptab = args.attn_page_table + static_cast<uint64_t>(seq) * args.attn_page_stride;
    const uint32_t rounds = begin < end ? end - begin : 0u;
    const uint32_t nblk = (rounds + kTcN - 1) / kTcN;
    // the buffer -> barrier maps and block -> buffer / parity
    const auto bfk = [](const uint32_t& i) -> uint32_t { return i < 2 ? 2u + i : 17u + (i - 2); };
    const auto bfv = [](const uint32_t& i) -> uint32_t { return i < 2 ? 4u + i : 17u + (kDep - 2) + (i - 2); };
    const auto bek = [](const uint32_t& i) -> uint32_t { return i < 2 ? 6u + i : 17u + 2 * (kDep - 2) + (i - 2); };
    const auto bev = [](const uint32_t& i) -> uint32_t { return i < 2 ? 8u + i : 17u + 3 * (kDep - 2) + (i - 2); };
    const auto bidx = [](const uint32_t& b) -> uint32_t { return b % kDep; };
    const auto bpar = [](const uint32_t& b) -> uint32_t { return (b / kDep) & 1; };
    const auto qkv_out = reinterpret_cast<const nv_bfloat16*>(args.arena + fcfg::kOff_qkv_out);
    const auto attn_ws = reinterpret_cast<float*>(args.arena + fcfg::kOff_attn_ws);

    auto bars = reinterpret_cast<uint64_t*>(region + 16);
    // bounded polls: try_wait's hardware suspend makes a poll microsecond-class, so 2^24 polls trap a deadlock
    // in seconds (legitimate waits are < 3 us)
    constexpr uint64_t kBarPollCap = 16777216;
    const auto bar_wait = [&](const uint32_t& i, const uint32_t& ph) {
        uint64_t polls = 0;
        while (!::ptx::mbar_try_wait_parity(&bars[i], ph))
            ::ptx::device_assert((++polls) < kBarPollCap && "attention mbarrier wait");
    };
    // The tail box: the block that straddles `len` boxes too. Its base comes from the block's first in-window
    // row (the swa table may hold -1 below lo; rows past len have no entry), every in-window row votes on
    // page-consecutiveness, and the issuer zeroes V's rows past `end` once the box has landed, ahead of the PV
    // (P = 0 x NaN poisons PV); K's rows past `end` reach masked logits only. Without it that block, the one
    // holding the newest rows, takes the per-row scatter.
    uint8_t* const kv0 =
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(region) + kHdr + 1023) & ~static_cast<uintptr_t>(1023));
    // in-carve KV staging: K0..K(d-1) | V0..V(d-1), one kTcN x 128 B slot each, then the merge's (m, l) words
    const auto kbuf = [&](const uint32_t& i) { return kv0 + i * (kTcN * 128); };
    const auto vbuf = [&](const uint32_t& i) { return kv0 + (kDep + i) * (kTcN * 128); };
    uint8_t* const mlbuf = kv0 + 2 * kDep * (kTcN * 128);
    // 16 KB of Q staging (128 rows x 128 B in canonical SWIZZLE_128B atoms, the K boxes' landing, so the A
    // descriptor reads it like kdesc reads K)
    uint8_t* const qsm = mlbuf + 1024;
    ::ptx::device_assert(*reinterpret_cast<const volatile uint32_t*>(region) == 0);

    // the instruction descriptors
    constexpr uint32_t rt_qk = ::ptx::mma_inst_desc_f16(kUnitRows, kTcN);
    constexpr uint32_t rt_pv = ::ptx::mma_inst_desc_f16(kUnitRows, 64, ::ptx::F16Type::BF16, ::ptx::F16Type::BF16,
                                                        ::ptx::DType::F32, ::ptx::Major::K, ::ptx::Major::MN);
    // the descriptor bases, built once per issue from the slot address
    const auto kdesc = [&](const uint32_t& i) -> uint64_t {
        return ::ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(::ptx::to_shared(kbuf(i)));
    };
    const auto vdesc = [&](const uint32_t& i) -> uint64_t {
        return ::ptx::mma_smem_desc_mn_major<uint16_t, kTcN, 64, 128>(::ptx::to_shared(vbuf(i)));
    };
    const auto qk_issue = [&](const uint32_t& idx, const uint32_t& scol) {
        const uint64_t kd0 = kdesc(idx);
        // Q rides an smem A descriptor (SS form); each 16-dim k atom steps both descriptors by 32 B
        const uint64_t ad0 = ::ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(::ptx::to_shared(qsm));
#pragma unroll
        for (uint32_t kk = 0; kk < 4; ++kk) {
            const uint64_t ad = ad0 + kk * 2;
            const uint64_t kd = kd0 + kk * 2;
            attn_mma_ss(ad, kd, scol, kk > 0 ? 1u : 0u, rt_qk);
        }
    };
    const auto pv_issue = [&](const uint32_t& idx, const uint32_t& pcol, const uint32_t& ocol, const bool& acc) {
        const uint64_t vd0 = vdesc(idx);
#pragma unroll
        for (uint32_t kk = 0; kk < kTcN / 16; ++kk) {
            const uint64_t vd = vd0 + kk * 128;  // one 16-row k atom = 2 KB
            attn_mma_ts(pcol + kk * 8, vd, ocol, (acc || kk > 0) ? 1u : 0u, rt_pv);
        }
    };
    // the tail box's V rows past `end`, zeroed by the issuer once the box has
    // landed: the barrier's completed transaction orders the box's writes ahead of these stores, the async-proxy
    // fence orders the stores ahead of the PV
    const auto tail_zero = [&](const uint32_t& idx, const uint32_t& b) {
        const uint32_t nr = min_u32(end - (begin + kTcN * b), kTcN);
        if (nr < kTcN) {
            auto vz = reinterpret_cast<uint4*>(vbuf(idx) + nr * 128);
            for (uint32_t z = lane_idx; z < (kTcN - nr) * 8; z += 32)
                vz[z] = make_uint4(0u, 0u, 0u, 0u);
            ptx::fence_async_smem();
            __syncwarp();
        }
    };

    const uint32_t epi_warp = warp_idx - kEpiBase / 32;  // softmax WGs only
    constexpr float kL2eScale = 0.125f * 1.4426950408889634f;
    const float kNegInf = -__int_as_float(0x7f800000);

    // ---- the K / V feed: kFeed warps below the issuer, one parity stream each (buffer == stream, so each warp
    // has its own K / V barriers); lanes own rows ----
    constexpr uint32_t kFeed = kGlobalFeed;
    constexpr uint32_t kIssuerWarp = kEpiBase / 32 - 1;
    // the KV feed is read once: evict-first
    constexpr uint64_t kKvHint = ::ptx::kEvictFirst;
    const bool is_feed = (warp_idx + kFeed >= kIssuerWarp && warp_idx < kIssuerWarp);
    if (is_feed) {
        // this warp's lane: which parity stream (== buffer idx) it feeds
        // (feed 1/2 warps carry both species)
        const uint32_t fw = kIssuerWarp - 1u - warp_idx;  // 0..kFeed-1
        const uint32_t my_s = fw;
        if (rounds > 0) {
            if (::ptx::elect_one_sync()) {
                ::ptx::prefetch_tensormap(tmap_k);
                ::ptx::prefetch_tensormap(tmap_v);
                ::ptx::prefetch_tensormap(tmap_kb);
                ::ptx::prefetch_tensormap(tmap_vb);
            }
            // this warp's stream's blocks: b = my_s, my_s + 2, ...
            const uint32_t b0 = (my_s);
            const uint32_t bstep = 2u;
            for (uint32_t b = b0; b < nblk; b += bstep) {
                // buffer = b mod kDep, parity = (b / kDep) & 1
                const uint32_t idx = bidx(b), ph = bpar(b);
                const uint32_t r0 = begin + kTcN * b;
                const uint32_t nr = min_u32(end - r0, kTcN);
                uint64_t off2[kTcN / 32];
#pragma unroll
                for (uint32_t i = 0; i < kTcN / 32; ++i)
                    off2[i] = 0;  // the rows past `end` hold no entry; the vote below skips them
#pragma unroll
                for (uint32_t i = 0; i < kTcN / 32; ++i)
                    if (lane_idx + 32 * i < nr) off2[i] = static_cast<uint64_t>(__ldg(ptab + r0 + lane_idx + 32 * i));
                // Box eligibility: a full block over consecutive pool slots (every row's entry
                // checked; page-grain allocation makes them consecutive). One box per species then
                // lands the same SWIZZLE_128B bytes the per-row issues would; other blocks take the
                // per-row scatter.
                uint64_t obase = 0;
                bool boxed = false;
                // the base from the first in-window row (lo <= row < end: its entry is live): every
                // slot's entry shuffled, the slot selected after (a guarded shuffle per slot
                // if-converts into a dynamically indexed read of off2, a local-memory frame)
                const uint32_t d0 = lo > r0 ? lo - r0 : 0u;
#pragma unroll
                for (uint32_t i = 0; i < kTcN / 32; ++i) {
                    const uint64_t v = __shfl_sync(0xffffffffu, off2[i], d0 & 31u);
                    obase = (d0 >> 5) == i ? v - d0 * kKvRowBytes : obase;
                }
                bool ok = true;
#pragma unroll
                for (uint32_t i = 0; i < kTcN / 32; ++i) {
                    const uint32_t rr = lane_idx + 32 * i;
                    if (rr >= d0 && rr < nr) ok &= off2[i] == obase + rr * kKvRowBytes;
                }
                boxed = __all_sync(0xffffffffu, ok);
                // K first, each species on its own empty barrier (a joint wait deadlocks through K
                // full / V empty)
                bar_wait(bek(idx), ph ^ 1);  // K buffer free
                if (boxed) {
                    if (::ptx::elect_one_sync()) {
                        ::ptx::mbar_arrive_expect_tx(&bars[bfk(idx)], (kTcN) * 128);
                        ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(kbuf(idx)), tmap_kb, kvh * 64,
                                                                 static_cast<int32_t>(obase / kKvRowBytes),
                                                                 &bars[bfk(idx)], kKvHint);
                    }
                    __syncwarp();
                } else {
                    if (::ptx::elect_one_sync()) ::ptx::mbar_arrive_expect_tx(&bars[bfk(idx)], nr * 128);
                    __syncwarp();
#pragma unroll
                    for (uint32_t i = 0; i < kTcN / 32; ++i) {
                        const uint32_t rr = lane_idx + 32 * i;
                        if (rr < nr)
                            ::ptx::cp_async_bulk_tensor_2d_load_hint(
                                ::ptx::to_shared(kbuf(idx) + rr * 128), tmap_k, kvh * 64,
                                static_cast<int32_t>(off2[i] / kKvRowBytes), &bars[bfk(idx)], kKvHint);
                    }
                }
                bar_wait(bev(idx), ph ^ 1);  // V buffer free
                if (boxed) {
                    if (::ptx::elect_one_sync()) {
                        ::ptx::mbar_arrive_expect_tx(&bars[bfv(idx)], (kTcN) * 128);
                        ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(vbuf(idx)), tmap_vb, kvh * 64,
                                                                 static_cast<int32_t>(obase / kKvRowBytes),
                                                                 &bars[bfv(idx)], kKvHint);
                    }
                    __syncwarp();
                    continue;
                }
                if (nr < kTcN) {
                    // V tail zero-fill (P = 0 x NaN poisons PV)
                    auto vz = reinterpret_cast<uint4*>(vbuf(idx) + nr * 128);
                    for (uint32_t z = lane_idx; z < (kTcN - nr) * 8; z += 32)
                        vz[z] = make_uint4(0u, 0u, 0u, 0u);
                    ptx::fence_async();
                }
                if (::ptx::elect_one_sync()) ::ptx::mbar_arrive_expect_tx(&bars[bfv(idx)], nr * 128u);
                __syncwarp();
#pragma unroll
                for (uint32_t i = 0; i < kTcN / 32; ++i) {
                    const uint32_t rr = lane_idx + 32 * i;
                    if (rr < nr)
                        ::ptx::cp_async_bulk_tensor_2d_load_hint(::ptx::to_shared(vbuf(idx) + rr * 128), tmap_v,
                                                                 kvh * 64, static_cast<int32_t>(off2[i] / kKvRowBytes),
                                                                 &bars[bfv(idx)], kKvHint);
                }
            }
        }
        {
            // retire the final empty arrivals: the attention invalidates the barriers and hands the carve back to the
            // post-attention ring
            for (uint32_t i = 0; i < kDep && i < nblk; ++i) {
                const uint32_t last = i + ((nblk - 1 - i) / kDep) * kDep;
                if ((last & 1) == my_s) {
                    bar_wait(bek(i), bpar(last));
                    bar_wait(bev(i), bpar(last));
                }
            }
        }
        return;  // no epilogue duties
    }

    // ---- the MMA issuer: the warp below the softmax band, every QK and PV, gated per stream ----
    if (warp_idx == kIssuerWarp) {
        if (rounds > 0) {
            bar_wait(16, 0u);  // Q staged
#pragma unroll 1
            for (uint32_t b = 0; b < min_u32(nblk, 2u); ++b) {
                bar_wait(bfk(bidx(b)), bpar(b));
                ::ptx::tcgen05_fence_after_thread_sync();
                qk_issue(bidx(b), kTS[b]);
                umma_arrive(&bars[b]);
                umma_arrive(&bars[bek(bidx(b))]);
            }
#pragma unroll 1
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint32_t s = b & 1, t = b >> 1;
                const uint32_t tg = t;
                if (b + 2 < nblk) {
                    // QK(b + 2) at the drain-site S free: it issues during softmax(b), before
                    // PV(b)
                    const uint32_t bq = bidx(b + 2), pq = bpar(b + 2);
                    bar_wait(12 + s, tg & 1);  // drain(s,t) freed S
                    bar_wait(bfk(bq), pq);     // K_{b+2} landed
                    ::ptx::tcgen05_fence_after_thread_sync();
                    qk_issue(bq, kTS[s]);
                    umma_arrive(&bars[s]);
                    umma_arrive(&bars[bek(bq)]);
                }
                bar_wait(14 + s, tg & 1);  // P(s,t) ready (O rescaled in-WG before it)
                const uint32_t bp = bidx(b), pp = bpar(b);
                bar_wait(bfv(bp), pp);  // V_b landed
                tail_zero(bp, b);
                ::ptx::tcgen05_fence_after_thread_sync();
                pv_issue(bp, kTP[s], kTO[s], t > 0);
                umma_arrive(&bars[10 + s]);
                umma_arrive(&bars[bev(bp)]);
            }
        }
        return;  // no epilogue duties
    }

    const auto mask_free = [&](const uint32_t& kvp0) {
        // every row unmasked: no tail, causal-safe for posq_min = len - kM, window-safe for posq_max = len - 1
        // (uint-safe rearrangements)
        return kvp0 + kTcN <= end && kvp0 + kTcN + kM - 1 <= len && kvp0 + win + 1 >= len;
    };
    if constexpr (kM == 16) {
        // ---- the softmax warpgroups: stream = warpgroup, row = TMEM lane ----
        // Exp + sum + pack run in registers ahead of the PV-done wait; only the P store and the O
        // rescale wait behind it (the rescale after the pack, so the P stores overlap the O round trip).
        const uint32_t wg = (epi_warp >> 2);  // stream = warpgroup
        const bool wg0 = epi_warp < 4;
        const uint32_t r = (epi_warp & 3) * 32 + lane_idx;  // q row = TMEM lane
        const uint32_t posq =
            fcfg::kTree ? len - kM + __ldg(args.attn_kv_len + fcfg::kTreeDepthWord + r % kM) : len - kM + r % kM;
        // a tree window's row: the window's first slot and the row's ancestor mask (read only by tree images)
        [[maybe_unused]] const uint32_t base = len - kM;
        [[maybe_unused]] const uint32_t tmask =
            fcfg::kTree ? __ldg(args.attn_kv_len + fcfg::kTreeMaskWord + r % kM) : 0u;
        float o[64];
#pragma unroll
        for (uint32_t i = 0; i < 64; ++i)
            o[i] = 0.0f;
        float ml_m = kMSent, ml_l = 0.0f;

        // a 64-row drain, never called; kept for the schedule (README.md); the linear window's mask only
        const auto drain_max = [&](const uint32_t& scol, const uint32_t& kvp0, const bool& mfree, uint32_t* lv) {
            float mnew = ml_m;
            if (mfree) {
                float r0, r1;
                ptx::tcgen05_ld_red_max_32x32b_x32(scol, lv, r0);
                ptx::tcgen05_ld_red_max_32x32b_x32(scol + 32, lv + 32, r1);
                ::ptx::tcgen05_wait_ld();
                mnew = fmaxf(mnew, fmaxf(r0, r1) * kL2eScale);
            } else {
                tmem_ld32(scol, lv);
                tmem_ld32(scol + 32, lv + 32);
                ::ptx::tcgen05_wait_ld();
#pragma unroll
                for (uint32_t c = 0; c < 64; ++c) {
                    const uint32_t pos = kvp0 + c;
                    const bool okp = pos < end && pos <= posq && pos + win >= posq;
                    const float f = okp ? reinterpret_cast<const float&>(lv[c]) * kL2eScale : kNegInf;
                    lv[c] = reinterpret_cast<const uint32_t&>(f);
                    mnew = fmaxf(mnew, f);
                }
            }
            return mnew;
        };
        const auto rescale = [&](const uint32_t& ocol, const float& al) {
            // vote-gated O rescale (al == 1 rows are exact no-ops; the warp skips only when all its rows agree),
            // in 32-column halves for the register budget
            if (!__all_sync(0xffffffffu, al == 1.0f)) {
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t ov[32];
                    tmem_ld32(ocol + 32 * h, ov);
                    ::ptx::tcgen05_wait_ld();
#pragma unroll
                    for (uint32_t c = 0; c < 32; ++c) {
                        const float f = reinterpret_cast<const float&>(ov[c]) * al;
                        ov[c] = reinterpret_cast<const uint32_t&>(f);
                    }
                    tmem_st32(ocol + 32 * h, ov);
                }
            }
        };
        const auto exp_pack = [&](uint32_t* lv, const bool& mfree, const float& mnew) {
            // fl(lv x scale) - mnew keeps the op sequence of the reference (m, l bitwise); masked lanes are
            // -inf: ftz ex2 gives an exact 0
            float lsum = 0.0f;
#pragma unroll
            for (uint32_t cp = 0; cp < 32; ++cp) {
                float p2[2];
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    float lf = reinterpret_cast<const float&>(lv[2 * cp + h]);
                    if (mfree) lf *= kL2eScale;
                    p2[h] = softmax_exp2(lf - mnew);
                }
                lsum += p2[0] + p2[1];
                const nv_bfloat162 pb = __float22bfloat162_rn({p2[0], p2[1]});
                lv[cp] = reinterpret_cast<const uint32_t&>(pb);
            }
            return lsum;
        };

        if (rounds > 0) {
            if (wg0) {  // warpgroup 0 stages Q
                // Q -> smem once per split, warpgroup 0's rows
                const uint32_t tok = sk_row(seq, r), qh = r >> 4;
                const auto qrow = reinterpret_cast<const uint4*>(qkv_out + static_cast<uint64_t>(tok) * kQkvRow +
                                                                 (kvh * 8 + qh) * 64);
                uint4 qv[8];
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i)
                    qv[i] = __ldg(qrow + i);
                // canonical SWIZZLE_128B placement (16 B chunk i of row r lands at i ^ (r & 7)); the async-proxy fence
                // makes it visible to the QK descriptor
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i)
                    *reinterpret_cast<uint4*>(qsm + r * 128 + ((i ^ (r & 7)) << 4)) = qv[i];
                ptx::fence_async();
                ::ptx::mbar_arrive(&bars[16]);  // Q visible to the issuer
            }

            // local blocks b = wg + 2t
            const uint32_t wl = wg, n_s = nblk > wl ? (nblk - wl + 1) >> 1 : 0u;
            for (uint32_t t = 0; t < n_s; ++t) {
                const uint32_t b = wl + 2 * t, tg = t;
                const uint32_t kvp0 = begin + kTcN * b;
                bar_wait(wg, tg & 1);  // S(s) ready
                ::ptx::tcgen05_fence_after_thread_sync();
                uint32_t lv[64];
                const bool mfree = mask_free(kvp0);
                // The half drain: the full 128-column row max first (half A stays in lv, half B keeps only
                // its max), then one (m, l) update, vote and P window per 128 rows. Half B re-drains after
                // the PV-done wait: S is held to its last read, so the S-free arrive moves there.
                float mnew = ml_m;
                if (mfree) {
                    float r0, r1, r2, r3;
                    ptx::tcgen05_ld_red_max_32x32b_x32(kTS[wg] + 64, lv, r2);
                    ptx::tcgen05_ld_red_max_32x32b_x32(kTS[wg] + 96, lv + 32, r3);
                    ptx::tcgen05_ld_red_max_32x32b_x32(kTS[wg], lv, r0);
                    ptx::tcgen05_ld_red_max_32x32b_x32(kTS[wg] + 32, lv + 32, r1);
                    ::ptx::tcgen05_wait_ld();
                    mnew = fmaxf(mnew, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)) * kL2eScale);
                } else {
                    tmem_ld32(kTS[wg] + 64, lv);
                    tmem_ld32(kTS[wg] + 96, lv + 32);
                    ::ptx::tcgen05_wait_ld();
#pragma unroll
                    for (uint32_t c = 0; c < 64; ++c) {
                        const uint32_t pos = kvp0 + 64 + c;
                        if constexpr (fcfg::kTree) {
                            if (tree_key_visible(pos, end, posq, win, base, tmask))
                                mnew = fmaxf(mnew, reinterpret_cast<const float&>(lv[c]) * kL2eScale);
                        } else {
                            if (pos < end && pos <= posq && pos + win >= posq)
                                mnew = fmaxf(mnew, reinterpret_cast<const float&>(lv[c]) * kL2eScale);
                        }
                    }
                    tmem_ld32(kTS[wg], lv);
                    tmem_ld32(kTS[wg] + 32, lv + 32);
                    ::ptx::tcgen05_wait_ld();
#pragma unroll
                    for (uint32_t c = 0; c < 64; ++c) {
                        const uint32_t pos = kvp0 + c;
                        bool okp;
                        if constexpr (fcfg::kTree)
                            okp = tree_key_visible(pos, end, posq, win, base, tmask);
                        else
                            okp = pos < end && pos <= posq && pos + win >= posq;
                        const float f = okp ? reinterpret_cast<const float&>(lv[c]) * kL2eScale : kNegInf;
                        lv[c] = reinterpret_cast<const uint32_t&>(f);
                        mnew = fmaxf(mnew, f);
                    }
                }
                const float al = softmax_exp2(ml_m - mnew);
                const float lsum_a = exp_pack(lv, mfree, mnew);
                // the PV(s, t - 1) wait guards the P store (P is reused across blocks)
                if (t > 0) {
                    bar_wait(10 + wg, (tg + 1) & 1);  // PV(s,t-1) done
                    ::ptx::tcgen05_fence_after_thread_sync();
                }
                tmem_st32(kTP[wg], lv);  // P half A
                // half B re-drained (a bit-identical reload)
                tmem_ld32(kTS[wg] + 64, lv);
                tmem_ld32(kTS[wg] + 96, lv + 32);
                ::ptx::tcgen05_wait_ld();
                __syncwarp();  // S free at the LAST S read
                if (::ptx::elect_one_sync()) ::ptx::mbar_arrive(&bars[12 + wg]);
                if (!mfree) {
#pragma unroll
                    for (uint32_t c = 0; c < 64; ++c) {
                        const uint32_t pos = kvp0 + 64 + c;
                        bool okp;
                        if constexpr (fcfg::kTree)
                            okp = tree_key_visible(pos, end, posq, win, base, tmask);
                        else
                            okp = pos < end && pos <= posq && pos + win >= posq;
                        const float f = okp ? reinterpret_cast<const float&>(lv[c]) * kL2eScale : kNegInf;
                        lv[c] = reinterpret_cast<const uint32_t&>(f);
                    }
                }
                const float lsum_b = exp_pack(lv, mfree, mnew);
                tmem_st32(kTP[wg] + 32, lv);  // P half B
                ml_l = ml_l * al + (lsum_a + lsum_b);
                ml_m = mnew;
                if (t > 0) rescale(kTO[wg], al);
                ::ptx::tcgen05_wait_st();
                __syncwarp();
                if (::ptx::elect_one_sync()) ::ptx::mbar_arrive(&bars[14 + wg]);  // P(s,t) ready
            }
        }

        // The streams merge through TMEM: each warpgroup drains half the columns of both streams' O (its lane
        // quarter reaches every column), the (m, l) pairs cross in the streams' dead S columns, and each warpgroup
        // merges and stores its half of the row: o0 x al + o1 x bl, o1 x bl rounded first.
        uint32_t oa[32], ob[32];
        float mh = kMSent, lh = 0.0f;
        if (rounds > 0) {
            uint32_t n_own = nblk > wg ? (nblk - wg + 1) >> 1 : 0u,
                     n_oth = nblk > (wg ^ 1) ? (nblk - (wg ^ 1) + 1) >> 1 : 0u;
            const auto pv_wait = [&](const uint32_t& s, const uint32_t& n) {  // the stream's last PV-done phase
                bar_wait(10 + s, ((n + 1) & 1));
            };
            uint32_t mx = __float_as_uint(kMSent), lx = 0u;
            // no QK writes a stream's S after its last block
            constexpr Half2 kTX{kTS[0], kTS[1]};
            ::ptx::tcgen05_st_32x32b_x2(kTX[wg], reinterpret_cast<const uint32_t&>(ml_m),
                                        reinterpret_cast<const uint32_t&>(ml_l));
            ::ptx::tcgen05_wait_st();
            if (n_own > 0) pv_wait(wg, n_own);
            ::ptx::tcgen05_fence_before_thread_sync();
            ::ptx::sync_aligned(kAttnBarId, 256);
            // the other stream's last PV is complete by now: its wait cannot alias an earlier phase
            if (n_oth > 0) pv_wait(wg ^ 1, n_oth);
            ::ptx::tcgen05_fence_after_thread_sync();
            ::ptx::tcgen05_ld_32x32b_x2(kTX[wg ^ 1], mx, lx);
            tmem_ld32(kTO[0] + 32 * wg, oa);
            if (nblk > 1) tmem_ld32(kTO[1] + 32 * wg, ob);
            ::ptx::tcgen05_wait_ld();
            if (nblk <= 1) {
#pragma unroll
                for (uint32_t c = 0; c < 32; ++c)
                    ob[c] = 0u;
            }
            const float m0 = wg ? reinterpret_cast<const float&>(mx) : ml_m,
                        l0 = wg ? reinterpret_cast<const float&>(lx) : ml_l;
            const float m1 = wg ? ml_m : reinterpret_cast<const float&>(mx),
                        l1 = wg ? ml_l : reinterpret_cast<const float&>(lx);
            mh = fmaxf(m0, m1);
            const float al = softmax_exp2(m0 - mh), bl = softmax_exp2(m1 - mh);
#pragma unroll
            for (uint32_t c = 0; c < 32; ++c) {
                const float f = __fmaf_rn(reinterpret_cast<const float&>(oa[c]), al,
                                          __fmul_rn(reinterpret_cast<const float&>(ob[c]), bl));
                oa[c] = reinterpret_cast<const uint32_t&>(f);
            }
            lh = __fmaf_rn(l0, al, __fmul_rn(l1, bl));
        } else {
#pragma unroll
            for (uint32_t c = 0; c < 32; ++c)
                oa[c] = 0u;
        }
        float* const ws = attn_ws + static_cast<uint64_t>(sm_idx) * kWsRow;
        float* const wrow = ws + r * 64 + 32 * wg;
#pragma unroll
        for (uint32_t d = 0; d < 4; ++d) {
            const uint4 s0 = make_uint4(oa[8 * d], oa[8 * d + 1], oa[8 * d + 2], oa[8 * d + 3]);
            const uint4 s1 = make_uint4(oa[8 * d + 4], oa[8 * d + 5], oa[8 * d + 6], oa[8 * d + 7]);
            ptx::st_row_u8_cg(reinterpret_cast<uint4*>(wrow + 8 * d), s0, s1);
        }
        if (wg0) {
            const uint2 ml2 = make_uint2(reinterpret_cast<const uint32_t&>(mh), reinterpret_cast<const uint32_t&>(lh));
            ptx::st_row_u2_cg(reinterpret_cast<uint2*>(ws + kUnitRows * 64 + r * 2), ml2);
        }
    } else {
        // ---- the 8-row window: M = 64 tiles, row = TMEM lane 16 x quadrant + i, two threads per row ----
        // The 64 rows sit in the first 16 lanes of each quadrant (the M = 64 layout), so each warp drains its
        // quadrant's 16 lanes with the .16x32bx2 shape: thread half h holds S columns [64 h, + 64) and O columns
        // [32 h, + 32) of its row. A thread's S share fits its registers, so S frees at the drain (QK(b + 2) issues
        // under the exp); the row's max and sum cross the pair by shuffle, in the 16-row path's order (bitwise its
        // rows).
        const uint32_t wg = (epi_warp >> 2);  // stream = warpgroup
        const bool wg0 = epi_warp < 4;
        const uint32_t half = lane_idx >> 4;
        const uint32_t r = (epi_warp & 3) * 16 + (lane_idx & 15);  // q row (q head x 8 + token)
        const uint32_t posq = len - kM + r % kM;
        float ml_m = kMSent, ml_l = 0.0f;

        const auto rescale = [&](const uint32_t& ocol, const float& al) {
            // vote-gated O rescale of the thread's 32 columns (al == 1 rows are exact no-ops)
            if (!__all_sync(0xffffffffu, al == 1.0f)) {
                uint32_t ov[32];
                ::ptx::tcgen05_ld_16x32bx2_x32<32>(ocol, ov);
                ::ptx::tcgen05_wait_ld();
#pragma unroll
                for (uint32_t c = 0; c < 32; ++c) {
                    const float f = reinterpret_cast<const float&>(ov[c]) * al;
                    ov[c] = reinterpret_cast<const uint32_t&>(f);
                }
                ::ptx::tcgen05_st_16x32bx2_x32<32>(ocol, ov);
            }
        };
        const auto exp_pack = [&](uint32_t* lv, const bool& mfree, const float& mnew) {
            // the 16-row path's sequence per column pair (fl(lv x scale) - mnew; masked lanes -inf, an exact 0)
            float lsum = 0.0f;
#pragma unroll
            for (uint32_t cp = 0; cp < 32; ++cp) {
                float p2[2];
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    float lf = reinterpret_cast<const float&>(lv[2 * cp + h]);
                    if (mfree) lf *= kL2eScale;
                    p2[h] = softmax_exp2(lf - mnew);
                }
                lsum += p2[0] + p2[1];
                const nv_bfloat162 pb = __float22bfloat162_rn({p2[0], p2[1]});
                lv[cp] = reinterpret_cast<const uint32_t&>(pb);
            }
            return lsum;
        };

        if (rounds > 0) {
            if (wg0) {  // Q -> smem once per split: each thread four of its row's eight 16 B chunks
                const uint32_t tok = sk_row(seq, r), qh = r / kM;
                const auto qrow = reinterpret_cast<const uint4*>(qkv_out + static_cast<uint64_t>(tok) * kQkvRow +
                                                                 (kvh * 8 + qh) * 64) +
                                  4 * half;
                uint4 qv[4];
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i)
                    qv[i] = __ldg(qrow + i);
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i)
                    *reinterpret_cast<uint4*>(qsm + r * 128 + (((4 * half + i) ^ (r & 7)) << 4)) = qv[i];
                ptx::fence_async();
                ::ptx::mbar_arrive(&bars[16]);  // Q visible to the issuer
            }

            // local blocks b = wg + 2t
            const uint32_t n_s = nblk > wg ? (nblk - wg + 1) >> 1 : 0u;
            for (uint32_t t = 0; t < n_s; ++t) {
                const uint32_t kvp0 = begin + kTcN * (wg + 2 * t);
                bar_wait(wg, t & 1);  // S(s) ready
                ::ptx::tcgen05_fence_after_thread_sync();
                uint32_t lv[64];
                const bool mfree = mask_free(kvp0);
                float mx;
                if (mfree) {
                    float r0, r1;
                    ptx::tcgen05_ld_red_max_16x32bx2_x32<64>(kTS[wg], lv, r0);
                    ptx::tcgen05_ld_red_max_16x32bx2_x32<64>(kTS[wg] + 32, lv + 32, r1);
                    ::ptx::tcgen05_wait_ld();
                    mx = fmaxf(r0, r1) * kL2eScale;
                } else {
                    ::ptx::tcgen05_ld_16x32bx2_x32<64>(kTS[wg], lv);
                    ::ptx::tcgen05_ld_16x32bx2_x32<64>(kTS[wg] + 32, lv + 32);
                    ::ptx::tcgen05_wait_ld();
                    mx = kNegInf;
#pragma unroll
                    for (uint32_t c = 0; c < 64; ++c) {
                        const uint32_t pos = kvp0 + 64 * half + c;
                        const bool okp = pos < end && pos <= posq && pos + win >= posq;
                        const float f = okp ? reinterpret_cast<const float&>(lv[c]) * kL2eScale : kNegInf;
                        lv[c] = reinterpret_cast<const uint32_t&>(f);
                        mx = fmaxf(mx, f);
                    }
                }
                __syncwarp();  // S free: the stream's S is in registers
                if (::ptx::elect_one_sync()) ::ptx::mbar_arrive(&bars[12 + wg]);
                const float mnew = fmaxf(ml_m, fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16)));
                const float al = softmax_exp2(ml_m - mnew);
                const float lsum = exp_pack(lv, mfree, mnew);
                const float lpair = __shfl_xor_sync(0xffffffffu, lsum, 16);
                // the PV(s, t - 1) wait guards the P store and the O rescale
                if (t > 0) {
                    bar_wait(10 + wg, (t + 1) & 1);  // PV(s,t-1) done
                    ::ptx::tcgen05_fence_after_thread_sync();
                }
                ::ptx::tcgen05_st_16x32bx2_x32<32>(kTP[wg], lv);
                ml_l = ml_l * al + (lsum + lpair);  // both halves add the same pair (the 16-row path's half A + half B)
                ml_m = mnew;
                if (t > 0) rescale(kTO[wg], al);
                ::ptx::tcgen05_wait_st();
                __syncwarp();
                if (::ptx::elect_one_sync()) ::ptx::mbar_arrive(&bars[14 + wg]);  // P(s,t) ready
            }
        }

        // the streams merge as the 16-row path's: each warpgroup takes 32 columns of both streams' O (a thread 16 of
        // them), the (m, l) pairs cross in the streams' dead S columns (both halves write the pair)
        uint32_t oa[16], ob[16];
        float mh = kMSent, lh = 0.0f;
        if (rounds > 0) {
            uint32_t n_own = nblk > wg ? (nblk - wg + 1) >> 1 : 0u,
                     n_oth = nblk > (wg ^ 1) ? (nblk - (wg ^ 1) + 1) >> 1 : 0u;
            const auto pv_wait = [&](const uint32_t& s, const uint32_t& n) {  // the stream's last PV-done phase
                bar_wait(10 + s, ((n + 1) & 1));
            };
            uint32_t mx = __float_as_uint(kMSent), lx = 0u;
            // no QK writes a stream's S after its last block
            constexpr Half2 kTX{kTS[0], kTS[1]};
            ::ptx::tcgen05_st_16x32bx2_x2<2>(kTX[wg], reinterpret_cast<const uint32_t&>(ml_m),
                                             reinterpret_cast<const uint32_t&>(ml_l));
            ::ptx::tcgen05_wait_st();
            if (n_own > 0) pv_wait(wg, n_own);
            ::ptx::tcgen05_fence_before_thread_sync();
            ::ptx::sync_aligned(kAttnBarId, 256);
            // the other stream's last PV is complete by now: its wait cannot alias an earlier phase
            if (n_oth > 0) pv_wait(wg ^ 1, n_oth);
            ::ptx::tcgen05_fence_after_thread_sync();
            ::ptx::tcgen05_ld_16x32bx2_x2<2>(kTX[wg ^ 1], mx, lx);
            ::ptx::tcgen05_ld_16x32bx2_x16<16>(kTO[0] + 32 * wg, oa);
            if (nblk > 1) ::ptx::tcgen05_ld_16x32bx2_x16<16>(kTO[1] + 32 * wg, ob);
            ::ptx::tcgen05_wait_ld();
            if (nblk <= 1) {
#pragma unroll
                for (uint32_t c = 0; c < 16; ++c)
                    ob[c] = 0u;
            }
            const float m0 = wg ? reinterpret_cast<const float&>(mx) : ml_m,
                        l0 = wg ? reinterpret_cast<const float&>(lx) : ml_l;
            const float m1 = wg ? ml_m : reinterpret_cast<const float&>(mx),
                        l1 = wg ? ml_l : reinterpret_cast<const float&>(lx);
            mh = fmaxf(m0, m1);
            const float al = softmax_exp2(m0 - mh), bl = softmax_exp2(m1 - mh);
#pragma unroll
            for (uint32_t c = 0; c < 16; ++c) {
                const float f = __fmaf_rn(reinterpret_cast<const float&>(oa[c]), al,
                                          __fmul_rn(reinterpret_cast<const float&>(ob[c]), bl));
                oa[c] = reinterpret_cast<const uint32_t&>(f);
            }
            lh = __fmaf_rn(l0, al, __fmul_rn(l1, bl));
        } else {
#pragma unroll
            for (uint32_t c = 0; c < 16; ++c)
                oa[c] = 0u;
        }
        float* const ws = attn_ws + static_cast<uint64_t>(sm_idx) * kWsRow;
        float* const wrow = ws + r * 64 + 32 * wg + 16 * half;
#pragma unroll
        for (uint32_t d = 0; d < 2; ++d) {
            const uint4 s0 = make_uint4(oa[8 * d], oa[8 * d + 1], oa[8 * d + 2], oa[8 * d + 3]);
            const uint4 s1 = make_uint4(oa[8 * d + 4], oa[8 * d + 5], oa[8 * d + 6], oa[8 * d + 7]);
            ptx::st_row_u8_cg(reinterpret_cast<uint4*>(wrow + 8 * d), s0, s1);
        }
        if (wg0 && half == 0) {
            const uint2 ml2 = make_uint2(reinterpret_cast<const uint32_t&>(mh), reinterpret_cast<const uint32_t&>(lh));
            ptx::st_row_u2_cg(reinterpret_cast<uint2*>(ws + kUnitRows * 64 + r * 2), ml2);
        }
    }
    ::ptx::sync_aligned(kAttnBarId, 256);  // rows before the release
    if (threadIdx.x == kEpiBase) ptx::arrival_count_publish(args.fctr + kCtrAttnParts + u * kAttnPartStride + j, 1);
}

}  // namespace attn

// The decode attention phase (one layer's attention, split-KV) and its combine. Inputs are the previous launch's
// outputs: the qkv rows (q / k already normed, roped and pre-scaled) and the bf16 paged KV pool. A unit is one
// (sequence, kv head) Q tile of 8 x kM rows (row = q head x kM + token, GQA 8:1, head_dim 64; kM = attn_cfg::kM, 16 or
// 8); the 16-row window [max(0, P - window), P + 16) is cut into kSplits chunks, one split per CTA, and a split
// attends its chunk's part of the window's [max(0, P - window), P + kM). A split streams its chunk's KV rows, runs the
// online softmax in fp32 log2 space and writes a log2-space (m, l, O) partial to attn_ws, then publishes its per-split
// ready word. Combine CTAs (kGroups row groups per unit, on the unit's own first split CTAs) wait for their unit's
// split CTAs, merge the partials in fixed split order (deterministic split-KV), apply the gate-norm epilogue (per-head
// RMS norm, eps 1e-5, then x * sigmoid(o_gate)) and store x_attn, then publish kCtrAttnOut. Two attention variants,
// selected per attention type (kImage.global_attn):
//  sliding window  cp.async KV slices into an smem ring, warp-level mma.sync QK / PV on the 8 epilogue warps
//  global          TMA KV blocks (128 rows), tcgen05 QK / PV with S, P and O in TMEM, two feed warps, one MMA issuer
//                  warp and two softmax warpgroups on alternate blocks (the "parity streams"), merged in the split

namespace attn {

// The head phase: the split CTAs and the combine on every CTA of the grid, inside the post-attention ring (slots of
// kSlotBytes; the 256 epilogue threads from kEpiBase). kGlobalAttn: the global attention, its carve ring slots 1..4
// (the feed and issuer warps below the epilogue band; its feeds retire their last empty arrivals before the carve
// returns to the ring); else the mma.sync attention loop on the epilogue warps, its ring one 16-row block per slot in
// the slots' x regions (swa_ring), its combine publishing x_attn for TMA reads (an async-proxy fence). kEpochWait: the
// combine waits for the gate warp's epoch bump (the sliding-window attention inside the post-attention body, whose
// bump lands under the attention).
template <bool kGlobalAttn, bool kEpochWait, uint32_t kSlotBytes, uint32_t kEpiBase>
static __device__ __forceinline__ void attn_head_phase(const AttnArgs& args, uint8_t* smem_buffer,
                                                       const uint32_t& sm_idx, const uint32_t& warp_idx,
                                                       const uint32_t& lane_idx, const uint32_t& epoch,
                                                       const CUtensorMap* tmap_k, const CUtensorMap* tmap_v,
                                                       const CUtensorMap* tmap_kb, const CUtensorMap* tmap_vb,
                                                       uint8_t* const swa_ring) {
    namespace fcfg = attn_cfg;
    constexpr uint32_t kM = fcfg::kM;
    constexpr uint32_t kUnits = attn_units(kM);  // (seq, kv-head) Q tiles
    constexpr uint32_t kNumSMs = cfg::kGrid;
    // the splits: one per CTA
    constexpr uint32_t kSplits = attn_splits(kNumSMs, kM);
    constexpr uint32_t kAttnCtasUsed = kUnits * kSplits;
    // the combine width: kGroups row groups per unit, kRowsPG rows per combine CTA, kTPR threads per row
    constexpr uint32_t kGroups = kAttnGroups;
    constexpr uint32_t kRowsPG = kUnitRows / kGroups;                   // rows per combine CTA
    constexpr uint32_t kTPR = 256 / kRowsPG < 32 ? 256 / kRowsPG : 32;  // threads per row (one warp at most)
    constexpr uint32_t kDPT = 64 / kTPR;                                // dims per thread
    constexpr uint32_t kActive = kRowsPG * kTPR;                        // epi threads owning a row slice
    constexpr uint32_t kCombines = kUnits * kGroups;
    static_assert(kCombines == attn_combines(kNumSMs, kM), "combine count must match the dispatch gate target");
    constexpr uint32_t kWindow = fcfg::kAttnWindow;  // SWA 2047 | GLOBAL lo:0 sentinel
    // The sliding-window ring: only the unit's kv head's 128 B k / v slices are fetched, in 16 B cp.async pieces spread
    // over all 256 epilogue threads (one piece each); the warp mma.sync path consumes 16-row blocks. A slot is
    // [k 128 | v 128 | 16 B pad]: the 256 B stride lands every k-slice row on one bank (an 8-way conflict per QK
    // load), 272 makes the warp's k loads conflict-free ((68r + q) mod 32 = 4r + q) and halves the ldmatrix row
    // collisions.
    constexpr uint32_t kSlotB = 272;
    // the ring: kKDepth blocks of 16 slots in flight per split
    constexpr uint32_t kKDepth = fcfg::kSwaDepth;
    constexpr uint32_t kKSlots = 16 * kKDepth;
    static_assert(kSplits >= kGroups && kSplits <= kAttnPartStride && kAttnCtasUsed <= kNumSMs &&
                      kCombines <= kNumSMs && kDPT == 2 && kActive <= 256,
                  "attn_ws / combine capacity (own-unit aligned mapping): two dims per thread");
    static_assert(block::kPhaseBarBase != kAttnBarId && block::kPhaseBarBase != block::kItem,
                  "the helper barrier is distinct from the sliding-window attention's barriers");

    const uint32_t epi_tid = threadIdx.x - kEpiBase;
    const uint32_t epi_warp = warp_idx - kEpiBase / 32;

    // the global attention's carve (ring slots 1..4) and the mma.sync attention loop's ring rows (row r: slot r / 16,
    // 272-B slot r % 16)
    uint8_t* const region = smem_buffer + kSlotBytes;
    const auto swa_row = [&](const uint32_t row) { return swa_ring + (row / 16) * kSlotBytes + (row % 16) * kSlotB; };

    // the global attention: its feed warps and MMA issuer run below the softmax band, with no epilogue duties
    if constexpr (kGlobalAttn) {
        if (warp_idx < kEpiBase / 32) {
            if (sm_idx < kAttnCtasUsed)
                attn_global<kEpiBase>(args, smem_buffer, region, sm_idx, warp_idx, lane_idx, tmap_k, tmap_v, tmap_kb,
                                      tmap_vb);
            return;
        }
    }

    const auto qkv_out = reinterpret_cast<const nv_bfloat16*>(args.arena + fcfg::kOff_qkv_out);
    const auto attn_ws = reinterpret_cast<float*>(args.arena + fcfg::kOff_attn_ws);
    uint32_t* const fctr = args.fctr;

    // ---- the sliding-window attention: one (unit, chunk) per CTA ----
    if (!kGlobalAttn && sm_idx < kAttnCtasUsed) {
        const uint32_t u = sm_idx / kSplits, j = sm_idx % kSplits;
        const uint32_t seq = u / fcfg::kNumKvHeads, kvh = u % fcfg::kNumKvHeads;
        // the 16-row window's chunks and blocks (flen = kv_len = P + 16), so the block loop (the pair round
        // included) is the 16-row window's on every split; the rows past the window's end, len, are causally masked
        // for every row
        const uint32_t flen = static_cast<uint32_t>(__ldg(args.attn_kv_len + seq));
        const uint32_t len = flen - (fcfg::kFeedRows - kM);
        const uint32_t lo = flen > 16 + kWindow ? flen - 16 - kWindow : 0u;
        const uint32_t chunk = (flen - lo + kSplits - 1) / kSplits;
        const uint32_t begin = lo + j * chunk;
        const uint32_t end = min_u32(begin + chunk, flen);

        const int64_t* ptab = args.attn_page_table + static_cast<uint64_t>(seq) * args.attn_page_stride;
        const uint32_t rounds = begin < end ? end - begin : 0u;  // one row per KV position
        const uint32_t nblk = (rounds + 15) / 16;
        // each thread's next table offset preloads one block ahead (a single pending register; blocks issue
        // in order), so the __ldg(ptab) -> cp.async chain does not carry two serial memory latencies
        uint64_t next_page_off = 0;
        const uint32_t rd0 = epi_tid >> 4;
        if (rd0 < rounds) next_page_off = static_cast<uint64_t>(__ldg(ptab + begin + rd0));
        // one 16-row block per commit group: 256 x 16 B cp.async pieces, one per epilogue thread; cp.async
        // groups are per thread, so every thread commits one group per block slot and waits its own
        const auto issue_block = [&](const uint32_t& b) {
            const uint32_t p = epi_tid;
            const uint32_t rd = 16 * b + (p >> 4);
            const uint32_t idx = p & 15, sub = p & 7;
            uint8_t* const dptr = swa_row(rd % kKSlots) + (idx < 8 ? 0 : 128) + sub * 16;
            if (rd < rounds) {
                const uint64_t off = next_page_off;
                const uint8_t* const src = (idx < 8 ? args.k_cache : args.v_cache) + off + kvh * 128 + sub * 16;
                const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dptr));
                ptx::cp_async_cg_16(dst, src);
                if (rd + 16 < rounds) next_page_off = static_cast<uint64_t>(__ldg(ptab + begin + rd + 16));
            } else {
                // tail rows: zero-filled, so the PV mma (which spans all 16 KV rows, P = 0 on masked
                // lanes) never meets NaN bits
                *reinterpret_cast<uint4*>(dptr) = make_uint4(0u, 0u, 0u, 0u);
            }
            ptx::cp_async_commit_group();
        };
        // group-count invariant: exactly kKDepth + b groups are committed per thread at the top of block b,
        // so wait_group kKDepth - 1 proves block b landed
        const auto commit_empty = [&]() { ptx::cp_async_commit_group(); };
        // The follower's wait. The static prologue (sm_epoch, kv_len, the first table offset) already ran
        // under the predecessor's tail; the KV pool's newest rows and the qkv rows are the predecessor's, so
        // every epilogue thread waits here, ahead of the Q load. The ring's first kKDepth blocks are older
        // positions (< flen - 16, written by earlier steps) on every split but a unit's last, so those issue
        // ahead of the wait; the last split, or a window too short to hold them, issues behind it.
        const bool prefix_static = rounds > 0 && flen >= 16u + 16u * kKDepth && begin + 16u * kKDepth <= flen - 16u;
        if (prefix_static) {
            for (uint32_t b = 0; b < kKDepth; ++b) {
                if (b < nblk)
                    issue_block(b);
                else
                    commit_empty();
            }
        }
        ::ptx::griddepcontrol_wait();
        if (rounds > 0 && !prefix_static) {
            for (uint32_t b = 0; b < kKDepth; ++b) {
                if (b < nblk)
                    issue_block(b);
                else
                    commit_empty();
            }
        }

        // The mma.sync path: warp w = the tile's rows [16 w, 16 w + 16) (q head w's 16 tokens; at kM = 8 q
        // heads 2 w and 2 w + 1, 8 tokens each, so 4 warps cover the tile and the rest only stage KV); QK^T and
        // PV via m16n8k16 (bf16 operands, fp32 accumulators) over 16-row KV blocks. The softmax scale applies to
        // the fp32 logits (q stays raw bf16 in the A fragments); online (m, l) in log2 space per row, the block
        // max by quad shuffles. A thread holds fragment rows tq and tq + 8. Each row's arithmetic is its own, so
        // an 8-row tile's rows equal the 16-row tile's first 8 tokens bitwise.
        const uint32_t quad = lane_idx & 3, tq = lane_idx >> 2;
        const uint32_t qh = epi_warp;
        const bool tile_warp = kM == 16 || epi_warp < kUnitRows / 16;
        const uint32_t tok0 = seq * kM + tq;
        // fragment row tq + 8: token tq + 8 of q head qh, or (kM = 8) token tq of q head 2 qh + 1
        // a tree window: fragment rows tq / tq + 8 at P + their depths, seeing the window keys of their ancestor masks
        const uint32_t base = len - kM;
        const uint32_t posq0 = fcfg::kTree ? base + __ldg(args.attn_kv_len + fcfg::kTreeDepthWord + tq) : len - kM + tq;
        const uint32_t posq1 = fcfg::kTree ? base + __ldg(args.attn_kv_len + fcfg::kTreeDepthWord + tq + 8)
                               : kM == 16  ? posq0 + 8
                                           : posq0;
        const uint32_t tmask0 = fcfg::kTree ? __ldg(args.attn_kv_len + fcfg::kTreeMaskWord + tq) : 0u;
        const uint32_t tmask1 = fcfg::kTree ? __ldg(args.attn_kv_len + fcfg::kTreeMaskWord + tq + 8) : 0u;
        constexpr float kL2eScale = 0.125f * 1.4426950408889634f;
        const float kNegInf = -__int_as_float(0x7f800000);
        uint32_t aq[16];  // 4 k-blocks x {a0..a3} raw bf16 pairs
        if (tile_warp) {
            const auto q0 = reinterpret_cast<const uint32_t*>(qkv_out + static_cast<uint64_t>(tok0) * kQkvRow +
                                                              (kvh * 8 + (kM == 16 ? qh : 2 * qh)) * 64);
            const auto q1 = reinterpret_cast<const uint32_t*>(
                qkv_out + static_cast<uint64_t>(kM == 16 ? tok0 + 8 : tok0) * kQkvRow +
                (kvh * 8 + (kM == 16 ? qh : 2 * qh + 1)) * 64);
#pragma unroll
            for (uint32_t kb = 0; kb < 4; ++kb) {
                const uint32_t ci = kb * 8 + quad;  // u32 (bf16 pair) index
                aq[4 * kb + 0] = __ldg(q0 + ci);
                aq[4 * kb + 1] = __ldg(q1 + ci);
                aq[4 * kb + 2] = __ldg(q0 + ci + 4);
                aq[4 * kb + 3] = __ldg(q1 + ci + 4);
            }
        }
        float oc[32];  // 8 dim-tiles x {c0..c3}
#pragma unroll
        for (uint32_t i = 0; i < 32; ++i)
            oc[i] = 0.0f;
        float m2[2] = {kMSent, kMSent}, l2[2] = {0.0f, 0.0f};

        // the pair round: at nblk == 2 both blocks are already resident (issued at priming, kKDepth == 2),
        // so one wait + sync runs both blocks' QK / PV back to back with a single online-softmax pass (the
        // (m, l) update reassociates: tolerance-equal, not bitwise, to the two-round loop)
        bool pair_done = false;
        if (nblk == 2) {
            ptx::cp_async_wait_group<0>();
            ::ptx::sync_aligned(kAttnBarId, 256);
            if (tile_warp) {
                float lg[16];
#pragma unroll
                for (uint32_t t = 0; t < 4; ++t) {
                    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                    const auto krow = reinterpret_cast<const uint32_t*>(swa_row(8 * t + tq));
#pragma unroll
                    for (uint32_t kb = 0; kb < 4; ++kb)
                        ptx::mma_m16n8k16_bf16(c0, c1, c2, c3, aq[4 * kb], aq[4 * kb + 1], aq[4 * kb + 2],
                                               aq[4 * kb + 3], krow[kb * 8 + quad], krow[kb * 8 + quad + 4]);
                    lg[4 * t + 0] = c0;
                    lg[4 * t + 1] = c1;
                    lg[4 * t + 2] = c2;
                    lg[4 * t + 3] = c3;
                }
#pragma unroll
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint32_t pos = begin + 8 * (i >> 2) + quad * 2 + (i & 1);
                    const uint32_t pq = ((i & 3) < 2) ? posq0 : posq1;
                    const bool ok = fcfg::kTree
                                        ? tree_key_visible(pos, end, pq, kWindow, base, ((i & 3) < 2) ? tmask0 : tmask1)
                                        : pos < end && pos <= pq && pos + kWindow >= pq;
                    lg[i] = ok ? lg[i] * kL2eScale : kNegInf;
                }
                float bm0 = fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[4], lg[5]));
                float bm1 = fmaxf(fmaxf(lg[2], lg[3]), fmaxf(lg[6], lg[7]));
                bm0 = fmaxf(bm0, fmaxf(fmaxf(lg[8], lg[9]), fmaxf(lg[12], lg[13])));
                bm1 = fmaxf(bm1, fmaxf(fmaxf(lg[10], lg[11]), fmaxf(lg[14], lg[15])));
#pragma unroll
                for (uint32_t ofs = 1; ofs < 4; ofs <<= 1) {
                    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, ofs));
                    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, ofs));
                }
                // fresh accumulators (the pair round is the only round):
                // m = max(sentinel, bm), l = rs, O accumulates from zero
                m2[0] = fmaxf(m2[0], bm0);
                m2[1] = fmaxf(m2[1], bm1);
                float pr[16];
#pragma unroll
                for (uint32_t i = 0; i < 16; ++i)
                    pr[i] = softmax_exp2(lg[i] - (((i & 3) < 2) ? m2[0] : m2[1]));
                float rs0 = ((pr[0] + pr[1]) + (pr[4] + pr[5])) + ((pr[8] + pr[9]) + (pr[12] + pr[13]));
                float rs1 = ((pr[2] + pr[3]) + (pr[6] + pr[7])) + ((pr[10] + pr[11]) + (pr[14] + pr[15]));
#pragma unroll
                for (uint32_t ofs = 1; ofs < 4; ofs <<= 1) {
                    rs0 += __shfl_xor_sync(0xffffffffu, rs0, ofs);
                    rs1 += __shfl_xor_sync(0xffffffffu, rs1, ofs);
                }
                l2[0] = rs0;
                l2[1] = rs1;
#pragma unroll
                for (uint32_t c = 0; c < 2; ++c) {
                    uint32_t pa[4];
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const nv_bfloat162 v = __float22bfloat162_rn({pr[8 * c + 2 * k], pr[8 * c + 2 * k + 1]});
                        pa[k] = reinterpret_cast<const uint32_t&>(v);
                    }
                    const uint8_t* const vsb = swa_row(c * 16) + 128;
#pragma unroll
                    for (uint32_t d = 0; d < 8; ++d) {
                        uint32_t v0, v1;
                        ptx::ldmatrix_x2_trans(v0, v1, ptx::to_shared(vsb + (lane_idx & 15) * kSlotB + d * 16));
                        ptx::mma_m16n8k16_bf16(oc[4 * d], oc[4 * d + 1], oc[4 * d + 2], oc[4 * d + 3], pa[0], pa[1],
                                               pa[2], pa[3], v0, v1);
                    }
                }
            }
            pair_done = true;
        }
        if (!pair_done)
            for (uint32_t blk = 0; blk < nblk; ++blk) {
                // wait_group kKDepth - 1 proves this block landed (the group-count invariant above)
                ptx::cp_async_wait_group<kKDepth - 1>();
                ::ptx::sync_aligned(kAttnBarId, 256);
                if (tile_warp) {
                    constexpr uint32_t kRowB = kSlotB;
                    const uint8_t* const sbase = swa_row((blk % kKDepth) * 16);
                    const uint8_t* const vsbase = sbase + 128;
                    const uint32_t kv0 = begin + 16 * blk;
                    // QK^T: two n8 tiles over the block (B = plain 4B lds of k slices)
                    float lg[8];
#pragma unroll
                    for (uint32_t t = 0; t < 2; ++t) {
                        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                        const auto krow = reinterpret_cast<const uint32_t*>(sbase + (8 * t + tq) * kRowB);
#pragma unroll
                        for (uint32_t kb = 0; kb < 4; ++kb)
                            ptx::mma_m16n8k16_bf16(c0, c1, c2, c3, aq[4 * kb], aq[4 * kb + 1], aq[4 * kb + 2],
                                                   aq[4 * kb + 3], krow[kb * 8 + quad], krow[kb * 8 + quad + 4]);
                        lg[4 * t + 0] = c0;
                        lg[4 * t + 1] = c1;
                        lg[4 * t + 2] = c2;
                        lg[4 * t + 3] = c3;
                    }
                // scale + causal/SWA/tail mask (masked -> -inf so exp2f gives 0
                // even while the row max is still the sentinel)
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) {
                        const uint32_t pos = kv0 + 8 * (i >> 2) + quad * 2 + (i & 1);
                        const uint32_t pq = ((i & 3) < 2) ? posq0 : posq1;
                        const bool ok =
                            fcfg::kTree ? tree_key_visible(pos, end, pq, kWindow, base, ((i & 3) < 2) ? tmask0 : tmask1)
                                        : pos < end && pos <= pq && pos + kWindow >= pq;
                        lg[i] = ok ? lg[i] * kL2eScale : kNegInf;
                    }
                    // block max + row sum per fragment row (quad shuffles)
                    float bm0 = fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[4], lg[5]));
                    float bm1 = fmaxf(fmaxf(lg[2], lg[3]), fmaxf(lg[6], lg[7]));
#pragma unroll
                    for (uint32_t ofs = 1; ofs < 4; ofs <<= 1) {
                        bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, ofs));
                        bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, ofs));
                    }
                    const float mn0 = fmaxf(m2[0], bm0), mn1 = fmaxf(m2[1], bm1);
                    const float al0 = softmax_exp2(m2[0] - mn0), al1 = softmax_exp2(m2[1] - mn1);
                    m2[0] = mn0;
                    m2[1] = mn1;
                    float pr[8];
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i)
                        pr[i] = softmax_exp2(lg[i] - (((i & 3) < 2) ? mn0 : mn1));
                    float rs0 = (pr[0] + pr[1]) + (pr[4] + pr[5]);
                    float rs1 = (pr[2] + pr[3]) + (pr[6] + pr[7]);
#pragma unroll
                    for (uint32_t ofs = 1; ofs < 4; ofs <<= 1) {
                        rs0 += __shfl_xor_sync(0xffffffffu, rs0, ofs);
                        rs1 += __shfl_xor_sync(0xffffffffu, rs1, ofs);
                    }
                    l2[0] = l2[0] * al0 + rs0;
                    l2[1] = l2[1] * al1 + rs1;
                // O rescale, then P*V: the QK^T D fragment re-packs in-register
                // into the P*V A fragment (bf16); V B fragments via ldmatrix.trans
#pragma unroll
                    for (uint32_t d = 0; d < 8; ++d) {
                        oc[4 * d + 0] *= al0;
                        oc[4 * d + 1] *= al0;
                        oc[4 * d + 2] *= al1;
                        oc[4 * d + 3] *= al1;
                    }
                    uint32_t pa[4];
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const nv_bfloat162 v = __float22bfloat162_rn({pr[2 * k], pr[2 * k + 1]});
                        pa[k] = reinterpret_cast<const uint32_t&>(v);
                    }
#pragma unroll
                    for (uint32_t d = 0; d < 8; ++d) {
                        uint32_t v0, v1;
                        ptx::ldmatrix_x2_trans(v0, v1, ptx::to_shared(vsbase + (lane_idx & 15) * kRowB + d * 16));
                        ptx::mma_m16n8k16_bf16(oc[4 * d], oc[4 * d + 1], oc[4 * d + 2], oc[4 * d + 3], pa[0], pa[1],
                                               pa[2], pa[3], v0, v1);
                    }
                }
                // all readers past this half before its next cp.async block; the
                // empty commit keeps the group-count invariant at the tail
                ::ptx::sync_aligned(kAttnBarId, 256);
                if (blk + kKDepth < nblk)
                    issue_block(blk + kKDepth);
                else
                    commit_empty();
            }

        // the log2-space (m, l, O) partial -> attn_ws (.cg stores: cross-SM rows of this kernel, under the
        // release / acquire of the ready word)
        float* const ws = attn_ws + static_cast<uint64_t>(sm_idx) * kWsRow;
        if (tile_warp) {
            const uint32_t r0 = qh * 16 + tq, r1 = r0 + 8;
            const auto pub_u2 = [](uint2* p, const uint2& v) { ptx::st_row_u2_cg(p, v); };
#pragma unroll
            for (uint32_t d = 0; d < 8; ++d) {
                uint2 s0, s1;
                s0.x = reinterpret_cast<const uint32_t&>(oc[4 * d]);
                s0.y = reinterpret_cast<const uint32_t&>(oc[4 * d + 1]);
                s1.x = reinterpret_cast<const uint32_t&>(oc[4 * d + 2]);
                s1.y = reinterpret_cast<const uint32_t&>(oc[4 * d + 3]);
                pub_u2(reinterpret_cast<uint2*>(ws + r0 * 64 + 8 * d + quad * 2), s0);
                pub_u2(reinterpret_cast<uint2*>(ws + r1 * 64 + 8 * d + quad * 2), s1);
            }
            if (quad == 0) {
                uint2 ml0, ml1;
                ml0.x = reinterpret_cast<const uint32_t&>(m2[0]);
                ml0.y = reinterpret_cast<const uint32_t&>(l2[0]);
                ml1.x = reinterpret_cast<const uint32_t&>(m2[1]);
                ml1.y = reinterpret_cast<const uint32_t&>(l2[1]);
                pub_u2(reinterpret_cast<uint2*>(ws + kUnitRows * 64 + r0 * 2), ml0);
                pub_u2(reinterpret_cast<uint2*>(ws + kUnitRows * 64 + r1 * 2), ml1);
            }
        }
        ::ptx::sync_aligned(kAttnBarId, 256);  // rows before the release
        // the per-split ready word (a unit's words are 64 apart = 2 lines, so a unit's poller never shares
        // a line with another unit's writers)
        if (epi_tid == 0) ptx::arrival_count_publish(fctr + kCtrAttnParts + u * kAttnPartStride + j, 1);
    }

    // ---- the global attention's softmax warpgroups on the epilogue warps ----
    if constexpr (kGlobalAttn) {
        if (sm_idx < kAttnCtasUsed)
            attn_global<kEpiBase>(args, smem_buffer, region, sm_idx, warp_idx, lane_idx, tmap_k, tmap_v, tmap_kb,
                                  tmap_vb);
    }

    // ---- the combine: fixed-order split-KV merge + gate-norm epilogue ----
    // The combine CTAs are the first kGroups split CTAs of each unit, so a unit's merge starts the moment its own
    // split CTAs publish.
    if (sm_idx < kAttnCtasUsed && sm_idx % kSplits < kGroups) {
        const uint32_t u = sm_idx / kSplits, g = sm_idx % kSplits;
        const uint32_t seq = u / fcfg::kNumKvHeads, kvh = u % fcfg::kNumKvHeads;
        const uint32_t r = g * kRowsPG + epi_tid / kTPR;  // row in the unit's tile
        const uint32_t d4 = (epi_tid % kTPR) * kDPT;
        const uint32_t qh = r / kM, ti = r % kM;
        const uint32_t tok = seq * kM + ti;
        // one parallel spin over the unit's split words (thread j spins word j: one round trip in total),
        // then the merge in batches of 8 splits in fixed order; an empty split (l = 0, m = kMSent, O = 0)
        // merges as an exact no-op
        const auto load_og = [](const nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint32_t*>(p)); };
        if constexpr (kEpochWait) {
            // the in-body attention's epoch is the gate warp's bump, landing under the attention (zero until then)
            while (*reinterpret_cast<const volatile uint32_t*>(&epoch) == 0u) {
            };
        }
        // the o-gate row is the predecessor's, not the split CTAs', so its load issues ahead of the spin and lands
        // under it
        uint32_t og_pre = 0;
        if (kActive == 256 || epi_tid < kActive) {
            const auto og_p = qkv_out + static_cast<uint64_t>(tok) * kQkvRow + kOgCol + (kvh * 8 + qh) * 64 + d4;
            og_pre = load_og(og_p);
        }
        if (epi_tid < kSplits) ptx::count_spin(fctr + kCtrAttnParts + u * kAttnPartStride + epi_tid, epoch);
        ::ptx::sync_aligned(kAttnBarId, 256);
        if (kActive == 256 || epi_tid < kActive) {
            float M = kMSent, L = 0.0f;
            float acc[kDPT] = {};
            // the merge: every split's (m, l) and O loads in batches, then the ordered merges
            // one batch of eight splits: its (m, l) + O loads, then its eight ordered merges (a partial last batch
            // guards its splits)
            const auto load8 = [&](const uint32_t j0, uint2(&ml8)[8], uint4(&ov8)[8]) {
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    if (kSplits % 8 == 0 || j0 + k < kSplits) {
                        const float* const ws = attn_ws + static_cast<uint64_t>(u * kSplits + j0 + k) * kWsRow;
                        ml8[k] = ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(ws + kUnitRows * 64 + r * 2));
                        const uint2 o2 = ptx::ld_row_u2_cg(reinterpret_cast<const uint2*>(ws + r * 64 + d4));
                        ov8[k].x = o2.x;
                        ov8[k].y = o2.y;
                    }
                }
            };
            const auto merge8 = [&](const uint32_t j0, const uint2(&ml8)[8], const uint4(&ov8)[8]) {
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    if (kSplits % 8 == 0 || j0 + k < kSplits) {
                        const auto of = reinterpret_cast<const float*>(&ov8[k]);
                        const float mjk = reinterpret_cast<const float&>(ml8[k].x);
                        const float ljk = reinterpret_cast<const float&>(ml8[k].y);
                        const float mn = fmaxf(M, mjk);
                        const float al = softmax_exp2(M - mn), bl = softmax_exp2(mjk - mn);
#pragma unroll
                        for (uint32_t q4 = 0; q4 < kDPT; ++q4)
                            acc[q4] = acc[q4] * al + of[q4] * bl;
                        L = L * al + ljk * bl;
                        M = mn;
                    }
                }
            };
            if constexpr (kSplits % 16 == 8) {
                // an odd number of whole batches (TP8: 72 splits): two batches per rolled iteration, batch b + 1
                // loading under batch b's merges; the last batch merges after the loop
                uint2 mlA[8], mlB[8];
                uint4 ovA[8], ovB[8];
                load8(0, mlA, ovA);
#pragma unroll 1
                for (uint32_t j0 = 0; j0 + 8 < kSplits; j0 += 16) {
                    load8(j0 + 8, mlB, ovB);
                    merge8(j0, mlA, ovA);
                    load8(j0 + 16, mlA, ovA);
                    merge8(j0 + 8, mlB, ovB);
                }
                merge8(kSplits - 8, mlA, ovA);
            } else {
                // the batches unrolled (TP4: 36 splits, the last batch partial); the same order
#pragma unroll
                for (uint32_t j0 = 0; j0 < kSplits; j0 += 8) {
                    uint2 ml8[8];
                    uint4 ov8[8];
                    load8(j0, ml8, ov8);
                    merge8(j0, ml8, ov8);
                }
            }
            // x = O / L (L > 0: the row's own append position is always visible), then the per-head
            // weightless RMS norm (eps 1e-5, fp32), then x * sigmoid(og)
            float x[kDPT];
#pragma unroll
            for (uint32_t k = 0; k < kDPT; ++k)
                x[k] = acc[k] / L;
            float rstd = 1.0f;
            float ss = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < kDPT; ++k)
                ss += x[k] * x[k];
#pragma unroll
            for (uint32_t ofs = 1; ofs < kTPR; ofs <<= 1)  // kTPR-lane row group
                ss += __shfl_xor_sync(0xffffffffu, ss, ofs);
            rstd = rsqrtf(ss * (1.0f / 64) + 1e-5f);
            const auto sig = [](const float& v) { return 1.0f / (1.0f + __expf(-v)); };
            const uint64_t x_off =
                fcfg::kOff_x_attn + (static_cast<uint64_t>(tok) * kXAttnRow + (kvh * 8 + qh) * 64 + d4) * 2;
            uint8_t* const x_row = args.arena + x_off;
            const uint32_t ogv = og_pre;
            const auto ogb = reinterpret_cast<const nv_bfloat16*>(&ogv);
            const float s0 = sig(__bfloat162float(ogb[0]));
            const float s1 = sig(__bfloat162float(ogb[1]));
            const nv_bfloat162 xb = __float22bfloat162_rn({x[0] * rstd * s0, x[1] * rstd * s1});
            const uint32_t pack = reinterpret_cast<const uint32_t&>(xb);
            ptx::st_row_u32_volatile(reinterpret_cast<uint32_t*>(x_row), pack);
        }
        if constexpr (!kGlobalAttn) ptx::fence_async_global();
        ::ptx::sync_aligned(kAttnBarId, 256);  // rows before the release
        if (epi_tid == 0) ptx::arrival_count_publish(fctr + kCtrAttnOut, 1);
    }
}

}  // namespace attn

}  // namespace mk

// The global layers' attention policy: the global attention as the post-attention phase's leading part, one split
// per CTA over the (seq, kv head) units, then a CTA barrier. The attention's carve is ring slots 1..4 (the
// post-attention header, out tile, exchange buffer and barriers above the ring are untouched) and it owns the whole
// TMEM pool; the post-attention 64-column allocation follows the attention's dealloc. Roles: the post-attention W
// producer and gate warps are the attention's two feed warps, the MMA warp its issuer, the eight epilogue warps
// its softmax warpgroups; the aux warp prefetches the code. A short context takes the mma.sync attention loop instead
// (use_tcgen05). The post-attention body then starts on a free TMEM pool: its policy part is only the post-attention
// allocation and the x gate, whose counters are the swa policy's (layer.cu SwaAttention).

// the global layers run the tcgen05 attention, every chunk able to hold a 128-row box
static_assert(!mk::kImage.global_attn || mk::attn::attn_box_live(), "the global attention's 128-row box");

namespace mk {

// What both attention policies share (GlobalAttention below, layer.cu SwaAttention): the attention's epoch word and
// done barrier in unused post-attention header bytes (past the 64-column TMEM pointer, past the mbarrier pool), and
// the expert prefetch (moe.cuh) on the post-attention body's aux warp behind its claims, which follow the attention,
// so the expert prefetch's loads never share the attention's KV stream. The expert prefetch's 1 KB
// histogram and its picks (FrontScratch::hot, read by the front item's tile list) live in the front unit's scratch
// (static, at the post-attention carve's tail): the ring's activation regions are live from the first x slot on and
// get no fence against generic-proxy writes.
struct AttnWords {
    using PrefetchArgs = expert_prefetch::Args;
    static constexpr uint32_t kGrid = cfg::kGrid;
    static constexpr uint32_t kEpochOff = postattn::kTmemOff + 8;
    static constexpr uint32_t kDoneOff = postattn::kBarOff + postattn::kBars * sizeof(uint64_t);
    static_assert(kEpochOff + sizeof(uint64_t) <= postattn::kBarOff &&
                      kDoneOff + sizeof(uint64_t) <= postattn::kSmemBytes,
                  "attention synchronization fits the unused post-attention header bytes");
    static_assert(sizeof(MoeFrontItem<true>::Scratch) >=
                      (expert_prefetch::kExperts + expert_prefetch::kSelected + 1) * sizeof(uint32_t),
                  "the histogram fits the front unit's scratch");

    static __device__ __forceinline__ uint64_t* epoch(uint8_t* smem) {
        return reinterpret_cast<uint64_t*>(smem + kEpochOff);
    }
    static __device__ __forceinline__ uint64_t* done(uint8_t* smem) {
        return reinterpret_cast<uint64_t*>(smem + kDoneOff);
    }
    static __device__ __forceinline__ void retire(uint8_t* smem) { ptx::mbar_inval(done(smem)); }
    // the attention done on this CTA
    static __device__ __forceinline__ void wait_attn(uint8_t* smem) { ptx::mbar_wait_parity_poll(done(smem), 0); }
    // the expert prefetch on the global layers only at the wide budget (cfg::kWorld.prefetch_wide)
    static __device__ __forceinline__ void run_prefetch(const PrefetchArgs& p, uint32_t* histogram, uint32_t* hot,
                                                        const uint32_t lane) {
        if constexpr (cfg::kWorld.prefetch_wide || !kImage.global_attn)
            expert_prefetch::run_uniform<kGrid, postattn::kM>(p, histogram, hot, blockIdx.x, lane);
        else if (lane < expert_prefetch::kExperts / 32)
            hot[lane] = 0;
    }
};

struct GlobalAttention : AttnWords {
    static constexpr bool kGlobalAttn = true;
    struct Args {
        AttnArgs a;
        CUtensorMap tk, tv, tkb, tvb;  // the paged pool's per-row maps and their block-grain box variants
    };
    // the carve from slot 1 (attn_head_phase)
    static constexpr uint32_t kRegionOff = postattn::kSlotBytes;
    static constexpr uint32_t kAttnPhaseBar = block::kPhaseBarBase + 3, kAttnThreads = 11 * 32;
    static constexpr uint32_t kEpiBase = kPostAttnEpi0 * 32;
    static_assert(attn_cfg::kM >= postattn::kM && attn::kGlobalFeed == 2,
                  "the post-attention W producer and gate warps feed");
    static_assert(kRegionOff % 1024 == 0 && kRegionOff + attn_cfg::kAttnSmemBytes + 1024 <= postattn::kRingBytes,
                  "the global attention's carve fits the post-attention ring (slots 1..4)");
    static_assert(kPostAttnGate == 1 && kPostAttnMma == 2 && kPostAttnEpi0 == 4 && kPostAttnWProducer == 0 &&
                      kPostAttnAux == 3,
                  "the warp remap below is the twelve-warp role map");
    static __device__ __forceinline__ void init(uint8_t* smem) { ptx::mbar_init(done(smem), 1); }
    // the attention's warp number for a post-attention warp: the W producer and the gate warp feed (the two warps
    // below the issuer, one parity stream each), the MMA warp issues; the epilogue warps keep their numbers
    static __device__ __forceinline__ uint32_t attn_warp(const uint32_t warp) {
        return warp == kPostAttnWProducer
                   ? kEpiBase / 32 - 3
                   : (warp == kPostAttnGate ? kEpiBase / 32 - 2 : (warp == kPostAttnMma ? kEpiBase / 32 - 1 : warp));
    }
    // the path selection on the device-read context length (CTA-uniform): the tcgen05 path when a split's chunk
    // holds a whole 128-row box, else the mma.sync attention loop
    static __device__ __forceinline__ bool use_tcgen05(const Args& a) {
        constexpr uint32_t kSplits = attn::attn_splits(kGrid, attn_cfg::kM);
        constexpr uint32_t kBoxRows = attn::kGlobalTcN;
        const uint32_t seq = (blockIdx.x / kSplits) / attn_cfg::kNumKvHeads;
        const uint32_t len = static_cast<uint32_t>(__ldg(a.a.attn_kv_len + seq));
        return (len + kSplits - 1) / kSplits >= kBoxRows;
    }
    // every attention warp: the epoch, the tcgen05 entry (the attention's barriers, the whole-pool TMEM allocation
    // into the carve's first word: the pool must be free, which is why the post-attention allocation follows the
    // attention), the grid wait
    static __device__ __forceinline__ void enter(const Args& a, uint8_t* smem, const uint32_t warp, const uint32_t lane,
                                                 const bool tc) {
        uint8_t* const region = smem + kRegionOff;
        if (warp == kPostAttnEpi0 && lane == 0) *epoch(smem) = ++a.a.sm_epoch[blockIdx.x];
        if (tc) {
            if (warp == kPostAttnMma) {
                ptx::tcgen05_alloc(ptx::to_shared(region), 512);
            } else if (warp == kPostAttnGate && ptx::elect_one_sync()) {
                attn::attn_init_bars(reinterpret_cast<uint64_t*>(region + 16));
                ptx::fence_mbarrier_init_release_cluster();
                ptx::prefetch_tensormap(&a.tk);
                ptx::prefetch_tensormap(&a.tv);
                ptx::prefetch_tensormap(&a.tkb);
                ptx::prefetch_tensormap(&a.tvb);
            }
        }
        ptx::sync_aligned(kAttnPhaseBar, kAttnThreads);
        ptx::griddepcontrol_wait();
    }
    static __device__ __forceinline__ void phase(const Args& a, uint8_t* smem, const uint32_t warp, const uint32_t lane,
                                                 const bool tc) {
        const uint32_t ep = static_cast<uint32_t>(*epoch(smem));
        // the leader bumps the epoch ahead of the attention barrier, so neither path's combine waits for it
        if (tc) {
            // the tcgen05 path in the ring from slot 1 (attn_head_phase)
            attn::attn_head_phase<true, false, postattn::kSlotBytes, kEpiBase>(
                a.a, smem, blockIdx.x, attn_warp(warp), lane, ep, &a.tk, &a.tv, &a.tkb, &a.tvb, nullptr);
        } else if (warp >= kPostAttnEpi0) {
            // the mma.sync attention loop in the slots' unused activation regions, as the sliding-window attention
            attn::attn_head_phase<false, false, postattn::kSlotBytes, kEpiBase>(
                a.a, smem, blockIdx.x, warp, lane, ep, nullptr, nullptr, nullptr, nullptr, smem + postattn::kSlotXOff);
            ptx::cp_async_wait_group<0>();
        }
    }
    // the x gate: the attention done, then every combine of this launch published
    static __device__ __forceinline__ void wait(const Args& a, uint8_t* smem) {
        wait_attn(smem);
        const uint32_t target = static_cast<uint32_t>(*epoch(smem)) * attn::attn_combines(kGrid, attn_cfg::kM);
        handoff::ticket_wait(a.a.fctr + kCtrAttnOut, target);
    }

    // the attention ahead of the post-attention body (layer.cu PostAttnPhase::run): the W producer, gate and MMA
    // warps feed and issue, the epilogue warps are the softmax band, the aux warp prefetches the code
    static __device__ __forceinline__ void attend(const Args& a, uint8_t* smem, const uint32_t warp,
                                                  const uint32_t lane) {
        if (warp == kPostAttnAux) {
            if (code_prefetch_cta()) code_prefetch_issue();
        } else {
            const bool tc = use_tcgen05(a);
            enter(a, smem, warp, lane, tc);
            phase(a, smem, warp, lane, tc);
            ptx::fence_async_smem();
            ptx::sync_aligned(kAttnPhaseBar, kAttnThreads);
            if (warp == kPostAttnEpi0 && tc) {
                ptx::tcgen05_dealloc(0, 512);
                uint64_t* const bars = reinterpret_cast<uint64_t*>(smem + kRegionOff + 16);
                if (lane < attn::kNumAttnBars) ptx::mbar_inval(bars + lane);
                ptx::tcgen05_fence_before_thread_sync();
            }
        }
    }
    // inside the post-attention body: the post-attention accumulators on the freed pool, then the done barrier its
    // waiters read
    static __device__ __forceinline__ void run(const Args&, uint8_t* smem, const uint32_t warp, const uint32_t lane) {
        if (warp == kPostAttnEpi0) {
            ptx::tcgen05_fence_after_thread_sync();
            ptx::tcgen05_alloc(ptx::to_shared(smem + postattn::kTmemOff), postattn::kTmemCols);
            ptx::tcgen05_fence_before_thread_sync();
            __syncwarp();
            if (lane == 0) ptx::mbar_arrive(done(smem));
        }
        ptx::sync_aligned(block::kItem, handoff::kItemThreads);  // the post-attention TMEM address is published
    }
};

}  // namespace mk
