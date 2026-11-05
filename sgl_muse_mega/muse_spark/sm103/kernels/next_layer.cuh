#pragma once
// The next-layer phase of a layer image: the cross-layer norm item, the combine item, and the phase body (combine
// into y, the unpooler GEMM, the cross-layer norm, the next layer's qkv GEMM with the rowops epilogue, the next
// layer's KV prefetch).

#include <cuda_bf16.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "framework.cuh"
#include "ptx.cuh"
#include "moe.cuh"

// The cross-layer NORM ITEM of the next-layer phase: the residual add + post-FFN norm + the NEXT layer's
// pre-attention norm (framework.cuh, the multi-virtual-warp row on 8 warps) on warps 4..11 of the launch's last
// kOwnerCtas CTAs, which own no unpooler-GEMM block (norm_first), after the unpooler GEMM's tiles landed. It writes
// the new residual stream and h3 (the next layer's qkv input), then publishes h3_ready, the ticketed count the qkv
// GEMM's x producer waits on.

namespace mk {

// The window's rows (16, or 8). Its template arguments are part of the layer kernel's mangled name; the one variant
// is kSplit = 1 with 256-bit operand loads and stores and the branch polled per packet (the unpooler's all-gathered
// row, framework.cuh).
template <uint32_t kHidden, uint32_t kSplit, bool kV8 = true, bool kV8St = true, bool kPollBranch = false>
struct NormItem {
    static_assert(kSplit == 1 && kV8 && kV8St && kPollBranch, "the one row variant");
    struct Args {
        const nv_bfloat16* branch;  // the unpooler's gathered row [rows][kHidden] bf16 (the phase sets it per launch)
        const float* stream_in;     // the residual in [rows][kHidden] fp32
        const float* alpha;
        const float* beta;
        const nv_bfloat16* pre_w;
        const nv_bfloat16* post_w;  // the input-scale post weight (enters both sums of squares)
        float* stream_out;          // the residual out [rows][kHidden] fp32
        nv_bfloat16* h3;            // [rows][kHidden] bf16 — the qkv x
        uint32_t* h3_ready;         // the ticketed count: +kArrivals per launch
        uint32_t* launch_ctr;       // unused; kept for the schedule (README.md)
    };
    static constexpr uint32_t kRows = cfg::kWindowM;        // the window's tokens (postattn::kM)
    static constexpr uint32_t kOwnerCtas = kRows * kSplit;  // CTA c hosts row c / kSplit, split c % kSplit
    static constexpr uint32_t kArrivals = kOwnerCtas;       // publishes per launch
    // kArrivals divides 2^32, so no launch's arrivals straddle the u32 wrap and the target N x (ticket + 1) wraps in
    // step with the word
    static_assert((uint64_t(1) << 32) % kArrivals == 0, "the u32 count would wrap mid-launch");
    static constexpr uint32_t kStaging = 33;  // the body's staging floats
    static __device__ __forceinline__ const uint32_t* ready(const Args& a) { return a.h3_ready; }
    // the model's post-FFN / pre-attention norm eps
    static constexpr float kPostEps = 1e-8f, kPreEps = 1e-5f;

    // The per-layer operands load AHEAD of the tiles' count wait and the row then loads only the branch behind it.
    using Operands = rowops::RowNormOperands<(32 + handoff::kItemWarps - 1) / handoff::kItemWarps>;
    static __device__ __forceinline__ void preload(const Args& a, const uint32_t cta, const uint32_t item_warp,
                                                   const uint32_t lane, Operands& ops) {
        const size_t off = size_t(cta / kSplit) * kHidden;
        rowops::row_norm_preload_operands<kHidden, handoff::kItemWarps>(item_warp, lane, a.stream_in + off, a.alpha,
                                                                        a.beta, a.pre_w, a.post_w, ops);
    }
    // item_warp = the item-local warp 0..7; `staging` = kStaging floats of smem private to the item's warps; dry = the
    // row's instruction-cache warm pass (no poll, no stores, no publish)
    static __device__ __forceinline__ void run_hoisted(const Args& a, const uint32_t cta, const uint32_t item_warp,
                                                       const uint32_t lane, float* staging, const Operands& ops,
                                                       const bool dry) {
        const size_t off = size_t(cta / kSplit) * kHidden;
        // the body takes its scalars by reference: locals, not the constexpr members
        const float post_eps = kPostEps, pre_eps = kPreEps;
        const uint32_t bar_idx = handoff::kItemBarId;
        rowops::row_norm_residual<kHidden, handoff::kItemWarps>(item_warp, lane, a.branch + off, post_eps, pre_eps,
                                                                a.stream_out + off, a.h3 + off, staging, bar_idx, &ops,
                                                                dry);
        handoff::item_publish(a.h3_ready, !dry && item_warp == 0 && lane == 0, 1u);
    }

    // The idle warps' L2 prefetch of the row's operands: this CTA's residual row and a 1 / kOwnerCtas slice of the four
    // shared vectors (alpha, beta, pre_w, post_w; cold every layer), behind the caller's grid wait (the residual is a
    // predecessor's output). Prefetches only.
    static __device__ __forceinline__ void prefetch_operands_l2(const Args& a, const uint32_t cta, const uint32_t warp,
                                                                const uint32_t lane, const uint32_t warps) {
        const uint32_t t = warp * 32 + lane, stride = warps * 32 * 128;
        const char* const row = reinterpret_cast<const char*>(a.stream_in + size_t(cta / kSplit) * kHidden);
        for (uint32_t off = t * 128; off < kHidden * 4; off += stride)
            ptx::prefetch_global_l2(row + off);
        const char* const vec[4] = {reinterpret_cast<const char*>(a.alpha), reinterpret_cast<const char*>(a.beta),
                                    reinterpret_cast<const char*>(a.pre_w), reinterpret_cast<const char*>(a.post_w)};
        constexpr uint32_t kVecBytes[4] = {kHidden * 4, kHidden * 4, kHidden * 2, kHidden * 2};
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t slice = kVecBytes[r] / kOwnerCtas;
            for (uint32_t off = t * 128; off < slice; off += stride)
                ptx::prefetch_global_l2(vec[r] + size_t(cta) * slice + off);
        }
    }
};

// The MoE combine as an ITEM of the next-layer phase (warps 4..11 of the owner CTA pairs, ahead of the unpooler
// GEMM): per token, the weighted sum of its top-k expert outputs after each slot's RMSNorm (moe.cuh), computed
// redundantly on every rank from the gathered expert outputs, so every rank holds bitwise the same y. One CTA pair per
// token: 512 threads, each CTA's 16 lane groups over half of the hidden's n-blocks, one 8-column chunk per lane, the
// pair's partials exchanged through DSMEM before the final reduction. The item polls its own zb elements until every
// rank's expert outputs of this layer have landed (moe.cuh combine_wait), writes y (the unpooler GEMM's input) and
// publishes y_ready (the next MoE phase's entry restores the elements); the unpooler GEMM's x producer acquires it,
// and block 0's x producer then closes the MoE phase's epoch.

// The item runs HOSTED inside the layer launch: the row's partials come from the caller's scratch, the peer's CLUSTER
// rank is an operand (in the width-4 cluster pair_rank ^ 1 is not the peer) and the row's token comes from the item's
// claimed block id, not from blockIdx.

struct CombineItem {
    struct Args {
        const int32_t* tok_slot;  // [kMaxSlots] the MoE phase's inverse route table
        const float* topk_w;      // [B][kTopK]
        const __nv_bfloat16* zb;  // [kZbRing][zb_par_slots][kH]: the ring base of this rank's replica
        uint32_t zb_par_slots;
        const uint32_t* epoch_dev;      // the MoE phase's epoch counter
        const __nv_bfloat16* gamma_p1;  // [kH]
        __nv_bfloat16* y;               // [B][kH] bf16 — the unpooler GEMM's x
        uint32_t* y_ready;              // the ticketed count: +kArrivals per launch
        uint32_t* gsync;  // the MoE phase's sync words (+ 3 the final-claim count the close waits for, + 4 the cursor)
        int B, b_live;
        float eps;
    };
    static constexpr uint32_t kRows = cfg::kWindowM;       // the window's tokens (postattn::kM)
    static constexpr uint32_t kPair = moe_tp::kCombPair;   // CTAs per token
    static constexpr uint32_t kOwnerCtas = kRows * kPair;  // CTA pair p hosts token p
    // publishes per launch: the owners' count rounded up to a power of two, the first kExtra owners publishing two
    static constexpr uint32_t kArrivals = handoff::pow2_at_least(kOwnerCtas);
    static constexpr uint32_t kExtra = kArrivals - kOwnerCtas;
    static_assert(kExtra < kOwnerCtas && (uint64_t(1) << 32) % kArrivals == 0, "the u32 count would wrap mid-launch");
    static constexpr int kGroups = int(handoff::kItemThreads) / 16;  // 16 lane groups per CTA
    // the row's shared scratch (moe.cuh s_red + s_sc) in floats, carved by the caller
    static constexpr uint32_t kRedFloats = uint32_t(moe_tp::kCombNb * moe_tp::kTopK + moe_tp::kTopK);
    static_assert(kGroups * 2 == moe_tp::kCombGroups, "the pair runs the combine's lane groups per token");
    static __device__ __forceinline__ const uint32_t* ready(const Args& a) { return a.y_ready; }
    // the counter holds the last CLOSED epoch
    static __device__ __forceinline__ uint32_t epoch(const Args& a) {
        return *reinterpret_cast<const volatile uint32_t*>(a.epoch_dev) + 1u;
    }

    // the row's arguments (one token per cluster pair)
    static __device__ __forceinline__ moe_tp::CombArgs row_args(const Args& a, const uint32_t e,
                                                                const uint32_t pair_rank, uint64_t* xchg_bar,
                                                                const uint32_t pair_peer, const uint32_t cta) {
        moe_tp::CombArgs c;
        c.tok_slot = a.tok_slot;
        c.topk_w = a.topk_w;
        c.zb = a.zb + size_t(e % moe_tp::kZbRing) * size_t(a.zb_par_slots) * moe_tp::kH;
        c.gamma_p1 = a.gamma_p1;
        c.y = a.y;
        c.eps = a.eps;
        c.B = a.B;
        c.b_live = a.b_live;
        c.pair_rank = pair_rank;
        c.xchg_bar = xchg_bar;
        c.pair_peer = pair_peer;
        c.tok0 = int32_t(cta / kPair);
        return c;
    }
    // Warps 4..11 of the owner CTAs. pair_rank = %cluster_ctarank; xchg_bar = the exchange mbarrier (one phase per
    // token); pair_peer = the peer's cluster rank; tp_rank = this rank (the wait's timeout report); ext_red = the
    // caller's scratch. The row runs twice in one rolled loop (one copy of its text): first DRY (its loads and
    // arithmetic on this CTA's scratch, no exchange, no y store, no publish) between the caller's
    // cluster-barrier arrive and wait, to warm the instruction cache, then for real behind the wait and the row's
    // polls.
    static __device__ __forceinline__ void run(const Args& a, const uint32_t cta, const uint32_t item_warp,
                                               const uint32_t lane, const uint32_t pair_rank, uint64_t* xchg_bar,
                                               const uint32_t pair_peer, const uint32_t tp_rank, float* ext_red) {
        const uint32_t e = epoch(a);
        // one token per CTA pair (its exchange barrier has one phase per token): the host binds B <= kRows
        if (a.B > int(kRows)) __trap();
        {
            // the row's slot ids and weights ahead of the wait: the tables are complete before the item runs
            const moe_tp::CombArgs c = row_args(a, e, pair_rank, xchg_bar, pair_peer, cta);
            const moe_tp::CombPre pre = moe_tp::combine_prep(c, int(cta) / int(kPair));
            uint4 zv[moe_tp::kTopK] = {};  // the row's chunks (a dead slot's stay zero)
#pragma unroll 1
            for (int pass = 1; pass >= 0; --pass) {
                const bool dry = pass > 0;
                if (!dry) {
                    ptx::cluster_barrier_wait();
                    moe_tp::combine_wait(c, &pre, zv, int(tp_rank));
                    ptx::sync_aligned(handoff::kItemBarId, handoff::kItemThreads);
                } else {
                    moe_tp::combine_load(c, &pre, zv);
                }
                moe_tp::combine_rows<kGroups>(c, handoff::kItemBarId, &pre, ext_red, dry, zv);
            }
        }
        handoff::item_publish(a.y_ready, item_warp == 0 && lane == 0, kExtra != 0 && cta < kExtra ? 2u : 1u);
    }

    // The MoE phase's epoch close (the g2 cursor reset and the epoch store), by block 0's x producer after its y_ready
    // acquire (every pair read the epoch before it published) and behind the unpooler GEMM's stages. Every CTA's final
    // g2 claim precedes its count (moe.cuh g2_producer), so the cursor reset follows every claim; the count then starts
    // the next layer's MoE phase at zero.
    static __device__ __forceinline__ void close(const Args& a) {
        const uint32_t e = epoch(a);
        ptx::count_spin(a.gsync + 3, gridDim.x);
        ptx::st_relaxed_gpu_u32(a.gsync + 3, 0u);
        ptx::st_relaxed_gpu_u32(a.gsync + 4, 0u);
        ptx::st_relaxed_gpu_u32(const_cast<uint32_t*>(a.epoch_dev), e);  // its one writer; Args is the readers'
    }
};

}  // namespace mk

// The next-layer phase: two dense GEMMs with the cross-layer norm item between them, run as the layer launch's last
// phase. The combine item turns the MoE phase's expert outputs into y; the unpooler GEMM (K split over the width-4
// cluster) writes the FFN output, gathered across the rank's 4-rank group; the norm item on the owner CTAs adds the
// residual and normalizes it into h3; the qkv GEMM (the next layer's qkv projection, K split over the cluster) runs
// the rowops epilogue on it. One W/x ring and one accumulator serve both GEMMs, so the qkv GEMM's W stream starts the
// moment a CTA's unpooler-GEMM block drains and the norm item's latency sits under the qkv W prefill.
// Words: ffn_ready gains blocks1 * kFfnArrival per launch (each of a block's four K-quarter ranks adds one); the
// items' ready words gain their kArrivals; the per-CTA launch ticket is read once at entry and bumped once at exit
// by the x producer (launches of this body never overlap: each waits on its predecessor's completion).

// The unpooler is sharded over the 4-rank gather group (the TP ranks are node-major): the rank's CTAs own its
// blocks1 = n1 / 64 / 4 column blocks, and every finished bf16 tile row goes by system-scope stores into each group
// rank's symmetric row (ag_bases[r] + ag_off, empty-marked), which the norm item polls per packet
// (framework.cuh kPollBranch). A group never waits on another node's landing.

// The tail image (kImage.tail, the model's last layer): the combine item, the unpooler GEMM and the norm item (the
// model's final norm: its h3 is the decoder's output) and nothing past them. The qkv-GEMM operands stay bound but are
// never streamed: the producers and the MMA see zero qkv-GEMM stages and the epilogue returns behind the norm item.

// The unpooler GEMM's K split over the width-4 cluster (the world's unpool_ksplit, TP8): the physical cluster owns
// unpooler-GEMM block ticket_block / 4 and rank r its K quarter, so the rank's blocks stream on four times the CTAs
// with a quarter of the x bytes each. Rank r keeps the block's token quarter r: the three peers' partials of it land
// by st.async in a staging area past the carve, and it sums the four in rank order before storing and gathering those
// rows.
// The staging area: the exchange barrier (16 B) + three [64 rows][m / 4 tokens] fp32 partials. Without the split
// (TP4) each claimed block below blocks1 streams one unpooler-GEMM block at full K.
__host__ __device__ constexpr uint32_t unpool_stage_bytes(uint32_t m) {
    return mk::cfg::kWorld.unpool_ksplit ? 16u + 3u * 64u * (m / 4u) * 4u : 0u;
}

// The qkv GEMM, the next layer's qkv row (2 (q + kv heads) x 64 columns), is split over the grid's CTAs: one W2 block
// per cluster at TP8 (36 blocks) and per CTA pair at TP4 (72). The cluster split: CTA q streams K quarter q, ships
// each peer its kM / 4 tokens' fp32 partials (st.async, tx on the peer's cmb_bar), reduces its own kM / 4 tokens over
// the four quarters in the fixed order ((p0 + p1) + p2) + p3 and runs the rowops epilogue on them. The pair split: each
// CTA streams its K half, ships the peer the partial of the peer's token half, adds the two halves' partials (rank 0's
// first) and runs the rowops on its kM / 2 tokens. The combine item keeps the pair claims.

namespace mk {

inline constexpr uint32_t kNextQkvBlocks = 2 * (cfg::kQHeads + cfg::kKvHeads);
inline constexpr uint32_t kNextQkvCk = cfg::kGrid / kNextQkvBlocks;
static_assert((kNextQkvCk == 4 || kNextQkvCk == 2) && kNextQkvCk * kNextQkvBlocks == cfg::kGrid,
              "the qkv GEMM: one W2 block per cluster or per CTA pair");

// The role map is the dense executor's (framework.cuh) and the item warps are its item span; every warp duty
// below reads these names.
inline constexpr block::RoleMap kNextLayerRoles = kDenseRoles;
static_assert(block::valid(kNextLayerRoles), "the next-layer phase's role map has the framework's shape");
inline constexpr block::WarpSpan kNextLayerItemSpan = block::item_span(kNextLayerRoles);
static_assert(kNextLayerItemSpan.warp0 == 4 && block::fits(kNextLayerItemSpan, block::kItemWarps), "item warps 4..11");
inline constexpr uint32_t kNextLayerWProducer = kNextLayerRoles.producer, kNextLayerGate = kNextLayerRoles.gate;
inline constexpr uint32_t kNextLayerMma = kNextLayerRoles.mma, kNextLayerTmem = kNextLayerRoles.aux,
                          kNextLayerEpi0 = kNextLayerRoles.epi0;
inline constexpr uint32_t kNextLayerEpiEnd = kNextLayerRoles.epi0 + kNextLayerRoles.n_epi,
                          kNextLayerPrefetch = kNextLayerEpiEnd - 1;
// the idle item warps (the item warps past the epilogue quad): the norm item's operand prefetch and the next layer's
// KV prefetch
inline constexpr uint32_t kNextLayerIdleWarps = block::kItemWarps - kNextLayerRoles.n_epi;

// The accumulator drain: one x16 tcgen05.ld per 16 columns (one x8 at m 8).
template <uint32_t kM>
static __device__ __forceinline__ void next_layer_drain(const uint32_t taddr, uint32_t (&v)[kM]) {
    ptx::tcgen05_ld_32x32b_cols<kM>(taddr, v);
    ptx::tcgen05_wait_ld();
}

// A K-split partial's n consecutive token columns v[c0, c0 + n) to a peer's staging (st.async, tx on its barrier): 16 B
// packets of four tokens, or one 8 B packet for a token pair (the m8 window's quarter).
template <uint32_t kN, uint32_t kM>
static __device__ __forceinline__ void next_layer_ship_tokens(const uint32_t dst, const uint32_t (&v)[kM],
                                                              const uint32_t c0, const uint32_t mbar) {
    static_assert(kN % 4 == 0 || kN == 2, "packets of four tokens, or one pair");
    if constexpr (kN == 2) {
        ptx::st_async_cluster_v2(dst, v[c0], v[c0 + 1], mbar);
    } else {
#pragma unroll
        for (uint32_t j = 0; j < kN; j += 4)
            ptx::st_async_cluster_v4(dst + j * 4, make_uint4(v[c0 + j], v[c0 + j + 1], v[c0 + j + 2], v[c0 + j + 3]),
                                     mbar);
    }
}

// Runtime arguments of the next-layer phase (m, pack, ring and ck are template constants).
struct NextLayerArgs {
    uint32_t stages1;          // K1 / 64: unpooler-GEMM W stages (= x k-blocks) per block
    uint32_t blocks1;          // n1 / 64: unpooler-GEMM blocks (CTAs block_id < blocks1 own one)
    uint32_t stages2;          // K2 / 64: qkv-GEMM W stages per FULL row (split over the cluster)
    uint32_t n1;               // the unpooler GEMM's out row width (elements)
    nv_bfloat16* out1;         // unused (the unpooler's rows gather in the AR slot); kept for the schedule (README.md)
    uint32_t* ffn_ready;       // the unpooler-GEMM arrival count (+blocks1 per launch)
    uint32_t* launch_ctr;      // [grid] the per-CTA launch tickets
    nv_bfloat16* out2;         // [kM, n2] bf16: the qkv row
    uint32_t n2;               // the qkv row width (elements)
    const uint64_t* ag_bases;  // [cfg::kRanks] the ranks' symmetric workspaces (device addresses)
    uint32_t ag_rank;          // this rank: its column blocks start at (ag_rank % 4) * blocks1
    uint32_t ag_off;           // bf16 elements from a workspace base to the gathered [kM, n1] row
};

// The carve: the dense ck carve plus the tmem-empty barrier (the accumulator handoff between the GEMMs) and the
// combine's exchange barrier. Under the unpooler-GEMM split the out tile moves under x slot 0, making room for the
// unpooler GEMM's exchange staging past the carve: the x ring is dead under the tile, the unpooler GEMM's from its
// tmem_full to the hand-over, the qkv GEMM's from its tmem_full on (the last stream).
__host__ __device__ constexpr DenseSmemLayout next_layer_smem_layout(uint32_t m, uint32_t pack, uint32_t ring,
                                                                     uint32_t ck) {
    DenseSmemLayout l = dense_smem_layout(m, pack, ring, ck);
    if (cfg::kWorld.unpool_ksplit) {
        const uint32_t out_bytes = l.cmb_off - l.out_off;
        l.out_off = l.x_off;
        l.cmb_off -= out_bytes;
        l.tmem_off -= out_bytes;
        l.bar_off -= out_bytes;
    }
    l.size = (l.bar_off + (2 * (ring / pack) + 1 + (ck > 1 ? 1 : 0) + 1 + 1) * 8 + 1023) / 1024 * 1024;
    return l;
}

// The next-layer phase body, run as the layer launch's last phase on a width-4 cluster grid (layer.cu
// NextLayerPhase). block_id is the pair's claimed block; ticket_block is the physical CTA (its launch ticket, its
// cluster's unpooler-GEMM and qkv-GEMM blocks). The statics (the ticket, the item's staging, the combine row's
// partials) live in the post-attention phase's dead FrontZsOvl union (ext_scratch): the launch's static smem sits at
// the part's cap.
template <uint32_t kM, uint32_t kStages, uint32_t kPack, uint32_t kCk, class kTileEpi, class kItem, class kCombItem>
static __device__ __forceinline__ void next_layer_body(
    const NextLayerArgs& args, const CUtensorMap& tmap_w1, const CUtensorMap& tmap_x1, const CUtensorMap& tmap_w2,
    const CUtensorMap& tmap_x2, const CUtensorMap& tmap_out2, const typename kTileEpi::Args& tile_epi,
    const typename kItem::Args& item, const typename kCombItem::Args& comb, uint8_t* smem_buf, const uint32_t block_id,
    void* ext_scratch, const uint32_t ticket_block) {
    constexpr uint32_t BN = 64;
    constexpr uint32_t kStageBytes = BN * 128;
    constexpr uint32_t kXTile = kM * 128;
    constexpr uint32_t kNumSlots = kStages / kPack;
    constexpr uint32_t kAtoms = 4;
    constexpr uint32_t kTmemCols = dense_tmem_cols(kM);    // the accumulator at the alloc grain
    constexpr uint32_t kCmbStride = dense_cmb_stride(kM);  // the combine stage's fp32 row stride
    constexpr uint32_t kEpiBarId = block::kEpi;
    constexpr uint32_t kEpiThreads = kNextLayerRoles.n_epi * 32;
    constexpr uint32_t kThreads = block::kThreads;
    static_assert(kM == 16 || kM == 8, "the 16-row window, or 8 (the drains in registers)");
    static_assert(kStages % kPack == 0 && kPack <= kStages / 2, "pack must divide the ring and be <= half of it");
    static_assert(kCk == 2,
                  "the claim block is a CTA pair's (the claim unit; both GEMMs split K over the 4 cluster ranks)");
    static_assert(kTileEpi::kCkSafe, "tile epi is not ck-safe (the hook is rank-0-only at ck>1)");
    static_assert((uint64_t(1) << 32) % kItem::kArrivals == 0, "the item's u32 count would wrap mid-launch");
    // the pair variant rides the cluster pairs; the solo variant runs one CTA per token
    static_assert(kCombItem::kPair == kCk && kM == kCombItem::kRows, "the combine item rides this body's clusters");
    static_assert((uint64_t(1) << 32) % kCombItem::kArrivals == 0,
                  "the combine item's u32 count would wrap mid-launch");

    constexpr DenseSmemLayout kSmem = next_layer_smem_layout(kM, kPack, kStages, kCk);
    const uint32_t smem_w = ptx::to_shared(smem_buf);
    const uint32_t smem_x = smem_w + kSmem.x_off;
    auto out_tile = reinterpret_cast<nv_bfloat16*>(smem_buf + kSmem.out_off);
    auto cmb_stage = reinterpret_cast<float*>(smem_buf + kSmem.cmb_off);
    auto tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_buf + kSmem.tmem_off);
    uint64_t* const bar_base = reinterpret_cast<uint64_t*>(smem_buf + kSmem.bar_off);
    uint64_t* const full_bars = bar_base;
    uint64_t* const empty_bars = bar_base + kNumSlots;
    uint64_t* const tmem_full_bar = bar_base + 2 * kNumSlots;
    uint64_t* const cmb_bar = bar_base + 2 * kNumSlots + 1;
    uint64_t* const tmem_empty_bar = bar_base + 2 * kNumSlots + 2;
    // the unpooler-GEMM exchange past the carve (the caller budgets the ring for it): its barrier, then [source
    // slot][64 rows][kM / 4 tokens] fp32, slot = the source rank, less one past ours
    uint64_t* const s1_bar = reinterpret_cast<uint64_t*>(smem_buf + kSmem.size);
    float* const s1_stage = reinterpret_cast<float*>(smem_buf + kSmem.size + 16);
    uint64_t* const comb_xchg_bar = bar_base + 2 * kNumSlots + 3;
    // the next-layer phase's statics in the post-attention phase's union
    struct alignas(16) NextLayerScratch {
        alignas(16) float comb_red[kCombItem::kRedFloats];  // first: the pair's st.async targets need 16-B alignment
        float item_staging[kItem::kStaging];
        uint32_t ticket;
    };
    auto* const sc = static_cast<NextLayerScratch*>(ext_scratch);
    float* const comb_red_p = sc->comb_red;

    const uint32_t warp_idx = ptx::warp_idx_sync();
    const uint32_t lane_idx = ptx::lane_idx();
    // the pair-local rank (the combine's token half and n-blocks) and the peer's CLUSTER rank (every mapa
    // operand): in the width-4 cluster the pair is (ctarank & 1) and its peer ctarank ^ 1
    const uint32_t rank = (ptx::cluster_ctarank() & (kCk - 1u));
    const uint32_t pair_peer = (ptx::cluster_ctarank() ^ 1u);
    const auto umin = [](const uint32_t& a, const uint32_t& b) { return a < b ? a : b; };

    // ---- GEMM geometry ----
    // the unpooler GEMM: one block per physical cluster (ticket_block / 4), its K quarter per rank; or one block per
    // claimed block at full K
    constexpr bool kS1Split = cfg::kWorld.unpool_ksplit;
    constexpr uint32_t kS1Q = kS1Split ? kM / 4 : kM;  // the token rows a CTA keeps
    static_assert(kS1Q % 4 == 0 || kS1Q == 2, "the token quarter in 16 B chunks (one 8 B pair at m 8)");
    const uint32_t s1_rank = kS1Split ? ptx::cluster_ctarank() : 0u;
    const uint32_t s1_block = kS1Split ? ticket_block / 4u : block_id;
    const bool did1 = s1_block < args.blocks1;
    static_assert(cfg::kRanks % 4 == 0, "the gather groups tile the ranks");
    constexpr uint32_t kFfnArrival = 4;
    const uint32_t w1_outer = (args.ag_rank % 4 * args.blocks1 + s1_block) * BN;
    // (the host checks stages1 % 4 == 0)
    const uint32_t k1_0 = kS1Split ? s1_rank * (args.stages1 / 4u) : 0u;
    const uint32_t k1_iters = did1 ? (kS1Split ? args.stages1 / 4u : args.stages1) : 0u;
    // the qkv GEMM: the physical cluster owns W2 block ticket_block / 4, the cluster rank its K quarter; or the claimed
    // pair owns block block_id / 2, the pair rank its K half (the remainder stages to the low ranks)
    constexpr uint32_t kCk2 = kNextQkvCk;
    const uint32_t q2 = kCk2 == 4 ? ptx::cluster_ctarank() : rank;
    const uint32_t w2_outer = (kCk2 == 4 ? ticket_block / 4u : block_id / kCk) * BN;
    const uint32_t k2_base = args.stages2 / kCk2;
    const uint32_t k2_rem = args.stages2 % kCk2;
    const uint32_t k2_0 = q2 * k2_base + umin(q2, k2_rem);
    const uint32_t k2_iters = !kImage.tail ? k2_base + (q2 < k2_rem ? 1u : 0u) : 0u;
    // the norm rows run on CTAs that own no unpooler-GEMM block, so a row never waits behind an unpooler-GEMM drain and
    // no unpooler-GEMM block (nor its cluster's K-quarter reduction) behind a row: under the cluster split the launch's
    // last kOwnerCtas CTAs (the unpooler GEMM sits on the first 4 x blocks1), else the claimed blocks right past the
    // unpooler GEMM's (the next MoE phase exits)
    const uint32_t norm_first = kS1Split ? gridDim.x - kItem::kOwnerCtas : args.blocks1;
    const uint32_t norm_cta = kS1Split ? ticket_block : block_id;
    const bool owner = norm_cta >= norm_first && (kS1Split || norm_cta < norm_first + kItem::kOwnerCtas);
    const uint32_t norm_row = norm_cta - norm_first;

    // ---- init (warp 1) + TMEM alloc (warp 3) + the launch ticket ----
    if (warp_idx == kNextLayerGate && ptx::elect_one_sync()) {
        ptx::prefetch_tensormap(&tmap_w1);
        ptx::prefetch_tensormap(&tmap_x1);
        ptx::prefetch_tensormap(&tmap_w2);
        ptx::prefetch_tensormap(&tmap_x2);
        for (uint32_t i = 0; i < kNumSlots; ++i) {
            ptx::mbar_init(&full_bars[i], 1);
            ptx::mbar_init(&empty_bars[i], 1);
        }
        ptx::mbar_init(tmem_full_bar, 1);
        ptx::mbar_init(cmb_bar, 1);
        ptx::mbar_init(tmem_empty_bar, 1);
        ptx::mbar_init(comb_xchg_bar, 1);
        if constexpr (kS1Split) ptx::mbar_init(s1_bar, 1);
        ptx::fence_mbarrier_init_release_cluster();
        sc->ticket = args.launch_ctr[ticket_block];
        // the norm CTAs own no unpooler-GEMM block and fit the grid
        if (kS1Split ? 4u * args.blocks1 > norm_first : norm_first + kItem::kOwnerCtas > gridDim.x) __trap();
    }
    if (warp_idx == kNextLayerTmem) ptx::tcgen05_alloc(ptx::to_shared(tmem_addr_storage), kTmemCols);
    // the cluster barrier split around the combine item: every thread arrives; the owner pairs' item warps run
    // the combine row (a dry pass first) and wait inside the item; every other thread
    // waits here
    ptx::cluster_barrier_arrive();
    if (warp_idx >= kNextLayerEpi0 && block_id < kCombItem::kOwnerCtas) {
        ptx::griddepcontrol_wait();
        kCombItem::run(comb, block_id, warp_idx - uint32_t(kNextLayerItemSpan.warp0), lane_idx, rank, comb_xchg_bar,
                       pair_peer, args.ag_rank, comb_red_p);
    } else {
        ptx::cluster_barrier_wait();
    }
    const uint32_t taddr = tmem_addr_storage[0];
    const uint32_t ticket = sc->ticket;

    // ---- warp 1: x producer ----
    // The unpooler GEMM's x is the combine item's y and the qkv GEMM's x the norm item's h3, one ticketed wait each;
    // the ring's slot / phase cursor runs through the hand-over.
    if (warp_idx == kNextLayerGate) {
        ptx::griddepcontrol_wait();
        // the qkv GEMM's W stages past the ring depth to L2 (below)
        const auto tail_prefetch = [&] {
#pragma unroll(cfg::kWorld.text_diet ? 1 : 4)
            for (uint32_t q = kStages; q < k2_iters; ++q)
                dense_prefetch_w(&tmap_w2, static_cast<int32_t>((k2_0 + q) * 64), static_cast<int32_t>(w2_outer));
        };
        // the release-free variant: the norm owners' qkv-GEMM tail prefetch issued here, inside the combine wait, so
        // their h3 publish has none of it in flight (the norm owners here hold no unpooler-GEMM block, and a pair split
        // of the qkv GEMM prefetches behind the unpooler GEMM)
        const bool tail_pf_early = cfg::kWorld.relfree && owner && (k1_iters != 0 || kCk2 == 4u);
        if constexpr (cfg::kWorld.relfree) {
            if (tail_pf_early && ptx::elect_one_sync()) tail_prefetch();
            __syncwarp();
        }
        // the unpooler GEMM's x rows are the combine item's y: one ticketed wait
        handoff::ticket_wait(kCombItem::ready(comb), kCombItem::kArrivals * (ticket + 1));
        uint32_t slot = 0, phase = 0;
        for (uint32_t s = 0; s < k1_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k1_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(smem_x + (slot * kPack + d) * kXTile, &tmap_x1,
                                                           static_cast<int32_t>((k1_0 + s + d) * 64), /*y=*/0,
                                                           &full_bars[slot], ptx::kEvictNormal);
                }
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
        // the combine's epoch close, by block 0 behind its y_ready acquire (every pair read the epoch) and
        // its x stages
        if (block_id == 0 && ptx::elect_one_sync()) kCombItem::close(comb);
        __syncwarp();
        // the qkv GEMM's W stages past the ring depth go to L2 behind the last unpooler-GEMM publish (ffn_ready), in
        // this warp's idle span before h3: the ring's qkv-GEMM stages fill at the unpooler GEMM's drain and the W
        // producer then waits on the MMA, which waits on h3 through the norm item, so HBM would otherwise idle for
        // the item's span and the qkv GEMM would stream its tail HBM-bound afterwards. Issued earlier, or by the W
        // producer, the burst lands under the unpooler-GEMM publishes and slows them. Every CTA under the cluster
        // split of the qkv GEMM (each streams a quarter), the unpooler-GEMM CTAs under the pair split.
        if (k1_iters != 0 || kCk2 == 4u) {
            handoff::ticket_wait(args.ffn_ready, args.blocks1 * kFfnArrival * (ticket + 1));
            if (!tail_pf_early && ptx::elect_one_sync()) tail_prefetch();
            __syncwarp();
        }
        // the hand-over: the qkv GEMM's x rows are the norm item's h3; the launch's one PDL trigger (layer.cu)
        handoff::ticket_wait(kItem::ready(item), kItem::kArrivals * (ticket + 1));
        if (ptx::elect_one_sync()) ptx::griddepcontrol_launch_dependents();
        __syncwarp();
        for (uint32_t s = 0; s < k2_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k2_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(smem_x + (slot * kPack + d) * kXTile, &tmap_x2,
                                                           static_cast<int32_t>((k2_0 + s + d) * 64), /*y=*/0,
                                                           &full_bars[slot], ptx::kEvictNormal);
                }
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
        // this launch's ticket consumed: the next launch of this body reads it + 1
        if (ptx::elect_one_sync()) args.launch_ctr[ticket_block] = ticket + 1;
    }

    // ---- warp 0: W producer ---- the unpooler GEMM's K quarter, then the qkv GEMM's, one ring
    if (warp_idx == kNextLayerWProducer) {
        uint32_t slot = 0, phase = 0;
        for (uint32_t s = 0; s < k1_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k1_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(
                        smem_w + (slot * kPack + d) * kStageBytes, &tmap_w1, static_cast<int32_t>((k1_0 + s + d) * 64),
                        static_cast<int32_t>(w1_outer), &full_bars[slot], ptx::kEvictFirst);
                }
                ptx::mbar_arrive_expect_tx(&full_bars[slot], deliveries * (kStageBytes + kXTile));
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
        for (uint32_t s = 0; s < k2_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k2_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(
                        smem_w + (slot * kPack + d) * kStageBytes, &tmap_w2, static_cast<int32_t>((k2_0 + s + d) * 64),
                        static_cast<int32_t>(w2_outer), &full_bars[slot], ptx::kEvictFirst);
                }
                ptx::mbar_arrive_expect_tx(&full_bars[slot], deliveries * (kStageBytes + kXTile));
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
    }

    // ---- warp 2: MMA ---- one accumulator; the qkv GEMM opens after the epilogue's drain of the unpooler GEMM
    // (tmem_empty), which the ring absorbs
    if (warp_idx == kNextLayerMma) {
        constexpr uint32_t idesc = ptx::mma_inst_desc_f16(BN, kM);
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_x);
        uint32_t slot = 0, phase = 0;
        for (uint32_t gemm = 0; gemm < 2; ++gemm) {
            const uint32_t k_iters = gemm == 0 ? k1_iters : k2_iters;
            if (k_iters == 0) continue;
            if (gemm == 1 && did1) ptx::mbar_wait_parity_poll(tmem_empty_bar, 0);
            ptx::tcgen05_fence_after_thread_sync();
            for (uint32_t s = 0; s < k_iters; s += kPack) {
                const uint32_t deliveries = umin(kPack, k_iters - s);
                ptx::mbar_wait_parity_poll(&full_bars[slot], phase);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d) {
                        const uint32_t sl = slot * kPack + d;
#pragma unroll
                        for (uint32_t kk = 0; kk < kAtoms; ++kk) {
                            const uint64_t da = w_desc0 + sl * (kStageBytes / 16) + kk * 2;
                            const uint64_t db = x_desc0 + sl * (kXTile / 16) + kk * 2;
                            const uint32_t scale_c = (s > 0 || d > 0 || kk > 0) ? 1u : 0u;
                            ptx::tcgen05_mma_f16(taddr, da, db, idesc, scale_c);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&empty_bars[slot]);
                    if (s + kPack >= k_iters) ptx::tcgen05_commit_arrive(tmem_full_bar);
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // ---- warps 4..7 ---- the unpooler GEMM's drain, the K-quarter reduction, this rank's [kM / 4, 64] bf16 rows
    // gathered to the group, the ffn_ready arrival; then (with warps 8..11, owner CTAs) the norm item; then the qkv
    // GEMM's drain, its K-quarter reduction, the rowops epilogue and the qkv store.
    if (warp_idx >= kNextLayerEpi0) {
        const uint32_t quad = warp_idx - kNextLayerEpi0;
        // the unpooler-GEMM drain, the count, then (the norm CTAs) the row
        {
            if (warp_idx < kNextLayerEpiEnd && did1) {
                ptx::mbar_wait_parity_poll(tmem_full_bar, 0);
                ptx::tcgen05_fence_after_thread_sync();
                uint32_t v[kM];
                next_layer_drain<kM>(taddr, v);
                // every quadrant drained -> the accumulator goes back to the MMA
                // (the qkv GEMM opens under the tile's store)
                ptx::sync_aligned(kEpiBarId, kEpiThreads);
                if (warp_idx == kNextLayerEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive(tmem_empty_bar);
                // the K-quarter reduction: each row's partial of peer p's token quarter goes to p's
                // staging slot (every peer entered the phase: the init's cluster barrier), then this
                // rank's quarter is summed over the four ranks in rank order into the tile's rows [q
                // kM/4, (q+1) kM/4); at full K the tile is the accumulator's
                if constexpr (!kS1Split) {
                    if (lane_idx < 16) {
                        const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                        for (uint32_t c = 0; c < kM; ++c)
                            out_tile[c * BN + row] = __float2bfloat16_rn(__int_as_float(v[c]));
                    }
                } else {
                    if (lane_idx < 16) {
                        const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                        for (uint32_t p = 0; p < 4; ++p) {
                            if (p == s1_rank) continue;
                            const uint32_t slot = s1_rank - (s1_rank > p ? 1u : 0u);  // our slot at p
                            const uint32_t dst =
                                ptx::mapa_shared_cluster(ptx::to_shared(s1_stage + (slot * 64 + row) * kS1Q), p);
                            const uint32_t bar = ptx::mapa_shared_cluster(ptx::to_shared(s1_bar), p);
                            next_layer_ship_tokens<kS1Q>(dst, v, p * kS1Q, bar);
                        }
                    }
                    if (warp_idx == kNextLayerEpi0 && ptx::elect_one_sync())
                        ptx::mbar_arrive_expect_tx(s1_bar, 3u * 64u * kS1Q * 4u);
                    ptx::mbar_wait_parity_acquire_cluster_poll(s1_bar, 0u);
                    if (lane_idx < 16) {
                        const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                        for (uint32_t q = 0; q < 4; ++q) {
                            if (q != s1_rank) continue;
#pragma unroll
                            for (uint32_t j = 0; j < kS1Q; ++j) {
                                float acc = 0.0f;
#pragma unroll
                                for (uint32_t r = 0; r < 4; ++r)
                                    acc += r == q ? __int_as_float(v[q * kS1Q + j])
                                                  : s1_stage[((r - (r > q ? 1u : 0u)) * 64 + row) * kS1Q + j];
                                out_tile[(q * kS1Q + j) * BN + row] = __float2bfloat16_rn(acc);
                            }
                        }
                    }
                }
                ptx::sync_aligned(kEpiBarId, kEpiThreads);
                // the tile out, row-major: thread t stores row t / 8, 8-column chunk t % 8 (one
                // uint4); this rank's token rows only
                const uint32_t s1_row0 = s1_rank * kS1Q;
#pragma unroll
                for (uint32_t i = 0; i < (kS1Q * 8 + kEpiThreads - 1) / kEpiThreads; ++i) {
                    const uint32_t t = i * kEpiThreads + quad * 32 + lane_idx;
                    if (kS1Q * 8 % kEpiThreads != 0 && t >= kS1Q * 8) break;
                    const uint32_t r = s1_row0 + t / 8, chunk = t % 8;
                    const uint4 val = *reinterpret_cast<const uint4*>(out_tile + r * BN + chunk * 8);
                    const size_t col = args.ag_off + size_t(r) * args.n1 + w1_outer + chunk * 8;
                    // the group's far half first, each source rotated by its rank (in rank order every source reached
                    // the group's last rank last)
                    const uint32_t g0 = args.ag_rank - args.ag_rank % 4;
#pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t q = g0 + (args.ag_rank + 2 + i) % 4;
                        ptx::st_relaxed_sys_u4(
                            reinterpret_cast<uint4*>(reinterpret_cast<nv_bfloat16*>(args.ag_bases[q]) + col), val);
                    }
                }
                ptx::sync_aligned(kEpiBarId, kEpiThreads);
                // one arrival per unpooler-GEMM block (a quarter per rank under the split), after every epi
                // thread's stores: the word gains blocks1 x kFfnArrival per launch
                if (warp_idx == kNextLayerEpi0 && ptx::elect_one_sync()) {
                    if constexpr (cfg::kWorld.relfree)
                        // relaxed: no reader dereferences anything behind the word (the gathered row's packets are
                        // their own readiness)
                        ptx::red_add_b32(args.ffn_ready, kS1Split ? kFfnArrival / 4u : kFfnArrival);
                    else
                        // the release variant: the x producer's prefetch is ordered behind the word
                        ptx::red_add_rel_b32(args.ffn_ready, kS1Split ? kFfnArrival / 4u : kFfnArrival);
                }
            }
            if (owner) {
                // the norm item on warps 4..11 of the owner CTAs, behind the grid wait (the residual
                // stream is a prior grid's output) and after every unpooler-GEMM block landed; it publishes
                // its ready word
                ptx::griddepcontrol_wait();
                // the row is the gathered one: every rank's tiles land in this rank's symmetric row, polled per packet
                typename kItem::Args item_row = item;
                item_row.branch = reinterpret_cast<const nv_bfloat16*>(args.ag_bases[args.ag_rank]) + args.ag_off;
                // the idle warps prefetch the row's per-layer operands into L2 behind the grid wait
                // (cold per layer in a real model, warm in a microbenchmark); the epi warps are on
                // the unpooler GEMM here
                if (warp_idx >= kNextLayerEpiEnd)
                    kItem::prefetch_operands_l2(item, norm_row, warp_idx - kNextLayerEpiEnd, lane_idx,
                                                kNextLayerIdleWarps);
                // the row runs twice in one rolled loop (one copy of its text): first DRY while the row is still
                // landing, to warm the instruction cache, then for real. Each pass issues the hoisted operand round:
                // the row's per-layer operands ahead of the readiness wait, consumed by the row behind it (the
                // residual is upstream data, ordered by the grid wait above)
#pragma unroll 1
                for (int pass = 1; pass >= 0; --pass) {
                    // the lane id through an opaque zero: the row's addresses stay per pass (hoisted out of the
                    // loop they would stay live across both passes and spill)
                    const uint32_t lane_p = ptx::opaque_u32(lane_idx);
                    typename kItem::Operands ops;
                    kItem::preload(item, norm_row, warp_idx - uint32_t(kNextLayerItemSpan.warp0), lane_p, ops);
                    // the gathered row's packets are their own readiness (no count to wait on)
                    ptx::sync_aligned(handoff::kItemBarId, handoff::kItemThreads);
                    kItem::run_hoisted(item_row, norm_row, warp_idx - uint32_t(kNextLayerItemSpan.warp0), lane_p,
                                       sc->item_staging, ops, pass > 0);
                }
            }
        }
        if (warp_idx >= kNextLayerEpiEnd) return;  // the idle warps: no dense role
        if constexpr (kImage.tail) {
            // the tail: no qkv GEMM — the accumulator held the unpooler GEMM alone (drained above)
            if (warp_idx == kNextLayerEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);
            return;
        }
        if constexpr (kCk2 == 4) {
            // ---- the qkv GEMM ---- every CTA ships each peer its kM / 4 tokens' fp32 partials, then reduces its own
            // kM / 4 tokens over the four K quarters in the fixed order ((p0 + p1) + p2) + p3 (staging [row][source
            // quarter][kM / 4]; the receiver's own slot unused) and runs the rowops
            constexpr uint32_t kQ = kM / 4;  // shipped as 16 B packets of four tokens (one 8 B pair at m 8)
            if (warp_idx == kNextLayerEpi0 && ptx::elect_one_sync())
                ptx::mbar_arrive_expect_tx(cmb_bar, 3 * 64 * kQ * 4);
            ptx::mbar_wait_parity_poll(tmem_full_bar, did1 ? 1u : 0u);
            ptx::tcgen05_fence_after_thread_sync();
            uint32_t v[kM];
            next_layer_drain<kM>(taddr, v);
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                for (uint32_t p = 0; p < 4; ++p) {
                    if (p == q2) continue;
                    const uint32_t dst = ptx::mapa_shared_cluster(ptx::to_shared(cmb_stage + (row * 4 + q2) * kQ), p);
                    const uint32_t mbar = ptx::mapa_shared_cluster(ptx::to_shared(cmb_bar), p);
                    next_layer_ship_tokens<kQ>(dst, v, p * kQ, mbar);
                }
            }
            ptx::mbar_wait_parity_acquire_cluster_poll(cmb_bar, 0);
            // (the unpooler-GEMM tile's readers all passed the post-store barrier above; the tile is free)
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
                // a uniform branch per quarter: constant indices keep v in registers
                const auto reduce_part = [&](auto qc) {
                    constexpr uint32_t Q = decltype(qc)::value;
#pragma unroll
                    for (uint32_t j = 0; j < kQ; ++j) {
                        float acc = 0.f;
#pragma unroll
                        for (uint32_t src = 0; src < 4; ++src) {
                            const float part =
                                src == Q ? __int_as_float(v[Q * kQ + j]) : cmb_stage[(row * 4 + src) * kQ + j];
                            acc = src == 0 ? part : acc + part;
                        }
                        out_tile[(Q * kQ + j) * BN + row] = __float2bfloat16_rn(acc);
                    }
                };
                if (q2 == 0u)
                    reduce_part(std::integral_constant<uint32_t, 0>{});
                else if (q2 == 1u)
                    reduce_part(std::integral_constant<uint32_t, 1>{});
                else if (q2 == 2u)
                    reduce_part(std::integral_constant<uint32_t, 2>{});
                else
                    reduce_part(std::integral_constant<uint32_t, 3>{});
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            const uint32_t tok0 = q2 * kQ;
            kTileEpi::template pre_store<kQ>(tile_epi, w2_outer, out_tile, quad, lane_idx, tok0);
            if constexpr (kTileEpi::kMutatesTile) ptx::sync_aligned(kEpiBarId, kEpiThreads);
            {
                const uint32_t t = quad * 32 + lane_idx;
                if (t < kQ * 8) {  // kQ * 8 <= the 128 epi threads at every kM
                    const uint32_t c = tok0 + (t >> 3), j = t & 7;
                    *reinterpret_cast<uint4*>(args.out2 + size_t(c) * args.n2 + w2_outer + j * 8) =
                        *reinterpret_cast<const uint4*>(out_tile + c * BN + j * 8);
                }
            }
        } else {
            // ---- the qkv GEMM on the pair ---- each CTA ships the peer its K half's partial of the peer's token half
            // (64 x kM / 2 fp32), adds the peer's partial of its own half (rank 0's addend first) and runs the rowops
            // on its kM / 2 tokens
            static_assert(kM % 8 == 0, "the peer's half in 16 B groups of four fp32");
            constexpr uint32_t kHalf = kM / 2;
            if (warp_idx == kNextLayerEpi0 && ptx::elect_one_sync())
                ptx::mbar_arrive_expect_tx(cmb_bar, 64 * kHalf * 4);
            ptx::mbar_wait_parity_poll(tmem_full_bar, did1 ? 1u : 0u);
            ptx::tcgen05_fence_after_thread_sync();
            uint32_t v[kM];
            next_layer_drain<kM>(taddr, v);
            // the peer's cluster rank addresses its smem; its token half is the pair-local index
            const uint32_t peer = pair_peer, tok0 = kHalf * rank;
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
                const uint32_t dst = ptx::mapa_shared_cluster(ptx::to_shared(cmb_stage + row * kCmbStride), peer);
                const uint32_t mbar = ptx::mapa_shared_cluster(ptx::to_shared(cmb_bar), peer);
                // the token halves by a uniform branch on the rank: constant indices keep v in registers
                if (rank == 0u) {
#pragma unroll
                    for (uint32_t g = 0; g < kHalf / 4; ++g) {
                        const uint32_t c0 = kHalf + 4 * g;
                        const uint4 pk = {v[c0], v[c0 + 1], v[c0 + 2], v[c0 + 3]};
                        ptx::st_async_cluster_v4(dst + c0 * 4, pk, mbar);
                    }
                } else {
#pragma unroll
                    for (uint32_t g = 0; g < kHalf / 4; ++g) {
                        const uint32_t c0 = 4 * g;
                        const uint4 pk = {v[c0], v[c0 + 1], v[c0 + 2], v[c0 + 3]};
                        ptx::st_async_cluster_v4(dst + c0 * 4, pk, mbar);
                    }
                }
            }
            ptx::mbar_wait_parity_acquire_cluster_poll(cmb_bar, 0);
            // (the unpooler-GEMM tile's readers all passed the post-store barrier above; the tile is free)
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
                if (rank == 0u) {
#pragma unroll
                    for (uint32_t c = 0; c < kHalf; ++c) {
                        float acc = __int_as_float(v[c]);
                        acc += cmb_stage[row * kCmbStride + c];
                        out_tile[c * BN + row] = __float2bfloat16_rn(acc);
                    }
                } else {
#pragma unroll
                    for (uint32_t c = kHalf; c < 2 * kHalf; ++c) {
                        float acc = __int_as_float(v[c]);
                        acc += cmb_stage[row * kCmbStride + c];
                        out_tile[c * BN + row] = __float2bfloat16_rn(acc);
                    }
                }
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            kTileEpi::template pre_store<kHalf>(tile_epi, w2_outer, out_tile, quad, lane_idx, tok0);
            if constexpr (kTileEpi::kMutatesTile) ptx::sync_aligned(kEpiBarId, kEpiThreads);
            // this rank's token rows of the tile to the qkv row (16 B per thread, kHalf * 8 threads)
#pragma unroll
            for (uint32_t i = 0; i < (kHalf * 8 + kEpiThreads - 1) / kEpiThreads; ++i) {
                const uint32_t t = i * kEpiThreads + quad * 32 + lane_idx;
                if (t < kHalf * 8) {
                    const uint32_t c = tok0 + (t >> 3), j = t & 7;
                    *reinterpret_cast<uint4*>(args.out2 + size_t(c) * args.n2 + w2_outer + j * 8) =
                        *reinterpret_cast<const uint4*>(out_tile + c * BN + j * 8);
                }
            }
        }
        if (warp_idx == kNextLayerEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);
        return;
    }
}

}  // namespace mk
