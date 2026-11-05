// One decoder layer = one persistent launch: mk::decoder_layer<4>, built once per image with
// -DMK_IMAGE=swa|global|swa_bf16|global_bf16_tail; the host loads the cubin. In order: the image the
// build selects and the code prefetch (both read by the phases; the code prefetch stays ahead of the phase headers:
// README.md, "Order-dependent placement"), the phases' headers, the phase program, and the kernel's instantiation.

// The layer image a build selects: -DMK_IMAGE=swa|global|swa_bf16|global_bf16_tail names one row of the table below.
// One kernel serves all four; the image's axes pick the attention policy, the MoE variant's weight format and the
// next-layer phase's tail.

namespace mk {

struct LayerImage {
    bool global_attn;  // the global attention type: the global attention (attention.cuh) ahead of the post-attention
                       // phase; else the sliding-window attention (attention.cuh) beside the post-attention prologue
                       // roles
    bool bf16;         // the bf16 MoE phase: kind::f16 MMAs on bf16 W1 / W2 and bf16 token rows, no scale factors
    bool tail;  // the last layer: the next-layer phase runs the combine, the unpooler GEMM and the final norm (no
                // qkv GEMM, no KV rows)
};
namespace image {
inline constexpr LayerImage swa{.global_attn = false, .bf16 = false, .tail = false};            // 45 layers
inline constexpr LayerImage global{.global_attn = true, .bf16 = false, .tail = false};          // 15 layers
inline constexpr LayerImage swa_bf16{.global_attn = false, .bf16 = true, .tail = false};        // layer 0
inline constexpr LayerImage global_bf16_tail{.global_attn = true, .bf16 = true, .tail = true};  // layer 61
}  // namespace image
inline constexpr LayerImage kImage = image::MK_IMAGE;

// the NVFP4 variant's MoE phase gathers h as the row window (moe.cuh: per-row SF recalibration of the h codes)
inline constexpr bool kHRow = !kImage.bf16;

// Code prefetch: the kernel's own TEXT made L2-hot from the DATA side at launch. The layer's text (200-320 KB) does not
// stay in L2 between two launches of its module: the layers in between stream more normal-priority bytes than L2 holds
// (the expert prefetch first of all). An instruction line past the ~64 KB launch prefetch is fetched from HBM at first
// use, and a plain data load of one word per 128-B line of the kernel's code puts the line in L2; so every launch
// prefetches the whole text again, the window's lines too. One warp issues the loads on each of the nine CTAs with
// blockIdx.x % 16 == 0, every one of them loading every line. Evict-last keeps the lines longer than the normal
// priority, not for good. The text range [pc, pc + bytes) comes from g_code_prefetch_native, bound by the host at
// module load; an unbound range prefetches nothing. Loads only: the arithmetic is untouched.

struct CodePrefetchRange {
    unsigned long long base;
    unsigned long long bytes;
};

inline constexpr unsigned kCodePrefetchCtaMask = 15;  // the nine CTAs with blockIdx.x % 16 == 0 issue the prefetch
__device__ CodePrefetchRange g_code_prefetch;         // the range the host binds (zero: nothing to prefetch)
__device__ unsigned g_code_prefetch_sink;

// The kernel's native text range as emitted: {&kernel, 0} with an R_CUDA_DESCRIPTOR relocation on `entry`. Before
// loading the cubin the host retypes that relocation to R_CUDA_64 (the absolute text address) and writes the kernel's
// .text size into `bytes`; it then copies the loaded value into g_code_prefetch.
struct CodePrefetchNative {
    void (*entry)();
    unsigned long long bytes;
};

// ONE converged warp: one 4-B load per 128-B line of the range, eight per lane per iteration (nvcc unrolls the loop
// twice: sixteen in flight), at the L2 evict-last policy (a cache-policy operand; the eviction qualifier on ld would
// need a 256-bit load). The prefetch is bound by the SM's request rate (about 13 cycles a line). The values
// feed a store behind an improbable condition so the loads stay live; the L2 fill is their only effect. Out of line:
// ptxas places the routine behind the kernel body, so it adds no text ahead of the phases that follow (text ahead of
// the MoE phase would push it out of the launch prefetch window).
__device__ __noinline__ void code_prefetch_issue() {
    const CodePrefetchRange r = g_code_prefetch;
    const unsigned n = static_cast<unsigned>((r.bytes + 127ull) >> 7);
    const unsigned lane = threadIdx.x & 31u;
    unsigned acc = 0;
    for (unsigned i = lane; i < n; i += 32u * 8u) {
        unsigned v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned k = i + 32u * j;
            unsigned x = 0;
            if (k < n)
                asm volatile(
                    "{ .reg .b64 pol; createpolicy.fractional.L2::evict_last.b64 pol, 1.0; "
                    "ld.global.cg.L2::cache_hint.u32 %0, [%1], pol; }"
                    : "=r"(x)
                    : "l"(r.base + static_cast<unsigned long long>(k) * 128ull)
                    : "memory");
            v[j] = x;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc ^= v[j];
    }
    if (acc == 0x9E3779B9u) atomicAdd(&g_code_prefetch_sink, 1u);
}

__device__ __forceinline__ bool code_prefetch_cta() {
    return (blockIdx.x & kCodePrefetchCtaMask) == 0u;
}

}  // namespace mk

// The one-launch decoder layer: decoder_layer runs the attention phase, the post-attention phase
// (o_proj, all-reduce + norm, pooler || router, top-k, the route tables), the MoE phase (g1 -> g2) and the next-layer
// phase (combine, unpooler, norm, the next layer's qkv) in ONE persistent cluster launch, written as a compile-time
// phase program.
// Same-launch handoffs are ticketed counts (framework.cuh): the front CTA publishes front_done (+1 per
// launch) after its tables, every nq item publishes nq_done (+postattn::kRouteArrivals per launch) after its row, and
// the MoE phase waits both against its CTA's launch ticket instead of a griddepcontrol.wait. Carve: every phase's
// dynamic layout lies inside the MoE phase's (kLayerSmem); the static FrontZsOvl union is the attention's histogram,
// the MoE phase's z staging and the next-layer phase's scratch in turn. TMEM: the post-attention phase's 64 columns,
// the MoE phase's 512 and the next-layer phase's 32 are allocated and freed in sequence, one owner at a time.

#include "postattn.cuh"
#include "moe.cuh"

// the launch's dynamic carve: the MoE phase's (kMoeSmem), or on the bf16 MoE variant (kImage.bf16) its whole budget;
// every other phase's layout lies inside it
inline constexpr int kLayerSmem = moe_tp::kBf16 ? moe_tp::kSmemBudget : moe_tp::kMoeSmem;
static_assert(moe_tp::kMoeSmem <= kLayerSmem, "the MoE phase's carve fits the launch's");
// the post-attention body's carve inside the MoE phase's: its own layout plus the top-k arrays at its tail
inline constexpr int kPostAttnSmem = int(mk::postattn::kSmemBytes + mk::postattn::kTopkBytes);
static_assert(kPostAttnSmem <= kLayerSmem,
              "the post-attention carve and its top-k arrays (the carve's tail) fit inside the MoE phase's");
static_assert(moe_tp::kMoeThreads == int(mk::block::kThreads), "one twelve-warp block for both bodies");

namespace mk {

// The MoE phase behind the post-attention phase's two ticketed counts (front_done, nq_done). Thread 0's store of its
// epoch is read by nothing (the next-layer phase reads epoch_dev itself), kept for the schedule (README.md).
static __device__ __forceinline__ void moe_body(const moe_tp::LayerArgs& a, const uint32_t ticket,
                                                const uint32_t* front_done, const uint32_t* nq_done,
                                                moe_tp::ZStageBuf* zst, uint32_t* moe_epoch) {
    using namespace moe_tp;
    const uint32_t e = layer_epoch(a);
    if (moe_epoch != nullptr && threadIdx.x == 0) *moe_epoch = e;
    const int par = int(e % kZbRing);
    moe_body(a, a.zb + size_t(par) * size_t(a.zb_par_slots) * kH, e, zst, front_done,
             postattn::kFrontPubs * (ticket + 1), nq_done, postattn::kRouteArrivals * (ticket + 1));
}

// The layer as a PHASE PROGRAM: a compile-time list of phases run in order by one executor (phase_program). Each later
// phase runs behind the boundary the executor emits: the earlier phase's retire (what the next carve needs gone), a CTA
// barrier, then the next phase's entry and body. A phase holds references to its operands and declares whether it
// has a kernel prologue (thread 0, ahead of every body).

// the statics the phases hand to one another: the front scratch union, the launch ticket, the qkv GEMM's coordinates,
// the MoE phase's epoch (unread) and the next-layer phase's claim words
struct LayerShared {
    moe_tp::FrontZsOvl* fovl;
    uint32_t* ticket;
    moe_tp::QkvCoords* pf;
    uint32_t* moe_epoch;
    uint64_t* claim_bar;
    uint32_t* pair_arrive;
    uint32_t* claim;
    uint64_t* entry_rdy;  // unread (nullptr); kept for the schedule (README.md)
};

// the post-attention phase with its attention policy: the global attention (Attention::kGlobalAttn) runs on every
// warp ahead of the post-attention body, then a CTA barrier; the sliding-window attention runs inside the
// post-attention body, on the epilogue warps beside the post-attention prologue roles
template <uint32_t kW, class Attention>
struct PostAttnPhase {
    const PostAttnArgs& args;
    const CUtensorMap &tmap_wo, &tmap_xa, &tmap_wp, &tmap_xh, &tmap_pooled;
    const RouterArgs& ra;
    const MoeFrontItem<true>::Args& fa;
    const CUtensorMap &tmap_wr, &tmap_xf;
    const typename Attention::Args& attention;
    const typename Attention::PrefetchArgs& prefetch;
    static constexpr bool kPrologue = false, kClusterInit = false;
    __device__ __forceinline__ void enter(const LayerShared&) const {}
    __device__ __forceinline__ void run(const LayerShared& sh, const PostAttnArgs&, uint8_t* smem_buf) const {
        if constexpr (Attention::kGlobalAttn) {
            Attention::attend(attention, smem_buf, ptx::warp_idx_sync(), ptx::lane_idx());
            __syncthreads();
        }
        postattn_body<kW, Attention>(args, tmap_wo, tmap_xa, tmap_wp, tmap_xh, tmap_pooled, ra, fa, tmap_wr, tmap_xf,
                                     sh.fovl->front, *sh.ticket, smem_buf, attention, prefetch);
    }
    // the attention's barriers (retired by the body's exit cluster barrier) and the post-attention dynamic mbarrier
    // pool [postattn::kBarOff, + postattn::kBars * 8), still live at the body's return: the MoE phase's g2 carve
    // overlays those bytes, so they are retired before any MoE byte lands there (a TMA transaction onto a live
    // barrier's bytes corrupts it)
    __device__ __forceinline__ void retire(uint8_t* smem_buf) const {
        if (threadIdx.x == 0) Attention::retire(smem_buf);
        if (threadIdx.x < postattn::kBars)
            ptx::mbar_inval(reinterpret_cast<uint64_t*>(smem_buf + postattn::kBarOff) + threadIdx.x);
    }
};

// the MoE phase
struct MoePhase {
    const moe_tp::LayerArgs& la;
    static constexpr bool kPrologue = true, kClusterInit = false;
    // the MoE phase's tensormaps prefetched under the post-attention fill: nothing in the MoE phase prefetches them,
    // so its first TMA otherwise fetches the descriptor at first use
    __device__ __forceinline__ void prologue(const LayerShared&) const {
        for (int i = 0; i < moe_tp::kTmaps; ++i)
            ptx::prefetch_tensormap(&la.tmaps.tm[i]);
        ptx::prefetch_tensormap(&la.tm_sfab);
    }
    __device__ __forceinline__ void enter(const LayerShared&) const {}
    __device__ __forceinline__ void run(const LayerShared& sh, const PostAttnArgs& args, uint8_t*) const {
        moe_body(la, *sh.ticket, args.front_done, args.nq_done, moe_tp::zstage_window(*sh.fovl), sh.moe_epoch);
    }
    __device__ __forceinline__ void retire(uint8_t*) const {}
};

// the executor: every phase's prologue, the first phase, each later phase behind the emitted boundary
template <class Boundary, class Prev>
static __device__ __forceinline__ void phase_program_boundaries(const Boundary&, const Prev&) {}
template <class Boundary, class Prev, class Next, class... Rest>
static __device__ __forceinline__ void phase_program_boundaries(const Boundary& boundary, const Prev& prev,
                                                                const Next& next, const Rest&... rest) {
    boundary(prev, next);
    phase_program_boundaries(boundary, next, rest...);
}
template <class First, class... Later>
static __device__ __forceinline__ void phase_program(const LayerShared& sh, const PostAttnArgs& args, uint8_t* smem_buf,
                                                     const First& first, const Later&... later) {
    __builtin_assume(threadIdx.x < block::kThreads);  // the kernels' launch bounds, for the text inlined from here
    if (threadIdx.x == 0) {
        (
            [&] {
                if constexpr (Later::kPrologue) later.prologue(sh);
            }(),
            ...);
        if constexpr ((Later::kClusterInit || ...))
            ptx::fence_mbarrier_init_release_cluster();  // published to the pair by the post-attention body's entry
                                                         // cluster_sync
    }
    static_assert(!First::kPrologue && !First::kClusterInit, "the first phase needs no kernel prologue");
    first.enter(sh);
    first.run(sh, args, smem_buf);
    const auto boundary = [&](const auto& prev, const auto& next) {
        prev.retire(smem_buf);
        __syncthreads();
        next.enter(sh);
        next.run(sh, args, smem_buf);
    };
    phase_program_boundaries(boundary, first, later...);
}

}  // namespace mk

// The attention policies: the sliding-window attention overlapping the post-attention prologue roles (below) and the
// global attention leading the post-attention phase (attention.cuh).
#include "attention.cuh"

static_assert(mk::cfg::kGrid % mk::cfg::kW == 0, "the launch grid is whole clusters");

namespace mk {

struct SwaAttention : AttnWords {
    static constexpr bool kGlobalAttn = false;
    using Args = AttnArgs;
    static_assert(attn_cfg::kM >= postattn::kM, "the attention computes every row the post-attention phase reads");
    static_assert(16 * 272 <= postattn::kSlotStages * postattn::kXTile,
                  "one sliding-window KV block fits each unused activation slot");

    static __device__ __forceinline__ void init(uint8_t* smem) {
        ptx::mbar_init(done(smem), 1);
        *epoch(smem) = 0;  // the combine and the x gate read the word: zero until the gate warp's bump lands
    }

    // the attention's epoch: the gate warp's bump behind its grid wait, off the attention's path (the word's last
    // writer is the previous launch)
    static __device__ __forceinline__ void bump_epoch(const Args& a, uint8_t* smem, const uint32_t lane) {
        if (lane == 0) *epoch(smem) = ++a.sm_epoch[blockIdx.x];
        __syncwarp();
    }

    // the sliding-window attention on the epilogue warps: the head phase over the ring slots' x regions, then the done
    // arrive
    static __device__ __forceinline__ void run(const Args& a, uint8_t* smem, const uint32_t warp, const uint32_t lane) {
        const bool leader = warp == kPostAttnEpi0 && lane == 0;
        // the attention's KV ring blocks occupy the ring slots' unused x regions (block b in slot b % kSlots); the W
        // producer touches only W
        uint8_t* const kv = smem + postattn::kSlotXOff;
        ptx::sync_aligned(block::kItem, handoff::kItemThreads);
        // the combine reads the shared word itself: the gate warp's bump lands under the attention (bump_epoch)
        const uint32_t& ep = *reinterpret_cast<const uint32_t*>(epoch(smem));
        // the mma.sync attention loop on the epilogue warps, its ring in the slots' x regions; the combine waits for
        // the gate warp's epoch bump
        attn::attn_head_phase<false, true, postattn::kSlotBytes, kPostAttnEpi0 * 32>(
            a, smem, blockIdx.x, warp, lane, ep, nullptr, nullptr, nullptr, nullptr, kv);
        ptx::cp_async_wait_group<0>();
        ptx::sync_aligned(block::kItem, handoff::kItemThreads);
        if (leader) ptx::mbar_arrive(done(smem));
    }

    // The local gate drains cp.async; the global gate publishes every X row before the X TMA.
    static __device__ __forceinline__ void wait(const Args& a, uint8_t* smem) {
        ptx::mbar_wait_parity_poll(done(smem), 0);
        const uint32_t target = static_cast<uint32_t>(*epoch(smem)) * attn::attn_combines(kGrid, attn_cfg::kM);
        handoff::ticket_wait(a.fctr + kCtrAttnOut, target);
    }
};
using AttnPolicy = std::conditional_t<kImage.global_attn, GlobalAttention, SwaAttention>;

}  // namespace mk

// The next-layer phase (next_layer.cuh: the combine item, the unpooler GEMM, the cross-layer norm item, the next
// layer's qkv GEMM with the rowops epilogue) runs on every CTA the moment its MoE phase exits, with no grid-wide join;
// its own readiness words order it (the combine item's zb elements, y_ready, ffn_ready, h3_ready). The body keeps a
// standalone kernel's shape: its griddepcontrol.waits (satisfied at once), its ring and its TMEM (32 columns,
// allocated after the MoE phase frees its 512). The K-split pair is the width-4 cluster's (ctarank & 1, peer
// ctarank ^ 1); the phase's statics ride the FrontZsOvl union, dead after g1. Its mbarrier pool sits on bytes the MoE
// phase's g2 ring covered, every TMA into them consumed before the MoE phase exits (no live barrier under a TMA
// landing zone, no TMA in flight under a barrier).

#include "next_layer.cuh"

#include <type_traits>

namespace mk {

// the next-layer phase's tile epilogue: the qkv rowops at the post-attention window
using NextLayerTileEpi = TileEpiQkvRowops<cfg::kQHeads, cfg::kKvHeads, int(postattn::kM)>;
// the norm item at the post-attention rows (its name is part of the kernel's mangled name)
using NextLayerNormItem = NormItem<cfg::kHidden, 1, true, true, true>;
inline constexpr uint32_t kNextLayerCk = 2;
// the next-layer phase's ring: the post-attention pack at the deepest ring that fits the MoE phase's carve
__host__ __device__ constexpr uint32_t next_layer_ring_fit(uint32_t pack, int budget) {
    uint32_t stages = 2 * pack;
    while (static_cast<int>(next_layer_smem_layout(postattn::kM, pack, stages + pack, kNextLayerCk).size) <= budget)
        stages += pack;
    return stages;
}
inline constexpr uint32_t kNextLayerPack = cfg::kPack;
// under the unpooler GEMM's K split the exchange's staging past the carve comes out of the ring's budget and the
// next-layer phase takes the deepest ring at its pack that fits beside it; at full K, the post-attention ring's depth
inline constexpr uint32_t kNextLayerStages =
    cfg::kWorld.unpool_ksplit ? next_layer_ring_fit(kNextLayerPack, kLayerSmem - int(unpool_stage_bytes(postattn::kM)))
                              : uint32_t(cfg::kStages);
inline constexpr int kNextLayerSmem =
    static_cast<int>(next_layer_smem_layout(postattn::kM, kNextLayerPack, kNextLayerStages, kNextLayerCk).size);
static_assert(kNextLayerSmem + int(unpool_stage_bytes(postattn::kM)) <= kLayerSmem,
              "the next-layer phase's carve (and the unpooler GEMM exchange's staging) fits inside the MoE phase's");
static_assert(cfg::kStages == int(postattn::kSlots * postattn::kSlotStages) && cfg::kPack == int(postattn::kSlotStages),
              "the post-attention ring is the pins' (the next-layer phase shares its pack)");
static_assert(kNextLayerStages % kNextLayerPack == 0 && kNextLayerPack <= kNextLayerStages / 2,
              "the next-layer phase's ring packs");
// the next-layer phase's statics inside the FrontZsOvl union (the ticket, the norm item's staging, the combine row's
// partials)
inline constexpr uint32_t kNextLayerScratchBytes =
    4 + 4 * NextLayerNormItem::kStaging + 4 * CombineItem::kRedFloats + 16;
static_assert(kNextLayerScratchBytes <= sizeof(moe_tp::FrontZsOvl),
              "the next-layer phase's statics fit the dead union");

// The next-layer KV prefetch: the idle warps (back from the next-layer body at its start on non-owner CTAs, after the
// norm item on the owners) pull the successor sliding-window attention's KV rows into L2 in the launch's last stretch,
// after the MoE phase's
// weight stream has turned L2 over. A persistent successor lands as this launch's CTAs exit, with no exit spread to
// hide its first ring blocks under, so the rows it reads first are made L2-hot here. Only older positions are read
// (the newest sixteen are this launch's own epilogue writes); the loads feed a store behind an improbable condition,
// so the L2 fill is their only effect.
// The operands select the mode on one image: the successor's pool (the prefetch), splits = 0 (an immediate return), or
// another layer's pool (the same loads and timing, nothing the successor reads).
// kNextKvUnits: the successor attention's (seq, kv head) units (attention.cuh attn_units: one sequence, whichever
// row count the successor's attention computes).
inline constexpr uint32_t kNextKvHeads = cfg::kKvHeads, kNextKvUnits = attn::attn_units(attn_cfg::kM);
struct NextKvPrefetch {
    const int64_t* page_table;  // [seq, page_stride] byte offsets into the KV pool (the sliding-window pool's table)
    const int32_t* kv_len;      // [seq]
    const uint8_t* k_cache;     // the pool the prefetch reads: the successor layer's (newest rows: this launch's)
    const uint8_t* v_cache;
    uint32_t page_stride;
    uint32_t splits;  // the successor attention's split count (0: no swa successor)
    uint32_t window;  // the swa window: the rows before the newest sixteen that the attention reads
};
__device__ unsigned g_next_kv_sink;
static __device__ __forceinline__ void next_kv_prefetch(const NextKvPrefetch& n, const uint32_t warp,
                                                        const uint32_t lane) {
    if (n.splits == 0u) return;
    const uint8_t* const k_cache = n.k_cache;
    const uint8_t* const v_cache = n.v_cache;
    const uint32_t kv_items = kNextKvUnits * n.splits;
    uint32_t acc = 0;
    for (uint32_t kv_item = blockIdx.x; kv_item < kv_items; kv_item += gridDim.x) {
        const uint32_t u = kv_item / n.splits, j = kv_item % n.splits, seq = u / kNextKvHeads, kvh = u % kNextKvHeads;
        const uint32_t len = static_cast<uint32_t>(__ldg(n.kv_len + seq));
        const uint32_t lo = len > 16u + n.window ? len - 16u - n.window : 0u;
        const uint32_t chunk = (len - lo + n.splits - 1u) / n.splits;
        const uint32_t begin = lo + j * chunk;
        const uint32_t end = umin(umin(begin + chunk, len), len > 16u ? len - 16u : 0u);
        const int64_t* const ptab = n.page_table + static_cast<uint64_t>(seq) * n.page_stride;
        // the idle warps split the attention's rows; one row per lane, its k and v slices (sixteen 16 B pieces)
        for (uint32_t r0 = begin + 32u * (warp - kNextLayerEpiEnd); r0 < end; r0 += 32u * kNextLayerIdleWarps) {
            const uint32_t row = umin(r0 + lane, end - 1u);
            const uint64_t off = static_cast<uint64_t>(__ldg(ptab + row)) + kvh * 128u;
            uint4 v[16];
#pragma unroll
            for (uint32_t p = 0; p < 16; ++p)
                v[p] = ptx::ld_global_l2_hint_b128((p < 8 ? k_cache : v_cache) + off + (p & 7) * 16, ptx::kEvictNormal);
#pragma unroll
            for (uint32_t p = 0; p < 16; ++p)
                acc ^= v[p].x ^ v[p].y ^ v[p].z ^ v[p].w;
        }
    }
    if (acc == 0x9E3779B9u) atomicAdd(&g_next_kv_sink, 1u);
}

// the next-layer phase: its body on the block this CTA's pair claims in MoE-exit order, then the idle warps'
// next-layer KV prefetch
struct NextLayerPhase {
    const moe_tp::LayerArgs& la;
    const NextLayerArgs& cargs;
    const CUtensorMap &tmap_w1, &tmap_x1;
    const CUtensorMap &tmap_w2, &tmap_x2, &tmap_out2;
    const NextLayerTileEpi::Args& tile_epi;
    const NextLayerNormItem::Args& item;
    const CombineItem::Args& comb;
    const NextKvPrefetch& nextkv;
    static constexpr bool kPrologue = true, kClusterInit = true;
    // the qkv GEMM's coordinates (QkvCoords: unread, the stores kept for the schedule (README.md)) and the claim words,
    // armed here because the phase's own carve is the MoE phase's smem until its exit (the executor's init fence and
    // the post-attention body's entry cluster_sync publish them to the pair)
    __device__ __forceinline__ void prologue(const LayerShared& sh) const {
        sh.pf->tm_w2 = &tmap_w2;
        sh.pf->w2_outer = int32_t((blockIdx.x / kNextLayerCk) * 64);
        sh.pf->k2_0 = int32_t((ptx::cluster_ctarank() & (kNextLayerCk - 1u)) * (cargs.stages2 / kNextLayerCk));
        *sh.pair_arrive = 0u;
        ptx::mbar_init(sh.claim_bar, 1);
    }
    // the next-layer phase's work in MoE-exit order, so the earliest-exiting pairs run the combine owners' blocks and
    // the unpooler GEMM's first blocks (where a separately launched kernel's scheduler put its first CTAs): the pair's
    // SECOND-exiting CTA claims the next block pair from the MoE phase's cursor (gsync word 7) and hands it to the
    // first through its claim mbarrier; the launch's last claim resets the cursor (every claim of this launch is
    // done; the next launch's claims sit behind its post-attention phase's griddepcontrol.wait). The launch ticket
    // stays the physical CTA's.
    __device__ __forceinline__ void enter(const LayerShared& sh) const {
        if (threadIdx.x == 0) {
            const uint32_t ctarank = ptx::cluster_ctarank();
            const uint32_t peer = ctarank ^ 1u;
            const uint32_t arrived = ptx::atom_add_shared_cluster_u32(
                ptx::mapa_shared_cluster(ptx::to_shared(sh.pair_arrive), ctarank & ~1u), 1u);
            if (arrived == 0u) {
                ptx::mbar_arrive_expect_tx(sh.claim_bar, 4u);
                ptx::mbar_wait_parity_acquire_cluster_poll(sh.claim_bar, 0u);
            } else {
                const uint32_t p = ptx::atomic_add_relaxed_b32(la.gsync + 7, 1u);
                if (p + 1u == gridDim.x / kNextLayerCk) ptx::st_relaxed_gpu_u32(la.gsync + 7, 0u);
                *sh.claim = p;
                ptx::st_async_cluster_b32(ptx::mapa_shared_cluster(ptx::to_shared(sh.claim), peer), p,
                                          ptx::mapa_shared_cluster(ptx::to_shared(sh.claim_bar), peer));
            }
        }
        __syncthreads();
    }
    __device__ __forceinline__ void run(const LayerShared& sh, const PostAttnArgs&, uint8_t* smem_buf) const {
        const uint32_t next_layer_block = *sh.claim * kNextLayerCk + (ptx::cluster_ctarank() & (kNextLayerCk - 1u));
        next_layer_body<postattn::kM, kNextLayerStages, kNextLayerPack, kNextLayerCk, NextLayerTileEpi,
                        NextLayerNormItem, CombineItem>(cargs, tmap_w1, tmap_x1, tmap_w2, tmap_x2, tmap_out2, tile_epi,
                                                        item, comb, smem_buf, next_layer_block, sh.fovl, blockIdx.x);
        // then the idle warps' next-layer KV prefetch
        if (threadIdx.x / 32 >= kNextLayerEpiEnd) next_kv_prefetch(nextkv, threadIdx.x / 32, threadIdx.x & 31);
    }
    __device__ __forceinline__ void retire(uint8_t*) const {}
};

// The one-launch kernel. Its one boundary is next-layer phase(L) -> attention(L+1), and its one PDL trigger is the
// next-layer phase's hand-over (next_layer.cuh), which every CTA reaches. With one CTA per SM a successor CTA lands
// only where a CTA of this launch has exited, so an earlier trigger lands nothing sooner; one fired right behind the
// grid wait leaves the dependent grid released through the whole layer, and it then lands out of blockIdx order and
// later. The trigger orders no data: the successor's grid wait returns at this launch's completion; the cross-rank
// buffers' reuse is ordered by that plus the MoE phase's anchor (moe.cuh (C)). Every image (the 4 MK_IMAGE rows) is
// this kernel; the image picks the attention policy, the weight format and the tail.

// the layer's program: the post-attention phase with the image's attention policy, the MoE phase, and the next-layer
// block claimed in MoE-exit order
template <uint32_t kW>
static __device__ __forceinline__ void layer_body(
    const PostAttnArgs& args, const CUtensorMap& tmap_wo, const CUtensorMap& tmap_xa, const CUtensorMap& tmap_wp,
    const CUtensorMap& tmap_xh, const CUtensorMap& tmap_pooled, const RouterArgs& ra,
    const MoeFrontItem<true>::Args& fa, const CUtensorMap& tmap_wr, const CUtensorMap& tmap_xf,
    const moe_tp::LayerArgs& la, const AttnPolicy::Args& attention, const expert_prefetch::Args& prefetch,
    const NextLayerArgs& cargs, const CUtensorMap& tmap_w1, const CUtensorMap& tmap_x1, const CUtensorMap& tmap_w2,
    const CUtensorMap& tmap_x2, const CUtensorMap& tmap_out2, const NextLayerTileEpi::Args& tile_epi,
    const NextLayerNormItem::Args& item, const CombineItem::Args& comb, const NextKvPrefetch& nextkv, uint8_t* smem_buf,
    const LayerShared& sh) {
    const PostAttnPhase<kW, AttnPolicy> post_attention{args, tmap_wo, tmap_xa, tmap_wp, tmap_xh,   tmap_pooled,
                                                       ra,   fa,      tmap_wr, tmap_xf, attention, prefetch};
    const MoePhase moe{la};
    const NextLayerPhase next_layer{la,        cargs,    tmap_w1, tmap_x1, tmap_w2, tmap_x2,
                                    tmap_out2, tile_epi, item,    comb,    nextkv};
    phase_program(sh, args, smem_buf, post_attention, moe, next_layer);
}

template <uint32_t kW>
__launch_bounds__(384, 1) __global__
    void decoder_layer(const __grid_constant__ PostAttnArgs args, const __grid_constant__ CUtensorMap tmap_wo,
                       const __grid_constant__ CUtensorMap tmap_xa, const __grid_constant__ CUtensorMap tmap_wp,
                       const __grid_constant__ CUtensorMap tmap_xh, const __grid_constant__ CUtensorMap tmap_pooled,
                       const __grid_constant__ RouterArgs ra, const __grid_constant__ MoeFrontItem<true>::Args fa,
                       const __grid_constant__ CUtensorMap tmap_wr, const __grid_constant__ CUtensorMap tmap_xf,
                       const __grid_constant__ moe_tp::LayerArgs la, const __grid_constant__ AttnPolicy::Args attention,
                       const __grid_constant__ expert_prefetch::Args prefetch,
                       const __grid_constant__ NextLayerArgs cargs, const __grid_constant__ CUtensorMap tmap_w1,
                       const __grid_constant__ CUtensorMap tmap_x1, const __grid_constant__ CUtensorMap tmap_w2,
                       const __grid_constant__ CUtensorMap tmap_x2, const __grid_constant__ CUtensorMap tmap_out2,
                       const __grid_constant__ NextLayerTileEpi::Args tile_epi,
                       const __grid_constant__ NextLayerNormItem::Args item,
                       const __grid_constant__ CombineItem::Args comb, const __grid_constant__ NextKvPrefetch nextkv) {
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    __shared__ __align__(16)
        moe_tp::FrontZsOvl s_fovl;  // the attention's histogram / the MoE z staging / the next-layer scratch
    __shared__ uint32_t s_ticket;
    __shared__ moe_tp::QkvCoords s_pf;  // the next-layer prologue's qkv GEMM coordinates (unread)
    __shared__ uint32_t s_moe_epoch;    // this MoE phase's epoch (its store; unread)
    __shared__ __align__(8) uint64_t s_claim_bar;
    __shared__ uint32_t s_pair_arrive;
    __shared__ uint32_t s_claim;
    const LayerShared sh{&s_fovl, &s_ticket, &s_pf, &s_moe_epoch, &s_claim_bar, &s_pair_arrive, &s_claim, nullptr};
    // the global attention's first global accesses precede every CTA barrier; there ptxas holds the global-memory
    // descriptor in a vector register pair and moves it to the uniform datapath before every global access (~1000 R2UR)
    if constexpr (kImage.global_attn) __syncthreads();
    layer_body<kW>(args, tmap_wo, tmap_xa, tmap_wp, tmap_xh, tmap_pooled, ra, fa, tmap_wr, tmap_xf, la, attention,
                   prefetch, cargs, tmap_w1, tmap_x1, tmap_w2, tmap_x2, tmap_out2, tile_epi, item, comb, nextkv,
                   smem_buf, sh);
}

}  // namespace mk

static_assert(kLayerSmem <= mk::cfg::kSmemOptin, "the layer's dynamic smem fits the sm_103a opt-in cap");

#ifndef MK_HOST
namespace mk {
// The kernel's own native text range for the code prefetch: nvcc emits {&kernel, 0} with an
// R_CUDA_DESCRIPTOR relocation; the host retypes it to R_CUDA_64 and writes the .text size into `bytes` before
// cuModuleLoadData, so after load the global holds the kernel's text [pc, pc + bytes).
__device__ CodePrefetchNative g_code_prefetch_native = {reinterpret_cast<void (*)()>(&decoder_layer<mk::cfg::kW>), 0};
}  // namespace mk

// The kernel template is instantiated by taking its address.
[[maybe_unused]] static constexpr auto kLayerKernel = &mk::decoder_layer<mk::cfg::kW>;
#endif
