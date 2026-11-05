#pragma once
// The framework every image shares: the typed configuration (the model per rank and per tensor-parallel world, the
// window's rows, the smem ceiling, the dense ring), the twelve-warp block contract and its role maps, the carve
// descriptors, the ticketed counts between phases, the row ops (the norms, the qk-norm + rope, the all-reduce consume +
// norm), and the dense executor (TMA ring, tcgen05 MMAs, tile epilogues; plain and persistent) with the qkv rowops
// epilogue.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_fp8.h>

#include "ptx.cuh"

// The window and the model as one rank sees it: every image's constants. The tensor-parallel world is one row of the
// world table below (-DMK_WORLD=tp8|tp4); the layer images' per-image axes (attention type, expert dtype, last layer)
// are layer.cu's.

// the verify window's rows (bs 1 x 16 tokens; -DMK_WINDOW_M=8 builds the 8-row window's images): the MMA N of the
// post-attention phase, the next-layer phase and the dense head / tail
#ifndef MK_WINDOW_M
#define MK_WINDOW_M 16
#endif

namespace mk::cfg {

inline constexpr int kWindowM = MK_WINDOW_M;

// A tensor-parallel world: the per-rank shard of the model and the code variants whose shape is the world's. The ranks
// are node-major, 4 per node (the pooler / unpooler gathers stay inside a node); every rank runs the same images.
struct World {
    int ranks;         // tensor-parallel ranks
    int i_shard;       // the expert intermediate per rank (4096 / ranks)
    int q_heads;       // query heads per rank (head_dim 64)
    int kv_heads;      // KV heads per rank
    int router_split;  // the router's K split (the reduction order of the logits)
    bool k_window;     // the pooler and the router stream the rank's K window of h3 / h3f and all-reduce their fp32
                       // partials (the top-k row and the nq rows sum the ranks' planes); else the pooler streams the
                       // rank's N shard at full K (its bf16 tiles gathered in the node) and the router runs replicated
    bool unpool_ksplit;  // the next-layer phase's unpooler GEMM splits K over the width-4 cluster (one block per
                         // cluster); else one block per CTA at full K
    bool topk_redux;     // each top-k round's winner by two warp reductions instead of the shuffle butterfly
    bool route_early;    // the top-k row publishes its route arrival behind the int32 ids and weights alone
    bool front_regs;     // the front item keeps its entries in registers between its histogram and rank passes
    bool text_diet;      // loops that carry no per-element register state stay rolled (fewer instruction bytes
                         // fetched per launch); else unrolled
    bool prefetch_wide;  // the expert prefetch takes the most-routed experts that fit 48 MiB of L2 on every NVFP4
                         // layer; else 4 experts, on the sliding-window layers only
    int combine_groups;  // the attention combine's row groups per (sequence, kv head) unit (combine CTAs per unit)
    bool relfree;        // release-free phase handoffs: the router units release onto their count and the top-k
                         // rows acquire it (no relay of the rows' arrivals), and the next-layer phase's ffn_ready is a
                         // relaxed add with the norm owners' qkv-GEMM tail prefetch issued inside the combine wait
};
namespace world {
inline constexpr World tp8{.ranks = 8,  // 2 nodes x 4 GPUs, one NVLink domain
                           .i_shard = 512,
                           .q_heads = 16,
                           .kv_heads = 2,
                           .router_split = 12,
                           .k_window = false,
                           .unpool_ksplit = true,
                           .topk_redux = false,
                           .route_early = false,
                           .front_regs = false,
                           .text_diet = true,
                           .prefetch_wide = true,
                           .combine_groups = 32,
                           .relfree = false};
inline constexpr World tp4{.ranks = 4,  // one node of 4 GPUs
                           .i_shard = 1024,
                           .q_heads = 32,
                           .kv_heads = 4,
                           .router_split = 8,
                           .k_window = true,
                           .unpool_ksplit = false,
                           .topk_redux = true,
                           .route_early = true,
                           .front_regs = true,
                           .text_diet = false,
                           .prefetch_wide = false,
                           .combine_groups = 16,
                           .relfree = true};
}  // namespace world
inline constexpr World kWorld = world::MK_WORLD;

// The window and the launch.
inline constexpr int kRanks = kWorld.ranks;
inline constexpr int kW = 4;       // cluster width of the layer launch
inline constexpr int kGrid = 144;  // the layer launch's grid: 36 width-4 clusters, every CTA co-resident at its smem
inline constexpr int kMoeM = 64;   // the MoE phase's token capacity (route tables, tiles): >= the window's rows
// the part's per-block dynamic smem maximum (the sm_103a opt-in cap)
inline constexpr int kSmemOptin = 232448;
// the largest dynamic smem the MoE phase's launch was measured to take on GB300 (a larger request fails the opt-in),
// 1,020 B under kSmemOptin: every one-CTA-per-SM launch (the layer, the DFlash draft) sizes to it
inline constexpr int kSmemCeil = 231428;

// The model as one rank sees it.
inline constexpr int kHidden = 8192;     // the residual stream
inline constexpr int kMoeHidden = 4096;  // the MoE input (the pooled row)
inline constexpr int kExperts = 256;     // routed experts
inline constexpr int kTopK = 8;          // experts per token
inline constexpr int kIShard = kWorld.i_shard;
inline constexpr int kFbW = 128;  // the MoE phase's g1 feature-block width
inline constexpr int kQHeads = kWorld.q_heads;
inline constexpr int kKvHeads = kWorld.kv_heads;
inline constexpr int kRouterSplit = kWorld.router_split;
static_assert(kIShard * kRanks == 4096 && kQHeads * kRanks == 128 && kKvHeads * kRanks == 16,
              "the world row shards the model's 4096-wide expert intermediate, 128 query and 16 KV heads");

// The dense ring of every dense GEMM (post-attention, next-layer phase, qkv head, lm_head): 20 stages of 64-element K
// blocks in packs of 4; the head / tail images stream their weights evict-first and the qkv head splits K over a CTA
// pair.
inline constexpr int kStages = 20;
inline constexpr int kPack = 4;
inline constexpr bool kEvictFirst = true;
inline constexpr int kQkvCk = 2;

}  // namespace mk::cfg

// The twelve-warp block contract every body of the layer launch shares (compile time only): the block size, the one
// named-barrier namespace, a body's per-phase ROLE MAP (which warp holds which duty) and the WARP SPAN a row item runs
// on.
// Every member is constexpr: a body that spells its roles through a map folds to the same literals.

namespace mk::block {

inline constexpr uint32_t kWarps = 12;
inline constexpr uint32_t kThreads = kWarps * 32;

// Named barriers (`bar.sync id, count`): ONE namespace for every body of the launch, because consecutive phases run in
// the same CTA and an id live at two thread counts at once is undefined. 0 the CTA barrier; 1 the MoE phase's
// role-scoped publish; 2 the dense epilogue quad (kEpiBarId, 128 threads); 3 the ITEM barrier (256 threads: the eight
// warps of an item's warp span); 8.. the phase boundaries of a body that also carries an item.
enum Bar : uint32_t { kCta = 0, kEpi = 2, kItem = 3, kPhaseBarBase = 8 };

inline constexpr int kNone = -1;

// A phase's role map: which warp holds which duty. The duties are the contract's; the warp numbers are each body's
// declaration. `aux` may sit on the MMA issuer's own warp or be absent; `join` absent means the last epilogue warp
// joins.
struct RoleMap {
    int producer;  // the weight (A) stream — pre-wait: weights are never a predecessor's output
    int gate;      // the PDL-ordered activation feeder (kNone: the whole CTA waits at entry and the producer gathers)
    int mma;       // the tcgen05 issuer, owner of the ring's commits
    int aux;       // the scale-factor copy issuer (the MoE phase) / the TMEM allocator (the dense executor), or kNone
    int join;      // the join / emit warp (the MoE phase), or kNone
    int epi0;      // the first epilogue warp
    int n_epi;     // epilogue warps: one quad or two
};

constexpr bool in_block(int w) {
    return w >= 0 && w < int(kWarps);
}

constexpr bool holds_epi(const RoleMap& r, int w) {
    return w >= r.epi0 && w < r.epi0 + r.n_epi;
}

// The shape every declared map must have: distinct issue-side warps, the epilogue quads whole and inside the
// block, no issue-side warp inside the epilogue range (aux on the MMA warp is the one admitted alias).
constexpr bool valid(const RoleMap& r) {
    if (!in_block(r.producer) || !in_block(r.mma) || !in_block(r.epi0)) return false;
    if (r.n_epi != 4 && r.n_epi != 8) return false;
    if (r.epi0 + r.n_epi > int(kWarps)) return false;
    if (r.producer == r.mma) return false;
    if (r.gate != kNone && (!in_block(r.gate) || r.gate == r.producer || r.gate == r.mma)) return false;
    if (r.aux != kNone && (!in_block(r.aux) || r.aux == r.producer || r.aux == r.gate)) return false;
    if (r.join != kNone &&
        (!in_block(r.join) || r.join == r.producer || r.join == r.mma || r.join == r.gate || r.join == r.aux))
        return false;
    // the issue side never sits inside the epilogue range (a join on an epilogue warp is the one admitted overlap)
    if (holds_epi(r, r.producer) || holds_epi(r, r.mma)) return false;
    if (r.gate != kNone && holds_epi(r, r.gate)) return false;
    if (r.aux != kNone && r.aux != r.mma && holds_epi(r, r.aux)) return false;
    return true;
}

// The WARP SPAN a row ITEM runs on: the eight warps from the first epilogue warp (the epilogue quad plus the idle quad
// of the dense executor), the 256-thread layout every row body is written for.
struct WarpSpan {
    int warp0;
    int n_warps;
};
inline constexpr int kItemWarps = 8;
constexpr WarpSpan item_span(const RoleMap& r) {
    return WarpSpan{r.epi0, kItemWarps};
}
constexpr bool fits(const WarpSpan& s, int item_warps) {
    return s.warp0 >= 0 && item_warps <= s.n_warps && s.warp0 + s.n_warps <= int(kWarps);
}

}  // namespace mk::block

// The carve descriptor a phase body exports (compile time only): what a launch must budget to host the body (dynamic
// smem, declared static smem, mbarrier words, TMEM columns) and the composition rules as code. Two SEQUENTIAL phases
// of one launch alias the dynamic smem (max) and keep their static smem and mbarriers apart (sum); an ITEM adds static
// smem only; TMEM is one allocation per launch that the phases re-carve. A body static_asserts its export against its
// real carve, and the launch static_asserts the composition against its budget.

namespace mk::carve {

struct Carve {
    int dyn_smem;     // bytes of the dynamic carve
    int static_smem;  // bytes of static smem the body declares
    int mbars;        // mbarrier words
    int tmem_cols;    // TMEM columns of the phase's allocation
};

// Two sequential phases of a launch: dynamic smem the max, static smem and mbarrier words the sum, TMEM the max.
__host__ __device__ constexpr Carve phases(Carve a, Carve b) {
    return Carve{a.dyn_smem > b.dyn_smem ? a.dyn_smem : b.dyn_smem, a.static_smem + b.static_smem, a.mbars + b.mbars,
                 a.tmem_cols > b.tmem_cols ? a.tmem_cols : b.tmem_cols};
}

// An item running on a body's warp span: static smem only.
__host__ __device__ constexpr Carve item(Carve host, int item_static_smem) {
    return Carve{host.dyn_smem, host.static_smem + item_static_smem, host.mbars, host.tmem_cols};
}

}  // namespace mk::carve

// Kernel-parameter structs of the dense executor and of the post-attention norm + quant item.
// Their layouts are part of the kernel ABI: the host packs them field for field.

namespace mk {

// One 64-row W block per CTA (grid = N / 64), every tensor through a tensor map; only the K extent is runtime.
struct DenseArgs {
    uint32_t stages_total;  // K / 64: W stages (= x k-blocks) per CTA
};

// The persistent variant: grid < N / 64 CTAs, CTA i taking blocks b = i + j * gridDim.x. The schedule may pad N past
// the weight's rows (lm_head): the tensor maps keep the real extents, so W loads past them zero-fill and the stores
// clip; n_out skips the stores of fully padded blocks.
struct DensePersistArgs {
    uint32_t stages_total;  // K / 64: W stages (= x k-blocks) per block
    uint32_t num_blocks;    // the schedule's N / 64 (>= the weight's n / 64)
    uint32_t n_out;         // the weight's out columns
};

// The pre-expert norm + dynamic NVFP4 quant of each pooled row (the post-attention nq item); the rows come from the
// pooler's gathered row (or the K-window variant's partial planes) (postattn.cuh). pooled, pooled_ready and
// arrivals are unused, kept for the schedule (README.md).
struct NormQuantArgs {
    void* pooled;       // unused
    const void* gamma;  // [H] bf16: the layer's effective pre-expert gamma
    float eps;
    uint8_t* q;              // [kM, H/2] e2m1 codes
    uint8_t* sf;             // [kM, H/16] e4m3 block scales
    float* x_scale;          // [kM] the per-row global scale
    uint64_t* pooled_ready;  // unused
    uint32_t arrivals;       // unused
};

// The dense executor: a warp-specialized GEMM for skinny activations, out[m, n] = x[m, k] @ w[n, k]^T in bf16 with fp32
// accumulation in TMEM. Each CTA owns 64-row W blocks (grid = N / 64): W streams in 8-KB (64-element) stages through
// a deep ring, x k-blocks ride the same stage barriers (streamed, not resident), one TMA barrier in and one MMA commit
// out per pack of stages, the ring head is L2-prefetched before the init, and the epilogue stores the tile by TMA.
// kCk = 2 / 4 splits K over a 2- / 4-CTA cluster (the partials meet in smem through st.async). Layer 0's qkv head
// runs this body (kCk 2 with the rowops tile hook) and the lm_head its persistent variant (below); the layer's
// post-attention and next-layer phases share its role map, carve conventions and drain, with bodies of their own.

// The ring-head W L2 prefetch lands evict-first: the W stream demands those lines right after, and an unhinted
// prefetch would be the layer's largest normal-priority stream.
static __device__ __forceinline__ void dense_prefetch_w(const CUtensorMap* tmap, int32_t x, int32_t y) {
    ptx::cp_async_bulk_prefetch_tensor_2d_l2_hint(tmap, x, y, ptx::kEvictFirst);
}

// The executor's role map: w0 the W producer (before the grid wait: weights are never a predecessor's output), w1 the x
// producer (the PDL-ordered gate), w2 the MMA issuer, w3 the TMEM allocator, w4..w7 the epilogue quad (its last warp
// issues the pre-init W prefetch); the 384-thread launch leaves w8..w11 idle.
inline constexpr block::RoleMap kDenseRoles{/*producer=*/0,        /*gate=*/1, /*mma=*/2,  /*aux=*/3,
                                            /*join=*/block::kNone, /*epi0=*/4, /*n_epi=*/4};
static_assert(block::valid(kDenseRoles), "the dense executor's role map has the framework's shape");
inline constexpr uint32_t kDenseWProducer = kDenseRoles.producer, kDenseGate = kDenseRoles.gate,
                          kDenseMma = kDenseRoles.mma;
inline constexpr uint32_t kDenseTmem = kDenseRoles.aux, kDenseEpi0 = kDenseRoles.epi0;
inline constexpr uint32_t kDenseEpiEnd = kDenseRoles.epi0 + kDenseRoles.n_epi, kDensePrefetch = kDenseEpiEnd - 1;

// Dynamic-smem carve [W ring][x ring][out tile][ck combine stage][tmem word][barrier pool]: the one spelling of the
// layout, used by the device carve and by the host's smem sizing. The barriers live in dynamic smem (a static-smem pool
// costs ~60 cycles per handshake). Every slot carries `pack` W stages and `pack` x k-blocks behind one barrier pair
// (the x stream through L2 bounds the ring, so x is as deep as W).
struct DenseSmemLayout {
    uint32_t x_off;     // x ring, after the ring's 8KB W stages
    uint32_t out_off;   // [m, 64] bf16 epilogue tile
    uint32_t cmb_off;   // ck 2: rank 1's [64, m] fp32 partial (row-major, row stride dense_cmb_stride); ck 4:
                        // the peers' [64 rows][4 sources][m / 4 tokens] fp32 partials of this CTA's token quarter
    uint32_t tmem_off;  // tmem addr word (8-B slot)
    uint32_t bar_off;   // pool: [full][empty] per slot + tmem full (+ ck cmb)
    uint32_t size;      // total dynamic smem, 1-KB rounded
};

// The TMEM alloc grain: the power of two >= max(32, cols).
__host__ __device__ constexpr uint32_t dense_pow2_cols(uint32_t cols) {
    uint32_t p = 32;
    while (p < cols)
        p *= 2;
    return p;
}

// TMEM accumulator columns of the m64n{m} D tile: the pow2 alloc grain, >= 32.
__host__ __device__ constexpr uint32_t dense_tmem_cols(uint32_t m) {
    return dense_pow2_cols(m);
}

// fp32 row stride of the ck combine stages (the pair's, and the 4-way [64 rows][4 sources][m / 4 tokens]). Past m 16
// a stride of m floats is a 128-B multiple and lands a quad's 16 active lanes on one bank per reduction column; +4
// floats spreads them and keeps 16-B alignment (m 16 keeps m).
__host__ __device__ constexpr uint32_t dense_cmb_stride(uint32_t m) {
    return m == 16 ? m : m + 4;
}

__host__ __device__ constexpr DenseSmemLayout dense_smem_layout(uint32_t m, uint32_t pack, uint32_t ring, uint32_t ck) {
    const uint32_t stage = 64 * 128;  // 64 W rows x one 128-B K-row
    const uint32_t x_tile = m * 128;
    const uint32_t x_off = ring * stage;
    const uint32_t out_off = x_off + ring * x_tile;
    const uint32_t cmb_off = out_off + m * 128;  // past the [m, 64] bf16 out tile
    const uint32_t cmb_bytes = ck > 1 ? 64 * dense_cmb_stride(m) * 4 : 0;
    const uint32_t tmem_off = cmb_off + cmb_bytes;
    const uint32_t end = tmem_off + 8 + (2 * (ring / pack) + 1 + (ck > 1 ? 1 : 0)) * 8;
    return {x_off, out_off, cmb_off, tmem_off, tmem_off + 8, (end + 1023) / 1024 * 1024};
}

// The executor's carve export: its dynamic carve, no static smem, two barrier words per ring slot
// plus two (TMEM full, the ck combine), one TMEM allocation.
__host__ __device__ constexpr mk::carve::Carve dense_carve(uint32_t m, uint32_t pack, uint32_t ring, uint32_t ck) {
    return mk::carve::Carve{static_cast<int>(dense_smem_layout(m, pack, ring, ck).size), 0,
                            2 * static_cast<int>(ring / pack) + 2, static_cast<int>(dense_tmem_cols(m))};
}

// The epilogue's tile hook: called by the 128 epilogue threads (warps 4..7), converged, after the out-tile sync (on the
// K-split pair by the tile's owner). `out_tile` = the [kM tokens x 64] bf16 token-major tile (stride 64) of output
// block [w_outer, w_outer + 64), already rounded to bf16. A policy may rewrite the tile (the TMA store then ships the
// rewrite) and issue its own stores; kMutatesTile adds the republish sync. The default does nothing; the qkv rowops
// epilogue is TileEpiQkvRowops (below).
struct TileEpiNone {
    struct Args {};
    static constexpr bool kMutatesTile = false;
    static constexpr bool kCkSafe = true;  // rank-0-only invocation at ck>1 is fine for a no-op
    static __device__ __forceinline__ void pre_store(const Args&, const uint32_t /*w_outer*/, nv_bfloat16* /*out_tile*/,
                                                     const uint32_t /*quad*/, const uint32_t /*lane_idx*/) {}
};
// The executor as a device function, over a caller-given block id (the kernels below, and the layer's phases), on a
// 256-thread block. `kArenaStore` = false skips the tile's TMA store.
// `kPrefetchSlice`: the pre-init W L2 prefetch covers the CTA's whole K slice, not just the ring head.
template <uint32_t kM, uint32_t kStages, uint32_t kPack, bool kEvictFirst, uint32_t kCk = 1,
          class kTileEpi = TileEpiNone, bool kArenaStore = true, bool kPrefetchSlice = false>
static __device__ __forceinline__ void dense_body(const DenseArgs& args, const CUtensorMap& tmap_w,
                                                  const CUtensorMap& tmap_x, const CUtensorMap& tmap_out,
                                                  const typename kTileEpi::Args& tile_epi, uint8_t* smem_buf,
                                                  const uint32_t block_id) {
    // ---- the static geometry ---------------------------------------------------
    constexpr uint32_t kPfr = kStages;          // the pre-init W L2 prefetch depth: one ring
    constexpr uint32_t BN = 64;                 // W rows per CTA block
    constexpr uint32_t kStageBytes = BN * 128;  // one 64-elem W stage (8KB)
    constexpr uint32_t kXTile = kM * 128;       // one 64-elem x k-block
    constexpr uint32_t kClusterCtas = kCk;
    constexpr uint32_t kNumSlots = kStages / kPack;
    constexpr uint32_t kAtoms = 4;                       // k16 MMA atoms per stage
    constexpr uint32_t kTmemCols = dense_tmem_cols(kM);  // the accumulator at the alloc grain
    constexpr uint32_t kEpiBarId = block::kEpi;
    constexpr uint32_t kEpiThreads = kDenseRoles.n_epi * 32;
    constexpr uint32_t kCmbStride = dense_cmb_stride(kM);
    static_assert((kM == 8 || kM % 16 == 0) && kM <= 64, "8- or 16k-row windows up to the m64 D tile's 64 columns");
    static_assert((kXTile) % 1024 == 0, "x k-block tiles the 128B-swizzle atom");
    static_assert(kStages % kPack == 0 && kPack <= kStages / 2, "pack must divide the ring and be <= half of it");
    static_assert(kCk == 1 || kCk == 2 || kCk == 4, "kCk: 1 flat | 2- or 4-CTA cluster K-split");
    // ck 4 (the K 12288 qkv: 36 W blocks on 144 CTAs): the SYMMETRIC reduction — CTA q of the cluster streams K
    // quarter q, ships each peer p the fp32 partial of p's kM / 4 tokens (st.async, tx on p's cmb_bar), reduces its own
    // kM / 4 tokens over the four quarters in the fixed order ((p0 + p1) + p2) + p3 and TMA-stores them (the host's out
    // box is kM / 4 rows at ck 4); a plain tile sink only
    static_assert(kCk != 4 || (kArenaStore && !kTileEpi::kMutatesTile), "ck 4: the arena store, no mutating hook");
    // ck>1 runs the tile hook on rank 0 only — an every-rank-store policy
    // (the AR push) would half-push
    static_assert(kCk == 1 || kTileEpi::kCkSafe, "tile epi is not ck-safe (the hook is rank-0-only at ck>1)");

    // ---- smem: the shared dense_smem_layout carve (note above) ----
    constexpr DenseSmemLayout kSmem = dense_smem_layout(kM, kPack, kStages, kCk);
    static_assert(dense_carve(kM, kPack, kStages, kCk).dyn_smem == static_cast<int>(kSmem.size) &&
                      dense_carve(kM, kPack, kStages, kCk).mbars == static_cast<int>(2 * kNumSlots + 2) &&
                      dense_carve(kM, kPack, kStages, kCk).tmem_cols == static_cast<int>(kTmemCols),
                  "the dense executor's exported carve is its real carve");
    const uint32_t smem_w = ptx::to_shared(smem_buf);
    const uint32_t smem_x = smem_w + kSmem.x_off;
    auto out_tile = reinterpret_cast<nv_bfloat16*>(smem_buf + kSmem.out_off);
    auto cmb_stage = reinterpret_cast<float*>(smem_buf + kSmem.cmb_off);
    auto tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_buf + kSmem.tmem_off);
    uint64_t* const bar_base = reinterpret_cast<uint64_t*>(smem_buf + kSmem.bar_off);
    uint64_t* const full_bars = bar_base;
    uint64_t* const empty_bars = bar_base + kNumSlots;
    uint64_t* const tmem_full_bar = bar_base + 2 * kNumSlots;
    uint64_t* const cmb_bar = bar_base + 2 * kNumSlots + 1;  // kCk>1 only

    const uint32_t warp_idx = ptx::warp_idx_sync();
    const uint32_t lane_idx = ptx::lane_idx();
    // ck: cluster c owns W block c (its ranks stream disjoint K slices of the SAME 64 rows) and reduces the partials
    const uint32_t rank = kClusterCtas > 1 ? ptx::cluster_ctarank() : 0;
    const uint32_t k_rank = rank % kCk;
    const uint32_t w_outer = (block_id / kCk + rank / kCk) * BN;  // rank / kCk is 0

    const auto umin = [](const uint32_t& a, const uint32_t& b) { return a < b ? a : b; };
    // this rank's K span (contiguous split of the stages)
    const uint32_t k_base = args.stages_total / kCk;
    const uint32_t k_rem = args.stages_total % kCk;
    const uint32_t k0 = k_rank * k_base + umin(k_rank, k_rem);
    const uint32_t k_iters = k_base + (k_rank < k_rem ? 1u : 0u);

    // the pre-init W L2 prefetch of one ring of stages: warp 7 idles until the epilogue, so it issues the prefetch in
    // the init / alloc window
    if (kPfr > 0 && warp_idx == kDensePrefetch && ptx::elect_one_sync()) {
        const uint32_t kq = umin(kPfr, k_iters);
        for (uint32_t q = 0; q < kq; ++q) {
            dense_prefetch_w(&tmap_w, static_cast<int32_t>((k0 + q) * 64), static_cast<int32_t>(w_outer));
        }
        // the rest of the slice too, at normal priority (evict-first lines would evict each other before the refill
        // reads them): a block resident ahead of its input lands its whole W in L2 before the wait
        if constexpr (kPrefetchSlice)
            for (uint32_t q = kq; q < k_iters; ++q)
                ptx::cp_async_bulk_prefetch_tensor_2d_l2_hint(&tmap_w, static_cast<int32_t>((k0 + q) * 64),
                                                              static_cast<int32_t>(w_outer), ptx::kEvictNormal);
    }

    // ---- init (warp 1) + TMEM alloc (warp 3) ------------------------------
    if (warp_idx == kDenseGate && ptx::elect_one_sync()) {
        ptx::prefetch_tensormap(&tmap_w);
        ptx::prefetch_tensormap(&tmap_x);
        for (uint32_t i = 0; i < kNumSlots; ++i) {
            ptx::mbar_init(&full_bars[i], 1);
            ptx::mbar_init(&empty_bars[i], 1);
        }
        ptx::mbar_init(tmem_full_bar, 1);
        if constexpr (kCk > 1) ptx::mbar_init(cmb_bar, 1);
        ptx::fence_mbarrier_init_release_cluster();
    }
    if (warp_idx == kDenseTmem) ptx::tcgen05_alloc(ptx::to_shared(tmem_addr_storage), kTmemCols);
    if constexpr (kClusterCtas > 1)
        ptx::cluster_sync();  // peer mbars armed before any st.async lands
    else
        __syncthreads();
    // the TMEM address: read after the init sync by every warp
    const uint32_t taddr = tmem_addr_storage[0];

    // =======================================================================
    // warp 1: x producer — PDL follower discipline: x is the predecessor's
    // output, so every tmap_x issue orders after prior-grid completion while
    // W streams pre-wait (ring-capped). The dependents trigger fires right
    // AFTER the wait (a no-op without the PSS launch attribute). x k-blocks
    // ride the pack's stage barrier 1:1 with the W stages (the streamed-x
    // winner coupling); the W producer's expect_tx covers both byte streams.
    // =======================================================================
    if (warp_idx == kDenseGate) {
        ptx::griddepcontrol_wait();
        if (ptx::elect_one_sync()) ptx::griddepcontrol_launch_dependents();
        uint32_t slot = 0, phase = 0;
        for (uint32_t s = 0; s < k_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(smem_x + (slot * kPack + d) * kXTile, &tmap_x,
                                                           static_cast<int32_t>((k0 + s + d) * 64), /*y=*/0,
                                                           &full_bars[slot], ptx::kEvictNormal);
                }
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
    }

    // =======================================================================
    // warp 0: W producer: kPack 8-KB stage loads per barrier; kEvictFirst marks the read-once W stream so it does not
    // displace the activations other kernels re-read from L2
    // =======================================================================
    if (warp_idx == kDenseWProducer) {
        uint32_t slot = 0, phase = 0;
        for (uint32_t s = 0; s < k_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k_iters - s);
            ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
            if (ptx::elect_one_sync()) {
                for (uint32_t d = 0; d < deliveries; ++d) {
                    ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + (slot * kPack + d) * kStageBytes, &tmap_w,
                                                           static_cast<int32_t>((k0 + s + d) * 64),
                                                           static_cast<int32_t>(w_outer), &full_bars[slot],
                                                           kEvictFirst ? ptx::kEvictFirst : ptx::kEvictNormal);
                }
                ptx::mbar_arrive_expect_tx(&full_bars[slot], deliveries * (kStageBytes + kXTile));
            }
            __syncwarp();
            slot = slot == kNumSlots - 1 ? 0 : slot + 1;
            phase ^= slot == 0;
        }
    }

    // =======================================================================
    // warp 2: MMA — one wait per pack (covers W + x), m64n{kM}k16 ::1 atoms
    // in k order, one empty commit per pack, one tmem_full commit at the end
    // =======================================================================
    if (warp_idx == kDenseMma) {
        constexpr uint32_t idesc = ptx::mma_inst_desc_f16(BN, kM);
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_x);
        uint32_t slot = 0, phase = 0;
        for (uint32_t s = 0; s < k_iters; s += kPack) {
            const uint32_t deliveries = umin(kPack, k_iters - s);
            ptx::mbar_wait_parity_poll(&full_bars[slot], phase);
            ptx::tcgen05_fence_after_thread_sync();
            if (ptx::elect_one_sync()) {  // hoisted over the whole pack
                for (uint32_t d = 0; d < deliveries; ++d) {
                    const uint32_t sl = slot * kPack + d;
#pragma unroll
                    for (uint32_t kk = 0; kk < kAtoms; ++kk) {
                        // +2 units = one MMA atom's K advance (32 B)
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

    // =======================================================================
    // warps 4..7: drain the m64 D layout (quadrant q's first 16 lanes = W
    // rows 16q..16q+15, all kM columns regular — the decoded ground truth;
    // columns >= the runtime m are exact zeros from the x OOB fill and the
    // out tensormap's row clip drops them) into a [kM, 64] bf16 tile, then
    // ONE async TMA store; the CTA exits without draining the store group
    // (bulk async stores complete before the grid's PDL memory-flush point)
    // =======================================================================
    if constexpr (kCk == 4 && requires { kTileEpi::kCk4PairFixup; }) {
        // a ck 4 pair fixup's other four warps: the producer / gate / MMA / aux warps, past their loops, join the row
        // band the epilogue quad reaches after its quarter's stores (their rows of the pair's token quarter)
        if (warp_idx < kDenseEpi0)
            kTileEpi::template ck4_pair_fixup<kM>(tile_epi, w_outer, k_rank, warp_idx, lane_idx, out_tile,
                                                  /*ticket_thread=*/false);
    }
    if (warp_idx >= kDenseEpi0) {
        const uint32_t quad = warp_idx - kDenseEpi0;  // the TMEM lane band this warp drains
        // ck: arm the combine expect early (a complete-tx landing before the arm is legal — count 1 closes the phase)
        if constexpr (kCk == 2) {
            if (k_rank == 0 && warp_idx == kDenseEpi0 && ptx::elect_one_sync())
                ptx::mbar_arrive_expect_tx(cmb_bar, 64 * kM * 4);
        } else if constexpr (kCk == 4) {
            if (warp_idx == kDenseEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive_expect_tx(cmb_bar, 3 * 64 * kM);
        }
        ptx::mbar_wait_parity_poll(tmem_full_bar, 0);
        ptx::tcgen05_fence_after_thread_sync();
        // one x16 drain per 16 columns (one at m 16; one x8 at m 8)
        uint32_t v[kM];
        ptx::tcgen05_ld_32x32b_cols<kM>(taddr, v);
        ptx::tcgen05_wait_ld();
        if constexpr (kCk == 4) {
            constexpr uint32_t kQ = kM / 4;
            constexpr uint32_t kStride = dense_cmb_stride(kM);
            static_assert(kQ % 4 == 0, "ck 4 ships each quarter's tokens as 16 B packets of four");
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
                const uint32_t mbar_l = ptx::to_shared(cmb_bar);
#pragma unroll
                for (uint32_t p = 0; p < 4; ++p) {
                    if (p == k_rank) continue;
                    const uint32_t dst =
                        ptx::mapa_shared_cluster(ptx::to_shared(cmb_stage + row * kStride + k_rank * kQ), p);
                    const uint32_t mbar = ptx::mapa_shared_cluster(mbar_l, p);
#pragma unroll
                    for (uint32_t g = 0; g < kQ / 4; ++g) {
                        const uint32_t c0 = p * kQ + 4 * g;
                        const uint4 pk = {v[c0], v[c0 + 1], v[c0 + 2], v[c0 + 3]};
                        ptx::st_async_cluster_v4(dst + g * 16, pk, mbar);
                    }
                }
            }
            // the TMEM columns are drained (each warp's wait above): free them once every epi warp passed it
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            if (warp_idx == kDenseEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);
            ptx::mbar_wait_parity_acquire_cluster(cmb_bar, 0);
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
                // a uniform branch per quarter: constant indices keep v in registers
                const auto reduce_part = [&](auto qc) {
                    constexpr uint32_t Q = decltype(qc)::value;
#pragma unroll
                    for (uint32_t j0 = 0; j0 < kQ; j0 += 4) {
                        float part[4][4];
#pragma unroll
                        for (uint32_t src = 0; src < 4; ++src) {
                            if (src == Q) {
#pragma unroll
                                for (uint32_t e = 0; e < 4; ++e)
                                    part[src][e] = __int_as_float(v[Q * kQ + j0 + e]);
                            } else {
                                const float4 f =
                                    *reinterpret_cast<const float4*>(cmb_stage + row * kStride + src * kQ + j0);
                                part[src][0] = f.x, part[src][1] = f.y, part[src][2] = f.z, part[src][3] = f.w;
                            }
                        }
#pragma unroll
                        for (uint32_t e = 0; e < 4; ++e) {
                            const float acc = ((part[0][e] + part[1][e]) + part[2][e]) + part[3][e];
                            out_tile[(Q * kQ + j0 + e) * BN + row] = __float2bfloat16_rn(acc);
                        }
                    }
                };
                if (k_rank == 0u)
                    reduce_part(std::integral_constant<uint32_t, 0>{});
                else if (k_rank == 1u)
                    reduce_part(std::integral_constant<uint32_t, 1>{});
                else if (k_rank == 2u)
                    reduce_part(std::integral_constant<uint32_t, 2>{});
                else
                    reduce_part(std::integral_constant<uint32_t, 3>{});
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            if constexpr (requires { kTileEpi::kCk4PairFixup; }) {
                // the pair fixup: this CTA's quarter leaves by generic stores as below, then the row ops run on the
                // second CTA of each (head, quarter) pair, all eight warps
                const uint32_t tid = quad * 32 + lane_idx;
                if (tid < kQ * 8 && k_rank * kQ + (tid >> 3) < tile_epi.rows) {
                    const uint32_t c = k_rank * kQ + (tid >> 3), j = tid & 7u;
                    *reinterpret_cast<uint4*>(tile_epi.out + static_cast<uint64_t>(c) * tile_epi.out_stride + w_outer +
                                              j * 8) = *reinterpret_cast<const uint4*>(out_tile + c * BN + j * 8);
                }
                kTileEpi::template ck4_pair_fixup<kM>(tile_epi, w_outer, k_rank, warp_idx, lane_idx, out_tile,
                                                      warp_idx == kDenseEpi0 && lane_idx == 0);
                return;
            }
            if constexpr (requires { kTileEpi::kCk4RowsFixup; }) {
                // a row fixup behind the reduction: this CTA's quarter leaves by generic stores (the head's last CTA
                // reads every quarter back from L2 behind the ticket's fences)
                const uint32_t tid = quad * 32 + lane_idx;
                if (tid < kQ * 8 && k_rank * kQ + (tid >> 3) < tile_epi.rows) {
                    const uint32_t c = k_rank * kQ + (tid >> 3), j = tid & 7u;
                    *reinterpret_cast<uint4*>(tile_epi.out + static_cast<uint64_t>(c) * tile_epi.out_stride + w_outer +
                                              j * 8) = *reinterpret_cast<const uint4*>(out_tile + c * BN + j * 8);
                }
                kTileEpi::template ck4_fixup<kM>(tile_epi, w_outer, quad, lane_idx);
                return;
            }
            if (warp_idx == kDenseEpi0 && ptx::elect_one_sync()) {
                ptx::fence_async_smem();
                ptx::cp_async_bulk_tensor_2d_store(ptx::to_shared(out_tile + k_rank * kQ * BN), &tmap_out,
                                                   static_cast<int32_t>(w_outer), static_cast<int32_t>(k_rank * kQ));
                ptx::tma_store_commit();
            }
            return;  // the CTA exits without draining the store group (as below)
        }
        if constexpr (kCk > 1) {
            if (k_rank != 0) {
                // ship the fp32 partial to rank 0's stage straight from the
                // drain registers ([64, kM] row-major: a lane's kM columns
                // are contiguous -> kM/4 v4 st.async deliveries)
                if (lane_idx < 16) {
                    const uint32_t row = quad * 16 + lane_idx;
                    const uint32_t dst = ptx::mapa_shared_cluster(ptx::to_shared(cmb_stage + row * kCmbStride), 0);
                    const uint32_t mbar = ptx::mapa_shared_cluster(ptx::to_shared(cmb_bar), 0);
#pragma unroll
                    for (uint32_t g = 0; g < kM / 4; ++g) {
                        const uint4 pk = {v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3]};
                        ptx::st_async_cluster_v4(dst + g * 16, pk, mbar);
                    }
                }
                // order the dealloc after EVERY epi warp's tcgen05_ld has
                // drained (each warp's wait covers only its own loads; a
                // dependent CTA may re-alloc the freed columns)
                ptx::sync_aligned(kEpiBarId, kEpiThreads);
                if (warp_idx == kDenseEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);
                return;  // rank 0 owns the reduction + store
            }
            // acquire wait: the peer's st.async payload must be visible
            ptx::mbar_wait_parity_acquire_cluster(cmb_bar, 0);
        }
        if (lane_idx < 16) {
            const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
            for (uint32_t c = 0; c < kM; ++c) {
                float acc = __int_as_float(v[c]);
                if constexpr (kCk > 1) acc += cmb_stage[row * kCmbStride + c];  // fixed rank order
                out_tile[c * BN + row] = __float2bfloat16_rn(acc);
            }
        }
        ptx::sync_aligned(kEpiBarId, kEpiThreads);
        kTileEpi::pre_store(tile_epi, w_outer, out_tile, quad, lane_idx);
        if constexpr (kTileEpi::kMutatesTile) ptx::sync_aligned(kEpiBarId, kEpiThreads);
        if constexpr (kArenaStore) {
            if (warp_idx == kDenseEpi0 && ptx::elect_one_sync()) {
                ptx::fence_async_smem();
                ptx::cp_async_bulk_tensor_2d_store(ptx::to_shared(out_tile), &tmap_out, static_cast<int32_t>(w_outer),
                                                   /*y=*/0);
                ptx::tma_store_commit();
            }
        }
        if (warp_idx == kDenseEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);  // no relinquish
        // exit hands this smem to the next CTA without draining the store group (bulk async stores complete before
        // the grid's PDL memory-flush point)
    }
}

template <uint32_t kM, uint32_t kStages, uint32_t kPack, bool kEvictFirst, uint32_t kCk = 1,
          class kTileEpi = TileEpiNone>
__launch_bounds__(256, 1) __global__
    void dense_gemm(const __grid_constant__ DenseArgs args, const __grid_constant__ CUtensorMap tmap_w,
                    const __grid_constant__ CUtensorMap tmap_x, const __grid_constant__ CUtensorMap tmap_out,
                    const __grid_constant__ typename kTileEpi::Args tile_epi) {
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    dense_body<kM, kStages, kPack, kEvictFirst, kCk, kTileEpi>(args, tmap_w, tmap_x, tmap_out, tile_epi, smem_buf,
                                                               blockIdx.x);
}

// ============================================================================
// Persistent variant — the same deep-stage ring discipline with each CTA walking
// MULTIPLE 64-row W blocks (b = blockIdx.x + i * gridDim.x; the host picks a
// grid that balances the blocks), for the deep-wave shapes (lm_head: 792
// schedule blocks over ~152 SMs). One barrier init + tmem alloc + ring warmup
// per CTA instead of per wave, and the W/x ring streams THROUGH block
// boundaries; between blocks only the accumulator handoff (the MMA block
// loop's tmem_empty parity — single accumulator, MMA of block j waits the
// drain of j-1) and the out-tile republish gate sit on the critical path,
// both absorbed by the ring's in-flight stages. The schedule may pad N past
// the raw weight/out rows: the tensormaps keep the raw extents, so W loads
// past them zero-fill, the boundary block's store clips, and fully-OOB pad
// blocks skip the store (args.n_out). ::1 only — the ck K-split trades ring
// depth for wave width, the opposite of what a deep block loop needs.
// ============================================================================

// Persist smem carve = the dense ck=1 carve + one tmem-empty barrier at the
// pool tail (the persistent accumulator handoff). The persistent body drains
// the accumulator unrounded: its out tile is [m, 64] fp32, twice the bf16
// tile; f32_tile = false keeps the bf16 tile's carve (draft_head's body,
// which has a tile of its own).
__host__ __device__ constexpr DenseSmemLayout dense_persist_smem_layout(uint32_t m, uint32_t pack, uint32_t ring,
                                                                        bool f32_tile = true) {
    DenseSmemLayout l = dense_smem_layout(m, pack, ring, 1);
    const uint32_t grow = f32_tile ? m * 128 : 0;
    l.cmb_off += grow;
    l.tmem_off += grow;
    l.bar_off += grow;
    l.size = (l.bar_off + (2 * (ring / pack) + 2) * 8 + 1023) / 1024 * 1024;
    return l;
}

// The persistent variant's epilogue sink, PersistGather: the vocab all-gather folded into the lm_head epilogue. Each
// epilogue thread stores its two 16-B packets per 16 rows ONCE to the rows' multicast address (the switch replicates
// them into every rank's [kM, v_total] fp32 row) at this rank's column window. After its last block each CTA fences its
// pushes and counts itself on a local grid word; the grid's last CTA publishes ONE arrival (release, system scope,
// multicast) to every rank's arrival word, so a consumer waits for one arrival per rank per launch. Reuse of a peer's
// row across steps needs no fence here: rank A writes into rank B's row in step s + 1 only after the step's
// all-reduces, which need B's step s + 1 contributions, which B issues only after its step s logits were consumed.
struct PersistGather {
    struct Args {
        uint64_t mc_base;     // the rows' multicast base
        uint64_t arrive_off;  // bytes from the base to the s32 arrival word
        uint32_t* grid_done;  // local u32, zeroed once: this launch's CTAs that finished their pushes
        uint32_t v_total;     // gathered row width (elements)
        uint32_t col0;        // this rank's first column in the row
    };

    // one 16-token round per 16 rows of the [kM, 64] tile (half of one at m 8); thread p's packets are columns 4p and
    // 32 + 4p, so a row's 8 threads cover 128 contiguous bytes per store (and per smem read)
    template <uint32_t kM>
    static __device__ __forceinline__ void tile(const Args& epi, const uint32_t w_outer, const uint32_t n_out,
                                                const float* out_tile, const uint32_t quad, const uint32_t lane_idx) {
        const uint32_t tid = quad * 32 + lane_idx;
        const uint32_t p = tid & 7u;
#pragma unroll
        for (uint32_t r = 0; r < kM; r += 16) {
            const uint32_t t = r + (tid >> 3);
            if (kM % 16 != 0 && t >= kM) break;
#pragma unroll
            for (uint32_t c = p * 4; c < 64; c += 32) {
                if (w_outer + c >= n_out) break;  // the boundary block's packets past the raw shard
                const uint4 v = *reinterpret_cast<const uint4*>(out_tile + t * 64 + c);
                const uint64_t off = (static_cast<uint64_t>(t) * epi.v_total + epi.col0 + w_outer + c) * sizeof(float);
                ptx::multimem_st_v4_bf16x2_weak(reinterpret_cast<uint32_t*>(epi.mc_base + off), v.x, v.y, v.z, v.w);
            }
        }
    }

    // one thread per CTA, after the CTA's pushes are system-fenced
    static __device__ __forceinline__ void arrive(const Args& epi) {
        if (atomicAdd(epi.grid_done, 1u) != gridDim.x - 1) return;
        *epi.grid_done = 0;
        __threadfence_system();  // the other CTAs' pushes, observed through the count, ordered ahead of the release
        ptx::multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(epi.mc_base + epi.arrive_off), 1u);
    }
};

// a non-TMA tile sink fences its stores system-wide before its CTA's arrival unless it declares kFenceSys = false (the
// pull sinks: the stores are the arrival, read by the peers' fabric reduce, and the counter is the next launch's)
template <class kEpi>
__device__ __forceinline__ constexpr bool persist_sink_fences_sys() {
    if constexpr (requires { kEpi::kFenceSys; })
        return kEpi::kFenceSys;
    else
        return true;
}

template <uint32_t kM, uint32_t kStages, uint32_t kPack, bool kEvictFirst, class kEpi>
__device__ __forceinline__ void dense_persist_body(const DensePersistArgs& args, const CUtensorMap& tmap_w,
                                                   const CUtensorMap& tmap_x, const typename kEpi::Args& epi,
                                                   const uint32_t cta = blockIdx.x, const uint32_t ctas = gridDim.x) {
    // ---- static geometry (BN = 64, ck 1) ---------------------------------------
    constexpr uint32_t BN = 64;
    constexpr uint32_t kStageBytes = BN * 128;
    constexpr uint32_t kXTile = kM * 128;
    constexpr uint32_t kNumSlots = kStages / kPack;
    constexpr uint32_t kAtoms = 4;
    constexpr uint32_t kTmemCols = dense_tmem_cols(kM);
    constexpr uint32_t kEpiBarId = block::kEpi;
    constexpr uint32_t kEpiThreads = kDenseRoles.n_epi * 32;
    static_assert((kM == 8 || kM % 16 == 0) && kM <= 64, "8- or 16k-row windows up to the m64 D tile's 64 columns");
    static_assert(kStages % kPack == 0 && kPack <= kStages / 2, "pack must divide the ring and be <= half of it");

    // ---- smem: the shared persist carve (note above) -----------------------
    constexpr DenseSmemLayout kSmem = dense_persist_smem_layout(kM, kPack, kStages);
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    const uint32_t smem_w = ptx::to_shared(smem_buf);
    const uint32_t smem_x = smem_w + kSmem.x_off;
    auto out_tile = reinterpret_cast<float*>(smem_buf + kSmem.out_off);
    auto tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_buf + kSmem.tmem_off);
    uint64_t* const bar_base = reinterpret_cast<uint64_t*>(smem_buf + kSmem.bar_off);
    uint64_t* const full_bars = bar_base;
    uint64_t* const empty_bars = bar_base + kNumSlots;
    uint64_t* const tmem_full_bar = bar_base + 2 * kNumSlots;
    uint64_t* const tmem_empty_bar = bar_base + 2 * kNumSlots + 1;

    const uint32_t warp_idx = ptx::warp_idx_sync();
    const uint32_t lane_idx = ptx::lane_idx();
    const uint32_t nb = args.num_blocks;

    const auto umin = [](const uint32_t& a, const uint32_t& b) { return a < b ? a : b; };

    // the pre-init W L2 prefetch of one ring for the first block; later blocks stream ahead through the live ring
    if (warp_idx == kDensePrefetch && ptx::elect_one_sync()) {
        const uint32_t kq = umin(kStages, args.stages_total);
        for (uint32_t q = 0; q < kq; ++q)
            dense_prefetch_w(&tmap_w, static_cast<int32_t>(q * 64), static_cast<int32_t>(cta * BN));
    }

    // ---- init (warp 1) + TMEM alloc (warp 3) ------------------------------
    if (warp_idx == kDenseGate && ptx::elect_one_sync()) {
        ptx::prefetch_tensormap(&tmap_w);
        ptx::prefetch_tensormap(&tmap_x);
        for (uint32_t i = 0; i < kNumSlots; ++i) {
            ptx::mbar_init(&full_bars[i], 1);
            ptx::mbar_init(&empty_bars[i], 1);
        }
        ptx::mbar_init(tmem_full_bar, 1);
        ptx::mbar_init(tmem_empty_bar, 1);
        ptx::fence_mbarrier_init_release_cluster();
    }
    if (warp_idx == kDenseTmem) ptx::tcgen05_alloc(ptx::to_shared(tmem_addr_storage), kTmemCols);
    __syncthreads();
    const uint32_t taddr = tmem_addr_storage[0];

    // =======================================================================
    // warp 1: x producer, behind the grid wait; the x k-loop re-issues per block (its tiles stay L2-resident across the
    // blocks, so the re-read costs L2 hits, not HBM bytes)
    // =======================================================================
    if (warp_idx == kDenseGate) {
        ptx::griddepcontrol_wait();
        if (ptx::elect_one_sync()) ptx::griddepcontrol_launch_dependents();
        uint32_t slot = 0, phase = 0;
        for (uint32_t b = cta; b < nb; b += ctas) {
            for (uint32_t s = 0; s < args.stages_total; s += kPack) {
                const uint32_t deliveries = umin(kPack, args.stages_total - s);
                ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d) {
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_x + (slot * kPack + d) * kXTile, &tmap_x,
                                                               static_cast<int32_t>((s + d) * 64), /*y=*/0,
                                                               &full_bars[slot], ptx::kEvictNormal);
                    }
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // =======================================================================
    // warp 0: W producer — the ring streams through block boundaries; the
    // elected lane's expect_tx covers the pack's W AND x bytes
    // =======================================================================
    if (warp_idx == kDenseWProducer) {
        uint32_t slot = 0, phase = 0;
        for (uint32_t b = cta; b < nb; b += ctas) {
            const uint32_t w_outer = b * BN;
            for (uint32_t s = 0; s < args.stages_total; s += kPack) {
                const uint32_t deliveries = umin(kPack, args.stages_total - s);
                ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d) {
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + (slot * kPack + d) * kStageBytes, &tmap_w,
                                                               static_cast<int32_t>((s + d) * 64),
                                                               static_cast<int32_t>(w_outer), &full_bars[slot],
                                                               kEvictFirst ? ptx::kEvictFirst : ptx::kEvictNormal);
                    }
                    ptx::mbar_arrive_expect_tx(&full_bars[slot], deliveries * (kStageBytes + kXTile));
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // =======================================================================
    // warp 2: MMA — single accumulator; block j opens after the epilogue's
    // drain of block j-1 (tmem_empty parity), which the ring absorbs
    // =======================================================================
    if (warp_idx == kDenseMma) {
        constexpr uint32_t idesc = ptx::mma_inst_desc_f16(BN, kM);
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_x);
        uint32_t slot = 0, phase = 0, j = 0;
        for (uint32_t b = cta; b < nb; b += ctas, ++j) {
            if (j >= 1) ptx::mbar_wait_parity_poll(tmem_empty_bar, (j - 1) & 1);
            ptx::tcgen05_fence_after_thread_sync();
            for (uint32_t s = 0; s < args.stages_total; s += kPack) {
                const uint32_t deliveries = umin(kPack, args.stages_total - s);
                ptx::mbar_wait_parity_poll(&full_bars[slot], phase);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one_sync()) {  // hoisted over the whole pack
                    for (uint32_t d = 0; d < deliveries; ++d) {
                        const uint32_t sl = slot * kPack + d;
#pragma unroll
                        for (uint32_t kk = 0; kk < kAtoms; ++kk) {
                            // +2 units = one MMA atom's K advance (32 B)
                            const uint64_t da = w_desc0 + sl * (kStageBytes / 16) + kk * 2;
                            const uint64_t db = x_desc0 + sl * (kXTile / 16) + kk * 2;
                            const uint32_t scale_c = (s > 0 || d > 0 || kk > 0) ? 1u : 0u;
                            ptx::tcgen05_mma_f16(taddr, da, db, idesc, scale_c);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&empty_bars[slot]);
                    if (s + kPack >= args.stages_total) ptx::tcgen05_commit_arrive(tmem_full_bar);
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // =======================================================================
    // warps 4..7: per-block drain -> [kM, 64] fp32 tile (unrounded) -> the
    // sink. The accumulator releases the moment the drain registers hold
    // it; the tile republishes only after the prior sink consumed it.
    // =======================================================================
    if (warp_idx >= kDenseEpi0) {
        const uint32_t quad = warp_idx - kDenseEpi0;
        // a sink with a prologue (kPrologue) runs it on the epilogue warps ahead of their first accumulator, under
        // the W ring's fill
        if constexpr (requires { kEpi::kPrologue; }) kEpi::prologue(epi, quad, lane_idx);
        uint32_t j = 0;
        for (uint32_t b = cta; b < nb; b += ctas, ++j) {
            const uint32_t w_outer = b * BN;
            ptx::mbar_wait_parity_poll(tmem_full_bar, j & 1);
            ptx::tcgen05_fence_after_thread_sync();
            uint32_t v[kM];
            ptx::tcgen05_ld_32x32b_cols<kM>(taddr, v);
            ptx::tcgen05_wait_ld();
            // every quadrant drained -> hand the accumulator back before the
            // convert/store tail (the MMA opens block j+1 under it)
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            if (b + ctas < nb && warp_idx == kDenseEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive(tmem_empty_bar);
            if (j >= 1) {
                // out-tile reuse gate: the committing thread read-waits the
                // previous block's store before the tile is rewritten
                if (warp_idx == kDenseEpi0 && ptx::elect_one_sync()) ptx::tma_store_wait_read();
                ptx::sync_aligned(kEpiBarId, kEpiThreads);
            }
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                for (uint32_t c = 0; c < kM; ++c)
                    out_tile[c * BN + row] = __int_as_float(v[c]);
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            // the sink consumes the tile (pad blocks past the raw out skip their packets)
            kEpi::template tile<kM>(epi, w_outer, args.n_out, out_tile, quad, lane_idx);
        }
        // every epi thread's pushes system-visible before the CTA's arrivals
        if constexpr (persist_sink_fences_sys<kEpi>()) __threadfence_system();
        ptx::sync_aligned(kEpiBarId, kEpiThreads);
        if (warp_idx == kDenseEpi0 && ptx::elect_one_sync()) kEpi::arrive(epi);
        if (warp_idx == kDenseEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);  // no relinquish
    }
}

template <uint32_t kM, uint32_t kStages, uint32_t kPack, bool kEvictFirst>
__launch_bounds__(256, 1) __global__
    void dense_persist_gather(const __grid_constant__ DensePersistArgs args, const __grid_constant__ CUtensorMap tmap_w,
                              const __grid_constant__ CUtensorMap tmap_x,
                              const __grid_constant__ PersistGather::Args epi) {
    dense_persist_body<kM, kStages, kPack, kEvictFirst, PersistGather>(args, tmap_w, tmap_x, epi);
}

}  // namespace mk

// Row operations of the layer: the fused residual + post-attention norm + pre-MoE norm (the all-reduce row), the
// pre-expert norm + dynamic NVFP4 quant of a pooled row, and the polled all-gather row reads. Every op is the explicit
// IEEE form (__fadd_rn, __fmul_rn, div.rn) in one fixed association order, so no contraction moves a result.

namespace mk::rowops {

// The reference norm's per-bf16x2 sum of squares: two fma.rn.f32.bf16 into a fresh 0.0f accumulator (the bf16 square
// is exact in fp32, so the pair contributes one rounded add; the caller chains the pairs).
static __device__ __forceinline__ float bf16x2_sumsq_fma(const nv_bfloat162& v) {
    float sum = 0.0f;
    ptx::fma_sq_rn_f32_bf16(sum, reinterpret_cast<const uint16_t&>(v.x));
    ptx::fma_sq_rn_f32_bf16(sum, reinterpret_cast<const uint16_t&>(v.y));
    return sum;
}

// sum / kN for a power-of-two kN: the reciprocal is exact, so the multiply rounds the same real quotient div.rn does
// (bitwise over all 2^32 inputs) without div.rn's FCHK guard and its out-of-line slow path.
template <uint32_t kN>
static __device__ __forceinline__ float mean_pow2(const float& sum) {
    static_assert(kN != 0 && (kN & (kN - 1)) == 0, "the reciprocal is exact only for a power of two");
    return __fmul_rn(sum, 1.0f / static_cast<float>(kN));
}

// The dynamic per-token NVFP4 scale of one 16-element group: token_scale = max(row_amax, 1e-10) / (6 x 448),
// gs = 1 / token_scale and gs_inv = token_scale itself (not 1 / gs); the group's e4m3 scale byte is
// e4m3(max(amax, 1e-10) / 6 x gs) and the returned factor multiplies the values before the e2m1 rounding.
static __device__ __forceinline__ float nvfp4_out_scale_dyn(const float& vec_max, const float& gs, const float& gs_inv,
                                                            uint8_t& sf_byte) {
    const float per_block_scale = __fmul_rn(fmaxf(vec_max, 1e-10f), 1.0f / 6.0f);
    const auto sf_bits = __nv_cvt_float_to_fp8(__fmul_rn(per_block_scale, gs), __NV_SATFINITE, __NV_E4M3);
    sf_byte = static_cast<uint8_t>(sf_bits);
    const float sf_val = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf_bits, __NV_E4M3)));
    const float eff = __fmul_rn(sf_val, gs_inv);
    return __fdiv_rn(1.0f, fmaxf(eff, 1e-10f));
}

// The pre-expert norm + dynamic NVFP4 quant of a 4096-wide row on 256 threads (the reference's cta_wide layout: thread
// t owns 16 columns = one scale group). The thread's column slice; the row and gamma fragments load apart from the
// math so a caller looping over rows can prefetch the next row while this one reduces.
__device__ __forceinline__ uint32_t dyn_wide_col0(const uint32_t& warp8, const uint32_t& lane_idx) {
    return ((warp8 / 4) * 128u + (warp8 % 4) * 32u + lane_idx) * 16u;
}
__device__ __forceinline__ void dyn_wide_load_gamma(const nv_bfloat16* gamma, const uint32_t& col_0, uint4 (&gam)[2]) {
    gam[0] = __ldg(reinterpret_cast<const uint4*>(gamma + col_0));
    gam[1] = __ldg(reinterpret_cast<const uint4*>(gamma + col_0 + 8));
}

// The pre-expert norm on preloaded fragments (frag = the row's 16 bf16, gam = gamma's): the thread's 16 normed bf16
// and their fp32 amax (an order-free fmaxf chain). The cross-warp reduction is the reference's tree
// ((w0 + w4) + (w2 + w6)) + ((w1 + w5) + (w3 + w7)). `staging` = 8 floats.
template <uint32_t kHidden>
static __device__ __forceinline__ void dyn_wide_norm_frag(const uint32_t& warp8, const uint32_t& lane_idx,
                                                          const uint4 (&frag)[2], const uint4 (&gam)[2],
                                                          const float& eps, nv_bfloat162 (&nrm)[8], float& vec_max,
                                                          float* staging, const uint32_t& bar_idx) {
    static_assert(kHidden == 4096, "the reference's 256-thread row (H 4096)");
    float acc = 0.0f;
#pragma unroll
    for (uint32_t v = 0; v < 2; ++v) {
        const auto xb = reinterpret_cast<const nv_bfloat162*>(&frag[v]);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e)
            acc = __fadd_rn(acc, bf16x2_sumsq_fma(xb[e]));
    }
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, offset));

    ptx::sync_aligned(bar_idx, 256);
    if (lane_idx == 0) staging[warp8] = acc;
    ptx::sync_aligned(bar_idx, 256);
    const float total = __fadd_rn(__fadd_rn(__fadd_rn(staging[0], staging[4]), __fadd_rn(staging[2], staging[6])),
                                  __fadd_rn(__fadd_rn(staging[1], staging[5]), __fadd_rn(staging[3], staging[7])));
    const float norm_factor = rsqrtf(__fadd_rn(mean_pow2<kHidden>(total), eps));

    vec_max = 0.0f;
#pragma unroll
    for (uint32_t v = 0; v < 2; ++v) {
        const auto xb = reinterpret_cast<const nv_bfloat162*>(&frag[v]);
        const auto gb = reinterpret_cast<const nv_bfloat162*>(&gam[v]);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
            const float2 xf = __bfloat1622float2(xb[e]);
            nrm[v * 4 + e] = __hmul2(
                __float22bfloat162_rn(make_float2(__fmul_rn(xf.x, norm_factor), __fmul_rn(xf.y, norm_factor))), gb[e]);
            const float2 f = __bfloat1622float2(nrm[v * 4 + e]);
            vec_max = fmaxf(vec_max, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
}

// The norm, the row amax (order-free), token_scale and the group's quant: 8 B of e2m1 codes (low nibble first) and
// one e4m3 scale byte. Returns token_scale (the row's x_scale). `staging` = 16 floats.
template <uint32_t kHidden>
static __device__ __forceinline__ float row_pre_expert_norm_quant_nvfp4_dyn_wide_frag(
    const uint32_t& warp8, const uint32_t& lane_idx, const uint4 (&frag)[2], const uint4 (&gam)[2], const float& eps,
    uint8_t* q_out, uint8_t* sf_out, float* staging, const uint32_t& bar_idx) {
    const uint32_t col_0 = dyn_wide_col0(warp8, lane_idx);
    alignas(16) nv_bfloat162 nrm[8];
    float vec_max;
    dyn_wide_norm_frag<kHidden>(warp8, lane_idx, frag, gam, eps, nrm, vec_max, staging, bar_idx);
    // the row amax (order-free fmaxf chains): warp butterfly, staged, reduced
    float row_max = vec_max;
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, offset));
    if (lane_idx == 0) staging[8 + warp8] = row_max;
    ptx::sync_aligned(bar_idx, 256);
#pragma unroll
    for (uint32_t w = 0; w < 8; ++w)
        row_max = fmaxf(row_max, staging[8 + w]);
    const float token_scale = __fmul_rn(fmaxf(row_max, 1e-10f), 1.0f / (6.0f * 448.0f));
    const float gs = __fdiv_rn(1.0f, token_scale);

    uint8_t sf_byte;
    const float out_scale = nvfp4_out_scale_dyn(vec_max, gs, token_scale, sf_byte);
    uint32_t packed[2];
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
        packed[h] = 0;
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(nrm[h * 4 + e]);
            packed[h] |= ptx::cvt_pack_f32_to_e2m1x2(__fmul_rn(f.x, out_scale), __fmul_rn(f.y, out_scale)) << (e * 8);
        }
    }
    *reinterpret_cast<uint2*>(q_out + col_0 / 2) = make_uint2(packed[0], packed[1]);
    sf_out[col_0 / 16] = sf_byte;
    return token_scale;
}

// The bf16 MoE layers' row: the same norm, its normed bf16 stored as the expert input (no quant, no token scale).
template <uint32_t kHidden>
static __device__ __forceinline__ void row_pre_expert_norm_bf16_dyn_wide_frag(
    const uint32_t& warp8, const uint32_t& lane_idx, const uint4 (&frag)[2], const uint4 (&gam)[2], const float& eps,
    nv_bfloat16* x_out, float* staging, const uint32_t& bar_idx) {
    const uint32_t col_0 = dyn_wide_col0(warp8, lane_idx);
    alignas(16) nv_bfloat162 nrm[8];
    float vec_max;
    dyn_wide_norm_frag<kHidden>(warp8, lane_idx, frag, gam, eps, nrm, vec_max, staging, bar_idx);
    reinterpret_cast<uint4*>(x_out + col_0)[0] = *reinterpret_cast<const uint4*>(&nrm[0]);
    reinterpret_cast<uint4*>(x_out + col_0)[1] = *reinterpret_cast<const uint4*>(&nrm[4]);
}

// The all-gather row reads. A gathered row lives in a symmetric buffer whose 16-B packets each rank fills; a reader
// polls each packet at system scope until no bf16 half carries the empty mark, then re-marks it empty for the row's
// next use (one reader per packet). The mark is an all-ones half, never a cvt.rn result (a NaN rounds to 0x7FFF).
inline constexpr uint16_t kAgEmpty = 0xFFFFu;
// The row polls' budget (this one and the fp32 all-reduce consume below): a peer that never pushes traps instead of
// hanging the step. A minute, because eager warm-ups skew ranks by seconds.
inline constexpr uint64_t kRowPollBudgetNs = 60000000000;
static __device__ __forceinline__ bool ag_packet_has_empty(const uint4& v) {
    const auto h = reinterpret_cast<const uint16_t*>(&v);
    bool e = false;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
        e |= h[i] == kAgEmpty;
    return e;
}
static __device__ __forceinline__ uint4 ag_poll_take_u4(const uint4* p) {
    uint4 v;
    uint64_t t0 = 0;
    for (uint32_t polls = 0;; ++polls) {
        v = ptx::ld_relaxed_sys_u4(p);
        if (!ag_packet_has_empty(v)) break;
        if ((polls & 1023u) == 0) {
            const uint64_t now = ptx::globaltimer_ns();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > kRowPollBudgetNs) {
                ptx::trap();
            }
        }
    }
    ptx::st_row_u4(const_cast<uint4*>(p), make_uint4(~0u, ~0u, ~0u, ~0u));
    return v;
}
// kN packets at once: every packet's first load in flight before any spin (one system-scope round trip, not kN), then
// only the packets still carrying the empty mark are re-polled.
template <uint32_t kN>
static __device__ __forceinline__ void ag_poll_take_u4_n(const uint4* const (&p)[kN], const bool (&live)[kN],
                                                         uint4 (&v)[kN]) {
#pragma unroll
    for (uint32_t i = 0; i < kN; ++i)
        if (live[i]) v[i] = ptx::ld_relaxed_sys_u4(p[i]);
#pragma unroll
    for (uint32_t i = 0; i < kN; ++i)
        if (live[i] && ag_packet_has_empty(v[i]))
            v[i] = ag_poll_take_u4(p[i]);
        else if (live[i])
            ptx::st_row_u4(const_cast<uint4*>(p[i]), make_uint4(~0u, ~0u, ~0u, ~0u));
}

// The residual + norm row's per-thread operands (every owned virtual warp's pass-2 streams and both norm weights),
// loaded AHEAD of the row's wait by an item whose branch row is the only input that must follow the wait.
template <uint32_t kPer>
struct RowNormOperands {
    uint4 wv4[kPer];
    float4 sv[kPer][2], av[kPer][2], btv[kPer][2];
    uint4 wv[kPer];
};

// Load a row's operands ahead of the wait (the per-layer parameters are cold; only the branch row must follow it):
// one 256-bit load per stream per owned virtual warp.
template <uint32_t kHidden, uint32_t kWarps>
static __device__ __forceinline__ void row_norm_preload_operands(const uint32_t& warp_idx, const uint32_t& lane_idx,
                                                                 const float* stream_row, const float* alpha,
                                                                 const float* beta, const nv_bfloat16* pre_w,
                                                                 const nv_bfloat16* post_w,
                                                                 RowNormOperands<(32 + kWarps - 1) / kWarps>& o) {
    constexpr uint32_t kVws = 32;
    constexpr uint32_t kPer = (kVws + kWarps - 1) / kWarps;
    const auto vw_of = [&](const uint32_t& k) { return warp_idx + kWarps * k; };
    const auto has_vw = [&](const uint32_t& k) { return kVws % kWarps == 0 || vw_of(k) < kVws; };
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) {
            const uint32_t col_0 = (vw_of(k) * 32 + lane_idx) * 8;
            o.wv4[k] = __ldg(reinterpret_cast<const uint4*>(post_w + col_0));
            ptx::ld_row_u8_nc(reinterpret_cast<const uint4*>(stream_row + col_0), reinterpret_cast<uint4&>(o.sv[k][0]),
                              reinterpret_cast<uint4&>(o.sv[k][1]));
            ptx::ld_row_u8_nc(reinterpret_cast<const uint4*>(alpha + col_0), reinterpret_cast<uint4&>(o.av[k][0]),
                              reinterpret_cast<uint4&>(o.av[k][1]));
            ptx::ld_row_u8_nc(reinterpret_cast<const uint4*>(beta + col_0), reinterpret_cast<uint4&>(o.btv[k][0]),
                              reinterpret_cast<uint4&>(o.btv[k][1]));
            o.wv[k] = __ldg(reinterpret_cast<const uint4*>(pre_w + col_0));
        }
}

// The residual + post-FFN norm + next layer's pre-attention norm row at H 8192, bitwise the reference's 1024-thread
// kernel:
//   v         = x x post_w                                 x = the branch row (bf16), the weighted input in fp32
//   post_rstd = rsqrtf(sum(v^2) / H + post_eps)
//   ns        = fmaf(alpha, stream, beta x (v x post_rstd)) -> ns_row (the new residual stream, fp32)
//   pre_rstd  = rsqrtf(sum(ns^2) / H + pre_eps)
//   out       = bf16((ns x pre_rstd) x pre_w)              -> out_row (bf16)
// kWarps real warps emulate the reference's 32 virtual warps (vw = warp + kWarps x k). The branch is an all-gather
// row, polled per packet (above); every other operand was preloaded by row_norm_preload_operands. ns stays in
// registers into pass 3, and the reduction tree is the reference's verbatim (per-vw lane chains, the descending
// 16..1 butterfly, partials at staging[vw], a warp-0 combine, smem broadcast). `staging` = 33 floats shared by the
// kWarps x 32 threads, synchronized on `bar_idx`. dry: the row's instructions on a zero branch, without the poll or the
// stores (the instruction cache's warm pass).
template <uint32_t kHidden, uint32_t kWarps>
static __device__ __forceinline__ void row_norm_residual(const uint32_t& warp_idx, const uint32_t& lane_idx,
                                                         const nv_bfloat16* branch_row, const float& post_eps,
                                                         const float& pre_eps, float* ns_row, nv_bfloat16* out_row,
                                                         float* staging, const uint32_t& bar_idx,
                                                         const RowNormOperands<(32 + kWarps - 1) / kWarps>* pre,
                                                         const bool dry) {
    static_assert(kHidden == 8192, "the reference's 1024-thread row (H 8192)");
    constexpr uint32_t kVws = 32;  // reference virtual warps
    static_assert(kWarps >= 2 && kWarps <= kVws, "multi-vw emulation range");
    constexpr uint32_t kPer = (kVws + kWarps - 1) / kWarps;  // owned vws (max)
    constexpr uint32_t kThreads = kWarps * 32;
    const auto vw_butterfly = [&](float acc) {
#pragma unroll
        for (uint32_t offset = 16; offset >= 1; offset >>= 1)
            acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, offset));
        return acc;
    };
    // the warp-0 combine of the staged per-vw partials, broadcast through staging[32]
    const auto vw_combine = [&]() {
        ptx::sync_aligned(bar_idx, kThreads);
        if (warp_idx == 0) {
            const float total = vw_butterfly(staging[lane_idx]);
            if (lane_idx == 0) staging[32] = total;
        }
        ptx::sync_aligned(bar_idx, kThreads);
        return staging[32];
    };
    // vw_of(k) is warp-uniform, so every owned-vw guard below stays convergent; the modulus test folds the guard away
    // when kWarps divides 32 (the 8-warp variant)
    const auto vw_of = [&](const uint32_t& k) { return warp_idx + kWarps * k; };
    const auto has_vw = [&](const uint32_t& k) { return kVws % kWarps == 0 || vw_of(k) < kVws; };

    // the branch packets (pass 1's only operand) first, every owned vw in flight before the first barrier
    uint4 bv[kPer] = {}, wv4[kPer];
    {
        const uint4* bps[kPer];
        bool live[kPer];
#pragma unroll
        for (uint32_t k = 0; k < kPer; ++k) {
            live[k] = has_vw(k) && !dry;
            bps[k] = reinterpret_cast<const uint4*>(branch_row + (vw_of(k) * 32 + lane_idx) * 8);
        }
        ag_poll_take_u4_n<kPer>(bps, live, bv);
    }
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) wv4[k] = pre->wv4[k];
    float4 sv[kPer][2], av[kPer][2], btv[kPer][2];
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) {
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
                sv[k][c] = pre->sv[k][c];
                av[k][c] = pre->av[k][c];
                btv[k][c] = pre->btv[k][c];
            }
        }

    // Pass 1: post-norm sum of squares per owned vw (weighted v = x * w in fp32)
    float part[kPer];
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) {
            const auto xb = reinterpret_cast<const nv_bfloat162*>(&bv[k]);
            const auto xw = reinterpret_cast<const nv_bfloat162*>(&wv4[k]);
            float acc = 0.0f;
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                float2 f = __bfloat1622float2(xb[e]);
                const float2 w2 = __bfloat1622float2(xw[e]);
                f.x = __fmul_rn(f.x, w2.x);
                f.y = __fmul_rn(f.y, w2.y);
                acc = fmaf(f.x, f.x, acc);
                acc = fmaf(f.y, f.y, acc);
            }
            part[k] = vw_butterfly(acc);
        }
    // Staging stores hoisted out of the vw loop: a lane test inside the unrolled loop makes ptxas branch-split the
    // warp and duplicate the butterflies with a WARPSYNC collective per SHFL; the diverged region must hold only the
    // plain stores.
    if (lane_idx == 0)
#pragma unroll
        for (uint32_t k = 0; k < kPer; ++k)
            if (has_vw(k)) staging[vw_of(k)] = part[k];
    const float post_rstd = rsqrtf(__fadd_rn(__fdiv_rn(vw_combine(), static_cast<float>(kHidden)), post_eps));

    // Pass 2: gated residual add + new-stream store + pre-norm sum of squares (all operands resident)
    float4 ns[kPer][2];
    uint4 wv[kPer];
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) {
            const uint32_t col_0 = (vw_of(k) * 32 + lane_idx) * 8;
            const auto xb = reinterpret_cast<const nv_bfloat162*>(&bv[k]);
            const auto xw = reinterpret_cast<const nv_bfloat162*>(&wv4[k]);
            float acc = 0.0f;
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    float raw =
                        (e & 1u) ? __bfloat162float(xb[c * 2 + e / 2].y) : __bfloat162float(xb[c * 2 + e / 2].x);
                    // normed = v * rstd; ns = fmaf(alpha, s, beta * normed)
                    const nv_bfloat162 w2 = xw[c * 2 + e / 2];
                    raw = __fmul_rn(raw, __bfloat162float((e & 1u) ? w2.y : w2.x));
                    const float ns_val = fmaf((&av[k][c].x)[e], (&sv[k][c].x)[e],
                                              __fmul_rn((&btv[k][c].x)[e], __fmul_rn(raw, post_rstd)));
                    (&ns[k][c].x)[e] = ns_val;
                    acc = fmaf(ns_val, ns_val, acc);
                }
            }
            if (!dry)
                ptx::st_row_u8(reinterpret_cast<uint4*>(ns_row + col_0), reinterpret_cast<const uint4&>(ns[k][0]),
                               reinterpret_cast<const uint4&>(ns[k][1]));
            wv[k] = pre->wv[k];
            part[k] = vw_butterfly(acc);
        }
    // same hoist as pass 1 (see the divergence note there)
    if (lane_idx == 0)
#pragma unroll
        for (uint32_t k = 0; k < kPer; ++k)
            if (has_vw(k)) staging[vw_of(k)] = part[k];
    const float pre_rstd = rsqrtf(__fadd_rn(__fdiv_rn(vw_combine(), static_cast<float>(kHidden)), pre_eps));

    // Pass 3: pre-norm from the register-resident new stream
#pragma unroll
    for (uint32_t k = 0; k < kPer; ++k)
        if (has_vw(k)) {
            const uint32_t col_0 = (vw_of(k) * 32 + lane_idx) * 8;
            const auto wb = reinterpret_cast<const nv_bfloat162*>(&wv[k]);
            alignas(16) nv_bfloat162 ov[4];
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const float v = __fmul_rn(
                        __fmul_rn((&ns[k][c].x)[e], pre_rstd),
                        (e & 1u) ? __bfloat162float(wb[c * 2 + e / 2].y) : __bfloat162float(wb[c * 2 + e / 2].x));
                    const nv_bfloat16 o = __float2bfloat16(v);
                    if (e & 1u)
                        ov[c * 2 + e / 2].y = o;
                    else
                        ov[c * 2 + e / 2].x = o;
                }
            }
            if (!dry) *reinterpret_cast<uint4*>(out_row + col_0) = *reinterpret_cast<const uint4*>(&ov[0]);
        }
}

// The all-reduce + norm row pieces: the push-lamport all-reduce's consume half per 16 B packet and the
// post-attention norm-residual row, sliced across the AR row's CTAs, with the reduced x register-resident and the
// two rstd rendezvous done in shared / cluster memory.

// Lamport marker law: the push half rewrites every +0 (0x0000) element to -0 (0x8000) before storing, so +0 doubles
// as the "slot empty" marker the poll spins on. The consume never unmarks: an all-zero element legitimately sums to
// -0. The norm is not neutral to that sign in general (pass 2's fmaf(alpha, stream, (beta x rstd) x raw) exposes it
// when alpha x stream is also +-0), so the consume reads the marked values exactly.

inline constexpr uint16_t kBf16PosZero = 0;      // the "slot empty" marker
inline constexpr uint16_t kBf16NegZero = 32768;  // a pushed zero

// Any +0 bf16 element in a 16 B packet (8 x bf16) — the poll predicate.
static __device__ __forceinline__ bool ar_packet8_has_pos_zero(const uint4& v) {
    const auto h = reinterpret_cast<const uint16_t*>(&v);
    bool z = false;
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        z |= (h[e] == kBf16PosZero);
    return z;
}

// The push half's marking (clear_pos_zero): every +0 element -> -0.
static __device__ __forceinline__ void ar_lamport_mark_packet8(uint4& v) {
    const auto h = reinterpret_cast<uint16_t*>(&v);
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        if (h[e] == kBf16PosZero) h[e] = kBf16NegZero;
}

// The consume: polls the kRanks ranks' packets until none carries the empty marker, then sums them in rank order.
// The bounded variant's report names the packet and the ranks that arrived (bit r = rank r's packet landed).
template <uint32_t kRanks>
static __device__ __forceinline__ uint4 ar_lamport_consume_packet8(const nv_bfloat16* const (&plane_rows)[kRanks],
                                                                   const uint32_t& col_0, const uint32_t& rank) {
    uint4 pv[kRanks];
    const auto arrived = [&] {
        bool has_zero = false;
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            pv[r] = ptx::ld_relaxed_sys_u4(reinterpret_cast<const uint4*>(plane_rows[r] + col_0));
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            has_zero |= ar_packet8_has_pos_zero(pv[r]);
        return !has_zero;
    };
    ptx::bounded_spin(arrived, "AR lamport", int(rank), int(col_0 / 8), (1u << kRanks) - 1u, [&] {
        uint32_t landed = 0;
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            landed |= uint32_t(!ar_packet8_has_pos_zero(pv[r])) << r;
        return landed;
    });
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

// the consume's marker restore: st.global.v4 of +0 markers into every rank's slot, re-establishing "empty" for the
// phase's next use
template <uint32_t kRanks>
static __device__ __forceinline__ void ar_lamport_restore_packet8(nv_bfloat16* const (&restore_rows)[kRanks],
                                                                  const uint32_t& col_0) {
    const uint4 zeros = {0u, 0u, 0u, 0u};  // 8 x +0 bf16
#pragma unroll
    for (uint32_t r = 0; r < kRanks; ++r)
        ptx::st_row_u4(reinterpret_cast<uint4*>(restore_rows[r] + col_0), zeros);
}

// The fp32 variants for the K-window variant's partial planes (the pooled and logits partials): a packet is 4 fp32, the
// marker +0.0f per word (the push rewrites +0.0f to -0.0f, the poll spins while any word is +0), the consume sums the
// kRanks packets in rank order with __fadd_rn (identical on every rank), the restore writes 16 B of +0 into the
// consumer's own copies (no other rank polls a rank's copy). A peer that never pushes traps the consume after 60 s,
// with no report. The bound is the exception to ptx.cuh's order (a cross-rank
// bound under the ticket bound): the o_proj all-reduce's 30 s wait, earlier in the same layer, names a dead peer
// first; a peer that dies between its two pushes traps an in-rank ticket wait first.
static __device__ __forceinline__ bool ar_packet4f_has_pos_zero(const uint4& v) {
    return v.x == 0u || v.y == 0u || v.z == 0u || v.w == 0u;
}
static __device__ __forceinline__ void ar_lamport_mark_packet4f(uint4& v) {
    if (v.x == 0u) v.x = 0x80000000u;
    if (v.y == 0u) v.y = 0x80000000u;
    if (v.z == 0u) v.z = 0x80000000u;
    if (v.w == 0u) v.w = 0x80000000u;
}
template <uint32_t kRanks>
static __device__ __forceinline__ void ar_lamport_push_packet4f(float* const (&dst_rows)[kRanks], const uint32_t& col_0,
                                                                uint4 v) {
    ar_lamport_mark_packet4f(v);
#pragma unroll
    for (uint32_t r = 0; r < kRanks; ++r)
        ptx::st_relaxed_sys_u4(reinterpret_cast<uint4*>(dst_rows[r] + col_0), v);
}
template <uint32_t kRanks>
static __device__ __forceinline__ float4 ar_lamport_consume_packet4f(const float* const (&plane_rows)[kRanks],
                                                                     const uint32_t& col_0) {
    uint4 pv[kRanks];
    uint64_t t0 = 0;
    for (uint32_t polls = 0;; ++polls) {
        bool has_zero = false;
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            pv[r] = ptx::ld_relaxed_sys_u4(reinterpret_cast<const uint4*>(plane_rows[r] + col_0));
#pragma unroll
        for (uint32_t r = 0; r < kRanks; ++r)
            has_zero |= ar_packet4f_has_pos_zero(pv[r]);
        if (!has_zero) break;
        if ((polls & 1023u) == 0) {
            const uint64_t now = ptx::globaltimer_ns();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > kRowPollBudgetNs) {
                ptx::trap();
            }
        }
    }
    float4 acc = *reinterpret_cast<const float4*>(&pv[0]);
#pragma unroll
    for (uint32_t r = 1; r < kRanks; ++r) {
        const float4 f = *reinterpret_cast<const float4*>(&pv[r]);
        acc.x = __fadd_rn(acc.x, f.x);
        acc.y = __fadd_rn(acc.y, f.y);
        acc.z = __fadd_rn(acc.z, f.z);
        acc.w = __fadd_rn(acc.w, f.w);
    }
    return acc;
}
template <uint32_t kRanks>
static __device__ __forceinline__ void ar_lamport_restore_packet4f(float* const (&restore_rows)[kRanks],
                                                                   const uint32_t& col_0) {
    const uint4 zeros = {0u, 0u, 0u, 0u};  // 4 x +0.0f
#pragma unroll
    for (uint32_t r = 0; r < kRanks; ++r)
        ptx::st_row_u4(reinterpret_cast<uint4*>(restore_rows[r] + col_0), zeros);
}

// the per-vw descending shfl_xor butterfly
static __device__ __forceinline__ float arnorm_vw_butterfly(float acc) {
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, offset));
    return acc;
}

// one lane's pass-1 sum of squares from the downcast bf16 branch packet: the reference norm reads bf16, so the row
// squares the rounded values, never the fp32 accumulators
static __device__ __forceinline__ float arnorm_pass1_sumsq(const uint4& bv) {
    const auto xb = reinterpret_cast<const nv_bfloat162*>(&bv);
    float acc = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(xb[e]);
        acc = fmaf(f.x, f.x, acc);
        acc = fmaf(f.y, f.y, acc);
    }
    return acc;
}

// The consume-fed row (four CTAs per row, one packet per thread): the reference norm's tree, with the pass-2
// operands and pass 3's weight preloaded above the lamport poll (their latency hides under the arrival spin) and the
// two rendezvous done by `round(r, part)`, which stages the 32 vw partials and returns their butterfly in the same
// order (bitwise the gathered variant). The restore stays at the caller's tail.
template <uint32_t kHidden, uint32_t kRanks, class Round>
static __device__ __forceinline__ void row_arnorm_cn4_row(
    const uint32_t& vw, const uint32_t& lane_idx, const uint32_t& rank, const nv_bfloat16* const (&plane_rows)[kRanks],
    const float* stream_row, const float* alpha, const float* beta, const nv_bfloat16* pre_w, const float& post_eps,
    const float& pre_eps, float* ns_row, nv_bfloat16* out_row, float* router_row, Round&& round) {
    static_assert(kHidden == 8192, "the reference's 1024-thread row (H 8192)");
    const uint32_t col_0 = (vw * 32 + lane_idx) * 8;

    // Pass-2 operand preload ABOVE the poll (lands under the arrival spin)
    float4 sv[2], av[2], btv[2];
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
        sv[c] = __ldg(reinterpret_cast<const float4*>(stream_row + col_0 + c * 4));
        av[c] = __ldg(reinterpret_cast<const float4*>(alpha + col_0 + c * 4));
        btv[c] = __ldg(reinterpret_cast<const float4*>(beta + col_0 + c * 4));
    }
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(pre_w + col_0));  // pass 3's weight, under the spin too
    const uint4 bv = ar_lamport_consume_packet8<kRanks>(plane_rows, col_0, rank);

    // Pass 1: post-norm sum of squares from the downcast bf16 branch values
    const float post_rstd =
        rsqrtf(__fadd_rn(mean_pow2<kHidden>(round(0, arnorm_vw_butterfly(arnorm_pass1_sumsq(bv)))), post_eps));

    // Pass 2: gated residual add + new-stream store + pre-norm sum of squares
    const auto xb = reinterpret_cast<const nv_bfloat162*>(&bv);
    float4 ns[2];
    float acc = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
            const float raw = (e & 1u) ? __bfloat162float(xb[c * 2 + e / 2].y) : __bfloat162float(xb[c * 2 + e / 2].x);
            const float ns_val =
                fmaf((&av[c].x)[e], (&sv[c].x)[e], __fmul_rn(__fmul_rn((&btv[c].x)[e], post_rstd), raw));
            (&ns[c].x)[e] = ns_val;
            acc = fmaf(ns_val, ns_val, acc);
        }
        *reinterpret_cast<float4*>(ns_row + col_0 + c * 4) = ns[c];
    }
    const float pre_rstd = rsqrtf(__fadd_rn(mean_pow2<kHidden>(round(1, arnorm_vw_butterfly(acc))), pre_eps));

    // Pass 3: pre-norm from the REGISTER-resident new stream (owned slice)
    const auto wb = reinterpret_cast<const nv_bfloat162*>(&wv);
    alignas(16) nv_bfloat162 ov[4];
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
            const float v =
                __fmul_rn(__fmul_rn((&ns[c].x)[e], pre_rstd),
                          (e & 1u) ? __bfloat162float(wb[c * 2 + e / 2].y) : __bfloat162float(wb[c * 2 + e / 2].x));
            const nv_bfloat16 o = __float2bfloat16(v);
            if (e & 1u)
                ov[c * 2 + e / 2].y = o;
            else
                ov[c * 2 + e / 2].x = o;
        }
    }
    *reinterpret_cast<uint4*>(out_row + col_0) = *reinterpret_cast<const uint4*>(&ov[0]);
    if (router_row != nullptr) {
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            float4 rv;
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e)
                (&rv.x)[e] = (e & 1u) ? __bfloat162float(ov[c * 2 + e / 2].y) : __bfloat162float(ov[c * 2 + e / 2].x);
            *reinterpret_cast<float4*>(router_row + col_0 + c * 4) = rv;
        }
    }
}

}  // namespace mk::rowops

// Ticketed counts: how one phase of the launch hands work to another without a grid-wide barrier. A producer item
// publishes ONE monotone count word (+N arrivals per launch, never reset); a consumer waits for N x (ticket + 1), its
// CTA's own launch ticket, with a wrap-safe signed compare. No reset and no rounding of a read: replay-safe by
// construction.

namespace mk::handoff {

// The item's named barrier: warps 4..11 (id 3; the dense epilogue's id 2 runs on warps 4..7 at 128 threads).
constexpr uint32_t kItemBarId = block::kItem;
constexpr uint32_t kItemWarps = block::kItemWarps;
static_assert(kItemBarId == 3 && kItemWarps == 8, "the item barrier and warps are the framework's");
constexpr uint32_t kItemThreads = kItemWarps * 32;
// An item's arrivals per launch are a power of two (the count's width invariant): the one at or above n.
constexpr uint32_t pow2_at_least(uint32_t n) {
    uint32_t p = 1;
    while (p < n)
        p *= 2;
    return p;
}
// PUBLISH: every item thread fences the async proxy after its own last generic store, then the item barrier, then ONE
// elected thread releases the word. Consumers read these stores through TMA (the async proxy) and a barrier orders only
// the generic proxy, so the fence sits on each WRITING thread's path, never once after the barrier.
// `weight`: the arrivals this CTA publishes (an item whose owner count is not a power of two has its first owners
// publish 2, so the word still gains a power of two per launch).
__device__ __forceinline__ void item_publish(uint32_t* word, const bool elected, const uint32_t weight = 1u) {
    ptx::fence_async_global();
    ptx::sync_aligned(kItemBarId, kItemThreads);
    if (elected) {
        if (weight == 1u)
            ptx::red_add_rel_b32(word, 1u);
        else
            ptx::red_add_rel_b32(word, weight);
    }
}

// WAIT on the executing lane: the wrap-safe count spin to this launch's target, then the async-proxy fence so the
// caller's TMA reads see the published bytes.
__device__ __forceinline__ void ticket_wait(const uint32_t* word, const uint32_t target) {
    ptx::count_spin(word, target);
    ptx::fence_async_global();
}

}  // namespace mk::handoff

// The per-head qk-norm + rope of the qkv epilogue, bitwise the reference decode kernels.

namespace mk::rowops {

// One head (64 elements) on one warp, lane l holding the bf16 pair (2l, 2l + 1):
//   qk-norm: sum = fma(x, x, 0) then fma(y, y, sum), the descending 16..1 butterfly,
//            nf = scale x rsqrt.approx(fma(sum, 1 / 64, eps)), out = bf16(x x nf, y x nf)
//            (scale = q_scale on q heads, 1 on k heads; the norm is weightless)
//   rope:    GPT-J interleaved on the re-loaded bf16 (the reference round-trips through bf16 between its two
//            kernels), cos / sin from the fp32 row [cos 32 | sin 32] at lane l:
//            out_x = fma(x, cos, -(y x sin)), out_y = fma(y, cos, x x sin)
// Every operation is explicitly rounded, so the result does not depend on contraction in the caller.
template <bool kHasRope = true>
static __device__ __forceinline__ nv_bfloat162
head_pair_qknorm_rope(const nv_bfloat162 pv,  // this lane's bf16 pair (2l, 2l+1)
                      const float scale,      // q_scale on q heads, 1.0f on k heads
                      const float eps,        // rms_norm_eps (norm is weightless)
                      const float c,          // cos_sin_row[lane] (unused w/o rope)
                      const float s) {        // cos_sin_row[32 + lane]
    const float2 f = __bfloat1622float2(pv);
    float acc = fmaf(f.x, f.x, 0.0f);
    acc = fmaf(f.y, f.y, acc);
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, offset));
    const float nf = __fmul_rn(scale, ptx::rsqrt_approx(__fmaf_rn(acc, 1.0f / 64.0f, eps)));
    nv_bfloat162 out = __float22bfloat162_rn(make_float2(__fmul_rn(f.x, nf), __fmul_rn(nf, f.y)));
    if constexpr (kHasRope) {
        const float2 n = __bfloat1622float2(out);  // the rope reads the stored bf16
        out = __float22bfloat162_rn(
            make_float2(__fmaf_rn(n.x, c, -__fmul_rn(n.y, s)), __fmaf_rn(n.y, c, __fmul_rn(n.x, s))));
    }
    return out;
}

}  // namespace mk::rowops

// The qkv projection's rowops epilogue (the dense executor's tile hook): on each 64-column block of the projected
// [q | k | v | og] row, the qk-norm + q scale + rope + the global layers' per-token q temperature, in place on the
// bf16 tile, and the k / v rows appended to the paged KV pool. Bitwise the reference's qkv GEMM followed by its rowops
// kernel (the tile is rounded to bf16 first, as the reference's GEMM output). NoPE layers pass identity cos / sin.

namespace mk {

template <uint32_t kNumQHeads, uint32_t kNumKVHeads>
struct RowEpiQkvRowops {
    struct Args {
        const float* cos_sin;      // [m, 64] fp32 {cos[32]|sin[32]} per token
        const int32_t* cache_loc;  // [m] KV pool slots
        uint8_t* k_cache;
        uint8_t* v_cache;
        const float* temp_scale;  // [m] global-layer q temperature; nullable (always null here)
        float q_scale;
        float eps;
    };

    static constexpr uint32_t kQW = kNumQHeads * 64;   // q / og segment width
    static constexpr uint32_t kKW = kNumKVHeads * 64;  // k / v segment width
    static constexpr uint32_t kKvRowBytes = kKW * 2;
    // no 128-column block straddles a q / k / v / og boundary
    static_assert(kQW % 128 == 0 && kKW % 128 == 0, "segment grain");
};

// The tile hook: each 64-column block is exactly one head, and lane l reads bf16 pair l of token t's head row straight
// from the token-major tile. Warp `quad` owns tokens {quad, quad + 4, ...}; the rewritten tile ships through the
// executor's TMA store (kMutatesTile).
template <uint32_t kNumQHeads, uint32_t kNumKVHeads, uint32_t kM>
struct TileEpiQkvRowops {
    using Args = typename RowEpiQkvRowops<kNumQHeads, kNumKVHeads>::Args;
    static constexpr bool kMutatesTile = true;
    static constexpr bool kCkSafe = true;  // valid on the K-split pair (rank 0 owns the reduced tile)
    static constexpr uint32_t kQW = kNumQHeads * 64;
    static constexpr uint32_t kKW = kNumKVHeads * 64;
    static constexpr uint32_t kKvRowBytes = kKW * 2;

    // Tokens [tok0, tok0 + kNtok) of the tile (all kM by default; the K-split ranks take a part each): warp `quad` owns
    // kNtok / 4 of them (one each on the first two warps at kNtok 2). Every token's operands load ahead of the token
    // math, so the L2 round trips overlap instead of sitting behind each token's shuffle chain.
    template <uint32_t kNtok = kM>
    static __device__ __forceinline__ void pre_store(const Args& epi, const uint32_t w_outer, nv_bfloat16* out_tile,
                                                     const uint32_t quad, const uint32_t lane_idx,
                                                     const uint32_t tok0 = 0) {
        static_assert((kNtok % 4 == 0 || kNtok == 2) && kNtok <= kM, "whole token quads, or a token pair");
        constexpr uint32_t kTok = kNtok < 4 ? 1 : kNtok / 4;
        if constexpr (kNtok < 4) {
            if (quad >= kNtok) return;
        }
        const bool is_q = w_outer < kQW;
        const bool is_k = !is_q && w_outer < kQW + kKW;
        const bool is_v = !is_q && !is_k && w_outer < kQW + 2 * kKW;
        if (is_q || is_k) {
            // qk-norm + rope (+ q temperature) per token, in place on the tile
            const bool has_ts = is_q && epi.temp_scale != nullptr;
            float c[kTok], s[kTok], ts[kTok];
#pragma unroll
            for (uint32_t i = 0; i < kTok; ++i) {
                const uint32_t t = tok0 + quad + 4 * i;
                c[i] = __ldg(epi.cos_sin + t * 64 + lane_idx);
                s[i] = __ldg(epi.cos_sin + t * 64 + 32 + lane_idx);
                ts[i] = has_ts ? __ldg(epi.temp_scale + t) : 1.0f;
            }
#pragma unroll
            for (uint32_t i = 0; i < kTok; ++i) {
                const uint32_t t = tok0 + quad + 4 * i;
                const auto pp = reinterpret_cast<nv_bfloat162*>(out_tile + t * 64) + lane_idx;
                nv_bfloat162 pv =
                    rowops::head_pair_qknorm_rope<true>(*pp, is_q ? epi.q_scale : 1.0f, epi.eps, c[i], s[i]);
                if (has_ts) {
                    const float2 f = __bfloat1622float2(pv);
                    pv = __float22bfloat162_rn(make_float2(__fmul_rn(f.x, ts[i]), __fmul_rn(f.y, ts[i])));
                }
                *pp = pv;
            }
        }
        if (!is_k && !is_v) return;  // q leaves through the tile's TMA store; og needs nothing
        // k / v append: one 128-B bulk copy per token row from the tile to its pool slot (asynchronous, so the
        // scattered pool writes cost issue latency only). The k rows are complete under __syncwarp (this warp rewrote
        // them); the v rows were published by the pre-hook sync.
        uint8_t* const cache = is_k ? epi.k_cache : epi.v_cache;
        const uint32_t col = (is_k ? w_outer - kQW : w_outer - kQW - kKW) * 2;
        __syncwarp();
        if (ptx::elect_one_sync()) {
            int64_t slot[kTok];
#pragma unroll
            for (uint32_t i = 0; i < kTok; ++i)
                slot[i] = __ldg(epi.cache_loc + tok0 + quad + 4 * i);
            ptx::fence_async_smem();
#pragma unroll
            for (uint32_t i = 0; i < kTok; ++i) {
                const uint32_t t = tok0 + quad + 4 * i;
                ptx::cp_async_bulk_store(cache + slot[i] * kKvRowBytes + col, ptx::to_shared(out_tile + t * 64), 128);
            }
            ptx::tma_store_commit();
            // the rows' smem source outlives the engine's reads only if the issuing lane waits for them (exit gives
            // no such guarantee); the read-side wait leaves the global side asynchronous
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncwarp();
    }
};

}  // namespace mk
