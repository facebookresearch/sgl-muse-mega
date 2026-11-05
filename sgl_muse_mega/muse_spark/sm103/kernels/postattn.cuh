#pragma once
// The post-attention phase of a layer image: o_proj with the all-reduce push epilogue, the lamport all-reduce + norm,
// the pooler and router GEMMs, the routing top-k, the nq rows (norm + NVFP4 quant), the MoE front as an item, and the
// phase's layout, unit queue and operands.

#include <cuda_bf16.h>
#include <cstdint>
#include <cuda_runtime.h>

#include "framework.cuh"
#include "ptx.cuh"
#include "moe.cuh"

// The all-reduce push as a tile epilogue: the push-lamport all-reduce's push half inside the o_proj tile's epilogue,
// mark + store only, no spin (the consume + norm runs in the AR row, framework.cuh). Each epilogue thread owns
// one 16 B packet per 16 tokens of the [kM, 64] tile, marks +0 -> -0 in registers and stores it with one relaxed.sys
// multimem.st (the workspaces' multicast address) into slot [phase x kRanks + rank] of every rank's workspace.
// Out-of-order tile arrival is legal: the markers are the arrival sync.

namespace mk {

template <uint32_t kHidden, uint32_t kRanks, uint32_t kM>
struct TileEpiArPush {
    struct Args {
        const uint64_t*
            peer_bases;        // [kRanks + 1] the ranks' workspace bases (device addresses), then the multicast base
        int64_t plane_stride;  // elements; 2 phases x kRanks slots each
        uint32_t rank;
    };
    // the 128 epi threads <-> kM*8 packets: kM / 16 rounds of the m16 tile's bijection (one at m16; the first half
    // of one at m8)
    static constexpr uint32_t kEpiThreads = 128;
    static constexpr uint32_t kRounds = (kM * 8 + kEpiThreads - 1) / kEpiThreads;
    static_assert(kM % 16 == 0 || kM == 8, "whole packet rounds over the epi threads, or the half round of m8");

    // one round of the push: tokens 16 i .. 16 i + 15 of the tile over the 128 epi threads (a body that splits the
    // tile's sequences over its warps calls a round per sequence)
    static __device__ __forceinline__ void push_round(const Args& epi, const uint32_t phase, const uint32_t w_outer,
                                                      const nv_bfloat16* out_tile, const uint32_t quad,
                                                      const uint32_t lane_idx, const uint32_t i) {
        const int64_t slot_0 = static_cast<int64_t>(phase) * kRanks * epi.plane_stride + epi.rank * epi.plane_stride;
        // thread (quad, lane) -> token t, packet p of the [kM, 64] tile, one round per 16 tokens
        const uint32_t tid = quad * 32 + lane_idx;
        const uint32_t pk = tid + i * kEpiThreads;
        const uint32_t t = pk >> 3, p = pk & 7u;
        if (kM % 16 != 0 && t >= kM) return;
        const uint4 v = *reinterpret_cast<const uint4*>(out_tile + t * 64 + p * 8);
        const uint32_t col_0 = t * kHidden + w_outer + p * 8;
        uint4 m = v;
        rowops::ar_lamport_mark_packet8(m);
        ptx::multimem_st_relaxed_sys_v4_bf16x2(
            reinterpret_cast<uint32_t*>(reinterpret_cast<nv_bfloat16*>(epi.peer_bases[kRanks]) + slot_0 + col_0), m);
    }

    // the push at a given plane phase (the layer takes it from its launch ticket)
    static __device__ __forceinline__ void push(const Args& epi, const uint32_t phase, const uint32_t w_outer,
                                                const nv_bfloat16* out_tile, const uint32_t quad,
                                                const uint32_t lane_idx) {
#pragma unroll
        for (uint32_t i = 0; i < kRounds; ++i)
            push_round(epi, phase, w_outer, out_tile, quad, lane_idx, i);
    }
};

// The routing top-k: the window token's top-8 over the router logits (sigmoid scores plus the expert bias for the
// selection, the weights renormalized and scaled).

namespace topk {

// fp32 +inf (the top-k sentinels use its negation)
constexpr float kInfF = __builtin_huge_valf();
constexpr uint32_t kNumExperts = cfg::kExperts, kNumTopk = cfg::kTopK;

// The window token's top-8 (sigmoid scores + expert-bias selection, weights renormalized). Emits global ids and
// weights, plus the single-rank local view (id < num_local ? id : -1). post_scale: the route scale on the renormalized
// weights, an RN post-multiply. topk_idx32 (nullable): the global ids as int32.
// The eight selection rounds are one rolled loop body instead of eight unrolled copies (~16 KB of straight-line code
// that one warp runs once per launch, fetched cold); the round-k score and id are recorded by lane k (warp-uniform
// values), so no register array is indexed at runtime. Same operations in the same order as the unrolled form.
// kIdsOnly: the routine stores only what a reader inside the launch takes (the int32 ids, the weights) and hands the
// lane's id back through id_out, so the caller publishes its route arrival behind two stores and writes the int64 ids
// and the local view after it; topk_idx / topk_idx_local are not read.
template <bool kIdsOnly = false>
static __device__ __forceinline__ void topk_token(const float* logits, const float router_mult,
                                                  const float* expert_bias, const uint32_t lane_idx,
                                                  const uint32_t num_local, int64_t* topk_idx, int64_t* topk_idx_local,
                                                  float* topk_w, const float post_scale, int32_t* topk_idx32,
                                                  uint32_t* id_out = nullptr) {
    const float kLog2e = __uint_as_float(0x3FB8AA3Bu);
    constexpr uint32_t kExPerLane = kNumExperts / 32;
    float s[kExPerLane], sel[kExPerLane];
    float acc[kExPerLane];
#pragma unroll
    for (uint32_t j = 0; j < kExPerLane; ++j)
        acc[j] = logits[j * 32 + lane_idx];
    // the lane's bias values stay in registers: the winner's bias below comes by shuffle from lane bidx % 32
    // (bitwise the same fp32 value) instead of a dependent global load per top-k round
    float bias[kExPerLane];
#pragma unroll
    for (uint32_t j = 0; j < kExPerLane; ++j) {
        const uint32_t ex = j * 32 + lane_idx;
        const float neg_scaled = fmaf(acc[j], -router_mult, 0.0f);
        s[j] = ptx::div_full_f32(1.0f, __fadd_rn(1.0f, ptx::ex2_approx_f32(__fmul_rn(neg_scaled, kLog2e))));
        bias[j] = expert_bias[ex];  // a plain load, not __ldg: the caller may hand a shared-memory copy
        sel[j] = s[j] + bias[j];
    }
    float score_sum = 0.0f;
    float my_score = 0.0f;  // lane k's round-k score and id
    uint32_t my_id = 0;
#pragma unroll 1
    for (uint32_t k = 0; k < kNumTopk; ++k) {
        float best = -kInfF;
        uint32_t bj = 0;
#pragma unroll
        for (uint32_t j = 0; j < kExPerLane; ++j)
            if (sel[j] > best) {
                best = sel[j];
                bj = j;
            }
        uint32_t bidx = bj * 32 + lane_idx;
        if constexpr (cfg::kWorld.topk_redux) {
            // the round's winner by two reductions: the largest score across the lanes, then the smallest id among
            // the lanes holding it (the butterfly's winner: equal scores compare equal)
            const float top = ptx::redux_max_f32(best, 0xffffffffu);
            const uint32_t wid = __reduce_min_sync(0xffffffffu, best == top ? bidx : 0xffffffffu);
            best = top;
            bidx = wid;
        } else {
#pragma unroll
            for (uint32_t offset = 16; offset >= 1; offset >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, offset);
                const uint32_t oi = __shfl_xor_sync(0xffffffffu, bidx, offset);
                if (ov > best || (ov == best && oi < bidx)) {
                    best = ov;
                    bidx = oi;
                }
            }
        }
        const uint32_t win_j = bidx / 32;
        float bias_mine = bias[0];
#pragma unroll
        for (uint32_t j = 1; j < kExPerLane; ++j)
            if (j == win_j) bias_mine = bias[j];
        const float bias_win = __shfl_sync(0xffffffffu, bias_mine, bidx % 32);
        const float sv = __fsub_rn(best, bias_win);
        score_sum = __fadd_rn(score_sum, sv);
        if (lane_idx == k) {
            my_score = sv;
            my_id = bidx;
        }
#pragma unroll
        for (uint32_t j = 0; j < kExPerLane; ++j)
            if (j == win_j && lane_idx == bidx % 32) sel[j] = -kInfF;
    }
    // 1e-15: the router's score-division epsilon
    const float inv_sum = ptx::div_full_f32(1.0f, __fadd_rn(score_sum, 1e-15f));
    if (lane_idx < kNumTopk) {
        const uint32_t id = my_id;
        if constexpr (kIdsOnly) {
            topk_idx32[lane_idx] = static_cast<int32_t>(id);
            topk_w[lane_idx] = __fmul_rn(__fmul_rn(my_score, inv_sum), post_scale);
            *id_out = id;
        } else {
            // the int64 outputs: read by nothing here (the stack reads idx32), kept for the schedule (README.md)
            topk_idx[lane_idx] = static_cast<int64_t>(id);
            if (topk_idx32 != nullptr) topk_idx32[lane_idx] = static_cast<int32_t>(id);
            topk_idx_local[lane_idx] = id < num_local ? static_cast<int64_t>(id) : int64_t(-1);
            topk_w[lane_idx] = __fmul_rn(__fmul_rn(my_score, inv_sum), post_scale);
        }
    }
    __syncwarp();
}

}  // namespace topk

// The front as an item: the route tables, the tile list and n_tiles from this launch's router ids, built on one CTA
// of the twelve-warp block by moe.cuh front_tables (layer.cu: the front cluster's rank 0, behind the route
// count).

// kCohIds: the ids were stored by other CTAs of this launch behind a ticketed count, so every entry read bypasses L1
// (moe.cuh)
template <bool kCohIds>
struct MoeFrontItem {
    struct Args {
        const int32_t* topk_idx;    // [B][kTopK] this launch's router ids
        moe_tp::RouteTables route;  // the MoE phase's tables
        int B, b_live;
    };
    static constexpr uint32_t kWarps = block::kWarps;  // front_tables is CTA-wide: the whole block
    using Scratch = moe_tp::FrontScratch;
    // behind an ABI boundary, by value: inlining this glue into a kernel that also carries a tcgen05 mainloop
    // breaks ptxas's uniform-datapath compilation. The third operand is unused, kept for the schedule (README.md).
    static __device__ __noinline__ void run(Args a, Scratch& fs, uint64_t* /*unused*/) {
        moe_tp::front_tables<kCohIds, cfg::kWorld.front_regs>(moe_tp::FrontArgs{a.topk_idx, a.B, a.b_live}, a.route,
                                                              fs);
    }
};

// The post-attention phase's layout: its role map, the carve and ring constants, the unit
// queue's decode and the ring cursor, and its operands.

// The role map is the dense executor's (framework.cuh); the item warps are warps 4..11.
inline constexpr block::RoleMap kPostAttnRoles = kDenseRoles;
static_assert(block::valid(kPostAttnRoles), "the post-attention phase's role map has the framework's shape");
inline constexpr block::WarpSpan kPostAttnItemSpan = block::item_span(kPostAttnRoles);
static_assert(kPostAttnItemSpan.warp0 == 4 && block::fits(kPostAttnItemSpan, block::kItemWarps), "item warps 4..11");
inline constexpr uint32_t kPostAttnWProducer = kPostAttnRoles.producer, kPostAttnGate = kPostAttnRoles.gate;
inline constexpr uint32_t kPostAttnMma = kPostAttnRoles.mma, kPostAttnAux = kPostAttnRoles.aux,
                          kPostAttnEpi0 = kPostAttnRoles.epi0;

namespace postattn {
// kM: the window's rows (16 x the sequences, or the 8-row window of 7 drafts): the MMA's N, the drains' column
// loop, the row queues. kMc: the carve's rows; an 8-row window keeps the 16-row layout (the sliding-window
// attention's ring blocks and the top-k reduction staging live in its x and out regions).
inline constexpr uint32_t kM = cfg::kWindowM;
inline constexpr uint32_t kMc = kM < 16 ? 16 : kM;
inline constexpr uint32_t BN = 64;
// the route count's arrivals per launch: the width invariant's power of two (framework.cuh)
inline constexpr uint32_t kRouteArrivals = handoff::pow2_at_least(kM);
static_assert((uint64_t(1) << 32) % kRouteArrivals == 0, "the token rows' counts would wrap mid-launch");
static_assert(kM == 16 || kM == 8, "the post-attention rows: one sequence of 16 (or 8)");
// the front tables' publishes per launch to the MoE phase (front_done)
inline constexpr uint32_t kFrontPubs = 1;
inline constexpr uint32_t kHidden = cfg::kHidden, kRanks = cfg::kRanks;  // kRanks: the all-reduce's TP world
inline constexpr uint32_t kStageBytes = BN * 128;                        // one 64-elem bf16 W stage (8 KB)
inline constexpr uint32_t kXTile = kMc * 128;                            // one 64-elem x k-block's slot (2 KB)
inline constexpr uint32_t kXBytes = kM * 128;                            // its TMA bytes
inline constexpr uint32_t kSlotStages = cfg::kPack;                      // 32 KB of W per slot barrier
inline constexpr uint32_t kSlots = cfg::kStages / cfg::kPack;            // 20 stages in flight at m16
static_assert((kSlotStages == 4) && cfg::kStages % cfg::kPack == 0 && kSlots == 5, "the ring is the pins' carve");
inline constexpr uint32_t kWStageBytes = kStageBytes;
inline constexpr uint32_t kSlotXOff = kSlotStages * kWStageBytes;              // the x tiles behind the W stages
inline constexpr uint32_t kSlotBytes = kSlotStages * (kWStageBytes + kXTile);  // 40960 shared
inline constexpr uint32_t kRingBytes = kSlots * kSlotBytes;                    // 204800 shared
inline constexpr uint32_t kOutOff = kRingBytes;       // the [m, 64] bf16 tile (2 KB), or a [m, 128] fp32 partial (8 KB)
inline constexpr uint32_t kOutBytes = kMc * 128 * 4;  // the fp32 partial's 8 KB
// the pooler's K split: the kPoolerCk CTAs of a cluster-unit stream disjoint K halves of one tile; the odd CTA sends
// its fp32 accumulator [4 quads][16 rows][kM tokens] to the even CTA's xfer buffer, which adds (own + peer) and
// stores. The K-window variant (the world's k_window: each rank streams its quarter of K and the ranks' partials are
// summed) streams a quarter of K already: one CTA per tile.
inline constexpr bool kKWindow = cfg::kWorld.k_window;
inline constexpr bool kRelFree = cfg::kWorld.relfree;
inline constexpr uint32_t kPoolerCk = kKWindow ? 1 : 2;
inline constexpr uint32_t kXferBytes = 4 * 16 * kM * 4;  // 4096 (the odd CTA's send)
inline constexpr uint32_t kXferOff = kOutOff + kOutBytes;
inline constexpr uint32_t kTmemOff = kXferOff + 4 * 16 * kMc * 4;
inline constexpr uint32_t kBarOff = kTmemOff + 16;
// the claim queue's end marker
inline constexpr uint32_t kTerminator = 0xFFFFFFFFu;
// full[kSlots] empty[kSlots] tmem_full[2] tmem_empty[2] xchg[2] ticket xfer_full xfer_empty reduce
inline constexpr uint32_t kBars = 2 * kSlots + 4 + 6;
// the top-k row's reduction staging: both n-blocks' cfg::kRouterSplit partial rows (512 B each) land in out_tile +
// xfer by bulk copies
static_assert(2 * cfg::kRouterSplit * 512 <= kTmemOff - kOutOff, "the reduction staging fits out_tile + xfer");
inline constexpr uint32_t kSmemBytes = (kBarOff + kBars * 8 + 1023) / 1024 * 1024;
// the router's logits and bias rows (one float per expert each), past the post-attention carve inside the MoE
// phase's carve
inline constexpr uint32_t kTopkBytes = 2 * cfg::kExperts * sizeof(float);
// two accumulators of 32 columns (the m16 window at the 32-column grain), one 64-column allocation
inline constexpr uint32_t kAccCols = 32;
inline constexpr uint32_t kTmemCols = 2 * kAccCols;
static_assert(kSmemBytes <= cfg::kSmemOptin, "the post-attention carve fits the sm_103a opt-in cap");
// the o_proj K = the rank's query heads x head_dim 64, in 64-elem stages
// the pooler's K range: the rank's K window, or the whole hidden
inline constexpr uint32_t kOprojStages = cfg::kQHeads * 64 / BN,
                          kPoolerStages = (kKWindow ? kHidden / kRanks : kHidden) / 64;
inline constexpr uint32_t kPoolerUnitStages = kPoolerStages / kPoolerCk;  // per CTA of a pooler cluster-unit
inline constexpr uint32_t kOprojBlocks = kHidden / BN;           // 128 tiles = the static units of 32 width-4 clusters
inline constexpr uint32_t kPoolerBlocks = cfg::kMoeHidden / BN;  // 64 tiles
// the o_proj W stream's L2 hint: evict-first like the pooler's
inline constexpr uint64_t kOprojHint = ptx::kEvictFirst;
inline constexpr uint32_t kNqTilesPerWarp = kPoolerBlocks / block::kItemWarps;  // 8: the warp's 512 columns
// the router units: W rows [E = 256] fp32 read as tf32, K = 8192 in 32-wide stages of 8 KB (one ring granule); a
// unit is HALF an n-block (64 W rows) x one K-slice, unit r = (half r & 1, n-block (r >> 1) & 1, K-slice r >> 2),
// writing one [kM x 64] fp32 partial. The per-row accumulation is the 128-row unit's, so the logits are bitwise.
inline constexpr uint32_t kRouterBN = 128, kRouterBk = 32, kRouterRows = 64;
inline constexpr uint32_t kRouterStageBytes = kRouterRows * kRouterBk * 4;  // 8192
inline constexpr uint32_t kRouterSlotStages = kSlotStages;
inline constexpr uint32_t kRouterUnitsPerCluster = cfg::kW;  // one router unit per CTA of a cluster-unit
// the gathered rows inside the o_proj workspace slot, as bf16-element offsets from the slot base (an fp32 counts as
// two): past the o_proj packets [rows][kHidden], the pooled and router-logit partial planes, then the unpooler's
// gathered [rows][kHidden] row (next_layer.cuh) and the pooler's gathered [rows][kPooledRow] row. The host binds the
// same offsets. The K-window variant lays an 8-row window's slot on the 16-row layout (kMc): its partial planes start
// empty (+0) like the packets, so the windows' images can share one workspace only if no gather row (prefilled
// all-ones) lands on the other's planes.
inline constexpr uint32_t kPooledRow = kPoolerBlocks * BN;  // 4096: the pooled row width
inline constexpr uint32_t kSlotRows = kKWindow ? kMc : kM;
inline constexpr uint32_t kUnpoolAgOff =
    kSlotRows * kHidden + 2 * kSlotRows * kPooledRow + 2 * kSlotRows * cfg::kExperts;
inline constexpr uint32_t kPoolAgOff = kUnpoolAgOff + kSlotRows * kHidden;
// the gather group's ranks (the K-window variant computes every tile on every rank: a group of one)
inline constexpr uint32_t kPoolAgGroup = kKWindow ? 1 : 4;
inline constexpr uint32_t kPoolAgTiles = kPoolerBlocks / kPoolAgGroup;  // the rank's pooled tiles
static_assert((kPoolerBlocks % kPoolAgGroup == 0 && kRanks % kPoolAgGroup == 0),
              "whole pooled tiles per rank, the groups tile the ranks");
// the router's K range in 32-wide stages: the rank's K window, or the whole hidden
inline constexpr uint32_t kRouterKBlocks = (kKWindow ? kHidden / kRanks : kHidden) / 32;
// the K-window variant's partial planes in the o_proj slot (the AR's lamport protocol at fp32): the pooled partial
// [kM][kPooledRow] and the router-logit partial [kM][256], both fp32, past the o_proj packets; slot [phase][r] of
// every rank's workspace holds rank r's partials, and each consumer sums its own copy's kRanks planes in rank order
inline constexpr uint32_t kPrPooledOff = kSlotRows * kHidden;
inline constexpr uint32_t kPrLogitsOff = kPrPooledOff + kSlotRows * kPooledRow * 2;
static_assert(kPrLogitsOff + kSlotRows * 2 * kRouterBN * 2 <= kUnpoolAgOff, "the partial planes sit below the gathers");
inline constexpr uint32_t kReduceBlock = 32;
inline constexpr uint32_t kArrivalBlock = 4;  // a row's arrival word: two last arrivers + the owner + the owner's pad

// the queue: [AR rows][router cluster-units][top-k rows][pooler cluster-units]. The nq rows ride the first kQNqUnits
// pooler units and the front tables the cluster that ran top-k row 0: the layer's wall is the MoE phase's end, and
// the MoE phase's rate is the SMs it gets, so a unit on a cluster that is busy anyway costs the MoE phase nothing
// while a unit on its own cluster holds four CTAs for the whole launch.
enum Unit : uint32_t { kUnitArRow = 0, kUnitRouter, kUnitPooler, kUnitTopk, kUnitNq, kUnitFront, kUnitNone };
// the AR row's CTAs: the width-kW cluster, one row per cluster claim
inline constexpr uint32_t kRowCtas = cfg::kW;
inline constexpr uint32_t kQAr = kM / (cfg::kW / kRowCtas);
// 2 n-blocks x 2 halves x cfg::kRouterSplit K-slices, one unit per CTA of a cluster-unit (split 12 fixes the reduction
// order: bitwise logits)
inline constexpr uint32_t kQRouter = 4 * cfg::kRouterSplit / kRouterUnitsPerCluster;  // 2 n-blocks x 2 halves x split
inline constexpr uint32_t kTopkRows = 1;                                              // token rows per top-k CTA claim
static_assert(kTopkRows >= 1 && kM % (cfg::kW * kTopkRows) == 0, "whole rows on every CTA of a top-k unit");
inline constexpr uint32_t kQTopk = kM / (cfg::kW * kTopkRows);  // rows per unit-CTA
inline constexpr uint32_t kQNqUnits = kM / cfg::kW;             // the pooler units that also run an nq row
// the front tables run statically on the cluster that ran top-k row 0 (a claimed front waited behind a co-claimed
// pooler stream, or held one behind it)
inline constexpr uint32_t kFrontCluster = kQAr + kQRouter;  // an o_proj cluster, free exactly at the route count

struct Queue {
    static constexpr uint32_t kW = cfg::kW;
    // the pooler cluster-units: kW / kPoolerCk tiles each, the rank's 1 / kPoolAgGroup share of the 64
    static constexpr uint32_t kPooler = kPoolerBlocks * kPoolerCk / kW / kPoolAgGroup;
    static_assert(kPoolerBlocks * kPoolerCk % (kW * kPoolAgGroup) == 0, "whole pooler cluster-units per rank");
    // the nq rows' hosts: one per pooler cluster-unit
    static constexpr uint32_t kNqUnits = kQNqUnits;
    static_assert(kNqUnits <= kPooler, "one nq row per pooler cluster-unit");
    static constexpr uint32_t kLen = kQAr + kQRouter + kPooler + kQTopk;
    struct Entry {
        uint32_t type, index;
    };
    static __device__ __forceinline__ Entry decode(uint32_t d) {
        if (d == kTerminator) return {kUnitNone, 0u};
        if (d < kQAr) return {kUnitArRow, d};
        d -= kQAr;
        if (d < kQRouter) return {kUnitRouter, d};
        d -= kQRouter;
        // ORDER LAW: a CTA runs its claims in queue order, so a unit may wait only on units EARLIER in the
        // queue. The top-k rows wait on the router units; the nq rows wait on all 64 pooler tiles and go
        // last (an nq claimed ahead of a pooler on the same CTA deadlocks); the front is static on its own
        // cluster.
        if (d < kQTopk) return {kUnitTopk, d};
        d -= kQTopk;
        if (d < kPooler) return {kUnitPooler, d};
        d -= kPooler;
        return {kUnitNq, d};
    }
};

// the ring cursor one warp carries across units
struct RingCursor {
    uint32_t slot = 0, phase = 0;
    __device__ __forceinline__ void advance() {
        slot = slot == kSlots - 1 ? 0u : slot + 1;
        phase ^= slot == 0;
    }
};

}  // namespace postattn

// the router units' and the tails' operands
struct RouterArgs {
    float* ws;           // [split * 2][16][128] fp32 partials, unit-major
    uint32_t* counters;  // [2] the router units' ticketed count per n-block (+kReduceBlock per launch: split + the pad)
    uint32_t* arrival;   // [16] the top-k rows' own-ticket words (+kArrivalBlock per launch: three arrivals + the pad)
    float* logits;       // [16][256] fp32
    uint32_t split;      // the K-split width (cfg::kRouterSplit, checked at entry)
    uint32_t k_blocks;   // K / 32 (checked at entry)
    uint64_t w_hint;     // the router weights' TMA cache hint
    const float* bias;
    float router_mult;
    float route_scale;
    uint32_t num_local;
    int64_t* topk_idx;
    int64_t* topk_local;
    float* topk_w;
    int32_t* topk_idx32;
    NormQuantArgs nq;  // the nq rows' operands
};

struct PostAttnArgs {
    uint32_t n_clusters, queue_len;  // the queue is walked round-robin by the n_clusters clusters
    uint32_t* h3_ready;
    uint32_t* route_count;
    uint32_t* launch_ctr;        // [grid]
    uint64_t* stamps;            // unused; kept for the schedule (README.md)
    uint64_t* pooled_ready;      // [1 + kPoolerBlocks] the K-window variant's per-tile words (the nq rows' wait)
    const uint64_t* peer_bases;  // [kRanks] the lamport workspaces (device pointers), this rank's included
    int64_t plane_stride;        // elements: 2 phases x kRanks slots per workspace
    uint32_t rank;
    const float* stream_in;
    const float* alpha;
    const float* beta;
    const nv_bfloat16* prenorm_w;
    float* stream_out;
    nv_bfloat16* h3;
    float* h3f;
    uint32_t* front_done;  // the tables' TICKETED COUNT: +1 per launch by the front CTA (the MoE phase waits t + 1)
    uint32_t* nq_done;     // the token rows': +kRouteArrivals per launch (the MoE phase waits kRouteArrivals * (t + 1))
};

// The post-attention phase: o_proj -> lamport all-reduce + post-attention norm -> pooler || router -> top-k -> the nq
// rows (pre-expert norm + NVFP4 quant) -> the front tables, run as the layer launch's post-attention phase (layer.cu
// PostAttnPhase).
//
// Units and what each waits on (per rank, m = 16):
//  o_proj tile    static: cluster c's CTA r owns W block kW*c + r (kOprojStages 64-wide stages over the o_proj K), x
//                 behind the grid wait; the tile's epilogue pushes it into the kRanks ranks' lamport planes at
//                 phase = launch ticket % 2
//  AR + norm row  one row per cluster claim, its quarters on the cluster's CTAs (item warps 4..11); the ranks'
//                 packets are the signal (the lamport sentinel spin); the two partial-sum rounds ride a cluster-scope
//                 mbarrier; publishes h3 (bf16), h3f (fp32) and stream_out, then h3_ready
//  router unit    half an n-block x one K-slice of the tf32 router GEMM; its fp32 partial to ra.ws
//  top-k row      reduces its token's router partials, runs top-k, publishes the route count
//  pooler tile    a K-split pair (kPoolerCk = 2) per tile; x = h3 behind one ticketed wait; the finished bf16 tile
//                 is pushed to the 4-rank gather group's pooled rows (the rank streams 1 / 4 of the tiles)
//  nq row         rides a pooler unit: polls its gathered pooled row, norms + quantizes it, publishes nq_done
//  front tables   static on the cluster that ran top-k row 0, behind the route count; publishes front_done
// The queue is walked in dependency order by a static round-robin over the clusters (claim k of cluster c is entry
// k * n_clusters + c, computed by every role), so a unit waits only on earlier entries or on the ranks' static
// o_proj tiles and a persistent grid never waits on an unclaimed unit.
// The ring: 5 slots x 40 KB = 20 stages of (8 KB W + 2 KB x) at pack 4, one (full, empty) barrier pair per slot. The
// weight producer (warp 0) streams every claimed unit's stages back to back and never waits on an activation;
// the x producer (warp 1) waits on each unit's input. Two TMEM accumulators alternate per GEMM unit, so a unit's drain
// overlaps the next unit's MMAs.
// Words (one u32 allocation, zeroed at bind): [h3_ready | route_count | front_done | nq_done | launch_ctr[grid]].

namespace postattn {

// the nq row's math on the row's fragments: the pre-expert norm + the dynamic NVFP4 quant into q / sf / x_scale, or
// on the bf16 MoE layers (kBf16) the norm alone, its bf16 row into q
template <uint32_t kH_>
__device__ __forceinline__ void nq_row_math(const NormQuantArgs& nq, const uint32_t t, const uint32_t warp8,
                                            const uint32_t lane, const uint4 (&frag)[2], const uint4 (&gam)[2],
                                            float* staging) {
    constexpr uint32_t kBar = block::kItem;  // the item warps' barrier; a local: the body takes it by reference
    if constexpr (kImage.bf16) {
        rowops::row_pre_expert_norm_bf16_dyn_wide_frag<kH_>(
            warp8, lane, frag, gam, nq.eps, reinterpret_cast<nv_bfloat16*>(nq.q) + static_cast<size_t>(t) * kH_,
            staging, kBar);
    } else {
        const float ts = rowops::row_pre_expert_norm_quant_nvfp4_dyn_wide_frag<kH_>(
            warp8, lane, frag, gam, nq.eps, nq.q + static_cast<size_t>(t) * (kH_ / 2),
            nq.sf + static_cast<size_t>(t) * (kH_ / 16), staging, kBar);
        if (warp8 == 0 && lane == 0) nq.x_scale[t] = ts;
    }
}
// the nq row: the thread's 16 pooled columns as two bf16 packets of this rank's gathered row, each polled at system
// scope until neither half carries the empty mark and re-marked for the next layer (framework.cuh
// ag_poll_take_u4); then the row's math
template <uint32_t kM_, uint32_t kH_>
__device__ __noinline__ void nq_item_gathered(const NormQuantArgs nq, nv_bfloat16* row0, const uint32_t t,
                                              float* staging) {
    const uint32_t warp8 = ptx::warp_idx_sync() - 4, lane = ptx::lane_idx();
    const uint32_t col_0 = rowops::dyn_wide_col0(warp8, lane);
    uint4 gam[2], frag[2];
    rowops::dyn_wide_load_gamma(static_cast<const nv_bfloat16*>(nq.gamma), col_0, gam);
    const uint4* const pk = reinterpret_cast<const uint4*>(row0 + static_cast<size_t>(t) * kH_ + col_0);
    frag[0] = rowops::ag_poll_take_u4(pk);
    frag[1] = rowops::ag_poll_take_u4(pk + 1);
    nq_row_math<kH_>(nq, t, warp8, lane, frag, gam, staging);
}
// the top-k row's routine out of line: the router logits and bias from smem, the ids / local view / weights out
__device__ __noinline__ void topk_row_out(const float* logits, float router_mult, const float* bias, uint32_t lane_idx,
                                          uint32_t num_local, int64_t* idx, int64_t* local, float* w, float route_scale,
                                          int32_t* idx32) {
    topk::topk_token(logits, router_mult, bias, lane_idx, num_local, idx, local, w, route_scale, idx32);
}
// the route-early variant: the int32 ids and the weights stored, the lane's id (lanes < 8) handed back for the int64
// stores behind the route arrival
__device__ __noinline__ void topk_ids_out(const float* logits, float router_mult, const float* bias, uint32_t lane_idx,
                                          uint32_t num_local, float route_scale, int32_t* idx32, float* w,
                                          uint32_t& id_out) {
    topk::topk_token<true>(logits, router_mult, bias, lane_idx, num_local, nullptr, nullptr, w, route_scale, idx32,
                           &id_out);
}
// the K-window variant's nq row: the thread's 16 pooled columns as four fp32 packets summed over the kRanks partial
// planes of this rank's copy (`planes` = the copy's pooled partial at slot [phase][0], rank order), rounded to bf16
// once into the row's fragments, the copy's markers restored; then the row's math
template <uint32_t kM_, uint32_t kH_, uint32_t kRanks_>
__device__ __noinline__ void nq_item_planes(const NormQuantArgs nq, nv_bfloat16* planes, const int64_t plane_stride,
                                            const uint32_t t, float* staging) {
    const uint32_t warp8 = ptx::warp_idx_sync() - 4, lane = ptx::lane_idx();
    const uint32_t col_0 = rowops::dyn_wide_col0(warp8, lane);
    uint4 gam[2], frag[2];
    rowops::dyn_wide_load_gamma(static_cast<const nv_bfloat16*>(nq.gamma), col_0, gam);
    float* rrows[kRanks_];
    const float* rows[kRanks_];
#pragma unroll
    for (uint32_t r = 0; r < kRanks_; ++r) {
        rrows[r] = reinterpret_cast<float*>(planes + r * plane_stride) + static_cast<size_t>(t) * kH_;
        rows[r] = rrows[r];
    }
    alignas(16) nv_bfloat162 fb[8];
#pragma unroll
    for (uint32_t p = 0; p < 4; ++p) {
        const float4 s = rowops::ar_lamport_consume_packet4f<kRanks_>(rows, col_0 + p * 4);
        fb[2 * p] = __float22bfloat162_rn(make_float2(s.x, s.y));
        fb[2 * p + 1] = __float22bfloat162_rn(make_float2(s.z, s.w));
    }
#pragma unroll
    for (uint32_t p = 0; p < 4; ++p)
        rowops::ar_lamport_restore_packet4f<kRanks_>(rrows, col_0 + p * 4);
    frag[0] = *reinterpret_cast<const uint4*>(&fb[0]);
    frag[1] = *reinterpret_cast<const uint4*>(&fb[4]);
    nq_row_math<kH_>(nq, t, warp8, lane, frag, gam, staging);
}

}  // namespace postattn

template <uint32_t kW, class Attention>
static __device__ __forceinline__ void postattn_body(const PostAttnArgs& a, const CUtensorMap& tmap_wo,
                                                     const CUtensorMap& tmap_xa, const CUtensorMap& tmap_wp,
                                                     const CUtensorMap& tmap_xh, const CUtensorMap& tmap_pooled,
                                                     const RouterArgs& ra, const MoeFrontItem<true>::Args& fa,
                                                     const CUtensorMap& tmap_wr, const CUtensorMap& tmap_xf,
                                                     MoeFrontItem<true>::Scratch& front_scratch, uint32_t& s_ticket,
                                                     uint8_t* smem_buf, const typename Attention::Args& attention,
                                                     const typename Attention::PrefetchArgs& prefetch) {
    using namespace postattn;
    using Q = Queue;
    static_assert(kW == Q::kW, "the queue's clusters are the launch's");
    static_assert(kW == 4, "the cluster width is the AR row's split in quarters");
    // the AR rows' CTAs publish h3_ready: its arrivals per launch at the width invariant's power of two
    // (framework.cuh)
    constexpr uint32_t kH3Arrivals = handoff::pow2_at_least(kRowCtas * kM);
    static_assert((uint64_t(1) << 32) % kH3Arrivals == 0 && (kH3Arrivals - kRowCtas * kM) % kRowCtas == 0,
                  "h3_ready's width");
    static_assert(kW == cfg::kW, "the queue's constants are the module's width");
    static_assert(block::kThreads == 384, "the twelve-warp block");
    // the drain's warps: the epilogue quad
    constexpr uint32_t kDrainWarps = kPostAttnRoles.n_epi;
    constexpr uint32_t kEpiThreads = kDrainWarps * 32;
    constexpr uint32_t kEpiBarId = block::kEpi;
    // the composed sliding-window attention needs no post-attention resource (no TMEM, no DSMEM, nothing at cluster
    // scope) and owns its two words: its warps start at the CTA's entry and take the entry cluster barrier behind the
    // attention, ahead of their first cluster-scope use (the AR row's exchange), while the post-attention warps wait at
    // the cluster barrier
    constexpr bool kAttnAhead = !Attention::kGlobalAttn;

    const uint32_t smem_w = ptx::to_shared(smem_buf);
    auto out_tile = reinterpret_cast<nv_bfloat16*>(smem_buf + kOutOff);
    auto tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_buf + kTmemOff);
    uint64_t* const bar_base = reinterpret_cast<uint64_t*>(smem_buf + kBarOff);
    uint64_t* const full_bars = bar_base;
    uint64_t* const empty_bars = bar_base + kSlots;
    uint64_t* const tmem_full = bar_base + 2 * kSlots;
    uint64_t* const tmem_empty = tmem_full + 2;
    uint64_t* const xchg_bar = tmem_empty + 2;
    uint64_t* const ticket_bar = xchg_bar + 2;  // xchg_bar[0..1]: one exchange round each
    uint64_t* const xfer_full = xchg_bar + 3;   // the even CTA: the peer's 4 KB landed (arrive.expect_tx + complete_tx)
    uint64_t* const xfer_empty = xchg_bar + 4;  // the odd CTA: the even consumed the last send (a remote arrive)
    uint64_t* const reduce_bar = xchg_bar + 5;  // the top-k row's partials landed (bulk copies, tx-counted)
    float* const xfer = reinterpret_cast<float*>(smem_buf + kXferOff);
    __shared__ float cstage[2 * 32];  // the AR row's [2 rounds][the 4 CTAs' 8 warps] partials, scattered by st.async
    __shared__ float nq_staging[16];
    // the top-k arrays ride the post-attention phase's idle tail of the MoE phase's carve (layer.cu kPostAttnSmem); the
    // static smem budget is the MoE phase's
    float* const topk_logits = reinterpret_cast<float*>(smem_buf + kSmemBytes);
    float* const topk_bias = topk_logits + cfg::kExperts;
    // the MoE phase's capacity holds the window's token count
    static_assert(uint32_t(moe_tp::kBMax) >= kM, "the MoE phase holds the window's token count");

    const uint32_t warp_idx = ptx::warp_idx_sync();
    const uint32_t lane_idx = ptx::lane_idx();
    const uint32_t rank = ptx::cluster_ctarank();
    const uint32_t cluster = blockIdx.x / kW;
    const auto umin = [](uint32_t x, uint32_t y) { return x < y ? x : y; };
    const bool has_oproj = cluster < kOprojBlocks / kW;
    const uint32_t oproj_outer = (kW * cluster + rank) * BN;
    // the static front: the cluster that ran top-k row 0
    const bool is_front_cluster = cluster == kFrontCluster;
    // the k-th claim's unit: the static program's round-robin
    const auto take_unit = [&](const uint32_t k) {
        const uint32_t d = k * a.n_clusters + cluster;  // a cluster past the queue claims nothing
        return Q::decode(cluster < a.n_clusters && d < a.queue_len ? d : kTerminator);
    };

    // ---- init (warp 1) + TMEM alloc (warp 3) + the launch ticket ----
    if (warp_idx == kPostAttnGate && ptx::elect_one_sync()) {
        ptx::prefetch_tensormap(&tmap_wo);
        ptx::prefetch_tensormap(&tmap_xa);
        ptx::prefetch_tensormap(&tmap_wp);
        ptx::prefetch_tensormap(&tmap_xh);
        for (uint32_t i = 0; i < kSlots; ++i) {
            ptx::mbar_init(&full_bars[i], 1);
            ptx::mbar_init(&empty_bars[i], 1);
        }
        for (uint32_t b = 0; b < 2; ++b) {
            ptx::mbar_init(&tmem_full[b], 1);
            ptx::mbar_init(&tmem_empty[b], 1);
        }
        ptx::mbar_init(xchg_bar,
                       8);  // the AR row's exchange: the 8 item warps arrive, the 3 peers' partials complete tx
        ptx::mbar_init(xchg_bar + 1, 8);
        ptx::mbar_init(ticket_bar, 1);
        ptx::mbar_init(xfer_full, 1);
        ptx::mbar_init(xfer_empty, 1);
        ptx::mbar_init(reduce_bar, 1);
        if constexpr (!kAttnAhead) Attention::init(smem_buf);
        ptx::fence_mbarrier_init_release_cluster();
        // the static schedule is bound to this grid and queue: a mismatch would drift every claim silently
        const bool split_ok = 2 * ra.split <= kReduceBlock
                              // the router's K range is decoded from the compile-time
                              // split
                              && ra.split == cfg::kRouterSplit && ra.k_blocks == kRouterKBlocks;
        // the post-attention queue runs on a.n_clusters clusters (they land under attention's drain); the MoE phase
        // runs on every resident cluster, and the clusters past the queue claim nothing
        const bool grid_ok = gridDim.x >= kW * a.n_clusters && gridDim.x % kW == 0;
        if (!grid_ok || a.queue_len != Q::kLen || !split_ok) __trap();
    }
    // the composed global attention allocates the whole TMEM pool at its entry; the post-attention phase's 64 columns
    // are allocated by the first epilogue warp behind the attention's dealloc and read behind its done barrier
    if constexpr (!Attention::kGlobalAttn) {
        if (warp_idx == kPostAttnAux) ptx::tcgen05_alloc(ptx::to_shared(tmem_addr_storage), kTmemCols);
    }
    if constexpr (kAttnAhead) {
        // the attention's words are its leader's, initialized ahead of its arrival: the gate warp's epoch bump
        // and the post-attention done waits all sit behind their own cluster wait
        if (warp_idx == kPostAttnEpi0 && lane_idx == 0) {
            Attention::init(smem_buf);
            ptx::fence_mbarrier_init_release_cluster();
        }
        __syncwarp();
        ptx::cluster_barrier_arrive();  // peer mbars armed before any st.async lands
        if (warp_idx < kPostAttnEpi0) ptx::cluster_barrier_wait();
    } else {
        ptx::cluster_sync();  // peer mbars armed before any st.async lands
    }
    uint32_t taddr = Attention::kGlobalAttn ? 0u : tmem_addr_storage[0];
    // The launch ticket: the gate warp reads it behind its grid wait (the previous launch bumps it at exit,
    // after its trigger) but after issuing the first o_proj x stages, which need no ticket. Every other warp but
    // the weight producer takes it from ticket_bar; the item warps' first use is their o_proj push.
    // The item warps wait here. The MMA must not: the feeder's x stages reuse the ring's slots and need its releases.
    uint32_t ticket = 0;
    if (warp_idx >= kPostAttnEpi0) {
        Attention::run(attention, smem_buf, warp_idx, lane_idx);
        if constexpr (Attention::kGlobalAttn) taddr = tmem_addr_storage[0];
        if constexpr (kAttnAhead) {
            ptx::cluster_barrier_wait();   // the entry barrier, arrived ahead of the attention
            taddr = tmem_addr_storage[0];  // the aux warp's allocation, published by it
        }
        ptx::mbar_wait_parity_poll(ticket_bar, 0);
        ticket = *reinterpret_cast<const volatile uint32_t*>(&s_ticket);
    }
    const uint32_t phase = ticket & 1u;  // the lamport planes' phase: push and consume agree through the ticket
    // the rank's pooled tiles: [pool_tile0, pool_tile0 + kPoolAgTiles)
    [[maybe_unused]] const uint32_t pool_tile0 = (a.rank % kPoolAgGroup) * kPoolAgTiles;
    // the rank's K windows (the K-window variant): the pooler's first 64-wide stage, the router's first 32-wide k-block
    [[maybe_unused]] const uint32_t pr_k0 = kKWindow ? a.rank * kPoolerStages : 0u,
                                    pr_kb0 = kKWindow ? a.rank * kRouterKBlocks : 0u;
    // router unit r = kRouterUnitsPerCluster * cluster-unit + rank: its n-block, half and K-slice
    const auto router_geom = [&](const uint32_t u, uint32_t& nblock, uint32_t& half, uint32_t& slice, uint32_t& k_begin,
                                 uint32_t& k_iters) {
        const uint32_t r =
            kRouterUnitsPerCluster * u + rank;  // a cluster-unit = one K-slice, both n-blocks, both halves
        half = r & 1u;
        nblock = (r >> 1) & 1u;
        slice = r >> 2;
        constexpr uint32_t k_base = kRouterKBlocks / cfg::kRouterSplit, k_rem = kRouterKBlocks % cfg::kRouterSplit;
        k_begin = pr_kb0 + slice * k_base + umin(slice, k_rem);
        k_iters = k_base + (slice < k_rem ? 1u : 0u);
    };
    // the front tables on the whole CTA of the front cluster's rank 0: every warp calls it after its claim loop,
    // so the item's block barriers see all twelve warps
    const auto front_unit = [&]() {
        if (rank != 0) return;
        if (warp_idx == kPostAttnGate && lane_idx == 0) {
            handoff::ticket_wait(a.route_count, kRouteArrivals * (ticket + 1));
        }
        // the waiter's acquire reaches the block (the tables' inputs), then the item
        __syncthreads();
        MoeFrontItem<true>::run(fa, front_scratch, nullptr);
        // the tables' publish to the MoE phase (framework.cuh item_publish on the whole CTA): every
        // writing thread bridges the async proxy, the CTA barrier, one release
        ptx::fence_async_global();
        __syncthreads();
        if (threadIdx.x == 0) ptx::red_add_rel_b32(a.front_done, 1u);
    };

    // warp 0 carries one ring cursor through the o_proj slice and every claimed unit
    RingCursor weight_ring;
    // bf16: kSlotStages 8 KB stages per slot; tf32 (the router): kRouterSlotStages 8 KB stages
    const auto stream_w = [&](const CUtensorMap& tm, const uint32_t stages, const uint32_t k0, const uint32_t w_outer,
                              const uint32_t k_elems, const uint32_t stage_bytes, const uint32_t slot_stages,
                              const uint64_t hint) {
    // the role lambdas' stage loops are rolled: at the o_proj sites the constant 20-stage bounds would unroll
    // into a 6 KB TMA-issue body per site (single-warp loops behind mbarrier waits, so the loop costs nothing)
#pragma unroll 1
        for (uint32_t s = 0; s < stages; s += slot_stages) {
            const uint32_t deliveries = umin(slot_stages, stages - s);
            ptx::mbar_wait_parity_poll(&empty_bars[weight_ring.slot], weight_ring.phase ^ 1);
            if (ptx::elect_one_sync()) {
                ptx::mbar_arrive_expect_tx(&full_bars[weight_ring.slot], deliveries * (stage_bytes + kXBytes));
#pragma unroll(cfg::kWorld.text_diet ? 1 : 4)
                for (uint32_t d = 0; d < deliveries; ++d)
                    ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + weight_ring.slot * kSlotBytes + d * stage_bytes,
                                                           &tm, static_cast<int32_t>((k0 + s + d) * k_elems),
                                                           static_cast<int32_t>(w_outer), &full_bars[weight_ring.slot],
                                                           hint);
            }
            __syncwarp();
            weight_ring.advance();
        }
    };
    // ---- warp 0: the weight stream ----
    // every claimed unit's stages back to back on one ring; never waits on an activation
    if (warp_idx == kPostAttnWProducer) {
        if constexpr (Attention::kGlobalAttn) {
            // slot 0's four o_proj stages first (slots 1..4 are the attention's carve until its done barrier)
            if (has_oproj) stream_w(tmap_wo, kSlotStages, 0, oproj_outer, 64, kStageBytes, kSlotStages, kOprojHint);
        }
        if (has_oproj) {
            const auto stream_oproj_w = [&]() {
                if constexpr (Attention::kGlobalAttn) {
                    Attention::wait_attn(smem_buf);
                    stream_w(tmap_wo, kOprojStages - kSlotStages, kSlotStages, oproj_outer, 64, kStageBytes,
                             kSlotStages, kOprojHint);
                    return;
                }
                // the sliding-window attention: slot 0's four stages under the attention, the rest behind its done
                // (the stream's traffic off the attention's KV loads; the o_proj MMA starts on slot 0 at x and the
                // ring refills under it)
                stream_w(tmap_wo, kSlotStages, 0, oproj_outer, 64, kStageBytes, kSlotStages, kOprojHint);
                Attention::wait_attn(smem_buf);
                stream_w(tmap_wo, kOprojStages - kSlotStages, kSlotStages, oproj_outer, 64, kStageBytes, kSlotStages,
                         kOprojHint);
            };
            stream_oproj_w();
            // the claimed units' streams wait for this CTA's o_proj MMAs: their HBM traffic would slow the slower
            // clusters' o_proj refills, and every row waits for the slowest push
            ptx::mbar_wait_parity_poll(&tmem_full[0], 0);
        }
        // the global attention's carve is the ring; the sliding-window attention keeps every stream off its KV loads
        Attention::wait_attn(smem_buf);
        for (uint32_t k = 0;; ++k) {
            const auto u = take_unit(k);
            if (u.type == kUnitNone) break;
            if (u.type == kUnitPooler) {
                stream_w(tmap_wp, kPoolerUnitStages, pr_k0 + (rank % kPoolerCk) * kPoolerUnitStages,
                         (pool_tile0 + (kW / kPoolerCk) * u.index + rank / kPoolerCk) * BN, 64, kStageBytes,
                         kSlotStages, ptx::kEvictFirst);
            }
            if (u.type == kUnitRouter) {
                uint32_t nblock, half, slice, k_begin, k_iters;
                router_geom(u.index, nblock, half, slice, k_begin, k_iters);
                stream_w(tmap_wr, k_iters, k_begin, nblock * kRouterBN + half * kRouterRows, kRouterBk,
                         kRouterStageBytes, kRouterSlotStages, ra.w_hint);
            }
            __syncwarp();
        }
        if (is_front_cluster) front_unit();
    }

    // ---- warp 1: the activation feeder ----
    // the grid wait for o_proj's x, one ticketed wait for h3, the launch ticket
    if (warp_idx == kPostAttnGate) {
        RingCursor w;
        const auto stream_x = [&](const CUtensorMap& tm, const uint32_t stages, const uint32_t k0,
                                  const uint32_t k_elems, const uint32_t slot_stages) {
#pragma unroll 1
            for (uint32_t s = 0; s < stages; s += slot_stages) {
                const uint32_t deliveries = umin(slot_stages, stages - s);
                ptx::mbar_wait_parity_poll(&empty_bars[w.slot], w.phase ^ 1);
                if (ptx::elect_one_sync()) {
#pragma unroll(cfg::kWorld.text_diet ? 1 : 4)
                    for (uint32_t d = 0; d < deliveries; ++d)
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + w.slot * kSlotBytes + kSlotXOff + d * kXTile,
                                                               &tm, static_cast<int32_t>((k0 + s + d) * k_elems),
                                                               /*y=*/0, &full_bars[w.slot], ptx::kEvictNormal);
                }
                __syncwarp();
                w.advance();
            }
        };
        ptx::griddepcontrol_wait();
        if constexpr (!Attention::kGlobalAttn) {
            // the sliding-window attention: the epoch bump, then the post-attention x map prefetched behind the
            // attention's done, ahead of the x wait
            Attention::bump_epoch(attention, smem_buf, lane_idx);
            Attention::wait_attn(smem_buf);
            if (lane_idx == 0) ptx::prefetch_tensormap(&tmap_xa);
        }
        Attention::wait(attention, smem_buf);
        // the first slot's x goes out ahead of the ticket's cold global load, the ticket is read and
        // published while that TMA is in flight, and the rest follows; the item warps wait on the ticket, so it
        // is published before the feeder can block on a ring slot
        if (has_oproj) stream_x(tmap_xa, kSlotStages, 0, 64, kSlotStages);
        ticket = a.launch_ctr[blockIdx.x];
        if (ptx::elect_one_sync()) {
            s_ticket = ticket;
            ptx::mbar_arrive(ticket_bar);
        }
        if (has_oproj) stream_x(tmap_xa, kOprojStages - kSlotStages, kSlotStages, 64, kSlotStages);
        // every post-h3 unit waits once for h3 (the launch's PDL trigger is at the next-layer phase's hand-over,
        // layer.cu)
        bool h3_waited = false;
        const auto h3_gate = [&]() {
            if (!h3_waited) {
                handoff::ticket_wait(a.h3_ready, kH3Arrivals * (ticket + 1));
            }
            h3_waited = true;
        };
        if (lane_idx == 0) {
            ptx::prefetch_tensormap(&tmap_xh);
            ptx::prefetch_tensormap(&tmap_xf);
        }
        for (uint32_t k = 0;; ++k) {
            const auto u = take_unit(k);
            if (u.type == kUnitNone) break;
            if (u.type != kUnitArRow) h3_gate();
            if (u.type == kUnitPooler) {
                stream_x(tmap_xh, kPoolerUnitStages, pr_k0 + (rank % kPoolerCk) * kPoolerUnitStages, 64, kSlotStages);
            }
            if (u.type == kUnitRouter) {
                uint32_t nblock, half, slice, k_begin, k_iters;
                router_geom(u.index, nblock, half, slice, k_begin, k_iters);
                stream_x(tmap_xf, k_iters, k_begin, kRouterBk, kRouterSlotStages);
            }
            __syncwarp();
        }
        if (is_front_cluster) {
            h3_gate();
            front_unit();
        }
        // this launch's ticket consumed: the next launch of this body reads it + 1
        if (ptx::elect_one_sync()) a.launch_ctr[blockIdx.x] = ticket + 1;
    }

    // ---- warp 2: the MMA issuer ----
    // unit g on accumulator g & 1, behind that accumulator's previous drain
    if (warp_idx == kPostAttnMma) {
        if constexpr (Attention::kGlobalAttn) {
            Attention::wait_attn(smem_buf);
            ptx::tcgen05_fence_after_thread_sync();
            taddr = tmem_addr_storage[0];
        }
        constexpr uint32_t idesc = ptx::mma_inst_desc_f16(BN, kM);
        constexpr uint32_t idesc_tf32 = ptx::mma_inst_desc_tf32(kRouterRows, kM);
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w);
        constexpr uint32_t kXBase = kSlotXOff;  // the x tiles' base in the carve
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint16_t, 64, 128>(smem_w + kXBase);
        const uint64_t w_desc0_tf32 = ptx::mma_smem_desc_k_major<uint32_t, kRouterBk, 128>(smem_w);
        const uint64_t x_desc0_tf32 = ptx::mma_smem_desc_k_major<uint32_t, kRouterBk, 128>(smem_w + kXBase);
        RingCursor w;
        uint32_t g = 0;
        // tf32: the router unit, 64-row tf32 stages of 32 K, four k8 atoms each
        const auto mma_unit = [&](const uint32_t stages, const bool tf32) {
            const uint32_t b = g & 1u;
            const uint32_t slot_stages = tf32 ? kRouterSlotStages : kSlotStages;
            const uint32_t stage_bytes = tf32 ? kRouterStageBytes : kStageBytes;
            if (g >= 2) ptx::mbar_wait_parity_poll(&tmem_empty[b], ((g >> 1) - 1) & 1u);
            ptx::tcgen05_fence_after_thread_sync();
#pragma unroll 1
            for (uint32_t s = 0; s < stages; s += slot_stages) {
                const uint32_t deliveries = umin(slot_stages, stages - s);
                ptx::mbar_wait_parity_poll(&full_bars[w.slot], w.phase);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one_sync()) {
#pragma unroll 1
                    for (uint32_t d = 0; d < deliveries; ++d) {
#pragma unroll
                        for (uint32_t kk = 0; kk < 4; ++kk) {
                            const uint64_t da =
                                (tf32 ? w_desc0_tf32 : w_desc0) + (w.slot * kSlotBytes + d * stage_bytes) / 16 + kk * 2;
                            // the x tile: the slot's, behind its W stages
                            const uint64_t db =
                                (tf32 ? x_desc0_tf32 : x_desc0) + ((w.slot * kSlotBytes + d * kXTile) / 16) + kk * 2;
                            const uint32_t scale_c = (s > 0 || d > 0 || kk > 0) ? 1u : 0u;
                            if (tf32)
                                ptx::tcgen05_mma_tf32(taddr + kAccCols * b, da, db, idesc_tf32, scale_c);
                            else
                                ptx::tcgen05_mma_f16(taddr + kAccCols * b, da, db, idesc, scale_c);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&empty_bars[w.slot]);
                    if (s + slot_stages >= stages) ptx::tcgen05_commit_arrive(&tmem_full[b]);
                }
                __syncwarp();
                w.advance();
            }
            ++g;
        };
        if (has_oproj) mma_unit(kOprojStages, false);
        for (uint32_t k = 0;; ++k) {
            const auto u = take_unit(k);
            if (u.type == kUnitNone) break;
            if (u.type == kUnitPooler) {
                mma_unit(kPoolerUnitStages, false);
            }
            if (u.type == kUnitRouter) {
                uint32_t nblock, half, slice, k_begin, k_iters;
                router_geom(u.index, nblock, half, slice, k_begin, k_iters);
                mma_unit(k_iters, true);
            }
            __syncwarp();
        }
        if (is_front_cluster) front_unit();
        // the last commit on each slot's empty barrier is never retired by a producer refill: drain them
        // before this warp leaves, so every async arrival on the pool lands ahead of the exit cluster
        // barrier and the phase boundary's invalidation. Slots below w.slot were visited in the current turn (parity
        // w.phase), the rest in the previous one (w.phase ^ 1; a never-visited slot completes that parity
        // trivially).
#pragma unroll(cfg::kWorld.text_diet ? 1 : kSlots)
        for (uint32_t s = 0; s < kSlots; ++s)
            ptx::mbar_wait_parity_poll(&empty_bars[s], s < w.slot ? w.phase : w.phase ^ 1u);
    }

    // ---- warp 3: TMEM, the code prefetch, the expert prefetch ----
    if (warp_idx == kPostAttnAux) {
        // the code prefetch (layer.cu) ahead of the grid wait: this warp is idle until its claims begin; the
        // global attention ran it already
        if constexpr (!Attention::kGlobalAttn) {
            if (code_prefetch_cta()) code_prefetch_issue();
        }
        // this warp walks the static program like every role (it has no unit duty)
        for (uint32_t k = 0;; ++k) {
            const auto u = take_unit(k);
            if (u.type == kUnitNone) {
                break;
            }
            __syncwarp();
        }
        // the expert prefetch (moe.cuh) under the attention, behind this warp's claims and
        // ahead of its front-item part; it reads the previous step's history, nothing of the predecessor
        // grid. Its histogram is the front unit's scratch: written by the front item, at the post-attention carve's
        // tail, and on this warp only after the expert prefetch in program order; its picks (hot) feed the front item's
        // tile list.
        Attention::run_prefetch(prefetch, reinterpret_cast<uint32_t*>(&front_scratch), front_scratch.hot, lane_idx);
        if (is_front_cluster) front_unit();
    }

    // ---- warps 4..7: the drains and the unit epilogues; warps 4..11: the AR row and the nq row ----
    if (warp_idx >= kPostAttnEpi0) {
        // quad: the warp's index among the item warps, its TMEM lane band and tile rows
        const uint32_t quad = warp_idx - kPostAttnEpi0;
        const bool epi_warp = warp_idx < kPostAttnEpi0 + kDrainWarps;
        const uint32_t item_warp = warp_idx - uint32_t(kPostAttnItemSpan.warp0);
        const uint32_t item_tid = item_warp * 32 + lane_idx;
        uint32_t g = 0;
        uint32_t nq_row = kM;  // kM = this CTA carries no nq row; else the row its deferred item owns
        // this warp's 16-column blocks of the accumulator: all of them, or (the split) those of its half of the item
        // warps
        // the accumulator into registers (lane < 16 of each quad holds one W row's tokens), the TMEM half
        // released
        const auto drain_regs = [&](uint32_t (&v)[kM]) {
            const uint32_t b = g & 1u;
            ptx::mbar_wait_parity_poll(&tmem_full[b], (g >> 1) & 1u);
            ptx::tcgen05_fence_after_thread_sync();
            ptx::tcgen05_ld_32x32b_cols<kM>(taddr + kAccCols * b, v);
            ptx::tcgen05_wait_ld();
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            if (warp_idx == kPostAttnEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive(&tmem_empty[b]);
        };
        const auto store_tile = [&](const uint32_t (&v)[kM]) {
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                for (uint32_t c = 0; c < kM; ++c)
                    out_tile[c * BN + row] = __float2bfloat16_rn(__int_as_float(v[c]));
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
        };
        const auto drain = [&]() {
            uint32_t v[kM];
            drain_regs(v);
            store_tile(v);
        };
        // the pooled tile's bf16 packets (out_tile token-major) into every gather-group rank's pooled row:
        // one 16 B packet per epi thread per round, system scope, the group's far half first, each source
        // rotated by its rank (next_layer.cuh's order); the packets are the nq rows' readiness
        const auto ag_push_tile = [&](const uint32_t (&v)[kM], const uint32_t tile) {
            store_tile(v);
            const uint32_t g0 = a.rank - a.rank % kPoolAgGroup;
#pragma unroll
            for (uint32_t r = 0; r < (kM * 8 + kEpiThreads - 1) / kEpiThreads; ++r) {
                const uint32_t p = r * kEpiThreads + quad * 32 + lane_idx, tok = p >> 3, c8 = (p & 7u) * 8u;
                if (kM * 8 % kEpiThreads != 0 && tok >= kM) break;
                const uint4 val = *reinterpret_cast<const uint4*>(out_tile + tok * BN + c8);
                const size_t col = size_t(kPoolAgOff) + size_t(tok) * kPooledRow + size_t(tile) * BN + c8;
#pragma unroll
                for (uint32_t i = 0; i < kPoolAgGroup; ++i) {
                    const uint32_t q = g0 + (a.rank + kPoolAgGroup / 2 + i) % kPoolAgGroup;
                    ptx::st_relaxed_sys_u4(
                        reinterpret_cast<uint4*>(reinterpret_cast<nv_bfloat16*>(a.peer_bases[q]) + col), val);
                }
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the tile is free for the next drain
        };
        // the K-window variant: this rank's slot in every rank's o_proj plane set (bf16 elements) at the launch
        // ticket's phase; the pooler tile's fp32 [kM, 64] partial staged token-major in the out region and pushed as
        // kM x 16 packets into the pooled partial of slot [phase][a.rank] on every rank
        [[maybe_unused]] const int64_t pr_slot = (static_cast<int64_t>(phase) * kRanks + a.rank) * a.plane_stride;
        const auto pr_push_tile = [&](const uint32_t (&v)[kM], const uint32_t tile) {
            float* const stage = reinterpret_cast<float*>(smem_buf + kOutOff);
            if (lane_idx < 16) {
                const uint32_t row = quad * 16 + lane_idx;
#pragma unroll
                for (uint32_t c = 0; c < kM; ++c)
                    stage[c * BN + row] = __int_as_float(v[c]);
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);
            float* dst[kRanks];
#pragma unroll
            for (uint32_t r = 0; r < kRanks; ++r)
                dst[r] =
                    reinterpret_cast<float*>(reinterpret_cast<nv_bfloat16*>(a.peer_bases[r]) + pr_slot + kPrPooledOff);
            const uint32_t tid = quad * 32 + lane_idx;
#pragma unroll
            for (uint32_t i = 0; i < kM * 16 / kEpiThreads; ++i) {  // two packets per thread at m16
                const uint32_t p = tid + i * kEpiThreads, tok = p >> 4, c4 = (p & 15u) * 4u;
                rowops::ar_lamport_push_packet4f<kRanks>(dst, tok * kPooledRow + tile * BN + c4,
                                                         *reinterpret_cast<const uint4*>(stage + tok * BN + c4));
            }
            ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the stage is free for the next drain
        };
        uint32_t pooler_uses = 0;  // this CTA's pooler units so far (the xfer barriers' phase)
        uint32_t topk_uses = 0;    // this CTA's top-k rows so far (reduce_bar's phase)
        const uint32_t reduce_smem =
            ptx::to_shared(smem_buf + kOutOff);  // the reduction staging (out_tile + xfer are idle here)
        if (has_oproj && epi_warp) {
            using Push = TileEpiArPush<kHidden, kRanks, kM>;
            const typename Push::Args push{a.peer_bases, a.plane_stride, a.rank};
            drain();
            Push::push(push, phase, oproj_outer, out_tile, quad, lane_idx);
            ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the tile is free for the next drain
        }
        if (has_oproj) ++g;
        // the item warps' first claimed unit reads a predecessor's output (the residual stream)
        ptx::griddepcontrol_wait();
        if (warp_idx == kPostAttnEpi0 && lane_idx == 0) ptx::prefetch_tensormap(&tmap_pooled);
        uint32_t xparity = 0;
        for (uint32_t k = 0;; ++k) {
            const auto u = take_unit(k);
            if (u.type == kUnitNone) break;
            if (u.type == kUnitArRow) {
                // the row's quarter on this CTA: vw = rank * 8 + the item warp; the packets of the kRanks
                // ranks at this launch's phase; the two rounds on the cluster-scope exchange mbarrier
                const uint32_t row = u.index;
                const uint32_t vw = rank * 8 + item_warp;
                nv_bfloat16* const my_base = reinterpret_cast<nv_bfloat16*>(a.peer_bases[a.rank]);
                const int64_t phase_0 = static_cast<int64_t>(phase) * kRanks * a.plane_stride;
                const nv_bfloat16* plane_rows[kRanks];
                nv_bfloat16* restore_rows[kRanks];
#pragma unroll
                for (uint32_t r = 0; r < kRanks; ++r) {
                    nv_bfloat16* const slot = my_base + phase_0 + r * a.plane_stride + row * kHidden;
                    plane_rows[r] = slot;
                    restore_rows[r] = slot;
                }
                // the row's operand arrays, copied from the slots above (the copy is kept for the schedule, README.md)
                constexpr uint32_t kRowPlanes = kRanks;
                const nv_bfloat16* row_planes[kRowPlanes];
                nv_bfloat16* row_restore[kRowPlanes];
#pragma unroll
                for (uint32_t r = 0; r < kRanks; ++r) {
                    row_planes[r] = plane_rows[r];
                    row_restore[r] = restore_rows[r];
                }
                rowops::row_arnorm_cn4_row<kHidden, kRowPlanes>(
                    vw, lane_idx, a.rank, row_planes, a.stream_in + row * kHidden, a.alpha, a.beta, a.prenorm_w, 1e-8f,
                    1e-5f, a.stream_out + row * kHidden, a.h3 + row * kHidden, a.h3f + row * kHidden,
                    [&](const uint32_t& r, const float& part) {
                        // the exchange as a scatter: each warp's lane 0 writes its
                        // partial into every CTA's cstage (st.async to the peers,
                        // tx-counted on their round barrier; a plain store + an
                        // arrive locally), so no block barrier precedes the signal
                        // and no DSMEM gather follows it; the 32 partials land in
                        // the same vw order, so the butterfly is bitwise the
                        // gathered variant. The empty branch is kept for the schedule (README.md).
                        if (r == 0 && item_warp == 0 && lane_idx == 0 && k < 8) {
                        }
                        uint64_t* const bar = xchg_bar + r;
                        if (lane_idx == 0) {
                            cstage[r * 32 + vw] = part;
                            // the peers as a rotation in one rolled body: a
                            // `p != rank` predicate makes ptxas emit four blocks
                            // that each re-derive both addresses
                            const uint32_t src = ptx::to_shared(cstage + r * 32 + vw), sbar = ptx::to_shared(bar);
#pragma unroll 1
                            for (uint32_t q = 1; q < kW; ++q) {
                                const uint32_t p = (rank + q) % kW;
                                ptx::st_async_cluster_b32(ptx::mapa_shared_cluster(src, p), __float_as_uint(part),
                                                          ptx::mapa_shared_cluster(sbar, p));
                            }
                            if (item_warp == 0)
                                ptx::mbar_arrive_expect_tx(bar, (kW - 1) * 8 * 4);  // the peers' 24 partials
                            else
                                ptx::mbar_arrive(bar);
                        }
                        ptx::mbar_wait_parity_acquire_cluster_poll(bar, xparity);
                        if (r == 1) {
                            xparity ^= 1u;  // both round barriers complete once per row
                        }
                        return rowops::arnorm_vw_butterfly(cstage[r * 32 + lane_idx]);
                    });
                // publish first, restore after: h3_ready's waiters need h3, not the cleared planes;
                // the planes are reused two launches on, and this CTA (still running its restore)
                // keeps the next launch off its SM. framework.cuh item_publish's three
                // steps, inlined.
                ptx::fence_async_global();
                ptx::sync_aligned(block::kItem, handoff::kItemThreads);
                if (item_tid == 0) {
                    if constexpr (kH3Arrivals == kRowCtas * kM)
                        ptx::red_add_rel_b32(a.h3_ready, 1u);
                    else  // the first rows' CTAs add two: the count gains its power of two per launch
                        ptx::red_add_rel_b32(a.h3_ready, row < (kH3Arrivals - kRowCtas * kM) / kRowCtas ? 2u : 1u);
                }
                rowops::ar_lamport_restore_packet8<kRowPlanes>(row_restore, (vw * 32 + lane_idx) * 8);
            } else if (u.type == kUnitRouter) {
                if (epi_warp) {
                    uint32_t nblock, half, slice, k_begin, k_iters;
                    router_geom(u.index, nblock, half, slice, k_begin, k_iters);
                    const uint32_t b = g & 1u;
                    // the M = 64 accumulator's TMEM layout: W row r sits at lane 32 * (r / 16) + r % 16, so each
                    // quad's first 16 lanes hold rows 16q..16q+15
                    ptx::mbar_wait_parity_poll(&tmem_full[b], (g >> 1) & 1u);
                    ptx::tcgen05_fence_after_thread_sync();
                    uint32_t v[kM];
                    ptx::tcgen05_ld_32x32b_cols<kM>(taddr + kAccCols * b, v);
                    ptx::tcgen05_wait_ld();
                    ptx::sync_aligned(kEpiBarId, kEpiThreads);
                    if (warp_idx == kPostAttnEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive(&tmem_empty[b]);
                    if (lane_idx < 16) {
                        const uint32_t dp = quad * 16 + lane_idx;  // the W row within the half-block
                        float* ws_tile =
                            ra.ws + (static_cast<uint64_t>(slice) * 2 + nblock) * kM * kRouterBN + half * kRouterRows;
#pragma unroll
                        for (uint32_t c = 0; c < kM; ++c)
                            ws_tile[c * kRouterBN + dp] = __int_as_float(v[c]);
                    }
                    // the release-free variant: the top-k rows read the partials by bulk copies (each writing thread's
                    // async-proxy fence)
                    if constexpr (kRelFree) ptx::fence_async_global();
                    ptx::sync_aligned(kEpiBarId, kEpiThreads);
                    if constexpr (kRelFree) {
                        // the block's ticketed count, acquired by the top-k rows themselves: one release add per
                        // unit, the first kReduceBlock / 2 - split slices' units adding two (it gains kReduceBlock per
                        // launch)
                        static_assert(2 * cfg::kRouterSplit <= kReduceBlock && kReduceBlock <= 4 * cfg::kRouterSplit,
                                      "the weighted units");
                        if (quad == 0 && lane_idx == 0)
                            ptx::red_add_rel_b32(ra.counters + nblock,
                                                 slice < kReduceBlock / 2 - cfg::kRouterSplit ? 2u : 1u);
                    } else {
                        // the block's ticketed count: +1 per unit; the split-th arriver acquires the
                        // peers' partials, publishes every row's arrival and pads the count to
                        // kReduceBlock
                        if (quad == 0 && lane_idx == 0) {
                            const uint32_t tk = ptx::atomic_add_rel_b32(ra.counters + nblock, 1u);
                            if (tk % kReduceBlock + 1 == 2 * ra.split) {         // 2 halves x split units per n-block
                                (void)ptx::ld_acq_fenced(ra.counters + nblock);  // the peers' partials
                                ptx::fence_acq_rel_gpu();                        // one release for the burst
                                for (uint32_t row = 0; row < kM; ++row)
                                    ptx::red_add_b32(ra.arrival + row, 1u);
                                ptx::red_add_b32(ra.counters + nblock, kReduceBlock - 2 * ra.split);
                            }
                        }
                    }
                }
                ++g;
            } else if (u.type == kUnitTopk) {
                // one token per CTA: the row's own-ticket (the two n-blocks' last arrivers + this
                // owner), then the row reduces its own logits (n-block 0 on the first item warp,
                // n-block 1 on the second, the split partials in slice order), then top-k on the
                // first warp and the route arrival
                if (epi_warp) {
                    for (uint32_t tr = 0; tr < kTopkRows; ++tr) {
                        const uint32_t t = (kW * u.index + rank) * kTopkRows + tr;
                        constexpr uint32_t tgt = 3;
                        // the word advances by exactly kArrivalBlock per launch, so the
                        // launch's base is the ticket's
                        const uint32_t base = kArrivalBlock * ticket;
                        // the release-free variant: the two router counts this launch reaches (each gains kReduceBlock)
                        [[maybe_unused]] const uint32_t ctgt = kReduceBlock * (ticket + 1);
                        constexpr uint32_t kSplit = cfg::kRouterSplit;  // checked at entry
                        if (quad == 0) {
                            if constexpr (!kRelFree) {
                                if (lane_idx == 0) ptx::red_add_b32(ra.arrival + t, 1u);  // the owner's arrive
                            }
                            // the routine's expert bias (1 KB) staged in smem under the
                            // spin: 8 values per lane
                            float bstage[8];
#pragma unroll
                            for (uint32_t j = 0; j < 8; ++j)
                                bstage[j] = ra.bias[j * 32 + lane_idx];
                            if constexpr (kRelFree) {
                                // the ticket wait's async-proxy fence orders the bulk copies of the partials below
                                ptx::count_spin(ra.counters, ctgt);
                                handoff::ticket_wait(ra.counters + 1, ctgt);
                            } else {
                                ptx::count_spin(ra.arrival + t, base + tgt);
                            }
#pragma unroll
                            for (uint32_t j = 0; j < 8; ++j)
                                topk_bias[j * 32 + lane_idx] = bstage[j];
                            __syncwarp();
                            // both n-blocks' split partial rows (512 B each) come in by
                            // bulk copies behind one mbarrier: one L2 round trip for the
                            // whole row; the adds below stay in slice order (bitwise the
                            // serial reduction)
                            if (ptx::elect_one_sync()) {
                                ptx::mbar_arrive_expect_tx(reduce_bar, 2 * kSplit * kRouterBN * 4);
#pragma unroll(cfg::kWorld.text_diet ? 1 : 2)
                                for (uint32_t nb = 0; nb < 2; ++nb)
#pragma unroll(cfg::kWorld.text_diet ? 4 : kSplit)
                                    for (uint32_t sl = 0; sl < kSplit; ++sl)
                                        ptx::cp_async_bulk_load(reduce_smem + (nb * kSplit + sl) * kRouterBN * 4,
                                                                ra.ws + ((sl * 2 + nb) * kM + t) * kRouterBN,
                                                                kRouterBN * 4, reduce_bar);
                            }
                            __syncwarp();
                        }
                        ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the copies are in flight; warps 4 and 5 reduce
                        if (quad < 2) {
                            ptx::mbar_wait_parity_poll(reduce_bar, topk_uses & 1u);
                            const uint32_t nb = quad;
                            const float4* p =
                                reinterpret_cast<const float4*>(smem_buf + kOutOff + nb * kSplit * kRouterBN * 4) +
                                lane_idx;
                            constexpr uint32_t row_stride = kRouterBN / 4;  // float4s per staged row
                            float4 acc = p[0];
#pragma unroll
                            for (uint32_t sl = 1; sl < kSplit; ++sl) {
                                const float4 q = p[sl * row_stride];
                                acc.x += q.x;
                                acc.y += q.y;
                                acc.z += q.z;
                                acc.w += q.w;
                            }
                            // the K-window variant: the row's logits partial out and back — this rank's reduced packet
                            // to every rank's slot, then the kRanks planes of this rank's copy summed in rank order
                            // (the markers are the cross-rank signal) and the copy's markers restored
                            if constexpr (kKWindow) {
                                const uint32_t col = t * (2 * kRouterBN) + nb * kRouterBN + lane_idx * 4;
                                nv_bfloat16* const my = reinterpret_cast<nv_bfloat16*>(a.peer_bases[a.rank]) +
                                                        static_cast<int64_t>(phase) * kRanks * a.plane_stride +
                                                        kPrLogitsOff;
                                float* dst[kRanks];
                                float* rrows[kRanks];
                                const float* rows[kRanks];
#pragma unroll
                                for (uint32_t r = 0; r < kRanks; ++r) {
                                    dst[r] = reinterpret_cast<float*>(reinterpret_cast<nv_bfloat16*>(a.peer_bases[r]) +
                                                                      pr_slot + kPrLogitsOff);
                                    rrows[r] = reinterpret_cast<float*>(my + r * a.plane_stride);
                                    rows[r] = rrows[r];
                                }
                                rowops::ar_lamport_push_packet4f<kRanks>(dst, col,
                                                                         *reinterpret_cast<const uint4*>(&acc));
                                acc = rowops::ar_lamport_consume_packet4f<kRanks>(rows, col);
                                rowops::ar_lamport_restore_packet4f<kRanks>(rrows, col);
                            }
                            *reinterpret_cast<float4*>(ra.logits + static_cast<uint64_t>(t) * 256 + nb * kRouterBN +
                                                       lane_idx * 4) = acc;
                            *reinterpret_cast<float4*>(topk_logits + nb * kRouterBN + lane_idx * 4) = acc;
                        }
                        ++topk_uses;
                        ptx::sync_aligned(kEpiBarId, kEpiThreads);  // both halves of the row's logits before the top-k
                        if (quad == 0) {
                            // the row's logits from smem (token 0 of a one-row view; the outputs pre-offset to token
                            // t); the route-early variant stores the int32 ids and the weights (what the front and the
                            // combine read) ahead of the route arrival and the int64 ids / local view (host-only)
                            // behind it
                            [[maybe_unused]] uint32_t tk_id = 0;
                            if constexpr (cfg::kWorld.route_early)
                                topk_ids_out(topk_logits, ra.router_mult, topk_bias, lane_idx, ra.num_local,
                                             ra.route_scale, ra.topk_idx32 + t * 8, ra.topk_w + t * 8, tk_id);
                            else
                                topk_row_out(topk_logits, ra.router_mult, topk_bias, lane_idx, ra.num_local,
                                             ra.topk_idx + t * 8, ra.topk_local + t * 8, ra.topk_w + t * 8,
                                             ra.route_scale, ra.topk_idx32 + t * 8);
                            __syncwarp();
                            if (lane_idx == 0) {
                                if constexpr (kRouteArrivals == kM)
                                    ptx::red_add_rel_b32(a.route_count, 1u);
                                else  // the first rows add two: the count gains its power of two per launch
                                    ptx::red_add_rel_b32(a.route_count, t < kRouteArrivals - kM ? 2u : 1u);
                                if constexpr (!kRelFree)
                                    ptx::red_add_b32(ra.arrival + t, kArrivalBlock - tgt);  // the block's pad
                            }
                            if constexpr (cfg::kWorld.route_early) {
                                if (lane_idx < 8) {
                                    ra.topk_idx[t * 8 + lane_idx] = static_cast<int64_t>(tk_id);
                                    ra.topk_local[t * 8 + lane_idx] =
                                        tk_id < ra.num_local ? static_cast<int64_t>(tk_id) : int64_t(-1);
                                }
                            }
                        }
                    }
                }
            } else if (u.type == kUnitPooler) {
                if (epi_warp) {
                    if constexpr (kKWindow) {
                        // one CTA per tile: its fp32 partial over the rank's K window to every rank's planes, then the
                        // tile's word (what the nq rows wait on; the packets' markers signal the peers' partials)
                        uint32_t v[kM];
                        drain_regs(v);
                        pr_push_tile(v, kW * u.index + rank);
                        if (warp_idx == kPostAttnEpi0) {
                            if (lane_idx == 0) ptx::red_add_rel_u64(a.pooled_ready + 1 + kW * u.index + rank, 1ull);
                        }
                    } else {
                        // the pair (even, odd) of the cluster-unit: the odd CTA sends its fp32
                        // accumulator into the even's xfer buffer (st.async, tx on the even's
                        // xfer_full); the even adds own + peer in that order, then stores the tile.
                        // The odd waits for the even to consume its previous send before sending
                        // again.
                        uint32_t v[kM];
                        drain_regs(v);
                        const uint32_t xp = pooler_uses & 1u;
                        if (rank % kPoolerCk == 1) {
                            // the previous send's consume, from the second send on
                            if (pooler_uses > 0) ptx::mbar_wait_parity_poll(xfer_empty, xp ^ 1u);
                            if (lane_idx < 16) {
                                const uint32_t peer = rank - 1;
                                const uint32_t dst =
                                    ptx::mapa_shared_cluster(ptx::to_shared(xfer + (quad * 16 + lane_idx) * kM), peer);
                                const uint32_t bar = ptx::mapa_shared_cluster(ptx::to_shared(xfer_full), peer);
#pragma unroll
                                for (uint32_t c = 0; c < kM; c += 4)
                                    ptx::st_async_cluster_v4(dst + c * 4,
                                                             make_uint4(v[c], v[c + 1], v[c + 2], v[c + 3]), bar);
                            }
                        } else {
                            if (warp_idx == kPostAttnEpi0 && ptx::elect_one_sync())
                                ptx::mbar_arrive_expect_tx(xfer_full, kXferBytes);
                            ptx::mbar_wait_parity_acquire_cluster_poll(xfer_full, xp);
                            if (lane_idx < 16) {
                                const float* p = xfer + (quad * 16 + lane_idx) * kM;
#pragma unroll
                                for (uint32_t c = 0; c < kM; ++c)
                                    v[c] = __float_as_int(__int_as_float(v[c]) + p[c]);
                            }
                            ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the xfer buffer is consumed
                            if (warp_idx == kPostAttnEpi0 && ptx::elect_one_sync())
                                ptx::mbar_arrive_cluster_release_cluster(xfer_empty, rank + 1);
                            ag_push_tile(v, pool_tile0 + (kW / kPoolerCk) * u.index + rank / kPoolerCk);
                        }
                        ++pooler_uses;
                    }
                    ptx::sync_aligned(kEpiBarId, kEpiThreads);  // the tile's store completed before the next drain
                }
                ++g;
                // the row's nq rides this unit: the cluster is busy until its tile lands anyway, and
                // the row's own wait (every tile) ends at the same moment. The item itself runs
                // after the claim loop, behind the TMEM dealloc (below).
                if (u.index < Q::kNqUnits) {
                    nq_row = kW * u.index + rank;
                    if (warp_idx == kPostAttnEpi0 && lane_idx == 0) {
                        // an empty branch, kept for the schedule (README.md)
                    }
                }
            }
            __syncwarp();
        }
        bool tmem_freed = false;
        // The deferred nq item. Its TMEM is dead before it runs (the claim loop returned its terminator, and
        // the item touches no TMEM), so the dealloc keeps its position relative to the loop and the MMA
        // warp, and only the item moves behind it: the dealloc runs under the item's wait instead of after
        // the row. The front cluster keeps the original order.
        if (nq_row < kM) {
            if (warp_idx == kPostAttnEpi0 && !is_front_cluster) {
                ptx::tcgen05_dealloc(taddr, kTmemCols);
                tmem_freed = true;
            }
            {
                // the gathered variant: the row's packets are its readiness (polled inside nq_item_gathered), so
                // this warp-level vote never waits (kept for the schedule (README.md)); the K-window variant: lanes
                // 0..7 poll the warp's eight local tile words concurrently (the peers' partials signal by their
                // markers)
                const uint32_t t0 = item_warp * kNqTilesPerWarp;
                const uint64_t tgt = uint64_t(ticket) + 1ull;
                bool ready = !kKWindow || lane_idx >= kNqTilesPerWarp;
                const uint32_t word0 = t0;
                while (true) {
                    if (!ready)
                        ready = static_cast<int64_t>(ptx::ld_acq_fenced_u64(a.pooled_ready + 1 + word0 + lane_idx) -
                                                     tgt) >= 0;
                    if (__all_sync(0xffffffffu, ready)) break;
                    __nanosleep(200);
                }
                __syncwarp();
            }
            // this carrier's row: polled from the gathered row (or summed over the partial planes), normed,
            // quantized, published
            if constexpr (kKWindow)
                nq_item_planes<kM, kPooledRow, kRanks>(ra.nq,
                                                       reinterpret_cast<nv_bfloat16*>(a.peer_bases[a.rank]) +
                                                           static_cast<int64_t>(phase) * kRanks * a.plane_stride +
                                                           kPrPooledOff,
                                                       a.plane_stride, nq_row, nq_staging);
            else
                nq_item_gathered<kM, kPooledRow>(
                    ra.nq, reinterpret_cast<nv_bfloat16*>(a.peer_bases[a.rank]) + kPoolAgOff, nq_row, nq_staging);
            // the row's q / sf / x_scale to the MoE phase's TMA reads (the first rows publish two
            // where kM is not a power of two: the count gains kRouteArrivals per launch)
            handoff::item_publish(a.nq_done, item_tid == 0, nq_row < kRouteArrivals - kM ? 2u : 1u);
        }
        if (is_front_cluster) front_unit();
        if (warp_idx == kPostAttnEpi0) {
            if (!tmem_freed) ptx::tcgen05_dealloc(taddr, kTmemCols);
        }
    }
    // the peers' DSMEM reads (cstage, the xfer buffer) complete before any exit
    ptx::cluster_sync();
}

}  // namespace mk
