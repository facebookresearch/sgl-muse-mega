#pragma once
// The MoE phase (moe_tp): the tensor-parallel expert layer of a layer image. The shard pack's layouts and the MoE
// phase's carve, the row window's byte math, the two cross-rank gathers, the front (route tables and tile list), g1
// (gate / up + SwiGLU + requant, z gathered on every rank), g2 (the full-K down projection, zb gathered), the MoE
// phase's entry, the combine rows, and the expert-weight L2 prefetch.

#include <cstdint>
#include <type_traits>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "framework.cuh"
#include "ptx.cuh"

// The MoE phase's interface: geometry, the at-rest weight layouts, the MoE phase's arguments (LayerArgs) and the
// cross-rank contracts of the tensor-parallel MoE on cfg::kRanks ranks. Each rank holds a shard of every expert (W1
// split on the intermediate, W2 on the output columns); the token rows arrive pre-quantized and routed (the front's
// tables), every rank gathers the full intermediate z of every slot (g1's register fan-out), runs g2 over the full K
// for its output columns and gathers zb; the combine (next_layer.cuh) reduces the slots into y.

namespace moe_tp {

// ---- geometry ----
inline constexpr int kRanks = mk::cfg::kRanks;
inline constexpr int kH = mk::cfg::kMoeHidden;  // hidden
inline constexpr int kIpr = mk::cfg::kIShard;   // intermediate per rank
inline constexpr int kI = kIpr * kRanks;        // intermediate (full)
inline constexpr int kE = mk::cfg::kExperts;    // experts
inline constexpr int kTopK = mk::cfg::kTopK;
// the MoE phase's token capacity (its route tables and tiles)
inline constexpr int kBMax = mk::cfg::kMoeM;
// the rendezvous spins one lane per rank (wait_all)
static_assert(kRanks >= 2 && kRanks <= 32, "the layer kernel waits on one warp of rank lanes");
// kBf16: the bf16 MoE layers: x = the normed bf16 token rows (no x_scale / x_sf), W1 / W2 bf16 (the same
// pair-tile and h-block packs, 64 elements = 128 B per K stage), tcgen05 kind::f16 MMAs (4 x K16 a stage), the g1
// epilogue's h = bf16(silu(bf16 gate) x bf16 up) into a bf16 z surface (no SF / exponent planes), alpha 1 at g2; the
// front, both cross-rank gathers, zb and the combine are the fp4 layers'.
inline constexpr bool kBf16 = mk::kImage.bf16;
// bytes of n weight / activation elements in the layer's format
__host__ __device__ constexpr int elem_bytes(int n) {
    return kBf16 ? 2 * n : n / 2;
}

// The two shard axes:
//   W1 column-shard on I  -> rank r owns i-features [r*kIpr, (r+1)*kIpr)
//   W2 column-shard on H, FULL K -> rank r owns output cols [r*kHpr, ...)
// (the g1 feature-block split lives with kFbW below — its width is derived.)
inline constexpr int kHpr = kH / kRanks;
inline constexpr int kHbPr = kHpr / 128;  // per rank
// the h blocks per g2 cell
inline constexpr int kG2Hb = 1;
static_assert(kHbPr % kG2Hb == 0, "whole h blocks per g2 cell");
inline constexpr int kG2CellsPt = kHbPr / kG2Hb;  // g2 cells per token tile
static_assert(kH % (128 * kRanks) == 0);

// NVFP4 scale-factor basis: E2M1 data with an E4M3 block scale per 16 values.
inline constexpr int kSfVec = 16;
// Feed chunk of both GEMMs: 128 B rows = 4 MMAs a stage — 256 fp4 elements (4 x K64 mxf4nvf4) or 64 bf16 elements
// (4 x K16 kind::f16).
inline constexpr int kBK = kBf16 ? 64 : 256;
inline constexpr int kBKBytes = 128;
static_assert(elem_bytes(kBK) == kBKBytes, "a K stage is one 128 B row");

// The register fan-out of both cross-rank gathers: the epilogue that produces a byte writes it to every rank's surface
// from the register it holds, so no push phase follows the epilogue and no grid-wide sync orders one against the other.

// the combine scheme: every rank receives every shard row and reduces every token (the mirror combine), one CTA per
// token, no reduce split
// the arrive words: rank p's anchor word at [p]. The device-local gsync words: [2] the anchor's fan-in,
// [3] the g2 producers' final-claim count, [4] the g2 steal cursor (both reset by the combine item's close), [7] the
// next-layer phase's claim cursor (layer.cu); the rest of kGsyncWords is unused.
inline constexpr int kArriveWords = kRanks;

// ---- per-token (dynamic) activation scaling ----
// Two activation scales are runtime values while the weight scales stay static: g1's epilogue multiplies by
// g1_alphas[e] x x_scale[token] (an exact fp32 per-token scale supplied with the activations; block SFs are relative
// to it), and h is requantized with e4m3 block scale factors derived from each 16-group's own runtime amax, with no
// per-token shift and no static input scale. z is gathered, so every rank's g2 consumes all kRanks producers' slices
// of a row under one per-row epilogue alpha: a row's codes must share a scale every producer derives identically.
// kHRow (the row window): each group's code is taken at a provisional power-of-two exponent e_b
// (prov_exp: the scale factor lands in (224, 448], the top e4m3 binade) and the exponent rides beside the SF byte in
// a third gathered plane (z_ex); both planes are stage-major ([I / 256][slots][16 B], zsf_off), so g2
// lands a stage's rows as one 2D box. Every consumer reduces the row's max exponent e_row over the full intermediate
// (identical on every rank) and recalibrates the row's SF bytes down to it (scale_down_byte: exact
// while normal, RN onto the subnormal grid far under the row max) on w2 and w3, in place in the landed block and NSF
// stages ahead of the SF stagers (g2_hrow_recal); g2's epilogue multiplies the column by 2^e_row
// (exact). Only groups ~14 binades under their row's max lose precision.
// !kHRow (the absolute codes): a group's SF is its own amax / 6 in h units, a normal e4m3 for an amax in
// [0.094, 2688]; below it the SF is subnormal (and the group requants to 0 under 6 x 2^-10, never NaN), above it the
// SF saturates at 448.
inline constexpr bool kHRow = mk::kHRow;
// the plane rides the g1 epilogue's fan-out (stage-major, zsf_off)

inline constexpr int kMaxSlots = kBMax * kTopK;
// the z box height (the token-indexed TMA box) and the token-tile height: 32 rows, one box per tile
inline constexpr int kNCap = 32;
inline constexpr int kTokBox = ((kBMax < kNCap ? kBMax : kNCap) + 7) & ~7;
inline constexpr int kNPad = (kTokBox + 31) & ~31;
static_assert(kNPad == kTokBox, "one z box per tile");
// The SF copy route: the A side's scale factors by tcgen05.cp from a blocked second image of W1's SF (g1),
// the B side's staged per stage by the four stager warps from a per-cell SF block the producer gathers. The
// quantizer's row-major x_sf is the input.
inline constexpr int kMaxTiles = (kE < kMaxSlots ? kE : kMaxSlots) + kMaxSlots / kNPad;
inline constexpr int kFrontWarps = 12;  // the MoE phase's twelve warps
// The front is external: the route tables and the tile list are built by the post-attention phase's front item
// (postattn.cuh) and reach the MoE phase through the post-attention phase's ticketed count (front_done), so the
// MoE phase runs no front and no publish. The epoch counts completed MoE passes (layer_epoch); the epoch close
// (next_layer.cuh) resets the steal cursor.
static_assert(kMaxSlots <= 2048, "the front item ranks the whole slot stream on one CTA");
// The MoE phase's role maps (framework.cuh): g1 is w0 producer, w1 MMA issuer, w2 SF copy issuer, w3 join / emit,
// w4..w11 the eight drainers; g2 keeps w0 / w1 and puts the SF stagers on the first drainer quad and the drainers on
// the second. Every drainer-set literal of the MoE phase derives from the three constants below.
inline constexpr mk::block::RoleMap kG1Roles{/*producer=*/0, /*gate=*/mk::block::kNone,
                                             /*mma=*/1,      /*aux=*/2,
                                             /*join=*/3,
                                             /*epi0=*/4,     /*n_epi=*/8};
static_assert(mk::block::valid(kG1Roles), "the g1 role map has the framework's shape");
inline constexpr int kCopyWarp = kG1Roles.aux;   // the g1 SF copy issuer
inline constexpr int kEmitWarp = kG1Roles.join;  // the g1 join/emit warp
inline constexpr int kDrain0 =
    kG1Roles.epi0;  // the first of the eight g1 drainers; g2 stages on the first four of them
static_assert(kCopyWarp == 2 && kEmitWarp == 3 && kDrain0 == 4, "the role map's warps");
inline constexpr mk::block::RoleMap kG2Roles{/*producer=*/0,
                                             /*gate=*/mk::block::kNone,
                                             /*mma=*/1,
                                             /*aux=*/mk::block::kNone,
                                             /*join=*/mk::block::kNone,
                                             /*epi0=*/kDrain0,
                                             /*n_epi=*/8};
static_assert(mk::block::valid(kG2Roles), "the g2 role map has the framework's shape");
// the MoE phase's block: twelve warps (one constant feeds the launch bounds, the launch and the grid_sync occupancy
// assert)
inline constexpr int kMoeThreads = kFrontWarps * 32;

// ---- the g1 pair-tile width (kFbW) and the smem accounting it decides ----
// A g1 cell covers kFbW = 128 expert features and its A tile carries gate and up for them: rows 0..127 gate,
// 128..255 up; two M = 128 MMAs read one 256-row tile.
// the front's table scratch, counted rather than sizeof'd (FrontScratch is declared later):
// counts / offsets / scan / apos, the warp byte table, wsum, nehist / nebase and the expert prefetch's hot words
// the warp byte table's row width: the experts, and at least the slot stash it doubles as (the front: two ints per
// front thread)
inline constexpr int kWcE = kE > 8 * kMoeThreads / kFrontWarps ? kE : 8 * kMoeThreads / kFrontWarps;
inline constexpr int kFrontFieldsSmem =
    4 * (3 * kE + (kE + 1)) + kFrontWarps * kWcE + 4 * 3 * (kE / 32) + 4 * 2 + 4 * (kE / 32);
constexpr int zstage_bytes(int fbw, int npad, bool ex_run = kHRow) {
    // the bf16 layers stages the cell's bf16 h alone (no SF / exponent runs)
    return kBf16 ? 2 * npad * fbw * 2 : 2 * (npad * (fbw / 2) + npad * (fbw / kSfVec) * (ex_run ? 2 : 1));
}
// the bf16 layers' z staging (4x the fp4 bytes) rides the dynamic carve past g1's ring (g1::ZST_OFF)
// instead of the front overlay: the overlay is static smem, which every phase of the layer launch pays
inline constexpr bool kZStageDyn = kBf16;
// The union inherits ZStageBuf's alignment, so its sizeof rounds up.
constexpr int front_static_smem(int fbw, int npad, bool ex_run = kHRow) {
    const int z = kZStageDyn ? 0 : zstage_bytes(fbw, npad, ex_run);
    const int b = kFrontFieldsSmem > z ? kFrontFieldsSmem : z;
    return (b + 15) & ~15;
}
// the dynamic budget: the launch ceiling less the front's static union and the 1 KB slack the MoE phase's other static
// words ride (barriers, claims, the row window's ring); the row window's staged exponent run (<= 1 KB) rides that slack
// too
constexpr int smem_budget(int fbw, int npad) {
    return mk::cfg::kSmemCeil - front_static_smem(fbw, npad, /*ex_run=*/false) - 1024;
}
inline constexpr int kFbW = mk::cfg::kFbW;
inline constexpr int kARows = 2 * kFbW;      // a pack block's rows = g1's A tile rows (gate + up)
inline constexpr int kFbFull = kIpr / kFbW;  // pack blocks per expert
inline constexpr int kFbPr = kFbFull;        // g1 cells per rank: one per pack block
static_assert(kFbW == 128, "the g1 pair tile is 128 features wide");
static_assert(kIpr % kFbW == 0, "the rank's I shard must be whole g1 feature blocks");
inline constexpr int kFrontStaticSmem = front_static_smem(kFbW, kNPad);

// the staged z emission: per-cell token-major mirrors of the two global extents a (slot, cell) owns, the cell's z
// run (kFbW features x fp4) and its SF run; two buffers (buf = ci & 1), WAR-guarded by the MB_ZSE barrier pair
// (g1), the footprint the front's table scratch (FrontZsOvl). kHRow: the cell's
// provisional-exponent run rides beside the SF run.
struct ZStageEx {
    uint8_t ex[kNPad * (kFbW / kSfVec)];
};
struct ZStageNoEx {};
struct alignas(16) ZStageBufFp4 : std::conditional_t<kHRow, ZStageEx, ZStageNoEx> {
    uint8_t z[kNPad * (kFbW / 2)];
    uint8_t sf[kNPad * (kFbW / kSfVec)];
};
// the bf16 layers's cell staging: the bf16 h columns alone ([pos][kFbW] bf16)
struct alignas(16) ZStageBufBf16 {
    uint8_t z[kNPad * kFbW * 2];
};
using ZStageBuf = std::conditional_t<kBf16, ZStageBufBf16, ZStageBufFp4>;
// the buffer's SF run (nullptr on the bf16 layers) - a dependent template, as zst_ex below
template <class S>
__device__ __forceinline__ auto* zst_sf(S& st) {
    if constexpr (!kBf16)
        return st.sf;
    else
        return static_cast<uint8_t*>(nullptr);
}
// the buffer's exponent run (nullptr when off) - a dependent template, so the off variant never names the field
template <class S>
__device__ __forceinline__ auto* zst_ex(S& st) {
    if constexpr (kHRow)
        return st.ex;
    else
        return static_cast<uint8_t*>(nullptr);
}
// the zb ring: the combine of MoE pass e reads zb while a peer's pass e + 1 g2 may already be pushing; a plane's reuse
// two passes later is ordered by the anchor, as is z's (two parities by row, kZPar).
// Parity p is the block at zb + p x zb_par_slots x kH, the stride a runtime argument (the region's slot capacity).
inline constexpr int kZbRing = 2;

// ---- at-rest layouts (shard-only, built by the weight loader) ----
// W1: [E][fb < kFbPr][ks][kARows][kFeedBytes]; the (e, fb) pair-tile row order is kFbW's (above).
// W2: [E][hb < kHbPr][ks][128 rows][kFeedBytes] (row = output column h, K = i).
// W1 SF: [E][fb][kARows][kH / 16 bytes], the same row order. W2 SF: [E][kHpr rows][kI / 16 bytes].
inline constexpr int kFeedBytes = kBKBytes;  // TMA row width
inline constexpr int kW1KSlices = elem_bytes(kH) / kFeedBytes;
inline constexpr int kW2KSlices = elem_bytes(kI) / kFeedBytes;
inline constexpr size_t kW1TileBytes = size_t(kARows) * elem_bytes(kH);  // per (e,fb)
inline constexpr size_t kW1TileSf = size_t(kARows) * (kH / kSfVec);
inline constexpr size_t kW1Bytes = size_t(kE) * kFbFull * kW1TileBytes;
inline constexpr size_t kW1SfBytes = size_t(kE) * kFbFull * kW1TileSf;
// A cell's z and SF emissions are wide stores based at the rank's own run (kFbPr whole tiles): 16 B for z, 8 B for SF
static_assert(kIpr % 32 == 0 && (kIpr / kSfVec) % 8 == 0, "a rank's z / SF runs keep the wide emissions aligned");
// a tensor map's extent is a promise about the buffer: the full region holds kFbFull blocks per expert
static_assert(size_t(kE) * kFbFull * kW1TileBytes == kW1Bytes, "the full W1 map's extent is its region");
static_assert(size_t(kE) * kFbFull * kW1TileSf == kW1SfBytes, "the full SF map's extent is its region");
inline constexpr size_t kW2Bytes = size_t(kE) * kHpr * elem_bytes(kI);
inline constexpr size_t kW2SfBytes = size_t(kE) * kHpr * (kI / kSfVec);

// ---- slot order (the front's) ----
// Experts ascending; within an expert, (token, k) pairs in row-major [t][k] order. Pads (idx == -1 or t >= b_live)
// are never dispatched; unused slot_tok / slot_k entries are -1 within the launch's reach (indices < B x kTopK,
// tiles < the B-derived tile cap); storage beyond the reach may hold an earlier launch's residue, and no reader is
// bounded by capacity.

struct RouteTables {
    int32_t* counts;    // [E]
    int32_t* offsets;   // [E+1] exclusive prefix
    int32_t* slot_tok;  // [kMaxSlots], -1 pad fill
    int32_t* slot_k;    // [kMaxSlots], -1 pad fill
    int32_t* tok_slot;  // [kMaxSlots] inverse [t*kTopK+k] -> slot, -1 pad
    int32_t* tile_e;    // [kMaxTiles] expert id; -1 fill
    int32_t* tile_r0;   // [kMaxTiles] first slot row within the expert
    int32_t* n_tiles;   // [1]
};

// CUtensorMap carries its own 64 B alignment, and this is LayerArgs' first member, so the descriptors stay aligned
// at their param-space address.
inline constexpr int kTmapsBase = 8;
inline constexpr int kTmaps = kTmapsBase + (kHRow ? 1 : 0);  // kHRow: + the exponent plane's stage-major box map
struct TmapSet {
    CUtensorMap tm[kTmaps];
};

// the MoE phase's arguments (the layout is the image's: the host binds it field for field)
struct LayerArgs {
    TmapSet tmaps;  // FIRST member: keeps 64-B alignment

    // inputs (replicated on every rank): the routing and the pre-quantized token rows
    const int32_t* topk_idx;  // [B][kTopK]
    const float* topk_w;      // [B][kTopK] f32 routing weights
    const uint8_t* x_fp4;     // [B][kH/2]
    const uint8_t* x_sf;      // [B][kH/kSfVec]
    int b_live;
    int B;

    // Weights (this rank's shard packs; see at-rest layouts above)
    const uint8_t* w1_fp4;
    const uint8_t* w1_sf;  // unused (the A side reads w1_sf_blk); kept for the schedule (README.md)
    const uint8_t* w2_fp4;
    const uint8_t* w2_sf;
    // Per-expert epilogue scale constants ([E] f32):
    //   g1 epi: h = silu(bf16(acc_g*g1a)) * bf16(acc_u*g1a) * g1c
    //   g2 epi: z = acc * g2a, ONE bf16 round
    const float* g1_alphas;
    const float* g1_scale_c;
    const float* g2_alphas;

    // Scratch (this rank)
    RouteTables route;
    // the gathered intermediate: every rank ends up with the full i row per slot; rank r writes bytes [r x
    // kIpr / 2, ...) locally and into every peer's copy (rank-disjoint bytes, no reduce), MoE pass e's rows from
    // (e & 1) x kZPar
    uint8_t* z_fp4;  // [kMaxSlots][kI/2]
    uint8_t* z_sf;   // [kMaxSlots][kI / kSfVec] (kHRow: the provisional codes; z_ex below holds their exponents)
    // the gathered g2 output, rounded to bf16 once at g2's epilogue: [kZbRing][zb_par_slots][kH] (the window
    // fills the first kMaxSlots rows of each parity block); rank r owns columns [r x kHpr, ..)
    __nv_bfloat16* zb;
    uint8_t* z_peer[kRanks];         // = peer p's z_fp4
    uint8_t* zsf_peer[kRanks];       // = peer p's z_sf
    __nv_bfloat16* zb_peer[kRanks];  // = peer p's zb

    // the anchor words: monotone, written once per MoE pass, never cleared; arrive_self[p] = the last
    // pass epoch peer p anchored
    uint32_t* arrive_self;          // [kArriveWords]
    uint32_t* arrive_peer[kRanks];  // = peer p's arrive_self
    uint32_t* gsync;                // device-local grid_sync + fan-in words
    uint32_t* epoch_dev;            // [1] monotone MoE pass counter
    uint64_t* clk;                  // unused; kept for the schedule (README.md)

    int rank;
    int n_sm;  // unused; kept for the schedule (README.md)

    // the post-expert norm weight, precomputed as gamma + 1 in bf16 (re-deriving it as 1 + f32(gamma) differs in
    // ulp)
    const __nv_bfloat16* norm_gamma_p1;  // [kH]
    float norm_eps;                      // 1e-8f (the model pin)

    // Output
    __nv_bfloat16* y;  // [B][kH] FULL y on every rank

    // the fp32 per-token first-GEMM activation scale (the quantizer's x_scale); h's block scale factors are
    // absolute, so no static input scale reaches the kernel
    const float* x_scale;  // [B]
    // the zb parity stride in slot rows: the ring's capacity (>= kMaxSlots), host-set from the zb extent
    int zb_par_slots;
    const uint8_t* w1_sf_blk;  // the blocked W1 SF image [E x kFbFull tiles][64 KB] (the SF copy route's source)
    CUtensorMap tm_sfab;       // the blocked image's tensor map (32 rows x 2 KB per tile)
    uint8_t* sf_img;           // unused; kept for the schedule (README.md)
    // kHRow: the gathered exponent plane [kMaxSlots][kI / kSfVec] int8 (one provisional exponent per 16-group,
    // beside z_sf's codes) and the peers' copies; passed always, unread without the row window
    uint8_t* z_ex;
    uint8_t* zex_peer[kRanks];
};
// the g2 producer takes the next cell's claim and decodes its route-table entry (two dependent L2 round trips) this
// many stages before the current cell's stage loop ends, and the claim ring carries the decoded cell beside the
// claim, so no role reads the tables on the cell boundary
inline constexpr int kG2TablePf = 3;

// the bf16 layers' variant: the plain twelve-warp MoE phase (no row window)
static_assert(!kBf16 || !kHRow, "kBf16: the bf16 layers run the plain MoE phase (no row window)");
// the next-layer phase's qkv-GEMM coordinates, stored by its prologue and read by nothing; the stores are kept for the
// schedule (README.md)
struct QkvCoords {
    const CUtensorMap* tm_w2;  // the qkv W tensormap (64-row x 64-element boxes)
    int32_t w2_outer;          // this CTA's W block's first row
    int32_t k2_0;              // this rank's first K stage of the split
};

// the combine's geometry: per 128-column n-block, 16 lanes x 8 consecutive columns (level A); the block stays 32
// lane groups, each reducing kCombNb / 32 n-blocks
inline constexpr int kCombNb = kH / 128;  // n-block partials
inline constexpr int kCombGroups = kCombNb < 32 ? kCombNb : 32;
static_assert(kCombNb % kCombGroups == 0, "the n-blocks must divide over the lane groups");
inline constexpr int kGsyncWords = 8 + kBMax;

// The row window's byte math (kHRow): the provisional per-block exponent, the per-row scale-factor
// recalibration (the half-up rule of a block-exponent NVFP4 quantizer), and the word-wise variant the g2 recalibrators
// run on four scale-factor bytes at once. Free of the MoE phase's geometry.

namespace hrow {

// 2^x as fp32 bits, x in [-126, 127] (the callers keep it there: the
// provisional exponent is clamped at -120 and tops out near +117)
__device__ __forceinline__ float pow2i(int x) {
    return __int_as_float((x + 127) << 23);
}
// The provisional block exponent: amax * 2^-e in (1024, 2048], one octave
// lower where it provably stays <= 1344, so the block's e4m3 scale factor
// (scaled amax / 6) lands in (224, 448] — the top binade, full mantissa,
// never saturated, never subnormal.
__device__ __forceinline__ int prov_exp(float amax) {
    const uint32_t b = __float_as_uint(amax);
    // 1344 / 1024 = 1.3125: the octave boundary is mantissa 0x280000.
    // Preserve the quoted rule's exponent for positive NaNs as well.
    const int e = int(b >> 23) - 138 + int((b & 0x7FFFFFu) > 0x280000u || b > 0x7F800000u);
    return e > -120 ? e : -120;
}
// the same rule branch-free: both results computed, one selected per byte (the subnormal arithmetic on an unsigned
// wrap of delta - e is discarded by the select); ~12 ops per byte with no divergence
__device__ __forceinline__ uint32_t scale_down_byte_nb(uint32_t byte, uint32_t delta) {
    const uint32_t e = byte >> 3;
    const uint32_t fast = byte - (delta << 3);
    const uint32_t sig = (e ? 8u : 0u) | (byte & 7u);
    const uint32_t raw_shift = delta - e + (e ? 1u : 0u);
    const uint32_t shift = raw_shift < 8u ? raw_shift : 8u;
    const uint32_t slow = (sig + (shift ? 1u << (shift - 1) : 0u)) >> shift;
    return e > delta ? fast : slow;
}
// Four scale-factor bytes of one row (one K64 cell) recalibrated to the row's
// max exponent e_row; ex holds the four provisional exponents as int8. When
// every lane's four bytes stay normal or zero, the warp uses packed byte
// arithmetic. Zero bytes stay zero at every delta and must not force a warp
// of live rows onto the scalar path when a tile includes dead rows. Otherwise
// every lane runs the branch-free byte rule. Bytes >= 0x80 are outside the SF contract (dead rows only) and
// the two paths are not asked to agree on them. Inactive rows may vote for
// the packed path regardless of their unread result. Every lane still calls.
// ZeroFast=false leaves live zero bytes to the scalar rule.
template <bool ZeroFast = true>
__device__ __forceinline__ uint32_t recal_word(uint32_t sf, uint32_t ex, int e_row, bool live = true) {
    const uint32_t er4 = uint32_t(e_row & 0xFF) * 0x01010101u;
    const uint32_t d4 = __vsub4(er4, ex);  // per byte e_row - e_b, >= 0 since e_row is the max
    const uint32_t e4 = (sf >> 3) & 0x0F0F0F0Fu;
    const uint32_t zero = ZeroFast ? __vcmpeq4(sf, 0u) : 0u;
    const bool packed = (__vcmpgtu4(e4, d4) | zero) == 0xFFFFFFFFu;
    if (__all_sync(0xFFFFFFFFu, packed || !live)) return __vsub4(sf, (d4 << 3) & 0xF8F8F8F8u) & ~zero;
    uint32_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b)
        out |= scale_down_byte_nb((sf >> (8 * b)) & 0xFFu, (d4 >> (8 * b)) & 0xFFu) << (8 * b);
    return out;
}
// The row max over packed int8 exponents: a SIMD max per word, four lanes reduced at the end.
__device__ __forceinline__ uint32_t max4(uint32_t a, uint32_t b) {
    return __vmaxs4(a, b);
}
__device__ __forceinline__ int reduce_max4(uint32_t w) {
    int m = int(int8_t(w & 0xFFu));
    m = max(m, int(int8_t((w >> 8) & 0xFFu)));
    m = max(m, int(int8_t((w >> 16) & 0xFFu)));
    return max(m, int(int8_t((w >> 24) & 0xFFu)));
}

}  // namespace hrow

// The MoE phase's two cross-rank gathers carry their own validity, so no system-scope release sits on either gather's
// path (a legal one compiles to MEMBAR.ALL.SYS, microseconds behind the stores it orders). Gather 1: g1's emit warp
// stores z (and, on the NVFP4 variant, its SF and exponent planes) into every rank's copy; every rank's g2 TMA-loads a
// tile's rows. Gather 2: g2's drainers store zb into every rank's copy; every rank's combine reads a token's slots.
// Every 8 B element of these surfaces holds a sentinel when empty, which no producer writes: z / z_sf / z_ex bytes 0x80
// on the NVFP4 variant (an e4m3 SF byte never has bit 7, a provisional exponent is >= -120, and g1 rewrites a code
// group that would read as the sentinel, lam_group), all ones on the bf16 variant's z and on zb (cvt.rn.bf16 never
// yields 0xFFFF). An element is empty iff its low 32-bit word is the sentinel's. Producers store elements
// st.relaxed.sys; consumers poll the same elements ld.relaxed.sys until none is empty. The argument (PTX memory model):
//  (A) a polled element R and the store W that wrote it are both strong at system scope over the same 8 B, so R takes
//      all of one write (single-copy atomicity); the location's writes are coherence-ordered reset(e - 2) < W(e) <
//      reset(e) < W(e + 2) by (C), so a non-empty R read this pass's W(e).
//  (B) gather 1's consumer is TMA (the async proxy): every lane's observations precede the leader's
//      fence.proxy.async.global through __syncwarp, so the TMA read that follows sees W(e) or a write after its own
//      completion. Gather 2's combine computes from the values it observed.
//  (C) reuse: z sits at a parity of the MoE pass epoch e (rows (e & 1) x kZPar), zb on its two-plane ring. Pass e's MoE
//      phase entry resets parity e - 1's z rows and zb plane (pass e - 1's polls and its combine's reads ended with its
//      launch). A reset reaches a peer's next store through the anchor: once per layer and rank, behind the MoE phase
//      entry's CTA barrier, a fan-in (atom.acq_rel.gpu) whose last CTA runs fence.release.sys and raises its word on
//      every rank, and every CTA of every rank acquires all kRanks words behind its g2. The same edge orders the reuse
//      of the all-reduce planes and the gather rows, whose lamport elements are restored within the layer.
//  (D) liveness: a consumer polls exactly the elements its producers write (a tile's live rows, a token's live slots;
//      every rank builds identical route tables, since the top-k reads bitwise-equal h3 rows on every rank), the anchor
//      depends only on the rank's own CTAs, and every wait is bounded (kLamTrapNs) and names what it saw. The anchor's
//      acquire compares the u32 epoch ordered (>=), so the words go back to their bind values (Buffers.reset) before
//      the epoch wraps at 2^32 passes; this stack resets them per request.
// Shared with any cross-rank protocol here: a store through a peer mapping and a load through the owner's name one
// location, griddepcontrol.wait carries the predecessor launch, and the proxy fence composes with the warp's sync.

inline constexpr uint64_t kZSent = kBf16 ? ~0ull : 0x8080808080808080ull;
inline constexpr uint64_t kZbSent = ~0ull;
inline constexpr uint32_t kZSent32 = uint32_t(kZSent);
// the z parity: MoE pass e's rows start at (e & 1) x kZPar; the largest window (16 rows) routes kZLive slots, and the
// g2 B box reads kTokBox rows from a tile's first
inline constexpr int kZPar = kMaxSlots / 2;
inline constexpr int kZLive = 16 * kTopK;
static_assert(mk::cfg::kWindowM <= 16 && kZLive + kTokBox <= kZPar, "a window's slots and the B box fit a parity");

// the lamport waits' bound: shorter than the all-reduce's (ptx.cuh kTrapSpinNs) and the ticket waits'
// (kTicketSpinPolls), so a gather that never completes reports itself before the waits queued behind it do
inline constexpr uint64_t kLamTrapNs = 10000000000;  // 10 s: a decode layer runs in ~0.1 ms
template <typename Pred, typename Observe>
__device__ __forceinline__ void lam_spin(Pred pred, const char* what, int rank, int word, uint32_t target,
                                         Observe observe) {
    uint32_t polls = 0;
    uint64_t t0 = 0;
    while (!pred()) {
        if ((++polls & ptx::kSpinCheckMask) == 0) {
            const uint64_t now = ptx::globaltimer_ns();
            if (t0 == 0) t0 = now;
            if (now - t0 > kLamTrapNs) ptx::spin_timeout_trap(what, rank, word, observe(), target, now - t0);
        }
    }
}

// MoE pass e's first z row
__device__ __forceinline__ int zpar_row0(uint32_t e) {
    return int(e & 1u) * kZPar;
}
// the 16 B chunk k of the z surfaces' rows [row0, row0 + m): the codes (kZc chunks a row), then the kPlanes planes'
// stage pieces (kHRow: SF and exponents, stage-major, zsf_off: one 16 B piece per stage and row)
inline constexpr int kZc = elem_bytes(kI) / 16;
inline constexpr int kZSt = kI / kBK;
inline constexpr int kPlanes = kHRow ? 2 : 0;
static_assert((kZc & (kZc - 1)) == 0 && (kZSt & (kZSt - 1)) == 0, "the chunk loop shifts and masks");
__device__ __forceinline__ uint8_t* lam_chunk(const LayerArgs& a, int row0, int m, int k) {
    if (k < m * kZc) return a.z_fp4 + size_t(row0 + k / kZc) * elem_bytes(kI) + size_t(k % kZc) * 16;
    k -= m * kZc;
    uint8_t* const plane = k < m * kZSt ? a.z_sf : a.z_ex;
    k = k < m * kZSt ? k : k - m * kZSt;
    return plane + size_t(k % kZSt) * (size_t(kMaxSlots) * 16) + size_t(row0 + k / kZSt) * 16;
}
// the reset: MoE pass e - 1's rows (parity (e - 1) & 1, every slot a window can route) back to the sentinel; thread t
// of nt per CTA takes its share of the grid's chunks
__device__ __forceinline__ void lam_reset_z(const LayerArgs& a, uint32_t e, int t, int nt) {
    const int row0 = int((e - 1u) & 1u) * kZPar;
    constexpr int kTotal = kZLive * (kZc + kPlanes * kZSt);
#pragma unroll 1
    for (int k = int(blockIdx.x) * nt + t; k < kTotal; k += int(gridDim.x) * nt)
        ptx::st_relaxed_sys_v2b64(lam_chunk(a, row0, kZLive, k), kZSent, kZSent);
}
// ... and MoE pass e - 1's zb plane, kZLive slots like z's rows (the combine read it in the previous launch; with two
// planes pass e + 1 is its next writer, behind pass e's anchor like z's rows)
__device__ __forceinline__ void lam_reset_zb(const LayerArgs& a, uint32_t e, int t, int nt) {
    static_assert(kZbRing == 2, "pass e - 1's zb plane is pass e + 1's");
    __nv_bfloat16* const plane = a.zb + size_t((e - 1u) % kZbRing) * size_t(a.zb_par_slots) * kH;
    constexpr int kTotal = kZLive * (kH * 2 / 16);  // the slot rows in 16 B chunks
#pragma unroll 1
    for (int k = int(blockIdx.x) * nt + t; k < kTotal; k += int(gridDim.x) * nt)
        ptx::st_relaxed_sys_v2b64(plane + size_t(k) * 8, kZbSent, kZbSent);
}

// The g2 gate (warp-collective): passes over every element of a tile's live rows until no lane saw the sentinel, a
// short backoff between passes. A pass is rounds of kLamRows rows, each one round trip: per lane eight 32 B code
// chunks and (kHRow) two (SF, exponent) piece pairs in flight before any compare.
inline constexpr int kLamRows = 256 / (elem_bytes(kI) / 32);  // rows per round: 32 lanes x 8 code chunks
static_assert(kLamRows >= 1 && (!kHRow || kLamRows * kZSt == 64), "a round's loads are 8 + 2 per lane");
inline constexpr uint32_t kLamBackoffNs = 64;
__device__ __forceinline__ bool lam_empty(uint64_t lo, uint64_t hi) {
    return uint32_t(lo) == kZSent32 || uint32_t(hi) == kZSent32;
}
// a plane piece p (row p / kZSt, stage p % kZSt) of the rows from row0
__device__ __forceinline__ size_t lam_piece(int row0, int p) {
    return size_t(p % kZSt) * (size_t(kMaxSlots) * 16) + size_t(row0 + p / kZSt) * 16;
}
// the tile's surfaces by value (the out-of-line gate takes no reference into the kernel parameters)
struct LamPlanes {
    const uint8_t* z;
    const uint8_t* sf;
    const uint8_t* ex;
};
__device__ __forceinline__ bool lam_pass(const LamPlanes& a, int row0, int m, int lane) {
    constexpr int kRowC = elem_bytes(kI) / 32;
    bool empty = false;
    const uint8_t* const zc = a.z + size_t(row0) * elem_bytes(kI);
    const int nz = m * kRowC, np = kHRow ? m * kZSt : 0;
#pragma unroll 1
    for (int r = 0; r < m; r += kLamRows) {
        uint64_t v[8][4], sf[2][2], ex[2][2];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int k = r * kRowC + 32 * b + lane;
            if (k < nz) ptx::ld_relaxed_sys_v4b64(zc + size_t(k) * 32, v[b]);
        }
        if constexpr (kHRow)
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int p = r * kZSt + 32 * b + lane;
                if (p < np) {
                    ptx::ld_relaxed_sys_v2b64(a.sf + lam_piece(row0, p), sf[b][0], sf[b][1]);
                    ptx::ld_relaxed_sys_v2b64(a.ex + lam_piece(row0, p), ex[b][0], ex[b][1]);
                }
            }
#pragma unroll
        for (int b = 0; b < 8; ++b)
            if (r * kRowC + 32 * b + lane < nz) empty |= lam_empty(v[b][0], v[b][1]) || lam_empty(v[b][2], v[b][3]);
        if constexpr (kHRow)
#pragma unroll
            for (int b = 0; b < 2; ++b)
                if (r * kZSt + 32 * b + lane < np)
                    empty |= lam_empty(sf[b][0], sf[b][1]) || lam_empty(ex[b][0], ex[b][1]);
    }
    return __any_sync(0xFFFFFFFFu, empty);
}
// the gate, one out-of-line copy for its two sites (the report on the trap path: word = row0 x 100 + m, target = the
// tile, observed = the first 32 B code chunk still empty, row x chunks-per-row + chunk, ~0 none)
__device__ __noinline__ void lam_poll_tile(const LamPlanes a, int rank, int row0, int m, int lane, int ti) {
    if (!lam_pass(a, row0, m, lane)) return;
    lam_spin(
        [&] {
            if (!lam_pass(a, row0, m, lane)) return true;
            __nanosleep(kLamBackoffNs);
            return false;
        },
        "z-tile lamport", rank, row0 * 100 + m, ti,
        [&] {
            for (int k = 0; k < m * (elem_bytes(kI) / 32); ++k) {
                uint64_t v[4];
                ptx::ld_relaxed_sys_v4b64(a.z + size_t(row0) * elem_bytes(kI) + size_t(k) * 32, v);
                if (lam_empty(v[0], v[1]) || lam_empty(v[2], v[3])) return uint32_t(k);
            }
            return ~0u;
        });
}

// The anchor's arrival (one lane of each CTA, behind the MoE phase entry's CTA barrier): the fan-in's last CTA releases
// every CTA's resets at system scope and raises this rank's word on every rank. The counter self-resets on the winner.
__device__ __forceinline__ void lam_anchor_arrive(const LayerArgs& a, uint32_t e, uint32_t* counter) {
    if (ptx::atomic_add_acq_rel_b32(counter, 1u) == gridDim.x - 1) {
        ptx::st_relaxed_gpu_u32(counter, 0u);
        ptx::fence_release_sys();
#pragma unroll
        for (int r = 0; r < kRanks; ++r)
            ptx::st_relaxed_sys_u32(a.arrive_peer[r] + a.rank, e);
    }
}
// the anchor's acquire, behind the CTA's g2 (warp-collective): every rank's pass-e word, lane r polls rank r
__device__ __forceinline__ void lam_anchor_acquire(const LayerArgs& a, uint32_t e, int lane) {
    if (lane < kRanks) {
        const uint32_t* const w = a.arrive_self + lane;
        lam_spin([&] { return ptx::ld_acq_sys_u32(w) >= e; }, "lamport anchor", a.rank, lane, e,
                 [&] { return ptx::ld_acq_sys_u32(w); });
    }
    __syncwarp();
}

// the MoE pass epoch: the counter counts completed passes (raised by the epoch close, next_layer.cuh), so it is stable
// for the whole MoE phase and every warp reads it at entry
__device__ __forceinline__ uint32_t layer_epoch(const LayerArgs& a) {
    return *reinterpret_cast<const volatile uint32_t*>(a.epoch_dev) + 1u;
}

// The routing tables and the token-tile list, built by the post-attention phase's front item (postattn.cuh)
// from the top-k ids: integer only, so the slot map is bit-identical on every rank and every replay.

// the front's arguments (a small POD: the front needs a few fields, not the MoE phase's argument struct)
struct FrontArgs {
    const int32_t* topk_idx;
    int B, b_live;
};

// The front's strided loops are kept rolled (#pragma unroll 1): each has a trip count of 0-2 at this shape (256
// experts over 384 threads, at most a tile per expert), and unrolled, half the item's text would be prologue for
// iterations that never run; the unit runs once per launch from a cold instruction cache, so fewer bytes to fetch
// beats fewer branches.

// Inclusive scan within a warp — 5 barrier-free shuffle rounds.
__device__ __forceinline__ int warp_incl_scan(int v) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(0xFFFFFFFFu, v, d);
        if ((int(threadIdx.x) & 31) >= d) v += u;
    }
    return v;
}

// front_tables' scratch, declared by the caller: its lifetime is the front phase only (g1 and g2 read the routes
// from global memory), so the MoE phase recycles the block as its z staging window (FrontZsOvl makes the aliasing
// explicit)
struct FrontScratch {
    // the entries derive inline from topk_idx (two coalesced global re-reads per entry instead of an smem
    // staging of the ids)
    int32_t counts[kE];
    int32_t offsets[kE + 1];
    int32_t scan[kE];
    int32_t apos[kE];
    alignas(4) uint8_t wc[kFrontWarps][kWcE];
    int32_t wsum[3][kE / 32];
    int32_t nehist[1];
    int32_t nebase[1];
    uint32_t hot[kE / 32];  // the expert prefetch's picks (bit e % 32 of word e / 32), written by the aux warp
};

// the recycle window: the front scratch and the g1 loop's two staging buffers have strictly sequential lifetimes, so
// a union; the smem budget accounts its max (kFrontStaticSmem), so neither member can squeeze the other
struct ZStageDyn {};  // kZStageDyn: the staging is g1's dynamic window (zstage_window)
union FrontZsOvl {
    FrontScratch front;
    std::conditional_t<kZStageDyn, ZStageDyn, ZStageBuf> zstage[2];
};
static_assert(sizeof(FrontZsOvl) <= kFrontStaticSmem, "the front's static count covers its union");

static_assert(kFrontWarps * kWcE / 4 >= 2 * kMoeThreads,
              "the slot stash reuses the warp-count table — two ints per thread");

// The routing tables, fused warp-scan variant: two Hillis-Steele block scans over kE would cost 32 __syncthreads() in
// one CTA; fused into one warp-scan pass they cost two. kCohIds (the front item): the ids were stored by other CTAs of
// the same launch behind a ticketed count, so every entry read is a gpu-scope relaxed load (a weak load may return the
// previous launch's line). kRegs (the front item's register variant): the whole stream is this block's, so each thread
// keeps its <= kRegEnt entries in registers from the histogram pass to the rank pass (one ids load, not two) and the
// rank pass writes the masked entries' -1 itself (no pre-fill pass over tok_slot). The same tables.
template <bool kCohIds = false, bool kRegs = false>
__device__ inline void front_tables(const FrontArgs& a, const RouteTables& rt, FrontScratch& fs) {
    constexpr int kRegEnt = (kMaxSlots + kMoeThreads - 1) / kMoeThreads;  // entries per thread at the window
    [[maybe_unused]] int eh[kRegs ? kRegEnt : 1];
    int32_t* const s_counts = fs.counts;
    int32_t* const s_offsets = fs.offsets;
    int32_t* const s_scan = fs.scan;
    int32_t* const s_apos = fs.apos;
    auto& s_wc = fs.wc;

    const int tid = threadIdx.x;
    const int n_entries = a.B * kTopK;
    // the entry stream [j_lo, j_hi) is the whole stream (n_entries is a run-time value, so min(0, n) does not fold);
    // the bounds are kept for the schedule (README.md)
    const int j_lo = min(0, n_entries), j_hi = min(j_lo + n_entries, n_entries);
    // the live-masked expert of entry j, read straight from global memory
    auto entry_e = [&](int j) {
        if (j / kTopK >= a.b_live) return -1;
        if constexpr (kCohIds) return int(ptx::ld_relaxed_b32(reinterpret_cast<const uint32_t*>(a.topk_idx + j)));
        return int(a.topk_idx[j]);
    };
#pragma unroll 1
    for (int e = tid; e < kE; e += blockDim.x) {
        s_counts[e] = 0;
        s_scan[e] = 0;
    }
    __syncthreads();
    // one entry's histogram step; an entry before j_lo would also seed the rank base (none: j_lo <= 0)
    auto count_entry = [&](int j, int e) {
        if (e < 0) return;
        atomicAdd(&s_counts[e], 1);
        if (j < j_lo) atomicAdd(&s_scan[e], 1);
    };
    if constexpr (kRegs) {
        if (n_entries > kRegEnt * int(blockDim.x)) __trap();  // the window's stream fits the registers
#pragma unroll
        for (int u = 0; u < kRegEnt; ++u) {
            const int j = tid + u * int(blockDim.x);
            eh[u] = j < n_entries ? entry_e(j) : -1;
        }
#pragma unroll
        for (int u = 0; u < kRegEnt; ++u)
            count_entry(tid + u * int(blockDim.x), eh[u]);
    } else {
        for (int j = tid; j < n_entries; j += blockDim.x)
            count_entry(j, entry_e(j));
    }
    __syncthreads();
    // the expert offsets and tile positions: one warp-scan pass
    {
        auto& s_wsum = fs.wsum;
        const int lane = tid & 31, warp = tid >> 5;
        // the prefetched experts' tiles lead the list, so the CTAs that take a second g1 cell start from L2
        const int cnt = (tid < kE) ? s_counts[tid] : 0;
        const int til = cnt > 0 ? (cnt + kNPad - 1) / kNPad : 0;
        const bool hot = tid < kE && ((fs.hot[tid >> 5] >> (tid & 31)) & 1u);
        int cs = warp_incl_scan(cnt), ts = warp_incl_scan(hot ? 0 : til), th = warp_incl_scan(hot ? til : 0);
        if (tid < kE && lane == 31) {
            s_wsum[0][warp] = cs;
            s_wsum[1][warp] = ts;
            s_wsum[2][warp] = th;
        }
        __syncthreads();
        if (tid < kE) {
            int nh = 0;
#pragma unroll 1
            for (int w = 0; w < kE / 32; ++w) {  // <= 8 adds beats a barrier
                nh += s_wsum[2][w];
                if (w < warp) {
                    cs += s_wsum[0][w];
                    ts += s_wsum[1][w];
                    th += s_wsum[2][w];
                }
            }
            s_apos[tid] = hot ? th : nh + ts;
            s_offsets[tid + 1] = cs;
            rt.offsets[tid + 1] = cs;
            rt.counts[tid] = s_counts[tid];
        }
        if (tid == 0) {
            s_offsets[0] = 0;
            rt.offsets[0] = 0;
        }
        __syncthreads();
    }
    // the tile list
    int nt = 0;
#pragma unroll 1
    for (int w = 0; w < kE / 32; ++w)
        nt += fs.wsum[1][w] + fs.wsum[2][w];
    // the prefetched experts ascending, then the rest ascending
    if (tid < kE && s_counts[tid] > 0) {
        const int ne = (s_counts[tid] + kNPad - 1) / kNPad;
        const int base = s_apos[tid] - ne;
#pragma unroll 1
        for (int tt = 0; tt < ne; ++tt) {
            rt.tile_e[base + tt] = tid;
            rt.tile_r0[base + tt] = tt * kNPad;
        }
    }
    // the fills are bounded at the launch's reach (n_entries slots, nt_cap tiles), not the capacity:
    // every runtime reader is bounded by n_tiles / n_entries / live_total; storage beyond the reach may
    // hold an earlier launch's residue
    const int nt_cap = (kE < n_entries ? kE : n_entries) + n_entries / kNPad;
#pragma unroll 1
    for (int i = nt + tid; i < nt_cap; i += blockDim.x) {
        rt.tile_e[i] = -1;
        rt.tile_r0[i] = -1;
    }
    if (tid == 0) *rt.n_tiles = nt;
    __syncthreads();
    // stable slot rank: entry j's rank = #{p < j : e_p == e_j}, built with a ballot + a byte prefix over the warps of
    // one round (linear in the stream); s_scan carries the running count of the rounds before.
    const int warp = tid >> 5, n_warps = int(blockDim.x) >> 5;
    const uint32_t lane_lt = (1u << (tid & 31)) - 1u;
    if constexpr (!kRegs) {
#pragma unroll 1
        for (int j = j_lo + int(threadIdx.x); j < j_hi; j += blockDim.x)
            rt.tok_slot[j] = -1;
    }
    __syncthreads();
    if constexpr (kRegs) {
        // the rank pass over the registers' rounds: round u = entries u x blockDim + tid, every round's barriers kept
#pragma unroll
        for (int u = 0; u < kRegEnt; ++u) {
            const int j0 = u * int(blockDim.x);
            if (j0 >= n_entries) break;  // uniform
#pragma unroll 1
            for (int i = tid; i < n_warps * kWcE / 4; i += blockDim.x)
                reinterpret_cast<uint32_t*>(s_wc)[i] = 0u;
            const int j = j0 + tid, e = eh[u];
            const uint32_t match = __match_any_sync(0xFFFFFFFFu, e);
            const int r_warp = __popc(match & lane_lt);
            __syncthreads();
            if (e >= 0 && r_warp == 0) s_wc[warp][e] = uint8_t(__popc(match));
            __syncthreads();
            if (e >= 0) {
                int r = s_scan[e] + r_warp;
#pragma unroll 1
                for (int w = 0; w < warp; ++w)
                    r += s_wc[w][e];
                const int s = s_offsets[e] + r;
                rt.slot_tok[s] = j / kTopK;
                rt.slot_k[s] = j % kTopK;
                rt.tok_slot[j] = s;
            } else if (j < n_entries) {
                rt.tok_slot[j] = -1;
            }
            __syncthreads();
            if (e >= 0 && r_warp == 0) atomicAdd(&s_scan[e], __popc(match));
            __syncthreads();
        }
    } else {
        // each thread double-buffers its own next-round entry in a register, so the global load's latency hides
        // behind the current round's ballot work and barriers
        int e_cur = (j_lo + tid < j_hi) ? entry_e(j_lo + tid) : -1;
        for (int j0 = j_lo; j0 < j_hi; j0 += blockDim.x) {
            const int jn = j0 + int(blockDim.x) + tid;
            const int e_next = (jn < j_hi) ? entry_e(jn) : -1;
#pragma unroll 1
            for (int i = tid; i < n_warps * kWcE / 4; i += blockDim.x)
                reinterpret_cast<uint32_t*>(s_wc)[i] = 0u;
            const int j = j0 + tid;
            const int e = e_cur;
            const uint32_t match = __match_any_sync(0xFFFFFFFFu, e);
            const int r_warp = __popc(match & lane_lt);
            __syncthreads();
            if (e >= 0 && r_warp == 0) s_wc[warp][e] = uint8_t(__popc(match));
            __syncthreads();
            if (e >= 0) {
                int r = s_scan[e] + r_warp;
#pragma unroll 1
                for (int w = 0; w < warp; ++w)
                    r += s_wc[w][e];
                const int s = s_offsets[e] + r;
                rt.slot_tok[s] = j / kTopK;
                rt.slot_k[s] = j % kTopK;
                rt.tok_slot[j] = s;
            }
            __syncthreads();
            if (e >= 0 && r_warp == 0) atomicAdd(&s_scan[e], __popc(match));
            __syncthreads();
            e_cur = e_next;
        }
    }
    const int live_total = s_offsets[kE];
#pragma unroll 1
    for (int s = live_total + int(threadIdx.x); s < n_entries; s += blockDim.x) {
        rt.slot_tok[s] = -1;
        rt.slot_k[s] = -1;
    }
}

// The MoE phase's shared GEMM pieces: the tensor-map slots, the NVFP4 requant, the
// scale-factor geometry and loads, a tile's cell, the smem budget.

// ---- shared geometry (kBK / kBKBytes above) ----
inline constexpr int kKPerTile = kBf16 ? kBK / 16 : kBK / 64;  // 4 MMAs per stage (K64 mxf4nvf4, K16 kind::f16)
inline constexpr uint32_t kOobRow = 2147483392;                // gather4 OOB row (zero-fill)
// The weight tiles have zero reuse (each (e, fb) / (e, hb) tile is read once) while z and zb are re-read:
// evict-first on the weight loads alone keeps the stream from flushing the re-read traffic.

// TM_G1_SFA stays zero (g1's A-side scale factors come from LayerArgs::tm_sfab), kept for the schedule (README.md)
enum TmapSlot {
    TM_G1_A = 0,
    TM_G1_SFA = 1,
    TM_G1_B = 2,
    TM_G1_SFB = 3,
    TM_G2_A = 4,
    TM_G2_SFA = 5,
    TM_G2_B = 6,
    TM_G2_SFB = 7,
};
// kHRow: the exponent plane's map (TM_G2_SFB's geometry); off, the slot does not exist and the name aliases slot 0
// (read under `if constexpr (kHRow)` only)
inline constexpr int TM_G2_EXB = kHRow ? kTmapsBase + 0 : 0;
static_assert(TM_G2_EXB < kTmaps, "kHRow: the exponent plane's map slot is inside TmapSet");

__device__ __forceinline__ void wtma3d(uint32_t dst, const CUtensorMap* tm, int32_t x, int32_t y, int32_t z,
                                       uint64_t* bar) {
    ptx::cp_async_bulk_tensor_3d_load_hint(dst, tm, x, y, z, bar,
                                           ptx::make_l2_cache_policy<ptx::L2EvictPolicy::EVICT_FIRST>());
}

// the MMA N of a cell's m rows: 8-aligned for mxf4nvf4, 16-aligned for kind::f16 at M = 128 (kBf16); the
// producer's gather quads, the MMA and the drains all size by it, and the padded columns are zero-filled rows never
// stored
__device__ __forceinline__ int ceil8(int m) {
    return kBf16 ? (m + 15) & ~15 : (m + 7) & ~7;
}

// ---- NVFP4 requant + silu, bit-exact to the reference: e4m3 RN-satfinite SF from block amax x (1/6 in FP32 RN)
// with no amax floor; silu = __expf, RN add, rcp.approx.ftz, RN mul ----
__device__ __forceinline__ uint8_t amax_to_ue4m3(float amax) {
    const float sf = amax * __int_as_float(0x3E2AAAAB);  // 1/6 in FP32 RN
    return uint8_t(ptx::cvt_pack_f32x2_to<ptx::e4m3>(0.0f, sf) & 0xFFu);
}
__device__ __forceinline__ float inv_sf_from_ue4m3(uint8_t code) {
    const uint32_t exp = (uint32_t(code) >> 3) & 0xFu;
    const uint32_t man = uint32_t(code) & 0x7u;
    float value;
    if (exp == 0) {
        value = float(man) * (1.0f / 512.0f);
    } else {
        value = 1.0f + float(man) * (1.0f / 8.0f);
        value *= __int_as_float(uint32_t(int(exp) - 7 + 127) << 23);
    }
    return value > 0.0f ? 1.0f / value : 0.0f;
}
__device__ __forceinline__ float silu_fast(float g) {
    return __fmul_rn(g, ptx::rcp_approx_ftz_f32(1.0f + __expf(-g)));
}

// ---- scale-factor geometry (block-16): one staged u32 carries 4 SF bytes = one K64 MMA cell ----
inline constexpr int kSfU32H = (kH / kSfVec) / 4;  // u32 per SF row, K = kH
inline constexpr int kSfU32I = (kI / kSfVec) / 4;  // u32 per SF row, K = kI
inline constexpr int kSfWords = 4;                 // u32 per row per BK stage
inline constexpr int kSfCols = kSfWords * 4;       // TMEM cols per tile per slot

// one SF row's whole BK stage into registers first (the hoist takes the bank-conflicted load off the store path)
__device__ __forceinline__ void sf_load_stage(uint32_t (&w)[kSfWords], const uint32_t* p) {
#pragma unroll
    for (int kk = 0; kk < kSfWords; ++kk)
        w[kk] = p[kk];
}
// The SF TMEM layout: the B-side scale image is read per M quadrant, so every 32-lane band holds every live N row
// group, group g at column kk * 4 + g. A row group is staged only when the cell is tall enough to have it
// (n_mma > 32 g).
static_assert(kNPad <= 128, "SF staging carries at most four 32-row N groups");
inline constexpr int kNGroups = kNPad / 32;  // 32-row N groups per tile
// the B-side SF row groups of this stage: group g (rows 32g..) only at cells taller than 32g rows (its TMEM columns
// are unread below that)
template <int NG>
__device__ __forceinline__ void sf_load_groups(uint32_t (&wb)[NG][kSfWords], const uint32_t* sfb_row0, int stride,
                                               int n_live) {
    sf_load_stage(wb[0], sfb_row0);
#pragma unroll
    for (int gq = 1; gq < NG; ++gq) {
        if (n_live > 32 * gq) {
            sf_load_stage(wb[gq], sfb_row0 + gq * 32 * stride);
        }
    }
}
// ... and their store: the whole slot in one tcgen05.st.32x32b.x16. `taddr` is the slot's column base; the 16
// registers land at columns 0..15, group g's word at kk * 4 + g. The MMA computes N = n_live rows, so the columns of
// groups at or above ceil(n_live / 32) are never read; they take the last live group's word.
template <int NG>
__device__ __forceinline__ void sf_store_slot_x16(uint32_t taddr, const uint32_t (&wb)[NG][kSfWords], int n_live) {
    uint32_t r[kSfCols];
#pragma unroll
    for (int kk = 0; kk < kSfWords; ++kk) {
        uint32_t w = wb[0][kk];  // group 0 is live at every n_mma
        r[kk * 4] = w;
#pragma unroll
        for (int g = 1; g < 4; ++g) {
            if (g < NG && n_live > 32 * g) w = wb[g < NG ? g : 0][kk];
            r[kk * 4 + g] = w;
        }
    }
    ptx::tcgen05_st_32x32b_x16(taddr, r);
}

// ---- the MMA issuers' operands, built once and stepped ----
// A K-major smem descriptor's address field is addr >> 4 in bits 0-13 and every other field is a constant of the
// operand (ptx.cuh), so the descriptor of addr + b is the descriptor of addr plus b >> 4 for any 16 B-aligned
// b: no shared address reaches 2^18 B, so the field never carries.
__host__ __device__ constexpr uint64_t desc_step(int bytes) {
    return uint64_t(bytes) >> 4;
}
inline constexpr uint64_t kDescK = desc_step(32);  // one K piece: 32 B into the stage's 128 B swizzled rows
static_assert(kBKBytes == 4 * 32 && kKPerTile == 4, "a stage's 128 B rows carry four 32 B K pieces");
// The cell's instruction descriptor: N is its only runtime term (the SF ids mxf4nvf4 folds in are bits 30-31 of the
// TMEM SF addresses, the top of their lane field: zero, every SF operand sits at lane 0).
__device__ __forceinline__ uint32_t mma_idesc(int n) {
    if constexpr (kBf16)
        return ptx::mma_inst_desc_f16(128, uint32_t(n));
    else
        return ptx::mma_inst_desc_mxf4nvf4_block16(128, uint32_t(n));
}
// a ring index and its phase parity, advanced one stage
__device__ __forceinline__ void ring_next(int& i, uint32_t& parity, int n) {
    if (++i == n) {
        i = 0;
        parity ^= 1u;
    }
}

// the token-tile cell decode: a pure function of the tile index, so every role derives the same answer (ranks may
// order the tiles differently, hot-first; every cross-rank access is keyed by the expert's rows, not the tile index)
__device__ __forceinline__ void tile_cell(const RouteTables& rt, int ti, int& e, int& off, int& m) {
    e = rt.tile_e[ti];
    const int r0 = rt.tile_r0[ti];
    off = rt.offsets[e] + r0;
    const int rem = rt.counts[e] - r0;
    m = rem < kNPad ? rem : kNPad;
}

// the dynamic smem the MoE phase may claim: the launch ceiling minus the static union the front and the z staging
// window share (smem_budget)
inline constexpr int kSmemBudget = smem_budget(kFbW, kNPad);
// kHRow: the staged exponent run rides the budget's 1 KB slack (smem_budget takes the plain union's size)
static_assert(!kHRow || zstage_bytes(kFbW, kNPad, true) - zstage_bytes(kFbW, kNPad, false) <= 1024,
              "kHRow: the staged exponent run must fit the dynamic budget's 1 KB slack");

// g1: the rank's W1 intermediate shard (gate + up), then SwiGLU and NVFP4 requant into z, gathered on every rank
// (the engine).

// =============================== g1 ===============================
// Warps: w0 producer | w1 MMA issuer | w2 SF copy issuer | w3 join / emit | w4-11 epilogue (w4-7 also the B-side SF
// stagers).
namespace g1 {
inline constexpr int STAGES = kH / kBK;
inline constexpr int A_BYTES = kARows * kBKBytes;  // gate + up tiles
inline constexpr int B_BYTES = kNPad * kBKBytes;
inline constexpr int SLOT = A_BYTES + B_BYTES;
// The SF K chunk: how many K stages of SF the staging block holds. The whole-K block is kARows x H / 16 bytes, which
// leaves the ring nothing at a large hidden size; chunking it along K keeps the 128-wide tile buildable there.
// Derived: the largest chunk that still leaves three ring slots (one chunk wherever the whole-K block fits).
inline constexpr int NBUF = 1;  // one block buffer (the B-side block and its stagers' free arity live in smem)
constexpr int sf_chunk_fit(int nbuf) {
    if constexpr (kBf16) {
        return STAGES;  // no SF: one chunk, zero-byte blocks
    } else {
        for (int p = STAGES; p >= 1; --p) {
            if (STAGES % p) continue;
            // the stagers' B-side block plus the A-side block, NBUF-buffered
            const int blk = nbuf * (kARows + kNPad) * (p * kBK / kSfVec);
            if (kSmemBudget - blk >= 3 * SLOT) return p;  // three ring slots kept
        }
        return 1;
    }
}
inline constexpr int SFC = sf_chunk_fit(NBUF);
inline constexpr int SFC1 = sf_chunk_fit(1);  // the single-buffer chunk (the ring depth's derivation)
static_assert(STAGES % SFC == 0, "the SF chunk must divide the cell's K stages");
inline constexpr int NCHUNK = STAGES / SFC;
inline constexpr int SF_ROW = kBf16 ? 0 : SFC * kBK / kSfVec;  // SF bytes per row per chunk (none under kBf16)
// the B-side block's gather4 lands four 16 B SF rows per box at a 128 B-aligned smem address: a one-stage chunk (16
// B rows) faults with a misaligned address, so a chunk holds two stages at least
static_assert((4 * SF_ROW) % 128 == 0, "the chunk's SF rows fill 128 B gather boxes (>= 2 stages)");
inline constexpr int SFU32 = SF_ROW / 4;  // ... as u32, the smem row stride
inline constexpr int SFA_BLOCK = kARows * SF_ROW;
inline constexpr int SFB_BLOCK = kNPad * SF_ROW;
// kZStageDyn: the two z staging buffers past the blocks (the bf16 layers'; 0 bytes otherwise)
inline constexpr int ZST_BYTES = kZStageDyn ? 2 * int(sizeof(ZStageBuf)) : 0;
inline constexpr int NS_FIT = (kSmemBudget - ZST_BYTES - NBUF * (SFA_BLOCK + SFB_BLOCK)) / SLOT;
// the ring depth: the fit, capped at the single-buffer chunk's derivation (NS_SHIP) so the depth does not move with
// the buffers
inline constexpr int NS_SHIP =
    (kSmemBudget - kARows * (SFC1 * kBK / kSfVec) - kNPad * (SFC1 * kBK / kSfVec)) / (A_BYTES + B_BYTES);
inline constexpr int NS_PIN = (!kBf16) ? (NS_FIT < NS_SHIP ? NS_FIT : NS_SHIP) : NS_FIT;
inline constexpr int NS = NS_PIN < STAGES ? NS_PIN : STAGES;
inline constexpr int PRE = NS < SFC ? NS : SFC;  // the hoisted entry: cell 0's stages issued ahead of the rows' word
inline constexpr int RING = NS * SLOT;
inline constexpr int SFA_BLOCK_OFF = RING;
inline constexpr int SFB_BLOCK_OFF = RING + NBUF * SFA_BLOCK;
inline constexpr int ZST_OFF = SFB_BLOCK_OFF + NBUF * SFB_BLOCK;
static_assert(ZST_OFF % 16 == 0, "the z staging window keeps ZStageBuf's alignment");
inline constexpr int SMEM = ZST_OFF + ZST_BYTES;
static_assert(NS >= 3 && SMEM <= kSmemBudget, "g1 over the derived smem budget");
inline constexpr int NHALF = SF_ROW / 128;                         // 128-B half-images per block row
inline constexpr int HALF_A = kARows * 128, HALF_B = kNPad * 128;  // bytes per half-image
// the 128-wide tile carries two SF streams and two accumulators: gate and up
inline constexpr int PAIRS = 2;
inline constexpr int TM_SFA_N = PAIRS * kSfCols;  // gate + up
inline constexpr int TM_SFB_N = kSfCols;
// D double-buffered by cell parity: two TMEM accumulators serve different cells (each cell accumulates its full K in
// one), so the epilogue drain overlaps the next cell's MMAs
inline constexpr int ACC_SLOTS = 2;
inline constexpr int TM_DSTRIDE = PAIRS * kNPad;  // gate + up
inline constexpr int TM_DG = 0, TM_DU = kNPad;
inline constexpr int TM_SFA = ACC_SLOTS * TM_DSTRIDE;
inline constexpr int ALLOC_COLS = 512;
inline constexpr int SF_ARRIVERS = 4;  // warps arriving per SF stage and block hand-back: the four stager warps
// the one predicate for "this warp stages SF rows": the staging gate, the per-stage arrive, the block hand-back and
// the init pre-arrive all key on it, so the barrier counts and the arrive sites cannot disagree (a mismatch
// deadlocks the first chunk). Every MB_SFR / MB_BLK_FREE arrive site is inside code gated by this predicate, so each
// stager arrives exactly once per event.
__host__ __device__ constexpr bool sf_stager(int warp) {
    // the four stagers: the first four drainers (w4-w7, also g2's SF stagers)
    return warp >= kDrain0 && warp < kDrain0 + 4;
}
constexpr int sf_stager_count() {
    int n = 0;
    for (int w = 0; w < kFrontWarps; ++w)
        n += sf_stager(w) ? 1 : 0;
    return n;
}
static_assert(sf_stager_count() == SF_ARRIVERS, "the barrier arity is the number of staging warps");
// The SF copy route: the cell's weight SF lands from a second pool image in the block-scaled SF order (per (K64
// cell, 128-row half) one 512 B tile: byte n * 16 + g * 4 + k16 for tile row 32g + n and K16 sub-block k16), and the
// copy issuer's (w2's) elected lane places each cell's SF with one tcgen05.cp.32x128b.warpx4 per half: source row n
// = the four row groups' words, broadcast to the four quadrants at columns 4kk..4kk+3, the layout the MMAs read.
// SFPF: the blocked W1 SF image's L2 prefetches (never on the bf16 layers: no SF image, its map unencoded)
inline constexpr bool SFPF = !kBf16;
inline constexpr int SFCOPY_TILE = 512;  // bytes per (cell, half) source tile
inline constexpr int SF_CP_SBO = 128;    // the copy source's stride between 8-row core matrices: eight 16-B SF rows
inline constexpr int GATHER4_BYTES = 4 * kBKBytes;  // one gather4 box of the B feed: four token rows of one stage
// who frees the SF block each cell: the B-side stagers (their reads) and the copy issuer's commit (its copies'
// reads); the init count, the pre-arrive set and the arrive / commit sites all key on this one predicate
__host__ __device__ constexpr bool sfb_stager(int warp) {
    return warp >= kDrain0 && warp < kDrain0 + 4;
}
__host__ __device__ constexpr bool blk_free_arriver(int warp) {
    return sfb_stager(warp) || warp == kCopyWarp;  // the A-side stagers read no block under the copy route
}
constexpr int blk_free_arriver_count() {
    int n = 0;
    for (int w = 0; w < kFrontWarps; ++w)
        n += blk_free_arriver(w) ? 1 : 0;
    return n;
}
// the stage-ready barrier: the copy issuer's commit plus the four stagers' arrivals (the stagers stage the token side
// while the issuer copies the weight side)
inline constexpr int SFR_ARRIVERS = 1 + SF_ARRIVERS;
inline constexpr int BLK_FREE_ARRIVERS = blk_free_arriver_count();
static_assert(BLK_FREE_ARRIVERS == (SF_ARRIVERS) + 1, "the block-free arity is the stagers plus the copy issuer");
static_assert(kFbW == 128, "the SF copy route's blocked image is laid out for 128-wide tiles");
// the K-chunked route (NCHUNK > 1): the per-chunk block is the chunk's SFC stages of the blocked image, one TMA box
// of SFA_BLOCK / 2 KB rows x 2 KB; only the depth index moves
static_assert(kBf16 || SFA_BLOCK == SFC * kKPerTile * PAIRS * SFCOPY_TILE,
              "the blocked image carries the row-major block's bytes, per chunk");
static_assert((SFA_BLOCK % 2048 == 0 && SFA_BLOCK / 2048 <= 256),
              "the blocked image's TMA box is the chunk's block as whole 2 KB rows (at most 256)");
// the epilogue warps (the eight drainers arrive on MB_ACC_EMPTY, count 8): the count is tied to the predicate
// the same way
__host__ __device__ constexpr bool drainer(int warp) {
    return warp >= kDrain0 && warp < kDrain0 + 8;
}
constexpr int epi_warp_count() {
    int n = 0;
    for (int w = 0; w < kFrontWarps; ++w)
        n += drainer(w) ? 1 : 0;
    return n;
}
inline constexpr int EPI_ARRIVERS = 8;
static_assert(epi_warp_count() == EPI_ARRIVERS, "the epilogue warps (4-11) arrive once each on MB_ACC_EMPTY");
static_assert(kEmitWarp < kDrain0, "the join/emit warp is its own warp, ahead of the drainers");
inline constexpr int NSF = 2;  // SF TMEM slots
static_assert(NSF >= 2 && NSF <= STAGES, "g1 SF ring: double-buffered at least, a cell at most");
inline constexpr int TM_SFB = TM_SFA + NSF * TM_SFA_N;
static_assert(TM_SFB + NSF * TM_SFB_N <= ALLOC_COLS, "g1 TMEM over its allocation");
static_assert(TM_SFB % kSfCols == 0 && TM_SFB_N == kSfCols, "the g1 B-side SF slots are whole 16-column blocks");
enum {
    MB_FULL = 0,
    MB_EMPTY = MB_FULL + NS,
    MB_SFR = MB_EMPTY + NS,
    MB_SFF = MB_SFR + NSF,
    MB_BLK_FULL = MB_SFF + NSF,
    MB_BLK_FREE = MB_BLK_FULL + NBUF,
    MB_ACC_FULL = MB_BLK_FREE + NBUF,
    MB_ACC_EMPTY = MB_ACC_FULL + ACC_SLOTS,
    MB_COUNT = MB_ACC_EMPTY + ACC_SLOTS
};
// the staged-emission WAR pair past the zjoin pair [MB_COUNT, MB_COUNT + 2): count 1 = the emit warp's elected
// arrive after its last emission store of the cell; waited by the 8 epilogue warps before they re-stage that buffer
// two cells later
enum { MB_ZSE = MB_COUNT + 2 };
// the hoisted entry's relay of the tables' word: one arrival (the drainer's lane, after its acquire), cta-scope
// release / acquire, so the word keeps one gpu-scope poller per CTA
inline constexpr int MB_ROWS = MB_ZSE + 2;
inline constexpr int MB_END = MB_ROWS + 1;
}  // namespace g1

// Place stage sc's A-side SF (global stage index sg): per half one broadcast copy per K64 cell from the blocked
// image into TMEM SF slot sg % NSF. Ordering: (RAW) the MMAs that read the slot are issued later by the same thread,
// and tcgen05 operations of one thread execute in issue order; (WAR) the slot's previous reader (stage sg - NSF) is
// an asynchronous MMA, so the wait below is on MB_SFF[slot], which the MMA issuer's commit after those MMAs
// completes. Issued by the copy issuer's elected lane; its commit then arrives MB_SFR for the MMA issuer.
__device__ __forceinline__ void sf_copy_stage(int sc, int sg, uint32_t taddr, uint32_t smem, uint64_t* mbars) {
    const int sfs = sg % g1::NSF;
    ::ptx::mbar_wait_parity(&mbars[g1::MB_SFF + sfs], uint32_t((sg / g1::NSF) & 1));
    ptx::tcgen05_fence_after_thread_sync();
    if (ptx::elect_one()) {
#pragma unroll
        for (int h = 0; h < g1::PAIRS; ++h)  // gate, then up
#pragma unroll(mk::cfg::kWorld.text_diet ? 1 : kKPerTile)
            for (int kk = 0; kk < kKPerTile; ++kk)
                ptx::tcgen05_cp_32x128b_warpx4(
                    taddr + g1::TM_SFA + sfs * g1::TM_SFA_N + h * kSfCols + kk * 4,
                    ptx::mma_smem_desc(
                        smem + g1::SFA_BLOCK_OFF + ((sc * kKPerTile + kk) * g1::PAIRS + h) * g1::SFCOPY_TILE, 0u,
                        uint32_t(g1::SF_CP_SBO), 0u, 0));
    }
}
// a warp-uniform value the compiler can prove uniform (a full-mask broadcast of lane 0's copy; every lane holds the
// same value already)
__device__ __forceinline__ int uni(int v) {
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}
// g2's barrier set follows g1's at its own indices; it is armed after the phase boundary's join (g2_init)
inline constexpr int kG2MbarBase = g1::MB_END;  // after every g1 word

__device__ __forceinline__ void g1_init(uint64_t* mbars, uint32_t* tmem_addr_storage) {
    const int warp = int(threadIdx.x) >> 5;
    if (warp == kG1Roles.producer && ptx::elect_one()) {
        for (int i = 0; i < g1::NS; ++i) {
            ptx::mbar_init(&mbars[g1::MB_FULL + i], 1);   // expect_tx only
            ptx::mbar_init(&mbars[g1::MB_EMPTY + i], 1);  // commit-armed
        }
        for (int i = 0; i < g1::NSF; ++i) {
            ptx::mbar_init(&mbars[g1::MB_SFR + i],
                           g1::SFR_ARRIVERS);  // the SF-staging warps and/or the copy issuer's commit
            ptx::mbar_init(&mbars[g1::MB_SFF + i], 1);
        }
#pragma unroll
        for (int i = 0; i < g1::NBUF; ++i) {
            ptx::mbar_init(&mbars[g1::MB_BLK_FULL + i], 1);
            ptx::mbar_init(&mbars[g1::MB_BLK_FREE + i], g1::BLK_FREE_ARRIVERS);
        }
#pragma unroll
        for (int i = 0; i < g1::ACC_SLOTS; ++i) {
            ptx::mbar_init(&mbars[g1::MB_ACC_FULL + i], 2);                  // commit + belt
            ptx::mbar_init(&mbars[g1::MB_ACC_EMPTY + i], g1::EPI_ARRIVERS);  // the epilogue warps 2-9
        }
        // MB_ROWS: the hoisted entry's relay word (initialized here)
        ptx::mbar_init(&mbars[g1::MB_ROWS], 1);
        ptx::fence_mbarrier_init_release_cluster();
    } else if (warp == kG1Roles.mma) {
        ptx::tcgen05_alloc(ptx::to_shared(tmem_addr_storage), g1::ALLOC_COLS);
    }
    __syncthreads();
    if (warp == kG1Roles.producer && ptx::elect_one()) {
        for (int i = 0; i < g1::NS; ++i)
            ptx::mbar_arrive(&mbars[g1::MB_EMPTY + i]);
        for (int i = 0; i < g1::NSF; ++i)
            ptx::mbar_arrive(&mbars[g1::MB_SFF + i]);
    }
    // one pre-arrive set, keyed on the init's predicate: every block-free arriver (the B stagers, the copy
    // issuer) starts free in every block buffer, so the arity pre-armed equals the arity initialised (an extra
    // pre-arrive completes a phase early and lets the producer reload the block under its readers)
    if (g1::blk_free_arriver(warp) && ptx::elect_one())
#pragma unroll
        for (int i = 0; i < g1::NBUF; ++i)
            ptx::mbar_arrive(&mbars[g1::MB_BLK_FREE + i]);
    if (warp >= kDrain0 && ptx::elect_one())
#pragma unroll
        for (int i = 0; i < g1::ACC_SLOTS; ++i)
            ptx::mbar_arrive(&mbars[g1::MB_ACC_EMPTY + i]);
    __syncthreads();
}

// kHRow: z_sf and z_ex are stage-major, [g2::STAGES = kI / kBK][kMaxSlots][16 B]: a row's SF (or exponent) byte p
// lives at stage p / 16, row, offset p % 16, so g2's B-side box for a stage is 32 rows x 16 B contiguous (a 2D TMA
// box of 512 B x STAGES rows) landing [stage][row][16 B], the pieces the recalibrators rewrite in place and the stagers
// load coalesced. A cell's run never straddles a stage. Off, row-major [kMaxSlots][kI / 16].
__device__ __forceinline__ size_t zsf_off(int slot, int p) {
    if constexpr (kHRow)
        return size_t(p >> 4) * (size_t(kMaxSlots) * 16) + size_t(slot) * 16 + size_t(p & 15);
    else
        return size_t(slot) * (kI / kSfVec) + size_t(p);
}
// g1's two z staging buffers: the front overlay's recycle window, or under kZStageDyn (the bf16 layers)
// the dynamic carve past g1's ring (g1::ZST_OFF)
template <class Ovl>  // FrontZsOvl; a template, so the discarded branch is never instantiated
__device__ __forceinline__ ZStageBuf* zstage_window(Ovl& o) {
    if constexpr (kZStageDyn) {
        extern __shared__ __align__(1024) char smem_buf[];
        return reinterpret_cast<ZStageBuf*>(smem_buf + g1::ZST_OFF);
    } else {
        return o.zstage;
    }
}

// the cell's join / emit duty (w3): wait the eight drainers' join, emit the staged cell to every rank as lamport
// elements (every 8 B a st.relaxed.sys store, z at MoE pass e's parity, no release), release the staging
// buffer. dry: the body's instructions without its effects (no wait, no store, no arrive).
// A code group whose low word reads as the sentinel's (eight signed zeros of alternating sign) goes out with its first
// code's +0 as -0: the same values.
__device__ __forceinline__ uint64_t lam_group(uint64_t g) {
    if constexpr (kBf16) return g;  // a bf16 pair is never all ones
    return uint32_t(g) == kZSent32 ? g | 0x8ull : g;
}
__device__ __forceinline__ void g1_emit_cell(const LayerArgs& a, uint64_t* mbars, ZStageBuf* zst, int ci, int off,
                                             int m, int zb0, int sb0, uint32_t mep, int lane, bool dry) {
    if (!dry) ::ptx::mbar_wait_parity(&mbars[g1::MB_COUNT + (ci & 1)], uint32_t((ci >> 1) & 1));
    // the join gives this warp a CTA-scope acquire over the other epilogue warps' staging writes
    const ZStageBuf& st = zst[ci & 1];
    const int zr = zpar_row0(mep) + off;
    // a cell's z run is kZL 16 B lines a column (4 fp4, 16 bf16), so the warp emits 32 / kZL columns a round
    constexpr int kZL = elem_bytes(kFbW) / 16;
    const int zl = lane & (kZL - 1), zc = int(uint32_t(lane) / kZL);
    for (int t0 = 0; t0 < m; t0 += 32 / kZL) {
        const int col = t0 + zc;
        if (col < m) {
            const uint64_t* const v = reinterpret_cast<const uint64_t*>(st.z + col * elem_bytes(kFbW) + zl * 16);
            const uint64_t g0 = lam_group(v[0]), g1v = lam_group(v[1]);
            const size_t zo = size_t(zr + col) * elem_bytes(kI) + zb0 + zl * 16;
#pragma unroll(mk::cfg::kWorld.text_diet ? 4 : kRanks)
            for (int r = 0; r < (dry ? 0 : kRanks); ++r)
                ptx::st_relaxed_sys_v2b64(a.z_peer[r] + zo, g0, g1v);
        }
    }
    // kHRow: the cell's SF run and its exponent run, one 8 B element per column each (stage-major, zsf_off)
    for (int t0 = 0; t0 < (kHRow ? m : 0); t0 += 32) {
        const int col = t0 + lane;
        if (col < m) {
            const size_t so = zsf_off(zr + col, sb0);
            const uint64_t sf = *reinterpret_cast<const uint64_t*>(zst_sf(st) + col * (kFbW / kSfVec));
            const uint64_t ex = *reinterpret_cast<const uint64_t*>(zst_ex(st) + col * (kFbW / kSfVec));
#pragma unroll(mk::cfg::kWorld.text_diet ? 4 : kRanks)
            for (int r = 0; r < (dry ? 0 : kRanks); ++r) {
                ptx::st_relaxed_sys_b64(a.zsf_peer[r] + so, sf);
                ptx::st_relaxed_sys_b64(a.zex_peer[r] + so, ex);
            }
        }
    }
    // the staging buffer's WAR: every lane's reads ahead of the elected arrive
    __syncwarp();
    if (!dry && ptx::elect_one()) ptx::mbar_arrive(&mbars[g1::MB_ZSE + (ci & 1)]);
}

// the anchor's arrival on the emit warp's lane 0
__device__ __forceinline__ void g1_anchor(const LayerArgs& a, uint32_t e, int lane) {
    if (lane == 0) lam_anchor_arrive(a, e, a.gsync + 2);
    __syncwarp();
}

static_assert(g1::NCHUNK * g1::SFC == g1::STAGES, "the SF chunks tile the cell's K stages");

// the g2 producer's first cell, claimed and decoded by the g1 producer warp behind its last issue (and gated there
// on a CTA that ran two g1 cells), off the g1 -> g2 phase boundary
__device__ __forceinline__ void g2_preclaim(const LayerArgs& a, uint64_t* mbars, int32_t* sclaim, uint32_t e, int lane);

// The g1 loop: fb0 = the rank's first global feature block (z offsets); the weight map's depth indexes the shard
// pack (e x kFbPr + local block). A cell's z lands in every rank's copy as its emit warp stores it (the register
// fan-out); the consumers poll the elements themselves, so a cell has no close.
__device__ __forceinline__ void g1_loop(const LayerArgs& a, uint64_t* mbars, const uint32_t taddr,
                                        const RouteTables& rt, int fb0, uint32_t mep, ZStageBuf* zst, int32_t* sclaim,
                                        const int32_t* cell0) {
    extern __shared__ __align__(1024) char smem_buf[];
    const uint32_t smem = ptx::to_shared(smem_buf);
    const uint32_t* smem_u32 = reinterpret_cast<const uint32_t*>(smem_buf);
    const int tid = threadIdx.x, warp = uni(tid >> 5), lane = tid & 31;
    const int n_cells = uni(kFbPr * cell0[3]);
    // the resident-rows variant is off; its live-group scan (if (res) below) is kept for the schedule (README.md)
    const bool res = false;

    if (warp == kG1Roles.producer) {
        // ================= producer =================
        int s_g = 0;
        for (int ci = 0, c = blockIdx.x; c < n_cells; c += gridDim.x, ++ci)  // cells round-robin over the grid
        {
            const int fbl = c % kFbPr;  // shard-local block
            int e, off, m;
            // the hoisted entry: cell 0's tile was read by the block ahead of the rows' word (moe_body)
            if (ci == 0) {
                e = cell0[0];
                off = cell0[1];
                m = cell0[2];
            } else {
                tile_cell(rt, c / kFbPr, e, off, m);
            }
            e = uni(e);
            off = uni(off);
            m = uni(m);
            const int n_mma = min(kNPad, ceil8(m));
            // the hoisted entry: cell 0's row indices from the drainer's shared copy, every other cell's
            // from the tables (one pointer per cell, outside the chunk loop)
            const int32_t* const tok8 = ci == 0 ? cell0 + 4 : rt.slot_tok + off;
            for (int ch = 0; ch < g1::NCHUNK; ++ch) {
                const int cg = ci * g1::NCHUNK + ch;  // the running chunk (every role agrees)
                const int buf = cg % g1::NBUF;        // its block buffer
                const uint32_t bph = uint32_t((cg / g1::NBUF) & 1);
                // kBf16: no scale factors — no chunk block to free, issue or wait on
                if constexpr (!kBf16) {
                    ::ptx::mbar_wait_parity(&mbars[g1::MB_BLK_FREE + buf], bph);
                    uint32_t blk_tx = 0;
                    if (ptx::elect_one()) {
                        constexpr int kHf = 1;
                        // the blocked image: the chunk's block as one box (SFA_BLOCK / 2 KB
                        // rows x 2 KB) from the second pool image, depth = the tile's NCHUNK
                        // blocks in K order
                        const int sf_tile = e * kFbFull + fbl;
                        wtma3d(smem + g1::SFA_BLOCK_OFF + buf * g1::SFA_BLOCK, &a.tm_sfab, 0, 0,
                               sf_tile * g1::NCHUNK + ch, &mbars[g1::MB_BLK_FULL + buf]);
                        blk_tx += g1::SFA_BLOCK;
                        // a quad's 4 rows land 4 consecutive image rows: 4*SF_ROW
                        // apart row-major, 4*128 B in a half-image
                        constexpr int kQuadB = 4 * g1::SF_ROW;
                        // the token side's block: one gather4 per quad of rows
#pragma unroll 2
                        for (int gq = 0; gq < (n_mma + 3) / 4; ++gq) {
                            int32_t r[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int j = gq * 4 + i;
                                r[i] = (j < m) ? tok8[j] : int32_t(kOobRow);
                            }
#pragma unroll
                            for (int hf = 0; hf < kHf; ++hf)
                                ptx::cp_async_bulk_tensor_2d_gather4(
                                    smem + g1::SFB_BLOCK_OFF + buf * g1::SFB_BLOCK + hf * g1::HALF_B + gq * kQuadB,
                                    &a.tmaps.tm[TM_G1_SFB], ch * g1::SFU32 + hf * 32, r[0], r[1], r[2], r[3],
                                    &mbars[g1::MB_BLK_FULL + buf]);
                            blk_tx += 4 * g1::SF_ROW;
                        }
                        ptx::mbar_arrive_expect_tx(&mbars[g1::MB_BLK_FULL + buf], blk_tx);
                    }
                }  // !kBf16

                for (int u = 0, s = ch * g1::SFC; u < g1::SFC; ++u, ++s, ++s_g) {
                    const int slot = s_g % g1::NS;
                    const uint32_t gen = uint32_t(s_g / g1::NS);
                    ::ptx::mbar_wait_parity(&mbars[g1::MB_EMPTY + slot], gen & 1);
                    const uint32_t a_dst = smem + slot * g1::SLOT;
                    const uint32_t b_dst = a_dst + g1::A_BYTES;
                    if (ptx::elect_one()) {
                        // the weight tile, then the gathered token rows
                        wtma3d(a_dst, &a.tmaps.tm[TM_G1_A], 0, s * kARows, e * kFbFull + fbl,
                               &mbars[g1::MB_FULL + slot]);
                        uint32_t b_tx = 0;
#pragma unroll 2
                        for (int gq = 0; gq < (n_mma + 3) / 4; ++gq) {
                            int32_t r[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int j = gq * 4 + i;
                                r[i] = (j < m) ? tok8[j] : int32_t(kOobRow);
                            }
                            ptx::cp_async_bulk_tensor_2d_gather4(b_dst + gq * g1::GATHER4_BYTES, &a.tmaps.tm[TM_G1_B],
                                                                 s * kBK, r[0], r[1], r[2], r[3],
                                                                 &mbars[g1::MB_FULL + slot]);
                            b_tx += g1::GATHER4_BYTES;
                        }
                        ptx::mbar_arrive_expect_tx(&mbars[g1::MB_FULL + slot], g1::A_BYTES + b_tx);
                    }
                }
            }
        }
        g2_preclaim(a, mbars, sclaim, mep, lane);
    } else if (warp == kG1Roles.mma) {
        // ================= MMA issuer =================
        // the SF copies are w2's; this thread waits their commit (and the stagers' arrivals) on MB_SFR.
        // The operands are built once and stepped (desc_step): ring slot 0's descriptors, the
        // instruction descriptor per cell, the ring and SF slots with their parities as counters.
        const uint64_t da0 = ptx::mma_smem_desc_k_major<uint8_t, kBKBytes, 128>(smem);
        const uint64_t db0 = ptx::mma_smem_desc_k_major<uint8_t, kBKBytes, 128>(smem + g1::A_BYTES);
        constexpr uint64_t kDSlot = desc_step(g1::SLOT), kDUp = desc_step((kARows / 2) * kBKBytes);
        int slot = 0, sfs = 0;
        uint32_t par = 0, sfp = 0;
        for (int ci = 0, c = blockIdx.x; c < n_cells; c += gridDim.x, ++ci) {
            int e, off, m;
            tile_cell(rt, c / kFbPr, e, off, m);
            e = uni(e);
            off = uni(off);
            m = uni(m);
            const int n_mma = min(kNPad, ceil8(m));
            const int as = ci & 1;
            const uint32_t acc_ph = uint32_t((ci >> 1) & 1);
            const uint32_t id = mma_idesc(n_mma);
            const uint32_t d_g = taddr + g1::TM_DG + as * g1::TM_DSTRIDE, d_u = taddr + g1::TM_DU + as * g1::TM_DSTRIDE;
            ::ptx::mbar_wait_parity(&mbars[g1::MB_ACC_EMPTY + as], acc_ph);
            for (int s = 0; s < g1::STAGES; ++s) {
                ::ptx::mbar_wait_parity(&mbars[g1::MB_FULL + slot], par);
                if constexpr (!kBf16) ::ptx::mbar_wait_parity(&mbars[g1::MB_SFR + sfs], sfp);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one()) {
                    const uint64_t da = da0 + uint64_t(slot) * kDSlot, db = db0 + uint64_t(slot) * kDSlot;
#pragma unroll(mk::cfg::kWorld.text_diet ? 1 : kKPerTile)
                    for (int kb = 0; kb < kKPerTile; ++kb) {
                        // the "up" operands address the tile's second half: a second M = 128
                        // MMA at kFbW 128 (at 64 the interleaved tile carries both in one)
                        const uint64_t da_g = da + uint64_t(kb) * kDescK, da_u = da_g + kDUp;
                        const uint64_t dbk = db + uint64_t(kb) * kDescK;
                        const uint32_t scale_c = (s == 0 && kb == 0) ? 0u : 1u;
                        if constexpr (kBf16) {
                            // kBf16: bf16 x bf16 -> fp32, K16 a piece (the stage's
                            // 128 B = 64 elements = 4 pieces, 32 B apart like the fp4
                            // K64 pieces); no scale-factor operands
                            ptx::tcgen05_mma_f16(d_g, da_g, dbk, id, scale_c);
                            ptx::tcgen05_mma_f16(d_u, da_u, dbk, id, scale_c);
                        } else {
                            // the SF operand bases stay 4-column aligned: a stride-1 base traps tcgen05.mma
                            const uint32_t sfa_g = taddr + g1::TM_SFA + sfs * g1::TM_SFA_N + kb * 4;
                            const uint32_t sfa_u = sfa_g + kSfCols;
                            const uint32_t sfb_k = taddr + g1::TM_SFB + sfs * g1::TM_SFB_N + kb * 4;
                            ptx::tcgen05_mma_mxf4nvf4_block16(d_g, da_g, dbk, id, scale_c, sfa_g, sfb_k);
                            ptx::tcgen05_mma_mxf4nvf4_block16(d_u, da_u, dbk, id, scale_c, sfa_u, sfb_k);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&mbars[g1::MB_EMPTY + slot]);
                    if constexpr (!kBf16) ptx::tcgen05_commit_arrive(&mbars[g1::MB_SFF + sfs]);
                }
                ring_next(slot, par, g1::NS);
                if constexpr (!kBf16) ring_next(sfs, sfp, g1::NSF);
            }
            if (ptx::elect_one()) {
                ptx::tcgen05_commit_arrive(&mbars[g1::MB_ACC_FULL + as]);
                ptx::mbar_arrive_release(&mbars[g1::MB_ACC_FULL + as]);  // count-2 belt
            }
        }
    } else if (warp == kCopyWarp) {
        // ================= SF copy issuer (w2) =================
        // Per stage: the TMEM SF slot's previous reader is done (MB_SFF, waited inside sf_copy_stage) and
        // the cell's blocked A image has landed (MB_BLK_FULL, once per chunk); then the A copies and one
        // commit -> MB_SFR[slot], which the MMA issuer waits on. The block-free commit after the chunk's
        // last copies stays with the issuer of those copies.
        int s_g = 0;
        // kBf16: no scale factors to copy (the warp has no g1 role)
        if constexpr (!kBf16)
            for (int ci = 0, c = blockIdx.x; c < n_cells; c += gridDim.x, ++ci) {
                static_assert(g1::blk_free_arriver(kCopyWarp), "the copy issuer is a block-free arriver");
                for (int s = 0; s < g1::STAGES; ++s, ++s_g) {
                    const int slot = s_g % g1::NS;
                    const uint32_t gen = uint32_t(s_g / g1::NS);
                    // the chunk's block buffer and generation (the producer's cg = ci x
                    // NCHUNK + ch)
                    const int cg = ci * g1::NCHUNK + s / g1::SFC;
                    const int buf = cg % g1::NBUF;
                    if (s % g1::SFC == 0)
                        ::ptx::mbar_wait_parity(&mbars[g1::MB_BLK_FULL + buf], uint32_t((cg / g1::NBUF) & 1));
                    // the A copies read the cell's block, not the ring slot: issued without a
                    // slot wait
                    sf_copy_stage(g1::NCHUNK == 1 ? s : s % g1::SFC, s_g, taddr, smem + buf * g1::SFA_BLOCK, mbars);
                    if (ptx::elect_one()) ptx::tcgen05_commit_arrive(&mbars[g1::MB_SFR + s_g % g1::NSF]);
                    // release the chunk's buffer once its last stage's A copies are issued: one commit per block
                    // generation, the arity the producer consumes per chunk
                    if ((s + 1) % g1::SFC == 0 && ptx::elect_one())
                        ptx::tcgen05_commit_arrive(&mbars[g1::MB_BLK_FREE + buf]);
                }
            }
    } else if (warp == kEmitWarp) {
        // ================= join / emit (w3) =================
        // the dry pass (the hoisted loop body): cell 0's join / store body once ahead of it, without its effects. The
        // anchor's arrival rides this warp, idle until its first join: the MoE phase entry's CTA barrier
        // ordered the CTA's resets and its post-attention phase ahead of it.
        for (int ci = -1, c = int(blockIdx.x) - int(gridDim.x); c < n_cells; c += gridDim.x, ++ci) {
            const bool dry = ci < 0;
            if (dry && int(blockIdx.x) >= n_cells) continue;
            if (ci == 0) g1_anchor(a, mep, lane);
            const int cc = dry ? int(blockIdx.x) : c;
            const int fbl = cc % kFbPr;
            const int fbg = fb0 + fbl;
            const int zb0 = fbg * elem_bytes(kFbW);
            const int sb0 = fbg * (kFbW / kSfVec);
            int e, off, m;
            tile_cell(rt, cc / kFbPr, e, off, m);
            e = uni(e);
            off = uni(off);
            m = uni(m);
            g1_emit_cell(a, mbars, zst, ci, off, m, zb0, sb0, mep, lane, dry);
        }
        if (int(blockIdx.x) >= n_cells) g1_anchor(a, mep, lane);  // a CTA without a cell still arrives
    } else {
        // ================= B-side SF stagers (w4-7) + epilogue (w4-11) =================
        // The TMEM lane band is the hardware warp quadrant.
        const int band = warp & 3, half = (warp - kDrain0) >> 2;
        int s_g = 0;
        // the dry pass: ci = -1 runs cell 0's body once more ahead of it, with no waits and no arrives
        const bool prefetch_warp = half == 1;  // the non-stager half only
        for (int ci = prefetch_warp ? -1 : 0, c = int(blockIdx.x) - (prefetch_warp ? int(gridDim.x) : 0); c < n_cells;
             c += gridDim.x, ++ci) {
            const bool dry = ci < 0;
            if (dry && int(blockIdx.x) >= n_cells) continue;  // a CTA without a cell has nothing to prefetch
            const int cc = dry ? int(blockIdx.x) : c;
            const int fbl = cc % kFbPr;
            const int fbg = fb0 + fbl;  // the global block (z offsets)
            const int zb0 = fbg * elem_bytes(kFbW);
            const int sb0 = fbg * (kFbW / kSfVec);
            int e, off, m;
            tile_cell(rt, cc / kFbPr, e, off, m);
            e = uni(e);
            off = uni(off);
            m = uni(m);
            const int n_mma = min(kNPad, ceil8(m));
            // the stagers stage the token side's block (the bf16 layers have no SF: the drainers stage
            // nothing)
            if constexpr (!kBf16) {
                if (!dry && g1::sfb_stager(warp)) {  // w4-w7 (never on the dry pass: it moves TMEM and waits)
                    // the cell's own block: the chunk handshake at the cell boundary
                    int n_mma_s = n_mma;  // tile height of the cell being staged
                    // the chunk's block buffer (one buffer)
                    const int cg0 = ci * g1::NCHUNK;
                    ::ptx::mbar_wait_parity(&mbars[g1::MB_BLK_FULL + cg0 % g1::NBUF], uint32_t((cg0 / g1::NBUF) & 1));
                    const uint32_t* sfb_blk = smem_u32 + (g1::SFB_BLOCK_OFF + (cg0 % g1::NBUF) * g1::SFB_BLOCK) / 4;
                    for (int s = 0; s < g1::STAGES; ++s, ++s_g) {
                        const int sfs = s_g % g1::NSF;
                        // per warp
                        ::ptx::mbar_wait_parity(&mbars[g1::MB_SFF + sfs], uint32_t((s_g / g1::NSF) & 1));
                        const uint32_t lane_band = uint32_t(band * 32) << 16;
                        // the whole stage into registers first: the u32 row stride is a
                        // multiple of 32, so the raw read is a 32-way bank conflict; off the
                        // store's dependency chain it hides
                        uint32_t wb[kNGroups][kSfWords];
                        const int sc = g1::NCHUNK == 1 ? s : s % g1::SFC;  // stage within the chunk
                        const int kc0 = kSfWords * sc;
                        // every band stages every live B row group at its own lanes (the
                        // per-quadrant SFB image); a group only at cells tall enough to have
                        // it
                        const bool two = kNGroups > 1 && n_mma_s > 32;
                        if constexpr (kNGroups <= 2) {
                            sf_load_stage(wb[0], sfb_blk + (lane)*g1::SFU32 + kc0);
                            if (kNGroups > 1 && two)
                                sf_load_stage(wb[kNGroups - 1], sfb_blk + (32 + lane) * g1::SFU32 + kc0);
                        } else {
                            sf_load_groups(wb, sfb_blk + lane * g1::SFU32 + kc0, g1::SFU32, n_mma_s);
                        }
                        sf_store_slot_x16((taddr + g1::TM_SFB + sfs * g1::TM_SFB_N) | lane_band, wb, n_mma_s);
                        ptx::tcgen05_wait_st();
                        ptx::tcgen05_fence_after_thread_sync();
                        if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g1::MB_SFR + sfs]);
                    }
                    ptx::fence_async_smem();
                    __syncwarp();  // order ALL lanes' reads before the elected arrive
                    if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g1::MB_BLK_FREE + (cg0 + g1::NCHUNK - 1) % g1::NBUF]);
                }
            }  // !kBf16
            // ---- epilogue: drain gate + up, SwiGLU with the per-expert and per-token scales,
            // amax / 16, e4m3 SF, e2m1 pack ----
            // Columns (= tokens) interleave across the 8 epilogue warps at the 4-column load granule, so
            // all eight are busy at any n_mma. kBf16: alpha 1, no per-token scale.
            const float g1a = kBf16 ? 1.0f : __ldg(a.g1_alphas + e);
            const float g1c = kBf16 ? 1.0f : __ldg(a.g1_scale_c + e);
            // the per-column scale is derived once per cell, one column per lane, ahead of the
            // accumulator wait: the token's fp32 per-token scale (x_scale) times the per-expert alpha
            constexpr int kStgX = kNPad / 32;
            static_assert(kNPad % 32 == 0, "STAGEX puts one column per lane");
            float stgx[kStgX];
            int pos_reg = -1;
#pragma unroll
            for (int u = 0; u < kStgX; ++u) {
                const int col = u * 32 + lane;
                const int tok = col < m ? rt.slot_tok[off + col] : 0;
                stgx[u] = kBf16 ? 1.0f : __fmul_rn(g1a, a.x_scale[tok < 0 ? 0 : tok]);
            }
            // the live column groups: unread while res is off (the scan is kept, above)
            [[maybe_unused]] int g_lo = -1, g_hi = -1;
            if (res) {
                for (int j = 0; j < m; ++j)
                    if (rt.slot_tok[off + j] == lane) pos_reg = j;
                const uint32_t live = __ballot_sync(0xFFFFFFFFu, pos_reg >= 0);
                int k = 0;
                for (int g = 0; g < kNPad / 4; ++g)
                    if ((live >> (4 * g)) & 0xFu) {
                        if ((k & 1) == half) (k < 2 ? g_lo : g_hi) = g;
                        ++k;
                    }
            }
            const int as = ci & 1;
            const uint32_t acc_ph = uint32_t((ci >> 1) & 1);
            if (!dry) ::ptx::mbar_wait_parity(&mbars[g1::MB_ACC_FULL + as], acc_ph);
            // WAR: staging buffer ci & 1 may be re-staged only once the emit warp emitted its previous
            // use, two cells back (count 1, pre-armed)
            if (!dry) ::ptx::mbar_wait_parity(&mbars[g1::MB_ZSE + (ci & 1)], acc_ph);
            ptx::tcgen05_fence_after_thread_sync();
            const uint32_t lane_band = uint32_t(band * 32) << 16;
            constexpr int G = 4;  // columns per pass
            const int lz = lane;  // the lane's feature within the band
            if constexpr (kBf16) {
                // kBf16: the lane holds feature band x 32 + lane of the pass's G columns in each
                // accumulator; h = bf16(silu(bf16(gate)) x bf16(up)) (the bf16 gate_up GEMM's
                // rounding), staged token-major as bf16 (no alpha, no amax / SF / pack); the emit
                // warp stores every rank's copy after the join
                static_assert(G == 4, "kBf16: the 128-wide pass");
                ZStageBuf& st = zst[ci & 1];
                for (int c0 = half * G; c0 < n_mma; c0 += 2 * G) {
                    uint32_t gr[G], ur[G];
                    ptx::tcgen05_ld_32x32b_x4((taddr + g1::TM_DG + as * g1::TM_DSTRIDE + c0) | lane_band, gr[0], gr[1],
                                              gr[2], gr[3]);
                    ptx::tcgen05_ld_32x32b_x4((taddr + g1::TM_DU + as * g1::TM_DSTRIDE + c0) | lane_band, ur[0], ur[1],
                                              ur[2], ur[3]);
                    ptx::tcgen05_wait_ld();
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        const int col = c0 + j;
                        const float gate_b = __bfloat162float(__float2bfloat16_rn(__int_as_float(gr[j])));
                        const float up_b = __bfloat162float(__float2bfloat16_rn(__int_as_float(ur[j])));
                        const __nv_bfloat16 h = __float2bfloat16_rn(__fmul_rn(silu_fast(gate_b), up_b));
                        if (col < m)
                            *reinterpret_cast<__nv_bfloat16*>(st.z + col * (kFbW * 2) + (band * 32 + lz) * 2) = h;
                    }
                }
            } else
                for (int c0 = half * G; c0 < n_mma; c0 += 2 * G) {
                    static_assert(G == 4, "the resident variant's column groups are the pass width");
                    int cg = c0;
                    uint32_t gr[G], ur[G];
                    // the per-column scale (g1_alphas[e] x x_scale[token]), staged above and
                    // broadcast here
                    float g1d[G];
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        const int cb = cg + j;  // the pass's column
                        float sv = stgx[0];
#pragma unroll
                        for (int u = 1; u < kStgX; ++u)
                            if (cb >= u * 32) sv = stgx[u];
                        // the register pick above is warp-uniform (cb is the same on every lane)
                        const int sl = cb & 31;
                        g1d[j] = __shfl_sync(0xFFFFFFFFu, sv, sl);
                    }
                    ptx::tcgen05_ld_32x32b_x4((taddr + g1::TM_DG + as * g1::TM_DSTRIDE + cg) | lane_band, gr[0], gr[1],
                                              gr[2], gr[3]);
                    ptx::tcgen05_ld_32x32b_x4((taddr + g1::TM_DU + as * g1::TM_DSTRIDE + cg) | lane_band, ur[0], ur[1],
                                              ur[2], ur[3]);
                    ptx::tcgen05_wait_ld();  // gate and up are independent loads
                    float batch_z[G], batch_mx[G];
#pragma unroll
                    for (int j = 0; j < G; ++j) {
                        const int col = cg + j;
                        const int pos = col;
                        // the reference's truncation points: both SwiGLU halves round to
                        // bf16 after the per-expert alpha; the post-SwiGLU constants (the
                        // up / gate scale mismatch, the w2 input scale) ride g1c
                        float z, mx;
                        const float a1 = g1d[j];
                        const float gate_b =
                            __bfloat162float(__float2bfloat16_rn(__fmul_rn(__int_as_float(gr[j]), a1)));
                        const float up_b = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__int_as_float(ur[j]), a1)));
                        z = __fmul_rn(__fmul_rn(silu_fast(gate_b), up_b), g1c);
                        // amax over the 16-element SF granule: HALF a hardware
                        // band, so a 16-lane butterfly.
                        mx = fabsf(z);
#pragma unroll
                        for (int d = 8; d > 0; d >>= 1)
                            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, d));
                        // kHRow: the group's code at its provisional exponent
                        // (prov_exp: the SF byte in (224, 448]), the exponent
                        // staged beside the byte for the consumers' per-row recalibration;
                        // off, the absolute code: the group's e4m3 SF is its own amax / 6 in
                        // h units. Either way every producer's slice of a row derives its
                        // scale identically and g2's per-row alpha needs nothing from the
                        // producer.
                        int eb = 0;
                        float zs = z, mxs = mx;
                        if constexpr (kHRow) {
                            eb = hrow::prov_exp(mx);
                            const float sc = hrow::pow2i(-eb);
                            zs = __fmul_rn(z, sc);
                            mxs = __fmul_rn(mx, sc);
                        }
                        const uint8_t sf = amax_to_ue4m3(mxs);
                        const float q = zs * (inv_sf_from_ue4m3(sf));
                        const float qe = __shfl_sync(0xFFFFFFFFu, q, lane & ~1);
                        const float qo = __shfl_sync(0xFFFFFFFFu, q, lane | 1);
                        const uint32_t byte = ptx::cvt_pack_f32x2_to<ptx::e2m1>(qo, qe);
                        uint32_t w32 = byte;
                        w32 |= __shfl_sync(0xFFFFFFFFu, byte, (lane & 24) + 2) << 8;
                        w32 |= __shfl_sync(0xFFFFFFFFu, byte, (lane & 24) + 4) << 16;
                        w32 |= __shfl_sync(0xFFFFFFFFu, byte, (lane & 24) + 6) << 24;
                        if (col < m) {
                            // feature f = the cell's base + band x (kFbW / 4) + lz, so
                            // the z byte is f / 2 and the SF group f / 16
                            const bool put_z = (lane & 7) == 0;
                            const bool put_sf = (lane & 15) == 0;
                            // stage, not store: a token-major mirror of the two global
                            // extents; the emit warp stores every rank's copy as wide
                            // stores after the cell join, between its join wait and the
                            // release, so writer == releaser and the cross-rank ordering
                            // is program order
                            ZStageBuf& st = zst[ci & 1];
                            if (put_z)
                                *reinterpret_cast<uint32_t*>(st.z + pos * (kFbW / 2) + band * (kFbW / 8) +
                                                             (lz >> 3) * 4) = w32;
                            if (put_sf) zst_sf(st)[pos * (kFbW / kSfVec) + band * (kFbW / 64) + (lz >> 4)] = sf;
                            if constexpr (kHRow)
                                if (put_sf)
                                    zst_ex(st)[pos * (kFbW / kSfVec) + band * (kFbW / 64) + (lz >> 4)] = uint8_t(eb);
                        }
                    }
                }
            {
                // the 8 epilogue warps join on the cell and the emit warp counts the cell into its
                // tile; one async-proxy bridge per cell, on that warp only (the join gives it a
                // CTA-scope acquire over the other warps' stores, so its proxy fence is cumulative
                // over them)
                __syncwarp();
                if (!dry && ptx::elect_one()) ptx::mbar_arrive(&mbars[g1::MB_COUNT + (ci & 1)]);
            }
            if (!dry && ptx::elect_one()) ptx::mbar_arrive(&mbars[g1::MB_ACC_EMPTY + as]);
        }
    }
    __syncthreads();
}

// g2: the rank's W2 output shard over the FULL intermediate, the MoE pass's one bf16 rounding, into zb
// (the engine).

// =============================== g2 ===============================
// The full-K second GEMM: A = 128 output columns of W2 over the whole intermediate, B = the tile's gathered z rows.
// Every value it produces is final.
namespace g2 {
inline constexpr int STAGES = kI / kBK;
// The ring words' arrival counts, each named once and used at both barrier inits and the pre-arm: BLK_FREE = the
// four stager warps, plus w2 and w3 after their last rewrite of the block under the row window; RECAL = the two
// recalibrators (w2 + w3); RECAL_FREE = the four stager warps.
inline constexpr int BLK_FREE_ARRIVALS = kHRow ? 6 : 4;
inline constexpr int RECAL_ARRIVALS = 2;
inline constexpr int RECAL_FREE_ARRIVALS = 4;
inline constexpr int HB = kG2Hb;                     // h blocks per cell
inline constexpr int CELLS_PT = kG2CellsPt;          // cells per token tile
inline constexpr int NBLK = 2;                       // per-cell block buffers (the pair's blocks take the second)
inline constexpr int A_BYTES = HB * 128 * kBKBytes;  // the cell's A tiles, h block j at j * 128 rows
inline constexpr int SLOT = A_BYTES + kNPad * kBKBytes;
inline constexpr int SFA_HB = kBf16 ? 0 : 128 * (kI / kSfVec);  // one h block's W2-SF rows (none under kBf16)
inline constexpr int SFA_BLOCK = HB * SFA_HB;                   // the cell's, h block j at j * SFA_HB
inline constexpr int SFB_BLOCK = kBf16 ? 0 : kNPad * (kI / kSfVec);
// kHRow: the exponent plane's block beside the B-side SF block (the same rows and words, one int8 per group), on
// the same box loads and barrier; its bytes ride every B-side SF expect_tx (BSF_BOX_TX)
inline constexpr int EXB_BLOCK = kHRow ? SFB_BLOCK : 0;
// kHRow: the B-side SF block and the plane block land stage-major: per row box bx a [STAGES][kTokBox][16 B] run (one 2D
// TMA box of 512 B x STAGES over the stage-major z_sf / z_ex), so stage s's words of the box's rows are one contiguous
// 512 B piece. The recalibrators (w2 / w3, g2_hrow_recal) rewrite the SF piece in place ahead of the stagers, and the
// stagers load it coalesced. They ride their own full words (MB_EXB_FULL, MB_SFB_FULL); the A-side block alone rides
// MB_BLK_FULL.
inline constexpr int B_PIECE = kTokBox * 16;  // one stage of one row box: kTokBox rows x 4 words
static_assert(!kHRow || (kTokBox == 32 && kNPad % kTokBox == 0), "kHRow: the B-side row boxes are 32 rows");
inline constexpr int BSF_BOX_TX = kTokBox * (kI / kSfVec);  // one row box of the SF block (and of the plane)
inline constexpr int NS_FIT = (kSmemBudget - NBLK * (SFA_BLOCK + SFB_BLOCK + EXB_BLOCK)) / SLOT;
inline constexpr int NS = NS_FIT < NBLK * STAGES ? NS_FIT : NBLK * STAGES;
inline constexpr int RING = NS * SLOT;
inline constexpr int SFA_BLOCK_OFF = RING;
inline constexpr int SFB_BLOCK_OFF = RING + NBLK * SFA_BLOCK;
inline constexpr int EXB_BLOCK_OFF = SFB_BLOCK_OFF + NBLK * SFB_BLOCK;
inline constexpr int SMEM = EXB_BLOCK_OFF + NBLK * EXB_BLOCK;
static_assert(NS >= 3 && SMEM <= kSmemBudget, "g2 over the derived smem budget");
// the half-image geometry (unswizzled blocks: one 128 B half-image per row)
inline constexpr int NHALF = (kI / kSfVec) / 128;
inline constexpr int HALF_A = 128 * 128, HALF_B = kNPad * 128;  // bytes per half-image
static_assert(!kHRow || kSfU32I % 4 == 0, "kHRow: the row max reads the plane row as 16-B vectors");
inline constexpr int ACC_SLOTS = 2;    // D double-buffered by cell parity
inline constexpr int DRAIN_WARPS = 4;  // warps 8..11
// the g2 consumer sets: four stagers from the first drainer warp, four drainers after them (the role maps above)
inline constexpr int STAGE0 = kDrain0;
inline constexpr int DRAIN0 = kDrain0 + 4;
static_assert(STAGE0 == kG2Roles.epi0 && DRAIN0 == kG2Roles.epi0 + kG2Roles.n_epi / 2,
              "the g2 stager / drainer quads are the g2 role map's two epilogue quads (kG2Roles)");
inline constexpr int TM_SFA_N = HB * kSfCols, TM_SFB_N = kSfCols;  // the A-side slot holds every h block's words
inline constexpr int TM_D = 0, TM_DSTRIDE = HB * kNPad;            // one accumulator per h block per cell parity
inline constexpr int TM_SFA = ACC_SLOTS * TM_DSTRIDE;
inline constexpr int ALLOC_COLS = 512;
// The claim-ahead producer: it claims cell k + 1 while it issues cell k (a relaxed atomic in flight across the stage
// loop, first read after it) while every other CTA can still take two cells during this one (c + 2 x
// gridDim.x < n_cells); on a thin pool an early claim would strand a tile at the tail, so the plain post-cell claim
// runs there. Where the SF block is single-buffered (NBLK 1) the SF TMEM ring deepens to NS - 1, so MB_BLK_FREE
// rises before the producer's ring lead is spent.
inline constexpr int NSF = 2;
static_assert(NSF >= 2 && NSF <= STAGES, "g2 SF ring: double-buffered at least, a cell at most");
inline constexpr int TM_SFB = TM_SFA + NSF * TM_SFA_N;
static_assert(TM_SFB + NSF * TM_SFB_N <= ALLOC_COLS, "g2 TMEM over its allocation");
static_assert(TM_SFA % kSfCols == 0 && TM_SFA_N == HB * kSfCols, "the g2 A-side SF slots are whole 16-column blocks");
static_assert(TM_SFB % kSfCols == 0 && TM_SFB_N == kSfCols, "the g2 B-side SF slots are whole 16-column blocks");
// Each SF stage occupies distinct shared memory until all six block consumers finish.
inline constexpr int RECAL_SLOTS = NSF;
inline constexpr int RECAL_FREE_SLOTS = NSF;
enum {
    MB_FULL = 0,
    MB_EMPTY = MB_FULL + NS,
    MB_SFR = MB_EMPTY + NS,
    MB_SFF = MB_SFR + NSF,
    MB_BLK_FULL = MB_SFF + NSF,
    MB_BLK_FREE = MB_BLK_FULL + NBLK,
    // kHRow: the B-side blocks' own full words — the plane (issued first) and the SF block (w2 waits them; off, the
    // names alias the next word and no index moves)
    MB_EXB_FULL = MB_BLK_FREE + NBLK,
    MB_SFB_FULL = MB_EXB_FULL + (kHRow ? NBLK : 0),
    // kHRow: the recalibration ring against the stagers: RECAL[s % NSF] (count 2: w2 + w3 arrive once the stage's
    // SF piece is rewritten), RECAL_FREE[s % NSF] (count 4: the stagers' arrivals after their loads of the stage NSF
    // earlier; pre-armed), so the recalibrators run NSF stages ahead of the stagers and never inside the MMA's
    // stage cadence
    MB_RECAL = MB_SFB_FULL + (kHRow ? NBLK : 0),
    MB_RECAL_FREE = MB_RECAL + (kHRow ? RECAL_SLOTS : 0),
    MB_ACC_FULL = MB_RECAL_FREE + (kHRow ? RECAL_FREE_SLOTS : 0),
    MB_ACC_EMPTY = MB_ACC_FULL + ACC_SLOTS,
    MB_COUNT = MB_ACC_EMPTY + ACC_SLOTS
};
// the A-side word's bytes: the W2-SF block alone under kHRow (the B-side boxes ride their own words), else the block
// plus the cell's nb row boxes of z SF
__device__ __forceinline__ constexpr int blk_tx(int nb) {
    return kHRow ? SFA_BLOCK : SFA_BLOCK + nb * BSF_BOX_TX;
}
__device__ __forceinline__ int blk_of(int ci) {
    return ci & 1;
}
__device__ __forceinline__ uint32_t blk_ph(int ci) {
    return uint32_t((ci >> 1) & 1);
}
}  // namespace g2

inline constexpr int kG2DynSmem = g2::SMEM;
static_assert(kG2DynSmem <= kSmemBudget, "g2's carve fits the MoE phase's budget");

// the steal claim ring: 8 slots above g2's barrier set (depth 8 covers the producer's lead, which the ring and block
// handshakes bound)
inline constexpr int kG2Claim = g2::MB_COUNT;
inline constexpr int kG2ClaimN = 8;
// each slot carries {c, e, off, m}: the producer's decode of the claimed cell
inline constexpr int kClaimW = 4;
inline constexpr int kG2ClaimWords = kG2ClaimN * kClaimW;  // the ring's smem words (both GEMM bodies declare it)
static_assert(kG2TablePf <= g2::STAGES, "kG2TablePf: the claim-ahead variant, a lead of 2..STAGES stages");
// the first cell's W tiles issued ahead of its gate: one (more in flight delay the gate's z poll behind them)
inline constexpr int kG2PreW = 1;
static_assert(kG2PreW >= 0 && kG2PreW <= g2::NS && kG2PreW <= g2::STAGES, "kG2PreW: within the ring and the cell");
// a CTA that ran two g1 cells (round-robin over the grid) leaves g1 at its tail: its first g2 cell's gate stays on the
// g1 producer, behind its last issue (g2_preclaim), so the tail's gate opens as early as it can
__device__ __forceinline__ bool g1_two_cells(const RouteTables& rt) {
    return int(blockIdx.x + gridDim.x) < kFbPr * *rt.n_tiles;
}
__device__ __forceinline__ void claim_cell(const int32_t* sclaim, int ci, int& e, int& off, int& m) {
    const int32_t* w = sclaim + kClaimW * (ci & 7);
    e = w[1];
    off = w[2];
    m = w[3];
}

// kHRow: w2 and w3 are the row window's recalibrators. Per cell, on the claim ring like every consumer, each waits for
// the plane's word and takes the pre-pass (one row per lane per 32-row group: the row's max provisional exponent e_row
// over the full intermediate, STAGES coalesced 16 B reads of the stage-major plane block, identical on every rank),
// then waits for the SF block's word and, stage by stage, NSF stages ahead of the stagers (RECAL_FREE: the stagers
// consumed the words of the stage NSF earlier), reads the row's four SF words and four plane words, recalibrates its
// words (w2 the even K64 cells, w3 the odd: hrow::recal_word, branch-free), writes them back over the same bytes and
// arrives RECAL (count 2). The stagers place the B side by tcgen05.st off the rewritten pieces, behind RECAL. The
// row-max hand-off w2 -> drainers is two slots indexed by the accumulator buffer (as = ci & 1): w2 writes slot `as` for
// cell ci behind its own wait on ACC_EMPTY[as] at the issuer's parity (the drainers of cell ci - 2 released the slot),
// before its RECAL arrive of the cell's stage 0, which the stagers acquire before their SFR arrive, which the issuer
// acquires before the ACC_FULL release the drainers acquire before their first read.
inline constexpr int kHRowSlot = 2;
__device__ __forceinline__ void g2_hrow_recal(const LayerArgs& a, uint64_t* mbars, const int32_t* sclaim, int n_cells,
                                              int8_t* s_erow, int k) {
    extern __shared__ __align__(1024) char smem_buf[];
    const int lane = int(threadIdx.x) & 31;
    int s_g = 0;
    for (int ci = 0;; ++ci) {
        ::ptx::mbar_wait_parity(&mbars[kG2Claim + (ci & 7)], uint32_t((ci >> 3) & 1));
        const int c = sclaim[kClaimW * (ci & 7)];
        if (c >= n_cells) break;
        int e, off, m;
        claim_cell(sclaim, ci, e, off, m);
        e = uni(e);
        off = uni(off);
        m = uni(m);
        const int n_mma = min(kNPad, ceil8(m));
        const int bb = g2::blk_of(ci);
        const int as_w = ci & 1;
        char* const exb = smem_buf + g2::EXB_BLOCK_OFF + bb * g2::EXB_BLOCK;
        char* const sfb = smem_buf + g2::SFB_BLOCK_OFF + bb * g2::SFB_BLOCK;
        // the pre-pass (both warps: every group's row max stays in the lanes that need it; a row group past the
        // cell's live rows keeps a zero exponent — its columns are never stored; its hand-off slot is written either
        // way so the drainers never read stale)
        ::ptx::mbar_wait_parity(&mbars[g2::MB_EXB_FULL + bb], g2::blk_ph(ci));
        int erow[kNGroups];
#pragma unroll
        for (int gq = 0; gq < kNGroups; ++gq) {
            erow[gq] = 0;
            if (gq == 0 || n_mma > 32 * gq) {
                uint32_t acc = 0x80808080u;  // int8 -128 x4
                const uint4* p = reinterpret_cast<const uint4*>(exb + gq * g2::STAGES * g2::B_PIECE + lane * 16);
#pragma unroll 4
                for (int s = 0; s < g2::STAGES; ++s) {
                    const uint4 v = p[s * (g2::B_PIECE / 16)];
                    acc = hrow::max4(acc, hrow::max4(hrow::max4(v.x, v.y), hrow::max4(v.z, v.w)));
                }
                erow[gq] = hrow::reduce_max4(acc);
            }
        }
        if (k == 0) {
            // the hand-off (kHRowSlot): behind this warp's own ACC_EMPTY[as] wait at the issuer's parity
            // (the drainers of cell ci - 2 are done with slot `as`)
            ::ptx::mbar_wait_parity(&mbars[g2::MB_ACC_EMPTY + as_w], uint32_t((ci >> 1) & 1));
#pragma unroll
            for (int gq = 0; gq < kNGroups; ++gq) {
                int v = erow[gq];
                s_erow[as_w * kNPad + 32 * gq + lane] = int8_t(v);
            }
        }
        ::ptx::mbar_wait_parity(&mbars[g2::MB_SFB_FULL + bb], g2::blk_ph(ci));
        for (int s = 0; s < g2::STAGES; ++s, ++s_g) {
            const int sfs = s_g % g2::NSF;
            // the stagers consumed this ring slot's previous stage (s_g - NSF): its piece may be rewritten
            ::ptx::mbar_wait_parity(&mbars[g2::MB_RECAL_FREE + sfs], uint32_t((s_g / g2::NSF) & 1));
            // one shared-memory round trip per stage: every live group's SF and exponent words are
            // loaded before any group's rewrite (the groups' pieces are STAGES x B_PIECE apart, so
            // hoisting the loads is exact)
            uint4 sw_[kNGroups], xw_[kNGroups];
#pragma unroll
            for (int gq = 0; gq < kNGroups; ++gq)
                if (gq == 0 || n_mma > 32 * gq) {
                    const int po = (gq * g2::STAGES + s) * g2::B_PIECE + lane * 16;
                    sw_[gq] = *reinterpret_cast<const uint4*>(sfb + po);
                    xw_[gq] = *reinterpret_cast<const uint4*>(exb + po);
                }
            uint32_t out_[kNGroups][kSfWords];
#pragma unroll
            for (int gq = 0; gq < kNGroups; ++gq) {
                if (gq == 0 || n_mma > 32 * gq) {
                    const uint4 sw = sw_[gq], xw = xw_[gq];
                    const uint32_t sfw[4] = {sw.x, sw.y, sw.z, sw.w}, exw[4] = {xw.x, xw.y, xw.z, xw.w};
                    // the parity k is 0 or 1 at runtime: this warp's two words by a select, never an indexed array
                    const bool odd = k != 0;
                    out_[gq][k] = hrow::recal_word(odd ? sfw[1] : sfw[0], odd ? exw[1] : exw[0], erow[gq], true);
                    out_[gq][k + 2] = hrow::recal_word(odd ? sfw[3] : sfw[2], odd ? exw[3] : exw[2], erow[gq], true);
                }
            }
#pragma unroll
            for (int gq = 0; gq < kNGroups; ++gq)
                if (gq == 0 || n_mma > 32 * gq) {
                    uint32_t* const pw =
                        reinterpret_cast<uint32_t*>(sfb + (gq * g2::STAGES + s) * g2::B_PIECE + lane * 16);
#pragma unroll
                    for (int kk = k; kk < kSfWords; kk += 2)
                        pw[kk] = out_[gq][kk];
                }
            __syncwarp();  // every lane's rewrite (and w2's hand-off store) ahead of the elected lane's release
            const int rslot = sfs;
            if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_RECAL + rslot]);
        }
        ptx::fence_async_smem();  // the rewritten pieces (generic) ahead of the block's TMA reload (async proxy)
        __syncwarp();
        if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_BLK_FREE + bb]);
    }
}

// g2's barriers (its TMEM is g1's allocation, carried over)
__device__ __forceinline__ void g2_init(uint64_t* mbars) {
    const int warp = int(threadIdx.x) >> 5;
    if (warp == kG2Roles.producer && ptx::elect_one()) {
        for (int i = 0; i < g2::NS; ++i) {
            ptx::mbar_init(&mbars[g2::MB_FULL + i], 1);
            ptx::mbar_init(&mbars[g2::MB_EMPTY + i], 1);
        }
        for (int i = 0; i < g2::NSF; ++i) {
            ptx::mbar_init(&mbars[g2::MB_SFR + i], 4);
            ptx::mbar_init(&mbars[g2::MB_SFF + i], 1);
        }
        if constexpr (kHRow) {
            for (int i = 0; i < g2::RECAL_SLOTS; ++i)
                ptx::mbar_init(&mbars[g2::MB_RECAL + i], g2::RECAL_ARRIVALS);
            for (int i = 0; i < g2::RECAL_FREE_SLOTS; ++i)
                ptx::mbar_init(&mbars[g2::MB_RECAL_FREE + i], g2::RECAL_FREE_ARRIVALS);
        }
        for (int i = 0; i < g2::NBLK; ++i) {
            ptx::mbar_init(&mbars[g2::MB_BLK_FULL + i], 1);
            ptx::mbar_init(&mbars[g2::MB_BLK_FREE + i], g2::BLK_FREE_ARRIVALS);
            if constexpr (kHRow) {
                ptx::mbar_init(&mbars[g2::MB_EXB_FULL + i], 1);
                ptx::mbar_init(&mbars[g2::MB_SFB_FULL + i], 1);
            }
        }
        for (int i = 0; i < g2::ACC_SLOTS; ++i) {
            ptx::mbar_init(&mbars[g2::MB_ACC_FULL + i], 2);
            ptx::mbar_init(&mbars[g2::MB_ACC_EMPTY + i], g2::DRAIN_WARPS);
        }
        ptx::fence_mbarrier_init_release_cluster();
    }
    __syncthreads();
    if (warp == kG2Roles.producer && ptx::elect_one()) {
        for (int i = 0; i < g2::NS; ++i)
            ptx::mbar_arrive(&mbars[g2::MB_EMPTY + i]);
        for (int i = 0; i < g2::NSF; ++i)
            ptx::mbar_arrive(&mbars[g2::MB_SFF + i]);
    }
    if (((warp >= g2::STAGE0 && warp < g2::DRAIN0) || (kHRow && (warp == kCopyWarp || warp == kEmitWarp))) &&
        ptx::elect_one())
        for (int i = 0; i < g2::NBLK; ++i)
            ptx::mbar_arrive(&mbars[g2::MB_BLK_FREE + i]);
    if constexpr (kHRow)  // the stagers pre-arm the recalibration ring's free words
        if (warp >= g2::STAGE0 && warp < g2::DRAIN0 && ptx::elect_one())
            for (int i = 0; i < g2::RECAL_FREE_SLOTS; ++i)
                ptx::mbar_arrive(&mbars[g2::MB_RECAL_FREE + i]);
    if (warp >= g2::DRAIN0 && ptx::elect_one())
        for (int i = 0; i < g2::ACC_SLOTS; ++i)
            ptx::mbar_arrive(&mbars[g2::MB_ACC_EMPTY + i]);
    __syncthreads();
}

// The g2 steal loop: hb0 = the rank's first global h block (zb offsets); the weight map's depth is the expert over
// the shard pack. GATE: each cell polls its tile's z rows until every element is in. STEAL: cells are
// claimed off one shared cursor, so a claim is almost always for a tile the peers have already written. Claims reach
// the consumer roles through the smem claim ring and its per-slot barriers.

// kHRow: the B-side blocks of a cell, per row box bx: the plane first (the recalibrators' pre-pass starts on it
// while the SF block is still landing), then the SF block, each one box landing stage-major ([stage][row][16 B]) on
// its own word; the B side is not swizzled under the row window
__device__ __forceinline__ void g2_issue_b_box(const LayerArgs& a, uint32_t smem, uint64_t* mbars, int bb, int off,
                                               int bx) {
    // the 2D box: x = the box's first row in u32 units of the stage-major row (4 per row), y = stage 0; STAGES rows
    const int32_t x = (off + bx * kTokBox) * 4;
    ptx::cp_async_bulk_tensor_2d_load(smem + g2::EXB_BLOCK_OFF + bb * g2::EXB_BLOCK + bx * g2::STAGES * g2::B_PIECE,
                                      &a.tmaps.tm[TM_G2_EXB], x, 0, &mbars[g2::MB_EXB_FULL + bb]);
    ptx::cp_async_bulk_tensor_2d_load(smem + g2::SFB_BLOCK_OFF + bb * g2::SFB_BLOCK + bx * g2::STAGES * g2::B_PIECE,
                                      &a.tmaps.tm[TM_G2_SFB], x, 0, &mbars[g2::MB_SFB_FULL + bb]);
}

// The producer, on the whole warp. Per claimed cell: (NVFP4) the cell's W2 SF block issued ahead of the gate, its
// latency under the poll; the warp polls the tile's live z rows (lam_poll_tile), then the leader's proxy fence orders
// every lane's observations ahead of its async-proxy reads and it issues the cell at MoE pass e's parity: the B-side
// boxes (kHRow) and the stage loop. The claim-ahead: the next cell is claimed while this one issues while every other
// CTA can still take two cells during this one (c + 2 x gridDim.x < n_cells); on a thin pool an early claim would
// strand a tile at the tail, so the plain claim kG2TablePf stages before the loop's end runs there. The first cell was
// claimed behind the CTA's last g1 issue (g2_preclaim); on a CTA that ran one g1 cell its first W tiles fill the free
// ring ahead of its gate, so they land during the stop-3 wait. Behind its last claim the warp counts itself for the
// next-layer phase's close (next_layer.cuh: the cursor reset waits for every CTA's final claim) and acquires the
// anchor.
__device__ __forceinline__ void g2_producer(const LayerArgs& a, uint64_t* mbars, uint32_t smem, const RouteTables& rt,
                                            uint32_t mep, uint32_t* mcur, int32_t* sclaim, int n_cells, int lane) {
    const int zr0 = zpar_row0(mep);
    const bool leader = lane == 0;
    int s_g = 0;
    int cn = 0;
    int c = sclaim[0], e1 = sclaim[1], off1 = sclaim[2], m1 = sclaim[3];
    for (int ci = 0; c < n_cells; ++ci) {
        const bool pre = c + 2 * int(gridDim.x) < n_cells;
        if (leader) cn = pre ? int(ptx::atomic_add_relaxed_b32(mcur, 1u)) : 0;
        const int hbl = (c % g2::CELLS_PT) * g2::HB;  // the cell's first shard-local h block
        const int e = e1, off = off1, m = m1;
        const int bb = g2::blk_of(ci);
        if constexpr (!kBf16)
            if (leader) {
                ::ptx::mbar_wait_parity(&mbars[g2::MB_BLK_FREE + bb], g2::blk_ph(ci));
                wtma3d(smem + g2::SFA_BLOCK_OFF + bb * g2::SFA_BLOCK, &a.tmaps.tm[TM_G2_SFA], 0, hbl * 128, e,
                       &mbars[g2::MB_BLK_FULL + bb]);
                ptx::mbar_arrive_expect_tx(&mbars[g2::MB_BLK_FULL + bb], g2::blk_tx(1));
            }
        // each pre-issued slot's full word expects the W tile and the z box, which follows behind the gate
        const bool g1_gated = ci == 0 && g1_two_cells(rt);
        const int pre_w = ci == 0 && !g1_gated ? kG2PreW : 0;
        if (leader)
#pragma unroll 1
            for (int s = 0; s < pre_w; ++s) {
                const int slot = (s_g + s) % g2::NS;
                ::ptx::mbar_wait_parity(&mbars[g2::MB_EMPTY + slot], uint32_t((s_g + s) / g2::NS) & 1);
                wtma3d(smem + slot * g2::SLOT, &a.tmaps.tm[TM_G2_A], 0, (hbl * g2::STAGES + s) * 128, e,
                       &mbars[g2::MB_FULL + slot]);
                ptx::mbar_arrive_expect_tx(&mbars[g2::MB_FULL + slot], g2::A_BYTES + kTokBox * kBKBytes);
            }
        __syncwarp();
        if (!g1_gated) lam_poll_tile({a.z_fp4, a.z_sf, a.z_ex}, a.rank, zr0 + off, m, lane, c / g2::CELLS_PT);
        __syncwarp();
        if (leader) {
            ptx::fence_async_global();
            auto table_ahead = [&](int s) {
                if (s == g2::STAGES - kG2TablePf) {
                    if (!pre) cn = int(ptx::atomic_add_relaxed_b32(mcur, 1u));
                } else if (s == g2::STAGES - kG2TablePf + 1 && cn < n_cells) {
                    tile_cell(rt, cn / g2::CELLS_PT, e1, off1, m1);
                }
            };
            if constexpr (kHRow) {
                g2_issue_b_box(a, smem, mbars, bb, zr0 + off, 0);
                ptx::mbar_arrive_expect_tx(&mbars[g2::MB_EXB_FULL + bb], g2::BSF_BOX_TX);
                ptx::mbar_arrive_expect_tx(&mbars[g2::MB_SFB_FULL + bb], g2::BSF_BOX_TX);
            }
#pragma unroll(mk::cfg::kWorld.text_diet ? 1 : g2::STAGES)
            for (int s = 0; s < g2::STAGES; ++s, ++s_g) {
                const int slot = s_g % g2::NS;
                const uint32_t a_dst = smem + slot * g2::SLOT;
                if (s >= pre_w) {
                    ::ptx::mbar_wait_parity(&mbars[g2::MB_EMPTY + slot], uint32_t(s_g / g2::NS) & 1);
                    wtma3d(a_dst, &a.tmaps.tm[TM_G2_A], 0, (hbl * g2::STAGES + s) * 128, e, &mbars[g2::MB_FULL + slot]);
                }
                ptx::cp_async_bulk_tensor_2d_load(a_dst + g2::A_BYTES, &a.tmaps.tm[TM_G2_B], s * kBK, zr0 + off,
                                                  &mbars[g2::MB_FULL + slot]);
                if (s >= pre_w)
                    ptx::mbar_arrive_expect_tx(&mbars[g2::MB_FULL + slot], g2::A_BYTES + kTokBox * kBKBytes);
                table_ahead(s);
            }
            int32_t* const w = sclaim + kClaimW * ((ci + 1) & 7);
            w[1] = e1;
            w[2] = off1;
            w[3] = m1;
            w[0] = cn;
            ptx::mbar_arrive(&mbars[kG2Claim + ((ci + 1) & 7)]);
        }
        c = __shfl_sync(0xFFFFFFFFu, cn, 0);
        e1 = __shfl_sync(0xFFFFFFFFu, e1, 0);
        off1 = __shfl_sync(0xFFFFFFFFu, off1, 0);
        m1 = __shfl_sync(0xFFFFFFFFu, m1, 0);
    }
    if (leader) ptx::red_add_rel_b32(a.gsync + 3, 1u);
    lam_anchor_acquire(a, mep, lane);
}

// the first cell's claim, on the g1 producer warp behind its last issue (published on the claim ring at once: its
// consumers wait behind the phase boundary), and on a CTA that ran two g1 cells its gate (else the g2 producer's,
// behind the cell's first W tiles); the gate's observations reach the leader's proxy fence through the phase
// boundary's barriers
__device__ __forceinline__ void g2_preclaim(const LayerArgs& a, uint64_t* mbars, int32_t* sclaim, uint32_t e,
                                            int lane) {
    const RouteTables& rt = a.route;
    const int n_cells = g2::CELLS_PT * *rt.n_tiles;
    int c = 0, e1 = 0, off1 = 0, m1 = 0;
    if (lane == 0) {
        // relaxed: the cursor guards no data (claims travel through the smem ring and its CTA-scope barriers)
        c = int(ptx::atomic_add_relaxed_b32(a.gsync + 4, 1u));
        if (c < n_cells) tile_cell(rt, c / g2::CELLS_PT, e1, off1, m1);
        sclaim[1] = e1;
        sclaim[2] = off1;
        sclaim[3] = m1;
        sclaim[0] = c;
        ptx::mbar_arrive(&mbars[kG2MbarBase + kG2Claim]);
    }
    if (!g1_two_cells(rt)) return;
    c = __shfl_sync(0xFFFFFFFFu, c, 0);
    off1 = __shfl_sync(0xFFFFFFFFu, off1, 0);
    m1 = __shfl_sync(0xFFFFFFFFu, m1, 0);
    if (c < n_cells) lam_poll_tile({a.z_fp4, a.z_sf, a.z_ex}, a.rank, zpar_row0(e) + off1, m1, lane, c / g2::CELLS_PT);
}

__device__ __forceinline__ void g2_loop(const LayerArgs& a, uint64_t* mbars, const uint32_t taddr,
                                        const RouteTables& rt, __nv_bfloat16* zb, int hb0, uint32_t mep, uint32_t* mcur,
                                        int32_t* sclaim, int8_t* s_erow) {
    extern __shared__ __align__(1024) char smem_buf[];
    const uint32_t smem = ptx::to_shared(smem_buf);
    const uint32_t* smem_u32 = reinterpret_cast<const uint32_t*>(smem_buf);
    const int tid = threadIdx.x, warp = uni(tid >> 5), lane = tid & 31;
    const int n_cells = uni(g2::CELLS_PT * (*rt.n_tiles));

    if (warp == kG2Roles.producer) {
        // ================= producer =================
        g2_producer(a, mbars, smem, rt, mep, mcur, sclaim, n_cells, lane);
    } else if (warp == kG2Roles.mma) {
        // ================= MMA issuer =================
        // the operands built once and stepped, as g1's issuer (desc_step)
        const uint64_t da0 = ptx::mma_smem_desc_k_major<uint8_t, kBKBytes, 128>(smem);
        const uint64_t db0 = ptx::mma_smem_desc_k_major<uint8_t, kBKBytes, 128>(smem + g2::A_BYTES);
        constexpr uint64_t kDSlot = desc_step(g2::SLOT), kDHb = desc_step(128 * kBKBytes);
        int slot = 0, sfs = 0;
        uint32_t par = 0, sfp = 0;
        auto cell_body = [&](int ci, int c) {
            int e, off, m;
            claim_cell(sclaim, ci, e, off, m);
            e = uni(e);
            off = uni(off);
            m = uni(m);
            const int n_mma = min(kNPad, ceil8(m));
            const int as = ci & 1;  // ACC slot (cell parity)
            const uint32_t id = mma_idesc(n_mma);
            const uint32_t d = taddr + g2::TM_D + as * g2::TM_DSTRIDE;
            ::ptx::mbar_wait_parity(&mbars[g2::MB_ACC_EMPTY + as], uint32_t((ci >> 1) & 1));
            for (int s = 0; s < g2::STAGES; ++s) {
                ::ptx::mbar_wait_parity(&mbars[g2::MB_FULL + slot], par);
                if constexpr (!kBf16) ::ptx::mbar_wait_parity(&mbars[g2::MB_SFR + sfs], sfp);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one()) {
                    const uint64_t da = da0 + uint64_t(slot) * kDSlot, db = db0 + uint64_t(slot) * kDSlot;
#pragma unroll(mk::cfg::kWorld.text_diet ? 1 : kKPerTile)
                    for (int kb = 0; kb < kKPerTile; ++kb) {
                        const uint64_t dak = da + uint64_t(kb) * kDescK, dbk = db + uint64_t(kb) * kDescK;
                        const uint32_t scale_c = (s == 0 && kb == 0) ? 0u : 1u;
                        if constexpr (!kBf16) {
                            const uint32_t sfa = taddr + g2::TM_SFA + sfs * g2::TM_SFA_N + kb * 4;
                            const uint32_t sfb = taddr + g2::TM_SFB + sfs * g2::TM_SFB_N + kb * 4;
                            ptx::tcgen05_mma_mxf4nvf4_block16(d, dak, dbk, id, scale_c, sfa, sfb);
                        } else {
                            // kBf16: bf16 x bf16 -> fp32, no scale factors; h block hj's A tile and accumulator
                            // against the one B side
#pragma unroll
                            for (int hj = 0; hj < g2::HB; ++hj)
                                ptx::tcgen05_mma_f16(d + hj * kNPad, dak + uint64_t(hj) * kDHb, dbk, id, scale_c);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&mbars[g2::MB_EMPTY + slot]);
                    if constexpr (!kBf16) ptx::tcgen05_commit_arrive(&mbars[g2::MB_SFF + sfs]);
                }
                ring_next(slot, par, g2::NS);
                if constexpr (!kBf16) ring_next(sfs, sfp, g2::NSF);
            }
            if (ptx::elect_one()) {
                ptx::tcgen05_commit_arrive(&mbars[g2::MB_ACC_FULL + as]);
                ptx::mbar_arrive_release(&mbars[g2::MB_ACC_FULL + as]);
            }
        };
        for (int ci = 0;; ++ci) {
            ::ptx::mbar_wait_parity(&mbars[kG2Claim + (ci & 7)], uint32_t((ci >> 3) & 1));
            const int c = sclaim[kClaimW * (ci & 7)];
            if (c >= n_cells) break;
            cell_body(ci, c);
        }
    } else if (warp < g2::STAGE0) {
        // kHRow: w2 and w3 are the row window's recalibrators (g2_hrow_recal), w2 also the B-side placer;
        // otherwise w2 / w3 have no g2 role (block barriers only)
        if constexpr (kHRow) {  // w2 the even K64 cells' recalibrator, w3 the odd (after w3's emit duty in g1)
            if (warp == kCopyWarp || warp == kEmitWarp)
                g2_hrow_recal(a, mbars, sclaim, n_cells, s_erow, warp == kEmitWarp ? 1 : 0);
        }
    } else {
        // ================= B-side SF stagers (w4-7) | drain epilogue (w8-11) =================
        const int band = warp & 3;
        int s_g = 0;
        auto cell_body = [&](int ci, int c) {
            const int hbg = hb0 + (c % g2::CELLS_PT) * g2::HB;  // the cell's first GLOBAL block (zb offsets)
            int e, off, m;
            claim_cell(sclaim, ci, e, off, m);
            e = uni(e);
            off = uni(off);
            m = uni(m);
            const int n_mma = min(kNPad, ceil8(m));
            const int bb = g2::blk_of(ci);
            if (warp < g2::DRAIN0) {
                // kBf16: no scale factors to stage — the stagers have no g2 role
                if constexpr (!kBf16) {
                    ::ptx::mbar_wait_parity(&mbars[g2::MB_BLK_FULL + bb], g2::blk_ph(ci));
                    const uint32_t* sfa_blk = smem_u32 + (g2::SFA_BLOCK_OFF + bb * g2::SFA_BLOCK) / 4;
                    const uint32_t* sfb_blk = smem_u32 + (g2::SFB_BLOCK_OFF + bb * g2::SFB_BLOCK) / 4;
                    // kHRow: the B side comes off the stage-major pieces the recalibrators
                    // rewrote in place, behind RECAL[s % NSF]; the loads are coalesced (lane i:
                    // row i's 16 B) and the words go to TMEM by the same stores as the plain
                    // variant's
                    for (int s = 0; s < g2::STAGES; ++s, ++s_g) {
                        const int sfs = s_g % g2::NSF;
                        ::ptx::mbar_wait_parity(&mbars[g2::MB_SFF + sfs], uint32_t((s_g / g2::NSF) & 1));
                        const uint32_t lane_band = uint32_t(band * 32) << 16;
                        // the register hoist of the g1 staging, with the same multi-group
                        // B-side variant and tallness predicate
                        uint32_t wa[kSfWords], wb[kNGroups][kSfWords];
                        const int kc0 = kSfWords * s;
                        const bool two = kNGroups > 1 && n_mma > 32;
                        sf_load_stage(wa, sfa_blk + (band * 32 + lane) * kSfU32I + kc0);
                        if constexpr (!kHRow) {
                            if constexpr (kNGroups <= 2) {
                                sf_load_stage(wb[0], sfb_blk + lane * kSfU32I + kc0);
                                if (kNGroups > 1 && two)
                                    sf_load_stage(wb[kNGroups - 1], sfb_blk + (32 + lane) * kSfU32I + kc0);
                            } else {
                                sf_load_groups(wb, sfb_blk + lane * kSfU32I + kc0, kSfU32I, n_mma);
                            }
                        }
                        if constexpr (kHRow) {
                            const int rslot = sfs;
                            const uint32_t rphase = uint32_t((s_g / g2::NSF) & 1);
                            ::ptx::mbar_wait_parity(&mbars[g2::MB_RECAL + rslot], rphase);
#pragma unroll
                            for (int gq = 0; gq < kNGroups; ++gq) {
                                if (gq == 0 || n_mma > 32 * gq)
                                    sf_load_stage(wb[gq],
                                                  sfb_blk + ((gq * g2::STAGES + s) * g2::B_PIECE) / 4 + lane * 4);
                                else
#pragma unroll
                                    for (int kk = 0; kk < kSfWords; ++kk)
                                        wb[gq][kk] = 0u;
                            }
                            __syncwarp();
                            if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_RECAL_FREE + sfs]);
                        }
                        {
                            // the A side's word is the same for all four columns of its K64 cell
                            uint32_t ra[kSfCols];
#pragma unroll
                            for (int kk = 0; kk < kSfWords; ++kk)
                                ra[kk * 4 + 0] = ra[kk * 4 + 1] = ra[kk * 4 + 2] = ra[kk * 4 + 3] = wa[kk];
                            ptx::tcgen05_st_32x32b_x16((taddr + g2::TM_SFA + sfs * g2::TM_SFA_N) | lane_band, ra);
                        }
                        sf_store_slot_x16((taddr + g2::TM_SFB + sfs * g2::TM_SFB_N) | lane_band, wb, n_mma);
                        ptx::tcgen05_wait_st();
                        ptx::tcgen05_fence_after_thread_sync();
                        if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_SFR + sfs]);
                    }
                    ptx::fence_async_smem();
                    __syncwarp();
                    if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_BLK_FREE + bb]);
                }  // !kBf16
            } else {
                // ---- epilogue: the one bf16 rounding of the whole MoE pass ----
                // The tile's scale is the expert constant alone: h's SF codes are absolute (the g1
                // epilogue), so nothing per column is staged ahead of the accumulator wait.
                const float g2a = kBf16 ? 1.0f : __ldg(a.g2_alphas + e);  // kBf16: alpha 1
                constexpr int kStgRegs = kNPad / 32;
                static_assert(kNPad % 32 == 0, "staging puts one column per lane");
                int own[kStgRegs];
                const int as = ci & 1;
                ::ptx::mbar_wait_parity(&mbars[g2::MB_ACC_FULL + as], uint32_t((ci >> 1) & 1));
                ptx::tcgen05_fence_after_thread_sync();
                const uint32_t lane_band = uint32_t(band * 32) << 16;
                const int h = hbg * 128 + band * 32 + lane;  // the h block's column
                // kHRow: the column's alpha is g2a x 2^e_row (a power of two: exact), one column
                // per lane off the recalibrator's (w2's) hand-off slot of this accumulator
                // (kHRowSlot; ordering in g2_hrow_recal), broadcast per column by shuffle below; the
                // slot is free again once this cell's ACC_EMPTY arrive below
                [[maybe_unused]] float alpha[kStgRegs];
                if constexpr (kHRow) {
#pragma unroll
                    for (int u = 0; u < kStgRegs; ++u)
                        alpha[u] = __fmul_rn(g2a, hrow::pow2i(int(s_erow[as * kNPad + u * 32 + lane])));
                }
                const ptrdiff_t zpar = zb - a.zb;  // this epoch's zb plane
                {
                    constexpr int hj = 0;
                    for (int c0 = 0; c0 < n_mma; c0 += 8) {
                        int owner[8];
                        [[maybe_unused]] float ga[8];
                        if constexpr (kHRow) {
                            float av = alpha[0];
#pragma unroll
                            for (int u = 1; u < kStgRegs; ++u)
                                if (c0 >= u * 32) av = alpha[u];
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                                ga[j] = __shfl_sync(0xFFFFFFFFu, av, (c0 + j) & 31);
                        }
                        uint32_t r[8];
                        ptx::tcgen05_ld_32x32b_x8(
                            (taddr + g2::TM_D + as * g2::TM_DSTRIDE + hj * kNPad + c0) | lane_band, r[0], r[1], r[2],
                            r[3], r[4], r[5], r[6], r[7]);
                        ptx::tcgen05_wait_ld();
#pragma unroll 1
                        for (int j = 0; j < 8; ++j) {
                            const int col = c0 + j;
                            if (col < m) {
                                const uint32_t rv = j == 0   ? r[0]
                                                    : j == 1 ? r[1]
                                                    : j == 2 ? r[2]
                                                    : j == 3 ? r[3]
                                                    : j == 4 ? r[4]
                                                    : j == 5 ? r[5]
                                                    : j == 6 ? r[6]
                                                             : r[7];
                                float av = g2a;
                                if constexpr (kHRow) {
                                    av = j == 0   ? ga[0]
                                         : j == 1 ? ga[1]
                                         : j == 2 ? ga[2]
                                         : j == 3 ? ga[3]
                                         : j == 4 ? ga[4]
                                         : j == 5 ? ga[5]
                                         : j == 6 ? ga[6]
                                                  : ga[7];
                                }
                                const __nv_bfloat16 v = __float2bfloat16_rn(__fmul_rn(__int_as_float(rv), av));
                                const size_t o = size_t(off + col) * kH + h;
                                // gather 2 to every rank, from the register (the fan-out): four consecutive columns'
                                // values as one 8 B lamport element (the combine polls the same elements)
                                const uint32_t u = __bfloat16_as_ushort(v);
                                const uint32_t u2 = u | (__shfl_down_sync(0xFFFFFFFFu, u, 1) << 16);
                                const uint64_t u4 =
                                    uint64_t(u2) | (uint64_t(__shfl_down_sync(0xFFFFFFFFu, u2, 2)) << 32);
                                if ((lane & 3) == 0)
#pragma unroll
                                    for (int q = 0; q < kRanks; ++q)
                                        ptx::st_relaxed_sys_b64(a.zb_peer[q] + zpar + o, u4);
                            }
                        }
                    }
                }
                __syncwarp();  // order all lanes' tcgen05.wait::ld reads before the elected arrive
                if (ptx::elect_one()) ptx::mbar_arrive(&mbars[g2::MB_ACC_EMPTY + as]);
            }
        };
        for (int ci = 0;; ++ci) {
            ::ptx::mbar_wait_parity(&mbars[kG2Claim + (ci & 7)], uint32_t((ci >> 3) & 1));
            const int c = sclaim[kClaimW * (ci & 7)];
            if (c >= n_cells) break;
            cell_body(ci, c);
        }
    }
    __syncthreads();
    if (warp == kG2Roles.mma) {
        ptx::tcgen05_dealloc(taddr, g2::ALLOC_COLS);
    }
}

// The MoE phase's two GEMMs on the tensor parallel ranks: g1 (the rank's W1 intermediate shard, gate + up, then
// SwiGLU and NVFP4 requant into z) and g2 (the rank's W2 output shard over the FULL intermediate, the MoE pass's one
// bf16 rounding, into zb).
//
// Engine: tcgen05 mxf4nvf4.block_scale.block16 K64 (kind::f16 on the bf16 layers, kBf16), BK = 256 fp4 (128 B
// rows, SWIZZLE_128B, 4 K64 MMAs per stage), the weights by per-expert 3D TMA boxes (evict-first: zero reuse), the
// token rows by gather4 with out-of-bounds zero fill, the scale factors into TMEM by tcgen05.cp (weights) and
// tcgen05.st (tokens).
//
//  g1  cell = (token tile, feature block) over the rank's kFbPr blocks; z lands at the block's global offset in every
//      rank's gathered z (the epilogue's register fan-out). h = silu(bf16(acc_g x a)) x bf16(acc_u x a) x g1c, per-16
//      amax / 6 e4m3 scale factors (no floor), e2m1 codes.
//  g2  K is the full intermediate and the cell's 128 A rows are output columns, so every value it produces is final:
//      zb = bf16(acc x g2a).

// The hoisted entry (moe_body): the epoch read and the barrier / TMEM init run ahead of
// the tables' word. The first drainer warp acquires the tables' word, reads cell 0's tile, row indices and tile
// count into shared memory and issues cell 0's W L2 prefetch, while thread 0 prefetches the descriptors and polls
// the rows' word; the tables' acquire reaches the drainer through the MB_ROWS barrier (one gpu-scope poller per
// CTA), and one block sync publishes both.
// The dry pass: cell 0's g1 epilogue body runs once ahead of it on the emit warp and the non-stager drainer warps
// (w8-w11), in the tables -> rows window, on the accumulator's garbage, with no waits and no arrives: the same
// instruction addresses, so every CTA's first drain finds its lines in the instruction cache. The staging it writes
// is rewritten by the real cell before any emit.

inline constexpr int kMoeMbars = kG2MbarBase + g2::MB_COUNT + kG2ClaimN;        // the array follows the placement
inline constexpr int kMoeSmem = g1::SMEM > kG2DynSmem ? g1::SMEM : kG2DynSmem;  // kG2DynSmem: g2's carve
static_assert(kMoeSmem <= kSmemBudget, "fused stage over the smem budget");
// The carve descriptors (framework.cuh): what each phase exports and what the launch composes. Dynamic smem =
// the max of the phases (they alias one carve), the barrier words = the sum (g2's set sits at g1's end), one TMEM
// allocation re-carved per phase; g1's static smem is the front overlay (FrontZsOvl, recycled as the z staging
// window).
inline constexpr mk::carve::Carve kG1Carve{g1::SMEM, kFrontStaticSmem, g1::MB_END, g1::ALLOC_COLS};
inline constexpr mk::carve::Carve kG2Carve{kG2DynSmem, 0, g2::MB_COUNT + kG2ClaimN, g2::ALLOC_COLS};
inline constexpr mk::carve::Carve kMoeCarve = mk::carve::phases(kG1Carve, kG2Carve);
static_assert(kMoeCarve.dyn_smem == kMoeSmem, "the MoE phase's dynamic smem is the max of its phases' carves");
static_assert(kMoeCarve.mbars == kMoeMbars, "the MoE phase's mbarrier array is the sum of its phases' sets");
static_assert(kMoeCarve.tmem_cols == g1::ALLOC_COLS, "one TMEM allocation per launch, re-carved by the phases");

// the MoE phase's barrier words and TMEM: g1's set and the zjoin / staged-emission pairs
__device__ __forceinline__ void moe_init(uint64_t* mbars, uint32_t* tmem_addr_storage) {
    g1_init(mbars, tmem_addr_storage);
    if (int(threadIdx.x) >> 5 == 0 && ptx::elect_one()) {
        ptx::mbar_init(&mbars[g1::MB_COUNT + 0], 8);  // zjoin pair
        ptx::mbar_init(&mbars[g1::MB_COUNT + 1], 8);
        // Staged-emission WAR pair: count 1 = the sole emitter's elected
        // arrive; init pre-arrive x1 so cells 0/1 stage a buffer no
        // emission has owned yet.
        ptx::mbar_init(&mbars[g1::MB_ZSE + 0], 1);
        ptx::mbar_init(&mbars[g1::MB_ZSE + 1], 1);
        ptx::mbar_arrive(&mbars[g1::MB_ZSE + 0]);
        ptx::mbar_arrive(&mbars[g1::MB_ZSE + 1]);
        // g2's steal claim ring, armed at kernel start
        for (int i = 0; i < kG2ClaimN; ++i)
            ptx::mbar_init(&mbars[kG2MbarBase + kG2Claim + i], 1);
        ptx::fence_mbarrier_init_release_cluster();
    }
    __syncthreads();
}

// the row window's per-slot max exponents, stagers -> drainers, one slot per accumulator buffer (kHRowSlot): static
// smem only on the images that run the row window
template <bool kOn>
__device__ __forceinline__ int8_t* erow_smem() {
    if constexpr (kOn) {
        __shared__ __align__(16) int8_t s_erow[kHRowSlot * kNPad];
        return s_erow;
    } else {
        return nullptr;
    }
}

__device__ __forceinline__ void moe_body(const LayerArgs& a, __nv_bfloat16* zb, uint32_t e, ZStageBuf* zst,
                                         const uint32_t* front_word, uint32_t front_target, const uint32_t* rows_word,
                                         uint32_t rows_target) {
    // the MoE phase's static smem (its placement fixes every barrier's address)
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ __align__(8) uint64_t mbars[kMoeMbars];
    __shared__ __align__(4) uint32_t tmem_addr_storage[1];
    __shared__ __align__(4) int32_t s_claim[kG2ClaimWords];
    // the hoisted entry: cell 0's (e, off, m), its tile count and its kNPad row indices, read ahead of the rows' word
    __shared__ __align__(16) int32_t s_cell0[4 + kNPad];
    int8_t* const s_erow = erow_smem<kHRow>();
    moe_init(mbars, tmem_addr_storage);
    {
        static_assert(kNPad == 32, "cell 0's row indices are read one lane each");
        const int c = int(blockIdx.x);
        // the previous MoE pass's z rows and zb plane back to the sentinel, on the warps idle in this
        // window (the previous launch's readers are done: this MoE phase runs behind the front's word, behind its grid
        // wait)
        const int w = int(threadIdx.x) >> 5;
        if (w != 0 && w != kDrain0) {
            const int t = int(threadIdx.x) - 32 * (w < kDrain0 ? 1 : 2);
            lam_reset_z(a, e, t, kMoeThreads - 64);
            lam_reset_zb(a, e, t, kMoeThreads - 64);
        }
        if (threadIdx.x == 0) {
            // the tables' word, the descriptors, then the rows' word — no table read on this thread
            ptx::count_spin(front_word, front_target);
            ptx::fence_async_global();
            // the relay variant: the tables' acquire handed to the drainer warp through the MoE phase's MB_ROWS
            // mbarrier (initialized by moe_init, otherwise unused) — ONE gpu-scope poller per CTA on the tables' word
            (void)ptx::mbar_arrive(&mbars[g1::MB_ROWS]);
            for (int i = 0; i < kTmaps; ++i)
                ptx::prefetch_tensormap(&a.tmaps.tm[i]);
            ptx::prefetch_tensormap(&a.tm_sfab);
            ptx::count_spin(rows_word, rows_target);
            ptx::fence_async_global();
        } else if ((threadIdx.x >> 5) == kDrain0) {
            // the first drainer warp: the tables' word (thread 0's acquire relayed through MB_ROWS at
            // cta scope), then cell 0's tile, its row indices and its W prefetch, beside thread 0's
            // acquire instead of behind it
            const int lane7 = threadIdx.x & 31;
            if (lane7 == 0) {
                ptx::mbar_wait_parity(&mbars[g1::MB_ROWS], 0);
            }
            __syncwarp();
            const int nt = __ldcg(a.route.n_tiles);
            const bool has = c < kFbPr * nt;
            int e0 = 0, off0 = 0, m0 = 0;
            if (has) tile_cell(a.route, c / kFbPr, e0, off0, m0);
            if (lane7 == 0) {
                s_cell0[0] = e0;
                s_cell0[1] = off0;
                s_cell0[2] = m0;
                s_cell0[3] = nt;  // the tile count for every warp's entry
                if (has) {        // cell 0's weights into L2
                    const int fbl = c % kFbPr;
                    if constexpr (g1::SFPF)
                        ptx::cp_async_bulk_prefetch_tensor_3d_l2(&a.tm_sfab, 0, 0, (e0 * kFbFull + fbl) * g1::NCHUNK);
                    for (int s = 0; s < g1::PRE; ++s)
                        ptx::cp_async_bulk_prefetch_tensor_3d_l2(&a.tmaps.tm[TM_G1_A], 0, s * kARows,
                                                                 e0 * kFbFull + fbl);
                }
            }
            s_cell0[4 + lane7] = (has && lane7 < m0) ? a.route.slot_tok[off0 + lane7] : int32_t(kOobRow);
        }
        __syncthreads();  // the tables, the rows and cell 0's tile reach the block
    }
    g1_loop(a, mbars, tmem_addr_storage[0], a.route, a.rank * kFbPr, e, zst, s_claim, s_cell0);
    uint64_t* const mb2 = mbars + kG2MbarBase;
    __syncthreads();  // the phase boundary's join
    g2_init(mb2);
    // g2's set initialised after the phase boundary's join; the barrier keeps the phase mark and orders it for every
    // role
    __syncthreads();
    g2_loop(a, mb2, tmem_addr_storage[0], a.route, zb, a.rank * kHbPr, e, a.gsync + 4, s_claim, s_erow);
}

// The combine's row math: per-slot RMSNorm of the gathered zb rows, then the router-weighted sum into y, run on
// every rank over the full gathered surface (no cross-rank arithmetic, so y is bitwise-identical on every rank). Run
// as the next-layer phase's combine item (next_layer.cuh) on a CTA pair per token.
// The reduction order is fixed: level A, per lane an fma chain over 8 consecutive columns of a 128-column n-block,
// then the 16-lane xor butterfly 8 / 4 / 2 / 1; level B, ((x + y) + z) + w float4 trees over the kCombNb n-block
// partials, lane l summing trees l, l + 4, ... ascending, then (p0 + p1) + (p2 + p3); inv_rms =
// rcp.approx.ftz(sqrt.rn(fma(ss, 1 / H, eps))); one fp32 fma accumulator over ascending k; one bf16 round at the y
// store.

__device__ __forceinline__ float rstd_pinned(float ss, float eps) {
    const float m = fmaf(ss, 1.0f / float(kH), eps);
    return ptx::rcp_approx_ftz_f32(ptx::sqrt_rn_f32(m));
}

// Level B over the kCombNb n-block partials (stride kTopK rows in s_red): ((x + y) + z) + w float4 trees, then lane l
// sums trees l, l + 4, ... ascending, then (p0 + p1) + (p2 + p3).
__device__ __forceinline__ float level_b_reduce(const float* red, int stride) {
    float s4[kCombNb / 4];
#pragma unroll
    for (int i = 0; i < kCombNb / 4; ++i)
        s4[i] =
            __fadd_rn(__fadd_rn(__fadd_rn(red[(4 * i) * stride], red[(4 * i + 1) * stride]), red[(4 * i + 2) * stride]),
                      red[(4 * i + 3) * stride]);
    float p[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        p[l] = s4[l];
#pragma unroll
        for (int i = l + 4; i < kCombNb / 4; i += 4)
            p[l] = __fadd_rn(p[l], s4[i]);
    }
    return __fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3]));
}

// CTAs per token: the row's n-blocks split over a CTA pair
inline constexpr int kCombPair = 2;

// the combine's arguments
struct CombArgs {
    const int32_t* tok_slot;
    const float* topk_w;
    const __nv_bfloat16* zb;
    const __nv_bfloat16* gamma_p1;
    __nv_bfloat16* y;
    float eps;
    int B, b_live;
    uint32_t pair_rank;  // the pair rank: n-blocks [rank x kCombNb / kCombPair, ...)
    uint64_t* xchg_bar;  // the hosting body's exchange mbarrier (the pair's level-A partials)
    uint32_t pair_peer;  // the peer's cluster rank (mapa's operand; in a width-4 cluster not pair_rank ^ 1)
    int32_t tok0;        // the row's token (the item's claimed block id / kCombPair)
};

// The row tree over kGroups lane groups (16 x kGroups threads, kCombNb / (kGroups x kCombPair) n-blocks each), on the
// item's warps 4..11 behind their named barrier. s_red is indexed by n-block, so both CTAs of the pair fill the same 32
// partial slots and level_b_reduce reduces the same words in the same order.
// The item reads the token's slot ids and weights ahead of its readiness wait (the tables are complete before the
// item runs), so the row's first dependent round trip leaves the critical path.
struct CombPre {
    int sl[kTopK];
    float wk[kTopK];
};
// the pair variant's one token (t = its cluster index): its slot ids and weights, or -1 / 0 past B / b_live
__device__ __forceinline__ CombPre combine_prep(const CombArgs& a, const int t) {
    CombPre p;
    const int tok = t;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        p.sl[k] = (t < a.B && tok < a.b_live) ? a.tok_slot[size_t(tok) * kTopK + k] : -1;
        p.wk[k] = p.sl[k] >= 0 ? a.topk_w[size_t(tok) * kTopK + k] : 0.f;
    }
    return p;
}
// The row's readiness is its own elements (gather 2): every chunk combine_rows reads (this lane's 8 columns
// of each live slot, two 8 B lamport elements) polled until none is the sentinel; the observed chunks are the row's
// operands, and the next MoE pass's entry restores them (lam_reset_zb). rank: this rank (the
// report's).
__device__ __forceinline__ int combine_col0(const CombArgs& a) {
    return int(a.pair_rank) * (kCombNb / kCombPair) * 128 + (int(threadIdx.x) - 128) * 8;
}
__device__ __forceinline__ void zb_load(const __nv_bfloat16* p, uint4& v) {
    uint64_t lo, hi;
    ptx::ld_relaxed_sys_v2b64(p, lo, hi);
    v = make_uint4(uint32_t(lo), uint32_t(lo >> 32), uint32_t(hi), uint32_t(hi >> 32));
}
__device__ __forceinline__ bool zb_empty(const uint4& v) {
    return v.x == uint32_t(kZbSent) || v.z == uint32_t(kZbSent);  // each element's low word
}
__device__ __forceinline__ void combine_load(const CombArgs& a, const CombPre* pre, uint4 (&v)[kTopK]) {
    const int c0 = combine_col0(a);
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
        if (pre->sl[k] >= 0) zb_load(a.zb + size_t(pre->sl[k]) * kH + c0, v[k]);
}
__device__ __forceinline__ void combine_wait(const CombArgs& a, const CombPre* pre, uint4 (&v)[kTopK], int rank) {
    const auto missing = [&] {
        bool e = false;
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
            e |= pre->sl[k] >= 0 && zb_empty(v[k]);
        return e;
    };
    combine_load(a, pre, v);
    lam_spin(
        [&] {
            if (!missing()) return true;
            combine_load(a, pre, v);
            return false;
        },
        "combine zb lamport", rank, int(a.tok0), uint32_t(combine_col0(a)),
        [&] {
            int k = 0;
            while (k < kTopK - 1 && !(pre->sl[k] >= 0 && zb_empty(v[k])))
                ++k;
            return uint32_t(pre->sl[k]);  // the first slot still empty
        });
}
// the row's partials in the caller's scratch (ext_red: [kCombNb][kTopK] then [kTopK]); zv: the row's chunks
// (combine_wait's, the dry pass's combine_load); dry: the row's arithmetic without the exchange and the y store
template <int kGroups>
__device__ __forceinline__ void combine_rows(const CombArgs a, const uint32_t bar_idx, const CombPre* pre,
                                             float* ext_red, const bool dry, const uint4 (&zv)[kTopK]) {
    constexpr int kPair = kCombPair;
    static_assert(kCombNb % (kGroups * kPair) == 0, "the n-blocks must divide over the lane groups");
    constexpr int kThreads = 16 * kGroups;
    constexpr int kNbPer = kCombNb / (kGroups * kPair);
    static_assert(kNbPer == 1, "a lane reads one chunk per slot (combine_wait's)");
    // the pair: rank r owns n-blocks [r * kNbSplit, (r + 1) * kNbSplit) of the row's s_red slots
    constexpr int kNbSplit = kCombNb / kPair;
    const int nb0 = int(a.pair_rank) * kNbSplit;
    float (*const s_red)[kTopK] = reinterpret_cast<float (*)[kTopK]>(ext_red);
    float* const s_sc = ext_red + kCombNb * kTopK;
    // the item's lanes are warps 4..11 of the hosting CTA: item-local tid, its named barrier for the CTA-wide syncs
    const int tid = int(threadIdx.x) - 128;
    const int c0 = nb0 * 128 + tid * 8;  // this lane's 8 consecutive cols of its first n-block
    // The lane's strips sit one whole block-round apart, so lane group tid>>4
    // holds n-blocks (tid>>4) + nb * kGroups.
    constexpr int kRound = kThreads * 8;
    float gp1[kNbPer][8];
#pragma unroll
    for (int nb = 0; nb < kNbPer; ++nb) {
        const uint4 gv = *reinterpret_cast<const uint4*>(a.gamma_p1 + c0 + nb * kRound);
        const __nv_bfloat16* g8 = reinterpret_cast<const __nv_bfloat16*>(&gv);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            gp1[nb][j] = __bfloat162float(g8[j]);
    }

    // the hosted row takes the caller's token (its block id is a claimed one). The loop runs once (tstep = 144 / 2 > B,
    // the item traps B > kRows) on tok0's pre / zv; kept for the schedule (README.md)
    const int t0 = int(a.tok0), tstep = int(gridDim.x) / kPair;
    for (int t = t0; t < a.B; t += tstep) {
        const int tok = t;
        int sl[kTopK];
        float wk[kTopK];
        // slot ids are block-uniform, so every branch below is too
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            sl[k] = pre->sl[k];
            wk[k] = pre->wk[k];
        }
        for (int nb = 0; nb < kNbPer; ++nb) {
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const __nv_bfloat16* z8 = reinterpret_cast<const __nv_bfloat16*>(&zv[k]);
                float ss = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = __fmul_rn(__bfloat162float(z8[j]), gp1[nb][j]);
                    ss = __fmaf_rn(v, v, ss);
                }
#pragma unroll
                for (int d = 8; d > 0; d >>= 1)
                    ss = __fadd_rn(ss, __shfl_xor_sync(0xFFFFFFFFu, ss, d));
                if ((tid & 15) == 0) s_red[nb0 + (tid >> 4) + nb * kGroups][k] = ss;
            }
        }
        ptx::sync_aligned(bar_idx, kThreads);
        // the level-B reduction needs all kCombNb partials: this rank's kNbSplit x kTopK fp32 go to the peer's s_red
        // rows (st.async on the peer's exchange mbarrier) and the peer's land in ours; every lane then reduces the
        // same 32 words in the same order — rstd bitwise the single-CTA row's
        constexpr uint32_t kXchgBytes = kNbSplit * kTopK * sizeof(float);
        static_assert(kXchgBytes % 16 == 0 && kXchgBytes / 16 <= kThreads, "the exchange is one uint4 per lane");
        const uint32_t peer = a.pair_peer;
        if (!dry && tid == 0) ptx::mbar_arrive_expect_tx(a.xchg_bar, kXchgBytes);
        if (!dry && tid < int(kXchgBytes / 16)) {
            const uint4 v = *reinterpret_cast<const uint4*>(&s_red[nb0][0] + tid * 4);
            ptx::st_async_cluster_v4(ptx::mapa_shared_cluster(ptx::to_shared(&s_red[nb0][0]) + tid * 16, peer), v,
                                     ptx::mapa_shared_cluster(ptx::to_shared(a.xchg_bar), peer));
        }
        // one token per cluster (the item traps on B > kRows): the barrier's first phase
        if (!dry) ptx::mbar_wait_parity_acquire_cluster_poll(a.xchg_bar, 0u);
        // the level-B reduction and rstd are block-uniform (every lane reduces the same shared words in the
        // same order), so they run once on kTopK lanes and are broadcast
        if (tid < kTopK) s_sc[tid] = rstd_pinned(level_b_reduce(&s_red[0][tid], kTopK), a.eps);
        ptx::sync_aligned(bar_idx, kThreads);
        float sc[kTopK];
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
            sc[k] = sl[k] >= 0 ? __fmul_rn(wk[k], s_sc[k]) : 0.f;
        for (int nb = 0; nb < kNbPer; ++nb) {
            float acc[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] = 0.f;
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                if (sl[k] < 0) continue;
                const __nv_bfloat16* z8 = reinterpret_cast<const __nv_bfloat16*>(&zv[k]);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[j] = __fmaf_rn(sc[k], __fmul_rn(__bfloat162float(z8[j]), gp1[nb][j]), acc[j]);
            }
            uint4 out;
            __nv_bfloat16* o8 = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                o8[j] = __float2bfloat16_rn(acc[j]);
            if (!dry) *reinterpret_cast<uint4*>(a.y + size_t(tok) * kH + c0 + nb * kRound) = out;
        }
        ptx::sync_aligned(bar_idx, kThreads);  // s_red reuse across tokens
    }
}

}  // namespace moe_tp

// The expert-weight L2 prefetch: one warp per CTA (the post-attention aux warp, otherwise idle) reads this layer's
// routing ids from the PREVIOUS step (the binding's last top-k, read once with acquire; nothing of the predecessor
// grid), histograms the experts and L2-prefetches the shard packs (W1, W2 and their scale factors) of the most-routed
// experts that fit the L2 budget, spread over every CTA of the launch, under the attention phase, long before this
// step's router runs. Consecutive steps route alike, so the MoE phase's g1 / g2 weight streams mostly hit L2.

namespace mk::expert_prefetch {

inline constexpr uint32_t kExperts = cfg::kExperts, kTopK = cfg::kTopK;
inline constexpr uint32_t kStride = 32;
// The per-expert region bytes of the rank's shard pack (the expert-major layouts above: NVFP4, or the bf16
// variant's packs without scale-factor regions); scalars because device code cannot index a constexpr array.
inline constexpr uint32_t kW1PackBytes = 2 * moe_tp::kIpr * moe_tp::elem_bytes(moe_tp::kH);
inline constexpr uint32_t kW1SfBytes = moe_tp::kBf16 ? 0 : 2 * moe_tp::kIpr * (moe_tp::kH / moe_tp::kSfVec);
inline constexpr uint32_t kW2PackBytes = moe_tp::kHpr * moe_tp::elem_bytes(moe_tp::kI);
inline constexpr uint32_t kW2SfBytes = moe_tp::kBf16 ? 0 : moe_tp::kHpr * (moe_tp::kI / moe_tp::kSfVec);
inline constexpr uint32_t kWeightBytes[4] = {kW1PackBytes, kW1SfBytes, kW2PackBytes, kW2SfBytes};
inline constexpr uint32_t kExpertBytes = kW1PackBytes + kW1SfBytes + kW2PackBytes + kW2SfBytes;
// The expert prefetch's count: as many of the most-routed experts as fit a 48 MiB L2 budget (14 at the TP8 NVFP4 shard,
// 3.375 MiB an expert; the bf16 variant's at every world), or 4 (cfg::kWorld.prefetch_wide off: the TP4 NVFP4 shard,
// 6.75 MiB an expert, where the wide budget's traffic costs the post-attention window more than it saves the MoE
// phase).
inline constexpr uint32_t kPrefetchBudget = 50331648;
inline constexpr uint32_t kSelected = cfg::kWorld.prefetch_wide || moe_tp::kBf16 ? kPrefetchBudget / kExpertBytes : 4;
static_assert(kSelected >= 1 && kSelected * kExpertBytes <= kPrefetchBudget);
// The expert prefetch strides each region by expert: the pack must be tail-free (one region per expert).
static_assert(size_t(moe_tp::kE) * kWeightBytes[0] == moe_tp::kW1Bytes &&
                  (moe_tp::kBf16 || size_t(moe_tp::kE) * kWeightBytes[1] == moe_tp::kW1SfBytes) &&
                  size_t(moe_tp::kE) * kWeightBytes[2] == moe_tp::kW2Bytes &&
                  (moe_tp::kBf16 || size_t(moe_tp::kE) * kWeightBytes[3] == moe_tp::kW2SfBytes),
              "the regions are the shard pack's per-expert strides");
static_assert(kExperts == uint32_t(moe_tp::kE) && kTopK == uint32_t(moe_tp::kTopK),
              "the expert prefetch's table is the MoE phase's");
static_assert(kWeightBytes[0] % kStride == 0 && kWeightBytes[1] % kStride == 0 && kWeightBytes[2] % kStride == 0 &&
                  kWeightBytes[3] % kStride == 0,
              "the expert prefetch walks each region in kStride steps");

__device__ __forceinline__ uint32_t weight_bytes(uint32_t region) {
    return region == 0 ? kW1PackBytes : region == 1 ? kW1SfBytes : region == 2 ? kW2PackBytes : kW2SfBytes;
}

struct Args {
    const int32_t* ids;
    const uint32_t* route_count;
    const uint8_t* weights[4];
};

// No early returns: inside a kernel that also carries tcgen05 mainloops, a return under a predicate ptxas cannot prove
// warp-uniform costs every such loop its uniform-datapath schedule (more instructions per stage), even on other
// warps' paths. One warp-uniform predicate carries the expert prefetch instead. kRows = the window's
// tokens (the route count advances by their power of two per launch); ties go to the lower expert id.
template <uint32_t kGrid, uint32_t kRows>
__device__ __forceinline__ void run_uniform(const Args& a, uint32_t* histogram, uint32_t* hot, uint32_t cta,
                                            uint32_t lane) {
    bool ok = a.ids != nullptr;
    uint32_t completed = 0;
    if (ok && lane == 0) completed = ptx::ld_acq_fenced(a.route_count);
    completed = __shfl_sync(0xffffffffu, completed, 0);
    ok = ok && completed != 0 && completed % handoff::pow2_at_least(kRows) == 0;
    if (ok) {
        for (uint32_t e = lane; e < kExperts; e += 32)
            histogram[e] = 0;
        __syncwarp();
        for (uint32_t row = 0; row < kRows * kTopK; row += 32) {
            const int32_t id = a.ids[row + lane];
            ok = ok && __all_sync(0xffffffffu, id >= 0 && id < int32_t(kExperts));
            if (ok) atomicAdd(histogram + id, 1u);
        }
        __syncwarp();
    }
    uint32_t picked = 0;  // bit j: expert lane + 32 j
    if (ok) {
        uint32_t counts[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j)
            counts[j] = histogram[lane + 32 * j];
        bool more = true;
        for (uint32_t pick = 0; pick < kSelected; ++pick) {
            uint32_t best = 0, expert = kExperts;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t id = lane + 32 * j;
                if (counts[j] > best || (counts[j] == best && id < expert)) {
                    best = counts[j];
                    expert = id;
                }
            }
#pragma unroll
            for (uint32_t d = 16; d; d >>= 1) {
                const uint32_t other_count = __shfl_xor_sync(0xffffffffu, best, d);
                const uint32_t other_id = __shfl_xor_sync(0xffffffffu, expert, d);
                if (other_count > best || (other_count == best && other_id < expert)) {
                    best = other_count;
                    expert = other_id;
                }
            }
            more = more && best != 0;
            if (more) {
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j)
                    if (lane + 32 * j == expert) {
                        counts[j] = 0;
                        picked |= 1u << j;
                    }
                for (uint32_t region = 0; region < 4; ++region) {
                    const uint32_t bytes = weight_bytes(region);
                    const uint8_t* const base = a.weights[region] + size_t(expert) * bytes;
                    for (uint32_t offset = (cta * 32 + lane) * kStride; offset < bytes;
                         offset += kGrid * 32 * kStride) {
                        ptx::prefetch_global_l2(base + offset);
                    }
                }
            }
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t word = __ballot_sync(0xffffffffu, (picked >> j) & 1u);
        if (lane == j) hot[j] = word;
    }
}

}  // namespace mk::expert_prefetch
