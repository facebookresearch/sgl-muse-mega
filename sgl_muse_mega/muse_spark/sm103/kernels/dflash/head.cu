// The draft step's entry and its draft head, one image per build: -DMK_IMAGE=entry (with -DMK_DFLASH_B=8|16) or
// -DMK_IMAGE=draft_head instantiates that image's kernel (below; with -DMK_DFLASH_TREE=1 the head grows the token
// tree, dflash/tree.cuh; with -DMK_DFLASH_SAMPLE=1 it walks the lattice by sampling, dflash/sample.cuh); the host
// libraries include this file under MK_HOST for the argument structs.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "dflash/common.cuh"
#if MK_DFLASH_TREE
#include "dflash/tree.cuh"
#endif
#if MK_DFLASH_SAMPLE || defined(MK_HOST)
#include "dflash/sample.cuh"
#endif

// draft_entry<B>: the DFlash draft step's entry, one launch of 144 CTAs x 256 threads (one CTA per SM). B = the
// draft block (8 or 16 rows). Every output is computed identically on every TP rank from identical bytes, so it is
// bitwise equal across ranks.
//
//   fc      ctx_raw = bf16(bf16(cap) fc^T)   the 5 captured target features of the window rows (fp32 residuals) through
//           the fc projection [8192, 5 x 8192]. Each rank owns 8192 / R output columns: 128-row W blocks x K-slices on
//           CTAs 0-127 (TP8 8 blocks x 16 slices of 2560, TP4 16 x 8 of 5120). Each CTA writes its fp32 partial and
//           counts itself on its block; once all its slices are in, slice CTA s reduces the block's W rows
//           [128 s / split, + 128 / split) in slice order (every slice's load in flight per output) and multicasts
//           them as 16-B packets of each window row into every rank's ctx_raw.
//   conv_proj0 delta = bf16(h0 Wk^T)         layer 0's attention-conv coefficient projection of the anchor row. Rows
//           >= 1 of the block are the mask token, a bind constant, so only row 0 runs per step: each rank owns 2048 / R
//           of the coefficients (blocks x K-slices on CTAs 128-143: TP8 2 x 8, TP4 4 x 4), reduced and multicast the
//           same way into the coefficient row.
//   rows    the 16 conv_proj0 CTAs then poll the gathered rows and run the row work: hidden_norm of the ctx rows, the
//           conv prepare of block rows 0-1, the coefficient rows.
//
// The gather needs no fence and no arrival count: every gathered 16-B packet starts as 0xFFFF (a bf16 NaN, never a
// result), a multicast store replaces it whole, and its ONE reader per rank spins on it until no lane is 0xFFFF, then
// writes 0xFFFF back for the next step. A peer's next push into a packet comes after the verify's all-reduces, which
// need this rank's progress past this launch, so the re-mark always lands first.
//
// Bind constants (rows of the mask token, filled by two bind launches with anchor = mask): e_mask = E[mask],
// h_mask = rms(E[mask]), d_mask = its 2048 coefficients, xq_const = the prepared row of two mask rows. The step
// copies them into rows >= 1 (>= 2 for xq) of its outputs, so its outputs are complete whoever rewrites those
// buffers between steps.
//
// The GEMM: swap-AB warp mma.sync (m16n8k16, bf16 in, fp32 accumulate): W rows are M (warp w owns rows 32w..32w+31
// of the 128-row block, two m16 tiles sharing each B fragment), the window rows are N. Bandwidth bound: per stage one
// 1D bulk copy of a 128 x 128 W tile, pre-packed with its 16-B chunks XOR-swizzled by row (ldmatrix reads 8 rows of
// one chunk conflict-free), and B bulk copies of the fp32 feature rows (L2-resident: 8 blocks read each); the MMA
// warps round x to bf16 while forming their B fragments.
//
// PDL: the fc CTAs read only weights and the captured features (written >= 3 launches back by the verify's layers),
// and their pushes reach a peer only after the peer finished its previous entry launch (the gather argument above), so
// they never wait for the predecessor. The conv_proj0 CTAs read the anchor (the predecessor's output) and write every
// local output, after griddepcontrol.wait.

namespace mk::dflash::entry {

template <uint32_t kRows>
struct Shape {
    static_assert(kRows == 8 || kRows == 16, "the draft block");
    static constexpr uint32_t kHidden = cfg::kHidden;
    static constexpr uint32_t kFeatures = 5;
    static constexpr uint32_t kK = kFeatures * kHidden;
    static constexpr uint32_t kRanks = cfg::kRanks;
    static constexpr uint32_t kTaps = 2, kGroups = 512;       // conv: 16 channels per coefficient group
    static constexpr uint32_t kCoeffs = 2 * kTaps * kGroups;  // (side, tap, group) per row
    static constexpr uint32_t kFcN = kHidden / kRanks;        // fc outputs per rank
    static constexpr uint32_t kConvProjN = kCoeffs / kRanks;  // conv_proj0 outputs per rank
    static constexpr uint32_t kTileN = 128, kStageK = 128;    // one W stage: 128 rows x 128 K bf16 = 32 KB
    // the K-slices: the fc on 128 CTAs, conv_proj0 on 16 (the row phase's CTAs), whatever the rank's share of blocks
    static constexpr uint32_t kFcBlocks = kFcN / kTileN, kFcSplit = 128 / kFcBlocks;
    static constexpr uint32_t kConvProjBlocks = kConvProjN / kTileN, kConvProjSplit = 16 / kConvProjBlocks;
    // W rows per slice CTA's reduction, and their 16-B packets per window row
    static constexpr uint32_t kFcReduce = kTileN / kFcSplit, kFcPackets = kFcReduce / 8;
    static constexpr uint32_t kConvProjReduce = kTileN / kConvProjSplit, kConvProjPackets = kConvProjReduce / 8;
    static constexpr uint32_t kFcStagesTotal = kK / kStageK, kFcStages = kFcStagesTotal / kFcSplit;
    static constexpr uint32_t kConvProjStagesTotal = kHidden / kStageK,
                              kConvProjStages = kConvProjStagesTotal / kConvProjSplit;
    static constexpr uint32_t kFcCtas = kFcBlocks * kFcSplit, kConvProjCtas = kConvProjBlocks * kConvProjSplit;
    static constexpr uint32_t kGrid = kFcCtas + kConvProjCtas;
    static constexpr uint32_t kThreads = 256;
    static constexpr uint32_t kMmaWarps = 4, kMTiles = kTileN / (16 * kMmaWarps), kProducerWarp = kMmaWarps;
    static constexpr uint32_t kN8 = kRows / 8;
    static constexpr uint32_t kRowChunks = kHidden / 8;  // 16-B chunks per row: 4 per thread
    // the row phase: conv_proj0 CTA t < kRows norms ctx row t; CTA (kRows + p) % 16 prepares block row p < 2
    static_assert(kConvProjCtas == 16 && kRows <= kConvProjCtas, "a conv_proj0 CTA per row-phase item");
    static_assert(kFcStagesTotal % kFcSplit == 0 && kConvProjStagesTotal % kConvProjSplit == 0, "even K-slices");
    static_assert(kHidden % kStageK == 0, "a stage never straddles two captured features");
    static_assert(kRowChunks == 4 * kThreads, "row ops: 4 chunks per thread");

    // smem: kStages W tiles, kStages x tiles, h0 row, the bf16 out tile, mbarriers, row-sum scratch
    static constexpr uint32_t kWStage = kTileN * kStageK * 2;
    static constexpr uint32_t kXRowF = kStageK + 8;  // fp32 row stride: 8-B fragment loads hit 32 banks per half warp
    static constexpr uint32_t kXStage = kRows * kXRowF * 4;
    static constexpr uint32_t kFixed = kHidden * 2 + kRows * kTileN * 2 + 256;
    static constexpr uint32_t kStages = (cfg::kSmemCeil - kFixed) / (kWStage + kXStage + 16);  // no static smem
    static constexpr uint32_t kXOff = kStages * kWStage;
    static constexpr uint32_t kH0Off = kXOff + kStages * kXStage;
    static constexpr uint32_t kOutOff = kH0Off + kHidden * 2;
    static constexpr uint32_t kBarOff = kOutOff + kRows * kTileN * 2;
    static constexpr uint32_t kMiscOff = kBarOff + 2 * kStages * 8;
    static constexpr uint32_t kSmem = kMiscOff + 256;
    static_assert(kSmem <= cfg::kSmemCeil && kStages >= 4, "ring depth");

    // the symmetric region: ctx_raw [kRows, hidden] bf16 | coefficient row [kCoeffs] bf16, all 0xFFFF between uses
    static constexpr uint32_t kSymCtx = 0;
    static constexpr uint32_t kSymConvProj = kRows * kHidden * 2;
    static constexpr uint32_t kSymBytes = kSymConvProj + kCoeffs * 2;

    // fp32 partials: fc [block][slice][m][n8][warp][lane] float4, then conv_proj0 [block][slice][m][warp][lane] float4
    static constexpr uint32_t kWsFc = kFcBlocks * kFcSplit * kTileN * kRows;
    static constexpr uint32_t kWsFloats = kWsFc + kConvProjBlocks * kConvProjSplit * kTileN * 8;

    // u32 per-block slice counts (fc, then conv_proj0): zeroed once, +split per launch
    static constexpr uint32_t kCtrKap = kFcBlocks, kCtrWords = kFcBlocks + kConvProjBlocks;
};

struct Args {
    const float* cap[5];    // [>= rows, hidden] fp32 each, in capture order
    const int64_t* anchor;  // a': the block's row 0 token
    const uint4* embed[8];  // every rank's vocab shard [shard_rows, hidden] bf16 (the target's embedding), kRanks used
    int64_t shard_rows;
    const nv_bfloat16* w_fc;         // pack_fc: [kFcBlocks][320 stages][128][128] swizzled, this rank's kFcN rows
    const nv_bfloat16* w_conv_proj;  // pack_conv_proj: [kConvProjBlocks][64][128][128], this rank's rows
    const nv_bfloat16* in_norm;      // layer 0 input_layernorm [hidden]
    const nv_bfloat16* hid_norm;     // hidden_norm [hidden]
    const nv_bfloat16* conv_base;    // layer 0 attention_conv side 0 base [taps, hidden]
    const nv_bfloat16* e_mask;       // bind constants (header)
    const nv_bfloat16* h_mask;
    const nv_bfloat16* d_mask;
    const nv_bfloat16* xq_const;
    uint8_t* sym;     // this rank's symmetric region
    uint64_t sym_mc;  // its multicast address
    uint32_t rank;
    nv_bfloat16* ctx_n;   // [rows, hidden] hidden_norm(fc(F)): the ctx rows of every layer's K / V
    nv_bfloat16* xq;      // [rows, hidden] layer 0's prepared block rows (its qkv x)
    nv_bfloat16* resid;   // [rows, hidden] layer 0's residual (the block's embedding rows)
    nv_bfloat16* c_att1;  // [rows, taps, groups] layer 0's attention-conv side-1 coefficients
    nv_bfloat16* h_row;   // [hidden] rms(E[a']) (the bind reads it)
    nv_bfloat16* delta;   // [coeffs] the anchor row's gathered coefficients (the bind reads it)
    float* ws;
    uint32_t* ctr;
    float eps;
};

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// a gathered packet still holds the 0xFFFF fill in some lane
__device__ __forceinline__ bool unlanded(const uint4& v) {
    return (__vcmpeq2(v.x, 0xFFFFFFFFu) | __vcmpeq2(v.y, 0xFFFFFFFFu) | __vcmpeq2(v.z, 0xFFFFFFFFu) |
            __vcmpeq2(v.w, 0xFFFFFFFFu)) != 0;
}

// The single reader's take of a gathered packet: spin until it landed, re-mark it empty, return it.
__device__ __forceinline__ uint4 take_packet(uint4* p) {
    uint4 v;
    bounded_spin(
        [&] {
            v = ptx::ld_relaxed_sys_u4(p);
            return !unlanded(v);
        },
        [&] { spin_report("entry gather packet", reinterpret_cast<uint64_t>(p), 0); });
    *p = make_uint4(~0u, ~0u, ~0u, ~0u);
    return v;
}

// The row sum of one 8192-wide row over 256 threads: a serial sum per thread, warp butterfly, the 8 warp partials
// summed in order by every thread (one value per row, no broadcast round trip).
__device__ __forceinline__ float row_sum_256(float acc, float* scratch) {
#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    __syncthreads();  // the previous use of the scratch is read
    if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = acc;
    __syncthreads();
    float sum = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < 8; ++w)
        sum += scratch[w];
    return sum;
}

// rsqrt(mean(x^2) + eps) of a row held as 4 x 16-B bf16 chunks per thread (chunk i at element (tid + 256 i) x 8)
__device__ __forceinline__ float row_rstd(const uint4 (&v)[4], float eps, float* scratch) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const auto e = reinterpret_cast<const nv_bfloat162*>(&v[i]);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(e[j]);
            acc = fmaf(f.x, f.x, acc);
            acc = fmaf(f.y, f.y, acc);
        }
    }
    return rsqrtf(row_sum_256(acc, scratch) / 8192.0f + eps);
}

// bf16((x * rstd) * w), RMSNorm's operation order, for one 16-B chunk
__device__ __forceinline__ uint4 norm_chunk(const uint4& x, const uint4& w, float rstd) {
    const auto xb = reinterpret_cast<const nv_bfloat162*>(&x);
    const auto wb = reinterpret_cast<const nv_bfloat162*>(&w);
    uint4 o;
    auto ob = reinterpret_cast<nv_bfloat162*>(&o);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(xb[j]);
        const float2 g = __bfloat1622float2(wb[j]);
        ob[j] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(f.x, rstd), g.x), __fmul_rn(__fmul_rn(f.y, rstd), g.y));
    }
    return o;
}

// One output of a block: its kSplit slice partials (fp32, `stride` apart) summed in slice order, every load in flight.
template <uint32_t kSplit>
__device__ __forceinline__ nv_bfloat16 reduce_slices(const float* p, uint32_t stride) {
    float v[kSplit];
#pragma unroll
    for (uint32_t s = 0; s < kSplit; ++s)
        v[s] = __ldcg(p + s * stride);
    float acc = v[0];
#pragma unroll
    for (uint32_t s = 1; s < kSplit; ++s)
        acc += v[s];
    return __float2bfloat16_rn(acc);
}

}  // namespace mk::dflash::entry

namespace mk::dflash {

template <uint32_t kRows>
__global__ __launch_bounds__(256, 1) void draft_entry(const __grid_constant__ entry::Args a) {
    using namespace entry;
    using S = Shape<kRows>;
    constexpr uint32_t kTile = S::kTileN * S::kStageK;
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const bool conv_cta = blockIdx.x >= S::kFcCtas;
    const uint32_t job = conv_cta ? blockIdx.x - S::kFcCtas : blockIdx.x;
    const uint32_t blk = conv_cta ? job % S::kConvProjBlocks : job % S::kFcBlocks;
    const uint32_t slice = conv_cta ? job / S::kConvProjBlocks : job / S::kFcBlocks;
    const uint32_t stages = conv_cta ? S::kConvProjStages : S::kFcStages;
    const nv_bfloat16* const w_tiles =
        conv_cta ? a.w_conv_proj + (size_t(blk) * S::kConvProjStagesTotal + slice * S::kConvProjStages) * kTile
                 : a.w_fc + (size_t(blk) * S::kFcStagesTotal + slice * S::kFcStages) * kTile;
    const uint32_t k0 = slice * stages * S::kStageK;  // the slice's first K column

    uint64_t* const full = reinterpret_cast<uint64_t*>(smem + S::kBarOff);
    uint64_t* const empty = full + S::kStages;
    auto* const h0 = reinterpret_cast<nv_bfloat16*>(smem + S::kH0Off);
    auto* const out = reinterpret_cast<nv_bfloat16*>(smem + S::kOutOff);
    auto* const misc = reinterpret_cast<uint32_t*>(smem + S::kMiscOff);  // [8..15] row sums
    auto* const scratch = reinterpret_cast<float*>(misc + 8);
    const uint32_t smem_w = ptx::to_shared(smem);
    const uint32_t smem_x = smem_w + S::kXOff;

    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < S::kStages; ++i) {
            ptx::mbar_init(full + i, 1);
            ptx::mbar_init(empty + i, S::kMmaWarps);
        }
        ptx::fence_mbarrier_init_release_cluster();
    }
    __syncthreads();
    // the successors wait for this grid's completion before reading its outputs; let them launch onto freed SMs
    if (threadIdx.x == 0) ptx::griddepcontrol_launch_dependents();

    const uint32_t tx = S::kWStage + (conv_cta ? 0u : kRows * S::kStageK * 4u);
    const auto produce = [&](uint32_t from, uint32_t to) {
        for (uint32_t it = from; it < to; ++it) {
            const uint32_t slot = it % S::kStages;
            if (it >= S::kStages) ptx::mbar_wait_parity(empty + slot, ((it / S::kStages) + 1) & 1);
            if (lane == 0) ptx::mbar_arrive_expect_tx(full + slot, tx);
            __syncwarp();
            if (lane == 0)
                ptx::cp_async_bulk_load(smem_w + slot * S::kWStage, w_tiles + size_t(it) * kTile, S::kWStage,
                                        full + slot);
            if (!conv_cta && lane < kRows) {
                const uint32_t kk = k0 + it * S::kStageK;
                const float* src = a.cap[kk / S::kHidden] + size_t(lane) * S::kHidden + kk % S::kHidden;
                ptx::cp_async_bulk_load(smem_x + slot * S::kXStage + lane * S::kXRowF * 4, src, S::kStageK * 4,
                                        full + slot);
            }
        }
    };
    const uint32_t prologue = stages < S::kStages ? stages : S::kStages;
    if (warp == S::kProducerWarp) produce(0, prologue);  // empty ring: no waits, the W stream starts at launch

    // the row phase's weights, fetched ahead of every wait (immutable)
    const uint32_t prep =
        (job + S::kConvProjCtas - kRows % S::kConvProjCtas) % S::kConvProjCtas;  // job (kRows + p) % 16 -> p
    uint4 wrow[4];
    if (conv_cta) {
        const auto* w = reinterpret_cast<const uint4*>(job < kRows ? a.hid_norm : a.in_norm);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            wrow[i] = w[threadIdx.x + i * 256];
    }

    if (conv_cta) {
        // the anchor row: E[a'] from the owner's shard in place (the vocab shards are disjoint, so the embedding's
        // all-reduce is the owner's row), h0 = rms(E[a']) * in_norm
        uint4 wn[4];
        const auto* wi = reinterpret_cast<const uint4*>(a.in_norm);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            wn[i] = job < kRows ? wi[threadIdx.x + i * 256] : wrow[i];
        ptx::griddepcontrol_wait();
        const int64_t id = *a.anchor;
        ::ptx::device_assert(id >= 0 && id < a.shard_rows * S::kRanks);
        const int64_t owner = id / a.shard_rows;
        const uint4* src = a.embed[owner] + (id - owner * a.shard_rows) * S::kRowChunks;
        uint4 e[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            e[i] = src[threadIdx.x + i * 256];
        const float rstd = row_rstd(e, a.eps, scratch);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t c = threadIdx.x + i * 256;
            const uint4 h = norm_chunk(e[i], wn[i], rstd);
            reinterpret_cast<uint4*>(h0)[c] = h;
            if (job == 0) {
                reinterpret_cast<uint4*>(a.resid)[c] = e[i];
                reinterpret_cast<uint4*>(a.h_row)[c] = h;
            }
        }
        __syncthreads();  // h0 complete before the MMA warps form B fragments from it
    }

    if (warp == S::kProducerWarp) {
        produce(prologue, stages);
    } else if (warp < S::kMmaWarps) {
        constexpr uint32_t kN8 = S::kN8, kM = S::kMTiles;
        float acc[kM][kN8][4];
#pragma unroll
        for (uint32_t m = 0; m < kM; ++m)
#pragma unroll
            for (uint32_t j = 0; j < kN8; ++j)
                acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.0f;
        // ldmatrix x4 of m tile m: lanes 0-15 address rows 32w + 16m + (lane & 15) at chunk c, lanes 16-31 at c + 1
        const uint32_t g = lane / 4, q = lane % 4;
        for (uint32_t it = 0; it < stages; ++it) {
            const uint32_t slot = it % S::kStages;
            ptx::mbar_wait_parity(full + slot, (it / S::kStages) & 1);
            const uint32_t wt = smem_w + slot * S::kWStage;
            const float* xs = reinterpret_cast<const float*>(smem + S::kXOff + slot * S::kXStage);
#pragma unroll
            for (uint32_t ks = 0; ks < S::kStageK / 16; ++ks) {
                uint32_t bf[kN8][2];
                if (conv_cta) {
                    // x = [h0; 0 x 7]: only lanes 0-3 (column 0 = row 0) carry values
                    const auto* hp = reinterpret_cast<const uint32_t*>(h0 + k0 + it * S::kStageK + 16 * ks) + q;
                    bf[0][0] = g == 0 ? hp[0] : 0u;
                    bf[0][1] = g == 0 ? hp[4] : 0u;
                } else {
#pragma unroll
                    for (uint32_t j = 0; j < kN8; ++j) {
                        const float* xr = xs + (8 * j + g) * S::kXRowF + 16 * ks + 2 * q;
                        const float2 lo = *reinterpret_cast<const float2*>(xr);
                        const float2 hi = *reinterpret_cast<const float2*>(xr + 8);
                        bf[j][0] = pack_bf16x2(lo.x, lo.y);
                        bf[j][1] = pack_bf16x2(hi.x, hi.y);
                    }
                }
                const uint32_t chunk = 2 * ks + (lane >> 4);
#pragma unroll
                for (uint32_t m = 0; m < kM; ++m) {
                    const uint32_t row = 32 * warp + 16 * m + (lane & 15);
                    uint32_t af[4];
                    ::ptx::ldmatrix_x4(af, wt + row * (S::kStageK * 2) + ((chunk ^ (row & 7)) << 4));
                    if (conv_cta) {
                        ::ptx::mma_m16n8k16_bf16(acc[m][0][0], acc[m][0][1], acc[m][0][2], acc[m][0][3], af[0], af[1],
                                                 af[2], af[3], bf[0][0], bf[0][1]);
                    } else {
#pragma unroll
                        for (uint32_t j = 0; j < kN8; ++j)
                            ::ptx::mma_m16n8k16_bf16(acc[m][j][0], acc[m][j][1], acc[m][j][2], acc[m][j][3], af[0],
                                                     af[1], af[2], af[3], bf[j][0], bf[j][1]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) ptx::mbar_arrive(empty + slot);
        }

        // the K-slice partial [window row][W row]; once the block's slices are all in (a monotonic count, +split per
        // launch), slice CTA s reduces W rows [s R, (s + 1) R) (R = 128 / split) in slice order and pushes them
        constexpr uint32_t kMmaThreads = S::kMmaWarps * 32;
        const uint32_t split = conv_cta ? S::kConvProjSplit : S::kFcSplit;
        const uint32_t prow = conv_cta ? 8u : kRows;  // the partial's window rows
        float* const part = a.ws + (conv_cta ? S::kWsFc : 0u) + size_t(blk) * split * prow * S::kTileN;
        {
            float* const p = part + size_t(slice) * prow * S::kTileN;
#pragma unroll
            for (uint32_t m = 0; m < kM; ++m) {
#pragma unroll
                for (uint32_t j = 0; j < kN8; ++j) {
                    if (conv_cta && j > 0) break;
                    // D fragment: W rows r, r + 8; window rows n, n + 1
                    const uint32_t r = 32 * warp + 16 * m + g, n = 8 * j + 2 * q;
                    p[n * S::kTileN + r] = acc[m][j][0];
                    p[(n + 1) * S::kTileN + r] = acc[m][j][1];
                    p[n * S::kTileN + r + 8] = acc[m][j][2];
                    p[(n + 1) * S::kTileN + r + 8] = acc[m][j][3];
                }
            }
        }
        __threadfence();
        ::ptx::sync_aligned(1, kMmaThreads);
        if (threadIdx.x == 0) {
            uint32_t* const count = a.ctr + (conv_cta ? S::kCtrKap : 0u) + blk;
            const uint32_t target = (atomicAdd(count, 1u) / split + 1) * split;
            uint32_t seen = 0;  // wrap-safe: the count is u32, never reset
            bounded_spin([&] { return static_cast<int32_t>((seen = ptx::ld_acq_fenced(count)) - target) >= 0; },
                         [&] { spin_report("entry slice count", seen, target); });
        }
        ::ptx::sync_aligned(1, kMmaThreads);
        const uint32_t t = threadIdx.x;
        if (conv_cta) {
            constexpr uint32_t kR = S::kConvProjReduce;  // row 0's coefficients only
            if (t < kR) out[t] = reduce_slices<S::kConvProjSplit>(part + kR * slice + t, 8 * S::kTileN);
        } else {
            constexpr uint32_t kR = S::kFcReduce;
            const auto reduce_part = [&](const uint32_t u) {
                out[u] =
                    reduce_slices<S::kFcSplit>(part + (u / kR) * S::kTileN + kR * slice + u % kR, kRows * S::kTileN);
            };
            if constexpr (kR * kRows <= kMmaThreads) {
                if (t < kR * kRows) reduce_part(t);
            } else {
#pragma unroll
                for (uint32_t u = t; u < kR * kRows; u += kMmaThreads)
                    reduce_part(u);
            }
        }
        ::ptx::sync_aligned(1, kMmaThreads);
        // 16-B packets into every rank's replica: fc kFcPackets per window row, conv_proj0 kConvProjPackets of row 0
        if (t < (conv_cta ? S::kConvProjPackets : S::kFcPackets * kRows)) {
            const uint4 v = reinterpret_cast<const uint4*>(out)[t];
            const uint64_t off =
                conv_cta
                    ? S::kSymConvProj +
                          (uint64_t(a.rank) * S::kConvProjN + blk * S::kTileN + S::kConvProjReduce * slice + 8 * t) * 2
                    : S::kSymCtx + (uint64_t(t / S::kFcPackets) * S::kHidden + a.rank * S::kFcN + blk * S::kTileN +
                                    S::kFcReduce * slice + 8 * (t % S::kFcPackets)) *
                                       2;
            ptx::multimem_st_relaxed_sys_v4_bf16x2(reinterpret_cast<uint32_t*>(a.sym_mc + off), v);
        }
    }
    if (!conv_cta) return;

    // ---- the row phase (conv_proj0 CTAs) ----
    if (warp > S::kProducerWarp) {
        // the mask rows' bind constants into this step's rows (CTA t writes row t), on the warps the GEMM left idle
        if (job >= 1 && job < kRows) {
            const size_t row = size_t(job) * S::kRowChunks;
            const uint32_t t = threadIdx.x - (S::kProducerWarp + 1) * 32, n = S::kThreads - (S::kProducerWarp + 1) * 32;
            for (uint32_t c = t; c < S::kRowChunks; c += n) {
                reinterpret_cast<uint4*>(a.resid)[row + c] = reinterpret_cast<const uint4*>(a.e_mask)[c];
                if (job >= 2) reinterpret_cast<uint4*>(a.xq)[row + c] = reinterpret_cast<const uint4*>(a.xq_const)[c];
            }
            constexpr uint32_t kSide1 = S::kTaps * S::kGroups / 8;  // 16-B chunks of one row's side-1 coefficients
            for (uint32_t c = t; c < kSide1; c += n)
                reinterpret_cast<uint4*>(a.c_att1)[job * kSide1 + c] =
                    reinterpret_cast<const uint4*>(a.d_mask)[kSide1 + c];
        }
    }
    __syncthreads();

    auto* const conv_proj_row = reinterpret_cast<uint4*>(a.sym + S::kSymConvProj);
    if (job < kRows) {
        // hidden_norm of ctx row `job` as its packets land
        auto* src = reinterpret_cast<uint4*>(a.sym + S::kSymCtx) + size_t(job) * S::kRowChunks;
        uint4 x[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            x[i] = take_packet(src + threadIdx.x + i * 256);
        const float rstd = row_rstd(x, a.eps, scratch);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            reinterpret_cast<uint4*>(a.ctx_n)[size_t(job) * S::kRowChunks + threadIdx.x + i * 256] =
                norm_chunk(x[i], wrow[i], rstd);
    }
    // the conv prepare of block rows 0 and 1 (side 0, one fused fp32 expression, rounded once):
    //   x'[0] = (b0 + d[0, t0]) h0                  x'[1] = (b0 + dm[t0]) hm + (b1 + dm[t1]) h0
    // Their CTAs take the coefficient row's packets: prep 0 side 0 tap 0 and side 1, prep 1 side 0 tap 1 (read only for
    // the delta copy the bind uses), each copied out before the re-mark.
    if (prep < 2) {
        constexpr uint32_t kTapChunks = S::kGroups / 8;  // 64 packets per (side, tap)
        nv_bfloat16* const d0 = out;                     // side 0 tap 0, in the out tile (the slice reduction is done)
        if (threadIdx.x < (prep == 0 ? 3 : 1) * kTapChunks) {
            const uint32_t c = prep == 0 ? (threadIdx.x < kTapChunks ? threadIdx.x : threadIdx.x + kTapChunks)
                                         : threadIdx.x + kTapChunks;
            const uint4 v = take_packet(conv_proj_row + c);
            reinterpret_cast<uint4*>(a.delta)[c] = v;
            if (c < kTapChunks) reinterpret_cast<uint4*>(d0)[c] = v;
            if (c >= 2 * kTapChunks) reinterpret_cast<uint4*>(a.c_att1)[c - 2 * kTapChunks] = v;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t c = threadIdx.x + i * 256, grp = c / 2;  // 8 channels of one 16-channel group
            const uint4 hv = reinterpret_cast<const uint4*>(h0)[c];
            const uint4 b0v = reinterpret_cast<const uint4*>(a.conv_base)[c];
            const auto hb = reinterpret_cast<const nv_bfloat162*>(&hv);
            const auto bb0 = reinterpret_cast<const nv_bfloat162*>(&b0v);
            uint4 o;
            auto ob = reinterpret_cast<nv_bfloat162*>(&o);
            if (prep == 0) {
                const float d = __bfloat162float(d0[grp]);
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const float2 h = __bfloat1622float2(hb[j]), b = __bfloat1622float2(bb0[j]);
                    ob[j] = __floats2bfloat162_rn(__fmul_rn(__fadd_rn(b.x, d), h.x), __fmul_rn(__fadd_rn(b.y, d), h.y));
                }
            } else {
                const uint4 b1v = reinterpret_cast<const uint4*>(a.conv_base + S::kHidden)[c];
                const uint4 hmv = reinterpret_cast<const uint4*>(a.h_mask)[c];
                const auto bb1 = reinterpret_cast<const nv_bfloat162*>(&b1v);
                const auto hm = reinterpret_cast<const nv_bfloat162*>(&hmv);
                const float dm0 = __bfloat162float(a.d_mask[grp]), dm1 = __bfloat162float(a.d_mask[S::kGroups + grp]);
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const float2 h = __bfloat1622float2(hb[j]), m = __bfloat1622float2(hm[j]);
                    const float2 b = __bfloat1622float2(bb0[j]), b1 = __bfloat1622float2(bb1[j]);
                    ob[j] = __floats2bfloat162_rn(
                        __fadd_rn(__fmul_rn(__fadd_rn(b.x, dm0), m.x), __fmul_rn(__fadd_rn(b1.x, dm1), h.x)),
                        __fadd_rn(__fmul_rn(__fadd_rn(b.y, dm0), m.y), __fmul_rn(__fadd_rn(b1.y, dm1), h.y)));
                }
            }
            reinterpret_cast<uint4*>(a.xq)[size_t(prep) * S::kRowChunks + c] = o;
        }
    }
}

}  // namespace mk::dflash

// draft_head: the draft head over an e4m3 pre-score of the lm_head shard, with exact rescoring of what can reach
// the top-16.
//
// A bf16 head would stream the rank's bf16 vocab shard to score every vocab row of the block's rows 1 .. gamma. This
// head streams an e4m3 copy (one fp32 scale per vocab row: half the bytes) and recomputes in bf16 only the rows that
// can reach the top-16:
//   xq        y rows 1 .. gamma as e4m3 (one scale per row, amax / 448), written by the last draft layer's final norm
//             (dflash/layer.cu): every x producer reads them behind the grid wait (its W ring streams meanwhile)
//   prescore  the lm_head's persistent mainloop at kind::f8f6f4 (K 128 per stage, the same 8 KB stages); per [64 x 16]
//             tile the epilogue keys each score (accumulator x the vocab row's scale; xq's scale is per x row and
//             cannot reorder a row) into the CTA's per-row top-16 lists
//   rescore   the `group` CTAs of slot e take row e's survivors: every list entry at or above the ksurv-th largest list
//             head (>= ksurv entries: they hold the rank's pre-score top-ksurv), plus the rank's W_P rows; each exact
//             logit is bf16 of an fp32 dot over the bf16 shard row, one warp per row in a fixed order (lane l sums
//             the 16-B chunks l, l + 32, .. in sequence, then a xor butterfly 16, 8, 4, 2, 1). The group's last CTA
//             takes the rank's top-16 of the survivors' exact keys.
//   the rest  the tagged exchange of 16 candidate + the W_P y words per row to every rank, the global
//             top-16, the unary scores and slot e's lattice; each lattice CTA also stores its per-predecessor argmax,
//             so the last one's walk is gamma table lookups.
// The candidates are the exact top-16 (in this dot order) whenever every global candidate survives on its rank: in its
// CTA's pre-score top-16 and at or above that rank's ksurv-th list head.
//
// launch: grid = the lm_head's persistent grid over the fp8 shard's 64-row blocks, block 256, smem
// dense_persist_smem_layout(16, 4, 20, false).size, PDL (static smem: the sink's lists and score tile)

namespace mk::dflash::head {

inline constexpr uint32_t kRows = 16;                        // the persist tile's x rows (gamma <= 15 live)
inline constexpr uint32_t kBN = 64;                          // vocab rows per block
inline constexpr uint32_t kKStage = 128;                     // fp8 elements per stage (one 128-B swizzle row)
inline constexpr uint32_t kStageStride = 82;                 // 80 keys + 2: the warp's 4 rows on different banks
inline constexpr uint32_t kMaxSurvLists = 64;                // ksurv <= 64
inline constexpr uint32_t kMaxSurv = kMaxSurvLists * kTopK;  // survivors per row
inline constexpr uint32_t kMaxHidden = 8192;
// state words (u32, zeroed once): the launch epoch, CTA arrivals, lattices done, then one group count per slot
inline constexpr uint32_t kStEpoch = 0, kStCtas = 1, kStLattice = 2, kStGroup = 3;
inline constexpr uint32_t kStateWords = kStGroup + kRows;

// A pre-score's key: the fp32 order image over the score, the id below (a max over keys is the pre-score's order).
__device__ __forceinline__ uint64_t score_key(float s, uint32_t id) {
    const uint32_t b = __float_as_uint(s);
    const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<uint64_t>(o) << kIdBits) | ((1u << kIdBits) - 1u - id);
}

struct PrescoreTileSmem {
    uint64_t list[kRows][kTopK];
    uint64_t stage[kRows][kStageStride];
    float score[kRows][kBN];  // the epilogue's transpose: [x row][vocab row]
};

__device__ __forceinline__ PrescoreTileSmem& tile_smem() {
    __shared__ PrescoreTileSmem s;
    return s;
}

struct HeadArgs {
    const float* w_scale;   // [vocab_rank] the fp8 rows' scales
    uint32_t stages_total;  // hidden / 128
    uint32_t num_blocks;    // the fp8 shard's 64-row blocks (the schedule's)
    uint32_t vocab_rank;
    uint32_t id0;  // the shard's first vocab id
};

// The per-tile list update: thread (row t, packet p) holds 8 of the row's 64 scores. The row's 8 lanes stage the list
// and the keys above its 16th, then each staged key's rank (count of greater staged keys) places it.
__device__ __forceinline__ void tile_lists(const HeadArgs& a, const uint32_t w_outer, const uint32_t tid) {
    PrescoreTileSmem& s = tile_smem();
    const uint32_t t = tid >> 3, p = tid & 7;
    const float4 v0 = *reinterpret_cast<const float4*>(&s.score[t][p * 8]);
    const float4 v1 = *reinterpret_cast<const float4*>(&s.score[t][p * 8 + 4]);
    const float sv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    uint64_t k[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t col = w_outer + p * 8 + e;
        k[e] = col < a.vocab_rank ? score_key(sv[e], a.id0 + col) : 0;
    }
    const uint64_t thr = s.list[t][kTopK - 1];
    uint32_t mine = 0;
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        mine += k[e] > thr;
    uint32_t incl = mine;
#pragma unroll
    for (uint32_t o = 1; o < 8; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o, 8);
        if (p >= o) incl += y;
    }
    const uint32_t total = __shfl_sync(0xffffffffu, incl, 7, 8);
    uint64_t* st = s.stage[t];
    for (uint32_t j = p; j < kTopK; j += 8)
        st[j] = s.list[t][j];
    uint32_t w = kTopK + incl - mine;
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e)
        if (k[e] > thr) st[w++] = k[e];
    __syncwarp();
    const uint32_t n = kTopK + total;
    for (uint32_t i = p; i < n; i += 8) {
        const uint64_t key = st[i];
        uint32_t r = 0;
        for (uint32_t j = 0; j < n; ++j)
            r += st[j] > key;
        if (r < kTopK) s.list[t][r] = key;
    }
    __syncwarp();
}

// The persistent body (framework.cuh's dense_persist_body) over the e4m3 shard: the same roles, ring and
// accumulator handoff, kind::f8f6f4 MMAs, the x producer gated on xq, and the list sink in place of the bf16 tile.
template <uint32_t kStages, uint32_t kPack, bool kEvictFirst>
__device__ __forceinline__ void head_body(const HeadArgs& a, const CUtensorMap& tmap_w, const CUtensorMap& tmap_x) {
    constexpr uint32_t kM = kRows;
    constexpr uint32_t kStageBytes = kBN * 128;
    constexpr uint32_t kXTile = kM * 128;
    constexpr uint32_t kNumSlots = kStages / kPack;
    constexpr uint32_t kAtoms = 4;  // K 32 per kind::f8f6f4 MMA
    constexpr uint32_t kTmemCols = dense_tmem_cols(kM);
    constexpr uint32_t kEpiThreads = kDenseRoles.n_epi * 32;
    static_assert(kStages % kPack == 0 && kPack <= kStages / 2, "pack must divide the ring and be <= half of it");
    // kind::f8f6f4 instruction descriptor: the kind::f16 layout, A / B type E4M3 = 0, D F32, M 64, N kM
    constexpr uint32_t idesc = (1u << 4) | ((kM >> 3) << 17) | ((kBN >> 4) << 24);

    constexpr DenseSmemLayout kSmem = dense_persist_smem_layout(kM, kPack, kStages, false);
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    const uint32_t smem_w = ptx::to_shared(smem_buf);
    const uint32_t smem_x = smem_w + kSmem.x_off;
    auto tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_buf + kSmem.tmem_off);
    uint64_t* const bar_base = reinterpret_cast<uint64_t*>(smem_buf + kSmem.bar_off);
    uint64_t* const full_bars = bar_base;
    uint64_t* const empty_bars = bar_base + kNumSlots;
    uint64_t* const tmem_full_bar = bar_base + 2 * kNumSlots;
    uint64_t* const tmem_empty_bar = bar_base + 2 * kNumSlots + 1;

    const uint32_t warp_idx = ptx::warp_idx_sync();
    const uint32_t lane_idx = ptx::lane_idx();
    const uint32_t cta = blockIdx.x, ctas = gridDim.x, nb = a.num_blocks;
    const uint32_t umin_stages = a.stages_total < kStages ? a.stages_total : kStages;

    if (warp_idx == kDensePrefetch && ptx::elect_one_sync()) {
        for (uint32_t q = 0; q < umin_stages; ++q)
            dense_prefetch_w(&tmap_w, static_cast<int32_t>(q * kKStage), static_cast<int32_t>(cta * kBN));
    }
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

    // warp 1: x producer, behind the grid wait (xq is the predecessor's output)
    if (warp_idx == kDenseGate) {
        ptx::griddepcontrol_wait();
        if (ptx::elect_one_sync()) ptx::griddepcontrol_launch_dependents();
        uint32_t slot = 0, phase = 0;
        for (uint32_t b = cta; b < nb; b += ctas) {
            for (uint32_t s = 0; s < a.stages_total; s += kPack) {
                const uint32_t deliveries = a.stages_total - s < kPack ? a.stages_total - s : kPack;
                ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d)
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_x + (slot * kPack + d) * kXTile, &tmap_x,
                                                               static_cast<int32_t>((s + d) * kKStage), 0,
                                                               &full_bars[slot], ptx::kEvictNormal);
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // warp 0: W producer (never waits on the grid: the fp8 shard is a weight)
    if (warp_idx == kDenseWProducer) {
        uint32_t slot = 0, phase = 0;
        for (uint32_t b = cta; b < nb; b += ctas) {
            const uint32_t w_outer = b * kBN;
            for (uint32_t s = 0; s < a.stages_total; s += kPack) {
                const uint32_t deliveries = a.stages_total - s < kPack ? a.stages_total - s : kPack;
                ptx::mbar_wait_parity_poll(&empty_bars[slot], phase ^ 1);
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d)
                        ptx::cp_async_bulk_tensor_2d_load_hint(smem_w + (slot * kPack + d) * kStageBytes, &tmap_w,
                                                               static_cast<int32_t>((s + d) * kKStage),
                                                               static_cast<int32_t>(w_outer), &full_bars[slot],
                                                               kEvictFirst ? ptx::kEvictFirst : ptx::kEvictNormal);
                    ptx::mbar_arrive_expect_tx(&full_bars[slot], deliveries * (kStageBytes + kXTile));
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // warp 2: MMA (single accumulator; block j opens after the epilogue drained block j - 1)
    if (warp_idx == kDenseMma) {
        const uint64_t w_desc0 = ptx::mma_smem_desc_k_major<uint8_t, kKStage, 128>(smem_w);
        const uint64_t x_desc0 = ptx::mma_smem_desc_k_major<uint8_t, kKStage, 128>(smem_x);
        uint32_t slot = 0, phase = 0, j = 0;
        for (uint32_t b = cta; b < nb; b += ctas, ++j) {
            if (j >= 1) ptx::mbar_wait_parity_poll(tmem_empty_bar, (j - 1) & 1);
            ptx::tcgen05_fence_after_thread_sync();
            for (uint32_t s = 0; s < a.stages_total; s += kPack) {
                const uint32_t deliveries = a.stages_total - s < kPack ? a.stages_total - s : kPack;
                ptx::mbar_wait_parity_poll(&full_bars[slot], phase);
                ptx::tcgen05_fence_after_thread_sync();
                if (ptx::elect_one_sync()) {
                    for (uint32_t d = 0; d < deliveries; ++d) {
                        const uint32_t sl = slot * kPack + d;
#pragma unroll
                        for (uint32_t kk = 0; kk < kAtoms; ++kk) {
                            const uint64_t da = w_desc0 + sl * (kStageBytes / 16) + kk * 2;  // +32 B per atom
                            const uint64_t db = x_desc0 + sl * (kXTile / 16) + kk * 2;
                            const uint32_t scale_c = (s > 0 || d > 0 || kk > 0) ? 1u : 0u;
                            ::ptx::tcgen05_mma_f8f6f4(taddr, da, db, idesc, scale_c);
                        }
                    }
                    ptx::tcgen05_commit_arrive(&empty_bars[slot]);
                    if (s + kPack >= a.stages_total) ptx::tcgen05_commit_arrive(tmem_full_bar);
                }
                __syncwarp();
                slot = slot == kNumSlots - 1 ? 0 : slot + 1;
                phase ^= slot == 0;
            }
        }
    }

    // warps 4..7: lists, then per block: drain -> scores x the row scales -> transpose -> lists
    if (warp_idx >= kDenseEpi0) {
        const uint32_t quad = warp_idx - kDenseEpi0, tid = quad * 32 + lane_idx;
        PrescoreTileSmem& ts = tile_smem();
        reinterpret_cast<uint64_t*>(ts.list)[tid] = 0;
        reinterpret_cast<uint64_t*>(ts.list)[tid + 128] = 0;
        uint32_t j = 0;
        for (uint32_t b = cta; b < nb; b += ctas, ++j) {
            const uint32_t w_outer = b * kBN;
            const uint32_t row = quad * 16 + (lane_idx & 15);
            const float sw = w_outer + row < a.vocab_rank ? __ldg(a.w_scale + w_outer + row) : 0.f;
            ptx::mbar_wait_parity_poll(tmem_full_bar, j & 1);
            ptx::tcgen05_fence_after_thread_sync();
            uint32_t v[kM];
            ptx::tcgen05_ld_32x32b_x16(taddr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
                                       v[12], v[13], v[14], v[15]);
            ptx::tcgen05_wait_ld();
            ptx::sync_aligned(block::kEpi, kEpiThreads);
            if (b + ctas < nb && warp_idx == kDenseEpi0 && ptx::elect_one_sync()) ptx::mbar_arrive(tmem_empty_bar);
            if (lane_idx < 16) {
#pragma unroll
                for (uint32_t c = 0; c < kM; ++c)
                    ts.score[c][row] = __fmul_rn(__int_as_float(v[c]), sw);
            }
            ptx::sync_aligned(block::kEpi, kEpiThreads);
            tile_lists(a, w_outer, tid);
            ptx::sync_aligned(block::kEpi, kEpiThreads);  // the score tile is rewritten by the next block
        }
        if (warp_idx == kDenseEpi0) ptx::tcgen05_dealloc(taddr, kTmemCols);  // no relinquish
    }
}

struct PostArgs {
    uint64_t* inbox_peer[8];       // every rank's inbox [2 halves][ranks][16 rows][kInboxWords], this rank's at [rank]
    uint64_t* part;                // [grid][16 rows][16] per-CTA pre-score lists
    uint64_t* resc;                // [16 rows][kMaxSurv] the survivors' exact keys
    __nv_bfloat16* hloc;           // [16 rows][hproj] the rank's W_P y
    uint32_t* state;               // u32 [kStateWords]
    const __nv_bfloat16* wb;       // [vocab_rank + hproj][hidden] the bf16 shard, then the W_P band
    const __nv_bfloat16* y;        // [>= gamma + 1][hidden]
    const __nv_bfloat16* pred_cb;  // [vocab][256]
    const __nv_bfloat16* succ_cb;  // [vocab][256]
    const int64_t* words;          // the step words (the anchor)
    int32_t* cand;                 // [gamma][16] candidate ids, best first
    float* unary;                  // [gamma][16]
    __nv_bfloat16* hhat;           // [gamma][256] W_P y (every rank's slice)
    float* scores;                 // [gamma][16][16]
    int32_t* pick;                 // [gamma][16] slot e's argmax successor per predecessor
    int64_t* drafts;               // [gamma]
    int64_t* verify_ids;           // [16]
    uint32_t rank;
    uint32_t ranks;
    uint32_t gamma;
    uint32_t hproj;
    uint32_t vocab_rank;
    uint32_t hidden;
    uint32_t group;  // CTAs per slot (the rescoring group)
    uint32_t ksurv;  // the survivors cover the rank's pre-score top-ksurv
    float mult;
    float cap;
};

struct PostSmem {
    alignas(16) __nv_bfloat16 yrow[kMaxHidden];
    uint64_t surv[kMaxSurv];
    uint64_t head[kThreads];
    uint32_t wsum[kThreads / 32];
    uint64_t thr;
    uint32_t n;
    uint64_t mine[kTopK];
    uint64_t ke[8 * kTopK];
    uint64_t kp[8 * kTopK];
    uint64_t ce[kTopK], cp[kTopK];
    LatticeSmem lat;
    int32_t picks[kTopK * kTopK];
    int32_t cands[kTopK * kTopK];
    int64_t ids[kTopK];
    uint32_t epoch;
    bool last;
};

// One warp's exact dot of y (smem) with the bf16 row `w`: lane l sums the 16-B chunks l, l + 32, .. in order, then the
// xor butterfly; every lane returns the sum. (The products of two bf16 are exact in fp32, so each FMA rounds once.)
__device__ __forceinline__ float warp_dot(const __nv_bfloat16* w, const __nv_bfloat16* yrow, uint32_t hidden,
                                          uint32_t lane) {
    const uint4* w4 = reinterpret_cast<const uint4*>(w);
    const uint4* y4 = reinterpret_cast<const uint4*>(yrow);
    const uint32_t n = hidden / 8 / 32;  // chunks per lane
    float acc = 0.f;
    for (uint32_t i0 = 0; i0 < n; i0 += 16) {
        uint4 wv[16];
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i)
            if (i0 + i < n) wv[i] = __ldg(w4 + lane + 32 * (i0 + i));
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            if (i0 + i >= n) break;
            const uint4 yv = y4[lane + 32 * (i0 + i)];
            const uint32_t ww[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
            const uint32_t yy[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
                const __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&ww[q]);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&yy[q]);
                acc = __fmaf_rn(__low2float(a2), __low2float(b2), acc);
                acc = __fmaf_rn(__high2float(a2), __high2float(b2), acc);
            }
        }
    }
#pragma unroll
    for (uint32_t o = 16; o >= 1; o >>= 1)
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    return acc;
}

// Row e's survivors in a fixed order (list by list in CTA order, each list's entries >= the threshold in its order):
// every CTA of the group builds the same array. T = the ksurv-th largest head (0 below ksurv lists).
__device__ __forceinline__ void survivors(const PostArgs& a, uint32_t e, uint32_t ctas, PostSmem& sm) {
    const uint32_t t = threadIdx.x;
    const uint64_t* lists = a.part + e * kTopK;  // list c at lists + c x 16 x 16
    sm.head[t] = t < ctas ? __ldcg(lists + static_cast<uint64_t>(t) * kRows * kTopK) : 0;
    if (t == 0) sm.thr = 0;
    __syncthreads();
    if (t < ctas && ctas >= a.ksurv) {
        uint32_t r = 0;
        for (uint32_t j = 0; j < ctas; ++j)
            r += sm.head[j] > sm.head[t];
        if (r == a.ksurv - 1) sm.thr = sm.head[t];
    }
    __syncthreads();
    const uint64_t thr = sm.thr;
    uint64_t l[kTopK];
    uint32_t c = 0;
    if (t < ctas && sm.head[t] >= thr && sm.head[t] != 0) {
#pragma unroll
        for (uint32_t j = 0; j < kTopK; ++j)
            l[j] = __ldcg(lists + static_cast<uint64_t>(t) * kRows * kTopK + j);
#pragma unroll
        for (uint32_t j = 0; j < kTopK; ++j)
            c += l[j] >= thr && l[j] != 0;  // descending: a prefix
    }
    // exclusive scan of c over threads (CTA order)
    uint32_t incl = c;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
        if ((t & 31) >= o) incl += y;
    }
    if ((t & 31) == 31) sm.wsum[t / 32] = incl;
    __syncthreads();
    uint32_t base = 0;
    for (uint32_t w = 0; w < t / 32; ++w)
        base += sm.wsum[w];
    const uint32_t off = base + incl - c;
    if (off + c > kMaxSurv) ::ptx::trap();  // <= ksurv lists of 16 by construction
#pragma unroll
    for (uint32_t j = 0; j < kTopK; ++j)
        if (j < c) sm.surv[off + j] = l[j];
    if (t == kThreads - 1) sm.n = off + c;
    __syncthreads();
}

// The post phase; a tree head (MK_DFLASH_TREE) also keeps each lattice row's fan and grows the token tree in place of
// the tree walk (dflash/tree.cuh), its smem past PostSmem.
__device__ __forceinline__ void head_post(const PostArgs& a
#if MK_DFLASH_TREE
                                          ,
                                          const tree::TreeArgs& tr
#endif
#if MK_DFLASH_SAMPLE
                                          ,
                                          const sample::WalkArgs& sw
#endif
) {
    ptx::griddepcontrol_wait();  // every thread's reads of predecessor words (y, the anchor) behind the wait
    __syncthreads();             // the body is done with smem: the ring is drained and the lists are final
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    PostSmem& sm = *reinterpret_cast<PostSmem*>(smem_buf);
    const PrescoreTileSmem& ts = tile_smem();
    const uint32_t t = threadIdx.x, cta = blockIdx.x, ctas = gridDim.x;
    const uint32_t warp = t / 32, lane = t % 32;

    a.part[(cta * kRows + t / kTopK) * kTopK + t % kTopK] = ts.list[t / kTopK][t % kTopK];
    if (t == 0) sm.epoch = *reinterpret_cast<volatile uint32_t*>(a.state + kStEpoch);
    __syncthreads();
    if (t == 0) ptx::red_add_rel_b32(a.state + kStCtas, 1u);
    if (cta >= a.gamma * a.group) return;

    const uint32_t e = cta / a.group, m = cta % a.group;
    const uint32_t epoch = sm.epoch, half = epoch & 1u, tag = exchange_tag(epoch);
    {
        const uint4* src = reinterpret_cast<const uint4*>(a.y + static_cast<uint64_t>(e + 1) * a.hidden);
        uint4* dst = reinterpret_cast<uint4*>(sm.yrow);
        for (uint32_t i = t; i < a.hidden / 8; i += kThreads)
            dst[i] = src[i];
    }
    if (t == 0) {
        const uint32_t target = (epoch + 1) * ctas;
        bounded_spin([&] { return reached(ptx::ld_relaxed_b32(a.state + kStCtas), target); },
                     [&] { spin_report("draft_head: CTA arrivals", ptx::ld_relaxed_b32(a.state + kStCtas), target); });
        ::ptx::fence_acquire_gpu();
    }
    __syncthreads();
    survivors(a, e, ctas, sm);
    // this CTA's share of the rescoring: items m, m + group, .. of [survivors | W_P rows], one warp per item
    const uint32_t n = sm.n, items = n + a.hproj;
    for (uint32_t i = m + a.group * warp; i < items; i += a.group * (kThreads / 32)) {
        const bool sv = i < n;
        const uint32_t id = sv ? key_id(sm.surv[i]) : 0;
        const uint32_t row = sv ? id - a.rank * a.vocab_rank : a.vocab_rank + (i - n);
        const float d = warp_dot(a.wb + static_cast<uint64_t>(row) * a.hidden, sm.yrow, a.hidden, lane);
        const __nv_bfloat16 h = __float2bfloat16_rn(d);
        if (lane == 0) {
            if (sv)
                a.resc[e * kMaxSurv + i] = make_key(__bfloat16_as_ushort(h), id);
            else
                a.hloc[e * a.hproj + (i - n)] = h;
        }
    }
    __syncthreads();
    if (t == 0) {
        sm.last = ptx::atomic_add_rel_b32(a.state + kStGroup + e, 1u) + 1 == (epoch + 1) * a.group;
        if (sm.last) ::ptx::fence_acquire_gpu();
    }
    __syncthreads();
    if (!sm.last) return;

    // the group's last CTA: the rank's top-16 of the survivors' exact keys, then the cross-rank exchange
    for (uint32_t i = t; i < n; i += kThreads)
        sm.surv[i] = __ldcg(a.resc + e * kMaxSurv + i);
    __syncthreads();
    select16(sm.surv, n, sm.mine);
    {
        const uint32_t q = t / kInboxWords, w = t % kInboxWords;
        uint64_t word;
        if (w < kTopK) {
            word = cand_word(sm.mine[w], tag);
        } else {
            const unsigned short* h = reinterpret_cast<const unsigned short*>(a.hloc) + e * a.hproj + 2 * (w - kTopK);
            word = (static_cast<uint64_t>(__ldcg(h + 1)) << 48) | (static_cast<uint64_t>(__ldcg(h)) << 32) | tag;
        }
        if (q < a.ranks)
            ::ptx::st_relaxed_sys_u64(a.inbox_peer[q] + ((half * a.ranks + a.rank) * kRows + e) * kInboxWords + w,
                                      word);
    }
    {
        // every rank's words of rows e (kInboxWords each) and e - 1 (the 16 candidate words); a landed candidate's
        // codebook row (successor for row e, predecessor for row e - 1) is prefetched into L2 under the remaining polls
        const uint64_t* in = a.inbox_peer[a.rank] + half * a.ranks * kRows * kInboxWords;
        const uint32_t n_e = a.ranks * kInboxWords, nw = n_e + (e > 0 ? a.ranks * kTopK : 0);
        for (uint32_t i = t; i < nw; i += kThreads) {
            const bool cur = i < n_e;
            const uint32_t src = cur ? i / kInboxWords : (i - n_e) / kTopK;
            const uint32_t w = cur ? i % kInboxWords : (i - n_e) % kTopK;
            const uint64_t* p = in + (src * kRows + (cur ? e : e - 1)) * kInboxWords + w;
            uint64_t word;
            bounded_spin([&] { return word_tag(word = ::ptx::ld_relaxed_sys_u64(p)) == tag; },
                         [&] { spin_report("draft_head: inbox word", word_tag(word), tag); });
            if (!cur) {
                sm.kp[src * kTopK + w] = word_key(word);
                const uint8_t* r =
                    reinterpret_cast<const uint8_t*>(a.pred_cb + int64_t(key_id(word_key(word))) * kRank);
#pragma unroll
                for (uint32_t c = 0; c < kRank * 2; c += 128)
                    ptx::prefetch_global_l2(r + c);
            } else if (w < kTopK) {
                sm.ke[src * kTopK + w] = word_key(word);
                const uint8_t* r =
                    reinterpret_cast<const uint8_t*>(a.succ_cb + int64_t(key_id(word_key(word))) * kRank);
#pragma unroll
                for (uint32_t c = 0; c < kRank * 2; c += 128)
                    ptx::prefetch_global_l2(r + c);
            } else {
                const uint32_t r = src * a.hproj + 2 * (w - kTopK);
                sm.lat.hh[r] = bf16_bits_to_float(static_cast<uint32_t>(word >> 32) & 0xFFFFu);
                sm.lat.hh[r + 1] = bf16_bits_to_float(static_cast<uint32_t>(word >> 48));
            }
        }
    }
    __syncthreads();
    select16(sm.ke, a.ranks * kTopK, sm.ce);
    if (e > 0) select16(sm.kp, a.ranks * kTopK, sm.cp);
    if (t < kTopK) {
        sm.lat.cand[t] = static_cast<int32_t>(key_id(sm.ce[t]));
        sm.lat.u[t] = softcap(bf16_bits_to_float(key_bf16(sm.ce[t])), a.mult, a.cap);
        sm.lat.pred[t] = e == 0 ? static_cast<int32_t>(a.words[kWordAnchor]) : static_cast<int32_t>(key_id(sm.cp[t]));
    }
    __syncthreads();
    float* srow = a.scores + e * kTopK * kTopK;
    lattice_slot(sm.lat, a.pred_cb, a.succ_cb, srow);
    {
        // this slot's argmax successor per predecessor p (thread t = p x 16 + c): torch.argmax over c of row p
        float v = __ldcg(srow + t);  // this thread's own store, read back
#if MK_DFLASH_TREE
        tree::fan_row(tr, e, v);
#endif
        int32_t i = static_cast<int32_t>(t % kTopK);
#pragma unroll
        for (uint32_t o = 8; o >= 1; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o, 16);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, i, o, 16);
            argmax_take(v, i, ov, oi);
        }
        if (t % kTopK == 0) a.pick[e * kTopK + t / kTopK] = i;
    }
    if (t < kTopK) {
        a.cand[e * kTopK + t] = sm.lat.cand[t];
        a.unary[e * kTopK + t] = sm.lat.u[t];
    }
    a.hhat[e * kRank + t] = __float2bfloat16_rn(sm.lat.hh[t]);
    __syncthreads();
    if (t == 0) {
        sm.last = ptx::atomic_add_rel_b32(a.state + kStLattice, 1u) + 1 == (epoch + 1) * a.gamma;
        if (sm.last) ::ptx::fence_acquire_gpu();
    }
    __syncthreads();
    if (!sm.last) return;
#if MK_DFLASH_TREE
    tree::walk(*reinterpret_cast<tree::WalkSmem*>(smem_buf + (sizeof(PostSmem) + 15) / 16 * 16), tr, a.cand, a.gamma,
               a.words, a.drafts, a.verify_ids);
    __syncthreads();
#else
#if MK_DFLASH_SAMPLE
    // the sampled walk, its score rows past PostSmem
    sample::walk(sw, a.cand, a.scores, a.words, a.gamma,
                 reinterpret_cast<float*>(smem_buf + (sizeof(PostSmem) + 15) / 16 * 16), sm.cands, sm.ids);
#else
    // the walk: slot 0 takes row 0's pick (every predecessor of slot 0 is the anchor), slot e row pick(e - 1)'s
    if (t < a.gamma * kTopK) {
        sm.picks[t] = __ldcg(a.pick + t);
        sm.cands[t] = __ldcg(a.cand + t);
    }
    __syncthreads();
    if (t == 0) {
        const int64_t anchor = a.words[kWordAnchor];
        sm.ids[0] = anchor;
        int32_t prev = 0;
        for (uint32_t s = 0; s < a.gamma; ++s) {
            prev = sm.picks[s * kTopK + prev];
            sm.ids[1 + s] = sm.cands[s * kTopK + prev];
        }
    }
    __syncthreads();
#endif
    // the verify window's 16 ids: the live block, then pads that repeat it (dflash/common.cuh's walk explains them)
    const uint32_t bl = a.gamma + 1;
    if (t < kTopK) a.verify_ids[t] = sm.ids[t % bl];
    if (t < a.gamma) a.drafts[t] = sm.ids[1 + t];
    __syncthreads();
#endif
    // every CTA read the epoch before it counted itself, and the walk runs after the last count
    if (t == 0) a.state[kStEpoch] = epoch + 1;
}

template <uint32_t kStages, uint32_t kPack, bool kEvictFirst>
__launch_bounds__(256, 1) __global__
    void draft_head(const __grid_constant__ HeadArgs args, const __grid_constant__ CUtensorMap tmap_w,
                    const __grid_constant__ CUtensorMap tmap_x, const __grid_constant__ PostArgs post
#if MK_DFLASH_TREE
                    ,
                    const __grid_constant__ tree::TreeArgs tree
#endif
#if MK_DFLASH_SAMPLE
                    ,
                    const __grid_constant__ sample::WalkArgs walk
#endif
    ) {
    head_body<kStages, kPack, kEvictFirst>(args, tmap_w, tmap_x);
#if MK_DFLASH_TREE
    head_post(post, tree);
#elif MK_DFLASH_SAMPLE
    head_post(post, walk);
#else
    head_post(post);
#endif
}

}  // namespace mk::dflash::head

#define MK_DFLASH_HEAD_entry 1
#define MK_DFLASH_HEAD_draft_head 2
#define MK_DFLASH_HEAD_CAT_(x) MK_DFLASH_HEAD_##x
#define MK_DFLASH_HEAD_CAT(x) MK_DFLASH_HEAD_CAT_(x)
#define MK_DFLASH_HEAD MK_DFLASH_HEAD_CAT(MK_IMAGE)
#if defined(MK_HOST)
#elif MK_DFLASH_HEAD == MK_DFLASH_HEAD_entry
// The draft entry, one image per draft block: -DMK_DFLASH_B=8|16 (launch: grid 144, block 256,
// smem entry::Shape<B>::kSmem, PDL).
[[maybe_unused]] static constexpr auto kKernel = &mk::dflash::draft_entry<MK_DFLASH_B>;
#elif MK_DFLASH_HEAD == MK_DFLASH_HEAD_draft_head
// The draft head on the lm_head's ring pins.
[[maybe_unused]] static constexpr auto kKernel =
    &mk::dflash::head::draft_head<mk::cfg::kStages, mk::cfg::kPack, mk::cfg::kEvictFirst>;
#else
#error "dflash/head.cu: -DMK_IMAGE=entry|draft_head"
#endif
