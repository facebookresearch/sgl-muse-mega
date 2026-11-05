#pragma once
// The DFlash2 draft's shared pieces: the bounded wait of the entry and the head, the draft's dims per rank, the qkv
// images' pins, the two GEMM epilogues (SwiGLU; the qkv's hd128 rowops), and the head's candidate keys and step words.
// The bounded wait stays ahead of the framework (README.md, "Order-dependent placement").
#include <cstdint>
#include <cstdio>

// The bounded wait of every DFlash spin (a peer CTA's count, a cross-rank packet, an mbarrier phase): a wait that
// outlives the budget names itself with one print and traps, so a protocol bug fails the next host sync instead of
// hanging every rank. Eager warm-ups skew the ranks by seconds and a draft step runs in about 0.5 ms, so the budget is
// a minute. The global timer is read every 1024 polls.

namespace mk::dflash {

inline constexpr uint64_t kSpinBudgetNs = 60000000000ull;

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The report of a timed-out wait, out of line: a printf in the kernel's own text costs it a stack frame.
__device__ __noinline__ void spin_report(const char* what, uint64_t observed, uint64_t target) {
    printf("dflash: the %s wait timed out (block %u thread %u): observed %llu, target %llu\n", what, blockIdx.x,
           threadIdx.x, static_cast<unsigned long long>(observed), static_cast<unsigned long long>(target));
}

// Polls ready() until it holds; past the budget calls report() (which names the wait) and traps.
template <class Ready, class Report>
__device__ __forceinline__ void bounded_spin(Ready ready, Report report) {
    uint64_t t0 = 0;
    for (uint32_t polls = 1; !ready(); ++polls) {
        if ((polls & 1023u) == 0) {
            const uint64_t now = globaltimer();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > kSpinBudgetNs) {
                report();
                asm volatile("trap;");
            }
        }
    }
}

}  // namespace mk::dflash

#include <cuda_bf16.h>

#include "framework.cuh"

// The DFlash2 draft as one rank sees it: 64 q heads over 8 kv heads at head_dim 128 (GQA 8:1) and an MLP intermediate
// of 28672, split over the tensor-parallel world (framework.cuh): TP8 holds 8 q heads, 1 kv head and 3584 columns
// per rank, TP4 16, 2 and 7168.

namespace mk::dflash {

inline constexpr uint32_t kHd = 128;                         // head_dim
inline constexpr uint32_t kGqa = 8;                          // q heads per kv head
inline constexpr uint32_t kKvHeads = 8 / mk::cfg::kRanks;    // kv heads per rank
inline constexpr uint32_t kQHeads = kGqa * kKvHeads;         // q heads per rank
inline constexpr uint32_t kInter = 28672 / mk::cfg::kRanks;  // the MLP intermediate per rank
static_assert(8 % mk::cfg::kRanks == 0, "every rank holds whole kv heads");

// The pins of the qkv images (kernels/dflash/layer.cu), shared with their host packer (host/runtime.cu). The ring:
// the dense executor's 8-KB W stages with their x k-blocks, as deep as the opt-in smem allows next to the K-split
// reduction's staging (B16's 32-row x blocks cost the ring 4 stages).

template <uint32_t kB>
struct QkvPins {  // the qkv images: 4-CTA K split per 64-row W block
    static constexpr uint32_t kM = 2 * kB, kStages = kB == 8 ? 20 : 16, kPack = 4;
};

// The DFlash draft MLP's gate_up projection with SwiGLU as the dense executor's tile hook. The packed W
// (sm103/dflash/weights.py pack_gate_up) holds, in each 16-row group g of 64-row block b, the gate rows of
// intermediate columns 32 b + 8 g + l at W rows 16 g + l and the matching up rows at 16 g + 8 + l (l < 8): one output's
// two operands sit in lanes l and l + 8 of one epilogue warp's TMEM band (a shfl_xor 8 for an epilogue working from the
// drain registers). This hook reads both from the executor's staged bf16 tile instead, so every epilogue thread takes
// outputs: warp q, lane j -> column 32 b + j of tokens q, q + 4, ... (a 64-B store per token per warp). SGLang's op
// order on the bf16 GEMM outputs G, U: act = bf16((G / (1 + exp(-G))) * U) in fp32.

template <uint32_t kM>
struct DflashSwiGlu {
    struct Args {
        nv_bfloat16* act;     // [rows, act_stride] bf16, the rank's intermediate
        uint32_t act_stride;  // elements
        uint32_t rows;        // the window's live rows (<= kM; the x map zero-fills past them)
    };
    static constexpr bool kMutatesTile = false;
    static constexpr bool kCkSafe = true;

    // after the out-tile sync: the tile is [kM tokens][64 W rows] bf16
    static __device__ __forceinline__ void pre_store(const Args& a, const uint32_t w_outer, nv_bfloat16* out_tile,
                                                     const uint32_t quad, const uint32_t lane) {
        const uint32_t gate_row = (lane / 8) * 16 + lane % 8;
        nv_bfloat16* const col = a.act + w_outer / 2 + lane;
#pragma unroll
        for (uint32_t i = 0; i < kM / 4; ++i) {
            const uint32_t t = quad + 4 * i;
            if (t < a.rows) {
                const float g = __bfloat162float(out_tile[t * 64 + gate_row]);
                const float u = __bfloat162float(out_tile[t * 64 + gate_row + 8]);
                const nv_bfloat16 act = __float2bfloat16_rn(__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u));
                col[uint64_t(t) * a.act_stride] = act;
            }
        }
    }
};

// The DFlash draft's qkv projection and its hd128 rowops. One rank's qkv is its q heads, then its k heads, then its v
// heads (TP8 8 + 1 + 1, TP4 16 + 2 + 2), of 128 elements over K = the hidden width. The window has
// kM = 2B rows: rows 0..B-1 are the block (positions P'..), rows B..2B-1 the committed context rows (positions
// P'-c'..), so the context K/V ride the block's W stream.
//
// Per row type (SGLang's two paths: the draft block and the context fill):
//   block row t < B: q / k heads: per-head RMSNorm, rounded to bf16, neox RoPE, bf16; q -> `out` (the attention's Q),
//                    k -> the ring; v (the GEMM's bf16 row) -> the ring (kv head g at column g x 128 of a ring row)
//   context row j = t - B: k: per-head RMSNorm, NOT rounded, RoPE, bf16; k and v -> the ring only for j < c' (the rows
//                    past the commit carry positions >= P', whose slots this launch's block rows own); q is dropped
//
// The per-head math, every operation explicitly rounded (nothing contracts), x = the GEMM's bf16 head row, d < 128:
//   ss  = ((S0 + S1) + S2) + S3, S_c = the xor butterfly (16, 8, 4, 2, 1) of x_d * x_d over d in [32 c, 32 c + 32)
//   inv = 1 / sqrt(ss / 128 + eps);  n_d = (x_d * inv) * w_d
//   d < 64: o_d = n_d cos_d - n_(d+64) sin_d;  d >= 64: o_d = n_d cos_(d-64) + n_(d-64) sin_(d-64)
// test/reference/dflash2.py qkv_rowops_ref is the same sequence in torch.
//
// The GEMM is the dense executor's 4-CTA K split (framework.cuh: kQkvN / 64 W blocks x 4 CTAs on width-4
// clusters), where a 64-row W block is half a head: the norm sums all 128 elements and every RoPE pair (d, d + 64)
// straddles the two blocks. The rows of token quarter q of a head need exactly the CTAs (half 0, q) and (half 1, q);
// the pair fixup lets the second of them to arrive (a per-(head, quarter) ticket) run the quarter's rows, its own half
// from its smem tile, its partner's from L2. One head per M = 128 tile instead keeps a head in one CTA, but streams
// the qkv W on half the CTAs.

inline constexpr uint32_t kQkvHeads = kQHeads + 2 * kKvHeads;  // + the rank's k and v heads
inline constexpr uint32_t kQkvN = kQkvHeads * kHd;             // the qkv row (TP8 1280, TP4 2560)
inline constexpr uint32_t kRingRow = kKvHeads * kHd;           // a ring row: the rank's kv heads at one position

// a k / v head's roles: v (else k), and its kv head's columns in the ring row (one kv head: the TP8 images' own
// expressions, which keep those images' code unchanged)
__host__ __device__ constexpr bool is_v_head(const uint32_t head) {
    if constexpr (kKvHeads == 1)
        return head == kQHeads + 1;
    else
        return head >= kQHeads + kKvHeads;
}
__host__ __device__ constexpr uint32_t ring_col(const uint32_t head) {
    if constexpr (kKvHeads == 1)
        return 0;
    else
        return (head - kQHeads) % kKvHeads * kHd;
}

struct DflashQkvArgs {
    nv_bfloat16* out;           // [kM, kQkvN] qkv rows: the block rows' q heads end here (the attention's Q)
    uint32_t out_stride;        // elements
    uint32_t rows;              // kM (the executor's quarter-store bound)
    uint32_t* tickets;          // u32 per (head, token quarter), zeroed once, monotonic
    const float* cos_sin;       // [kM, 128] fp32 per row: {cos[64] | sin[64]} at the row's position (theta 5e5)
    const int32_t* slot;        // [kM] the rows' ring slots
    const int32_t* commit;      // c': context rows j >= c' store nothing
    const nv_bfloat16* q_norm;  // [128]
    const nv_bfloat16* k_norm;  // [128]
    nv_bfloat16* k_ring;        // [4096, kRingRow]: the draft KV ring (slot = position & 4095)
    nv_bfloat16* v_ring;        // [4096, kRingRow]
    float eps;
};

static __device__ __forceinline__ float bf16_round(const float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

// The operands of one (token, head) row on one warp that no CTA of the launch writes; lane l works on elements
// d = 32 c + l, c = 0..3, so the RoPE partners (d, d + 64) share a lane.
struct DflashRowOps {
    float cs[2], sn[2];  // cos / sin at d = l and d = 32 + l
    float w[4];          // the head's norm weight at the lane's four elements
    int32_t slot;
    bool keep;  // this row's output is stored (q: block rows; k / v: block rows and context rows below the commit)
};

template <uint32_t kB>
static __device__ __forceinline__ DflashRowOps dflash_row_ops(const DflashQkvArgs& a, const uint32_t head,
                                                              const uint32_t t, const int32_t commit,
                                                              const uint32_t lane) {
    DflashRowOps o;
    const bool is_q = head < kQHeads, is_v = is_v_head(head);
    o.keep = t < kB || (!is_q && int32_t(t - kB) < commit);
    o.slot = is_q ? 0 : __ldg(a.slot + t);
    const float* const cs = a.cos_sin + t * kHd;
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
        o.cs[c] = is_v ? 0.0f : __ldg(cs + 32 * c + lane);
        o.sn[c] = is_v ? 0.0f : __ldg(cs + 64 + 32 * c + lane);
    }
    const nv_bfloat16* const wn = is_q ? a.q_norm : a.k_norm;
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
        o.w[c] = is_v ? 0.0f : __bfloat162float(wn[32 * c + lane]);
    return o;
}

// The per-head norm + RoPE of one row on one warp (the header's sequence); x[c] = element 32 c + lane.
template <bool kRoundNormed>
static __device__ __forceinline__ void dflash_head_rowops(const float (&x)[4], const DflashRowOps& o, const float eps,
                                                          float (&out)[4]) {
    float s[4];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
        s[c] = __fmul_rn(x[c], x[c]);
#pragma unroll
    for (uint32_t off = 16; off >= 1; off >>= 1)
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c)
            s[c] = __fadd_rn(s[c], __shfl_xor_sync(0xffffffffu, s[c], off));
    const float ss = __fadd_rn(__fadd_rn(__fadd_rn(s[0], s[1]), s[2]), s[3]);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(ss, 1.0f / float(kHd)), eps)));
    float n[4];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        n[c] = __fmul_rn(__fmul_rn(x[c], inv), o.w[c]);
        if constexpr (kRoundNormed) n[c] = bf16_round(n[c]);
    }
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
        out[c] = __fsub_rn(__fmul_rn(n[c], o.cs[c]), __fmul_rn(n[c + 2], o.sn[c]));
        out[c + 2] = __fadd_rn(__fmul_rn(n[c + 2], o.cs[c]), __fmul_rn(n[c], o.sn[c]));
    }
}

// One (token t, head) row on one warp from its bf16 GEMM row: q heads into `out` in place, k / v into the ring.
template <uint32_t kB>
static __device__ __forceinline__ void dflash_row(const DflashQkvArgs& a, const uint32_t head, const uint32_t t,
                                                  const uint32_t lane, const float (&x)[4], const DflashRowOps& o) {
    if (!o.keep) return;
    const bool is_q = head < kQHeads, is_v = is_v_head(head);
    float y[4];
    if (is_v) {
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c)
            y[c] = x[c];
    } else if (t < kB) {
        dflash_head_rowops<true>(x, o, a.eps, y);
    } else {
        dflash_head_rowops<false>(x, o, a.eps, y);
    }
    nv_bfloat16* const dst = is_q ? a.out + uint64_t(t) * a.out_stride + head * kHd
                                  : (is_v ? a.v_ring : a.k_ring) + uint64_t(o.slot) * kRingRow + ring_col(head);
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
        dst[32 * c + lane] = __float2bfloat16_rn(y[c]);
}

// The pair fixup (kCk4PairFixup; the two CTAs holding a head's halves): after the cluster's K reduction each CTA
// stores its token quarter of its 64-row block to `out`; a per-(head, quarter) ticket (2 arrivals per launch,
// monotonic: no spin, no co-residency) lets the second of the pair run the quarter's kM / 4 rows on its epilogue quad,
// warp `quad` taking tokens quad, quad + 4, ... The role warps leave.
template <uint32_t kB>
struct DflashQkvPair {
    using Args = DflashQkvArgs;
    static constexpr bool kMutatesTile = false;
    static constexpr bool kCkSafe = true;
    static constexpr bool kCk4PairFixup = true;

    static __device__ __forceinline__ void pre_store(const Args&, const uint32_t, nv_bfloat16*, const uint32_t,
                                                     const uint32_t) {}

    // the fixup's row operands (no thread of the launch writes them), loadable ahead of the drain
    template <uint32_t kM>
    struct Ops {
        DflashRowOps r[kM / 16];
    };
    template <uint32_t kM>
    static __device__ __forceinline__ Ops<kM> ck4_pair_ops(const Args& a, const uint32_t w_outer, const uint32_t k_rank,
                                                           const uint32_t warp, const uint32_t lane) {
        constexpr uint32_t kQ = kM / 4, kRows = kQ / 4;
        Ops<kM> o{};
        const uint32_t head = w_outer / kHd, t0 = k_rank * kQ;
        if (warp < 4 || (head < kQHeads && t0 >= kB)) return o;
        const int32_t commit = __ldg(a.commit);
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i)
            o.r[i] = dflash_row_ops<kB>(a, head, t0 + (warp - 4) + 4 * i, commit, lane);
        return o;
    }
    template <uint32_t kM>
    static __device__ __forceinline__ void ck4_pair_fixup_ops(const Args& a, const Ops<kM>& o, const uint32_t w_outer,
                                                              const uint32_t k_rank, const uint32_t warp,
                                                              const uint32_t lane, const nv_bfloat16* out_tile,
                                                              const bool ticket_thread) {
        static_assert(kM == 2 * kB, "block rows then context rows");
        constexpr uint32_t kQ = kM / 4, kRows = kQ / 4;
        if (warp < 4) return;
        const uint32_t quad = warp - 4, head = w_outer / kHd, half = (w_outer % kHd) / 64, t0 = k_rank * kQ;
        // B is a multiple of the quarter, so a quarter is all block rows or all context rows: a q head's context
        // quarter has nothing to do on either CTA of its pair (neither takes the ticket)
        if (head < kQHeads && t0 >= kB) return;
        __shared__ uint32_t s_second;
        ptx::sync_aligned(block::kEpi, 128);  // the quarter's stores issued
        // the split-K "last block" ticket, acq_rel: releases the stores the CTA barrier above ordered before it,
        // acquires the other arrival's stores for the reads the barrier below orders after it
        if (ticket_thread) s_second = ::ptx::atomic_add_acq_rel_b32(a.tickets + head * 4 + k_rank, 1u) % 2 == 1;
        ptx::sync_aligned(block::kEpi, 128);
        if (!s_second) return;
        float x[kRows][4];
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
            const uint32_t t = t0 + quad + 4 * i;
#pragma unroll
            for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t d = 32 * c + lane;
                const nv_bfloat16 v = d / 64 == half ? out_tile[t * 64 + d % 64]
                                                     : __ldcg(a.out + uint64_t(t) * a.out_stride + head * kHd + d);
                x[i][c] = o.r[i].keep ? __bfloat162float(v) : 0.0f;
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i)
            dflash_row<kB>(a, head, t0 + quad + 4 * i, lane, x[i], o.r[i]);
    }

    template <uint32_t kM>
    static __device__ __forceinline__ void ck4_pair_fixup(const Args& a, const uint32_t w_outer, const uint32_t k_rank,
                                                          const uint32_t warp, const uint32_t lane,
                                                          const nv_bfloat16* out_tile, const bool ticket_thread) {
        if (warp < 4) return;
        const uint32_t head = w_outer / kHd, t0 = k_rank * (kM / 4);
        if (head < kQHeads && t0 >= kB) return;
        Ops<kM> o;
        const int32_t commit = __ldg(a.commit);
#pragma unroll
        for (uint32_t i = 0; i < kM / 16; ++i)
            o.r[i] = dflash_row_ops<kB>(a, head, t0 + (warp - 4) + 4 * i, commit, lane);
        ck4_pair_fixup_ops<kM>(a, o, w_outer, k_rank, warp, lane, out_tile, ticket_thread);
    }
};

// The DFlash draft head and the verify's accept tail: the pieces the head (dflash/head.cu) and the accept (run by
// verify_accept, dflash/verify_accept.cu) share. Every launch is 256 threads.
//
// Candidate keys. A candidate is (bf16 logit, vocab id); its 64-bit key orders candidates the way the reference's
// top-k does (a larger logit first, a NaN above every number, -0 equal to +0, equal logits by the smaller id), so the
// top-16 of any union of candidate sets is the top-16 of the union of their top-16s, and a max over keys is a total
// order (no ties):
//     key = ord16(bf16) << 18 | (2^18 - 1 - id)        (id < 2^18; key 0 = an empty slot, below every real key)
//
// Cross-rank words. A rank publishes its candidates and its slice of W_P y as 8-byte words that carry the launch's
// tag in their low 30 bits; one 8-byte store is single-copy atomic, so a reader that sees the tag sees the payload
// (no flag, no fence):
//     candidate word = bf16 << 48 | id << 30 | tag;   hhat word = bf16[2w + 1] << 48 | bf16[2w] << 32 | tag

inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kTopK = 16;   // candidates per slot
inline constexpr uint32_t kRank = 256;  // selector rank (the codebooks' width, W_P's rows)
// the exchange's words per (half, source rank, row): 16 candidate words, then the rank's band of W_P y (kRank / R
// rows) as bf16 pairs
inline constexpr uint32_t kInboxWords = kTopK + kRank / cfg::kRanks / 2;
inline constexpr uint32_t kIdBits = 18;  // vocab ids below 2^18 (202,048)
inline constexpr uint32_t kTagBits = 30;
static_assert(kTopK * kTopK == kThreads && kRank == kThreads, "one thread per edge, one per rank column");

// The step words (int64, zeroed at bind): written by verify_accept's last CTA, read by the next iteration's draft.
inline constexpr uint32_t kWordP = 0;         // the window's first position (the anchor's); the accept moves it to P'
inline constexpr uint32_t kWordAnchor = 1;    // the next window's first token (the accept's bonus)
inline constexpr uint32_t kWordAccept = 2;    // drafts accepted by the last verify
inline constexpr uint32_t kWordCommit = 3;    // tokens it committed (accept + 1)
inline constexpr uint32_t kWordIter = 4;      // accepts run so far
inline constexpr uint32_t kWordOutCount = 5;  // tokens appended to the out ring so far
inline constexpr uint32_t kWords = 8;

// wrap-safe "counter reached target" for the monotonic launch counters (never reset; target = launches x arrivals)
__device__ __forceinline__ bool reached(uint32_t seen, uint32_t target) {
    return static_cast<int32_t>(seen - target) >= 0;
}

// ---- candidate keys ----------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ord16(uint32_t b) {
    if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;  // NaN: above +inf
    if (b == 0x8000u) b = 0;                      // -0 == +0
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ uint32_t unord16(uint32_t o) {
    if (o == 0xFFFFu) return 0x7FC0u;
    return (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
}

__device__ __forceinline__ uint64_t make_key(uint32_t bf16_bits, uint32_t id) {
    return (static_cast<uint64_t>(ord16(bf16_bits)) << kIdBits) | ((1u << kIdBits) - 1u - id);
}

__device__ __forceinline__ uint32_t key_id(uint64_t key) {
    return (1u << kIdBits) - 1u - static_cast<uint32_t>(key & ((1u << kIdBits) - 1u));
}

__device__ __forceinline__ uint32_t key_bf16(uint64_t key) {
    return unord16(static_cast<uint32_t>(key >> kIdBits));
}

__device__ __forceinline__ uint32_t exchange_tag(uint32_t epoch) {
    return epoch % ((1u << kTagBits) - 1u) + 1u;  // never 0: a zeroed inbox holds no live word
}

__device__ __forceinline__ uint64_t cand_word(uint64_t key, uint32_t tag) {
    return (static_cast<uint64_t>(key_bf16(key)) << 48) | (static_cast<uint64_t>(key_id(key)) << kTagBits) | tag;
}

__device__ __forceinline__ uint64_t word_key(uint64_t w) {
    return make_key(static_cast<uint32_t>(w >> 48), static_cast<uint32_t>(w >> kTagBits) & ((1u << kIdBits) - 1u));
}

__device__ __forceinline__ uint32_t word_tag(uint64_t w) {
    return static_cast<uint32_t>(w) & ((1u << kTagBits) - 1u);
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t b) {
    return __uint_as_float(b << 16);
}

// The unary score of a candidate logit v: the target's own logit transform, the softcap expression of the logits
// tail (head.cu), cap * tanh(v * multiplier / cap) in fp32 in that operation order.
__device__ __forceinline__ float softcap(float v, float mult, float cap) {
    return __fmul_rn(tanhf(__fdiv_rn(__fmul_rn(v, mult), cap)), cap);
}

// ---- torch.argmax's order -----------------------------------------------------------------------------------------

// (av, ai) beats (bv, bi): a greater value, a NaN over a number, the lower index on equal values (or two NaNs). The
// identity is (-inf, INT32_MAX): every real pair beats it.
__device__ __forceinline__ bool argmax_beats(float av, int32_t ai, float bv, int32_t bi) {
    const bool an = av != av, bn = bv != bv;
    if (an != bn) return an;
    if (!an && av != bv) return av > bv;
    return ai < bi;
}

__device__ __forceinline__ void argmax_take(float& v, int32_t& i, float ov, int32_t oi) {
    if (argmax_beats(ov, oi, v, i)) {
        v = ov;
        i = oi;
    }
}

// the warp's best pair in every lane (a total order, so the butterfly's order does not matter)
__device__ __forceinline__ void warp_argmax(float& v, int32_t& i) {
#pragma unroll
    for (uint32_t o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, i, o);
        argmax_take(v, i, ov, oi);
    }
}

// ---- top-16 selection (block-wide, 256 threads) -------------------------------------------------------------------

// out[r] = the key of rank r among keys[0, n) (smem), r < 16, by counting: every key is compared with every other, so
// the cost is n^2 / 256 compares per thread (n <= ~300 on every hot path). Keys are unique (empty 0 keys may repeat:
// they all land on one slot, and slots no key reaches stay empty).
__device__ __forceinline__ void select16(const uint64_t* keys, uint32_t n, uint64_t* out) {
    if (threadIdx.x < kTopK) out[threadIdx.x] = 0;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const uint64_t k = keys[i];
        uint32_t r = 0;
        for (uint32_t j = 0; j < n; ++j)
            r += keys[j] > k;
        if (r < kTopK) out[r] = k;
    }
    __syncthreads();
}

// ---- the lattice (one CTA per slot) and the walk ------------------------------------------------------------------

// prod rows padded by 8 bf16 so the warp's two p rows sit on different banks
inline constexpr uint32_t kProdStride = kRank + 8;

struct alignas(16) LatticeSmem {
    float hh[kRank];                          // this slot's W_P y (the kernel's own bf16, widened)
    __nv_bfloat16 prod[kTopK * kProdStride];  // bf16(A[pred(p)][r] * hh[r])
    __nv_bfloat16 st[kRank * kTopK];          // S[cand(c)][r] transposed: one r row holds the 16 candidates
    int32_t pred[kTopK], cand[kTopK];
    float u[kTopK];
};

// x into the compensated sum (s, c) (Knuth's TwoSum: s + c carries the exact running sum to ~2^-48 relative). The
// intrinsics forbid FMA contraction, which would break the error term.
__device__ __forceinline__ void two_sum_acc(float& s, float& c, float x) {
    const float t = __fadd_rn(s, x);
    const float bp = __fsub_rn(t, s);
    c = __fadd_rn(c, __fadd_rn(__fsub_rn(s, __fsub_rn(t, bp)), __fsub_rn(x, bp)));
    s = t;
}

// Slot e's K x K edge scores into scores[p * K + c] (fp32):
//     score[p][c] = u[c] + float(bf16(sum_r bf16(A[pred(p)][r] * hh[r]) * S[cand(c)][r]))
// with the caller's sm.hh / sm.pred / sm.cand / sm.u filled and synced. Each product of two bf16 is exact in fp32 and
// the 256-term sum is compensated, so the dot is the fp32 rounding of the exact sum (to ~2^-28 per score): the bf16
// rounding of the reference's fp64 dot (which PyTorch also rounds through fp32), independent of the summation order.
__device__ __forceinline__ void lattice_slot(LatticeSmem& sm, const __nv_bfloat16* pred_cb,
                                             const __nv_bfloat16* succ_cb, float* scores) {
    const uint32_t t = threadIdx.x;
    {
        // thread t owns rank column t of every predecessor and successor row (each row read is one coalesced 512 B)
        __nv_bfloat16 a[kTopK], s[kTopK];
#pragma unroll
        for (uint32_t p = 0; p < kTopK; ++p)
            a[p] = pred_cb[static_cast<int64_t>(sm.pred[p]) * kRank + t];
#pragma unroll
        for (uint32_t c = 0; c < kTopK; ++c)
            s[c] = succ_cb[static_cast<int64_t>(sm.cand[c]) * kRank + t];
        const float h = sm.hh[t];
#pragma unroll
        for (uint32_t p = 0; p < kTopK; ++p)
            sm.prod[p * kProdStride + t] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(a[p]), h));
        uint4* dst = reinterpret_cast<uint4*>(sm.st + t * kTopK);
        dst[0] = *reinterpret_cast<const uint4*>(&s[0]);
        dst[1] = *reinterpret_cast<const uint4*>(&s[8]);
    }
    __syncthreads();
    const uint32_t p = t / kTopK, c = t % kTopK;
    const __nv_bfloat16* prow = sm.prod + p * kProdStride;
    float s[4] = {0.f, 0.f, 0.f, 0.f}, e[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t r = 0; r < kRank; r += 4) {
        const uint2 pv = *reinterpret_cast<const uint2*>(prow + r);
        const __nv_bfloat162 p01 = *reinterpret_cast<const __nv_bfloat162*>(&pv.x);
        const __nv_bfloat162 p23 = *reinterpret_cast<const __nv_bfloat162*>(&pv.y);
        const float pf[4] = {__low2float(p01), __high2float(p01), __low2float(p23), __high2float(p23)};
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q)
            two_sum_acc(s[q], e[q], __fmul_rn(pf[q], __bfloat162float(sm.st[(r + q) * kTopK + c])));
    }
    float sum = s[0], err = __fadd_rn(__fadd_rn(e[0], e[1]), __fadd_rn(e[2], e[3]));
    two_sum_acc(sum, err, s[1]);
    two_sum_acc(sum, err, s[2]);
    two_sum_acc(sum, err, s[3]);
    const float dot = __bfloat162float(__float2bfloat16_rn(__fadd_rn(sum, err)));
    scores[t] = __fadd_rn(sm.u[c], dot);
}

struct WalkSmem {
    float scores[kTopK * kTopK * kTopK];  // up to 16 slots
    int32_t cand[kTopK * kTopK];
    int64_t ids[kTopK];
};

// The greedy walk (the Triton walk's greedy branch): slot e reads the score row of slot e - 1's pick (row 0 for slot 0,
// whose predecessors are all the anchor) and takes torch.argmax's pick. Then the verify window's ids: [anchor, d_1 ..
// d_gamma], and for a window shorter than 16 the live ids repeated into the pad rows (pads routed like live rows take
// no extra expert-prefetch slots). The whole CTA stages the scores (they came from other CTAs: global, acquired by the
// caller), warp 0 walks.
__device__ __forceinline__ void walk(WalkSmem& sm, const float* scores, const int32_t* cand, uint32_t gamma,
                                     int64_t anchor, int64_t* drafts, int64_t* verify_ids) {
    const uint32_t t = threadIdx.x;
    {
        // one round of independent loads (thread t: element t of every slot's 256 scores)
        float v[kTopK];
#pragma unroll
        for (uint32_t e = 0; e < kTopK; ++e)
            if (e < gamma) v[e] = __ldcg(scores + e * kTopK * kTopK + t);
        const int32_t cv = t < gamma * kTopK ? __ldcg(cand + t) : 0;
#pragma unroll
        for (uint32_t e = 0; e < kTopK; ++e)
            if (e < gamma) sm.scores[e * kTopK * kTopK + t] = v[e];
        if (t < gamma * kTopK) sm.cand[t] = cv;
    }
    __syncthreads();
    if (t < 32) {
        int32_t prev = 0;
        if (t == 0) sm.ids[0] = anchor;
        for (uint32_t e = 0; e < gamma; ++e) {
            float v = t < kTopK ? sm.scores[(e * kTopK + prev) * kTopK + t] : -INFINITY;
            int32_t i = t < kTopK ? static_cast<int32_t>(t) : INT32_MAX;
            warp_argmax(v, i);
            prev = i;
            if (t == 0) sm.ids[1 + e] = sm.cand[e * kTopK + i];
        }
        __syncwarp();
        // the verify window's 16 ids (the fused entry reads them after its grid wait): the live block, then pads that
        // repeat it (row r = row r mod b). The layers mask a pad's MoE, but the router still routes it and the next
        // step's expert prefetch histograms those routes, where repeated live ids compete like the live rows instead of
        // as one clump on a single id.
        const uint32_t b = gamma + 1;
        if (t < kTopK) verify_ids[t] = sm.ids[t % b];
        if (t < gamma) drafts[t] = sm.ids[1 + t];
    }
}

}  // namespace mk::dflash
