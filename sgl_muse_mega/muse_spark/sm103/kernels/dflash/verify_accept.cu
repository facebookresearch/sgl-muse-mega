// The verify's lm_head with the greedy accept in its epilogue, on the lm_head's persist pins: the accept's operands,
// then verify_accept. One image (-DMK_DFLASH_TREE=1: the token tree's accept and tree_compact, dflash/tree.cuh;
// -DMK_DFLASH_SAMPLE=1: sample_accept, the sampled accept after the verify graph's lm_head and softcap,
// dflash/sample.cuh); the host library includes this file under MK_HOST for the argument structs.

#include "dflash/common.cuh"
#if MK_DFLASH_TREE
#include "dflash/tree.cuh"
#endif
#if MK_DFLASH_SAMPLE || defined(MK_HOST)
#include "dflash/sample.cuh"
#endif

namespace mk::dflash {

// The greedy accept's operands: its inputs, its outputs (the step words, the committed tokens) and the next
// iteration's feeds, which verify_accept's last CTA writes.
struct AcceptArgs {
    const int64_t* tokens;           // [b_live] the verified window [anchor, d_1 .. d_(b_live - 1)]
    int64_t* words;                  // the step words (kWord*)
    int32_t* top1;                   // [16] each live row's argmax
    const int32_t* accept_override;  // [n_override] a forced accept per iteration (tests on synthetic weights), or null
    int32_t* out_ring;               // [ring] committed tokens
    const float* cos_sin_cache;      // target rope rows [cos_sin_rows][d]
    float* cos_sin;                  // [16][d]
    int32_t* loc_full;               // [16]
    int32_t* loc_swa;                // [16]
    int32_t* kv_len;                 // [1]
    const float* dcos_sin_cache;     // draft rope rows [dcos_sin_rows][dd]
    float* dcos_sin;                 // [2 b_live][dd]
    int32_t* dslot;                  // [2 b_live]
    int32_t* dkv_len;                // [1] the draft attention's kv_len
    int64_t cos_sin_rows;
    int64_t dcos_sin_rows;
    uint32_t b_live;
    uint32_t n_override;
    uint32_t ring;
    uint32_t d;
    uint32_t dd;
    uint32_t dring;  // draft KV ring rows (a power of two)
};

inline constexpr uint32_t kWindowRows = 16;
inline constexpr uint32_t kMaxD = 64, kMaxDD = 128;  // rope row widths the accept's staging covers (checked at bind)
// float4 per thread: target (16 + 15) rows x 64 / 4, draft 32 rows x 128 / 4
inline constexpr uint32_t kTgtPer = ((kWindowRows + kWindowRows - 1) * kMaxD / 4 + kThreads - 1) / kThreads;
inline constexpr uint32_t kDrfPer = (2 * kWindowRows * kMaxDD / 4 + kThreads - 1) / kThreads;

}  // namespace mk::dflash

// verify_accept: the verify's lm_head and the greedy accept in one launch, with no logits row anywhere.
//
// The persistent dense body of the lm_head (framework.cuh; the lm_head image's pins) runs over the window's final
// normed rows with its own epilogue sink: each [16, 64] fp32 tile (the accumulator, unrounded) goes
// through the softcap image's expression (head.cu) and updates the CTA's per-row argmax of the softcapped fp32
// values in torch.argmax's order (a NaN first, then the larger value, then the lower id); rows >= b_live are skipped.
// After the body every CTA:
//   1. publishes its per-row best {softcapped value, id} and counts itself (release);
//   2. the last CTA (acquire) reduces each live row over the CTAs and stores one tagged word per row into every rank's
//      inbox, {softcapped value << 32 | id << 14 | tag}: one 8-B store is single-copy atomic, so no flag and no fence;
//   3. it waits for every rank's words of the live rows and merges them in the same order: every rank gets the
//      torch.argmax of the softcapped fp32 row, as SGLang's greedy verify takes it;
//   4. it runs the accept and writes the next iteration's feeds, with the reads the accept does not decide issued
//      before the exchange.
// The inbox has two halves by launch parity: a rank writes a half again only two launches later, after every rank read
// it (it needs every rank's words of the launch in between, which each rank sends only after its previous launch
// completed).
//
// launch: grid = the lm_head's persistent grid, block 256, smem dense_persist_smem_layout(16, 4, 20).size
// (the accept's staging reuses it after the body), PDL

namespace mk::dflash {

inline constexpr uint32_t kAcceptRows = kWindowRows;

// The exchanged word's tag: 14 bits (never 0) beside the fp32 value and the 18-bit id. A slot read by launch e holds 0
// (a reset zeroes the inbox with the epoch), launch e - 2's word (a half is rewritten every second launch) or launch
// e's, and the tag's odd period 2^14 - 1 never gives e - 2 the tag of e.
inline constexpr uint32_t kAcceptTagBits = 32 - kIdBits;

__device__ __forceinline__ uint32_t accept_tag(uint32_t epoch) {
    return epoch % ((1u << kAcceptTagBits) - 1u) + 1u;
}

struct AcceptTileSmem {
    float val[kAcceptRows];
    int32_t idx[kAcceptRows];
};

__device__ __forceinline__ AcceptTileSmem& accept_smem() {
    __shared__ AcceptTileSmem s;
    return s;
}

// a CTA's best {value, id} as one word: the fp32 bits high, the id low; the identity (-inf, INT32_MAX) round-trips
__device__ __forceinline__ uint64_t accept_pack(float v, int32_t i) {
    return (static_cast<uint64_t>(__float_as_uint(v)) << 32) | static_cast<uint32_t>(i);
}

// a rank's exchanged word of a row: {value << 32 | id << 14 | tag}
__device__ __forceinline__ uint64_t accept_word(float v, int32_t i, uint32_t tag) {
    return (static_cast<uint64_t>(__float_as_uint(v)) << 32) | (static_cast<uint32_t>(i) << kAcceptTagBits) | tag;
}

__device__ __forceinline__ uint32_t accept_word_tag(uint64_t w) {
    return static_cast<uint32_t>(w) & ((1u << kAcceptTagBits) - 1u);
}

struct AcceptSink {
    struct Args {
        uint32_t id0;  // the shard's first vocab id
        uint32_t b_live;
        float scale;
        float cap;
    };
    static constexpr bool kFenceSys = false;  // nothing leaves the CTA from a tile; the post-phase publishes
    static constexpr bool kPrologue = true;

    static __device__ __forceinline__ void prologue(const Args&, const uint32_t quad, const uint32_t lane_idx) {
        const uint32_t tid = quad * 32 + lane_idx;
        if (tid < kAcceptRows) {
            AcceptTileSmem& s = accept_smem();
            s.val[tid] = -INFINITY;
            s.idx[tid] = INT32_MAX;
        }
    }

    // Thread (row t, packet p) holds 8 of the row's 64 values, columns 4p .. 4p + 3 and 32 + 4p .. 32 + 4p + 3 (a
    // row's 8 lanes read 128 contiguous bytes per load); the row's 8 lanes reduce, lane p = 0 updates the row.
    template <uint32_t kM>
    static __device__ __forceinline__ void tile(const Args& a, const uint32_t w_outer, const uint32_t n_out,
                                                const float* out_tile, const uint32_t quad, const uint32_t lane_idx) {
        static_assert(kM == kAcceptRows);
        const uint32_t tid = quad * 32 + lane_idx, t = tid >> 3, p = tid & 7;
        if (t >= a.b_live) return;  // whole 8-lane groups leave together
        float bv = -INFINITY;
        int32_t bi = INT32_MAX;
#pragma unroll
        for (uint32_t c = p * 4; c < 64; c += 32) {
            const float4 x = *reinterpret_cast<const float4*>(out_tile + t * 64 + c);
            const float xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t col = w_outer + c + e;
                if (col < n_out) argmax_take(bv, bi, softcap(xs[e], a.scale, a.cap), a.id0 + col);
            }
        }
        const uint32_t group = 0xFFu << (lane_idx & 24u);
#pragma unroll
        for (uint32_t o = 1; o < 8; o <<= 1)
            argmax_take(bv, bi, __shfl_xor_sync(group, bv, o), __shfl_xor_sync(group, bi, o));
        if (p == 0) {
            AcceptTileSmem& s = accept_smem();
            argmax_take(s.val[t], s.idx[t], bv, bi);
        }
    }

    static __device__ __forceinline__ void arrive(const Args&) {}
};

struct AcceptPostArgs {
    uint64_t* inbox_peer[8];  // every rank's inbox [2 halves][ranks][16 rows] (symmetric), this rank's at [rank]
    uint64_t* part;           // [grid][16 rows] per-CTA bests (accept_pack)
    uint32_t* state;          // u32 [epoch, CTA arrivals], zeroed once, never reset
    AcceptArgs acc;
    uint32_t rank;
    uint32_t ranks;
};

// A tree window (MK_DFLASH_TREE): the accept follows the window's matching path (dflash/tree.cuh), commits its tokens,
// and records the path and the window's slots for tree_compact; the forced accept names the leaf row.
__device__ __forceinline__ void accept_post(const AcceptPostArgs& a
#if MK_DFLASH_TREE
                                            ,
                                            int32_t* const table
#endif
#if MK_DFLASH_SAMPLE
                                            ,
                                            const sample::Args& sa
#endif
) {
    ptx::griddepcontrol_wait();  // the step words and the window's tokens are the predecessors' outputs
    __syncthreads();             // the body is done with smem: the ring is drained and the sink's rows are final
    __shared__ uint32_t s_epoch;
    __shared__ bool s_last;
    const uint32_t tid = threadIdx.x;
#if !MK_DFLASH_SAMPLE
    const uint32_t lane = tid & 31, warp = tid >> 5, cta = blockIdx.x;
    constexpr uint32_t kWarps = kThreads / 32;
#endif
    const AcceptArgs& c = a.acc;
#if !MK_DFLASH_SAMPLE
    {
        const AcceptTileSmem& ts = accept_smem();
        if (tid < kAcceptRows) a.part[cta * kAcceptRows + tid] = accept_pack(ts.val[tid], ts.idx[tid]);
    }
#endif
    if (tid == 0) s_epoch = *reinterpret_cast<volatile uint32_t*>(a.state);
    __syncthreads();
    if (tid == 0) {
        // the release add publishes the partial before the arrival counts; the last arriver acquires every partial
        s_last = ptx::atomic_add_rel_b32(a.state + 1, 1u) + 1 == (s_epoch + 1) * gridDim.x;
        if (s_last) ::ptx::fence_acquire_gpu();
    }
    __syncthreads();
    if (!s_last) return;
    const uint32_t epoch = s_epoch;
#if !MK_DFLASH_SAMPLE
    const uint32_t half = epoch & 1u, tag = accept_tag(epoch);
#endif

    // the feeds' rope rows, prefetched into the drained ring
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    float4* const s_tgt = reinterpret_cast<float4*>(smem_buf);
    float4* const s_drf = s_tgt + kTgtPer * kThreads;
    __shared__ int64_t s_tok[kWindowRows], s_words[kWords];
#if MK_DFLASH_SAMPLE
    __shared__ uint32_t s_acc;
    __shared__ int64_t s_bonus;
#else
    __shared__ int32_t t1[kWindowRows];
#endif
    __shared__ int64_t s_c;
#if !MK_DFLASH_SAMPLE
    __shared__ uint64_t s_mine[kAcceptRows], s_in[8][kAcceptRows];
#endif
    const int64_t p = c.words[kWordP];
    if (tid < c.b_live) s_tok[tid] = c.tokens[tid];
    if (tid < kWords) s_words[tid] = c.words[tid];
#if MK_DFLASH_TREE
    // the table past the staging; the window's slots before the feeds below move to the next window
    int32_t* const s_tab = reinterpret_cast<int32_t*>(s_drf + kDrfPer * kThreads);
    if (tid < tree::kPath) s_tab[tid] = table[tid];
    if (tid < kWindowRows) {
        table[tree::kSlotFull + tid] = c.loc_full[tid];
        table[tree::kSlotSwa + tid] = c.loc_swa[tid];
    }
#endif
    if (p + c.b_live + kWindowRows > c.cos_sin_rows || p + 2 * c.b_live > c.dcos_sin_rows) {
        if (tid == 0) printf("draft accept: P %lld past the rope tables\n", static_cast<long long>(p));
        ::ptx::trap();
    }
    const uint32_t tq = c.d / 4, dq = c.dd / 4;
    const uint32_t n_t = (c.b_live + kWindowRows - 1) * tq, n_d = 2 * c.b_live * dq;
    const float4* tsrc = reinterpret_cast<const float4*>(c.cos_sin_cache) + (p + 1) * tq;
    const float4* dsrc = reinterpret_cast<const float4*>(c.dcos_sin_cache) + p * dq;
    for (uint32_t i = tid; i < n_t; i += kThreads)
        ::ptx::cp_async_cg_16(::ptx::to_shared(&s_tgt[i]), tsrc + i);
    for (uint32_t i = tid; i < n_d; i += kThreads)
        ::ptx::cp_async_cg_16(::ptx::to_shared(&s_drf[i]), dsrc + i);
    ::ptx::cp_async_commit_group();

#if MK_DFLASH_SAMPLE
    __syncthreads();  // the step words
    {
        uint32_t acc;
        int64_t bonus;
        // the row masses past the feeds' staging
        sample::decide(sa, static_cast<uint64_t>(s_words[kWordIter]),
                       reinterpret_cast<float*>(s_drf + kDrfPer * kThreads), acc, bonus);
        if (tid == 0) {
            s_acc = acc;
            s_bonus = bonus;
        }
    }
#else
    // the rank's best of each live row over the CTAs (warp w: rows w, w + 8)
    for (uint32_t r = warp; r < c.b_live; r += kWarps) {
        float bv = -INFINITY;
        int32_t bi = INT32_MAX;
        for (uint32_t k = lane; k < gridDim.x; k += 32) {
            const uint64_t w = __ldcg(a.part + k * kAcceptRows + r);
            argmax_take(bv, bi, __uint_as_float(static_cast<uint32_t>(w >> 32)), static_cast<int32_t>(w));
        }
        warp_argmax(bv, bi);
        if (lane == 0) s_mine[r] = accept_word(bv, bi, tag);
    }
    __syncthreads();
    // to every rank (thread q x 16 + r), then every rank's word of every live row (thread src x 16 + r)
    const uint32_t q = tid / kAcceptRows, r = tid % kAcceptRows;
    const bool mine = q < a.ranks && r < c.b_live;
    if (mine) ::ptx::st_relaxed_sys_u64(a.inbox_peer[q] + (half * a.ranks + a.rank) * kAcceptRows + r, s_mine[r]);
    if (mine) {
        const uint64_t* in = a.inbox_peer[a.rank] + (half * a.ranks + q) * kAcceptRows + r;
        uint64_t w;
        bounded_spin([&] { return accept_word_tag(w = ::ptx::ld_relaxed_sys_u64(in)) == tag; },
                     [&] { printf("draft accept: rank %u waits row %u of rank %u\n", a.rank, r, q); });
        s_in[q][r] = w;
    }
    __syncthreads();
    for (uint32_t row = warp; row < c.b_live; row += kWarps) {
        float bv = -INFINITY;
        int32_t bi = INT32_MAX;
        if (lane < a.ranks) {
            const uint64_t w = s_in[lane][row];
            bi = static_cast<int32_t>(static_cast<uint32_t>(w) >> kAcceptTagBits);
            bv = __uint_as_float(static_cast<uint32_t>(w >> 32));
        }
        warp_argmax(bv, bi);
        if (lane == 0) t1[row] = bi;
    }
#endif

    // the accept, then the next iteration's feeds
    __syncthreads();
#if MK_DFLASH_TREE
    if (warp == 0) {
        uint32_t leaf = tree::accept_leaf(s_tab, s_tok, t1);
        if (c.accept_override != nullptr) {
            const int32_t f = c.accept_override[s_words[kWordIter] % c.n_override];
            leaf = f < 0 ? 0u : min(static_cast<uint32_t>(f), tree::kRows - 1);
        }
        if (lane == 0) s_tab[tree::kPath] = static_cast<int32_t>(leaf);
    }
    __syncthreads();
#endif
    if (tid == 0) {
        const int64_t iter = s_words[kWordIter];
#if MK_DFLASH_TREE
        // the path's rows in ascending order (preorder: by depth), committed and recorded for tree_compact
        const uint32_t leaf = static_cast<uint32_t>(s_tab[tree::kPath]);
        const uint32_t acc = static_cast<uint32_t>(s_tab[tree::kDepth + leaf]);
        const int64_t cc = acc + 1, bonus = t1[leaf];
        const int64_t cnt = s_words[kWordOutCount];
        table[tree::kPath] = static_cast<int32_t>(leaf);
        uint32_t m = static_cast<uint32_t>(s_tab[tree::kMask + leaf]) & ~1u;
        for (uint32_t j = 0; m != 0; ++j, m &= m - 1) {
            const uint32_t r = __ffs(m) - 1;
            table[tree::kPath + 1 + j] = static_cast<int32_t>(r);
            c.out_ring[(cnt + j) % c.ring] = static_cast<int32_t>(s_tok[r]);
        }
#elif MK_DFLASH_SAMPLE
        const uint32_t acc = s_acc;
        const int64_t cc = acc + 1, bonus = s_bonus;
        const int64_t cnt = s_words[kWordOutCount];
        for (uint32_t j = 0; j < acc; ++j)
            c.out_ring[(cnt + j) % c.ring] = static_cast<int32_t>(s_tok[j + 1]);
#else
        uint32_t acc = 0;
        while (acc + 1 < c.b_live && s_tok[acc + 1] == static_cast<int64_t>(t1[acc]))
            ++acc;
        if (c.accept_override != nullptr) {
            const int32_t f = c.accept_override[iter % c.n_override];
            acc = f < 0 ? 0u : min(static_cast<uint32_t>(f), c.b_live - 1);
        }
        const int64_t cc = acc + 1, bonus = t1[acc];
        const int64_t cnt = s_words[kWordOutCount];
        for (uint32_t j = 0; j < acc; ++j)
            c.out_ring[(cnt + j) % c.ring] = static_cast<int32_t>(s_tok[j + 1]);
#endif
        c.out_ring[(cnt + acc) % c.ring] = static_cast<int32_t>(bonus);
        c.words[kWordP] = p + cc;
        c.words[kWordAnchor] = bonus;
        c.words[kWordAccept] = acc;
        c.words[kWordCommit] = cc;
        c.words[kWordIter] = iter + 1;
        c.words[kWordOutCount] = cnt + cc;
        s_c = cc;
    }
#if !MK_DFLASH_SAMPLE
    if (tid < c.b_live) c.top1[tid] = t1[tid];
#endif
    ::ptx::cp_async_wait_group<0>();  // each thread re-reads only the rows it copied
    __syncthreads();
    const int64_t cc = s_c, pn = p + cc;
    // target: staged row k is position P + 1 + k = P' + (k + 1 - c')
    float4* tdst = reinterpret_cast<float4*>(c.cos_sin);
    for (uint32_t i = tid; i < n_t; i += kThreads) {
        const int64_t t = static_cast<int64_t>(i / tq) + 1 - cc;
        if (t >= 0 && t < kWindowRows) tdst[t * tq + i % tq] = s_tgt[i];
    }
    if (tid < kWindowRows) {
        c.loc_full[tid] = static_cast<int32_t>(pn + tid);
        c.loc_swa[tid] = static_cast<int32_t>(pn + tid);
    }
    if (tid == 0) c.kv_len[0] = static_cast<int32_t>(pn + kWindowRows);
    // draft: staged row k is position P + k: context row k if k < c', else block row k - c'; the masked context
    // rows [c', b_live) carry P' (row c')
    float4* ddst = reinterpret_cast<float4*>(c.dcos_sin);
    for (uint32_t i = tid; i < n_d; i += kThreads) {
        const uint32_t k = i / dq, q4 = i % dq;
        const float4 v = s_drf[i];
        if (k < cc) {
            ddst[(c.b_live + k) * dq + q4] = v;
        } else {
            if (k - cc < c.b_live) ddst[(k - cc) * dq + q4] = v;
            if (k == cc)
                for (uint32_t m = cc; m < c.b_live; ++m)
                    ddst[(c.b_live + m) * dq + q4] = v;
        }
    }
    if (tid < 2 * c.b_live) {
        const bool ctx = tid >= c.b_live;
        const uint32_t j = ctx ? tid - c.b_live : tid;
        const int64_t pos = ctx ? p + j : pn + j;
        c.dslot[tid] = ctx && j >= cc ? -1 : static_cast<int32_t>(pos & (c.dring - 1));
    }
    if (tid == 0) c.dkv_len[0] = static_cast<int32_t>(pn + c.b_live);
    __syncthreads();
    // every CTA read the epoch before it counted itself, so the advance is the launch's last act
    if (tid == 0) a.state[0] = epoch + 1;
}

template <uint32_t kM, uint32_t kStages, uint32_t kPack, bool kEvictFirst>
__launch_bounds__(256, 1) __global__
    void verify_accept(const __grid_constant__ DensePersistArgs args, const __grid_constant__ CUtensorMap tmap_w,
                       const __grid_constant__ CUtensorMap tmap_x, const __grid_constant__ AcceptSink::Args epi,
                       const __grid_constant__ AcceptPostArgs post
#if MK_DFLASH_TREE
                       ,
                       int32_t* const tree_table  // tree.cuh kTableWords
#endif
    ) {
    dense_persist_body<kM, kStages, kPack, kEvictFirst, AcceptSink>(args, tmap_w, tmap_x, epi);
#if MK_DFLASH_TREE
    accept_post(post, tree_table);
#else
    accept_post(post);
#endif
}

#if MK_DFLASH_SAMPLE
// sample_accept: the sampled accept (dflash/sample.cuh) on the verify graph's gathered fp32 softcapped logits, then the
// same commit and feeds. launch: grid = kCtas x live rows in clusters of kCtas, block 256, smem the slice's scaled
// logits + RadixSmem (>= the feeds' staging), PDL
__launch_bounds__(256, 1) __global__
    void sample_accept(const __grid_constant__ sample::Args sa, const __grid_constant__ AcceptPostArgs post) {
    sample::rows(sa);
    accept_post(post, sa);
}
#endif

}  // namespace mk::dflash

#ifndef MK_HOST
#if MK_DFLASH_SAMPLE
[[maybe_unused]] static constexpr auto kKernel = &mk::dflash::sample_accept;
#else
[[maybe_unused]] static constexpr auto kKernel =
    &mk::dflash::verify_accept<mk::cfg::kWindowM, mk::cfg::kStages, mk::cfg::kPack, mk::cfg::kEvictFirst>;
#endif
#if MK_DFLASH_TREE
[[maybe_unused]] static constexpr auto kCompact = &mk::dflash::tree::tree_compact<256>;
#endif
#endif
