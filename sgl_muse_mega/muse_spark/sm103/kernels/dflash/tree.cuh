#pragma once
// The token-tree window (-DMK_DFLASH_TREE=1, block 16): the verify checks a tree of 15 drafts instead of a next-layer
// phase.
//
// A DFlash block never reads its own drafts, so every branch comes out of the one lattice the head already scores
// (score[e][p][c]: slot e's candidate c after slot e - 1's candidate p). The head keeps each score row's top kFan
// successors with their log-probabilities (a softmax over those kFan scores), then grows the tree best-first: the 15
// nodes of highest path probability (the product along the path), so rows go where the draft is unsure. The rows are
// laid out in preorder with the rank-1 child first: the rank-1 next-layer phase from the anchor holds rows 1, 2, .., so
// a step that accepts along it moves nothing. Row r carries position P + depth(r) (the head writes its rope row) and
// writes its K / V at the window's slot of row r; the attention attention CTAs mask the window's keys by the row's
// ancestor mask. The accept follows the one path whose tokens are the target's argmax at their parent (siblings carry
// distinct candidates), and tree_compact then moves the path's K / V rows (every layer) and captured rows (the draft's
// context) to the slots of rows 1 .. accept, so the next step sees a contiguous history.
//
// The tree table (int32 [kTableWords]) follows the window's kv_len word in the target feeds (attn_kv_len + 1), so the
// layer images' attention CTAs read it through the pointer they already hold (attention.cuh kTreeDepthWord /
// kTreeMaskWord).
#include <cstdint>
#include <cstdio>

#include "dflash/common.cuh"

namespace mk::dflash::tree {

inline constexpr uint32_t kRows = 16;    // window rows: the anchor and 15 nodes
inline constexpr uint32_t kFan = 4;      // successors kept per score row
inline constexpr uint32_t kDepth = 0;    // [16] the row's depth (the anchor 0)
inline constexpr uint32_t kMask = 16;    // [16] bit j: window row j is the row or one of its ancestors
inline constexpr uint32_t kParent = 32;  // [16] the parent row (the anchor -1)
inline constexpr uint32_t kPath = 48;    // [16] the accept's leaf row, then the accepted row at depth 1 .. accept
inline constexpr uint32_t kSlotFull =
    64;  // [16] the window rows' slots in the full pool (verify_accept's copy of loc_full)
inline constexpr uint32_t kSlotSwa = 80;  // [16] .. and in the SWA pool
inline constexpr uint32_t kTableWords = 96;
inline constexpr uint32_t kFrontier = 64;  // 4 + 15 x (kFan - 1) live candidates at most
static_assert(kFan == 4 && kFrontier >= kFan + (kRows - 1) * (kFan - 1), "the frontier holds every live candidate");

struct TreeArgs {
    int32_t* table;              // kTableWords int32
    int32_t* fan_idx;            // [gamma][16][kFan] each score row's top successors (candidate indices)
    float* fan_lp;               // [gamma][16][kFan] their log-probabilities within the row's top kFan
    float* cos_sin;              // [16][d] the verify window's rope rows
    const float* cos_sin_cache;  // [cos_sin_rows][d]
    int64_t cos_sin_rows;
    uint32_t d;
};

// ---- the head: the fan of slot e's score rows (in the slot's lattice CTA) ----

// Thread t = p x 16 + c holds score[e][p][c] (v). Row p's top kFan successors in torch.argmax's order and their
// log-probabilities lp_k = (v_k - v_0) - log(sum_j exp(v_j - v_0)) (fp32, j in order) go to row (e, p) of the fan.
__device__ __forceinline__ void fan_row(const TreeArgs& tr, uint32_t e, float v) {
    const uint32_t t = threadIdx.x, p = t / kTopK, c = t % kTopK;
    float fv[kFan];
    int32_t fi[kFan];
    bool taken = false;
#pragma unroll
    for (uint32_t k = 0; k < kFan; ++k) {
        float bv = taken ? -INFINITY : v;
        int32_t bi = taken ? INT32_MAX : static_cast<int32_t>(c);
#pragma unroll
        for (uint32_t o = 8; o >= 1; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o, 16);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o, 16);
            argmax_take(bv, bi, ov, oi);
        }
        fv[k] = bv;
        fi[k] = bi;
        taken |= bi == static_cast<int32_t>(c);
    }
    if (c == 0) {
        float s = 0.f;
#pragma unroll
        for (uint32_t k = 0; k < kFan; ++k)
            s = __fadd_rn(s, expf(__fsub_rn(fv[k], fv[0])));
        const float lse = logf(s);
        const uint32_t row = (e * kTopK + p) * kFan;
#pragma unroll
        for (uint32_t k = 0; k < kFan; ++k) {
            tr.fan_idx[row + k] = fi[k];
            tr.fan_lp[row + k] = __fsub_rn(__fsub_rn(fv[k], fv[0]), lse);
        }
    }
}

// ---- the head: the tree (the last lattice CTA, after every slot's fan landed) ----

struct WalkSmem {
    int32_t fidx[kTopK * kTopK * kFan];  // the fan, up to 16 slots
    float flp[kTopK * kTopK * kFan];
    int32_t cand[kTopK * kTopK];
    // the frontier: a candidate node = (path log-probability, slot, candidate index, parent row, rank, order)
    float q_lp[kFrontier];
    int32_t q_slot[kFrontier], q_c[kFrontier], q_parent[kFrontier], q_rank[kFrontier], q_seq[kFrontier];
    // the rows in selection order, then in preorder
    float lp[kRows];
    int32_t slot[kRows], c[kRows], parent[kRows], depth[kRows], rank[kRows], order[kRows];
    uint64_t key[kRows];
    int32_t o_parent[kRows], o_depth[kRows], o_mask[kRows];
    int64_t o_tok[kRows];
};

// The tree from the fan and the candidates: verify_ids[16], drafts[15] (rows 1 .. 15), the table's depth / mask /
// parent, and the window's rope rows (row r: position P + depth(r)). Every rank runs it on bitwise-equal inputs.
__device__ __forceinline__ void walk(WalkSmem& sm, const TreeArgs& tr, const int32_t* cand, uint32_t gamma,
                                     const int64_t* words, int64_t* drafts, int64_t* verify_ids) {
    const uint32_t t = threadIdx.x, lane = t % 32;
    for (uint32_t i = t; i < gamma * kTopK * kFan; i += kThreads) {
        sm.fidx[i] = __ldcg(tr.fan_idx + i);
        sm.flp[i] = __ldcg(tr.fan_lp + i);
    }
    if (t < gamma * kTopK) sm.cand[t] = __ldcg(cand + t);
    __syncthreads();
    if (t < 32) {
        // best-first growth: the frontier starts with the anchor's fan (slot 0, predecessor row 0)
        for (uint32_t i = lane; i < kFrontier; i += 32)
            sm.q_seq[i] = -1;  // free
        if (lane < kFan) {
            sm.q_lp[lane] = sm.flp[lane];
            sm.q_slot[lane] = 0;
            sm.q_c[lane] = sm.fidx[lane];
            sm.q_parent[lane] = 0;
            sm.q_rank[lane] = static_cast<int32_t>(lane) + 1;
            sm.q_seq[lane] = static_cast<int32_t>(lane);
        }
        if (lane == 0) sm.depth[0] = 0;  // the anchor (row 0 of selection order)
        __syncwarp();
        int32_t seq = kFan;
        for (uint32_t r = 1; r < kRows; ++r) {
            // the frontier's best: the larger path log-probability, then the earlier push (argmax's lower index)
            float bv = -INFINITY;
            int32_t bi = INT32_MAX;
#pragma unroll
            for (uint32_t h = 0; h < kFrontier / 32; ++h) {
                const uint32_t i = lane + 32 * h;
                if (sm.q_seq[i] >= 0)
                    argmax_take(bv, bi, sm.q_lp[i], sm.q_seq[i] * kFrontier + static_cast<int32_t>(i));
            }
            warp_argmax(bv, bi);
            if (bi == INT32_MAX) ::ptx::trap();  // an empty frontier: a lattice row without kFan successors
            const uint32_t w = static_cast<uint32_t>(bi) % kFrontier;
            const int32_t e = sm.q_slot[w], cw = sm.q_c[w];
            if (lane == 0) {
                sm.lp[r] = sm.q_lp[w];
                sm.slot[r] = e;
                sm.c[r] = cw;
                sm.parent[r] = sm.q_parent[w];
                sm.depth[r] = e + 1;
                sm.rank[r] = sm.q_rank[w];
                sm.q_seq[w] = -1;
            }
            __syncwarp();
            // its fan joins the frontier (slot e + 1, predecessor cw) unless the row sits on the last slot
            if (static_cast<uint32_t>(e) + 1 < gamma) {
                // free entries, ranked by index: lane k < kFan takes the k-th free one
                const uint32_t free0 = __ballot_sync(0xffffffffu, sm.q_seq[lane] < 0);
                const uint32_t free1 = __ballot_sync(0xffffffffu, sm.q_seq[lane + 32] < 0);
                if (lane < kFan) {
                    uint32_t k = lane, slot_i;
                    const uint32_t n0 = __popc(free0);
                    if (k < n0) {
                        uint32_t m = free0;
                        for (uint32_t j = 0; j < k; ++j)
                            m &= m - 1;
                        slot_i = __ffs(m) - 1;
                    } else {
                        uint32_t m = free1;
                        for (uint32_t j = n0; j < k; ++j)
                            m &= m - 1;
                        slot_i = 32 + __ffs(m) - 1;
                    }
                    const uint32_t f = ((e + 1) * kTopK + cw) * kFan + lane;
                    sm.q_lp[slot_i] = __fadd_rn(sm.lp[r], sm.flp[f]);
                    sm.q_slot[slot_i] = e + 1;
                    sm.q_c[slot_i] = sm.fidx[f];
                    sm.q_parent[slot_i] = static_cast<int32_t>(r);
                    sm.q_rank[slot_i] = static_cast<int32_t>(lane) + 1;
                    sm.q_seq[slot_i] = seq + static_cast<int32_t>(lane);
                }
                seq += kFan;
            }
            __syncwarp();
        }
        // preorder, rank-1 child first: key = the rank path, 3 bits per level from the most significant (a prefix
        // sorts before its extensions); the parents come first in selection order, so one pass builds the keys
        if (lane == 0) {
            sm.key[0] = 0;
            for (uint32_t r = 1; r < kRows; ++r)
                sm.key[r] = sm.key[sm.parent[r]] | (static_cast<uint64_t>(sm.rank[r])
                                                    << (3 * (kRows - 1 - static_cast<uint32_t>(sm.depth[r]))));
        }
        __syncwarp();
        if (lane < kRows) {
            uint32_t o = 0;
            for (uint32_t j = 0; j < kRows; ++j)
                o += sm.key[j] < sm.key[lane];
            sm.order[lane] = static_cast<int32_t>(o);
        }
        __syncwarp();
        if (lane < kRows) {
            const uint32_t o = sm.order[lane];
            sm.o_parent[o] = lane == 0 ? -1 : sm.order[sm.parent[lane]];
            sm.o_depth[o] = sm.depth[lane];
            sm.o_tok[o] =
                lane == 0 ? words[kWordAnchor] : static_cast<int64_t>(sm.cand[sm.slot[lane] * kTopK + sm.c[lane]]);
        }
        __syncwarp();
        if (lane == 0) {
            sm.o_mask[0] = 1;
            for (uint32_t r = 1; r < kRows; ++r)
                sm.o_mask[r] = sm.o_mask[sm.o_parent[r]] | (1 << r);
        }
        __syncwarp();
        if (lane < kRows) {
            verify_ids[lane] = sm.o_tok[lane];
            if (lane >= 1 && lane <= gamma) drafts[lane - 1] = sm.o_tok[lane];
            tr.table[kDepth + lane] = sm.o_depth[lane];
            tr.table[kMask + lane] = sm.o_mask[lane];
            tr.table[kParent + lane] = sm.o_parent[lane];
        }
    }
    __syncthreads();
    // the window's rope rows: row r at position P + depth(r) (P = the anchor's)
    const int64_t pos = words[kWordP];
    if (pos + kRows > tr.cos_sin_rows) {
        if (t == 0) printf("dflash tree walk: P %lld past the rope table\n", static_cast<long long>(pos));
        ::ptx::trap();
    }
    const uint32_t q4 = tr.d / 4;
    const float4* src = reinterpret_cast<const float4*>(tr.cos_sin_cache);
    float4* dst = reinterpret_cast<float4*>(tr.cos_sin);
    for (uint32_t i = t; i < kRows * q4; i += kThreads)
        dst[i] = src[(pos + sm.o_depth[i / q4]) * q4 + i % q4];
}

// ---- the accept (verify_accept's last CTA, warp 0) ----

// The accepted path's leaf, in every lane: row r matches iff its token is the target's argmax at its parent (t1), and
// is accepted iff it and every ancestor match. The accepted rows form one next-layer phase from the anchor (siblings
// carry distinct candidates), and rows are in preorder, so the leaf is the largest accepted row. tab = the table
// (smem).
__device__ __forceinline__ uint32_t accept_leaf(const int32_t* tab, const int64_t* tok, const int32_t* t1) {
    const uint32_t lane = threadIdx.x % 32;
    const bool m = lane >= 1 && lane < kRows && tok[lane] == static_cast<int64_t>(t1[tab[kParent + lane]]);
    const uint32_t match = __ballot_sync(0xffffffffu, m) | 1u;
    const bool on = lane < kRows && (static_cast<uint32_t>(tab[kMask + lane]) & ~match) == 0;
    return 31 - __clz(__ballot_sync(0xffffffffu, on));
}

// ---- tree_compact: the accepted path's rows to the slots of rows 1 .. accept ----

inline constexpr uint32_t kCapCtas = 4;  // CTAs per capture buffer (a captured row is 32 KB; a K / V row 256-512 B)

struct CompactArgs {
    const int64_t* words;   // after the accept: the commit
    const int32_t* table;   // the path and the window's slots
    const uint64_t* rows;   // [n_swa + n_full + n_cap] row-major buffers: the SWA layers' K and V pools, the full
                            // layers', then the capture buffers
    uint32_t n_swa;         // SWA pools: row = table[kSlotSwa + window row]
    uint32_t n_full;        // full pools: row = table[kSlotFull + window row]
    uint32_t n_cap;         // capture buffers: row = window row (kCapCtas CTAs each, a quarter of the row's bytes)
    uint32_t kv_row_bytes;  // multiples of 16
    uint32_t cap_row_bytes;
};

// One CTA per pool, kCapCtas per capture buffer: row i of 1 .. accept takes row path[i]'s bytes from before the launch.
// A source path[i] >= i can be a later destination (path 2, 3, 4 moves row 3 into row 2 after row 2 into row 1), so
// each thread owns 16-B chunk columns: it loads a column's sources for every accepted depth, then stores them.
// The last launch of an iteration's graph: the next iteration's entry reads the captured rows without a grid wait
// (dflash/head.cu), so it must not follow tree_compact in one PDL next-layer phase.
template <uint32_t kNumThreads>
__global__ __launch_bounds__(kNumThreads, 1) void tree_compact(const __grid_constant__ CompactArgs a) {
    __shared__ int32_t s_tab[kTableWords];
    ::ptx::griddepcontrol_wait();  // the path and the commit are verify_accept's
    if (threadIdx.x < kTableWords) s_tab[threadIdx.x] = a.table[threadIdx.x];
    __syncthreads();
    const uint32_t b = blockIdx.x, n_kv = a.n_swa + a.n_full;
    const bool cap = b >= n_kv;
    const uint32_t row_q = (cap ? a.cap_row_bytes : a.kv_row_bytes) / 16;  // 16-B chunks per row
    const uint32_t q = cap ? row_q / kCapCtas : row_q;                     // this CTA's chunk columns
    const uint32_t q0 = cap ? (b - n_kv) % kCapCtas * q : 0;
    const int32_t* slot = s_tab + (b < a.n_swa ? kSlotSwa : kSlotFull);
    const auto row = [&](int32_t r) -> int64_t { return cap ? r : slot[r]; };
    uint4* buf = reinterpret_cast<uint4*>(a.rows[cap ? n_kv + (b - n_kv) / kCapCtas : b]);
    const int32_t acc = static_cast<int32_t>(a.words[kWordCommit] - 1);
    for (uint32_t k = q0 + threadIdx.x; k < q0 + q; k += kNumThreads) {
        uint4 v[kRows - 1];
#pragma unroll
        for (int32_t i = 1; i < static_cast<int32_t>(kRows); ++i)
            if (i <= acc && s_tab[kPath + i] != i) v[i - 1] = buf[row(s_tab[kPath + i]) * row_q + k];
#pragma unroll
        for (int32_t i = 1; i < static_cast<int32_t>(kRows); ++i)
            if (i <= acc && s_tab[kPath + i] != i) buf[row(i) * row_q + k] = v[i - 1];
    }
}

}  // namespace mk::dflash::tree
