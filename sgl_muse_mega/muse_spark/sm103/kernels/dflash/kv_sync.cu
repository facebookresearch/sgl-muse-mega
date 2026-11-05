// draft_kv_sync: SGLang's draft KV pool kept in step with the fused iteration. The draft writes a committed position's
// context K / V into its private ring one iteration after the commit (the next iteration's qkv and layer launches);
// this launch copies those rows into SGLang's draft pool at the slots SGLang allocated for the positions, so a request
// that resumes on a prefix-cache hit, or a step SGLang's worker runs, drafts from the context the fused steps made.
//
// state (int64 [1 + 16], set at a request's start): the position up to which the pool holds the request's context, then
// the window slots of the iteration that committed the next positions (slot of position state[0] + i at 1 + i).
// lag 1, an iteration's last launch: the ring holds the positions below P - commit (the last commit's rows arrive with
// the next iteration's draft); the window's slots replace the saved ones. lag 0, after a draft that ran alone (the
// last fused step of a request): the ring holds the positions below P.
//
// launch: grid 1, block 256, PDL

#include "dflash/common.cuh"

namespace mk::dflash {

inline constexpr uint32_t kSyncRows = 16;  // the window's rows: a commit moves at most this many positions

struct KvSyncArgs {
    const int64_t* words;   // the step words
    const int64_t* window;  // [16] this iteration's window slots (positions P .. P + 15, SGLang's full-pool ids)
    int64_t* state;         // [1 + 16]
    const uint64_t* ring;   // [n_buf] the draft's K / V rings, bf16 [ring_rows][row_bytes / 2]
    const uint64_t* pool;   // [n_buf] SGLang's draft pool buffers in the same order, [slots][row_bytes / 2]
    uint32_t n_buf;
    uint32_t row_bytes;  // a multiple of 16
    uint32_t ring_rows;  // a power of two
    uint32_t lag;
};

__global__ __launch_bounds__(kThreads, 1) void draft_kv_sync(const __grid_constant__ KvSyncArgs a) {
    __shared__ int64_t s_slot[kSyncRows];
    __shared__ int64_t s_from;
    __shared__ uint32_t s_n;
    ::ptx::griddepcontrol_wait();  // the ring rows and the step words are the iteration's
    const uint32_t t = threadIdx.x;
    if (t == 0) {
        const int64_t from = a.state[0], to = a.words[kWordP] - (a.lag ? a.words[kWordCommit] : 0);
        if (to < from || to - from > kSyncRows) {
            printf("draft_kv_sync: the pool holds positions < %lld, the ring < %lld\n", static_cast<long long>(from),
                   static_cast<long long>(to));
            ::ptx::trap();
        }
        s_from = from;
        s_n = static_cast<uint32_t>(to - from);
    }
    if (t < kSyncRows) s_slot[t] = a.state[1 + t];
    __syncthreads();
    const int64_t from = s_from;
    const uint32_t n = s_n, q = a.row_bytes / 16;
    for (uint32_t i = t; i < a.n_buf * n * q; i += kThreads) {
        const uint32_t k = i % q, r = i / q % n, b = i / (q * n);
        const uint4* src = reinterpret_cast<const uint4*>(a.ring[b]) + ((from + r) & (a.ring_rows - 1)) * q + k;
        reinterpret_cast<uint4*>(a.pool[b])[s_slot[r] * q + k] = *src;
    }
    // each thread rewrites only the state words it read above
    if (t == 0) a.state[0] = from + n;
    if (a.lag && t < kSyncRows) a.state[1 + t] = a.window[t];
}

}  // namespace mk::dflash

#ifndef MK_HOST
[[maybe_unused]] static constexpr auto kKernel = &mk::dflash::draft_kv_sync;
#endif
