#pragma once
// The PTX wrappers, by purpose: device asserts, shared-memory addresses, global loads / reductions / atomics, MUFU
// math, packing conversions, fences / the cluster barrier / elect / PDL, mbarriers, TMA, tcgen05 and its descriptors,
// a cluster's peer CTAs, row loads / stores by coherence, NVLS multimem, and the waits on counter words.

#include <cuda/std/cstdint>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>
#include <cstdio>

// Device asserts: a failed condition traps (no message: the images carry no assert strings).

namespace ptx {

static __device__ __forceinline__ void device_assert(const bool cond) {
    if (!cond) asm("trap;");
}

// The bounded waits' and the bounds checks' unconditional exit.
static __device__ __forceinline__ void trap() {
    asm volatile("trap;");
}

// Shared-memory addresses for inline PTX: `.shared` instructions take a 32-bit offset in the shared window (not a
// generic pointer), and `.shared::cluster` ones a cluster-mapped offset that names the target CTA.

// Generic pointer -> 32-bit `.shared` offset.
template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Local `.shared` offset -> the `.shared::cluster` offset of the same bytes in CTA `cta_rank`'s smem (peer mbarrier
// arrives, st.async to a peer, the 2-CTA TMA's barrier).
static __device__ __forceinline__ uint32_t mapa_shared_cluster(uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

// Global-memory loads, reductions and atomics with explicit scope and strength: the cross-CTA and cross-rank counters
// (tickets, readiness words) and the L2-hinted loads; names and signatures in the style of DeepGEMM's PTX wrappers.

// prefetch.global.L2: a fire-and-forget DRAM -> L2 line fetch (no register, no dependency).
static __device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

// A 16-B load carrying an L2 cache-policy operand (createpolicy, below).
static __device__ __forceinline__ uint4 ld_global_l2_hint_b128(const void* ptr, uint64_t policy) {
    uint4 v;
    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr), "l"(policy));
    return v;
}

// The sys-scope acquire load of a signed counter a peer rank releases (a `.gpu` acquire would not pair with it).
static __device__ __forceinline__ int ld_acq_sys_s32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// Plain gpu-scope reduce-add: a counter whose ordering comes from a surrounding fence.
static __device__ __forceinline__ void red_add_b32(uint32_t* ptr, uint32_t value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

// Release reduce-add: publishes everything the thread wrote before it (pairs with a gpu-scope acquire of the count).
static __device__ __forceinline__ void red_add_rel_b32(uint32_t* ptr, uint32_t value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

// The 64-bit release reduce-add (the K-window variant's per-tile pooled-ready words).
static __device__ __forceinline__ void red_add_rel_u64(uint64_t* ptr, uint64_t value) {
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(ptr), "l"(value) : "memory");
}

// Release atomic add returning the old value.
static __device__ __forceinline__ uint32_t atomic_add_rel_b32(uint32_t* ptr, uint32_t value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// Relaxed atomic add: a bare claim counter that guards no data (no fence, so the issue does not stall).
static __device__ __forceinline__ uint32_t atomic_add_relaxed_b32(uint32_t* ptr, uint32_t value) {
    uint32_t ret;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

static __device__ __forceinline__ uint64_t ld_acq_fenced_u64(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Relaxed poll of a released word (no acquire on every poll; the reader fences once after its loop exits).
static __device__ __forceinline__ uint32_t ld_relaxed_b32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// fence.acq_rel.gpu: ONE release covering a burst of relaxed red_add_b32 (each red_add_rel_b32 pays its own release).
static __device__ __forceinline__ void fence_acq_rel_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

// fence.acquire.gpu: the loop-exit half of a relaxed poll (or the acquire half of a release counter's last arrival).
static __device__ __forceinline__ void fence_acquire_gpu() {
    asm volatile("fence.acquire.gpu;" ::: "memory");
}

// fence.release.sys: orders the thread's earlier writes ahead of its following relaxed sys-scope stores.
static __device__ __forceinline__ void fence_release_sys() {
    asm volatile("fence.release.sys;");
}

// gpu-scope acq_rel add returning the old value (a fan-in whose last arriver reads every arriver's writes).
static __device__ __forceinline__ uint32_t atomic_add_acq_rel_b32(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// A relaxed gpu-scope store of a word other CTAs poll (a counter's reset, an epoch).
static __device__ __forceinline__ void st_relaxed_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Relaxed sys-scope words a peer rank polls or writes (one 4-B or 8-B access is single-copy atomic).
static __device__ __forceinline__ void st_relaxed_sys_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v));
}
static __device__ __forceinline__ uint64_t ld_relaxed_sys_u64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void st_relaxed_sys_u64(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// A plain 4-B store pinned against elision.
static __device__ __forceinline__ void st_global_b32(uint32_t* p, uint32_t v) {
    asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// MUFU wrappers.

// rcp.approx.ftz.f32: one MUFU.RCP (the softmax's 1 / l normalize).
static __device__ __forceinline__ float rcp_approx_ftz_f32(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// The fixed-rounding arithmetic the kernels pin to an opcode: ex2.approx (with .ftz: one MUFU.EX2, no denormal
// fixup), div.full, sqrt.rn, and fma.rn.f32.bf16 (a bf16 square accumulated in fp32 with one rounding).
static __device__ __forceinline__ float ex2_approx_ftz_f32(const float& x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ __forceinline__ float ex2_approx_f32(const float& x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ __forceinline__ float div_full_f32(const float& a, const float& b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static __device__ __forceinline__ float sqrt_rn_f32(const float& x) {
    float y;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
static __device__ __forceinline__ void fma_sq_rn_f32_bf16(float& acc, const uint16_t& x) {
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;" : "+f"(acc) : "h"(x));
}

// rsqrt.approx.f32: the reference norm kernels' opcode (bitwise; not __frsqrt_rn).
static __device__ __forceinline__ float rsqrt_approx(const float& x) {
    float y;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// redux.sync.max.f32: the warp's largest value, one of the lanes' values bit for bit.
static __device__ __forceinline__ float redux_max_f32(float v, uint32_t mask) {
    float r;
    asm("redux.sync.max.f32 %0, %1, %2;" : "=f"(r) : "f"(v), "r"(mask));
    return r;
}

// Packing conversions: two fp32 values into one packed pair of a narrow format in a single PTX `cvt`
// (`cvt_pack_f32x2_to<ptx::e2m1>(a, b)` = "cvt + pack f32x2 to e2m1x2"). Used by the MoE phase's NVFP4 h codes and
// their e4m3 block scales.
//
// Byte order: `cvt.<fmt>x2.f32 d, a, b` puts cvt(a) in d's UPPER half and cvt(b) in its LOWER half, so for cells
// destined to adjacent slots [i, i + 1] pass (cell[i + 1], cell[i]).
//   e4m3: .satfinite is mandatory; |x| > 448 saturates (sign kept), NaN -> 0x7F.
//   e2m1: values {0, 0.5, 1, 1.5, 2, 3, 4, 6} (and negatives); NaN -> 0x7 (no NaN code). A .b8 result needs an inline
//         `.reg .b8` and a `cvt.u32.u8` to reach a C++ register (there is no 8-bit asm constraint).

// Destination-format tags: packed2_t is the container of one packed pair.
struct e4m3 {
    using packed2_t = uint16_t;
};
struct e2m1 {
    using packed2_t = uint8_t;
};

template <typename Dst>
static __device__ __forceinline__ typename Dst::packed2_t cvt_pack_f32x2_to(float a, float b);

template <>
__device__ __forceinline__ uint16_t cvt_pack_f32x2_to<e4m3>(float a, float b) {
    uint16_t d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(a), "f"(b));
    return d;
}

template <>
__device__ __forceinline__ uint8_t cvt_pack_f32x2_to<e2m1>(float a, float b) {
    uint32_t d;
    asm volatile(
        "{ .reg .b8 v;"
        "  cvt.rn.satfinite.e2m1x2.f32 v, %1, %2;"
        "  cvt.u32.u8 %0, v;"
        "}"
        : "=r"(d)
        : "f"(a), "f"(b));
    return uint8_t(d & 0xffu);
}

// Two fp32 -> one e2m1x2 byte (b in the low nibble).
static __device__ __forceinline__ uint32_t cvt_pack_f32_to_e2m1x2(const float& a, const float& b) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.u32.u8 %0, byte0;\n"
        "}"
        : "=r"(out)
        : "f"(a), "f"(b));
    return out;
}

// Proxy fences, the cluster barrier, elect.sync and the PDL (programmatic dependent launch) controls.

// fence.proxy.async.shared::cta: generic-proxy smem writes made visible to the async proxy (required before a TMA
// store, or a UMMA operand read, of smem written with ordinary stores).
static __device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// fence.proxy.async.global: the global-memory form (generic writes later read by the async proxy).
static __device__ __forceinline__ void fence_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}
// fence.proxy.async over every state space.
static __device__ __forceinline__ void fence_async() {
    asm volatile("fence.proxy.async;" ::: "memory");
}

// barrier.cluster: every thread of every CTA of the cluster. cluster_sync is arrive + wait (the default acq/rel
// semantics: smem writes, mbarrier inits and TMEM allocations before it are visible to the peers after it); the split
// halves let a CTA interleave work between them.
static __device__ __forceinline__ void cluster_barrier_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;");
}
static __device__ __forceinline__ void cluster_barrier_wait() {
    asm volatile("barrier.cluster.wait.aligned;");
}
static __device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
}

// elect.sync: true on exactly one lane of the (converged) warp, for single-issuer sites (mbarrier init, TMA, MMA).
static __device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}\n"
        : "=r"(pred));
    return pred != 0;
}

// PDL (with the launch attribute cudaLaunchAttributeProgrammaticStreamSerialization). griddepcontrol.wait blocks until
// the predecessor grid's memory is visible (a no-op without a PDL predecessor): place it before the first load of a
// predecessor's output. launch_dependents lets the successor grid start launching (implicit at CTA exit otherwise).
static __device__ __forceinline__ void griddepcontrol_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
static __device__ __forceinline__ void griddepcontrol_launch_dependents() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// Aligned named barrier `bar.sync id, count` (ids 1..15; 0 is __syncthreads).
static __device__ __forceinline__ void sync_aligned(uint32_t barrier_idx, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(barrier_idx), "r"(num_threads) : "memory");
}

// Warp index broadcast from lane 0: ptxas can prove it uniform and keeps the role and address arithmetic on the
// uniform datapath (threadIdx.x >> 5 alone leaves it on the vector pipes).
static __device__ __forceinline__ uint32_t warp_idx_sync() {
    return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0);
}

static __device__ __forceinline__ uint32_t lane_idx() {
    uint32_t lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(lid));
    return lid;
}

// v through an opaque zero add: what derives from the result is not hoisted out of the loop that issues it
static __device__ __forceinline__ uint32_t opaque_u32(uint32_t v) {
    uint32_t r;
    asm volatile("add.u32 %0, %1, 0;" : "=r"(r) : "r"(v));
    return r;
}

// elect.sync capturing the leader lane: compiles to a bare ELECT (the `_|p` form gets wrapped in WARPSYNC blocks).
static __device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

// mbarrier wrappers (parity waits only).
//
// Parity: after mbar_init the barrier is in phase 0; every completed phase (count arrivals + the expected tx bytes)
// flips it, and try_wait.parity(p) returns once the barrier's current phase parity differs from p. A consumer waiting
// for a producer's first completion starts at p = 0; a producer waiting for a slot nobody has used yet starts at p = 1
// (the first wait passes). Getting the start wrong deadlocks on the second phase.
//
// Visibility: try_wait's default .acquire makes the cp.async.bulk (TMA) bytes the barrier tracked visible to the
// waiting thread's generic loads, so no fence.proxy.async is needed after a TMA load + wait.
//
// The no-hint try_wait loop compiles to a full-rate TRYWAIT + BRA spin (no suspend); only the timeHint form parks
// the thread. The spin is right for the short operand-ring waits.

// Retire a barrier so its bytes can be reused by a later phase of the launch: a TMA landing on the bytes of a live
// barrier corrupts it, so a body that hands its smem on invalidates its barriers first.
static __device__ __forceinline__ void mbar_inval(uint64_t* bar) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}

static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}

// Barrier inits made visible to the async proxy (the TMA's complete_tx) and to the rest of the cluster.
static __device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

static __device__ __forceinline__ uint64_t mbar_arrive(uint64_t* bar) {
    uint64_t state;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)));
    return state;
}

// Release arrival: the thread's prior shared-memory writes are visible to a same-CTA acquiring waiter.
static __device__ __forceinline__ void mbar_arrive_release(uint64_t* bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;" ::"r"(to_shared(bar)) : "memory");
}

// Release arrival on the barrier at the same offset in CTA `cta_rank`'s smem, at cluster scope: a peer that acquires
// it at cluster scope then sees this thread's prior shared stores (e.g. through ld.shared::cluster).
static __device__ __forceinline__ void mbar_arrive_cluster_release_cluster(uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0], 1;" ::"r"(mapped) : "memory");
}

// Arrive and add `bytes` to the phase's expected transaction count (the TMA loads that complete it).
static __device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}

// Wait for the phase after `parity` (looping: try_wait may return early).
static __device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
        "r"(parity)
        : "memory");
}

// The cluster-scope acquire wait: for a local barrier completed by peer CTAs' arrivals or a cluster-scope async op.
static __device__ __forceinline__ void mbar_wait_parity_acquire_cluster(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
        "p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
        "r"(parity)
        : "memory");
}

// One non-blocking try_wait.parity probe (default .acquire: it orders the TMA bytes the barrier tracked).
static __device__ __forceinline__ bool mbar_try_wait_parity(uint64_t* bar, uint32_t parity) {
    uint32_t ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}\n"
        : "=r"(ready)
        : "r"(to_shared(bar)), "r"(parity)
        : "memory");
    return ready != 0;
}

// Parity waits by try_wait polling: the local (cta) form and the cluster-scope acquire form (for a phase landed by
// a peer's st.async payload or arrivals).
static __device__ __forceinline__ void mbar_wait_parity_poll(uint64_t* bar, uint32_t parity) {
    while (!mbar_try_wait_parity(bar, parity)) {
    }
}

static __device__ __forceinline__ void mbar_wait_parity_acquire_cluster_poll(uint64_t* bar, uint32_t parity) {
    for (;;) {
        uint32_t done;
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t}\n"
            : "=r"(done)
            : "r"(to_shared(bar)), "r"(parity)
            : "memory");
        if (done) return;
    }
}

// cp.async, the non-bulk async copy: 16-B global -> shared pieces in per-thread commit groups, waited by count.
static __device__ __forceinline__ void cp_async_cg_16(uint32_t dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_smem), "l"(src_gmem) : "memory");
}
static __device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
// until at most N of this thread's committed groups are pending
template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// TMA (cp.async.bulk) wrappers: 1D bulk copies, 2D tiled loads / stores through a tensor map, the 2D gather4 row
// gather, the 3D L2 prefetch, and the L2 cache-policy operand.
//
// Completion: loads complete on an mbarrier (arm it with mbar_arrive_expect_tx(bar, bytes), wait on its parity; the
// default .acquire wait makes the bytes visible, no proxy fence needed). Stores complete through bulk groups:
// fence_async_smem() after the threads' smem writes (generic -> async proxy), the store, tma_store_commit(), then a
// wait_group(.read) before the source smem is reused.
// Coordinates: the tensor map's dim 0 is the innermost (stride-1) axis, so a load of a row-major [rows, cols] tile
// takes (x = column offset, y = row offset).
// Alignment: 1D bulk sizes and both endpoints are multiples of 16 B; swizzled tile destinations are 128-B aligned
// (1024-B for the 128B swizzle the MMA operands use).

// Fetch a tensor map descriptor (a __grid_constant__ parameter) ahead of its first TMA, off the critical path.
static __device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// 1D bulk copy global -> shared::cta, completing on `bar`.
static __device__ __forceinline__ void cp_async_bulk_load(uint32_t dst_smem, const void* src_gmem, uint32_t bytes,
                                                          uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"(dst_smem),
        "l"(src_gmem), "r"(bytes), "r"(to_shared(bar))
        : "memory");
}

// 1D bulk copy shared::cta -> global (bulk-group completion; fence_async_smem() first).
static __device__ __forceinline__ void cp_async_bulk_store(void* dst_gmem, uint32_t src_smem, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_gmem), "r"(src_smem),
                 "r"(bytes)
                 : "memory");
}

// 2D tile load global -> shared::cta through a tensor map, completing on `bar`.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load(uint32_t dst_smem, const CUtensorMap* tmap,
                                                                    int32_t x, int32_t y, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
        "l"(tmap), "r"(x), "r"(y), "r"(to_shared(bar))
        : "memory");
}

// 2D tile store shared::cta -> global (bulk-group completion; fence_async_smem() first).
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_store(uint32_t src_smem, const CUtensorMap* tmap,
                                                                     int32_t x, int32_t y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];" ::"l"(tmap),
        "r"(x), "r"(y), "r"(src_smem)
        : "memory");
}

// Wait until at most N bulk groups still READ their smem source (the source-reuse gate; global visibility needs the
// full wait_group).
template <int N>
static __device__ __forceinline__ void cp_async_bulk_wait_group_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
// Wait until at most N bulk groups are pending (their writes complete and visible).
template <int N>
static __device__ __forceinline__ void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

// gather4: four arbitrary rows {row0..row3} (one shared start column) of a 2D tensor into four consecutive smem rows,
// completing on `bar` with 4 x the row box bytes. The tensor map's dim-1 box is 1.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_gather4(uint32_t dst_smem, const CUtensorMap* tmap,
                                                                       int32_t col, int32_t row0, int32_t row1,
                                                                       int32_t row2, int32_t row3, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst_smem),
        "l"(tmap), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(to_shared(bar))
        : "memory");
}

// L2 prefetch of the 3D box a later tile load fetches (hint only: no destination, no completion).
static __device__ __forceinline__ void cp_async_bulk_prefetch_tensor_3d_l2(const CUtensorMap* tmap, int32_t x,
                                                                           int32_t y, int32_t z) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(tmap), "r"(x), "r"(y),
                 "r"(z)
                 : "memory");
}

// An L2 cache-policy operand (createpolicy.fractional) for the `L2::cache_hint` forms: evict-first for bytes read once
// (the weight streams), evict-last for bytes that must outlive the streams around them.
enum class L2EvictPolicy : int { NORMAL = 0, EVICT_FIRST = 1, EVICT_LAST = 2 };

template <L2EvictPolicy Policy>
static __device__ __forceinline__ uint64_t make_l2_cache_policy(float fraction = 1.0f) {
    static_assert(Policy == L2EvictPolicy::EVICT_FIRST || Policy == L2EvictPolicy::EVICT_LAST,
                  "make_l2_cache_policy: only EVICT_FIRST / EVICT_LAST supported");
    uint64_t cookie;
    if constexpr (Policy == L2EvictPolicy::EVICT_FIRST) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n" : "=l"(cookie) : "f"(fraction));
    } else {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n" : "=l"(cookie) : "f"(fraction));
    }
    return cookie;
}

// Close this thread's bulk group of prior bulk_group copies.
static __device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

// Wait until at most N store groups still read their smem source (the store tile may then be rewritten; the global
// writes may still be in flight).
template <int N = 0>
static __device__ __forceinline__ void tma_store_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// L2 cache-hint operands for the .L2::cache_hint forms: the createpolicy cookies (64-bit policy values) of the
// evict-normal and evict-first policies, as constants
constexpr uint64_t kEvictNormal = 0x1000000000000000ull;
constexpr uint64_t kEvictFirst = 0x12f0000000000000ull;

// 2D / 3D tile loads carrying an L2 cache hint.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load_hint(uint32_t dst_smem, const CUtensorMap* tmap,
                                                                         int32_t x, int32_t y, uint64_t* bar,
                                                                         uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst_smem),
        "l"(tmap), "r"(to_shared(bar)), "r"(x), "r"(y), "l"(cache_hint)
        : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_3d_load_hint(uint32_t dst_smem, const CUtensorMap* tmap,
                                                                         int32_t x, int32_t y, int32_t z, uint64_t* bar,
                                                                         uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst_smem),
        "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(to_shared(bar)), "l"(cache_hint)
        : "memory");
}

// L2 prefetch of a 2D box with an L2 policy: the lines land at the hint's priority.
static __device__ __forceinline__ void cp_async_bulk_prefetch_tensor_2d_l2_hint(const CUtensorMap* tmap, int32_t x,
                                                                                int32_t y, uint64_t cache_hint) {
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0, {%1, %2}], %3;" ::"l"(tmap),
                 "r"(x), "r"(y), "l"(cache_hint)
                 : "memory");
}

// Warp-level MMA (mma.sync m16n8k16: bf16 inputs, fp32 accumulated in place, row.col) and its ldmatrix fragment loads
// (`.shared` offsets: to_shared); x2.trans turns a row-major [16 x 8] bf16 tile into the col-major B fragment.
static __device__ __forceinline__ void mma_m16n8k16_bf16(float& c0, float& c1, float& c2, float& c3, const uint32_t& a0,
                                                         const uint32_t& a1, const uint32_t& a2, const uint32_t& a3,
                                                         const uint32_t& b0, const uint32_t& b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
static __device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// tcgen05 (5th-generation tensor core) wrappers: TMEM allocation, TMEM <-> register moves, the MMAs and their commit,
// the thread-sync fences, and the smem -> TMEM copy of scale factors.
//
// Lifecycle: one warp allocates n_cols (a power of two in [32, 512]) and the TMEM address lands in smem; MMAs write
// TMEM; warps drain it with tcgen05.ld; the allocating warp deallocates.
// Lane bands: a warp reaches only TMEM lanes [(warp % 4) * 32, + 32) with the .32x32b shapes (the % 4 matters when
// the draining warps are not 0-3), and the band's first 16 lanes with the .16x32bx2 shapes. An M = 64 MMA's row r sits
// at lane 32 x (r / 16) + r % 16 (A in TMEM the same).
// Ordering: tcgen05.ld / .st results are usable after wait::ld / wait::st. An MMA's result is read after its commit
// arrives on an mbarrier, the reader waits on it and issues fence::after_thread_sync (without the fence the drain can
// read stale values). A warp handing TMEM back to the MMA issuer runs wait::ld and fence::before_thread_sync before its
// arrive.
// Descriptors: below.

static __device__ __forceinline__ void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr),
                 "r"(n_cols));
}

static __device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}

// TMEM <-> registers, .32x32b shape: .xN moves N consecutive TMEM columns, one b32 register per column per lane.
static __device__ __forceinline__ void tcgen05_st_32x32b_x2(uint32_t taddr, uint32_t r0, uint32_t r1) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};" ::"r"(taddr), "r"(r0), "r"(r1));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x2(uint32_t taddr, uint32_t& r0, uint32_t& r1) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x4(uint32_t taddr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                            uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_st_32x32b_x16(uint32_t taddr, const uint32_t* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15};" ::"r"(r[0]),
        "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
        "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                            uint32_t& r3, uint32_t& r4, uint32_t& r5, uint32_t& r6,
                                                            uint32_t& r7) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
        : "r"(taddr));
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                             uint32_t& r3, uint32_t& r4, uint32_t& r5, uint32_t& r6,
                                                             uint32_t& r7, uint32_t& r8, uint32_t& r9, uint32_t& r10,
                                                             uint32_t& r11, uint32_t& r12, uint32_t& r13, uint32_t& r14,
                                                             uint32_t& r15) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7), "=r"(r8), "=r"(r9), "=r"(r10),
          "=r"(r11), "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
        : "r"(taddr));
}
// A lane's kN accumulator columns from taddr: one x16 load per 16 columns, or one x8 load for an 8-column window.
template <uint32_t kN>
static __device__ __forceinline__ void tcgen05_ld_32x32b_cols(uint32_t taddr, uint32_t (&v)[kN]) {
    static_assert(kN == 8 || kN % 16 == 0, "whole x16 loads, or one x8");
    if constexpr (kN == 8) {
        tcgen05_ld_32x32b_x8(taddr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    } else {
#pragma unroll
        for (uint32_t j = 0; j < kN; j += 16)
            tcgen05_ld_32x32b_x16(taddr + j, v[j], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7],
                                  v[j + 8], v[j + 9], v[j + 10], v[j + 11], v[j + 12], v[j + 13], v[j + 14], v[j + 15]);
    }
}
static __device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
          "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
          "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
}

// TMEM <-> registers, .16x32bx2 shape: the first 16 lanes of the warp's lane band (where an M = 64 MMA keeps its
// rows), two threads per TMEM lane. Thread t moves TMEM lane t % 16's columns [taddr + (t / 16) x kHalf, + N) for .xN,
// one b32 register per column.
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_ld_16x32bx2_x2(uint32_t taddr, uint32_t& r0, uint32_t& r1) {
    asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0, %1}, [%2], %3;"
                 : "=r"(r0), "=r"(r1)
                 : "r"(taddr), "n"(kHalf));
}
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_st_16x32bx2_x2(uint32_t taddr, uint32_t r0, uint32_t r1) {
    asm volatile("tcgen05.st.sync.aligned.16x32bx2.x2.b32 [%0], %3, {%1, %2};" ::"r"(taddr), "r"(r0), "r"(r1),
                 "n"(kHalf));
}
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_ld_16x32bx2_x16(uint32_t taddr, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16], %17;"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr), "n"(kHalf));
}
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_ld_16x32bx2_x32(uint32_t taddr, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
          "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
          "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr), "n"(kHalf));
}
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_st_16x32bx2_x32(uint32_t taddr, const uint32_t* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.16x32bx2.x32.b32 [%32], %33, "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31};" ::"r"(r[0]),
        "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
        "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
        "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]),
        "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(taddr), "n"(kHalf));
}

// tcgen05.ld.red .max: the drain of 32 fp32 columns of the thread's TMEM lane fused with their row max (NaN dropped):
// r fully written, red = the max; on the 32x32b shape and on the 8-row window's .16x32bx2 shape (kHalf as
// tcgen05_ld_16x32bx2_x32).
static __device__ __forceinline__ void tcgen05_ld_red_max_32x32b_x32(const uint32_t& col, uint32_t* r, float& red) {
    asm volatile(
        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
          "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
          "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=f"(red)
        : "r"(col));
}
template <uint32_t kHalf>
static __device__ __forceinline__ void tcgen05_ld_red_max_16x32bx2_x32(const uint32_t& col, uint32_t* r, float& red) {
    asm volatile(
        "tcgen05.ld.red.sync.aligned.16x32bx2.x32.max.f32 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
          "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
          "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=f"(red)
        : "r"(col), "n"(kHalf));
}

// wait::ld / wait::st: every prior tcgen05.ld / .st of this thread has completed. The memory clobber keeps later
// shared-memory operations (an mbarrier arrive that frees the drained columns) behind the wait: the arrive reads no
// drained register, so without it ptxas could hoist the arrive above the drain.
static __device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
static __device__ __forceinline__ void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// The completion of this thread's prior MMAs arrives once on `bar` (no .shared::cta qualifier: the spec takes none).
static __device__ __forceinline__ void tcgen05_commit_arrive(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}

static __device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
static __device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// MMAs (cta_group::1): D (TMEM) = A x B (+ D when scale_c != 0). A and B are smem matrix descriptors (or A in TMEM for
// the _atmem form), the instruction descriptor's upper 32 bits select the shape and types. Shapes off the PTX table are
// not rejected at compile time: they trap at run time.
// kind::f16 (bf16 here): M 64 / 128, N a multiple of 8 up to 256, K 16.
static __device__ __forceinline__ void tcgen05_mma_f16(uint32_t d, uint64_t desc_a, uint64_t desc_b,
                                                       uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
        "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::f8f6f4: e4m3 A and B from smem descriptors.
static __device__ __forceinline__ void tcgen05_mma_f8f6f4(uint32_t d, uint64_t desc_a, uint64_t desc_b,
                                                          uint32_t inst_desc, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
        "l"(desc_a), "l"(desc_b), "r"(inst_desc), "r"(scale_c));
}

// kind::f16 with A in TMEM: lane = m, the 32-bit cell a_tmem + k / 2 holds the bf16 pair {A[m][k even], A[m][k + 1]}
// (low, high); a K-16 slice advances a_tmem by 8 columns.
static __device__ __forceinline__ void tcgen05_mma_f16_atmem(uint32_t d, uint32_t a_tmem, uint64_t desc_b,
                                                             uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d),
        "r"(a_tmem), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::tf32: fp32 operands in smem (the MMA truncates them to tf32), K 8.
static __device__ __forceinline__ void tcgen05_mma_tf32(uint32_t d, uint64_t desc_a, uint64_t desc_b,
                                                        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(d),
        "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::mxf4nvf4.block_scale.block16 (NVFP4): e2m1 A and B with a UE4M3 scale per 16 elements; the scales live in
// TMEM at sf_a / sf_b (four scale bytes per row at K 64, one 32-bit TMEM word).
static __device__ __forceinline__ void tcgen05_mma_mxf4nvf4_block16(uint32_t d, uint64_t desc_a, uint64_t desc_b,
                                                                    uint32_t inst_desc_high, uint32_t scale_c,
                                                                    uint32_t sf_a, uint32_t sf_b) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16"
        "  [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n" ::"r"(d),
        "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c), "r"(sf_a), "r"(sf_b));
}

// tcgen05.cp 32x128b.warpx4: stage a scale-factor tile from smem (a matrix descriptor) into TMEM, the same 32 x 128 bit
// source broadcast to all four lane bands. Completion is a store from the caller's view (wait::st, or the MMA commit).
static __device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(uint32_t tmem_dst, uint64_t smem_src_desc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem_dst), "l"(smem_src_desc));
}

// .32x32b.x32 TMEM store: 32 columns in one instruction.
static __device__ __forceinline__ void tcgen05_st_32x32b_x32(uint32_t taddr, const uint32_t* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31};" ::"r"(r[0]),
        "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
        "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
        "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]),
        "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(taddr));
}

// tcgen05.mma descriptors: the 64-bit shared-memory matrix descriptor and the 32-bit instruction descriptors of the
// three MMA kinds the layer issues (kind::f16 for bf16, kind::tf32 for the router, kind::mxf4nvf4 for NVFP4). The bit
// layouts below are the verified ones (the public PTX tables have errors around bits 46-60 of the smem descriptor).

// Operand major-ness (instruction descriptor bits 15 / 16): K = K innermost ("TN"), MN = M (A) or N (B) innermost.
// FP4 operands are K-major only.
enum class Major : uint8_t {
    K = 0,
    MN = 1,
};

// Shared-memory matrix descriptor:
//   0-13  start address >> 4         16-29  leading byte offset >> 4 (LBO)     32-45  stride byte offset >> 4 (SBO)
//   46-47 version = 1 (required: larger swizzled shapes read garbage without it)
//   49-51 base offset (0 when the matrix starts at its swizzle pattern's boundary)
//   52    LBO mode = 0 (relative)    61-63  layout: 0 none, 2 128B, 4 64B, 6 32B swizzle
// `swizzle_bytes` is the swizzle period: 0, 32, 64 or 128.
__host__ __device__ static __forceinline__ constexpr uint64_t mma_smem_desc(uint32_t matrix_addr, uint32_t lbo,
                                                                            uint32_t sbo, uint32_t base_offset,
                                                                            int swizzle_bytes) {
    auto enc = [](uint32_t x) -> uint64_t { return (uint64_t)((x & 0x3FFFFu) >> 4); };
    uint8_t code = (swizzle_bytes == 128) ? 2u : (swizzle_bytes == 64) ? 4u : (swizzle_bytes == 32) ? 6u : 0u;
    uint64_t d = 0;
    d |= enc(matrix_addr);
    d |= enc(lbo) << 16;
    d |= enc(sbo) << 32;
    d |= uint64_t(1u) << 46;
    d |= uint64_t(base_offset & 0x7u) << 49;
    d |= uint64_t(code & 0x7u) << 61;
    return d;
}

// K-major operand (A [M, K], B [N, K]): the swizzle period equals the K row (BLOCK_K x sizeof(T) bytes), LBO = 0 and
// SBO = 8 rows. T is a size proxy (uint16_t for bf16); the element type lives in the instruction descriptor.
template <typename T, int BLOCK_K, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t mma_smem_desc_k_major(uint32_t addr,
                                                                                    uint32_t base_offset = 0) {
    constexpr int K_BYTES = BLOCK_K * int(sizeof(T));
    static_assert(SWIZZLE_BYTES == K_BYTES, "K-major requires swizzle bytes == BLOCK_K * sizeof(T)");
    return mma_smem_desc(addr, /*lbo=*/0u, /*sbo=*/8u * uint32_t(K_BYTES), base_offset, SWIZZLE_BYTES);
}

// MN-major operand in swizzle-atom form: atoms of (SWIZZLE_BYTES / sizeof(T)) elements x 8 K rows, adjacent 8-row
// groups SBO = 8 x SWIZZLE_BYTES apart, adjacent MN atoms LBO = BLOCK_K x SWIZZLE_BYTES apart (one 2D TMA load per
// MN atom with 128B swizzle lands exactly this layout). Swizzle NONE is not a valid MN-major layout.
template <typename T, int BLOCK_K, int BLOCK_MN, int SWIZZLE_BYTES>
__host__ __device__ static __forceinline__ constexpr uint64_t mma_smem_desc_mn_major(uint32_t addr,
                                                                                     uint32_t base_offset = 0) {
    constexpr int DT_BYTES = int(sizeof(T));
    constexpr int BLOCK_MN_BYTES = BLOCK_MN * DT_BYTES;
    static_assert(SWIZZLE_BYTES == 32 || SWIZZLE_BYTES == 64 || SWIZZLE_BYTES == 128,
                  "MN-major requires SWZ in {32, 64, 128}; NONE is not a valid layout");
    static_assert(BLOCK_MN_BYTES % SWIZZLE_BYTES == 0,
                  "MN-major: BLOCK_MN * sizeof(T) must be a multiple of SWIZZLE_BYTES");
    constexpr uint32_t SBO = uint32_t(8 * SWIZZLE_BYTES);
    constexpr uint32_t LBO = uint32_t(BLOCK_K * SWIZZLE_BYTES);
    return mma_smem_desc(addr, LBO, SBO, base_offset, SWIZZLE_BYTES);
}

enum class F16Type : uint8_t { F16 = 0, BF16 = 1 };
enum class DType : uint8_t { F16 = 0, F32 = 1, S32 = 2 };

// Instruction descriptor, kind::f16: D type at 4-5, A / B type at 7-9 / 10-12, negate at 13 / 14, major-ness at
// 15 / 16, N >> 3 at 17-22, M >> 4 at 24-28.
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_f16(
    uint32_t M, uint32_t N, F16Type a_type = F16Type::BF16, F16Type b_type = F16Type::BF16, DType d_type = DType::F32,
    Major a_major = Major::K, Major b_major = Major::K, bool negate_a = false, bool negate_b = false) {
    uint32_t d = 0;
    d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
    d |= (static_cast<uint32_t>(a_type) & 0x7u) << 7;
    d |= (static_cast<uint32_t>(b_type) & 0x7u) << 10;
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= ((M >> 4) & 0x1Fu) << 24;
    return d;
}

// Instruction descriptor, kind::tf32: the kind::f16 layout with A / B type = TF32 (2). The operands sit in smem as
// fp32; the MMA truncates each to tf32 itself.
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_tf32(
    uint32_t M, uint32_t N, DType d_type = DType::F32, Major a_major = Major::K, Major b_major = Major::K,
    bool negate_a = false, bool negate_b = false) {
    constexpr uint32_t TF32 = 2;
    uint32_t d = 0;
    d |= (static_cast<uint32_t>(d_type) & 0x3u) << 4;
    d |= TF32 << 7;
    d |= TF32 << 10;
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= ((M >> 4) & 0x1Fu) << 24;
    return d;
}

enum class ScaleFormat : uint8_t { E4M3 = 0, E8M0 = 1 };

// Instruction descriptor, kind::mxf4nvf4.block_scale.block16 (NVFP4: e2m1 operands, a UE4M3 scale per 16 elements).
// The block-scaled kinds use their own layout:
//   2      sparse (A 4:8)               4-5    b_sf_id (top 2 bits of the SFB TMEM address)
//   7-9    A format (E2M1 = 1)          10-11  B format (E2M1 = 1)          13 / 14  negate A / B
//   15/16  major-ness (K only)          17-22  N >> 3                       23       scale format (0 = UE4M3)
//   24-26  reserved (0)                 27-28  M >> 7                       29-30    a_sf_id
//   31     K: dense 0 = K64, 1 = K96 (sm_103a); sparse 0 = K128
// An M of 64 encodes as 0, which these kinds reject (illegal instruction): the MMAs here are M = 128.
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_mxf4nvf4_block16(
    uint32_t M, uint32_t N, uint32_t tmem_sfa_addr = 0, uint32_t tmem_sfb_addr = 0, Major a_major = Major::K,
    Major b_major = Major::K, bool negate_a = false, bool negate_b = false, bool k_dim = false,
    ScaleFormat sf = ScaleFormat::E4M3, bool sparse = false) {
    constexpr uint32_t MXF4_E2M1 = 1;
    uint32_t d = 0;
    if (sparse) d |= 1u << 2;
    d |= ((tmem_sfb_addr & 0xC0000000u) >> 30) << 4;
    d |= MXF4_E2M1 << 7;
    d |= MXF4_E2M1 << 10;
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= (static_cast<uint32_t>(sf) & 0x1u) << 23;
    d |= ((M >> 7) & 0x3u) << 27;
    d |= ((tmem_sfa_addr & 0xC0000000u) >> 30) << 29;
    if (k_dim) d |= 1u << 31;
    return d;
}
static_assert(((mma_inst_desc_mxf4nvf4_block16(128, 64) >> 27) & 0x3u) == 1u &&
                  ((mma_inst_desc_mxf4nvf4_block16(256, 64) >> 27) & 0x3u) == 2u,
              "mma_inst_desc_mxf4nvf4_block16 encodes m_dim in bits 27-28");
static_assert(((mma_inst_desc_mxf4nvf4_block16(128, 64) >> 24) & 0x7u) == 0u &&
                  ((mma_inst_desc_mxf4nvf4_block16(256, 64) >> 24) & 0x7u) == 0u,
              "mma_inst_desc_mxf4nvf4_block16 keeps bits 24-26 reserved (zero)");

// A cluster's peer CTAs: this CTA's rank, and st.async / atom into a peer's smem (addresses from
// mapa_shared_cluster).

static __device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// st.async of 16 B / 8 B / 4 B into a peer CTA's smem, completing on the peer's mbarrier (both addresses
// cluster-mapped). Issued from registers, so the source CTA may exit with the stores in flight.
static __device__ __forceinline__ void st_async_cluster_v4(uint32_t dst_smem_cluster, const uint4& v,
                                                           uint32_t mbar_cluster) {
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
        " [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst_smem_cluster),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(mbar_cluster)
        : "memory");
}

static __device__ __forceinline__ void st_async_cluster_v2(uint32_t dst_smem_cluster, uint32_t x, uint32_t y,
                                                           uint32_t mbar_cluster) {
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(
                     dst_smem_cluster),
                 "r"(x), "r"(y), "r"(mbar_cluster)
                 : "memory");
}

static __device__ __forceinline__ void st_async_cluster_b32(uint32_t dst_smem_cluster, uint32_t v,
                                                            uint32_t mbar_cluster) {
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst_smem_cluster),
        "r"(v), "r"(mbar_cluster)
        : "memory");
}

// Returning-old add on a peer CTA's smem word, cluster scope.
static __device__ __forceinline__ uint32_t atom_add_shared_cluster_u32(uint32_t dst_smem_cluster, uint32_t value) {
    uint32_t ret;
    asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;"
                 : "=r"(ret)
                 : "r"(dst_smem_cluster), "r"(value)
                 : "memory");
    return ret;
}

// Row loads and stores of global memory by coherence: volatile and .cg forms for data written during the kernel,
// non-coherent 256-bit loads for data no thread writes, plain stores of outputs, and the sys-scope relaxed forms of
// the cross-rank planes.

static __device__ __forceinline__ void st_row_u32_volatile(uint32_t* p, const uint32_t& v) {
    asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v));
}

// 256-bit non-coherent load (32-B aligned) of data no thread writes during the kernel: one LDG.256 for two
// 128-bit loads. Not volatile: the volatile .v8 lowers to a STRONG.SYS load that serializes.
static __device__ __forceinline__ void ld_row_u8_nc(const uint4* p, uint4& v0, uint4& v1) {
    asm("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w), "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
        : "l"(p));
}

// Plain 256-bit / 128-bit stores of kernel outputs (asm volatile only pins them against elision).
static __device__ __forceinline__ void st_row_u8(uint4* p, const uint4& v0, const uint4& v1) {
    asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(p), "r"(v0.x), "r"(v0.y), "r"(v0.z),
                 "r"(v0.w), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w)
                 : "memory");
}

static __device__ __forceinline__ void st_row_u4(uint4* p, const uint4& v) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// sys-scope relaxed 16-B load / store: the all-reduce's peer-plane polls and pushes.
static __device__ __forceinline__ uint4 ld_relaxed_sys_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

static __device__ __forceinline__ void st_relaxed_sys_u4(uint4* p, const uint4& v) {
    asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// sys-scope relaxed 8 B elements (the MoE phase's two cross-rank gathers, moe.cuh): a v2.b64 / v4.b64 access is two /
// four 8 B scalar operations, so a 16 B or 32 B load and the 8 B stores it reads overlap element for element
static __device__ __forceinline__ void ld_relaxed_sys_v2b64(const void* p, uint64_t& a, uint64_t& b) {
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
static __device__ __forceinline__ void ld_relaxed_sys_v4b64(const void* p, uint64_t (&v)[4]) {
    asm volatile("ld.relaxed.sys.global.v4.b64 {%0, %1, %2, %3}, [%4];"
                 : "=l"(v[0]), "=l"(v[1]), "=l"(v[2]), "=l"(v[3])
                 : "l"(p)
                 : "memory");
}
static __device__ __forceinline__ void st_relaxed_sys_v2b64(void* p, uint64_t a, uint64_t b) {
    asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
static __device__ __forceinline__ void st_relaxed_sys_b64(void* p, uint64_t a) {
    asm volatile("st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p), "l"(a) : "memory");
}

// .cg (L2, L1-bypassing) row loads / stores: coherent like the volatile forms but pipelined (a volatile access
// serializes in the memory pipe). The release that publishes them orders them for the consumer.
static __device__ __forceinline__ void st_row_u2_cg(uint2* p, const uint2& v) {
    asm volatile("st.global.cg.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y));
}

static __device__ __forceinline__ void st_row_u8_cg(uint4* p, const uint4& v0, const uint4& v1) {
    asm volatile("st.global.cg.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(p), "r"(v0.x), "r"(v0.y),
                 "r"(v0.z), "r"(v0.w), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w));
}

static __device__ __forceinline__ uint2 ld_row_u2_cg(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_row_u4_cg(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// NVLS multimem (multicast) forms and the sys-scope acquire the cross-rank words pair with. A multicast VA is legal
// only as the operand of multimem.* instructions; a rank's own replica is read through its unicast VA.

// Release-strength counter add into every replica of a multicast word (the token-readiness counts).
static __device__ __forceinline__ void multimem_red_add_release_u32(uint32_t* mc_ptr, uint32_t v) {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(mc_ptr), "r"(v) : "memory");
}

// The acquire read of a cross-rank word on the local replica. `.sys` is required (a `.gpu` acquire does not
// synchronize with a peer rank's release); the clobber keeps the consumer's reads below the spin.
static __device__ __forceinline__ uint32_t ld_acq_sys_u32(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Weak 16-B multicast store (8 bf16 or 4 fp32; multimem.st takes no vector .b32, the packed-half form is the 16-B
// carrier and is bit-transparent). Weak on purpose: the publisher's later release orders it.
static __device__ __forceinline__ void multimem_st_v4_bf16x2_weak(uint32_t* mc_ptr, uint32_t a, uint32_t b, uint32_t c,
                                                                  uint32_t d) {
    asm volatile("multimem.st.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(mc_ptr), "r"(a), "r"(b), "r"(c), "r"(d)
                 : "memory");
}

// The all-reduce push's multicast form: one relaxed.sys 16-B packet, fanned out to every rank by the NVLink switch.
static __device__ __forceinline__ void multimem_st_relaxed_sys_v4_bf16x2(uint32_t* mc_ptr, uint4 v) {
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(mc_ptr), "r"(v.x), "r"(v.y),
                 "r"(v.z), "r"(v.w)
                 : "memory");
}

// Waits on counter words: the arrival-count protocol, the fenced acquire load, and the bounded spins (a dead peer
// traps with a named report instead of hanging the step).

// The arrival-count protocol: producers write their data with relaxed / plain stores, then ONE release add of the
// count; consumers acquire-spin on the count, then read the data plainly.
static __device__ __forceinline__ void arrival_count_publish(uint32_t* count_ptr, const uint32_t& value) {
    ptx::red_add_rel_b32(count_ptr, value);
}

// Acquire load with a compiler memory clobber (so consumer loads cannot be hoisted above the spin).
static __device__ __forceinline__ uint32_t ld_acq_fenced(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Bounded spins: a wait whose publisher never comes (a dead or desynced rank, a protocol bug) traps with a named
// print instead of hanging the step (the host's next sync raises). Two bounds, so that under a silent peer the first
// trap is a cross-rank wait naming the peer:
//  cross-rank waits (the all-reduce's lamport consume; the MoE phase's gathers bound theirs shorter, moe.cuh):
//      kTrapSpinNs of global timer, read every kSpinCheckMask + 1 polls (the first read starts the budget, so a
//      healthy wait never reads it)
//  in-rank ticket waits: kTicketSpinPolls polls, no timer state in the loop; a gpu-scope acquire poll is an L2 round
//      trip (~140 ns), so the bound (~38 s) outlasts the cross-rank budget
// The report is out of line: a printf in the kernel's own text costs it a stack frame and register spills in the hot
// paths; a call into a never-taken function does not.
inline constexpr uint64_t kTrapSpinNs = 30000000000;  // 30 s: a decode layer runs in ~0.2 ms
inline constexpr uint32_t kSpinCheckMask = 1023;
inline constexpr uint32_t kTicketSpinPolls = 1u << 28;  // ~38 s of L2-hit polls
static __device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// The reports never return (their call sites save no registers) and print once per warp: the calling warp's lowest
// active lane prints, the warp joins behind it, then traps (a dead peer stalls every lane of a row, and a line per
// thread overruns the printf FIFO).
static __device__ __forceinline__ bool lowest_active_lane(const uint32_t active) {
    uint32_t lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lt));
    return (active & lt) == 0;
}
// the cross-rank report: the wait, this rank, the polled word in the wait's own index (a peer, a tile, a packet), its
// value and target
__device__ __noinline__ __attribute__((noreturn)) void spin_timeout_trap(const char* what, int rank, int word,
                                                                         uint32_t observed, uint32_t target,
                                                                         uint64_t waited_ns) {
    const uint32_t active = __activemask();
    if (lowest_active_lane(active))
        printf("mk: the %s wait timed out on rank %d: word %d observed %u, target %u (%llu ms)\n", what, rank, word,
               unsigned(observed), unsigned(target), static_cast<unsigned long long>(waited_ns / 1000000ull));
    __syncwarp(active);
    __trap();
}
// the in-rank report: the ticket word and the waiting thread
__device__ __noinline__ __attribute__((noreturn)) void ticket_timeout_trap(const uint32_t* word, uint32_t observed,
                                                                           uint32_t target) {
    const uint32_t active = __activemask();
    if (lowest_active_lane(active))
        printf("mk: a ticket wait timed out on block %u thread %u: word %p observed %u, target %u (%u polls)\n",
               blockIdx.x, threadIdx.x, static_cast<const void*>(word), unsigned(observed), unsigned(target),
               kTicketSpinPolls);
    __syncwarp(active);
    __trap();
}
// a cross-rank wait: observe() is the polled value the report prints (read on the fault path only)
template <typename Pred, typename Observe>
static __device__ __forceinline__ void bounded_spin(Pred pred, const char* what, int rank, int word, uint32_t target,
                                                    Observe observe) {
    uint32_t polls = 0;
    uint64_t t0 = 0;
    while (!pred()) {
        if ((++polls & kSpinCheckMask) == 0) {
            const uint64_t now = globaltimer_ns();
            if (t0 == 0) t0 = now;
            if (now - t0 > kTrapSpinNs) spin_timeout_trap(what, rank, word, observe(), target, now - t0);
        }
    }
}

// Spin until the count reaches target. The words are u32, never reset and +n per launch, so the compare is the
// wrap-safe signed distance (an ordered < would open vacuously when the count crosses 2^32).
static __device__ __forceinline__ void count_spin(const uint32_t* ptr, const uint32_t& target) {
    uint32_t polls = 0;
#pragma unroll 1
    while (static_cast<int32_t>(ld_acq_fenced(ptr) - target) < 0)
        if (++polls == kTicketSpinPolls) ticket_timeout_trap(ptr, ld_acq_fenced(ptr), target);
}

}  // namespace ptx
