"""The math of the bs1 verify forward (TP8 or TP4), as the one-launch kernels compute it, in plain PyTorch.

This file is the spec of what every kernel of the verify graph computes, and the checker the kernel tests run on GPU
against them. One decoder layer is one launch; in data order it is:

    attention  attention of the window's 16 rows over this layer's KV (GQA, swa window 2047 or global), the
               per-head weightless o-norm and the sigmoid o-gate                                  -> x_attn
    postattn   o_proj on the rank's heads, the cross-rank sum (8 bf16 partials, rank order), the gated residual
               add + pre-MoE norm                                                                 -> stream, h3
               pooler (node N-shard + gather) || router (fp32 weights read as tf32) -> top-8
               pre-expert norm + dynamic NVFP4 quantization of the expert input
    MoE        the experts, tensor-parallel on the intermediate: g1 on the rank's I slice (SwiGLU), h
               requantized and gathered, g2 on the rank's H slice, gathered                       -> zb
    next layer per-expert post-norm + weighted combine, unpooler (node N-shard + gather), the cross-layer norm,
               the next layer's qkv projection + qk-norm + RoPE (swa) or temperature (global) + K/V rows
The bf16 layers (0 and 61) run the same program with bf16 experts; layer 61 ends at the final norm.

Precision policy. Every dot product accumulates in fp64 and is rounded once to the dtype the kernel stores; every
elementwise step is fp32 in the kernel's operation order, with the kernel's bf16 rounding points. So the reference
differs from a kernel only by the kernel's fp32 accumulation order and its approximate ops (ex2, rsqrt, rcp), which
moves a value only where it sits next to a rounding boundary. ``compare`` holds outputs to the bounds that implies.

Tensor parallelism. Every function runs on ONE rank's shard; ``Comm`` stands in for the kernels' cross-rank moves
(``all_gather`` over the 8 ranks, ``node_gather`` over the 4 ranks of a node). ``emulate_ranks`` runs the 8 ranks as
threads of one process for CPU tests; ``DistComm`` uses torch.distributed. Shapes are read off the tensors, so the
same code runs unsharded with ranks = 1. Weights are taken in the kernels' layouts (``pack_*`` builds them from
dense tensors, ``unpack_*`` inverts them exactly; the kernel tests build their synthetic weights with the same
helpers: sm103/weights.py). Code references name this package's kernels (``kernels/...``).
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Callable, Optional, Protocol, Union

import torch

# the accumulation dtype of every dot product (the precision policy above)
ACC = torch.float64
HEAD_DIM = 64
NVFP4_GROUP = 16  # elements per e4m3 scale factor
TOPK = 8

# Constants the kernels bake:
# head_dim ** -0.5, applied as 0.125 * log2(e) in exp2 (kernels/attention.cuh kL2eScale)
SOFTMAX_SCALE = 0.125
# the per-head o-norm (kernels/attention.cuh: the combine's gate-norm)
ONORM_EPS = 1e-5
# both residual norms (kernels/postattn.cuh the AR row; kernels/next_layer.cuh kPostEps / kPreEps)
POST_EPS, PRE_EPS = 1e-8, 1e-5
# the per-expert post-norm (host/runtime.cu kPostNormEps)
COMBINE_EPS = 1e-8
# the router's score-division eps (kernels/postattn.cuh)
SCORE_DIV_EPS = 1e-15
# the model's rms_norm_eps: qk-norm, embed norm, pre-attention norm
RMS_EPS = 1e-5


# ---------------------------------------------------------------------------------------------------------------------
# Number formats

# e2m1: code & 7 -> magnitude, bit 3 = sign
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# the kernels' fp32 constants 1.0f / 6.0f and 1.0f / (6.0f * 448.0f) (e2m1 max x e4m3 max)
_F32_SIXTH = (torch.tensor(1.0) / torch.tensor(6.0)).item()
_F32_INV_2688 = (torch.tensor(1.0) / torch.tensor(6.0 * 448.0)).item()


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=codes.device)
    return torch.cat((grid, -grid))[codes.long()]


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """cvt.rn.satfinite.e2m1x2.f32: round to nearest, ties to the even code, saturate at 6."""
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=x.device)
    mag = x.abs().clamp(max=6.0)
    code = torch.bucketize(mag, mids)  # a value on a midpoint lands on the lower code
    tie = (code < len(mids)) & (mag == mids[code.clamp(max=len(mids) - 1)])
    code = code + (tie & (code % 2 == 1)).long()
    return (code | (torch.signbit(x).long() << 3)).to(torch.uint8)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] 4-bit codes -> [..., K/2] bytes, element 2i in the low nibble (the cvt's lower half)."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(b: torch.Tensor) -> torch.Tensor:
    return torch.stack((b & 0xF, b >> 4), dim=-1).reshape(
        *b.shape[:-1], 2 * b.shape[-1]
    )


def e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """cvt.rn.satfinite.e4m3x2.f32 as a byte (torch's cast is RN-even; the clamp is the satfinite)."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def dequant_nvfp4(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 [..., K/2] + e4m3 group scales [..., K/16] -> f32 [..., K] in SCALED space (any per-tensor or
    per-token scale lives in the epilogue constants, as in the kernels)."""
    v = e2m1_decode(unpack_nibbles(q))
    return (v.view(*sf.shape, NVFP4_GROUP) * e4m3_decode(sf)[..., None]).view(v.shape)


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Static NVFP4 of a weight [..., K] (for synthetic weights): per 16-group scale e4m3(amax / 6), codes
    e2m1(w / scale) -> (packed codes [..., K/2] u8, scales [..., K/16] u8)."""
    g = w.float().reshape(*w.shape[:-1], -1, NVFP4_GROUP)
    sf = e4m3_encode(g.abs().amax(-1) * _F32_SIXTH)
    d = e4m3_decode(sf)
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    return pack_nibbles(e2m1_encode(g * inv[..., None]).reshape(w.shape)), sf


def tf32(x: torch.Tensor) -> torch.Tensor:
    """An fp32 operand as the tf32 MMA reads it: the low 13 mantissa bits ignored."""
    return (x.float().contiguous().view(torch.int32) & -8192).view(torch.float32)


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e as fp32 bits, e in [-126, 127] (kernels/moe.cuh pow2i)."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def _mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T accumulated in fp64."""
    return x.to(ACC) @ w.to(ACC).T


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fmaf(a, b, c): the fp32 product is exact in fp64, one rounding to fp32."""
    return (a.to(ACC) * b.to(ACC) + c.to(ACC)).float()


def _mean_sq(x: torch.Tensor) -> torch.Tensor:
    return x.to(ACC).square().mean(-1, keepdim=True).float()


def _silu(g: torch.Tensor) -> torch.Tensor:
    return g / (1.0 + torch.exp(-g))


# ---------------------------------------------------------------------------------------------------------------------
# Weight layouts (the kernels' ABI): sm103/weights.py

from sgl_muse_mega.muse_spark.sm103.weights import (  # noqa: E402
    FBW,
    unpack_w1_bf16,
    unpack_w1_fp4,
    unpack_w2_bf16,
    unpack_w2_fp4,
)

# ---------------------------------------------------------------------------------------------------------------------
# Tensor-parallel collectives


class Comm(Protocol):
    rank: int
    ranks: int
    node_ranks: int

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Every rank's x concatenated along dim, in rank order."""

    def node_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """The x of this rank's node (ranks are node-major) concatenated along dim, in rank order."""


class DistComm:
    """``Comm`` over torch.distributed (one process per rank). Every rank must construct it: the node groups are
    created collectively. Values cross ranks as host copies (the kernel tests' process group is gloo).
    """

    def __init__(self, node_ranks: int = 4):
        import torch.distributed as dist

        self._dist = dist
        self.rank, self.ranks, self.node_ranks = (
            dist.get_rank(),
            dist.get_world_size(),
            node_ranks,
        )
        assert self.ranks % node_ranks == 0, (self.ranks, node_ranks)
        self._node = None
        for n in range(self.ranks // node_ranks):
            group = dist.new_group(list(range(n * node_ranks, (n + 1) * node_ranks)))
            if self.rank // node_ranks == n:
                self._node = group

    def _gather(self, x: torch.Tensor, dim: int, group, size: int) -> torch.Tensor:
        h = x.detach().contiguous().cpu()
        parts = [torch.empty_like(h) for _ in range(size)]
        self._dist.all_gather(parts, h, group=group)
        return torch.cat(parts, dim).to(x.device)

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return self._gather(x, dim, None, self.ranks)

    def node_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return self._gather(x, dim, self._node, self.node_ranks)


class _Hub:
    def __init__(self, ranks: int):
        self.barrier = threading.Barrier(ranks)
        self.slots: list = [None] * ranks


class ThreadComm:
    """``Comm`` between the threads of one process (``emulate_ranks``)."""

    def __init__(self, hub: _Hub, rank: int, ranks: int, node_ranks: int):
        self._hub, self.rank, self.ranks, self.node_ranks = hub, rank, ranks, node_ranks

    def _exchange(self, x: torch.Tensor) -> list:
        self._hub.slots[self.rank] = x
        self._hub.barrier.wait()  # every rank posted
        parts = list(self._hub.slots)
        self._hub.barrier.wait()  # every rank read before the slots are reused
        return parts

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat(self._exchange(x), dim)

    def node_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        n0 = self.rank - self.rank % self.node_ranks
        return torch.cat(self._exchange(x)[n0 : n0 + self.node_ranks], dim)


def emulate_ranks(
    fn: Callable[[Comm], object], ranks: int = 8, node_ranks: int = 4
) -> list:
    """Run ``fn(comm)`` once per rank, each rank a thread of this process over an in-memory ``Comm`` -> the ranks'
    results. A rank that raises breaks the others' barriers; the first real error is re-raised.
    """
    hub, results, errors = _Hub(ranks), [None] * ranks, []

    def run(r: int) -> None:
        try:
            results[r] = fn(ThreadComm(hub, r, ranks, node_ranks))
        except BaseException as e:  # re-raised below, after every thread joined
            errors.append(e)
            hub.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
    if errors:
        raise (real or errors)[0]
    return results


# ---------------------------------------------------------------------------------------------------------------------
# Layer stages (one rank)


def split_qkv(
    qkv: torch.Tensor, q_heads: int, kv_heads: int
) -> tuple[torch.Tensor, ...]:
    """A qkv row [m, (2 q + 2 kv) 64] = [q | k | v | o-gate] -> per-head views; q head h uses kv head h // (q / kv)."""
    m, d = qkv.shape[0], HEAD_DIM
    q, k, v, og = torch.split(
        qkv, (q_heads * d, kv_heads * d, kv_heads * d, q_heads * d), dim=1
    )
    return (
        q.view(m, q_heads, d),
        k.view(m, kv_heads, d),
        v.view(m, kv_heads, d),
        og.view(m, q_heads, d),
    )


def attention(
    qkv: torch.Tensor, k_ctx: torch.Tensor, v_ctx: torch.Tensor, window: Optional[int]
) -> torch.Tensor:
    """The attention phase -> x_attn [m, q_heads 64] bf16. qkv = this layer's row (q / k normed and roped by the
    previous launch), k_ctx / v_ctx [n, kv_heads, 64] = the KV rows before the window in position order. Row t sits at
    position n + t and sees positions p with p <= n + t and, on swa layers, n + t - p <= window
    (kernels/attention.cuh: the attention CTAs' masks); scores scale by 1/8. The kernel's P is bf16 into the PV MMA, its
    row sum fp32; here the softmax is exact. Then per head x = O / L, x * rsqrt(mean(x^2) + 1e-5) * sigmoid(og),
    rounded once (kernels/attention.cuh: the combine)."""
    m, kv_heads = qkv.shape[0], k_ctx.shape[1]
    q_heads = (qkv.shape[1] // HEAD_DIM - 2 * kv_heads) // 2
    q, k_new, v_new, og = split_qkv(qkv, q_heads, kv_heads)
    k = torch.cat((k_ctx, k_new)).to(ACC)
    v = torch.cat((v_ctx, v_new)).to(ACC)
    n = k.shape[0]
    pos_q = torch.arange(n - m, n, device=qkv.device)[:, None]
    pos_k = torch.arange(n, device=qkv.device)[None, :]
    seen = pos_k <= pos_q
    if window is not None:
        seen &= pos_k + window >= pos_q
    qg = q.to(ACC).view(m, kv_heads, q_heads // kv_heads, HEAD_DIM)
    s = torch.einsum("tkgd,nkd->kgtn", qg, k) * SOFTMAX_SCALE
    p = torch.softmax(s.masked_fill(~seen, float("-inf")), dim=-1)
    o = torch.einsum("kgtn,nkd->tkgd", p, v).reshape(m, q_heads, HEAD_DIM).float()
    rstd = torch.rsqrt(_mean_sq(o) + ONORM_EPS)
    return _bf16(o * rstd * torch.sigmoid(og.float())).reshape(m, q_heads * HEAD_DIM)


def o_proj_allreduce(
    x_attn: torch.Tensor, w_o: torch.Tensor, comm: Comm, round_partials: bool = True
) -> torch.Tensor:
    """o_proj on the rank's heads (w_o [H, q_heads 64] = the rank's K slice), then the cross-rank sum -> [m, H] bf16.
    The kernel pushes its partial as bf16 packets and every rank sums the 8 packets in rank order in fp32
    (kernels/postattn.cuh, kernels/framework.cuh ar_lamport_consume_packet8). ``round_partials``
    False sums the unrounded partials: the one place TP changes the numerics, switched off only to hold the sharded
    layer against the unsharded one."""
    partial = _mm(x_attn, w_o)
    if round_partials:
        partial = _bf16(partial)
    parts = comm.all_gather(partial[None], 0)
    acc = parts[0].to(torch.float32 if round_partials else ACC)
    for r in range(1, parts.shape[0]):
        acc = acc + parts[r].to(acc.dtype)
    return _bf16(acc)


def residual_norm(
    branch: torch.Tensor,
    stream: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_w: torch.Tensor,
    post_w: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused post-norm + residual + pre-norm with a norm-preserving gated residual (alpha, beta from the gate) ->
    (new stream [m, H] f32, normed [m, H] bf16):
        v = branch (post-attention: weightless post-norm) or branch * post_w (cross-layer: input-scale post-norm)
        stream' = alpha * stream + beta * v * rsqrt(mean(v^2) + 1e-8)
        normed  = stream' * rsqrt(mean(stream'^2) + 1e-5) * pre_w
    in the kernels' operation order (post-attention: kernels/framework.cuh row_arnorm_cn4_row; cross-layer:
    kernels/framework.cuh row_norm_residual via kernels/next_layer.cuh)."""
    v = branch.float()
    if post_w is not None:
        v = v * post_w.float()
    post_rstd = torch.rsqrt(_mean_sq(v) + POST_EPS)
    scaled = (beta * post_rstd) * v if post_w is None else beta * (v * post_rstd)
    ns = _fma(alpha, stream, scaled)
    pre_rstd = torch.rsqrt(_mean_sq(ns) + PRE_EPS)
    return ns, _bf16((ns * pre_rstd) * pre_w.float())


def node_sharded_linear(x: torch.Tensor, w: torch.Tensor, comm: Comm) -> torch.Tensor:
    """The pooler / unpooler: a replicated weight [N, K] whose N is split over the node's ranks (rank r computes block
    r % node_ranks) and gathered inside the node -> bf16 [m, N] on every rank (kernels/postattn.cuh the pooler
    tiles' push, kernels/next_layer.cuh the unpooler GEMM). The pooler's two K halves meet in fp32 before the one bf16
    rounding (kernels/postattn.cuh: the pooler pair's transfer).
    """
    n = w.shape[0] // comm.node_ranks
    j = comm.rank % comm.node_ranks
    return comm.node_gather(_bf16(_mm(x, w[j * n : (j + 1) * n])), 1)


def router(h3: torch.Tensor, w_router: torch.Tensor) -> torch.Tensor:
    """Router logits [m, E] f32: the fp32 router weight through the tf32 MMA (the x operand is h3's fp32 copy, exact
    in tf32) (kernels/postattn.cuh: the router units); the kernel's 12 K-slice partials are summed in slice order
    by the top-k row (kernels/postattn.cuh: the top-k row's reduction)."""
    return _mm(h3.float(), tf32(w_router)).float()


def topk(
    logits: torch.Tensor,
    router_mult: float,
    expert_bias: torch.Tensor,
    route_scale: float,
    k: int = TOPK,
    ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing (kernels/postattn.cuh topk_token) -> (ids [m, k] i32 in selection order, weights
    [m, k] f32): s = sigmoid(router_mult * logit); the k largest s + bias (the smaller id wins a tie); each weight =
    (s + bias) - bias (the kernel recovers it so), normalized by the selection-order sum + 1e-15, times route_scale.
    The ids are always the reference's own selection; ``ids`` = the kernel's ids: the weights are then taken at them
    (a near tie can select differently).
    """
    s = torch.sigmoid(logits.to(ACC) * router_mult).float()
    bias = expert_bias.float()
    sel = s + bias
    own = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, :k]
    at = own if ids is None else ids.long()
    sv = sel.gather(-1, at) - bias[at]
    total = torch.zeros_like(sv[:, 0])
    for j in range(at.shape[1]):
        total = total + sv[:, j]
    return (
        own.to(torch.int32),
        (sv * (1.0 / (total + SCORE_DIV_EPS))[:, None]) * route_scale,
    )


def pre_expert_norm(
    pooled: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """The pre-expert RMSNorm -> bf16 [m, H]: bf16(bf16(x * rsqrt(mean(x^2) + eps)) * gamma), gamma = the effective
    weight (kernels/framework.cuh row_pre_expert_norm_bf16_dyn_wide_frag). On the bf16 layers this row is the
    experts' input.
    """
    x = pooled.float()
    return _bf16(_bf16(x * torch.rsqrt(_mean_sq(x) + eps)).float() * gamma.float())


def quantize_expert_input(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The dynamic per-token NVFP4 of the normed expert input (kernels/framework.cuh
    row_pre_expert_norm_quant_nvfp4_dyn_wide_frag) -> (codes
    [m, H/2] u8, scales [m, H/16] u8, x_scale [m] f32), x ~= code * scale * x_scale:
        x_scale   = max(row amax, 1e-10) / 2688                    (2688 = 6 * 448)
        scale_g   = e4m3(max(group amax, 1e-10) / 6 / x_scale)
        code      = e2m1(x / max(scale_g * x_scale, 1e-10))"""
    v = x.float()
    m = v.shape[0]
    x_scale = v.abs().amax(-1).clamp(min=1e-10) * _F32_INV_2688
    gs = 1.0 / x_scale
    g = v.view(m, -1, NVFP4_GROUP)
    sf = e4m3_encode((g.abs().amax(-1).clamp(min=1e-10) * _F32_SIXTH) * gs[:, None])
    out_scale = 1.0 / (e4m3_decode(sf) * x_scale[:, None]).clamp(min=1e-10)
    codes = e2m1_encode(g * out_scale[..., None])
    return pack_nibbles(codes.view(m, -1)), sf, x_scale


def _prov_exp(amax: torch.Tensor) -> torch.Tensor:
    """The h row window's provisional group exponent e: amax * 2^-e in (1024, 1344] or (1344, 2048] one octave up, so
    the group's scale amax * 2^-e / 6 lands in (224, 448] (kernels/moe.cuh prov_exp).
    """
    b = amax.contiguous().view(torch.int32)
    e = (b >> 23) - 138 + (((b & 0x7FFFFF) > 0x280000) | (b > 0x7F800000)).int()
    return e.clamp(min=-120)


def _scale_down_byte(byte: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """An e4m3 scale byte times 2^-delta (delta >= 0): an exponent-field move while normal, round-half-up onto the
    2^-9 subnormal grid below (kernels/moe.cuh scale_down_byte_nb)."""
    byte, delta = byte.long(), delta.long()
    e = byte >> 3
    sig = torch.where(e != 0, 8, 0) | (byte & 7)
    shift = (delta - e + (e != 0).long()).clamp(0, 8)
    half = torch.where(
        shift > 0,
        torch.bitwise_left_shift(torch.ones_like(shift), (shift - 1).clamp(min=0)),
        0,
    )
    return torch.where(e > delta, byte - (delta << 3), (sig + half) >> shift).to(
        torch.uint8
    )


def h_quant(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The g1 producer's h quantization on its I slice z [..., I_s] (kernels/moe.cuh: the g1 epilogue) -> the
    three planes the MoE phase gathers: (codes [..., I_s/2] u8, scales [..., I_s/16] u8, exponents [..., I_s/16] i32):
    each 16-group coded at its provisional exponent e, scale = e4m3(amax 2^-e / 6), code = e2m1(z 2^-e / scale).
    """
    g = z.reshape(*z.shape[:-1], -1, NVFP4_GROUP)
    mx = g.abs().amax(-1)
    eb = _prov_exp(mx)
    sc = _pow2(-eb)
    sf = e4m3_encode((mx * sc) * _F32_SIXTH)
    d = e4m3_decode(sf)
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    codes = e2m1_encode((g * sc[..., None]) * inv[..., None])
    return pack_nibbles(codes.reshape(z.shape)), sf, eb


def h_dequant(codes: torch.Tensor, sf: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """The g2 consumer's view of the gathered FULL h row (the row window, kernels/moe.cuh kHRow) -> f32 [..., I]:
    e_row = the row's max exponent, each scale moved down to it, value = code * scale' * 2^e_row (the 2^e_row rides
    g2's epilogue alpha: kernels/moe.cuh g2_hrow_recal and the g2 epilogue)."""
    e_row = eb.amax(-1, keepdim=True)
    v = e2m1_decode(unpack_nibbles(codes)).view(*sf.shape, NVFP4_GROUP)
    v = (
        v
        * e4m3_decode(_scale_down_byte(sf, e_row - eb))[..., None]
        * _pow2(e_row)[..., None]
    )
    return v.view(*codes.shape[:-1], 2 * codes.shape[-1])


@dataclasses.dataclass
class Fp4Experts:
    """One rank's NVFP4 experts in the kernel layout (``pack_w1_fp4`` / ``pack_w2_fp4``) and their epilogue constants:
    g1_alphas = the gate weight scale, g1_scale_c = up scale / gate scale, g2_alphas = the w2 weight scale (the dynamic
    per-token scaling)."""

    w1: torch.Tensor  # [E, I_s/128, H/256, 256, 128] u8
    w1_sf: torch.Tensor  # [E I_s/128, (H/64) 1024] u8
    w2: torch.Tensor  # [E, H_s/128, I/256, 128, 128] u8
    w2_sf: torch.Tensor  # [E, H_s, I/16] u8
    g1_alphas: torch.Tensor  # [E] f32
    g1_scale_c: torch.Tensor  # [E] f32
    g2_alphas: torch.Tensor  # [E] f32


@dataclasses.dataclass
class Bf16Experts:
    """One rank's bf16 experts in the kernel layout (``pack_w1_bf16`` / ``pack_w2_bf16``); no scales."""

    w1: torch.Tensor  # [E, I_s/128, H/64, 256, 64] bf16
    w2: torch.Tensor  # [E, H_s/128, I/64, 128, 64] bf16


def _slots_by_expert(ids: torch.Tensor):
    """(expert, (token, k) indices) for every routed expert: only these are computed."""
    for e in ids.unique().tolist():
        yield e, (ids == e).nonzero(as_tuple=True)


def experts_fp4(
    xq: torch.Tensor,
    x_sf: torch.Tensor,
    x_scale: torch.Tensor,
    ids: torch.Tensor,
    ex: Fp4Experts,
    comm: Comm,
) -> torch.Tensor:
    """The NVFP4 MoE phase -> zb [m, k, H] bf16: slot (t, j)'s expert ids[t, j] output for token t, on every rank.
        g1 (rank's I slice): gate = bf16(acc_g * g1a x_scale_t), up = bf16(acc_u * g1a x_scale_t),
                             h = silu(gate) * up * g1c                     (kernels/moe.cuh: the g1 epilogue)
        h quantized per 16-group (``h_quant``), the planes gathered over the ranks, read as the full row (``h_dequant``)
        g2 (rank's H slice, full I): zb = bf16(acc * g2a), gathered over the ranks (the g2 epilogue)
    The MMAs run in scaled space (codes times group scales); x_scale and the weight scales enter as the epilogue's
    alphas."""
    m, k = ids.shape
    x = dequant_nvfp4(xq, x_sf)
    i_s, h_s = ex.w1.shape[1] * FBW, ex.w2.shape[1] * 128
    codes = torch.zeros(m, k, i_s // 2, dtype=torch.uint8, device=xq.device)
    sfb = torch.zeros(m, k, i_s // NVFP4_GROUP, dtype=torch.uint8, device=xq.device)
    eb = torch.zeros(m, k, i_s // NVFP4_GROUP, dtype=torch.int32, device=xq.device)
    for e, slot in _slots_by_expert(ids):
        gate_q, gate_sf, up_q, up_sf = unpack_w1_fp4(
            ex.w1[e : e + 1], ex.w1_sf[e * (i_s // FBW) : (e + 1) * (i_s // FBW)]
        )
        xs, a1 = x[slot[0]], (ex.g1_alphas[e] * x_scale[slot[0]])[:, None]
        gate = _bf16(_mm(xs, dequant_nvfp4(gate_q[0], gate_sf[0])).float() * a1).float()
        up = _bf16(_mm(xs, dequant_nvfp4(up_q[0], up_sf[0])).float() * a1).float()
        codes[slot], sfb[slot], eb[slot] = h_quant(
            (_silu(gate) * up) * ex.g1_scale_c[e]
        )
    h = h_dequant(
        comm.all_gather(codes, 2), comm.all_gather(sfb, 2), comm.all_gather(eb, 2)
    )
    zb = torch.zeros(m, k, h_s, dtype=torch.bfloat16, device=xq.device)
    for e, slot in _slots_by_expert(ids):
        w2_q, w2_sf = unpack_w2_fp4(ex.w2[e : e + 1], ex.w2_sf[e : e + 1])
        zb[slot] = _bf16(
            _mm(h[slot], dequant_nvfp4(w2_q[0], w2_sf[0])).float() * ex.g2_alphas[e]
        )
    return comm.all_gather(zb, 2)


def experts_bf16(
    x: torch.Tensor, ids: torch.Tensor, ex: Bf16Experts, comm: Comm
) -> torch.Tensor:
    """The bf16 MoE phase -> zb [m, k, H] bf16: h = bf16(silu(bf16(acc_g)) * bf16(acc_u)) on the rank's I slice,
    gathered; zb = bf16(acc) on the rank's H slice, gathered (kernels/moe.cuh: g2's epilogues).
    """
    m, k = ids.shape
    i_s, h_s = ex.w1.shape[1] * FBW, ex.w2.shape[1] * 128
    z = torch.zeros(m, k, i_s, dtype=torch.bfloat16, device=x.device)
    for e, slot in _slots_by_expert(ids):
        gate_w, up_w = unpack_w1_bf16(ex.w1[e : e + 1])
        xs = x[slot[0]]
        gate, up = _bf16(_mm(xs, gate_w[0])).float(), _bf16(_mm(xs, up_w[0])).float()
        z[slot] = _bf16(_silu(gate) * up)
    z = comm.all_gather(z, 2)
    zb = torch.zeros(m, k, h_s, dtype=torch.bfloat16, device=x.device)
    for e, slot in _slots_by_expert(ids):
        zb[slot] = _bf16(_mm(z[slot], unpack_w2_bf16(ex.w2[e : e + 1])[0]))
    return comm.all_gather(zb, 2)


def combine(
    zb: torch.Tensor, w: torch.Tensor, gamma_p1: torch.Tensor, eps: float = COMBINE_EPS
) -> torch.Tensor:
    """The per-expert post-norm and the weighted combine -> y [m, H] bf16 (kernels/moe.cuh):
    per slot xg = zb * (gamma + 1) and inv = 1 / sqrt(mean(xg^2) + 1e-8); y = bf16(sum_j (w_j inv_j) xg_j), one fp32
    fma accumulator over j ascending."""
    xg = zb.float() * gamma_p1.float()
    sc = w[..., None] * (1.0 / torch.sqrt(_mean_sq(xg) + eps))
    acc = torch.zeros_like(xg[:, 0])
    for j in range(xg.shape[1]):
        acc = _fma(sc[:, j], xg[:, j], acc)
    return _bf16(acc)


def qkv_head(
    h3: torch.Tensor,
    w_qkv: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    q_scale: float,
    eps: float,
    cos_sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A layer's head, run by its predecessor's launch (layer 0: the entry's qkv_rowops) -> the qkv row [m, (2 q + 2 kv)
    64] bf16; its k / v heads are the window's 16 KV rows the same launch appends to the layer's pool
    (kernels/framework.cuh). qkv = bf16(h3 @ w_qkv^T), then per q and k head:
        x = bf16(x * scale * rsqrt(mean(x^2) + eps))      scale = q_scale on q, 1 on k   (kernels/framework.cuh)
        swa:    GPT-J RoPE of the bf16 pairs (2p, 2p+1) by cos_sin [m, 64] = {cos[32] | sin[32]}, rounded to bf16
        global: no RoPE (the kernel runs cos 1 / sin 0, an identity)
    v and the o-gate pass as bf16(h3 @ w^T)."""
    m = h3.shape[0]
    qkv = _bf16(_mm(h3, w_qkv))
    q, k, v, og = split_qkv(qkv, q_heads, kv_heads)

    def norm_rope(x: torch.Tensor, scale: float) -> torch.Tensor:
        f = x.float()
        out = _bf16(f * (scale * torch.rsqrt(_mean_sq(f) + eps)))
        if cos_sin is None:
            return out
        n, c, s = out.float(), cos_sin[:, None, :32], cos_sin[:, None, 32:]
        x0, x1 = n[..., 0::2], n[..., 1::2]
        return _bf16(
            torch.stack((_fma(x0, c, -(x1 * s)), _fma(x1, c, x0 * s)), dim=-1).view(
                n.shape
            )
        )

    q, k = norm_rope(q, q_scale), norm_rope(k, 1.0)
    return torch.cat([t.reshape(m, -1) for t in (q, k, v, og)], dim=1)


# ---------------------------------------------------------------------------------------------------------------------
# One layer, the entry, the logits


@dataclasses.dataclass
class LayerWeights:
    """One rank's weights of one decoder layer, in the kernels' layouts (H = hidden, P = moe_hidden). Norm vectors are
    the effective weights (gain centering applied); alpha / beta are the norm-preserving residual's cached
    coefficients (as SGLang's model precomputes them). On the last layer the x_* vectors are the final norm's.
    """

    w_o: torch.Tensor  # [H, q_heads 64] bf16: the rank's heads of the o_proj K
    post_alpha: torch.Tensor  # [H] f32: the post-attention residual gate
    post_beta: torch.Tensor  # [H] f32
    post_pre_w: torch.Tensor  # [H] bf16: the pre-MoE norm
    w_pool: torch.Tensor  # [P, H] bf16, replicated
    w_router: torch.Tensor  # [E, H] f32, replicated
    expert_bias: torch.Tensor  # [E] f32
    router_mult: float
    route_scale: float
    pre_gamma: torch.Tensor  # [P] bf16: the pre-expert norm
    pre_eps: float
    experts: Union[Fp4Experts, Bf16Experts]
    gamma_p1: torch.Tensor  # [P] bf16: the post-expert norm weight + 1
    w_unpool: torch.Tensor  # [H, P] bf16, replicated
    x_alpha: torch.Tensor  # [H] f32: the cross-layer residual gate
    x_beta: torch.Tensor  # [H] f32
    x_pre_w: torch.Tensor  # [H] bf16: the next layer's pre-attention norm
    x_post_w: torch.Tensor  # [H] bf16: the post-FFN input-scale norm


@dataclasses.dataclass
class LayerInputs:
    """One rank's inputs to one layer launch: ``qkv`` = this layer's head [m, (2 q + 2 kv) 64] bf16 (``qkv_head`` of
    the previous launch), ``k_ctx`` / ``v_ctx`` = its KV rows before the window [n, kv_heads, 64] bf16 in position
    order, ``stream_in`` = the residual [m, H] f32, ``window`` = 2047 on swa layers, None on global ones.
    """

    qkv: torch.Tensor
    k_ctx: torch.Tensor
    v_ctx: torch.Tensor
    stream_in: torch.Tensor
    window: Optional[int]


@dataclasses.dataclass
class NextHead:
    """The successor layer's head, which this layer's launch computes (none on the last layer): ``w_qkv`` = its rows
    [q | k | v | o-gate] [(2 q + 2 kv) 64, H] bf16, ``q_scale`` = the checkpoint's scale_query_by over sqrt(64);
    a swa successor takes RoPE rows ``cos_sin`` [m, 64] f32 (``rope_cos_sin``), a global one none.
    """

    w_qkv: torch.Tensor
    q_scale: float
    eps: float = RMS_EPS
    cos_sin: Optional[torch.Tensor] = None
    q_heads: int = 16
    kv_heads: int = 2


# the values ``ref_layer`` returns, in data order: each one can be taken from the kernel instead (``forced``)
LAYER_VALUES = (
    "x_attn",
    "branch",
    "stream_mid",
    "h3",
    "pooled",
    "logits",
    "ids",
    "w",
    "xq",
    "x_sf",
    "x_scale",
    "x",
    "zb",
    "y",
    "ffn_out",
    "stream_out",
    "h3_next",
    "qkv_next",
)


def ref_layer(
    w: LayerWeights,
    x: LayerInputs,
    comm: Comm,
    nxt: Optional[NextHead] = None,
    *,
    forced: Optional[dict] = None,
    round_ar_partials: bool = True,
    b_live: Optional[int] = None,
) -> dict:
    """One decoder layer's launch on one rank -> its named values (``LAYER_VALUES``: the ones the kernel stores, plus
    the AR sum and the expert outputs it keeps on chip):
        x_attn (the attention type's arena), branch (the o_proj sum), stream_mid / h3 (the post-attention stream_out /
        h3; h3f = h3 as f32), pooled (the gathered row), logits, ids / w (idx32 / topk_w), xq / x_sf / x_scale (fp4) or
        x (bf16) (the MoE phase's input), zb (the gathered expert outputs), y, ffn_out (the gathered row), stream_out /
        h3_next (the attention type's next-layer buffers; on the last layer the final norm's residual and normed rows),
        qkv_next (the successor's head: arena qkv_out, its k / v heads the appended KV rows).
    ``forced`` = values read back from the kernel, used downstream in place of the reference's (the returned values
    stay the reference's own): ``ids`` (idx32) keeps a near tie from routing differently; forcing every stage's inputs
    checks each stage alone, which the comparison needs across an NVFP4 quantizer (``TOLERANCES``).
    ``b_live``: only the first b_live rows take experts; a pad row's zb is 0, so its y is 0 (the front item, which
    builds the route tables, and the combine skip rows >= b_live: kernels/moe.cuh: the front, combine_rows). Every
    other stage runs on every row.
    """
    f = dict(forced or {})
    assert set(f) <= set(LAYER_VALUES), set(f) - set(LAYER_VALUES)
    o: dict = {}

    def v(name: str) -> torch.Tensor:
        return f.get(name, o[name])

    o["x_attn"] = attention(x.qkv, x.k_ctx, x.v_ctx, x.window)
    o["branch"] = o_proj_allreduce(v("x_attn"), w.w_o, comm, round_ar_partials)
    o["stream_mid"], o["h3"] = residual_norm(
        v("branch"), x.stream_in, w.post_alpha, w.post_beta, w.post_pre_w
    )
    o["pooled"] = node_sharded_linear(v("h3"), w.w_pool, comm)
    o["logits"] = router(v("h3"), w.w_router)
    o["ids"], o["w"] = topk(
        v("logits"), w.router_mult, w.expert_bias, w.route_scale, ids=f.get("ids")
    )
    nrm = pre_expert_norm(v("pooled"), w.pre_gamma, w.pre_eps)
    live = slice(None, b_live)
    if isinstance(w.experts, Fp4Experts):
        o["xq"], o["x_sf"], o["x_scale"] = quantize_expert_input(nrm)
        zb = experts_fp4(
            v("xq")[live],
            v("x_sf")[live],
            v("x_scale")[live],
            v("ids")[live],
            w.experts,
            comm,
        )
    else:
        o["x"] = nrm
        zb = experts_bf16(v("x")[live], v("ids")[live], w.experts, comm)
    pads = nrm.shape[0] - zb.shape[0]
    o["zb"] = torch.cat((zb, zb.new_zeros(pads, *zb.shape[1:])))
    o["y"] = combine(v("zb"), v("w"), w.gamma_p1)
    o["ffn_out"] = node_sharded_linear(v("y"), w.w_unpool, comm)
    o["stream_out"], o["h3_next"] = residual_norm(
        v("ffn_out"), v("stream_mid"), w.x_alpha, w.x_beta, w.x_pre_w, w.x_post_w
    )
    if nxt is not None:
        o["qkv_next"] = qkv_head(
            v("h3_next"),
            nxt.w_qkv,
            nxt.q_heads,
            nxt.kv_heads,
            nxt.q_scale,
            nxt.eps,
            nxt.cos_sin,
        )
    return o


def ref_entry(
    token_ids: torch.Tensor,
    embed_shard: torch.Tensor,
    prenorm_w: torch.Tensor,
    comm: Comm,
    embed_eps: float = RMS_EPS,
    prenorm_eps: float = RMS_EPS,
) -> dict:
    """The verify window's entry -> {emb, normed, residual}: the vocab-sharded lookup (rank r holds rows
    [r V/8, (r+1) V/8); a miss reads 0) summed over the ranks (exact: one rank holds each row), the weightless embed
    norm bf16(x rsqrt(mean(x^2) + eps)) (the embedding multiplier is 1), residual = its fp32 copy, normed =
    bf16(residual rsqrt(mean(residual^2) + eps) prenorm_w) (layer 0's pre-attention norm); all in one launch here
    (kernels/head.cu). Layer 0's head is then ``qkv_head(normed, ...)``."""
    rows = embed_shard.shape[0]
    local = token_ids.long() - comm.rank * rows
    hit = (local >= 0) & (local < rows)
    part = torch.where(
        hit[:, None],
        embed_shard[local.clamp(0, rows - 1)],
        embed_shard.new_zeros(()),
    )
    emb = _bf16(comm.all_gather(part[None], 0).float().sum(0))
    e = emb.float()
    residual = _bf16(e * torch.rsqrt(_mean_sq(e) + embed_eps)).float()
    normed = _bf16(
        (residual * torch.rsqrt(_mean_sq(residual) + prenorm_eps)) * prenorm_w.float()
    )
    return {"emb": emb, "normed": normed, "residual": residual}


def ref_logits(
    h: torch.Tensor,
    lm_head_shard: torch.Tensor,
    comm: Comm,
    logit_scale: float,
    softcap: float,
) -> torch.Tensor:
    """The logits -> f32 [m, V] on every rank: the rank's vocab shard h @ W_r^T in fp32, gathered over the ranks, then
    softcap * tanh(x * logit_scale / softcap) (kernels/head.cu softcap_body; SGLang's fp32 logits path).
    """
    x = comm.all_gather(_mm(h, lm_head_shard).float(), 1)
    return torch.tanh((x * logit_scale) / softcap) * softcap


def rope_cos_sin(positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """The RoPE rows the window feeds gather: [m, 64] f32 = {cos[32] | sin[32]} of position * theta^(-2p/64)
    (SGLang's fp32 cos / sin cache)."""
    inv = 1.0 / (
        theta
        ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=positions.device)
            / HEAD_DIM
        )
    )
    f = positions.float()[:, None] * inv[None, :]
    return torch.cat((f.cos(), f.sin()), dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# Comparison

# Row-relative L2 bounds, max over rows of |got_r - ref_r| / |ref_r|, for an output against the reference stage fed
# the kernel's own inputs to that stage (``ref_layer(forced=...)``). The two share every bf16 and fp32 rounding point,
# so they part only where the kernel's fp32 accumulation order or its approximate ops (ex2, rsqrt, rcp) move a value
# across a rounding boundary: a few one-ulp bf16 flips (2^-8) per row -> 4e-3. zb crosses the MoE phase's h requant,
# whose e2m1 codes flip where gate / up did (a code step is up to 1/3 of a value) -> 2e-2. The router multiplies exact
# tf32 x bf16 products and the lm_head exact bf16 x bf16 products, both into fp32 -> fp32 order alone.
# Across a quantizer the bounds do not compose: rounding every o_proj partial to bf16 (2.8e-3 on every element of the
# AR sum) moved the reduced-dims FP4 layer's zb / y by 0.10-0.15 and the bf16 layer's by 0.007 (CPU, seeds 11-13: an
# NVFP4 quantizer turns a perturbation d of its input into ~sqrt(d) downstream). Bounds from that analysis; the first
# GPU comparison measures them.
TOLERANCES = {
    "x_attn": 4e-3,
    "branch": 4e-3,
    "stream_mid": 4e-3,
    "h3": 4e-3,
    "pooled": 4e-3,
    "logits": 1e-4,
    "w": 1e-4,
    "x": 4e-3,
    "x_scale": 4e-3,
    "zb": 2e-2,
    "y": 4e-3,
    "ffn_out": 4e-3,
    "stream_out": 4e-3,
    "h3_next": 4e-3,
    "qkv_next": 4e-3,
    "emb": 0.0,
    "normed": 4e-3,
    "residual": 4e-3,
    "logits_out": 1e-4,
}


def compare(
    got: torch.Tensor, ref: torch.Tensor, name: str, tol: Optional[float] = None
) -> tuple[float, float, bool]:
    """(max_abs, max_rel, ok) of a kernel output against the reference; a 1-D tensor is one row. Float tensors:
    max_rel = the largest row-relative L2 error, ok = max_rel <= tol (``TOLERANCES[name]`` by default) and no NaN.
    Integer tensors (ids, codes): max_rel = the fraction of rows that differ, ok = equal.
    """
    assert got.shape == ref.shape, (name, tuple(got.shape), tuple(ref.shape))
    if not ref.is_floating_point():
        diff = got.long() - ref.long()
        rows = diff.reshape(diff.shape[0], -1) if diff.dim() > 1 else diff[None]
        return (
            float(diff.abs().max()),
            float((rows != 0).any(-1).float().mean()),
            bool((diff == 0).all()),
        )
    g, r = got.to(ACC), ref.to(ACC)
    rows_d = (g - r).reshape(g.shape[0], -1) if g.dim() > 1 else (g - r)[None]
    rows_r = r.reshape(r.shape[0], -1) if r.dim() > 1 else r[None]
    num, den = rows_d.norm(dim=-1), rows_r.norm(dim=-1)
    rel = torch.where(
        den > 0, num / den.clamp_min(1e-300), torch.where(num > 0, float("inf"), 0.0)
    )
    max_abs, max_rel = float((g - r).abs().max()), float(rel.max())
    tol = TOLERANCES[name] if tol is None else tol
    return max_abs, max_rel, bool(max_rel <= tol and not torch.isnan(g).any())
