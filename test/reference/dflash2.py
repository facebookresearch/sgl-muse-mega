"""One DFlash2 decode iteration at bs1, in plain PyTorch.

This file is the spec the DFlash kernel units are checked against: the math of SGLang's DFLASH path for a DFlash2
draft (5 non-causal sliding-window layers with grouped dynamic convolutions, a rank-256 candidate selector; the draft
borrows the target's embedding and lm_head), at batch 1 with a block of B rows (B = 8: 7 drafts, B = 16: 15 drafts).
One iteration, in data order:

    (A) context update  ctx = hidden_norm(fc(F)), F = the last verify's 5 captured target features of each committed
                        row; per draft layer K = RoPE(k_norm(ctx Wk)), V = ctx Wv, written to the draft KV
    (B) draft block     x = E_target[anchor, mask x (B-1)] at positions P..P+B-1; per layer
                          r += x; h = conv_in(rms(r)); q, k, v = h Wqkv with q/k head norm + RoPE; block K/V -> KV
                          o = conv_out(allreduce(attn(q; keys [p-2047, P+B-1], non-causal) Wo))
                          r += o; h = conv_in'(rms(r)); x = conv_out'(allreduce((silu(h Wg) * h Wu) Wd))
                        y = rms_final(r + x)
    (C) head, selector  rows 1..B-1: target lm_head -> global top-16 -> u = cap tanh(mult v / cap); the lattice
                        score[e, p, c] = u[e, c] + <A[pred(e, p)] * W_P y_e, S[cand(e, c)]>, walked greedily
    (D) greedy accept   t = argmax(verify logits); acc = #leading d_i == t_(i-1); bonus = t[acc]; commit = acc + 1

The verify itself (target forward of [anchor, drafts] -> post-softcap fp32 logits + the captured features F') is the
target stack's business; ``run_iteration`` takes it as a callable. F' is the raw residual each capture layer outputs
(the public checkpoint's convention, read at its ``target_layer_ids``); this file consumes F' as given. The block
embeds with the raw target embedding row, as OSS does.

Citations name files of SGLang's python/sglang tree: M/ = srt/models/dflash.py, W/ =
srt/speculative/dflash_worker_v2.py, U/ = srt/speculative/dflash_utils.py, K/ = kernels/ops/speculative/dflash.py,
F/ = kernels/ops/speculative/fused_kv_materialize.py, L/ = srt/layers/.

Precision policy (the one muse_spark.py uses). Every dot product accumulates in fp64 and is rounded once to the dtype
SGLang stores; every elementwise step is fp32 in SGLang's operation order. ``act`` is the activation storage dtype:
bf16 reproduces SGLang's rounding points (each op's bf16 output, the bf16 draft KV), fp32 is the same math with no
storage rounding. SGLang conventions this pins down:
  - top-k ties go to the smaller vocab id (SGLang: flashinfer radix top-k ``deterministic=True`` then ``torch.topk``,
    whose tie order is implementation-defined, M/); argmax ties go to the first index, NaN first
    (torch.argmax, the walk K/);
  - the block's q/k are rounded to bf16 after the head norm and before RoPE (U/); the context K is not
    (F/);
  - the conv is one inductor-fused fp32 expression, rounded once at its output (M/).

Tensor parallelism. Every function runs on ONE rank's shard; ``Comm.all_gather`` (rank order) is the only cross-rank
move, and the same code runs unsharded with ``SoloComm`` (ranks = 1). The shards (``shard``): qkv / gate_up by output
heads / rows, o_proj / down by input columns, embedding / lm_head by vocab rows, fc by output rows (SGLang replicates
fc; any output split is exact, as it would be for the conv projections), conv projections, norms and the selector
replicated. Where TP changes the numerics:
  1. o_proj and down are row-parallel: each rank's partial is rounded to ``act`` and the partials are summed in rank
     order in fp32 (``all_reduce``), where the unsharded form rounds one full dot product. These are the only two
     reduction-order changes in the draft (10 per step). At bf16 they move y by a few ulps (about 1 % per row on the
     test configs), which can move a top-16 boundary candidate or a near-tie walk step: draft tokens are NOT
     invariant to the reduction order. Everything after y is, so a check holds y (and the head logits) to a tolerance
     and checks tokens by running ``candidates`` / ``lattice`` / ``selector_walk`` on the kernel's own y.
  2. The embedding all-reduce has one non-zero partial per element, so it is exact.
  3. The head's local top-16 + gather + global top-16 equals the unsharded top-16 under the tie rule above.
  4. Kernel variants that WOULD change the order, so must be checked with a tolerance: the conv finish applied to
     per-rank partials before the all-reduce (linear, but rounded per rank), a two-shot (reduce-scatter) all-reduce, and
     fc K-split by captured feature with partials accumulated across the 5 capture launches.

The DFlash kernels' operation orders in torch follow (``# ---- the kernels' operation orders ----``): each unit's
emulation or model (the entry, the qkv epilogue, the attention, the head's keys and lattice, the layer's RMS tree, the
draft head's rescoring), held against the reference above by the unit tests and used by the kernel tests' checks.
Names are prefixed by their unit where they would collide.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import struct
import threading
import types
from typing import Callable, Optional, Protocol

import torch
from sgl_muse_mega.muse_spark.sm103.dflash import head

# the accumulation dtype of every dot product (the precision policy above)
ACC = torch.float64
# _mm streams the weight through fp64 in blocks of this many output rows, bounding the transient copy
_MM_ROWS = 8192


# ---------------------------------------------------------------------------------------------------------------------
# Weights (one rank's shard, or the whole model at ranks = 1): the package's (sm103/dflash/weights.py), plus the target
# tensors the draft borrows

from sgl_muse_mega.muse_spark.sm103.dflash.weights import (  # noqa: E402
    ConvWeights,
    DraftConfig,
    DraftWeights,
    LayerWeights,
    block,
    checkpoint_layer,
    read_safetensors,
    rope_cos_sin,
    shard_layer,
)

_block = block


@dataclasses.dataclass
class TargetShared:
    """The target's tensors the draft borrows (M/, W/), vocab-sharded like the target's."""

    embed: torch.Tensor  # [V, H] (the rank's vocab rows)
    lm_head: torch.Tensor  # [V, H] (the rank's vocab rows)


def shard(
    cfg: DraftConfig, w: DraftWeights, t: TargetShared, rank: int, ranks: int
) -> tuple[DraftWeights, TargetShared]:
    """The unsharded weights -> rank's shard (the splits in the module docstring)."""
    return (
        dataclasses.replace(
            w,
            layers=[shard_layer(cfg, lw, rank, ranks) for lw in w.layers],
            fc=_block(w.fc, rank, ranks),
        ),
        TargetShared(_block(t.embed, rank, ranks), _block(t.lm_head, rank, ranks)),
    )


def load_draft(path: str, cfg: DraftConfig) -> DraftWeights:
    """The whole unsharded draft (9.8 GB at the public config)."""
    c = "candidate_selector."
    names = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    names += [
        f"{c}predecessor_codebook",
        f"{c}successor_codebook",
        f"{c}hidden_projection.weight",
    ]
    g = read_safetensors(path, names)
    return DraftWeights(
        [checkpoint_layer(path, i) for i in range(cfg.layers)],
        fc=g[names[0]],
        hidden_norm=g[names[1]],
        norm=g[names[2]],
        pred_codebook=g[names[3]],
        succ_codebook=g[names[4]],
        hidden_proj=g[names[5]],
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tensor-parallel collectives (the Comm contract of muse_spark.py: all_gather in rank order)


class Comm(Protocol):
    rank: int
    ranks: int

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Every rank's x concatenated along dim, in rank order."""


class SoloComm:
    """The unsharded model: one rank."""

    rank, ranks = 0, 1

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return x


class _Hub:
    def __init__(self, ranks: int):
        self.barrier = threading.Barrier(ranks)
        self.slots: list = [None] * ranks


class ThreadComm:
    """``Comm`` between the threads of one process (``emulate_ranks``)."""

    def __init__(self, hub: _Hub, rank: int, ranks: int):
        self._hub, self.rank, self.ranks = hub, rank, ranks

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        self._hub.slots[self.rank] = x
        self._hub.barrier.wait()  # every rank posted
        parts = list(self._hub.slots)
        self._hub.barrier.wait()  # every rank read before the slots are reused
        return torch.cat(parts, dim)


def emulate_ranks(fn: Callable[[Comm], object], ranks: int = 8) -> list:
    """Run ``fn(comm)`` once per rank, each rank a thread of this process -> the ranks' results. A rank that raises
    breaks the others' barriers; the first real error is re-raised."""
    hub, results, errors = _Hub(ranks), [None] * ranks, []

    def run(r: int) -> None:
        try:
            results[r] = fn(ThreadComm(hub, r, ranks))
        except BaseException as e:  # re-raised below, after every thread joined
            errors.append(e)
            hub.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(ranks)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
    if errors:
        raise (real or errors)[0]
    return results


def all_reduce(partial: torch.Tensor, comm: Comm, act: torch.dtype) -> torch.Tensor:
    """The row-parallel all-reduce (L/linear.py): the ranks' act-rounded partials summed in rank order in fp32,
    rounded to act."""
    parts = comm.all_gather(partial[None], 0)
    acc = parts[0].float()
    for r in range(1, parts.shape[0]):
        acc = acc + parts[r].float()
    return acc.to(act)


# ---------------------------------------------------------------------------------------------------------------------
# Elementwise blocks


def _mm(x: torch.Tensor, w: torch.Tensor, out: torch.dtype) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T accumulated in fp64, rounded once to out."""
    xa = x.to(ACC)
    return torch.cat(
        [
            (xa @ w[n : n + _MM_ROWS].to(ACC).T).to(out)
            for n in range(0, w.shape[0], _MM_ROWS)
        ],
        1,
    )


def _mean_sq(x: torch.Tensor) -> torch.Tensor:
    return x.to(ACC).square().mean(-1, keepdim=True).float()


def rms_norm(
    x: torch.Tensor, w: torch.Tensor, eps: float, act: torch.dtype
) -> torch.Tensor:
    """RMSNorm (L/layernorm.py): (x * rsqrt(mean(x^2) + eps)) * w."""
    xf = x.float()
    return (xf * torch.rsqrt(_mean_sq(xf) + eps) * w.float()).to(act)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    act: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused residual add + RMSNorm (L/layernorm.py) -> (normed, residual'): the sum is fp32, the stored
    residual is its act rounding, the norm reads the unrounded sum."""
    s = x.float() + residual.float()
    return (s * torch.rsqrt(_mean_sq(s) + eps) * w.float()).to(act), s.to(act)


def silu_mul(gate: torch.Tensor, up: torch.Tensor, act: torch.dtype) -> torch.Tensor:
    """SiluAndMul in fp32 (L/activation.py)."""
    g = gate.float()
    return (g / (1.0 + torch.exp(-g)) * up.float()).to(act)


def head_norm_rope(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    act: torch.dtype,
    round_normed: bool,
) -> torch.Tensor:
    """Per-head RMSNorm over head_dim, then neox RoPE over the full head; x [n, heads, D]. round_normed: the block's
    q / k round the normed head to act before the rotation (U/), the context K does not (F/).
    """
    xf = x.float()
    xn = xf * torch.rsqrt(_mean_sq(xf) + eps) * w.float()
    if round_normed:
        xn = xn.to(act).float()
    x1, x2 = xn.chunk(2, -1)
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(act)


def grouped_conv(
    x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, act: torch.dtype
) -> torch.Tensor:
    """The grouped dynamic depthwise conv over one block (M/): x [B, H], delta [B, taps, groups], base
    [taps, H] ->
        out[n, ch] = sum_{t <= n} (base[t, ch] + delta[n, t, ch // group]) * x[n - t, ch]
    so row n mixes rows n, n-1, ... of its own block and row 0 has no tap 1. Inductor fuses the whole expression in
    fp32 and rounds once at the output."""
    b, h = x.shape
    taps, groups = delta.shape[1:]
    rows = x.float().view(b, groups, h // groups)
    coeff = base.float().view(1, taps, groups, h // groups) + delta.float()[..., None]
    out = coeff[:, 0] * rows
    for t in range(1, taps):
        out[t:] = out[t:] + coeff[t:, t] * rows[:-t]
    return out.reshape(b, h).to(act)


def conv_prepare(
    conv: ConvWeights, h: torch.Tensor, act: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """DFlashGroupedConv.prepare (M/): one projection of the sublayer input gives both sides' coefficients
    [B, 2, taps, groups]; side 0 convolves the input now, side 1 is returned for ``conv_finish``.
    """
    taps = conv.base.shape[1]
    c = _mm(h, conv.proj, act).view(h.shape[0], 2, taps, -1)
    return grouped_conv(h, c[:, 0], conv.base[0], act), c[:, 1]


def conv_finish(
    conv: ConvWeights, x: torch.Tensor, coeff: torch.Tensor, act: torch.dtype
) -> torch.Tensor:
    """DFlashGroupedConv.finish (M/), on the all-reduced sublayer output (M/)."""
    return grouped_conv(x, coeff, conv.base[1], act)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    window_left: int,
) -> torch.Tensor:
    """Non-causal sliding-window GQA (ENCODER_ONLY, M/; FA window (left, left), L/attention/
    flashattention_backend.py): q [B, qh, D], k / v [n, kvh, D] -> [B, qh D] fp64. A query at p sees keys
    p - window_left <= pos <= p + window_left; q head h reads kv head h // (qh / kvh); scale D^-0.5 (M/).
    """
    b, qh, d = q.shape
    kvh = k.shape[1]
    seen = (pos_k[None, :] >= pos_q[:, None] - window_left) & (
        pos_k[None, :] <= pos_q[:, None] + window_left
    )
    qg = q.to(ACC).view(b, kvh, qh // kvh, d)
    s = torch.einsum("bkgd,nkd->kgbn", qg, k.to(ACC)) * d**-0.5
    p = torch.softmax(s.masked_fill(~seen, float("-inf")), dim=-1)
    return torch.einsum("kgbn,nkd->bkgd", p, v.to(ACC)).reshape(b, qh * d)


def embed(
    ids: torch.Tensor, table: torch.Tensor, comm: Comm, act: torch.dtype
) -> torch.Tensor:
    """VocabParallelEmbedding (L/vocab_parallel_embedding.py): masked lookup in the rank's vocab rows, then the
    all-reduce (exact: one non-zero partial). No noise scale for this target (W/).
    """
    n = table.shape[0]
    local = ids - comm.rank * n
    hit = (local >= 0) & (local < n)
    rows = table[local.clamp(0, n - 1)].to(act).masked_fill(~hit[:, None], 0)
    return all_reduce(rows, comm, act)


def softcap(x: torch.Tensor, multiplier: float, cap: float) -> torch.Tensor:
    """cap * tanh(multiplier x / cap) in fp32 (M/; the target's own logit transform)."""
    return torch.tanh(x.float() * multiplier / cap) * cap


def first_argmax(x: torch.Tensor) -> torch.Tensor:
    """torch.argmax over the last dim: the first NaN if any, else the first maximum (K/)."""
    nan = torch.isnan(x)
    best = x.masked_fill(nan, float("-inf")).amax(-1, keepdim=True)
    hit = torch.where(nan.any(-1, keepdim=True), nan, x == best)
    idx = torch.arange(x.shape[-1]).expand_as(x)
    return idx.masked_fill(~hit, x.shape[-1]).amin(-1)


def topk_low_id(
    vals: torch.Tensor, ids: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The k largest vals per row with their ids, descending, ties to the smaller id."""
    by_id = torch.argsort(ids, dim=-1, stable=True)
    vals, ids = vals.gather(-1, by_id), ids.gather(-1, by_id)
    top = torch.argsort(vals, dim=-1, descending=True, stable=True)[..., :k]
    return vals.gather(-1, top), ids.gather(-1, top)


# ---------------------------------------------------------------------------------------------------------------------
# The draft KV and the four stages


@dataclasses.dataclass
class DraftKV:
    """The draft's KV, contiguous by position: per layer K / V [capacity, the rank's kv heads, D] in act (SGLang's draft
    pool is bf16). Positions [0, P) hold context rows; the block writes [P, P + B)."""

    k: list[torch.Tensor]
    v: list[torch.Tensor]

    @classmethod
    def empty(
        cls, cfg: DraftConfig, capacity: int, ranks: int, act: torch.dtype
    ) -> DraftKV:
        shape = (capacity, cfg.kv_heads // ranks, cfg.head_dim)
        return cls(
            [torch.zeros(shape, dtype=act) for _ in range(cfg.layers)],
            [torch.zeros(shape, dtype=act) for _ in range(cfg.layers)],
        )


def _heads(cfg: DraftConfig, lw: LayerWeights) -> tuple[int, int]:
    """(q, kv) heads of this shard, read off the weights."""
    q = lw.o.shape[1] // cfg.head_dim
    return q, (lw.qkv.shape[0] // cfg.head_dim - q) // 2


def context_update(
    cfg: DraftConfig,
    w: DraftWeights,
    feats: torch.Tensor,
    pos0: int,
    comm: Comm,
    act: torch.dtype,
    mm: Callable[[torch.Tensor, torch.Tensor, torch.dtype], torch.Tensor] = _mm,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(A) feats [n, features H] (row i at position pos0 + i) -> per layer (K [n, kvh, D], V [n, kvh, D]).
    ctx = hidden_norm(fc(feats)) is shared by every layer (M/; no per-layer norm, no conv, M/); each
    layer projects it through the k / v rows of its qkv (M/; every layer's in one GEMM, their dot products
    independent) and norms + ropes K (F/). ``mm``: the GEMMs, ``_mm``'s contract (another
    implementation of its fp64-class dot products).
    """
    ctx = rms_norm(
        comm.all_gather(mm(feats, w.fc, act), 1), w.hidden_norm, cfg.rms_eps, act
    )
    n, d = feats.shape[0], cfg.head_dim
    # the rope rows on the host (the kernels' tables are host-built), moved to the features' device (pinned: no
    # host wait)
    cos, sin = (
        (t.pin_memory() if ctx.is_cuda else t).to(ctx.device, non_blocking=True)
        for t in rope_cos_sin(pos0 + torch.arange(n), d, cfg.rope_theta)
    )
    heads = [_heads(cfg, lw) for lw in w.layers]
    rows = torch.cat([lw.qkv[q * d :] for lw, (q, _) in zip(w.layers, heads)])
    kvs = mm(ctx, rows, act).split([2 * kvh * d for _, kvh in heads], 1)
    out = []
    for lw, (_, kvh), kv in zip(w.layers, heads, kvs):
        kv = kv.view(n, 2, kvh, d)
        k = head_norm_rope(
            kv[:, 0], lw.k_norm, cfg.rms_eps, cos, sin, act, round_normed=False
        )
        out.append((k, kv[:, 1]))
    return out


def write_context(
    kv: DraftKV, ctx: list[tuple[torch.Tensor, torch.Tensor]], pos0: int, rows: int
) -> None:
    """The first ``rows`` context rows into the KV at pos0.. (SGLang writes commit_len rows, W/)."""
    for layer, (k, v) in enumerate(ctx):
        kv.k[layer][pos0 : pos0 + rows] = k[:rows]
        kv.v[layer][pos0 : pos0 + rows] = v[:rows]


def sublayer_out(
    conv: ConvWeights,
    x: torch.Tensor,
    w: torch.Tensor,
    coeff: torch.Tensor,
    comm: Comm,
    act: torch.dtype,
    order: str,
) -> torch.Tensor:
    """A row-parallel sublayer's output, conv finished. order 'post_ar' is SGLang's (each rank's partial rounded to
    act, all-reduced, then finished: M/); 'pre_ar' is the kernels' (each rank's fp32 partial
    finished and rounded once, then all-reduced: linear and rank-independent, so the same math rounded per rank).
    """
    if order == "post_ar":
        return conv_finish(conv, all_reduce(_mm(x, w, act), comm, act), coeff, act)
    assert order == "pre_ar", order
    return all_reduce(
        conv_finish(conv, _mm(x, w, torch.float32), coeff, act), comm, act
    )


def draft_layer(
    cfg: DraftConfig,
    lw: LayerWeights,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    pos: int,
    comm: Comm,
    act: torch.dtype,
    order: str = "post_ar",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DFlashDecoderLayer (M/) over the block rows x [B, H] at positions pos.. -> (x', residual').
    residual None = layer 0 (the input is the embedding). Writes the block's K / V into kv_k / kv_v [pos, pos + B)
    before attending (RadixAttention); the attention reads the KV rows the window can reach. ``order``: the two
    sublayers' all-reduce / conv finish order (``sublayer_out``).
    """
    if residual is None:
        residual, h = x, rms_norm(x, lw.input_norm, cfg.rms_eps, act)
    else:
        h, residual = add_rms_norm(x, residual, lw.input_norm, cfg.rms_eps, act)
    return draft_layer_normed(cfg, lw, h, residual, kv_k, kv_v, pos, comm, act, order)


def draft_layer_normed(
    cfg: DraftConfig,
    lw: LayerWeights,
    h: torch.Tensor,
    residual: torch.Tensor,
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    pos: int,
    comm: Comm,
    act: torch.dtype,
    order: str = "post_ar",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``draft_layer`` from its normed input h (the input norm applied) and the residual it was normed from."""
    b, d, eps = h.shape[0], cfg.head_dim, cfg.rms_eps
    qh, kvh = _heads(cfg, lw)
    h, c_out = conv_prepare(lw.attn_conv, h, act)
    q, k, v = _mm(h, lw.qkv, act).split([qh * d, kvh * d, kvh * d], 1)
    positions = pos + torch.arange(b)
    cos, sin = rope_cos_sin(positions, d, cfg.rope_theta)
    q = head_norm_rope(
        q.reshape(b, qh, d), lw.q_norm, eps, cos, sin, act, round_normed=True
    )
    kv_k[pos : pos + b] = head_norm_rope(
        k.reshape(b, kvh, d), lw.k_norm, eps, cos, sin, act, round_normed=True
    )
    kv_v[pos : pos + b] = v.reshape(b, kvh, d)
    lo = max(0, pos - cfg.window_left)  # the lowest position any block row sees
    o = attention(
        q,
        kv_k[lo : pos + b],
        kv_v[lo : pos + b],
        positions,
        torch.arange(lo, pos + b),
        cfg.window_left,
    ).to(act)
    a = sublayer_out(lw.attn_conv, o, lw.o, c_out, comm, act, order)
    h, residual = add_rms_norm(a, residual, lw.post_norm, eps, act)
    h, c_out = conv_prepare(lw.mlp_conv, h, act)
    gate, up = _mm(h, lw.gate_up, act).chunk(2, 1)
    m = silu_mul(gate, up, act)  # M/
    return sublayer_out(lw.mlp_conv, m, lw.down, c_out, comm, act, order), residual


def draft_block(
    cfg: DraftConfig,
    w: DraftWeights,
    t: TargetShared,
    kv: DraftKV,
    anchor: int,
    pos: int,
    block: int,
    comm: Comm,
    act: torch.dtype,
    order: str = "post_ar",
) -> torch.Tensor:
    """(B) The block forward (M/): ids [anchor, mask x (block-1)] (K/) at positions pos.. -> the
    final-normed hidden y [block, H]. Row 0 runs every layer: its K / V and its conv tap feed row 1.
    """
    ids = torch.full((block,), cfg.mask_id, dtype=torch.int64)
    ids[0] = anchor
    x, residual = embed(ids, t.embed, comm, act), None
    for layer, lw in enumerate(w.layers):
        x, residual = draft_layer(
            cfg, lw, x, residual, kv.k[layer], kv.v[layer], pos, comm, act, order
        )
    return add_rms_norm(x, residual, w.norm, cfg.rms_eps, act)[0]


def candidates(
    cfg: DraftConfig, t: TargetShared, y: torch.Tensor, comm: Comm, act: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """(C, head) y [γ, H] = the block's rows 1..B-1 (W/) -> (global candidate ids [γ, K] int64, unary scores
    [γ, K] fp32). Logits through the rank's lm_head rows (act, M/), local top-K, gather of the K values and
    global ids, global top-K (M/), then the softcap (M/)."""
    logits = _mm(y, t.lm_head, act).float()
    n = t.lm_head.shape[0]
    ids = (torch.arange(n) + comm.rank * n).expand_as(logits)
    vals, ids = topk_low_id(logits, ids, cfg.top_k)
    vals, ids = topk_low_id(
        comm.all_gather(vals, 1), comm.all_gather(ids, 1), cfg.top_k
    )
    return ids, softcap(vals, cfg.output_multiplier, cfg.softcap)


def lattice(
    w: DraftWeights,
    y: torch.Tensor,
    cand: torch.Tensor,
    unary: torch.Tensor,
    anchor: int,
    act: torch.dtype,
) -> torch.Tensor:
    """(C, lattice) -> score [γ, K(p), K(c)] fp32 (M/, _score_edges M/):
        score[e, p, c] = unary[e, c] + sum_r A[pred(e, p), r] * (W_P y_e)[r] * S[cand(e, c), r]
    pred(0, p) = the anchor, pred(e, p) = cand[e-1, p]. SGLang rounds W_P y and the product A * (W_P y) to act (bf16
    tensors into the compiled bmm) and the bmm output to act, then adds the fp32 unary.
    """
    hid = _mm(y, w.hidden_proj, act).float()
    pred = torch.cat(
        [torch.full((1, cand.shape[1]), anchor, dtype=cand.dtype), cand[:-1]]
    )
    prod = (w.pred_codebook[pred].float() * hid[:, None, :]).to(act)
    dot = torch.einsum("epr,ecr->epc", prod.to(ACC), w.succ_codebook[cand].to(ACC))
    return unary[:, None, :] + dot.to(act).float()


def selector_walk(
    cand: torch.Tensor,
    scores: torch.Tensor,
    greedy: bool = True,
    uniforms: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(C, walk) _selector_walk_kernel (K/) for one request -> (draft tokens [γ] int64, q [γ, K] fp32).
    Slot e reads the score row of the previous slot's pick (row 0 for slot 0: every row of slot 0 has the anchor as
    predecessor). Greedy: the first argmax, q one-hot. Sampled: index = #{c : u_e >= cumsum(softmax(row / T))_c},
    clamped to K-1."""
    slots, k = cand.shape
    tokens = torch.empty(slots, dtype=torch.int64)
    q = torch.empty(slots, k, dtype=torch.float32)
    prev = 0
    for e in range(slots):
        row = scores[e, prev].float()
        if greedy:
            idx = int(first_argmax(row))
            q[e] = (torch.arange(k) == idx).float()
        else:
            scaled = row / temperature
            ex = torch.exp(scaled - scaled.max())
            q[e] = ex / ex.sum()
            idx = min(int((uniforms[e] >= torch.cumsum(q[e], 0)).sum()), k - 1)
        tokens[e] = cand[e, idx]
        prev = idx
    return tokens, q


@dataclasses.dataclass
class Accept:
    """out_tokens [B] = d_1 .. d_(B-1), 0 with slot accept_len replaced by the bonus (K/)."""

    accept_len: int
    commit_len: int
    bonus: int
    out_tokens: torch.Tensor
    new_pos: int


def greedy_accept(tokens: torch.Tensor, logits: torch.Tensor, pos: int) -> Accept:
    """(D) tokens [B] = [anchor, d_1 .. d_(B-1)], the verify's post-softcap fp32 logits [B, V] -> the accept of
    _dflash_accept_bonus_contig_kernel (K/) on t = argmax(logits) (W/).
    """
    t = first_argmax(logits)
    b = tokens.shape[0]
    acc = 0
    while acc < b - 1 and int(tokens[acc + 1]) == int(t[acc]):
        acc += 1
    out = torch.cat([tokens[1:], torch.zeros(1, dtype=tokens.dtype)])
    out[acc] = t[acc]
    return Accept(acc, acc + 1, int(t[acc]), out, pos + acc + 1)


# ---------------------------------------------------------------------------------------------------------------------
# The sampled step (temperature > 0): the walk above with uniforms, then SGLang's chain sampling of the verify
# (kernels/ops/speculative/dspark/dspark_accept.py accept_sampling, kernels/ops/speculative/reject_sampling.py
# chain_speculative_sampling_triton, U/ build_speculative_verify_target_probs). The uniforms are the kernels' stream
# (kernels/dflash/sample.cuh): Philox4x32-10, counter {iter, stream, purpose, i / 4}, key = the 64-bit seed.

PHILOX_M = (0xD2511F53, 0xCD9E8D57)
PHILOX_W = (0x9E3779B9, 0xBB67AE85)
PURPOSE_WALK, PURPOSE_ACCEPT = 0, 1
FINAL_UNIFORM = 15  # the bonus's uniform; the accept's coins are 0 .. gamma - 1
_M32 = 0xFFFFFFFF


def philox(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Philox4x32-10 (Salmon et al., SC'11)."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0, p1 = PHILOX_M[0] * c0, PHILOX_M[1] * c2
        c0, c1, c2, c3 = (
            (p1 >> 32) ^ c1 ^ k0,
            p1 & _M32,
            (p0 >> 32) ^ c3 ^ k1,
            p0 & _M32,
        )
        k0, k1 = (k0 + PHILOX_W[0]) & _M32, (k1 + PHILOX_W[1]) & _M32
    return c0, c1, c2, c3


def uniforms(seed: int, stream: int, iteration: int, purpose: int) -> torch.Tensor:
    """u_i in [0, 1), i < 16: the top 24 bits of word i % 4 of philox({iteration, stream, purpose, i / 4}, seed)."""
    seed &= 2**64 - 1
    key = (seed & _M32, seed >> 32)
    out = []
    for i in range(16):
        w = philox((iteration & _M32, stream & _M32, purpose, i // 4), key)[i % 4]
        out.append((w >> 8) * 2.0**-24)
    return torch.tensor(out, dtype=torch.float32)


def target_probs(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """(U/ build_speculative_verify_target_probs, dspark_accept SoftmaxTemp) the verify rows' sampling distributions,
    fp64 [rows, V]: softmax(fp32(logits / T)) over the top_k largest (top_k >= V: all; equal values at the boundary all
    kept), then the top-p set within it: v stays while the probability above it is below top_p (flashinfer's
    top_p_renorm_probs: the mass of the strictly larger values), renormalized."""
    y = (logits.float() / torch.tensor(temperature, dtype=torch.float32)).double()
    rows, v = y.shape
    if top_k < v:
        kth = torch.topk(y, top_k, dim=-1).values[:, -1:]
        y = torch.where(y >= kth, y, torch.full_like(y, -math.inf))
    p = torch.softmax(y, dim=-1)
    if top_p < 1.0:
        srt, idx = torch.sort(p, dim=-1, descending=True)
        above = torch.zeros_like(p)
        for r in range(rows):
            # the mass of strictly larger probabilities: ties share their group's start
            cum = torch.cumsum(srt[r], 0) - srt[r]
            first = torch.searchsorted(-srt[r], -srt[r], side="left")
            above[r, idx[r]] = cum[first]
        p = torch.where(above < top_p, p, torch.zeros_like(p))
        p = p / p.sum(-1, keepdim=True)
    return p


@dataclasses.dataclass
class SampledAccept:
    accept_len: int
    bonus: int
    # the residual's scale (the accepted prefix's probability; 1 token-wise) and its mass
    scale: float
    norm: float


def sampled_accept(
    tokens: torch.Tensor,
    probs: torch.Tensor,
    cand: torch.Tensor,
    q: torch.Tensor,
    u: torch.Tensor,
    block: bool,
) -> SampledAccept:
    """(dspark_accept accept_sampling, chain_speculative_sampling_triton) tokens [B] = [anchor, d_1 .. d_(B-1)], probs
    [B, V] = target_probs of the verify rows, cand / q [B-1, K] = each slot's candidates and the walk's probabilities
    over them (the dense draft probabilities: q at the candidates, 0 elsewhere), u = the step's 16 uniforms.
    Block verification (Algorithm 2 of arXiv:2403.10444): the longest prefix whose probability test and residual test
    pass; token-wise: accept while u * q < p. The bonus: the CDF of max(s P_r - Q_r, 0) in vocabulary order (P_r alone
    when every draft is accepted) at u[15] times its mass."""
    gamma, v = tokens.shape[0] - 1, probs.shape[1]
    p64 = probs.double()
    dense = torch.zeros(gamma, v, dtype=torch.float64)
    dense.scatter_(1, cand.long(), q.double())
    pick = [float(p64[e, tokens[e + 1]]) for e in range(gamma)]
    qd = [float(dense[e, tokens[e + 1]]) for e in range(gamma)]

    def residual(r: int, s: float) -> torch.Tensor:
        return torch.clamp(s * p64[r] - dense[r], min=0.0)

    acc, scale, norm = 0, 1.0, -1.0
    if block:
        pp, pps = 1.0, [1.0]
        for e in range(gamma):
            pp = min(pp * pick[e] / qd[e], 1.0) if 0.0 < qd[e] <= 1.0 else 0.0
            pps.append(pp)
        for e in range(gamma, 0, -1):
            coin = float(u[e - 1])
            if coin < pps[e]:
                ap, rm = pps[e], -1.0
                if e < gamma and 0.0 < pps[e] < 1.0:
                    rm = float(residual(e, pps[e]).sum())
                    ap = rm / (rm + 1.0 - pps[e])
                if coin < ap:
                    acc, scale, norm = e, pps[e], rm
                    break
    else:
        while (
            acc < gamma and 0.0 < qd[acc] <= 1.0 and float(u[acc]) * qd[acc] < pick[acc]
        ):
            acc += 1
    dist = p64[acc] if acc == gamma else residual(acc, scale)
    if norm < 0.0:
        norm = float(dist.sum())
    cdf = torch.cumsum(dist, 0)
    hit = torch.nonzero(cdf > float(u[FINAL_UNIFORM]) * norm)
    bonus = int(hit[0]) if hit.numel() else v - 1
    return SampledAccept(acc, bonus, scale, norm)


# ---------------------------------------------------------------------------------------------------------------------
# One iteration


@dataclasses.dataclass
class DecodeState:
    """Carried between iterations (W/). pos = P, the anchor's position ([0, P) are committed tokens); feats
    [n >= commit_len, features H] = the last verify's capture, row i at position pos - commit_len + i. After prefill:
    pos = prompt length, anchor = the first sampled token, feats = the prompt's features, commit_len = pos.
    """

    pos: int
    anchor: int
    feats: torch.Tensor
    commit_len: int


@dataclasses.dataclass
class Draft:
    """tokens = [anchor, d_1 .. d_(B-1)], the verify window; ctx = (A)'s per layer K / V of every feats row."""

    tokens: torch.Tensor  # [B]
    ctx: list[tuple[torch.Tensor, torch.Tensor]]
    y: torch.Tensor  # [B, H]
    cand: torch.Tensor  # [B-1, K]
    unary: torch.Tensor  # [B-1, K]
    scores: torch.Tensor  # [B-1, K, K]
    q: torch.Tensor  # [B-1, K]


def draft_step(
    cfg: DraftConfig,
    w: DraftWeights,
    t: TargetShared,
    kv: DraftKV,
    state: DecodeState,
    block: int,
    comm: Comm,
    act: torch.dtype = torch.bfloat16,
    greedy: bool = True,
    uniforms: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    order: str = "post_ar",
) -> Draft:
    """(A) + (B) + (C): the draft's half of an iteration (``order``: draft_layer's)."""
    pos0 = state.pos - state.commit_len
    ctx = context_update(cfg, w, state.feats, pos0, comm, act)
    write_context(kv, ctx, pos0, state.commit_len)
    y = draft_block(cfg, w, t, kv, state.anchor, state.pos, block, comm, act, order)
    cand, unary = candidates(cfg, t, y[1:], comm, act)
    scores = lattice(w, y[1:], cand, unary, state.anchor, act)
    drafts, q = selector_walk(cand, scores, greedy, uniforms, temperature)
    tokens = torch.cat([torch.tensor([state.anchor]), drafts])
    return Draft(tokens, ctx, y, cand, unary, scores, q)


def run_iteration(
    cfg: DraftConfig,
    w: DraftWeights,
    t: TargetShared,
    kv: DraftKV,
    state: DecodeState,
    block: int,
    verify: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    comm: Comm,
    act: torch.dtype = torch.bfloat16,
) -> tuple[Draft, Accept, DecodeState]:
    """One greedy iteration. verify(tokens [B], pos) -> (post-softcap fp32 logits [B, V], features [B, features H])
    stands in for the target."""
    draft = draft_step(cfg, w, t, kv, state, block, comm, act)
    logits, feats = verify(draft.tokens, state.pos)
    acc = greedy_accept(draft.tokens, logits, state.pos)
    return draft, acc, DecodeState(acc.new_pos, acc.bonus, feats, acc.commit_len)


# ---- the kernels' operation orders ----

LANES, CHUNK = (
    32,
    8,
)  # the draft head's warp dot: lane l sums the 16-B chunks (8 bf16) l, l + 32, ..


# ---- the entry ----


@dataclasses.dataclass
class EntryWeights:
    """The draft weights the entry reads, unsharded as the checkpoint stores them."""

    fc: torch.Tensor  # [H, features H]
    hidden_norm: torch.Tensor  # [H]
    in_norm: torch.Tensor  # layers.0.input_layernorm [H]
    conv_base: torch.Tensor  # layers.0.attention_conv.base_kernel [2 sides, taps, H]
    proj: torch.Tensor  # layers.0.attention_conv.kernel_projection [2 taps groups, H]


def entry_reference(
    cfg: DraftConfig,
    w: EntryWeights,
    cap: list[torch.Tensor],
    emb: torch.Tensor,
    act: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """The entry's outputs for window rows cap[k][:B] and block embedding rows emb [B, H], in the reference's precision
    policy: ctx_n is context_update's ctx, xq / delta are layer 0's conv_prepare of rms(emb) (draft_layer's layer-0
    branch)."""
    b = emb.shape[0]
    feats = torch.cat([c[:b].to(act) for c in cap], 1)
    ctx_raw = _mm(feats, w.fc, act)
    h = rms_norm(emb, w.in_norm, cfg.rms_eps, act)
    delta = _mm(h, w.proj, act)
    c = delta.view(b, 2, cfg.conv_taps, cfg.conv_groups)
    return dict(
        ctx_raw=ctx_raw,
        ctx_n=rms_norm(ctx_raw, w.hidden_norm, cfg.rms_eps, act),
        h=h,
        delta=delta,
        xq=grouped_conv(h, c[:, 0], w.conv_base[0], act),
        c_att1=c[:, 1].reshape(b, -1),
        resid=emb.to(act),
    )


def entry_emulate_prep(
    h0: torch.Tensor,
    hm: torch.Tensor,
    d0: torch.Tensor,
    dm: torch.Tensor,
    base0: torch.Tensor,
    group: int,
) -> torch.Tensor:
    """The kernel's prepared rows 0-1 from its own h0 / delta row and the bind constants, in its operation order
    (fp32, no contraction, one bf16 rounding): x'[0] = (b0 + d0[t0]) h0, x'[1] = (b0 + dm[t0]) hm + (b1 + dm[t1]) h0.
    """
    g = torch.arange(h0.shape[0], device=h0.device) // group
    groups = h0.shape[0] // group
    b0, b1 = base0[0].float(), base0[1].float()
    x0 = (b0 + d0[g].float()) * h0.float()
    x1 = (b0 + dm[g].float()) * hm.float() + (b1 + dm[groups + g].float()) * h0.float()
    return torch.stack([x0, x1]).to(torch.bfloat16)


# ---- the qkv launch ----


def butterfly_sum32(x: torch.Tensor) -> torch.Tensor:
    """The warp xor butterfly (offsets 16, 8, 4, 2, 1) over the last dim of 32 fp32 values; every lane ends with this
    value (lane l and lane l ^ off add the same two operands)."""
    assert x.shape[-1] == 32 and x.dtype == torch.float32
    while x.shape[-1] > 1:
        half = x.shape[-1] // 2
        x = x[..., :half] + x[..., half:]
    return x[..., 0]


def head_norm_rope_tree(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    round_normed: bool,
) -> torch.Tensor:
    """dflash2.head_norm_rope (x [n, heads, D] bf16, cos / sin [n, D / 2] fp32) in the kernels' order:
    ss = ((S0 + S1) + S2) + S3 with S_c the butterfly of x^2 over elements [32 c, 32 c + 32), inv = 1 / sqrt(ss / D
    + eps), n = (x inv) w (rounded to bf16 when round_normed), then neox RoPE, every op an fp32 rounding.
    """
    d = x.shape[-1]
    assert d == 4 * 32, "the kernel's head is four 32-lane chunks"
    xf = x.float()
    s = butterfly_sum32((xf * xf).view(*xf.shape[:-1], 4, 32))
    ss = ((s[..., 0] + s[..., 1]) + s[..., 2]) + s[..., 3]
    root = torch.sqrt(ss * (1.0 / d) + torch.tensor(eps, dtype=torch.float32))
    inv = torch.ones_like(root) / root
    nrm = (xf * inv[..., None]) * w.float()
    if round_normed:
        nrm = nrm.to(torch.bfloat16).float()
    x1, x2 = nrm.chunk(2, -1)
    c, sn = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], -1).to(torch.bfloat16)


def qkv_rowops_ref(
    raw: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    eps: float,
    cos_sin: torch.Tensor,
    block: int,
    q_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """raw [2B, (q heads + 2) D] bf16 (the GEMM rows: block rows, then context rows), cos_sin [2B, D] fp32 -> (q [B,
    q heads, D], k [2B, D], v [2B, D]) bf16: the block's q / k rounded after the head norm, the context k not (SGLang's
    two paths, dflash2 module docstring)."""
    m, d = raw.shape[0], q_norm.shape[0]
    assert m == 2 * block and raw.shape[1] == (q_heads + 2) * d, tuple(raw.shape)
    cos, sin = cos_sin[:, : d // 2], cos_sin[:, d // 2 :]
    q, k, v = raw.split([q_heads * d, d, d], 1)
    qo = head_norm_rope_tree(
        q[:block].view(block, q_heads, d), q_norm, eps, cos[:block], sin[:block], True
    )
    kb = head_norm_rope_tree(
        k[:block].view(block, 1, d), k_norm, eps, cos[:block], sin[:block], True
    )
    kc = head_norm_rope_tree(
        k[block:].view(block, 1, d), k_norm, eps, cos[block:], sin[block:], False
    )
    return qo, torch.cat([kb, kc]).view(m, d), v.clone()


def swiglu_ref(gate_up_raw: torch.Tensor) -> torch.Tensor:
    """The GEMM's bf16 rows [rows, 2 I] in the reference's column order (gate, then up) -> dflash2.silu_mul."""
    gate, up = gate_up_raw.chunk(2, -1)
    return silu_mul(gate, up, torch.bfloat16)


def qkv_feeds(
    pos: int, block: int, commit: int, cfg: DraftConfig, ring: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A qkv launch's feeds -> (positions [2B] int64, cos_sin [2B, D] fp32, slots [2B] int32). Block row i sits at
    pos + i, context row j at pos - commit + j, for every j < B: the rows past the commit land on slots the block owns,
    so only the kernel's commit mask keeps them off."""
    positions = torch.cat(
        [pos + torch.arange(block), pos - commit + torch.arange(block)]
    )
    cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
    slots = (positions % ring).to(torch.int32)
    return positions, torch.cat([cos, sin], 1).contiguous(), slots


# ---- the draft attention ----


M_SENT = -1e30  # the kernel's running max of a row with no visible key


def attn_key_window(length: int, block: int, c: ctypes.Structure) -> tuple[int, int]:
    """(lo_w, blocks): the attention's 64-aligned key range [lo_w, lo_w + 64 x blocks) at kv_len = length = P + B (the
    device's window_of): attention CTA j reads block j."""
    lo = length - block - c.window if length > block + c.window else 0
    lo_w = lo - lo % c.block_keys
    return lo_w, -(-(length - lo_w) // c.block_keys)


def attn_to_ring(rows: torch.Tensor, length: int, ring_rows: int) -> torch.Tensor:
    """Per-position K (or V) rows [>= length, D] -> the draft ring [ring_rows, D]: position p at slot p mod ring_rows
    for the newest ring_rows positions; slots no position < length reaches stay zero."""
    ring = rows.new_zeros(ring_rows, rows.shape[1])
    p = torch.arange(max(0, length - ring_rows), length, device=rows.device)
    ring[p % ring_rows] = rows[p]
    return ring


def attn_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int, cfg: DraftConfig
) -> torch.Tensor:
    """fp64 [B, QH x D]: block row t (position pos + t) over per-position K / V [>= pos + B, D] (one kv head): the
    reference model's non-causal window (dflash2.attention)."""
    b, wl = q.shape[0], cfg.window_left
    lo = max(0, pos - wl)
    return attention(
        q,
        k[lo : pos + b, None],
        v[lo : pos + b, None],
        pos + torch.arange(b),
        torch.arange(lo, pos + b),
        wl,
    )


def attn_emulate(
    q: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    length: int,
    c: ctypes.Structure,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's decomposition on the CPU: attention CTA j takes block j of the window's aligned range from the ring;
    S = Q K^T (row = token x 8 + head), the row max over its valid keys (in log2 units), P = exp2(S x scale - m) and
    its fp32 row sum, P rounded to bf16, O = P V rounded to bf16 (the published partial); then each row's partials
    combined in the kernel's order (eighths, then the eighths in order) and x = acc / l rounded to bf16.
    Returns (x [B, QH x D] bf16, the partials' (m, l) [blocks, 8B, 2]). The device's exp2 and MMA K order differ: a
    numerics band, not bitwise."""
    b, qh, d = q.shape
    scale_log2 = d**-0.5 * math.log2(math.e)  # the softmax scale, folded into the exp2
    lo_w, blocks = attn_key_window(length, b, c)
    rows = qh * b
    tok = torch.arange(rows) // qh
    qm = q.reshape(rows, d).double()  # row = token x 8 + head
    posq = length - b + tok
    lo_r = (posq - c.window).clamp(min=0)
    parts = []
    for j in range(blocks):
        pos = lo_w + c.block_keys * j + torch.arange(c.block_keys)
        slot = pos % c.ring_rows
        s = (qm @ k_ring[slot].double().T).float()
        valid = (pos[None, :] >= lo_r[:, None]) & (pos[None, :] < length)
        s = s.masked_fill(~valid, float("-inf"))
        m = torch.clamp(s.max(1).values * scale_log2, min=M_SENT)
        e = torch.exp2(s * scale_log2 - m[:, None])
        o = (e.bfloat16().double() @ v_ring[slot].double()).bfloat16().float()
        parts.append((m, e.sum(1), o))

    def merge(acc, part):
        m0, l0, o0 = acc
        mk, lk, ok = part
        mn = torch.maximum(m0, mk)
        a, bs = torch.exp2(m0 - mn), torch.exp2(mk - mn)
        return mn, l0 * a + lk * bs, o0 * a[:, None] + ok * bs[:, None]

    fresh = (torch.full((rows,), M_SENT), torch.zeros(rows), torch.zeros(rows, d))
    eighth = -(-blocks // 8)
    total = fresh
    for w in range(8):
        acc = fresh
        for part in parts[w * eighth : min((w + 1) * eighth, blocks)]:
            acc = merge(acc, part)
        total = merge(total, acc)
    x = (total[2] / total[1][:, None]).bfloat16().reshape(b, qh * d)
    return x, torch.stack([torch.stack(p[:2], 1) for p in parts])


# ---- the head and the accept ----


def bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """bf16 values -> their bit patterns as int64."""
    return x.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def ord16(bits: torch.Tensor) -> torch.Tensor:
    """The kernels' order image of a bf16 bit pattern: a NaN above +inf, -0 equal to +0, else IEEE order."""
    b = torch.where(bits == 0x8000, torch.zeros_like(bits), bits)
    o = torch.where(b & 0x8000 != 0, ~b & 0xFFFF, b | 0x8000)
    return torch.where((bits & 0x7FFF) > 0x7F80, torch.full_like(bits, 0xFFFF), o)


def keys(bits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """key = ord16 << 18 | (2^18 - 1 - id): a larger key is a better candidate (kernels/dflash/common.cuh)."""
    return (ord16(bits) << head.ID_BITS) | ((1 << head.ID_BITS) - 1 - ids)


def topk_keys(
    vals: torch.Tensor, ids: torch.Tensor, k: int = head.TOPK
) -> torch.Tensor:
    """The k best keys per row of bf16 vals with their ids, best first."""
    return keys(bf16_bits(vals), ids).topk(k, -1).values


def key_ids(k: torch.Tensor) -> torch.Tensor:
    return (1 << head.ID_BITS) - 1 - (k & ((1 << head.ID_BITS) - 1))


def lattice_hhat(
    pred_cb: torch.Tensor,
    succ_cb: torch.Tensor,
    hhat: torch.Tensor,
    cand: torch.Tensor,
    unary: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """dflash2.lattice given the slots' W_P y (the kernel's own bf16 hhat [gamma, 256]) instead of y: an
    identity W_P makes the reference's projection hand hhat through exactly."""
    w = types.SimpleNamespace(
        hidden_proj=torch.eye(hhat.shape[1], dtype=torch.bfloat16),
        pred_codebook=pred_cb,
        succ_codebook=succ_cb,
    )
    return lattice(w, hhat, cand, unary, anchor, torch.bfloat16)


def verify_ids(anchor: int, drafts: torch.Tensor) -> torch.Tensor:
    """The verify window's 16 ids: [anchor, d_1 .. d_gamma], the live ids repeated into the pad rows."""
    live = torch.cat([torch.tensor([anchor]), drafts.long()])
    return live[torch.arange(head.ROWS) % live.numel()]


@dataclasses.dataclass
class AcceptRef:
    accept: int
    commit: int
    bonus: int
    pos: int  # P'
    top1: torch.Tensor  # [b_live]
    committed: torch.Tensor  # the out ring's new tokens


def accept_ref(
    tokens: torch.Tensor, top1: torch.Tensor, pos: int, forced: int | None = None
) -> AcceptRef:
    """dflash2.greedy_accept on the live rows (tokens [b_live], top1 = each row's argmax [b_live]): the
    drafts accepted while each equals the previous row's argmax; ``forced`` = the bench's accept override (the bonus
    is still the argmax at the forced position)."""
    top1 = top1.long()
    acc = 0
    while acc < tokens.numel() - 1 and int(tokens[acc + 1]) == int(top1[acc]):
        acc += 1
    if forced is not None:
        acc = max(0, min(forced, tokens.numel() - 1))
    committed = torch.cat([tokens[1 : acc + 1], top1[acc : acc + 1]])
    return AcceptRef(acc, acc + 1, int(top1[acc]), pos + acc + 1, top1, committed)


# ---- the layer's row units ----


def rms_tree_sum(s: torch.Tensor) -> torch.Tensor:
    """The kernel's sum of squares of each row of s [n, 8192] fp32: thread (vw, lane) owns channels (32 vw + lane) x 8
    .. + 7 and sums their squares left to right; a descending xor butterfly over the 32 lanes, then the same over the
    32 virtual warps (all IEEE fp32, no fused multiply-add, so CPU torch reproduces it bit for bit) -> [n].
    """
    assert s.dtype == torch.float32 and s.shape[-1] == 32 * 32 * 8, (s.dtype, s.shape)
    sq = s.reshape(-1, 32, 32, 8)
    sq = sq * sq
    acc = sq[..., 0]
    for e in range(1, 8):
        acc = acc + sq[..., e]
    lane = torch.arange(32)
    for off in (16, 8, 4, 2, 1):
        acc = acc + acc[..., lane ^ off]
    part = acc[..., 0]  # [n, 32 vw]
    for off in (16, 8, 4, 2, 1):
        part = part + part[..., lane ^ off]
    return part[..., 0]


def add_rms_norm_tree(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's add + RMSNorm row: s = x + residual (fp32) -> (bf16((s x rstd) x w), bf16(s)), mean = the tree
    sum x 2^-13, rstd = 1 / sqrt(mean + eps)."""
    f32 = torch.float32
    s = x.float() + residual.float()
    mean = rms_tree_sum(s) * torch.tensor(1.0 / s.shape[-1], dtype=f32)
    rstd = torch.tensor(1.0, dtype=f32) / torch.sqrt(
        mean + torch.tensor(eps, dtype=f32)
    )
    return ((s * rstd[:, None]) * w.float()).to(torch.bfloat16), s.to(torch.bfloat16)


# ---- the draft head ----


def quantize_y(y: torch.Tensor) -> torch.Tensor:
    """The kernel's activation rows (quantize_row): x 448 / amax in fp32, e4m3 round to nearest even (uint8 [rows, H]).
    An all-zero row keeps scale 1."""
    a = y.float().abs().amax(1, keepdim=True)
    inv = torch.where(a > 0, torch.tensor(448.0) / a, torch.ones_like(a))
    return (y.float() * inv).to(torch.float8_e4m3fn).view(torch.uint8)


def warp_dot(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The kernel's rescoring dot of each bf16 row of ``w`` [n, H] with ``y`` [H] (fp32 [n]): lane l accumulates the
    16-B chunks j = l + 32 i, i ascending, each chunk's 8 products in order (one rounding per FMA: a product of two
    bf16 is exact in fp32), then the xor butterfly 16, 8, 4, 2, 1."""
    n, hid = w.shape
    per = hid // (LANES * CHUNK)
    wf = w.float().view(n, per, LANES, CHUNK)
    yf = y.float().view(per, LANES, CHUNK)
    acc = torch.zeros(n, LANES, dtype=torch.float32, device=w.device)
    for i in range(per):
        for u in range(CHUNK):
            acc = acc + wf[:, i, :, u] * yf[i, :, u]
    lane = torch.arange(LANES, device=w.device)
    for o in (16, 8, 4, 2, 1):
        acc = acc + acc[:, lane ^ o]
    return acc[:, 0]


def exact_logits(w: torch.Tensor, y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """bf16 logits of every row of the shard ``w`` for each row of ``y`` [g, H] in the rescoring's order ([g, rows])."""
    out = torch.empty(y.shape[0], w.shape[0], dtype=torch.bfloat16, device=w.device)
    for r in range(y.shape[0]):
        for c0 in range(0, w.shape[0], chunk):
            out[r, c0 : c0 + chunk] = warp_dot(w[c0 : c0 + chunk], y[r]).to(
                torch.bfloat16
            )
    return out


def key_values(k: torch.Tensor) -> torch.Tensor:
    """The bf16 logits of candidate keys (head.keys' inverse on the value half)."""
    o = (k >> head.ID_BITS) & 0xFFFF
    bits = torch.where(o & 0x8000 != 0, o & 0x7FFF, ~o & 0xFFFF)
    bits = torch.where(o == 0xFFFF, torch.full_like(bits, 0x7FC0), bits)
    return bits.to(torch.int16).view(torch.bfloat16)


def survived(resc: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """Per slot: is every candidate of ``cand`` [gamma, 16] (the exact global top-16) among some rank's survivors?
    ``resc`` = the ranks' survivor keys [ranks, gamma, max] (zeroed before the launch).
    """
    ids = key_ids(resc)
    ids = torch.where(resc != 0, ids, torch.full_like(ids, -1))
    flat = ids.permute(1, 0, 2).reshape(cand.shape[0], -1)
    return torch.stack(
        [torch.isin(cand[e].long(), flat[e]).all() for e in range(cand.shape[0])]
    )


# ---- the token tree (kernels/dflash/tree.cuh) ----

TREE_ROWS, TREE_FAN = 16, 4


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def tree_fan_lp(vals: list[float]) -> list[float]:
    """A score row's TREE_FAN best scores, best first -> their log-probabilities within them (tree.cuh fan_row: fp32
    per operation; exp and log from the C library, so a value may differ from the device's expf / logf by an ulp, which
    moves a tree only where two path log-probabilities tie to that precision)."""
    s = 0.0
    for v in vals:
        s = _f32(s + _f32(math.exp(_f32(v - vals[0]))))
    lse = _f32(math.log(s))
    return [_f32(_f32(v - vals[0]) - lse) for v in vals]


@dataclasses.dataclass
class TokenTree:
    """The window's rows in preorder: parent (-1 for the anchor), depth, ancestor mask (bit j: row j), token."""

    parent: list[int]
    depth: list[int]
    mask: list[int]
    tokens: list[int]


def token_tree(anchor: int, cand: torch.Tensor, scores: torch.Tensor) -> TokenTree:
    """The head's tree (tree.cuh walk) from its lattice: cand [gamma, 16] ids, scores [gamma, 16, 16]. Each score row
    keeps its TREE_FAN successors in torch.argmax's order; the tree takes the TREE_ROWS - 1 nodes of the highest path
    log-probability best-first (ties: the earlier push), then lays the rows out in preorder, the rank-1 child first.
    """
    gamma = cand.shape[0]
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :TREE_FAN]
    vals = scores.gather(-1, order).tolist()
    order = order.tolist()
    lps: dict = {}

    def row_lp(e: int, p: int) -> list[float]:
        if (e, p) not in lps:
            lps[e, p] = tree_fan_lp(vals[e][p])
        return lps[e, p]

    # frontier entries: (lp, push order, slot, candidate index, parent row, rank)
    front = [(row_lp(0, 0)[k], k, 0, order[0][0][k], 0, k + 1) for k in range(TREE_FAN)]
    seq = TREE_FAN
    rows = [(-1, -1, -1, 0, 0)]  # (slot, candidate index, parent, depth, rank)
    lp_row = [0.0]
    while len(rows) < TREE_ROWS:
        best = max(front, key=lambda q: (q[0], -q[1]))
        front.remove(best)
        blp, _, e, c, par, rank = best
        r = len(rows)
        rows.append((e, c, par, e + 1, rank))
        lp_row.append(blp)
        if e + 1 < gamma:
            for k, lp in enumerate(row_lp(e + 1, c)):
                front.append(
                    (_f32(blp + lp), seq + k, e + 1, order[e + 1][c][k], r, k + 1)
                )
            seq += TREE_FAN
    key = [0] * TREE_ROWS
    for r in range(1, TREE_ROWS):
        _, _, par, d, rank = rows[r]
        key[r] = key[par] | (rank << (3 * (TREE_ROWS - 1 - d)))
    o = [sum(key[j] < key[r] for j in range(TREE_ROWS)) for r in range(TREE_ROWS)]
    parent, depth, tokens = [0] * TREE_ROWS, [0] * TREE_ROWS, [0] * TREE_ROWS
    for r, (e, c, par, d, _) in enumerate(rows):
        parent[o[r]] = -1 if r == 0 else o[par]
        depth[o[r]] = d
        tokens[o[r]] = anchor if r == 0 else int(cand[e, c])
    mask = [1] + [0] * (TREE_ROWS - 1)
    for r in range(1, TREE_ROWS):
        mask[r] = mask[parent[r]] | (1 << r)
    return TokenTree(parent, depth, mask, tokens)


def tree_accept(
    tree: TokenTree, top1: torch.Tensor, pos: int, forced: int | None = None
) -> tuple[AcceptRef, list[int]]:
    """verify_accept's tree accept -> (the accept, the path rows at depths 1 .. accept): the leaf is the deepest row whose own
    token and every ancestor's equal the target's argmax at their parent (``forced``: the leaf row the tests name).
    """
    t1 = [int(x) for x in top1]
    match = 1
    for r in range(1, TREE_ROWS):
        match |= int(tree.tokens[r] == t1[tree.parent[r]]) << r
    leaf = max(r for r in range(TREE_ROWS) if tree.mask[r] & ~match == 0)
    if forced is not None:
        leaf = max(0, min(forced, TREE_ROWS - 1))
    path = [r for r in range(1, TREE_ROWS) if (tree.mask[leaf] >> r) & 1]
    acc = tree.depth[leaf]
    committed = torch.tensor([tree.tokens[r] for r in path] + [t1[leaf]])
    return AcceptRef(acc, acc + 1, t1[leaf], pos + acc + 1, top1, committed), path
