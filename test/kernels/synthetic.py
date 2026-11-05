"""Synthetic weights in the kernels' layouts, the routing knob, and the setup the kernel tests and the static
benchmark share: identity slot tables over a seeded KV, the window's ids, the draft rings, the runs' start and reset,
the sampled accept's inputs.

Every tensor is drawn from a generator keyed by (seed, name, layer[, rank]): replicated tensors (norms, router, pooler,
unpooler, expert scales and bias) use no rank in the key, so every rank holds identical bytes, which the kernels
require (each rank routes on its own copy of the router and must agree with the others). The experts are drawn in
the per-rank shard form (gate / up / down codes and scale bytes) and packed by ``sm103.weights``, the one source of
the kernel layouts (the reference reads the same tensors back through its unpackers).

Value ranges keep every activation O(1): dense weights ~ N(0, 1 / K) with K the full contraction; e2m1 codes are any
byte (all 16 codes are finite); e4m3 scale factors are drawn from a finite band around 1 (never the NaN codes); the
per-expert weight scales put the dequantized weights at the same 1 / sqrt(K) RMS.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as dist
from sgl_muse_mega.muse_spark.config import Config
from sgl_muse_mega.muse_spark.sm103.artifacts import LAYER_IMAGES, build
from sgl_muse_mega.muse_spark.sm103.buffers import Buffers, Feeds
from sgl_muse_mega.muse_spark.sm103.dflash import iteration
from sgl_muse_mega.muse_spark.sm103.dflash.head import WORDS
from sgl_muse_mega.muse_spark.sm103.dflash.weights import (
    ConvWeights,
    DraftConfig,
    DraftWeights,
)
from sgl_muse_mega.muse_spark.sm103.dflash.weights import LayerWeights as DraftLayer
from sgl_muse_mega.muse_spark.sm103.verify import (
    M8Stack,
    layer_libraries,
    share_gather_rows,
)
from sgl_muse_mega.muse_spark.sm103.weights import (
    LayerWeights,
    ModelWeights,
    pack_w1_bf16,
    pack_w1_fp4,
    pack_w2_bf16,
    pack_w2_fp4,
)

# ---- synthetic draws ----

# a uniformly random code's RMS
E2M1_RMS = math.sqrt(sum(v * v for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6)) / 8)
# e4m3 exponent fields of the band: values 2^-1 .. 1.875, sign 0, never NaN
SF_EXPONENTS = (6, 7)


def _gen(device, *key) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(hashlib.sha256(repr(key).encode()).hexdigest()[:15], 16))
    return g


def _normal(shape, std, dtype, device, *key) -> torch.Tensor:
    return (torch.randn(shape, generator=_gen(device, *key), device=device) * std).to(
        dtype
    )


def _uniform(shape, lo, hi, dtype, device, *key) -> torch.Tensor:
    return (
        torch.rand(shape, generator=_gen(device, *key), device=device) * (hi - lo) + lo
    ).to(dtype)


def _codes(shape, device, *key) -> torch.Tensor:
    return torch.randint(
        0, 256, shape, generator=_gen(device, *key), device=device, dtype=torch.uint8
    )


def _sf(shape, device, *key) -> torch.Tensor:
    g = _gen(device, *key)
    exp = torch.tensor(SF_EXPONENTS, device=device, dtype=torch.uint8)[
        torch.randint(0, len(SF_EXPONENTS), shape, generator=g, device=device)
    ]
    return (exp << 3) | torch.randint(
        0, 8, shape, generator=g, device=device, dtype=torch.uint8
    )


def _sf_rms() -> float:
    vals = [2.0 ** (e - 7) * (1 + m / 8) for e in SF_EXPONENTS for m in range(8)]
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def _experts(cfg: Config, rank: int, layer: int, device, seed: int) -> dict:
    """The rank's experts of one layer in kernel layout: W1 = its I slice of every expert's gate and up rows, W2 its H
    slice of the output rows (bf16 variant: no scales)."""
    e, h, i = cfg.experts, cfg.moe_hidden, cfg.moe_inter
    i_s, h_s = cfg.inter_rank, cfg.moe_hidden_rank
    if cfg.is_bf16(layer):

        def dense(name, shape, k):
            return _normal(
                shape, k**-0.5, torch.bfloat16, device, seed, name, layer, rank
            )

        return dict(
            w1=pack_w1_bf16(dense("gate", (e, i_s, h), h), dense("up", (e, i_s, h), h)),
            w2=pack_w2_bf16(dense("w2", (e, h_s, i), i)),
        )

    def codes(name, rows, k):
        return _codes((e, rows, k // 2), device, seed, name, layer, rank), _sf(
            (e, rows, k // 16), device, seed, name + "_sf", layer, rank
        )

    w1, w1_sf = pack_w1_fp4(*codes("gate", i_s, h), *codes("up", i_s, h))
    w2, w2_sf = pack_w2_fp4(*codes("w2", h_s, i))
    # the weight scales: dequantized RMS 1 / sqrt(K) (K = H for W1, I for W2; equal here)
    s2 = 1.0 / (math.sqrt(h) * E2M1_RMS * _sf_rms())
    return dict(
        w1=w1,
        w1_sf_blk=w1_sf,
        w2=w2,
        w2_sf=w2_sf,
        g1_alphas=_uniform(
            (e,), 0.9 * s2, 1.1 * s2, torch.float32, device, seed, "g1a", layer
        ),
        g1_scale_c=_uniform((e,), 0.9, 1.1, torch.float32, device, seed, "g1c", layer),
        g2_alphas=_uniform(
            (e,), 0.9 * s2, 1.1 * s2, torch.float32, device, seed, "g2a", layer
        ),
    )


def layer_weights(
    cfg: Config, rank: int, layer: int, device, seed: int = 0
) -> LayerWeights:
    H, mh, E = cfg.hidden, cfg.moe_hidden, cfg.experts
    bf, f32 = torch.bfloat16, torch.float32

    def norm_w(name, n):
        return _uniform((n,), 0.8, 1.2, bf, device, seed, name, layer)

    return LayerWeights(
        qkv=_normal((cfg.qkv_row, H), H**-0.5, bf, device, seed, "qkv", layer, rank),
        o_proj=_normal(
            (H, cfg.oproj_k),
            (cfg.oproj_k * cfg.tp) ** -0.5,
            bf,
            device,
            seed,
            "o_proj",
            layer,
            rank,
        ),
        pooler=_normal((mh, H), H**-0.5, bf, device, seed, "pooler", layer),
        # bf16 values in f32 storage, as the checkpoint's router is: exact in the kernel's tf32 MMA
        router=_normal((E, H), H**-0.5, bf, device, seed, "router", layer).float(),
        unpooler=_normal((H, mh), mh**-0.5, bf, device, seed, "unpooler", layer),
        expert_bias=torch.zeros(E, dtype=f32, device=device),
        post_alpha=_uniform((H,), 0.9, 1.0, f32, device, seed, "post_alpha", layer),
        post_beta=_uniform((H,), 0.4, 0.6, f32, device, seed, "post_beta", layer),
        post_pre_w=norm_w("post_pre_w", H),
        cross_alpha=_uniform((H,), 0.9, 1.0, f32, device, seed, "cross_alpha", layer),
        cross_beta=_uniform((H,), 0.4, 0.6, f32, device, seed, "cross_beta", layer),
        cross_pre_w=norm_w("cross_pre_w", H),
        cross_post_w=norm_w("cross_post_w", H),
        pre_gamma=norm_w("pre_gamma", mh),
        gamma_p1=norm_w("gamma_p1", mh),
        **_experts(cfg, rank, layer, device, seed),
    )


def model_weights(
    cfg: Config, rank: int, device, embed_out: torch.Tensor, seed: int = 0
) -> ModelWeights:
    """``embed_out``: the [vocab_rank, H] bf16 destination of the vocab shard (a symmetric-memory view)."""
    H = cfg.hidden
    embed_out.copy_(
        _normal((cfg.vocab_rank, H), 1.0, torch.bfloat16, device, seed, "embed", rank)
    )
    return ModelWeights(
        embed=embed_out,
        entry_pre_w=_uniform(
            (H,), 0.8, 1.2, torch.bfloat16, device, seed, "entry_pre_w"
        ),
        lm_head=_normal(
            (cfg.vocab_rank, H), H**-0.5, torch.bfloat16, device, seed, "lm_head", rank
        ),
        layers=[layer_weights(cfg, rank, L, device, seed) for L in range(cfg.layers)],
    )


def popularity(cfg: Config, layer: int, device, seed: int = 0) -> torch.Tensor:
    """A fixed per-layer expert popularity profile (replicated): the routing calibration scales it into the bias."""
    return _normal(
        (cfg.experts,), 1.0, torch.float32, device, seed, "popularity", layer
    )


# ---- routing: expert_bias = scale x the popularity profile ----


def set_routing(cfg: Config, weights: ModelWeights, scale: float, device) -> None:
    """Selection is sigmoid(logit) + bias, so the scale concentrates the rows' choices on the popular experts (scale 0:
    the bias's widest routing). Read by the kernels at run time (no rebind)."""
    for L, lw in enumerate(weights.layers):
        lw.expert_bias.copy_(popularity(cfg, L, device) * scale)


def distinct_per_layer(bufs, rows: int | None = None) -> list[int]:
    """Distinct experts the window's first ``rows`` rows (default all) routed to in every layer (the last forward)."""
    return [torch.unique(st.idx32[:rows]).numel() for st in bufs.layers]


def distinct_experts(cfg: Config, bufs, rows: int | None = None) -> float:
    """Distinct experts per NVFP4 layer, averaged over those layers."""
    d = [n for L, n in enumerate(distinct_per_layer(bufs, rows)) if not cfg.is_bf16(L)]
    return sum(d) / len(d)


def calibrate_routing(
    cfg: Config,
    weights: ModelWeights,
    bufs,
    forward,
    target: float,
    iters: int,
    hi: float,
    group,
    rows: int | None = None,
) -> tuple[float, float]:
    """Bisects the popularity scale until the window (its first ``rows`` rows: the live ones) routes to ~``target``
    distinct experts per NVFP4 layer, then takes the measured scale closest to it (the count is a step function of the
    scale, not monotone at its steps). ``forward`` runs one eager step; every rank runs the same forwards and must
    route identically (each routes on identical replicated weights)."""
    tried = []

    def measure(scale: float) -> float:
        set_routing(cfg, weights, scale, bufs.device)
        forward()
        torch.cuda.synchronize()
        d = distinct_per_layer(bufs)
        seen = [None] * dist.get_world_size(group)
        dist.all_gather_object(seen, d, group=group)
        assert all(x == d for x in seen), f"the ranks routed differently: {seen}"
        n = distinct_experts(cfg, bufs, rows)
        tried.append((abs(n - target), scale))
        return n

    lo_d, hi_d = measure(0.0), measure(hi)
    assert (
        hi_d < target < lo_d
    ), f"target {target} outside [{hi_d}, {lo_d}] over scales [0, {hi}]"
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if measure(mid) > target:
            lo = mid
        else:
            hi = mid
    measure((lo + hi) / 2)
    scale = min(tried)[1]
    return scale, measure(scale)


# ---- the DFlash2 draft ----


def _randn(shape, std: float, *key) -> torch.Tensor:
    g = torch.Generator().manual_seed(
        int(hashlib.sha256(repr(key).encode()).hexdigest()[:15], 16)
    )
    return (torch.randn(shape, generator=g) * std).to(torch.bfloat16)


def _unit_norm(n: int, *key) -> torch.Tensor:
    return (1 + _randn((n,), 0.1, *key).float()).to(torch.bfloat16)


def draft_layer(cfg: DraftConfig, layer_i: int, rank: int, ranks: int) -> DraftLayer:
    """Rank's shard of a draft layer at the checkpoint's scales: dense rows ~ N(0, 1 / K) keyed by (layer, rank); the
    replicated convs (tap 0 near identity) and norms keyed by the layer alone."""
    q, kv, d, inter = (
        cfg.q_heads // ranks,
        cfg.kv_heads // ranks,
        cfg.head_dim,
        cfg.inter // ranks,
    )

    def conv(name: str) -> ConvWeights:
        base = _randn((2, cfg.conv_taps, cfg.hidden), 0.05, name, layer_i)
        base[:, 0] += 1
        proj = _randn(
            (2 * cfg.conv_taps * cfg.conv_groups, cfg.hidden),
            0.01,
            name + "_proj",
            layer_i,
        )
        return ConvWeights(base, proj)

    return DraftLayer(
        input_norm=_unit_norm(cfg.hidden, "in", layer_i),
        post_norm=_unit_norm(cfg.hidden, "post", layer_i),
        q_norm=_unit_norm(d, "q", layer_i),
        k_norm=_unit_norm(d, "k", layer_i),
        qkv=_randn(
            ((q + 2 * kv) * d, cfg.hidden), cfg.hidden**-0.5, "qkv", layer_i, rank
        ),
        o=_randn((cfg.hidden, q * d), (cfg.q_heads * d) ** -0.5, "o", layer_i, rank),
        gate_up=_randn(
            (2 * inter, cfg.hidden), cfg.hidden**-0.5, "gate_up", layer_i, rank
        ),
        down=_randn((cfg.hidden, inter), cfg.inter**-0.5, "down", layer_i, rank),
        attn_conv=conv("att"),
        mlp_conv=conv("mlp"),
    )


def draft(rank: int, ranks: int) -> DraftWeights:
    """The rank's shard of a synthetic DFlash2 draft (``weights.load_shard``'s layout)."""
    cfg = DraftConfig()
    return DraftWeights(
        [draft_layer(cfg, i, rank, ranks) for i in range(cfg.layers)],
        fc=_randn(
            (cfg.hidden // ranks, cfg.features * cfg.hidden),
            (cfg.features * cfg.hidden) ** -0.5,
            "fc",
            rank,
        ),
        hidden_norm=_unit_norm(cfg.hidden, "hidden_norm"),
        norm=_unit_norm(cfg.hidden, "norm"),
        pred_codebook=_randn((cfg.vocab, cfg.selector_rank), 0.05, "pred"),
        succ_codebook=_randn((cfg.vocab, cfg.selector_rank), 0.05, "succ"),
        hidden_proj=_randn((cfg.selector_rank, cfg.hidden), 0.01, "wp"),
    )


# ---- the test setup ----


def identity_feeds(cfg: Config, device, tree: bool = False) -> Feeds:
    """Feeds over identity tables: slots = positions in both pools, one request whose slot-table row reaches every
    position a window can (context_length + rows), and the rope rows of position x 500000^(-2i/64) from fp64
    (``tree``: with a token-tree window's table).
    """
    width, d = cfg.context_length + cfg.rows, cfg.head_dim
    inv_freq = cfg.rope_theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = torch.arange(width, dtype=torch.float64)[:, None] * inv_freq[None]
    return Feeds(
        cfg,
        torch.arange(width, dtype=torch.int32, device=device)[None],
        torch.arange(width, dtype=torch.int64, device=device),
        torch.cat([ang.cos(), ang.sin()], 1).float().to(device),
        device,
        tree=tree,
    )


def set_context(f: Feeds, ctx: int, rows: int) -> None:
    """The held context of identity feeds: the window's rows sit at positions [ctx, ctx + rows); replays consume it in
    place."""
    assert ctx + rows <= f.width
    pos = torch.arange(ctx, ctx + rows, dtype=torch.int64, device=f.positions.device)
    f.positions.copy_(pos)
    f.out_cache_loc.copy_(pos)
    f.swa_out_cache_loc.copy_(pos)
    f.seq_lens.fill_(ctx)


def kv_fill(cfg: Config, rank: int, rows: int, device) -> list[tuple]:
    """Every layer's K and V rows by position, [rows, kv heads x head_dim] bf16 U(-0.1, 0.1), from one generator
    keyed by the rank."""
    g = torch.Generator(device=device)
    g.manual_seed(1 + rank)
    out = []
    for _ in range(cfg.layers):
        k, v = (
            torch.empty(
                rows, cfg.kv_row_bytes // 2, dtype=torch.bfloat16, device=device
            )
            for _ in range(2)
        )
        k.uniform_(-0.1, 0.1, generator=g)
        v.uniform_(-0.1, 0.1, generator=g)
        out.append((k, v))
    return out


def seed_ids(bufs: Buffers, cfg: Config) -> None:
    """The window's token ids: seeded, identical on every rank."""
    g = torch.Generator(device=bufs.ids.device)
    g.manual_seed(0)
    bufs.ids.copy_(
        torch.randint(0, cfg.vocab, (cfg.rows,), generator=g, device=bufs.ids.device)
    )


def seed_rings(it: iteration.Iteration) -> None:
    """Every draft layer's K / V ring: U(-0.1, 0.1) bf16 rows from a generator keyed by (rank, layer)."""
    cx = it.cx
    for i, lb in enumerate(it.lb):
        g = torch.Generator(device=cx.device).manual_seed(77 + 1000 * cx.rank + i)
        for ring in (lb.k_ring, lb.v_ring):
            ring.copy_(
                (torch.rand(ring.shape, generator=g, device=cx.device) * 0.2 - 0.1).to(
                    torch.bfloat16
                )
            )


def target_feeds(cx: iteration.Context, p0: int) -> None:
    """The verify window's feeds at P0 (window_feeds over identity feeds: rope rows, cache slots, kv_len, the page
    tables); afterwards verify_accept writes them each iteration."""
    with torch.cuda.stream(cx.main):
        set_context(cx.feeds, p0, cx.cfg.rows)
        cx.stack.feeds(cx.main.cuda_stream, False)
    torch.cuda.synchronize(cx.device)


# the start state of a DFlash2 run: the anchor and the last commit
ANCHOR, COMMIT0 = 123457, 4


def forced_accepts(b: int, n: int = 64) -> torch.Tensor:
    """verify_accept's forced accept per iteration word (mod n): every count 0 .. B - 1 in turn (3 is coprime to B)."""
    return (torch.arange(n, dtype=torch.int32) * 3) % b


def reset_all(cx: iteration.Context, its: dict) -> None:
    """Every protocol word and cross-rank region of the stack and of every block size to its bind-time value."""
    cx.bufs.reset(cx.group)
    for it in its.values():
        it.reset()
    cx.published()


def begin(cx: iteration.Context, its: dict, it: iteration.Iteration, p0: int) -> None:
    """A run of block size it.b from P0: the bind-time words, the anchor, the last commit, the previous verify's
    features (seeded N(0, 1) rows, identical on every rank), the verify window's feeds at P0.
    """
    reset_all(cx, its)
    it.start(p0, COMMIT0)
    it.words[WORDS["ANCHOR"]].fill_(ANCHOR)
    g = torch.Generator(device=cx.device).manual_seed(4242)
    for L in cx.capture:
        cap = cx.stack.cap(L)
        cap.copy_(torch.randn(cap.shape, generator=g, device=cx.device))
    it.accept_buf.out_ring.zero_()
    target_feeds(cx, p0)
    cx.published()


HANG = {
    1: "x-ready count",
    2: "row-ready count",
    3: "lamport consume",
    4: "coefficient poll",
}


def hang_report(diag: torch.Tensor) -> str | None:
    """The trapped wait of a draft layer, if a spin outlived its budget (its host-pinned hang record)."""
    code, block, thread, observed, target = (int(x) for x in diag[:5])
    if code == 0:
        return None
    return f"HANG {HANG.get(code, code)}: block {block} thread {thread} observed {observed} target {target}"


def context(
    draft_weights: DraftWeights,
    max_ctx: int,
    rank: int,
    device,
    group,
    cfg: Config,
    tree: bool = False,
):
    """A DFlash2 ``iteration.Context`` on synthetic target weights, a synthetic KV of ``max_ctx`` positions and identity
    slot tables; the verify's lm_head binds the head weight's vocab rows (the bytes of the synthetic lm_head).
    ``tree``: with the token-tree window's images.
    """
    paths = build(dflash=True, world=cfg.world)
    libs = layer_libraries(paths)
    consts = {image: libs[image].constants for image in LAYER_IMAGES}
    bufs = Buffers(cfg, consts, device, group)
    seed_ids(bufs, cfg)
    weights = model_weights(cfg, rank, device, bufs.embed_shard(cfg))
    draft = iteration.DraftDevice(draft_weights, weights.lm_head, cfg, rank, device)
    weights.lm_head = draft.hw[: cfg.vocab_rank]
    kv = kv_fill(cfg, rank, (max_ctx + cfg.rows + 127) // 128 * 128, device)
    feeds = identity_feeds(cfg, device, tree)
    stack = M8Stack(cfg, paths, libs, bufs, weights, rank, device, kv, feeds)
    share_gather_rows(bufs, consts["swa"], stack.libs8["swa"].constants)
    return iteration.Context(stack, draft, group, tree)


# ---- the sampled accept's inputs ----

TOP_K_ALL = 1 << 30  # SGLang's "no top-k"
# (temperature, top_k, top_p): the model's defaults (T 1, top-k 64), top-p alone, both, the plain softmax, a peaked row
# (the k-th weight below 2^-30: the selection over the whole key) and a larger top-k with top-p
SAMPLING_CONFIGS = {
    "t1_k64": (1.0, 64, 1.0),
    "t07_p09": (0.7, TOP_K_ALL, 0.9),
    "t1_k20_p095": (1.0, 20, 0.95),
    "t13_dense": (1.3, TOP_K_ALL, 1.0),
    "t005_k64": (0.05, 64, 1.0),
    "t1_k1000_p08": (1.0, 1000, 0.8),
}


def sampled_case(g: torch.Generator, b: int, vocab: int, anchor: int) -> tuple:
    """A sampled verify's inputs: each slot's 16 candidates, the walk's q over them and its path, the window's tokens,
    and logits rows that favor the drafts' candidates by a random margin (softly capped at 20).
    """
    cand = torch.stack(
        [torch.randperm(vocab, generator=g)[:16] for _ in range(b - 1)]
    ).int()
    q = torch.softmax(torch.randn(b - 1, 16, generator=g) * 2, -1)
    path = torch.randint(0, 16, (b - 1,), generator=g)
    tokens = torch.cat([torch.tensor([anchor]), cand[torch.arange(b - 1), path].long()])
    logits = torch.randn(b, vocab, generator=g) * 3
    boost = torch.rand(b - 1, generator=g) * 12
    for r in range(b - 1):
        logits[r, cand[r]] += boost[r]
        logits[r, tokens[r + 1]] += boost[r]
    return cand, q, path, tokens, 20 * torch.tanh(logits / 20)
