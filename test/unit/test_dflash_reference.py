"""CPU tests of the DFlash2 reference (test/reference/dflash2.py) and of the DFlash iteration's composition.

The reference: the conv against a per-position loop, RoPE against a complex rotation, the attention window against a
per-query loop, the lattice and the selector walk against brute force, the top-k tie rule across shards, the greedy
accept, the checkpoint reader, B = 8 / 16 shapes, TP8 == unsharded over a multi-iteration decode, and one real
checkpoint layer when it is staged.

The composition (sm103/dflash/iteration.py, test/kernels/test_dflash_iteration.py): the interfaces the units hand each
other (the entry's side-1 coefficient row in the queue's packet order, the accept's draft feeds in the order of the
draft qkv's x rows), the synthetic draft's shard shapes, the checks' comparisons, and the forced-suffix localization on
emulated ranks. No GPU is touched.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import struct

import pytest
import synthetic
import test_dflash_iteration as run
import torch
from reference import dflash2 as R
from sgl_muse_mega.muse_spark.sm103.artifacts import DFLASH_BLOCKS
from sgl_muse_mega.muse_spark.sm103.dflash import head, iteration, layer
from sgl_muse_mega.muse_spark.sm103.dflash import weights as dweights
from sgl_muse_mega.muse_spark.sm103.dflash.head import WORDS

# the emulated ranks are threads: keep their BLAS pools from oversubscribing the box
torch.set_num_threads(16)

# the staged public DFlash2 draft's directory (model.safetensors + config.json); unset: its tests skip
CKPT_DIR = (
    pathlib.Path(os.environ["SGL_MUSE_MEGA_DRAFT_PATH"])
    if "SGL_MUSE_MEGA_DRAFT_PATH" in os.environ
    else None
)
# Small dims with the real structure: every TP8 split divides, the window (24) is shorter than the context so the
# low window edge is live, and a block of 16 still fits inside it.
SMALL = R.DraftConfig(
    hidden=128,
    layers=2,
    q_heads=16,
    kv_heads=8,
    head_dim=16,
    inter=256,
    vocab=1024,
    sliding_window=24,
    capture=(1, 2, 3),
    mask_id=1000,
    selector_rank=32,
)
BF16, F32 = torch.bfloat16, torch.float32


def _randn(g: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=g) * scale).to(BF16)


def _weights(
    cfg: R.DraftConfig, seed: int = 0
) -> tuple[R.DraftWeights, R.TargetShared]:
    """Unsharded synthetic weights at the real checkpoint's scales (norm gains near 1, conv tap 0 near 1)."""
    g = torch.Generator().manual_seed(seed)
    h, d, f = cfg.hidden, cfg.head_dim, cfg.features

    def gain(n: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(n, generator=g)).to(BF16)

    def conv() -> R.ConvWeights:
        base = torch.randn(2, cfg.conv_taps, h, generator=g) * 0.1
        base[:, 0] += 1
        return R.ConvWeights(
            base.to(BF16),
            _randn(g, 2 * cfg.conv_taps * cfg.conv_groups, h, scale=0.3 * h**-0.5),
        )

    layers = [
        R.LayerWeights(
            gain(h),
            gain(h),
            gain(d),
            gain(d),
            qkv=_randn(g, (cfg.q_heads + 2 * cfg.kv_heads) * d, h, scale=h**-0.5),
            o=_randn(g, h, cfg.q_heads * d, scale=(cfg.q_heads * d) ** -0.5),
            gate_up=_randn(g, 2 * cfg.inter, h, scale=h**-0.5),
            down=_randn(g, h, cfg.inter, scale=cfg.inter**-0.5),
            attn_conv=conv(),
            mlp_conv=conv(),
        )
        for _ in range(cfg.layers)
    ]
    r = cfg.selector_rank
    w = R.DraftWeights(
        layers,
        fc=_randn(g, h, f * h, scale=(f * h) ** -0.5),
        hidden_norm=gain(h),
        norm=gain(h),
        pred_codebook=_randn(g, cfg.vocab, r, scale=r**-0.25),
        succ_codebook=_randn(g, cfg.vocab, r, scale=r**-0.25),
        hidden_proj=_randn(g, r, h, scale=h**-0.5),
    )
    return w, R.TargetShared(
        _randn(g, cfg.vocab, h), _randn(g, cfg.vocab, h, scale=4 * h**-0.5)
    )


def _prefill(cfg: R.DraftConfig, prompt: int, seed: int = 1) -> R.DecodeState:
    g = torch.Generator().manual_seed(seed)
    return R.DecodeState(
        prompt, 7, _randn(g, prompt, cfg.features * cfg.hidden), prompt
    )


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float) -> None:
    """Every |a - b| within rtol of |b| + rms(b): relative on large elements, rtol x the tensor's scale on small
    ones."""
    a, b = a.double(), b.double()
    bound = rtol * (b.abs() + b.square().mean().sqrt())
    excess = ((a - b).abs() - bound).max().item()
    assert excess <= 0, (excess, (a - b).abs().max().item(), bound.max().item())


def _close_rows(a: torch.Tensor, b: torch.Tensor, rtol: float) -> None:
    """Every row's ||a - b|| within rtol of its ||b|| (the bf16 comparisons: a rounded partial can dominate an element
    whose partials cancel, so the bound is on the row)."""
    a, b = a.double().flatten(1), b.double().flatten(1)
    worst = ((a - b).norm(dim=1) / b.norm(dim=1)).max().item()
    assert worst <= rtol, worst


def _kv_slice(kv: R.DraftKV, rank: int, ranks: int) -> R.DraftKV:
    return R.DraftKV(
        [R._block(k, rank, ranks, 1) for k in kv.k],
        [R._block(v, rank, ranks, 1) for v in kv.v],
    )


# ---------------------------------------------------------------------------------------------------------------------
# Elementwise blocks


def _naive_conv(x, delta, base) -> torch.Tensor:
    b, h = x.shape
    taps, groups = delta.shape[1:]
    out = torch.zeros(b, h, dtype=torch.float64)
    for n in range(b):
        for ch in range(h):
            acc = 0.0
            for t in range(min(taps, n + 1)):
                acc += (
                    float(base[t, ch]) + float(delta[n, t, ch // (h // groups)])
                ) * float(x[n - t, ch])
            out[n, ch] = acc
    return out


@pytest.mark.parametrize("block", [8, 16])
@pytest.mark.parametrize("taps", [2, 3])
def test_grouped_conv_matches_per_position_loop(block, taps):
    g = torch.Generator().manual_seed(block + taps)
    h, groups = 64, 4
    x, base = _randn(g, block, h), _randn(g, taps, h)
    delta = _randn(g, block, taps, groups, scale=0.3)
    ref = _naive_conv(x, delta, base)
    _close(R.grouped_conv(x, delta, base, F32), ref, 1e-6)
    # bf16: the fp32 expression rounded once (a value next to a rounding boundary may land one ulp away)
    out = R.grouped_conv(x, delta, base, BF16).double()
    assert ((out - ref).abs() <= ref.abs() * 2**-7 + 1e-6).all()


def test_conv_prepare_finish_coefficient_layout():
    """The projection's output n = (side, tap, group) row-major: side 0 with base[0] convolves the input, side 1 with
    base[1] the output (M/)."""
    cfg = SMALL
    w, _ = _weights(cfg)
    conv = w.layers[0].attn_conv
    g = torch.Generator().manual_seed(3)
    h, o = _randn(g, 8, cfg.hidden), _randn(g, 8, cfg.hidden)
    flat = h.double() @ conv.proj.double().T
    taps, groups = cfg.conv_taps, cfg.conv_groups

    def coeff(side):
        return torch.stack(
            [
                torch.stack(
                    [
                        flat[:, side * taps * groups + t * groups + gg]
                        for gg in range(groups)
                    ],
                    1,
                )
                for t in range(taps)
            ],
            1,
        )

    h_conv, c_out = R.conv_prepare(conv, h, F32)
    _close(h_conv, _naive_conv(h, coeff(0), conv.base[0]), 1e-5)
    _close(
        R.conv_finish(conv, o, c_out, F32), _naive_conv(o, coeff(1), conv.base[1]), 1e-5
    )


def test_head_norm_rope_matches_complex_rotation():
    g = torch.Generator().manual_seed(4)
    n, heads, d, theta = 6, 3, 16, 5e5
    x, w = _randn(g, n, heads, d), (1 + 0.1 * torch.randn(d, generator=g)).to(BF16)
    pos = torch.tensor([0, 1, 2, 17, 2047, 3999])
    cos, sin = R.rope_cos_sin(pos, d, theta)
    out = R.head_norm_rope(x, w, 1e-5, cos, sin, F32, round_normed=False).double()
    xd = x.double()
    xn = xd / torch.sqrt(xd.square().mean(-1, keepdim=True) + 1e-5) * w.double()
    angle = pos.double()[:, None] * theta ** (-torch.arange(0, d, 2).double() / d)
    z = (
        torch.complex(xn[..., : d // 2], xn[..., d // 2 :])
        * torch.polar(torch.ones_like(angle), angle)[:, None]
    )
    # fp32 angles at position ~4000 drift ~1e-4 rad
    _close(out, torch.cat([z.real, z.imag], -1), 2e-3)


def test_attention_window_matches_per_query_loop():
    g = torch.Generator().manual_seed(5)
    qh, kvh, d, wl, block, pos = 4, 2, 8, 5, 4, 9
    q = _randn(g, block, qh, d)
    k, v = _randn(g, pos + block, kvh, d), _randn(g, pos + block, kvh, d)
    pos_q, pos_k = pos + torch.arange(block), torch.arange(pos + block)
    out = R.attention(q, k, v, pos_q, pos_k, wl).view(block, qh, d)
    for i in range(block):
        keys = [j for j in range(pos + block) if abs(j - (pos + i)) <= wl]
        # [p - window_left, P + B - 1]: non-causal
        assert keys == list(range(pos + i - wl, pos + block))
        for hq in range(qh):
            kv_head = hq // (qh // kvh)
            p = torch.softmax(
                k[keys, kv_head].double() @ q[i, hq].double() * d**-0.5, 0
            )
            ref = p @ v[keys, kv_head].double()
            torch.testing.assert_close(out[i, hq], ref, rtol=1e-12, atol=1e-12)
    # draft_layer reads only from the lowest visible position on
    lo = pos - wl
    cut = R.attention(q, k[lo:], v[lo:], pos_q, pos_k[lo:], wl).view(block, qh, d)
    torch.testing.assert_close(cut, out, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# Head, selector, accept


def test_topk_tie_rule_across_shards():
    """Exact logit ties (duplicated lm_head rows, some across shard boundaries): TP8 picks the same candidates as one
    rank, ties going to the smaller id."""
    cfg = SMALL
    w, t = _weights(cfg)
    g = torch.Generator().manual_seed(6)
    y = _randn(g, 7, cfg.hidden)
    lm = t.lm_head.clone()
    top = (y[0].double() @ lm.double().T).topk(6).indices
    shard_rows = cfg.vocab // 8
    for i, src in enumerate(top):
        lm[(src + shard_rows * (i + 1)) % cfg.vocab] = lm[src]
    t = R.TargetShared(t.embed, lm)
    ids1, u1 = R.candidates(cfg, t, y, R.SoloComm(), BF16)
    for ids, u in R.emulate_ranks(
        lambda c: R.candidates(cfg, R.shard(cfg, w, t, c.rank, c.ranks)[1], y, c, BF16)
    ):
        assert torch.equal(ids, ids1) and torch.equal(u, u1)
    logits = (y.double() @ lm.double().T).to(BF16).float()
    for e in range(7):
        order = sorted(range(cfg.vocab), key=lambda j: (-logits[e, j].item(), j))
        assert ids1[e].tolist() == order[: cfg.top_k]
    assert torch.equal(
        u1, R.softcap(logits.gather(1, ids1), cfg.output_multiplier, cfg.softcap)
    )


def test_lattice_matches_brute_force():
    cfg = dataclasses.replace(SMALL, selector_rank=8)
    w, _ = _weights(cfg)
    g = torch.Generator().manual_seed(7)
    slots, k = 7, cfg.top_k
    y = _randn(g, slots, cfg.hidden)
    cand = torch.randint(0, cfg.vocab, (slots, k), generator=g)
    unary = torch.randn(slots, k, generator=g)
    anchor = 11
    scores = R.lattice(w, y, cand, unary, anchor, F32)
    hid = (y.double() @ w.hidden_proj.double().T).tolist()
    a, s = w.pred_codebook.double().tolist(), w.succ_codebook.double().tolist()
    for e in range(slots):
        for p in range(k):
            pred = anchor if e == 0 else int(cand[e - 1, p])
            for c in range(k):
                succ = s[int(cand[e, c])]
                ref = float(unary[e, c])
                ref += sum(a[pred][r] * hid[e][r] * succ[r] for r in range(8))
                assert abs(float(scores[e, p, c]) - ref) <= 1e-5 * (1 + abs(ref))


def _brute_walk(cand, scores, greedy, uniforms=None, temperature=1.0):
    prev, tokens = 0, []
    for e in range(cand.shape[0]):
        row = [float(v) for v in scores[e, prev]]
        if greedy:
            nans = [i for i, v in enumerate(row) if math.isnan(v)]
            idx = nans[0] if nans else row.index(max(row))
        else:
            m = max(v / temperature for v in row)
            ex = [math.exp(v / temperature - m) for v in row]
            cdf, acc = [], 0.0
            for v in ex:
                acc += v / sum(ex)
                cdf.append(acc)
            idx = min(sum(1 for c in cdf if float(uniforms[e]) >= c), len(row) - 1)
        tokens.append(int(cand[e, idx]))
        prev = idx
    return tokens


@pytest.mark.parametrize("slots", [7, 15])
def test_selector_walk_greedy_matches_brute_force(slots):
    g = torch.Generator().manual_seed(slots)
    k = 16
    cand = torch.randint(0, 1000, (slots, k), generator=g)
    scores = torch.randn(slots, k, k, generator=g)
    # a tie for the max: the first index wins; NaN wins, as in torch.argmax
    scores[1, :, 3] = scores[1, :, 9] = scores[1].amax() + 1
    scores[2, :, 5] = float("nan")
    tokens, q = R.selector_walk(cand, scores)
    assert tokens.tolist() == _brute_walk(cand, scores, True)
    assert int(tokens[1]) == int(cand[1, 3]) and int(tokens[2]) == int(cand[2, 5])
    assert torch.equal(q.sum(1), torch.ones(slots))
    assert set(q.unique().tolist()) == {0.0, 1.0}


def test_selector_walk_sampled_matches_brute_force():
    g = torch.Generator().manual_seed(8)
    slots, k = 7, 16
    cand = torch.randint(0, 1000, (slots, k), generator=g)
    scores = torch.randn(slots, k, k, generator=g) * 2
    u = torch.rand(slots, generator=g)
    tokens, q = R.selector_walk(cand, scores, greedy=False, uniforms=u, temperature=0.7)
    assert tokens.tolist() == _brute_walk(cand, scores, False, u, 0.7)
    torch.testing.assert_close(q.sum(1), torch.ones(slots))


def _brute_accept(tokens, logits):
    t = [max(range(len(row)), key=lambda j: (row[j], -j)) for row in logits.tolist()]
    acc = 0
    while acc < len(tokens) - 1 and int(tokens[acc + 1]) == t[acc]:
        acc += 1
    return acc, t[acc]


@pytest.mark.parametrize("block", [8, 16])
@pytest.mark.parametrize("accept", [0, 3, "all"])
def test_greedy_accept(block, accept):
    acc_n = block - 1 if accept == "all" else accept
    g = torch.Generator().manual_seed(block)
    tokens = torch.randint(0, 500, (block,), generator=g)
    logits = torch.randn(block, 600, generator=g)
    for i in range(acc_n):
        logits[i, tokens[i + 1]] = 50.0
    if acc_n < block - 1:
        # tied maxima: the first index is the bonus
        wrong = (int(tokens[acc_n + 1]) + 1) % 500
        logits[acc_n, wrong] = logits[acc_n, wrong + 50] = 50.0
    a = R.greedy_accept(tokens, logits, 100)
    assert (a.accept_len, a.bonus) == _brute_accept(tokens, logits)
    assert (a.accept_len, a.commit_len, a.new_pos) == (acc_n, acc_n + 1, 101 + acc_n)
    expect = torch.cat([tokens[1:], torch.zeros(1, dtype=tokens.dtype)])
    expect[acc_n] = a.bonus
    assert torch.equal(a.out_tokens, expect)


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint reader


def _write_safetensors(path: pathlib.Path, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = bytes(t.contiguous().view(torch.uint8).flatten().tolist())
        header[name] = {
            "dtype": "BF16",
            "shape": list(t.shape),
            "data_offsets": [off, off + len(b)],
        }
        blobs.append(b)
        off += len(b)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_load_draft_from_hf_names(tmp_path):
    cfg = SMALL
    w, _ = _weights(cfg)
    d = cfg.head_dim
    tensors = {}
    for i, lw in enumerate(w.layers):
        p = f"layers.{i}."
        q, k, v = lw.qkv.split([cfg.q_heads * d, cfg.kv_heads * d, cfg.kv_heads * d])
        gate, up = lw.gate_up.chunk(2)
        tensors |= {
            f"{p}self_attn.q_proj.weight": q,
            f"{p}self_attn.k_proj.weight": k,
            f"{p}self_attn.v_proj.weight": v,
            f"{p}self_attn.o_proj.weight": lw.o,
            f"{p}self_attn.q_norm.weight": lw.q_norm,
            f"{p}self_attn.k_norm.weight": lw.k_norm,
            f"{p}mlp.gate_proj.weight": gate,
            f"{p}mlp.up_proj.weight": up,
            f"{p}mlp.down_proj.weight": lw.down,
            f"{p}input_layernorm.weight": lw.input_norm,
            f"{p}post_attention_layernorm.weight": lw.post_norm,
        }
        for name, c in (("attention_conv", lw.attn_conv), ("mlp_conv", lw.mlp_conv)):
            tensors |= {
                f"{p}{name}.base_kernel": c.base,
                f"{p}{name}.kernel_projection.weight": c.proj,
            }
    tensors |= {
        "fc.weight": w.fc,
        "hidden_norm.weight": w.hidden_norm,
        "norm.weight": w.norm,
        "candidate_selector.predecessor_codebook": w.pred_codebook,
        "candidate_selector.successor_codebook": w.succ_codebook,
        "candidate_selector.hidden_projection.weight": w.hidden_proj,
    }
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    got = R.load_draft(str(tmp_path / "model.safetensors"), cfg)

    def flat(x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, list):
            return [t for e in x for t in flat(e)]
        return [t for f in dataclasses.fields(x) for t in flat(getattr(x, f.name))]

    assert all(torch.equal(a, b) for a, b in zip(flat(got), flat(w), strict=True))


def test_config_from_hf_is_the_default():
    if CKPT_DIR is None:
        pytest.skip("SGL_MUSE_MEGA_DRAFT_PATH is not set")
    path = CKPT_DIR / "config.json"
    assert R.DraftConfig.from_hf(json.loads(path.read_text())) == R.DraftConfig()


# ---------------------------------------------------------------------------------------------------------------------
# The iteration: shapes, TP8 == unsharded, state carry


def _draft(cfg, w, t, state, block, act, ranks, capacity):
    """draft_step on one rank (ranks = 1) or emulated TP ranks -> [(Draft, DraftKV)] per rank."""
    if ranks == 1:
        kv = R.DraftKV.empty(cfg, capacity, 1, act)
        return [(R.draft_step(cfg, w, t, kv, state, block, R.SoloComm(), act), kv)]

    def rank(comm):
        ws, ts = R.shard(cfg, w, t, comm.rank, comm.ranks)
        kv = R.DraftKV.empty(cfg, capacity, comm.ranks, act)
        return R.draft_step(cfg, ws, ts, kv, state, block, comm, act), kv

    return R.emulate_ranks(rank, ranks)


@pytest.mark.parametrize("block", [8, 16])
def test_draft_step_shapes(block):
    cfg = SMALL
    w, t = _weights(cfg)
    state = _prefill(cfg, 40)
    ((dr, kv),) = _draft(cfg, w, t, state, block, BF16, 1, 64)
    k = cfg.top_k
    assert dr.tokens.shape == (block,) and dr.tokens.dtype == torch.int64
    assert int(dr.tokens[0]) == state.anchor
    assert ((dr.tokens >= 0) & (dr.tokens < cfg.vocab)).all()
    assert dr.y.shape == (block, cfg.hidden) and dr.y.dtype == BF16
    assert dr.cand.shape == dr.unary.shape == dr.q.shape == (block - 1, k)
    assert dr.scores.shape == (block - 1, k, k) and dr.scores.dtype == F32
    assert torch.equal(
        dr.tokens[1:], dr.cand.gather(1, dr.q.argmax(1, keepdim=True))[:, 0]
    )
    assert len(dr.ctx) == cfg.layers
    assert dr.ctx[0][0].shape == (40, cfg.kv_heads, cfg.head_dim)
    assert (kv.k[0][: 40 + block].abs().sum((1, 2)) > 0).all()
    assert (kv.k[0][40 + block :] == 0).all()


@pytest.mark.parametrize("block", [8, 16])
def test_tp8_equals_unsharded_fp32(block):
    cfg = SMALL
    w, t = _weights(cfg)
    state = _prefill(cfg, 40)
    ((one, kv1),) = _draft(cfg, w, t, state, block, F32, 1, 64)
    for r, (dr, kv) in enumerate(_draft(cfg, w, t, state, block, F32, 8, 64)):
        assert torch.equal(dr.tokens, one.tokens) and torch.equal(dr.cand, one.cand)
        for got, ref in [(dr.y, one.y), (dr.unary, one.unary), (dr.scores, one.scores)]:
            _close(got, ref, 1e-5)
        ref_kv = _kv_slice(kv1, r, 8)
        for got, ref in zip(kv.k + kv.v, ref_kv.k + ref_kv.v, strict=True):
            _close(got, ref, 1e-5)


def test_tp8_bf16_moves_only_y():
    """bf16: the per-rank rounding of the row-parallel partials moves y by a few ulps (~1 % per row here), which can
    move top-16 boundary candidates and near-tie walk steps. Everything after y is rank-invariant: the unsharded
    head + walk on a rank's own y reproduce that rank's candidates, scores and tokens exactly.
    """
    cfg = SMALL
    w, t = _weights(cfg)
    state = _prefill(cfg, 40)
    ((one, _),) = _draft(cfg, w, t, state, 8, BF16, 1, 64)
    for dr, _ in _draft(cfg, w, t, state, 8, BF16, 8, 64):
        _close_rows(dr.y, one.y, 2**-6)
        cand, unary = R.candidates(cfg, t, dr.y[1:], R.SoloComm(), BF16)
        scores = R.lattice(w, dr.y[1:], cand, unary, state.anchor, BF16)
        tokens, _ = R.selector_walk(cand, scores)
        assert torch.equal(cand, dr.cand) and torch.equal(scores, dr.scores)
        assert torch.equal(tokens, dr.tokens[1:])


def test_context_write_of_all_rows_is_equivalent():
    """Writing every context row, not just commit_len, leaves the next draft bitwise unchanged: the rows past commit
    land in [P, P + B), which each layer's block write replaces before its attention reads it.
    """
    cfg = SMALL
    w, t = _weights(cfg)
    block, commit = 8, 3
    g = torch.Generator().manual_seed(9)
    state = R.DecodeState(43, 5, _randn(g, block, cfg.features * cfg.hidden), commit)
    kv_a = R.DraftKV.empty(cfg, 64, 1, BF16)
    prompt = R.context_update(cfg, w, _prefill(cfg, 40).feats, 0, R.SoloComm(), BF16)
    R.write_context(kv_a, prompt, 0, 40)
    kv_b = R.DraftKV(list(map(torch.clone, kv_a.k)), list(map(torch.clone, kv_a.v)))
    ref = R.draft_step(cfg, w, t, kv_a, state, block, R.SoloComm(), BF16)
    R.write_context(kv_b, ref.ctx, 40, block)
    y = R.draft_block(
        cfg, w, t, kv_b, state.anchor, state.pos, block, R.SoloComm(), BF16
    )
    assert torch.equal(y, ref.y)
    assert all(
        torch.equal(a, b) for a, b in zip(kv_a.k + kv_a.v, kv_b.k + kv_b.v, strict=True)
    )


def _fake_verify(cfg, plan):
    """A target stand-in: accepts plan[pos] drafts at that window, then a bonus; features seeded by position."""

    def verify(tokens, pos):
        g = torch.Generator().manual_seed(pos)
        b, acc = tokens.shape[0], plan[pos]
        logits = torch.randn(b, cfg.vocab, generator=g)
        for i in range(acc):
            logits[i, tokens[i + 1]] = 100.0
        if acc < b - 1:
            logits[acc, (int(tokens[acc + 1]) + 1) % cfg.vocab] = 100.0
        return logits, _randn(g, b, cfg.features * cfg.hidden)

    return verify


@pytest.mark.parametrize("block", [8, 16])
def test_iterations_carry_state_and_tp8_matches(block):
    """Three greedy iterations through a fake target, unsharded and TP8 (fp32): the same tokens, positions and anchors
    every iteration; the committed context rows in the KV are the context update of the last verify's features.
    """
    cfg = SMALL
    w, t = _weights(cfg)
    plan = {40: 3, 44: 0, 45: block - 1}

    def run(ws, ts, kv, comm):
        state, log = _prefill(cfg, 40), []
        for _ in range(3):
            dr, acc, nxt = R.run_iteration(
                cfg, ws, ts, kv, state, block, _fake_verify(cfg, plan), comm, F32
            )
            log.append((state, dr, acc))
            state = nxt
        lo = state.pos - state.commit_len
        ctx = R.context_update(cfg, ws, state.feats, lo, comm, F32)
        R.write_context(kv, ctx, lo, state.commit_len)
        return log, state, kv

    kv1 = R.DraftKV.empty(cfg, 96, 1, F32)
    log1, end1, _ = run(w, t, kv1, R.SoloComm())
    assert [s.pos for s, _, _ in log1] == [40, 44, 45] and end1.pos == 45 + block
    for (s, dr, acc), nxt in zip(log1, [s for s, _, _ in log1[1:]] + [end1]):
        n = acc.accept_len
        assert n == plan[s.pos] and nxt.anchor == acc.bonus and nxt.commit_len == n + 1
        assert torch.equal(acc.out_tokens[:n], dr.tokens[1 : n + 1])
    # every committed row holds the context K / V of its feature
    for s in [s for s, _, _ in log1] + [end1]:
        lo = s.pos - s.commit_len
        ctx = R.context_update(cfg, w, s.feats, lo, R.SoloComm(), F32)
        for i, (k, v) in enumerate(ctx):
            assert torch.equal(kv1.k[i][lo : s.pos], k[: s.commit_len])
            assert torch.equal(kv1.v[i][lo : s.pos], v[: s.commit_len])

    def rank(comm):
        ws, ts = R.shard(cfg, w, t, comm.rank, comm.ranks)
        return run(ws, ts, R.DraftKV.empty(cfg, 96, comm.ranks, F32), comm)

    for r, (log, end, kv) in enumerate(R.emulate_ranks(rank, 8)):
        assert end.pos == end1.pos and end.anchor == end1.anchor
        for (s, dr, acc), (s1, dr1, acc1) in zip(log, log1, strict=True):
            assert (s.pos, s.anchor) == (s1.pos, s1.anchor)
            assert torch.equal(dr.tokens, dr1.tokens)
            _close(dr.y, dr1.y, 1e-5)
        ref = _kv_slice(kv1, r, 8)
        for got, want in zip(kv.k + kv.v, ref.k + ref.v, strict=True):
            _close(got[: end.pos], want[: end.pos], 1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# One real layer


def test_real_layer_tp8_equals_unsharded():
    """Draft layer 0 of the staged DFlash2 checkpoint at the real dims: the loader's fused layouts feed a finite layer,
    and TP8 equals the unsharded layer (fp32 to 1e-5, bf16 to the row-parallel rounding), at a context past the 2048
    window. The random input is off the trained distribution, so the output magnitudes mean nothing.
    """
    if CKPT_DIR is None:
        pytest.skip("SGL_MUSE_MEGA_DRAFT_PATH is not set")
    path = CKPT_DIR / "model.safetensors"
    cfg = dataclasses.replace(
        R.DraftConfig.from_hf(json.loads((CKPT_DIR / "config.json").read_text())),
        layers=1,
    )
    lw = R.checkpoint_layer(str(path), 0)
    g = torch.Generator().manual_seed(10)
    block, pos = 8, 3000
    x = _randn(g, block, cfg.hidden, scale=0.05)
    ctx_k = _randn(g, pos, cfg.kv_heads, cfg.head_dim)
    ctx_v = _randn(g, pos, cfg.kv_heads, cfg.head_dim)

    def layer(lw_r, comm, act):
        kvh = cfg.kv_heads // comm.ranks
        k = torch.zeros(pos + block, kvh, cfg.head_dim, dtype=act)
        v = torch.zeros_like(k)
        k[:pos] = R._block(ctx_k, comm.rank, comm.ranks, 1)
        v[:pos] = R._block(ctx_v, comm.rank, comm.ranks, 1)
        out, residual = R.draft_layer(cfg, lw_r, x, None, k, v, pos, comm, act)
        return out, residual, k[pos:]

    for act, close, rtol in [(F32, _close, 1e-5), (BF16, _close_rows, 2**-6)]:
        one, res1, k1 = layer(lw, R.SoloComm(), act)
        assert torch.isfinite(one.float()).all() and one.float().abs().max() > 0
        ranks = R.emulate_ranks(
            lambda c: layer(R.shard_layer(cfg, lw, c.rank, c.ranks), c, act), 8
        )
        for r, (out, res, k) in enumerate(ranks):
            close(out, one, rtol)
            close(res, res1, rtol)
            # the block K is computed before the first reduction: bitwise
            assert torch.equal(k, R._block(k1, r, 8, 1))


BF16 = torch.bfloat16
RING = 4096  # the draft KV ring (kernels/dflash/layer.cu kRing)


def test_side1_packet_order_is_the_finish_packet():
    """The gathered side-1 coefficients of the permuted projection are the packed rows the o_proj tile polls, and
    side 0 is untouched (the entry's prepare reads it in the checkpoint's order)."""
    taps, groups, k = 2, 512, 64
    g = torch.Generator().manual_seed(3)
    proj = torch.randn(2 * taps * groups, k, generator=g).to(BF16)
    h = torch.randn(3, k, generator=g).to(BF16)
    plain = R._mm(h, proj, BF16).view(3, 2, taps, groups)
    got = R._mm(h, dweights.side1_packet_order(proj, taps), BF16)
    n = taps * groups
    assert torch.equal(got[:, :n], plain[:, 0].reshape(3, n))
    assert torch.equal(got[:, n:], dweights.pack_coeff(plain[:, 1]))
    assert torch.equal(dweights.unpack_coeff(got[:, n:], taps), plain[:, 1])


@pytest.mark.parametrize("b", DFLASH_BLOCKS)
@pytest.mark.parametrize("commit", [1, 3, 8, 16])
@pytest.mark.parametrize("pos", [4096, 4093, 128000])
def test_accept_draft_feeds_are_the_qkv_rows(b, commit, pos):
    """verify_accept's draft feeds row for row the qkv's x order (R.qkv_feeds: block rows, then context rows); a row
    the commit masks carries slot -1; the attention's kv_len is P' + B."""
    if commit > b:
        pytest.skip("c' <= B")
    cfg = R.DraftConfig()
    cache = iteration.draft_rope_cache(pos + 2 * b + 1, cfg)
    f = head.draft_feeds(cache, pos, commit, b, RING)
    positions, cos_sin, slots = R.qkv_feeds(pos, b, commit, cfg, RING)
    kept = torch.cat([torch.arange(b), b + torch.arange(commit)])
    masked = b + torch.arange(commit, b)
    assert torch.equal(f["dslot"][kept], slots[kept])
    assert torch.equal(f["dcos_sin"][kept], cache[positions[kept]])
    # the same rope rows up to the libm's vector / scalar cos (the cache is computed once over every position)
    assert torch.allclose(f["dcos_sin"][kept], cos_sin[kept], rtol=0, atol=1e-6)
    assert bool((f["dslot"][masked] == -1).all())
    assert f["dkv_len"].tolist() == [pos + b]


def test_synthetic_draft_shards():
    cfg, ranks = R.DraftConfig(), 8
    q_heads, _ = layer.qkv_heads(cfg, ranks)
    inter = cfg.inter // ranks
    lw = synthetic.draft_layer(cfg, 0, 3, ranks)
    assert lw.qkv.shape == (layer.qkv_cols(cfg, ranks), cfg.hidden)
    assert lw.o.shape == (cfg.hidden, q_heads * cfg.head_dim)
    assert lw.gate_up.shape == (2 * inter, cfg.hidden) and lw.down.shape == (
        cfg.hidden,
        inter,
    )
    assert lw.attn_conv.proj.shape == (2 * cfg.conv_taps * cfg.conv_groups, cfg.hidden)
    other = synthetic.draft_layer(cfg, 0, 4, ranks)
    assert torch.equal(lw.attn_conv.proj, other.attn_conv.proj)  # replicated
    assert torch.equal(lw.input_norm, other.input_norm)
    assert not torch.equal(lw.o, other.o)  # the rank's own shard
    assert torch.equal(
        dweights.unpack_gate_up(dweights.pack_gate_up(lw.gate_up)), lw.gate_up
    )


def test_comparisons():
    a = torch.randn(4, 64).to(BF16)
    assert run.bits_equal(a, a.clone()) and not run.bits_equal(a, -a)
    assert run.row_band(a, a)["ok"] and not run.row_band(a, a * 1.1)["ok"]
    z = torch.zeros(2, 4, dtype=torch.float32)
    assert not run.bits_equal(z, -z)  # -0 and +0 differ bitwise
    rec = dict(y=a, words=torch.arange(8))
    one = dict(recs=[rec, rec], rings=a, caps=a)
    same = run.same_run(
        one,
        dict(recs=[rec, dict(y=-a, words=torch.arange(8))], rings=a, caps=a),
        keys=("y", "words"),
    )
    assert (
        same["y"] == dict(ok=False, first_diff=1)
        and same["words"]["ok"]
        and same["rings"]["ok"]
    )


def test_sublayer_orders():
    """dflash2's 'pre_ar' sublayer is the layer kernel's (each rank's fp32 partial finished and rounded, then
    the rank-order fp32 sum rounded once); 'post_ar' is SGLang's; the two differ only by rounding.
    """
    ranks, b, k, h, groups = 2, 8, 64, 128, 8
    g = torch.Generator().manual_seed(11)
    base = (torch.randn(2, 2, h, generator=g) * 0.05).to(BF16)
    base[:, 0] += 1
    conv = R.ConvWeights(base, torch.zeros(1, 1))
    coeff = (torch.randn(b, 2, groups, generator=g) * 0.05).to(BF16)
    xs = [torch.randn(b, k, generator=g).to(BF16) for _ in range(ranks)]
    ws = [(torch.randn(h, k, generator=g) * 0.1).to(BF16) for _ in range(ranks)]

    def sublayer(order):
        return R.emulate_ranks(
            lambda comm: R.sublayer_out(
                conv, xs[comm.rank], ws[comm.rank], coeff, comm, BF16, order
            ),
            ranks,
        )[0]

    finished = [
        R.grouped_conv(R._mm(x, w, torch.float32), coeff, base[1], BF16)
        for x, w in zip(xs, ws)
    ]
    want = (finished[0].float() + finished[1].float()).to(BF16)
    pre, post = sublayer("pre_ar"), sublayer("post_ar")
    assert torch.equal(pre, want)
    assert not torch.equal(pre, post) and run.row_band(pre, post)["ok"]


# ---- the reference checks on emulated ranks: the reference standing in for the kernels must read exact ----


def _kernel_run(cfg, w, t, rings0, p0, b, iters, override, comm, ring):
    """The reference in the kernels' sublayer order ('pre_ar'), laid out as an eager run's records: per iteration
    every draft layer's buffers, the entry's ctx rows, the head's outputs, a synthetic per-row argmax and the step
    words the forced accept writes, the next iteration's features."""
    eps, wl, hd = cfg.rms_eps, cfg.window_left, cfg.head_dim
    layers = cfg.layers
    qh, kvh = R._heads(cfg, w.layers[0])
    kv = R.DraftKV.empty(cfg, p0 + (iters + 2) * b, comm.ranks, BF16)
    pos = torch.arange(0, p0)
    for i in range(layers):
        kv.k[i][pos], kv.v[i][pos] = (
            rings0[i, 0][pos % ring].view(-1, kvh, hd),
            rings0[i, 1][pos % ring].view(-1, kvh, hd),
        )
    words = torch.zeros(head.N_WORDS, dtype=torch.int64)
    # an anchor inside the small vocab
    words[WORDS["P"]], words[WORDS["ANCHOR"]] = p0, 77
    words[WORDS["ACCEPT"]], words[WORDS["COMMIT"]] = (
        synthetic.COMMIT0 - 1,
        synthetic.COMMIT0,
    )
    gen = torch.Generator().manual_seed(5)
    caps = torch.randn(cfg.features, b, cfg.hidden, generator=gen)
    recs = []

    def ring_of(i: int, j: int) -> torch.Tensor:
        out = torch.zeros(ring, kvh * hd, dtype=BF16)
        n = int(words[WORDS["P"]]) + b
        p = torch.arange(max(0, n - ring), n)
        out[p % ring] = (kv.k if j == 0 else kv.v)[i][p].flatten(1)
        return out

    for _ in range(iters):
        P, anchor, commit, itn = (
            int(words[WORDS[k]]) for k in ("P", "ANCHOR", "COMMIT", "ITER")
        )
        words_in = words.clone()
        feats = torch.cat([caps[k].to(BF16) for k in range(cfg.features)], 1)
        for i, (k, v) in enumerate(
            R.context_update(cfg, w, feats, P - commit, comm, BF16)
        ):
            kv.k[i][P - commit : P], kv.v[i][P - commit : P] = k[:commit], v[:commit]
        ctx_n = R.rms_norm(
            comm.all_gather(R._mm(feats, w.fc, BF16), 1), w.hidden_norm, eps, BF16
        )
        ids = torch.full((b,), cfg.mask_id, dtype=torch.int64)
        ids[0] = anchor
        r_a = R.embed(ids, t.embed, comm, BF16)
        h = R.rms_norm(r_a, w.layers[0].input_norm, eps, BF16)
        lo, positions = max(0, P - wl), P + torch.arange(b)
        cos, sin = R.rope_cos_sin(positions, hd, cfg.rope_theta)
        bufs = []
        for i, lw in enumerate(w.layers):
            hh, c_att = R.conv_prepare(lw.attn_conv, h, BF16)
            q, k, v = R._mm(hh, lw.qkv, BF16).split([qh * hd, kvh * hd, kvh * hd], 1)
            qn = R.head_norm_rope(
                q.reshape(b, qh, hd), lw.q_norm, eps, cos, sin, BF16, True
            )
            kv.k[i][P : P + b] = R.head_norm_rope(
                k.reshape(b, kvh, hd), lw.k_norm, eps, cos, sin, BF16, True
            )
            kv.v[i][P : P + b] = v.reshape(b, kvh, hd)
            o = R.attention(
                qn,
                kv.k[i][lo : P + b],
                kv.v[i][lo : P + b],
                positions,
                torch.arange(lo, P + b),
                wl,
            )
            o = o.to(BF16)
            a = R.sublayer_out(lw.attn_conv, o, lw.o, c_att, comm, BF16, "pre_ar")
            h2, r_b = R.add_rms_norm(a, r_a, lw.post_norm, eps, BF16)
            x_gu, c_mlp = R.conv_prepare(lw.mlp_conv, h2, BF16)
            act = R.silu_mul(*R._mm(x_gu, lw.gate_up, BF16).chunk(2, 1), BF16)
            d = R.sublayer_out(lw.mlp_conv, act, lw.down, c_mlp, comm, BF16, "pre_ar")
            nxt = w.layers[i + 1].input_norm if i + 1 < layers else w.norm
            h_next, r_c = R.add_rms_norm(d, r_b, nxt, eps, BF16)
            qout = torch.zeros(b, (qh + 2 * kvh) * hd, dtype=BF16)
            qout[:, : qh * hd] = qn.reshape(b, qh * hd)
            bufs.append(
                dict(
                    r_a=r_a,
                    x_attn=o,
                    r_b=r_b,
                    h2=h2,
                    x_gu=x_gu,
                    act=act,
                    r_c=r_c,
                    h=h_next,
                    qout=qout,
                )
            )
            r_a, h = r_c, h_next
        y = h
        for i, lb in enumerate(bufs):
            lb.update(k_ring=ring_of(i, 0), v_ring=ring_of(i, 1))
        cand, unary = R.candidates(cfg, t, y[1:], comm, BF16)
        hhat = R._mm(y[1:], w.hidden_proj, BF16)
        scores = R.lattice_hhat(
            w.pred_codebook, w.succ_codebook, hhat, cand, unary, anchor
        )
        drafts, _ = R.selector_walk(cand, scores)
        win = R.verify_ids(anchor, drafts)
        top1 = R.first_argmax(torch.randn(b, cfg.vocab, generator=gen))
        acc = R.accept_ref(
            win[:b], top1, P, forced=int(override[itn % override.numel()])
        )
        words[WORDS["P"]], words[WORDS["ANCHOR"]] = acc.pos, acc.bonus
        words[WORDS["ACCEPT"]], words[WORDS["COMMIT"]] = (
            acc.accept,
            acc.commit,
        )
        words[WORDS["ITER"]] += 1
        recs.append(
            dict(
                words_in=words_in,
                caps_in=caps,
                y=y,
                cand=cand.int(),
                unary=unary,
                hhat=hhat,
                scores=scores,
                drafts=drafts,
                ids=win,
                top1=top1,
                words=words.clone(),
                layers=bufs,
                ctx_n=ctx_n,
            )
        )
        caps = torch.randn(cfg.features, b, cfg.hidden, generator=gen)
    rings = torch.stack(
        [torch.stack([ring_of(i, 0), ring_of(i, 1)]) for i in range(layers)]
    )
    return dict(recs=recs, rings=rings, caps=caps)


@pytest.mark.parametrize("ranks", [8, 4])
@pytest.mark.parametrize("b", DFLASH_BLOCKS)
def test_checks_read_exact_on_the_reference(b, ranks):
    """reference_check (tokens, accept, the forced suffixes, the end-to-end record) and stage_check (every unit) fed an
    eager run whose 'kernel' is the reference itself in the kernels' order: every check passes and every distance is 0
    (the attention's: its bf16 rounding).
    """
    import types

    from test_dflash_reference import SMALL, _weights

    cfg, ring, p0, iters = SMALL, 64, 40, 3
    kvh = cfg.kv_heads // ranks
    w, t = _weights(cfg, seed=3)
    override = synthetic.forced_accepts(b)
    target = types.SimpleNamespace(
        vocab=cfg.vocab, vocab_rank=cfg.vocab // ranks, tp=ranks
    )
    g = torch.Generator().manual_seed(6)
    rings0 = (
        torch.rand(cfg.layers, 2, ring, kvh * cfg.head_dim, generator=g) * 0.2 - 0.1
    ).to(BF16)

    def rank_run(comm):
        wr, tr = R.shard(cfg, w, t, comm.rank, comm.ranks)
        eager = _kernel_run(cfg, wr, tr, rings0, p0, b, iters, override, comm, ring)
        cx = types.SimpleNamespace(
            cfg=target,
            dcfg=cfg,
            bufs=types.SimpleNamespace(embed_shard=lambda _cfg: tr.embed),
            hw=tr.lm_head,
            ring=ring,
            capture=cfg.capture,
            rank=comm.rank,
            target=True,
        )
        it = types.SimpleNamespace(b=b, override=override, tree=False)
        clean = run.reference_check(
            cx, wr, it, eager, rings0, p0, comm
        ), run.stage_check(cx, wr, it, eager, comm)
        # the negative control: layer 0's output norm of iteration 1 off by 3 %
        eager["recs"][1]["layers"][0]["h"] = (
            eager["recs"][1]["layers"][0]["h"].float() * 1.03
        ).to(BF16)
        return clean, run.reference_check(cx, wr, it, eager, rings0, p0, comm)

    for (ref, st), bad in R.emulate_ranks(rank_run, ranks):
        assert (
            not bad["ok"] and not bad["iterations"][1]["forced"]["layer"][0]["h"]["ok"]
        )
        assert (
            bad["iterations"][1]["forced"]["layer"][0]["r"]["ok"]
            and bad["iterations"][0]["forced"]["ok"]
        )
        assert ref["ok"] and st["ok"], (ref, st)
        assert run.checks_pass({"reference": ref, "stages": st})
        for g_ in ref["iterations"]:
            assert g_["forced"]["y_rel"] == [0.0] * cfg.layers
            assert all(
                x["r"]["rel_max"] == 0 and x["h"]["rel_max"] == 0
                for x in g_["forced"]["layer"]
            )
            assert g_["y_end_to_end"]["rel_max"] == 0 and g_["y_end_to_end"]["ok"]
            assert (
                g_["accept"]["ok"]
                and g_["window_ids"]["ok"]
                and g_["head_drafts"]["ok"]
            )
        assert all(v["rel_max"] == 0 for v in ref["ring_end_to_end"].values())
        for k, v in st["worst"].items():
            if "rel_max" in v:
                # the attention's reference is the fp64 attention: the kernel's x_attn is its bf16 rounding
                assert (
                    v["rel_max"] < 2**-8
                    if k.endswith("attn_combine")
                    else v["rel_max"] == 0
                ), (k, v["rel_max"])
        assert st["worst"]["entry_resid_equal"]["ok"]
        g = {
            "reference": ref,
            "stages": st,
            "draft_head": {"ok": True, "per_iteration": []},
            "verify": {"ok": True},
        }
        lines = run.reference_lines(b, g)
        assert (
            lines[0].startswith(f"b{b} reference PASS") and "outside: []" in lines[1]
        ), lines
