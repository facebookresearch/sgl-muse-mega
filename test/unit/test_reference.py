"""CPU tests of the verify reference (test/reference/muse_spark.py): the number formats, the weight layouts (round
trips), the attention mask, TP8 == unsharded, and one layer at the real TP8 dims (no GPU is touched).
"""

import math

import pytest
import torch
from reference import muse_spark as R
from sgl_muse_mega.muse_spark.config import Config
from sgl_muse_mega.muse_spark.sm103 import weights as L

# the emulated ranks are threads: keep their BLAS pools from oversubscribing the box
torch.set_num_threads(16)


def _sf_bytes(*shape, gen, lo=0x28, hi=0x38):
    """Finite e4m3 scale bytes in a band (0x28..0x37 = 0.25 .. 0.94): no NaN code, O(1) products."""
    return torch.randint(lo, hi, shape, generator=gen, dtype=torch.uint8)


def _codes(*shape, gen):
    return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# Number formats


def test_e2m1_codes_and_ties():
    codes = torch.arange(16, dtype=torch.uint8)
    # -0 keeps its sign bit
    assert torch.equal(R.e2m1_encode(R.e2m1_decode(codes)), codes)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0, -100.0])
    # ties to the even code, saturating
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, -6.0])
    assert torch.equal(R.e2m1_decode(R.e2m1_encode(mids)), want)
    b = _codes(3, 8, gen=torch.Generator().manual_seed(0))
    assert torch.equal(R.pack_nibbles(R.unpack_nibbles(b)), b)
    # element 0 = the low nibble
    assert int(R.unpack_nibbles(torch.tensor([0x21], dtype=torch.uint8))[0]) == 1


def test_e4m3_round_trip_and_rne():
    finite = torch.arange(0, 0x7F, dtype=torch.uint8)
    vals = R.e4m3_decode(finite)
    assert torch.equal(R.e4m3_encode(vals), finite)
    mids = (vals[:-1] + vals[1:]) / 2  # exact in fp32; a tie goes to the even byte
    want = torch.where(finite[:-1] % 2 == 0, finite[:-1], finite[1:])
    assert torch.equal(R.e4m3_encode(mids), want)
    x = torch.rand(10000, generator=torch.Generator().manual_seed(1)) * 448
    nearest = (x[:, None] - vals[None, :]).abs().argmin(-1).to(torch.uint8)
    assert torch.equal(R.e4m3_encode(x), nearest)
    assert int(R.e4m3_encode(torch.tensor([1e6]))[0]) == 0x7E  # satfinite


def test_tf32_truncates():
    x = torch.tensor([1 + 2**-10, 1 + 2**-11, -(1 + 2**-10 + 2**-12)])
    assert torch.equal(R.tf32(x), torch.tensor([1 + 2**-10, 1.0, -(1 + 2**-10)]))


def test_quantize_expert_input_bounds():
    x = (
        torch.randn(16, 4096, generator=torch.Generator().manual_seed(2)) * 3
    ).bfloat16()
    q, sf, xs = R.quantize_expert_input(x)
    assert q.shape == (16, 2048) and sf.shape == (16, 256) and xs.shape == (16,)
    v = x.float()
    assert torch.equal(xs, v.abs().amax(-1) * R._F32_INV_2688)
    err = (R.dequant_nvfp4(q, sf) * xs[:, None] - v).view(16, 256, 16).abs().amax(-1)
    # one e2m1 unit (half the widest code step) of a scale e4m3(amax / 6), which rounds up by at most 2^-4
    bound = v.view(16, 256, 16).abs().amax(-1) / 6 * (1 + 2**-4) * (1 + 1e-6)
    assert bool((err <= bound).all())


def test_quantize_nvfp4_weight():
    w = torch.randn(4, 256, generator=torch.Generator().manual_seed(15))
    q, sf = R.quantize_nvfp4(w)
    err = (R.dequant_nvfp4(q, sf) - w).view(4, 16, 16).abs().amax(-1)
    assert bool(
        (err <= w.view(4, 16, 16).abs().amax(-1) / 6 * (1 + 2**-4) * (1 + 1e-6)).all()
    )


def test_layer_kinds():
    cfg = Config()
    kinds = [
        (cfg.attn_type(i), "bf16" if cfg.is_bf16(i) else "fp4")
        for i in (0, 1, 2, 5, 60, 61)
    ]
    assert kinds == [
        ("swa", "bf16"),
        ("global", "fp4"),
        ("swa", "fp4"),
        ("global", "fp4"),
        ("swa", "fp4"),
        ("global", "bf16"),
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Weight layouts


def test_packs_round_trip_at_real_dims():
    d, gen, e = Config(), torch.Generator().manual_seed(3), 3
    hb, ib = d.moe_hidden // 2, d.moe_inter // 2
    gq, uq = _codes(e, d.inter_rank, hb, gen=gen), _codes(e, d.inter_rank, hb, gen=gen)
    gs, us = _sf_bytes(e, d.inter_rank, d.moe_hidden // 16, gen=gen), _sf_bytes(
        e, d.inter_rank, d.moe_hidden // 16, gen=gen
    )
    w1, w1_sf = L.pack_w1_fp4(gq, gs, uq, us)
    assert w1.shape == (e, 4, 16, 256, 128) and w1_sf.shape == (e * 4, 65536)
    for a, b in zip(L.unpack_w1_fp4(w1, w1_sf), (gq, gs, uq, us)):
        assert torch.equal(a, b)
    w2q, w2s = _codes(e, d.moe_hidden_rank, ib, gen=gen), _sf_bytes(
        e, d.moe_hidden_rank, d.moe_inter // 16, gen=gen
    )
    w2, w2_sf = L.pack_w2_fp4(w2q, w2s)
    assert w2.shape == (e, 4, 16, 128, 128) and w2_sf.shape == (e, 512, 256)
    assert all(
        torch.equal(a, b) for a, b in zip(L.unpack_w2_fp4(w2, w2_sf), (w2q, w2s))
    )
    gate, up = (
        torch.randn(e, d.inter_rank, d.moe_hidden, generator=gen).bfloat16()
        for _ in range(2)
    )
    w1b = L.pack_w1_bf16(gate, up)
    assert w1b.shape == (e, 4, 64, 256, 64)
    assert all(torch.equal(a, b) for a, b in zip(L.unpack_w1_bf16(w1b), (gate, up)))
    w2h = torch.randn(e, d.moe_hidden_rank, d.moe_inter, generator=gen).bfloat16()
    assert L.pack_w2_bf16(w2h).shape == (e, 4, 64, 128, 64)
    assert torch.equal(L.unpack_w2_bf16(L.pack_w2_bf16(w2h)), w2h)


def test_attention_mask_and_gqa():
    gen = torch.Generator().manual_seed(7)
    m, n, hq, hkv, window = 16, 30, 8, 2, 9
    qkv = torch.randn(m, (2 * hq + 2 * hkv) * 64, generator=gen).bfloat16()
    k_ctx, v_ctx = (torch.randn(n, hkv, 64, generator=gen).bfloat16() for _ in range(2))
    for win in (window, None):
        got = R.attention(qkv, k_ctx, v_ctx, win).view(m, hq, 64)
        q, k_new, v_new, og = R.split_qkv(qkv, hq, hkv)
        k, v = torch.cat((k_ctx, k_new)).double(), torch.cat((v_ctx, v_new)).double()
        for t in range(m):
            pos = n + t
            lo = 0 if win is None else max(0, pos - win)
            for h in range(hq):
                kv = h // (hq // hkv)
                s = (k[lo : pos + 1, kv] @ q[t, h].double()) / 8
                o = (torch.softmax(s, 0) @ v[lo : pos + 1, kv]).float()
                o = (
                    o
                    * torch.rsqrt(o.double().square().mean().float() + 1e-5)
                    * torch.sigmoid(og[t, h].float())
                )
                assert torch.equal(got[t, h], o.bfloat16()), (win, t, h)


# ---------------------------------------------------------------------------------------------------------------------
# TP8 == unsharded (reduced dims, the real head and quant grains)

SMALL = dict(
    hidden=1024,
    moe_hidden=1024,
    inter=1024,
    experts=16,
    q_heads=64,
    kv_heads=8,
    n_ctx=40,
    window=7,
)


def _norm_gates(n, gen):
    """alpha / beta of a norm-preserving residual from a random gate w: alpha = sqrt(sigmoid(-w) (1 + beta)),
    beta = sigmoid(w)."""
    w = torch.randn(n, generator=gen) / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1 + beta), min=1e-3)), beta


def _bf16_ones(n, gen):
    return (1 + 0.1 * torch.randn(n, generator=gen)).bfloat16()


def _qkv_parts(rows_q, rows_kv, cols, gen, scale):
    return [
        torch.randn(r, cols, generator=gen).mul(scale).bfloat16()
        for r in (rows_q, rows_kv, rows_kv, rows_q)
    ]


def _rank_rows(parts, r, ranks):
    """The rank's rows of [q | k | v | og] parts (heads split evenly)."""
    return torch.cat(
        [p[p.shape[0] // ranks * r : p.shape[0] // ranks * (r + 1)] for p in parts]
    )


class Model:
    """Full (unsharded) synthetic weights of one layer + its successor's head, and their per-rank shards."""

    def __init__(self, moe: str, window, nxt: str, seed: int, s=SMALL):
        gen = torch.Generator().manual_seed(seed)
        self.s, self.moe, self.window, self.nxt = s, moe, window, nxt
        h, mh, i, e, hq, hkv = (
            s[k]
            for k in ("hidden", "moe_hidden", "inter", "experts", "q_heads", "kv_heads")
        )
        m, n = 16, s["n_ctx"]
        # [q | k | v | og] columns, stored transposed
        self.qkv_in = _qkv_parts(hq * 64, hkv * 64, m, gen, 1.0)
        self.k_ctx, self.v_ctx = (
            torch.randn(n, hkv, 64, generator=gen).bfloat16() for _ in range(2)
        )
        self.stream = torch.randn(m, h, generator=gen)
        self.w_o = (
            torch.randn(h, hq * 64, generator=gen)
            .mul(1 / math.sqrt(hq * 64))
            .bfloat16()
        )
        self.pa = (*_norm_gates(h, gen), _bf16_ones(h, gen))
        self.w_pool = torch.randn(mh, h, generator=gen).mul(1 / math.sqrt(h)).bfloat16()
        self.w_router = torch.randn(e, h, generator=gen).mul(2 / math.sqrt(h))
        self.bias = 0.1 * torch.randn(e, generator=gen)
        self.pre_gamma, self.gamma_p1 = _bf16_ones(mh, gen), _bf16_ones(mh, gen)
        if moe == "fp4":
            self.w13 = (
                _codes(e, 2 * i, mh // 2, gen=gen),
                _sf_bytes(e, 2 * i, mh // 16, gen=gen),
            )
            self.w2 = (
                _codes(e, mh, i // 2, gen=gen),
                _sf_bytes(e, mh, i // 16, gen=gen),
            )
            self.alphas = (
                0.03 * (1 + 0.1 * torch.rand(e, generator=gen)),
                1 + 0.1 * torch.rand(e, generator=gen),
                0.03 * (1 + 0.1 * torch.rand(e, generator=gen)),
            )
        else:
            self.w13 = (
                torch.randn(e, 2 * i, mh, generator=gen)
                .mul(1 / math.sqrt(mh))
                .bfloat16()
            )
            self.w2 = (
                torch.randn(e, mh, i, generator=gen).mul(1 / math.sqrt(i)).bfloat16()
            )
        self.w_unpool = (
            torch.randn(h, mh, generator=gen).mul(1 / math.sqrt(mh)).bfloat16()
        )
        self.x = (*_norm_gates(h, gen), _bf16_ones(h, gen), _bf16_ones(h, gen))
        self.next_qkv = (
            _qkv_parts(hq * 64, hkv * 64, h, gen, 1 / math.sqrt(h)) if nxt else None
        )
        pos = torch.arange(n, n + m)
        self.cos_sin = R.rope_cos_sin(pos) if nxt == "swa" else None

    def weights(self, r: int, ranks: int) -> R.LayerWeights:
        if self.moe == "fp4":
            i, mh = self.w13[0].shape[1] // 2, self.w2[0].shape[1]
            (gq, uq), (gs, us) = L.shard_w13(self.w13[0], r, ranks, i), L.shard_w13(
                self.w13[1], r, ranks, i
            )
            w1, w1_sf = L.pack_w1_fp4(gq, gs, uq, us)
            w2, w2_sf = L.pack_w2_fp4(
                L.shard_w2(self.w2[0], r, ranks, mh),
                L.shard_w2(self.w2[1], r, ranks, mh),
            )
            experts = R.Fp4Experts(w1, w1_sf, w2, w2_sf, *self.alphas)
        else:
            i, mh = self.w13.shape[1] // 2, self.w2.shape[1]
            experts = R.Bf16Experts(
                L.pack_w1_bf16(*L.shard_w13(self.w13, r, ranks, i)),
                L.pack_w2_bf16(L.shard_w2(self.w2, r, ranks, mh)),
            )
        k = self.w_o.shape[1] // ranks
        return R.LayerWeights(
            self.w_o[:, k * r : k * (r + 1)],
            *self.pa,
            self.w_pool,
            self.w_router,
            self.bias,
            0.8,
            2.5,
            self.pre_gamma,
            1e-5,
            experts,
            self.gamma_p1,
            self.w_unpool,
            *self.x,
        )

    def inputs(self, r: int, ranks: int) -> R.LayerInputs:
        hkv = self.s["kv_heads"] // ranks
        kv = slice(hkv * r, hkv * (r + 1))
        qkv = _rank_rows(self.qkv_in, r, ranks).T.contiguous()
        return R.LayerInputs(
            qkv, self.k_ctx[:, kv], self.v_ctx[:, kv], self.stream, self.window
        )

    def head(self, r: int, ranks: int):
        if self.next_qkv is None:
            return None
        return R.NextHead(
            _rank_rows(self.next_qkv, r, ranks),
            43.7840518911 / 8,
            cos_sin=self.cos_sin,
            q_heads=self.s["q_heads"] // ranks,
            kv_heads=self.s["kv_heads"] // ranks,
        )

    def run(self, ranks: int, node_ranks: int, **kw) -> list:
        return R.emulate_ranks(
            lambda c: R.ref_layer(
                self.weights(c.rank, ranks),
                self.inputs(c.rank, ranks),
                c,
                self.head(c.rank, ranks),
                **kw,
            ),
            ranks,
            node_ranks,
        )


def _rank_cols(x, r, ranks, s=SMALL):
    """The rank's columns of an unsharded qkv / x_attn row."""
    hq, hkv = s["q_heads"] * 64, s["kv_heads"] * 64
    if x.shape[1] == hq:
        return x[:, hq // ranks * r : hq // ranks * (r + 1)]
    parts = torch.split(x, (hq, hkv, hkv, hq), dim=1)
    return torch.cat(
        [p[:, p.shape[1] // ranks * r : p.shape[1] // ranks * (r + 1)] for p in parts],
        1,
    )


SHARDED = {"x_attn", "qkv_next"}
CASES = [
    ("fp4", 7, "global"),
    ("fp4", None, "swa"),
    ("bf16", 7, "global"),
    ("bf16", None, None),
]


@pytest.mark.parametrize(
    "moe,window,nxt", CASES, ids=["fp4-swa", "fp4-global", "bf16-swa", "bf16-tail"]
)
def test_tp8_equals_unsharded(moe, window, nxt):
    model = Model(moe, window, nxt, seed=10)
    tp = model.run(8, 4, round_ar_partials=False)
    one = model.run(1, 1)[0]
    assert set(tp[0]) == set(one)
    for name, ref in one.items():
        for r, out in enumerate(tp):
            got = out[name]
            if name in SHARDED:
                ref_r = _rank_cols(ref, r, 8)
            else:
                ref_r = ref
                # replicated values agree bitwise on every rank
                assert torch.equal(got, tp[0][name]), (
                    name,
                    r,
                )
            assert (
                torch.equal(got, ref_r)
                if not got.is_floating_point()
                else R.compare(got, ref_r, name, tol=1e-6)[2]
            ), (name, r)
    assert all(torch.isfinite(v.float()).all() for v in one.values())


@pytest.mark.parametrize(
    "moe,window,nxt", CASES[:3], ids=["fp4-swa", "fp4-global", "bf16-swa"]
)
def test_forced_stages(moe, window, nxt):
    """The kernel-faithful TP8 layer (bf16 AR packets) against the unsharded one forced to its post-attention values and
    routing (the kernel tests' per-stage check): every downstream value agrees."""
    model = Model(moe, window, nxt, seed=11)
    tp = model.run(8, 4)
    forced = {k: tp[0][k] for k in ("stream_mid", "h3", "ids")}
    one = model.run(1, 1, forced=forced)[0]
    for name in (
        "pooled",
        "logits",
        "w",
        "zb",
        "y",
        "ffn_out",
        "stream_out",
        "h3_next",
    ):
        _, rel, ok = R.compare(tp[0][name], one[name], name, tol=1e-6)
        assert ok, (name, rel)
    with pytest.raises(AssertionError):
        model.run(1, 1, forced={"h3f": tp[0]["h3"]})


def _dist_worker(rank: int, store: str, out_dir: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(4)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=8
    )
    comm = R.DistComm(node_ranks=4)
    model = Model("fp4", 7, "global", seed=10)
    out = R.ref_layer(
        model.weights(rank, 8), model.inputs(rank, 8), comm, model.head(rank, 8)
    )
    torch.save(out, f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


def test_distcomm_matches_threads(tmp_path):
    """The same layer over torch.distributed (8 gloo processes, two 4-rank node groups) and over the thread
    emulation: bitwise."""
    import torch.multiprocessing as mp

    mp.spawn(_dist_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=8)
    threads = Model("fp4", 7, "global", seed=10).run(8, 4)
    for r, ref in enumerate(threads):
        got = torch.load(tmp_path / f"rank{r}.pt")
        assert all(torch.equal(got[k], ref[k]) for k in ref), r


def test_entry_and_logits_tp8_equal_unsharded():
    gen = torch.Generator().manual_seed(12)
    v, h = 8 * 96, 1024
    embed = torch.randn(v, h, generator=gen).bfloat16()
    lm = torch.randn(v, h, generator=gen).mul(1 / math.sqrt(h)).bfloat16()
    ids = torch.randint(0, v, (16,), generator=gen)
    pre_w = _bf16_ones(h, gen)
    hfin = torch.randn(16, h, generator=gen).bfloat16()

    def run(c):
        rows = v // c.ranks
        shard = slice(rows * c.rank, rows * (c.rank + 1))
        return R.ref_entry(ids, embed[shard], pre_w, c), R.ref_logits(
            hfin, lm[shard], c, 0.75, 30.0
        )

    tp, (one_entry, one_logits) = (
        R.emulate_ranks(run, 8, 4),
        R.emulate_ranks(run, 1, 1)[0],
    )
    assert torch.equal(one_entry["emb"], embed[ids])
    for entry, logits in tp:
        assert all(torch.equal(entry[k], one_entry[k]) for k in one_entry)
        assert torch.equal(logits, one_logits)
    assert float(one_logits.abs().max()) < 30.0


# ---------------------------------------------------------------------------------------------------------------------
# One FP4 swa layer at the real TP8 dims (only the activated experts exist)


def test_real_dims_fp4_layer():
    d, gen = Config(), torch.Generator().manual_seed(13)
    # the bias below keeps every row's top-8 inside this set
    hot = torch.randperm(d.experts, generator=gen)[:24]
    bias = torch.zeros(d.experts)
    bias[hot] = 1.0  # a sigmoid score is < 1: any hot expert outranks every cold one
    w_pool = (
        torch.randn(d.moe_hidden, d.hidden, generator=gen)
        .mul(1 / math.sqrt(d.hidden))
        .bfloat16()
    )
    w_router = torch.randn(d.experts, d.hidden, generator=gen).mul(
        2 / math.sqrt(d.hidden)
    )
    w_unpool = (
        torch.randn(d.hidden, d.moe_hidden, generator=gen)
        .mul(1 / math.sqrt(d.moe_hidden))
        .bfloat16()
    )
    stream = torch.randn(d.rows, d.hidden, generator=gen)
    # the replicated vectors: one copy for every rank
    pa = (*_norm_gates(d.hidden, gen), _bf16_ones(d.hidden, gen))
    xw = (
        *_norm_gates(d.hidden, gen),
        _bf16_ones(d.hidden, gen),
        _bf16_ones(d.hidden, gen),
    )
    pre_gamma, gamma_p1 = _bf16_ones(d.moe_hidden, gen), _bf16_ones(d.moe_hidden, gen)
    n_ctx = 4096
    pos = torch.arange(n_ctx, n_ctx + d.rows)
    weights, inputs, heads = [], [], []
    for r in range(d.tp):
        g = torch.Generator().manual_seed(100 + r)
        # the packs at their full size, filled only for the hot experts: no other expert is read
        w1 = torch.empty(
            d.experts,
            d.inter_rank // 128,
            d.moe_hidden // 256,
            256,
            128,
            dtype=torch.uint8,
        )
        w1_sf = torch.empty(
            d.experts * d.inter_rank // 128,
            d.moe_hidden // 64 * 1024,
            dtype=torch.uint8,
        )
        w2 = torch.empty(
            d.experts,
            d.moe_hidden_rank // 128,
            d.moe_inter // 256,
            128,
            128,
            dtype=torch.uint8,
        )
        w2_sf = torch.empty(
            d.experts, d.moe_hidden_rank, d.moe_inter // 16, dtype=torch.uint8
        )
        for e in hot.tolist():
            q = [_codes(1, d.inter_rank, d.moe_hidden // 2, gen=g) for _ in range(2)]
            s = [
                _sf_bytes(1, d.inter_rank, d.moe_hidden // 16, gen=g) for _ in range(2)
            ]
            w1[e : e + 1], w1_sf[e * 4 : (e + 1) * 4] = L.pack_w1_fp4(
                q[0], s[0], q[1], s[1]
            )
            w2[e : e + 1], w2_sf[e : e + 1] = L.pack_w2_fp4(
                _codes(1, d.moe_hidden_rank, d.moe_inter // 2, gen=g),
                _sf_bytes(1, d.moe_hidden_rank, d.moe_inter // 16, gen=g),
            )
        ones = torch.full((d.experts,), 0.02)
        experts = R.Fp4Experts(w1, w1_sf, w2, w2_sf, ones, torch.ones(d.experts), ones)
        weights.append(
            R.LayerWeights(
                torch.randn(d.hidden, d.q_heads_rank * 64, generator=g)
                .mul(1 / 32)
                .bfloat16(),
                *pa,
                w_pool,
                w_router,
                bias,
                1.0,
                2.5,
                pre_gamma,
                1e-5,
                experts,
                gamma_p1,
                w_unpool,
                *xw,
            )
        )
        kv = [
            torch.empty(n_ctx, d.kv_heads_rank, 64)
            .uniform_(-0.1, 0.1, generator=g)
            .bfloat16()
            for _ in range(2)
        ]
        inputs.append(
            R.LayerInputs(
                torch.randn(d.rows, d.qkv_row, generator=g).bfloat16(),
                *kv,
                stream,
                d.swa_window,
            )
        )
        w_next = (
            torch.randn(d.qkv_row, d.hidden, generator=g)
            .mul(1 / math.sqrt(d.hidden))
            .bfloat16()
        )
        heads.append(R.NextHead(w_next, 43.7840518911 / 8, cos_sin=R.rope_cos_sin(pos)))

    def run(c, forced=None):
        return R.ref_layer(
            weights[c.rank], inputs[c.rank], c, heads[c.rank], forced=forced
        )

    out = R.emulate_ranks(run, d.tp, d.node_ranks)
    o = out[0]
    shapes = {
        "x_attn": (16, 1024),
        "branch": (16, 8192),
        "h3": (16, 8192),
        "pooled": (16, 4096),
        "logits": (16, 256),
        "ids": (16, 8),
        "w": (16, 8),
        "xq": (16, 2048),
        "x_sf": (16, 256),
        "x_scale": (16,),
        "zb": (16, 8, 4096),
        "y": (16, 4096),
        "ffn_out": (16, 8192),
        "stream_out": (16, 8192),
        "h3_next": (16, 8192),
        "qkv_next": (16, 2304),
    }
    for name, shape in shapes.items():
        assert tuple(o[name].shape) == shape, name
        assert torch.isfinite(o[name].float()).all(), name
    assert bool(torch.isin(o["ids"], hot).all())
    for name in set(o) - SHARDED:
        # replicated on every rank, bitwise
        assert all(torch.equal(x[name], o[name]) for x in out), name
    again = R.emulate_ranks(
        lambda c: run(c, {"ids": out[c.rank]["ids"]}), d.tp, d.node_ranks
    )
    # deterministic; the kernel-ids path
    assert all(torch.equal(a[k], b[k]) for a, b in zip(out, again) for k in o)


def test_compare_semantics():
    ref = torch.randn(4, 64, generator=torch.Generator().manual_seed(14))
    assert R.compare(ref.clone(), ref, "h3") == (0.0, 0.0, True)
    bumped = ref.clone()
    bumped[2] *= 1.01
    _, rel, ok = R.compare(bumped, ref, "h3")
    assert not ok and abs(rel - 0.01) < 1e-6
    nan = ref.clone()
    nan[0, 0] = float("nan")
    assert not R.compare(nan, ref, "h3", tol=float("inf"))[2]
    ids = torch.arange(16, dtype=torch.int32).view(2, 8)
    flipped = ids.clone()
    flipped[1, 3] = 99
    assert R.compare(ids, ids, "ids")[2] and R.compare(flipped, ids, "ids")[1:] == (
        0.5,
        False,
    )
