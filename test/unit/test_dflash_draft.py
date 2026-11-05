"""CPU checks of the DFlash draft's launches (sm103/dflash/head.py's entry, sm103/dflash/layer.py,
sm103/dflash/weights.py's layouts; kernels/dflash/{head,layer}.cu), no GPU.

The entry:
- the W tile packer round-trips and its layout is the kernel's: a walk of the packed tiles with the kernel's own
  addressing (block / K-slice base, stage, row, swizzled chunk) and reduction order is the plain GEMM;
- the reference is dflash2's context update and layer-0 conv prepare, and the kernel's prepared-row order
  (emulate_prep) is grouped_conv's rows 0-1 bitwise;
- both images build with no spills and fit the smem opt-in; the packed parameter block is the cubin's, every bind
  pointer reaches it, and the constants are the kernel's.

The dense pieces (the qkv launch + hd128 rowops; the gate_up + SwiGLU unit of the layer images): the weight packers
against their index formulas and round trips, the op-order references against dflash2 (bitwise where the sum of
squares is exact, a one-ulp band otherwise) and against a literal 32-lane butterfly, the feeds, and the qkv image: it
builds with no spill, its parameter block is its cubin's, every bind pointer reaches the block and the tensor maps are
the executor's.

The draft attention (it runs inside the draft-layer images): its geometry; the window cut covers exactly every visible
key and the ring holds it; the kernel's decomposition (emulated: one block per CTA, the masks, bf16 P and partials, the
combine order) holds the band against the reference model's attention, with masked keys inert. The GPU check is
test/kernels/test_dflash_iteration.py's (stages: the attention and its combine on the kernel's own q rows and rings).

The layer's unit queue: the kernel's RMS order against the reference and against a literal transcription of the
kernel's per-thread indexing, the coefficient packets of the finish and the prepare transcribed per thread against the
reference's grouped conv, the packet layouts, and every (world, B, image) build: its resources (the 1-CTA/SM smem
ceiling) and its packed parameter block against the cubin, at TP8 and TP4.
"""

from __future__ import annotations

import re
import subprocess
from functools import cache
from pathlib import Path

import pytest
import test_param_blocks as T
import torch
from reference import dflash2 as R
from sgl_muse_mega.common import host
from sgl_muse_mega.common.jit import toolkit
from sgl_muse_mega.muse_spark.sm103 import artifacts
from sgl_muse_mega.muse_spark.sm103.dflash import head, layer
from sgl_muse_mega.muse_spark.sm103.dflash import weights as dweights

CUOBJDUMP = toolkit("cuobjdump")
OPTIN_SMEM = 232448
ENTRY_PREFIX = "_ZN2mk6dflash11draft_entryI"


def _rand_bf16(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


TN, SK = dweights.TILE_N, dweights.STAGE_K  # packed tile rows, its K columns
CH = SK // 8  # 16-B chunks per tile row


@pytest.mark.parametrize(
    "n,k",
    [(TN, SK), (2 * TN, 3 * SK), (256, 8192)],
    ids=["1x1", "2x3", "conv-proj-rank"],
)
def test_pack_roundtrip(n, k):
    w = _rand_bf16(n, k, seed=n + k)
    p = dweights.pack_tiles(w)
    assert p.shape == (n // TN, k // SK, TN, SK)
    assert torch.equal(dweights.unpack_tiles(p), w)
    # one tile by hand: row r, chunk c sits at chunk c ^ (r & 7)
    t = p[0, 0].view(TN, CH, 8)
    for r in (0, 5, 13, TN - 1):
        for c in (0, 3, CH - 1):
            assert torch.equal(t[r, c ^ (r & 7)], w[r, 8 * c : 8 * c + 8])


def kernel_order_matmul(
    packed: torch.Tensor, x: torch.Tensor, split: int
) -> torch.Tensor:
    """out [rows, N] from the packed tiles read the way draft_entry reads them: CTA (blk, slice) walks stages
    [slice S, (slice + 1) S) from tile base (blk stages_total + slice S); stage `it` covers K columns
    k0 + it SK; the ldmatrix of tile row r = 32 w + 16 m + (lane & 15) reads logical chunk 2 ks + (lane >> 4) at
    physical chunk chunk ^ (r & 7); the slice partials sum in slice order. fp64 throughout (the layout, not the
    rounding).
    """
    blocks, total = packed.shape[:2]
    stages = total // split
    flat = packed.reshape(-1, TN * SK)
    out = torch.zeros(x.shape[0], blocks * TN, dtype=torch.float64)
    for blk in range(blocks):
        parts = []
        for sl in range(split):
            base, k0 = blk * total + sl * stages, sl * stages * SK
            acc = torch.zeros(TN, x.shape[0], dtype=torch.float64)
            for it in range(stages):
                tile = flat[base + it].view(TN, CH, 8)
                xs = x[:, k0 + it * SK : k0 + (it + 1) * SK].double()
                for ks in range(SK // 16):
                    w = torch.empty(TN, 16, dtype=torch.float64)
                    for row in range(TN):
                        for half in range(2):  # lanes 0-15 / 16-31 of the x4 load
                            chunk = 2 * ks + half
                            w[row, 8 * half : 8 * half + 8] = tile[
                                row, chunk ^ (row & 7)
                            ].double()
                    acc += w @ xs[:, 16 * ks : 16 * ks + 16].T
            parts.append(acc)
        s = parts[0]
        for p in parts[1:]:
            s = s + p
        out[:, blk * TN : (blk + 1) * TN] = s.T
    return out


@pytest.mark.parametrize(
    "rows,blocks,stages,split", [(8, 1, 4, 2), (16, 2, 8, 4)], ids=["b8", "b16"]
)
def test_packed_tiles_are_the_gemm(rows, blocks, stages, split):
    w = _rand_bf16(blocks * TN, stages * SK, seed=rows)
    x = _rand_bf16(rows, stages * SK, seed=rows + 1)
    got = kernel_order_matmul(dweights.pack_tiles(w), x, split)
    want = x.double() @ w.double().T
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def small_cfg() -> R.DraftConfig:
    return R.DraftConfig(
        hidden=256,
        q_heads=8,
        kv_heads=2,
        head_dim=32,
        inter=512,
        vocab=64,
        conv_group=16,
    )


def small_weights(cfg: R.DraftConfig) -> R.EntryWeights:
    H, F = cfg.hidden, cfg.features
    return R.EntryWeights(
        fc=_rand_bf16(H, F * H, seed=1, scale=(F * H) ** -0.5),
        hidden_norm=_rand_bf16(H, seed=2, scale=0.1) + 1,
        in_norm=_rand_bf16(H, seed=3, scale=0.1) + 1,
        conv_base=_rand_bf16(2, cfg.conv_taps, H, seed=4, scale=0.1) + 0.5,
        proj=_rand_bf16(2 * cfg.conv_taps * cfg.conv_groups, H, seed=5, scale=H**-0.5),
    )


@pytest.mark.parametrize("rows", [8, 16])
def test_reference_is_the_dflash_reference(rows):
    cfg = small_cfg()
    w = small_weights(cfg)
    cap = [
        torch.randn(rows, cfg.hidden, generator=torch.Generator().manual_seed(10 + k))
        for k in range(5)
    ]
    emb = _rand_bf16(rows, cfg.hidden, seed=20)
    out = R.entry_reference(cfg, w, cap, emb)
    # layer 0's conv prepare (draft_layer, residual None)
    h = R.rms_norm(emb, w.in_norm, cfg.rms_eps, torch.bfloat16)
    xq, c1 = R.conv_prepare(R.ConvWeights(w.conv_base, w.proj), h, torch.bfloat16)
    assert torch.equal(out["h"], h) and torch.equal(out["xq"], xq)
    assert torch.equal(out["c_att1"], c1.reshape(rows, -1))
    # context_update's K / V come from this ctx
    feats = torch.cat([c.to(torch.bfloat16) for c in cap], 1)
    dw = R.DraftWeights(
        layers=[
            R.LayerWeights(
                input_norm=w.in_norm,
                post_norm=w.in_norm,
                q_norm=_rand_bf16(cfg.head_dim, seed=30) + 1,
                k_norm=_rand_bf16(cfg.head_dim, seed=31) + 1,
                qkv=_rand_bf16(
                    (cfg.q_heads + 2 * cfg.kv_heads) * cfg.head_dim,
                    cfg.hidden,
                    seed=32,
                    scale=0.05,
                ),
                o=torch.zeros(
                    cfg.hidden, cfg.q_heads * cfg.head_dim, dtype=torch.bfloat16
                ),
                gate_up=torch.zeros(1, 1),
                down=torch.zeros(1, 1),
                attn_conv=None,
                mlp_conv=None,
            )
        ],
        fc=w.fc,
        hidden_norm=w.hidden_norm,
        norm=w.in_norm,
        pred_codebook=None,
        succ_codebook=None,
        hidden_proj=None,
    )
    ((k, v),) = R.context_update(cfg, dw, feats, 100, R.SoloComm(), torch.bfloat16)
    lw, d = dw.layers[0], cfg.head_dim
    kv = R._mm(out["ctx_n"], lw.qkv[cfg.q_heads * d :], torch.bfloat16).view(
        rows, 2, cfg.kv_heads, d
    )
    assert torch.equal(kv[:, 1], v)
    cos, sin = R.rope_cos_sin(100 + torch.arange(rows), d, cfg.rope_theta)
    assert torch.equal(
        R.head_norm_rope(
            kv[:, 0], lw.k_norm, cfg.rms_eps, cos, sin, torch.bfloat16, False
        ),
        k,
    )


def test_emulate_prep_is_grouped_conv():
    cfg = small_cfg()
    w = small_weights(cfg)
    h = _rand_bf16(4, cfg.hidden, seed=40)
    delta = _rand_bf16(4, 2, cfg.conv_taps, cfg.conv_groups, seed=41, scale=0.3)
    want = R.grouped_conv(h, delta[:, 0], w.conv_base[0], torch.bfloat16)[:2]
    got = R.entry_emulate_prep(
        h[0],
        h[1],
        delta[0].reshape(-1),
        delta[1].reshape(-1),
        w.conv_base[0],
        cfg.conv_group,
    )
    assert torch.equal(got, want)


# ---- the images ----


@cache
def paths(world: str = "tp8") -> dict[str, Path]:
    return artifacts.build_set(artifacts.dflash_artifacts(world))


def resources(cubin: Path) -> dict[str, int]:
    text = subprocess.run(
        [CUOBJDUMP, "-res-usage", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    line = re.search(r"Function " + re.escape(ENTRY_PREFIX) + r"\S*:\n\s*(.*)", text)[1]
    return {k: int(v) for k, v in re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line)}


@pytest.mark.parametrize("rows", [8, 16])
def test_entry_image_resources(rows):
    import test_param_blocks as T

    p = paths()
    res = resources(p[f"dflash_entry_b{rows}"])
    assert res["LOCAL"] == 0, res  # no spills
    assert res["STACK"] <= 32, res  # the timed-out wait's report (its stack frame) only
    lib = host.Library(p[f"libmk_dflash_entry_b{rows}"])
    c = head.entry_constants(lib)
    assert (c.rows, c.grid, c.block, c.hidden, c.features, c.fc_n, c.conv_proj_n) == (
        rows,
        144,
        256,
        8192,
        5,
        1024,
        256,
    )
    assert c.smem + res["SHARED"] - T.SMEM_RESERVE <= OPTIN_SMEM, (c.smem, res)
    fakes = T.Fakes()
    values = {
        name: (
            [fakes() for _ in range(ftype._length_)]
            if hasattr(ftype, "_length_")
            else fakes()
        )
        for name, ftype in host.STRUCTS["MkDflashEntryBind"]._fields_
    }
    values.update(rank=3, shard_rows=25256, eps=1e-5)
    launch = lib.bind(
        lib.dll.mk_dflash_entry_bind, None, host.make("MkDflashEntryBind", **values)
    )
    info = launch.info()
    assert (info.grid, info.block, info.cluster, info.smem) == (144, 256, 1, c.smem)
    blob, slots = launch.params()
    size, kslots = T.params_of(p[f"dflash_entry_b{rows}"], ENTRY_PREFIX)
    assert len(blob) == size and slots == kslots, (len(blob), size, slots, kslots)
    words = {
        int.from_bytes(blob[i : i + 8], "little") for i in range(0, len(blob) - 7, 8)
    }
    for name, v in values.items():
        if name in ("rank", "shard_rows", "eps"):
            continue
        for ptr in v if isinstance(v, list) else [v]:
            assert ptr in words, f"{name} = {ptr:#x} never reaches the block"


# the cubin / parameter-block / tensor-map readers and the encoder shim

BF16, F32 = torch.bfloat16, torch.float32
CFG, RANKS = R.DraftConfig(), 8
H = CFG.hidden  # K of both launches
INTER = CFG.inter // RANKS  # the rank's intermediate
HD = CFG.head_dim
Q_HEADS, _ = layer.qkv_heads(CFG, RANKS)
QKV_N = layer.qkv_cols(CFG, RANKS)
RING = 4096  # the draft KV ring (kernels/dflash/layer.cu kRing)
EPS = CFG.rms_eps
QKV_PTRS = (
    "w",
    "x",
    "out",
    "tickets",
    "cos_sin",
    "slot",
    "commit",
    "q_norm",
    "k_norm",
    "k_ring",
    "v_ring",
)


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 ulps (same-sign finite values: the order of the int16 patterns)."""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    ia = torch.where(ia < 0, -32768 - ia, ia)
    ib = torch.where(ib < 0, -32768 - ib, ib)
    return (ia - ib).abs()


# ---- packers ----


@pytest.mark.parametrize(
    "inter,h",
    [(32, 64), (96, 128), (INTER, 64)],
    ids=["one_block", "three_blocks", "rank_inter"],
)
def test_pack_gate_up_layout(inter: int, h: int):
    g = torch.Generator().manual_seed(inter)
    gate_up = torch.randn(2 * inter, h, generator=g).to(BF16)
    w = dweights.pack_gate_up(gate_up)
    gate, up = gate_up.chunk(2)
    for c in [0, 7, 8, 31, inter - 1, inter // 2 + 5]:
        b, grp, lane = c // 32, (c % 32) // 8, c % 8
        assert torch.equal(w[64 * b + 16 * grp + lane], gate[c]), c
        assert torch.equal(w[64 * b + 16 * grp + 8 + lane], up[c]), c
    assert torch.equal(dweights.unpack_gate_up(w), gate_up)


def test_pack_qkv_is_the_reference_shard_order():
    cfg = R.DraftConfig(hidden=256)
    g = torch.Generator().manual_seed(1)
    d = cfg.head_dim
    qkv = torch.randn((cfg.q_heads + 2 * cfg.kv_heads) * d, cfg.hidden, generator=g).to(
        BF16
    )
    lw = R.LayerWeights(
        *(torch.ones(1) for _ in range(4)),
        qkv=qkv,
        o=torch.zeros(cfg.hidden, cfg.q_heads * d),
        gate_up=torch.zeros(16, 1),
        down=torch.zeros(1, 8),
        attn_conv=None,
        mlp_conv=None,
    )
    qr = cfg.q_heads // RANKS
    for rank in (0, 5):
        sh = R.shard_layer(cfg, lw, rank, RANKS).qkv
        q, k, v = dweights.unpack_qkv(dweights.pack_qkv(sh, cfg, RANKS), cfg, RANKS)
        fq, fk, fv = qkv.split([cfg.q_heads * d, cfg.kv_heads * d, cfg.kv_heads * d])
        assert torch.equal(q, fq[rank * qr * d : (rank + 1) * qr * d])
        assert torch.equal(k, fk[rank * d : (rank + 1) * d]) and torch.equal(
            v, fv[rank * d : (rank + 1) * d]
        )


# ---- op-order references ----


def test_butterfly_is_the_lane_shuffle():
    """The torch emulation against a literal 32-lane xor butterfly, every lane's result."""
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(64, 32, generator=g) * torch.logspace(-3, 3, 32)).to(F32)
    lanes = x.clone()
    for off in (16, 8, 4, 2, 1):
        lanes = lanes + lanes[:, torch.arange(32) ^ off]
    want = R.butterfly_sum32(x)
    assert all(torch.equal(lanes[:, i], want) for i in range(32))


def _rows(g: torch.Generator, n: int, heads: int, exact: bool) -> torch.Tensor:
    # exact: multiples of 1/8 in [-1, 1], so every square and partial sum is exact in fp32 and fp64
    if exact:
        return (torch.randint(-8, 9, (n, heads, HD), generator=g).float() / 8).to(BF16)
    return (torch.randn(n, heads, HD, generator=g) * 3).to(BF16)


@pytest.mark.parametrize("round_normed", [True, False], ids=["block", "context"])
@pytest.mark.parametrize("pos", [0, 4096, 128000], ids=["p0", "short", "long"])
def test_head_norm_rope_tree_vs_reference(round_normed: bool, pos: int):
    g = torch.Generator().manual_seed(pos + round_normed)
    w = (1 + 0.1 * torch.randn(HD, generator=g)).to(BF16)
    cos, sin = R.rope_cos_sin(pos + torch.arange(16), HD, CFG.rope_theta)
    # an exact sum of squares: the two are the same sequence of fp32 roundings
    x = _rows(g, 16, 8, exact=True)
    assert torch.equal(
        R.head_norm_rope_tree(x, w, EPS, cos, sin, round_normed),
        R.head_norm_rope(x, w, EPS, cos, sin, BF16, round_normed),
    )
    # general rows: only the sum-of-squares order differs (fp32 butterfly vs the fp64 mean): one bf16 ulp; the context
    # rows' unrounded norm feeds RoPE's subtraction, whose cancellation can double it
    x = _rows(g, 16, 8, exact=False)
    ulp = _bf16_ulp_diff(
        R.head_norm_rope_tree(x, w, EPS, cos, sin, round_normed),
        R.head_norm_rope(x, w, EPS, cos, sin, BF16, round_normed),
    )
    assert (
        ulp.max() <= (1 if round_normed else 2) and (ulp == 0).float().mean() > 0.97
    ), (
        ulp.max(),
        (ulp == 0).float().mean(),
    )


@pytest.mark.parametrize("block", [8, 16])
def test_qkv_rowops_ref_row_types(block: int):
    g = torch.Generator().manual_seed(block)
    raw = (torch.randn(2 * block, QKV_N, generator=g) * 2).to(BF16)
    qn, kn = [(1 + 0.1 * torch.randn(HD, generator=g)).to(BF16) for _ in range(2)]
    _, cos_sin, _ = R.qkv_feeds(4096, block, 3, CFG, RING)
    q, k, v = R.qkv_rowops_ref(raw, qn, kn, EPS, cos_sin, block, Q_HEADS)
    cos, sin = cos_sin[:, : HD // 2], cos_sin[:, HD // 2 :]
    rq, rk, rv = raw.split([Q_HEADS * HD, HD, HD], 1)
    tree = R.head_norm_rope_tree
    assert torch.equal(
        q,
        tree(
            rq[:block].view(block, Q_HEADS, -1), qn, EPS, cos[:block], sin[:block], True
        ),
    )
    assert torch.equal(
        k[:block],
        tree(rk[:block].view(block, 1, -1), kn, EPS, cos[:block], sin[:block], True)[
            :, 0
        ],
    )
    ctx = tree(rk[block:].view(block, 1, -1), kn, EPS, cos[block:], sin[block:], False)[
        :, 0
    ]
    assert torch.equal(k[block:], ctx)
    assert torch.equal(v, rv)
    # the two row types differ (the block's pre-RoPE rounding is live)
    assert not torch.equal(
        ctx,
        tree(rk[block:].view(block, 1, -1), kn, EPS, cos[block:], sin[block:], True)[
            :, 0
        ],
    )


@pytest.mark.parametrize("block", [8, 16])
@pytest.mark.parametrize("pos", [4096, 4093, 128000], ids=["short", "wrap", "long"])
def test_feeds(block: int, pos: int):
    for commit in sorted({1, 3, block}):
        positions, cos_sin, slot = R.qkv_feeds(pos, block, commit, CFG, RING)
        assert positions[:block].tolist() == list(range(pos, pos + block))
        assert positions[block : block + commit].tolist() == list(
            range(pos - commit, pos)
        )
        assert torch.equal(slot.long(), positions % RING)
        cos, sin = R.rope_cos_sin(positions, HD, CFG.rope_theta)
        assert torch.equal(cos_sin, torch.cat([cos, sin], 1))
        # the rows past the commit collide with the block's slots: the kernel's commit mask is what keeps the block's
        # rows
        assert set(slot[block + commit :].tolist()) <= set(slot[:block].tolist())


def test_swiglu_ref_is_reference():
    g = torch.Generator().manual_seed(3)
    gu = (torch.randn(16, 2 * 64, generator=g) * 4).to(BF16)
    assert torch.equal(R.swiglu_ref(gu), R.silu_mul(gu[:, :64], gu[:, 64:], BF16))


# ---- images: build, census, parameter blocks ----


def _res_usage(cubin: Path) -> dict[str, int]:
    text = subprocess.run(
        [T.CUOBJDUMP, "-res-usage", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    line = next(ln for ln in text.splitlines() if "REG:" in ln)
    return {
        kv.split(":")[0]: int(kv.split(":")[1])
        for kv in line.split()
        if ":" in kv and kv.split(":")[1].isdigit()
    }


@pytest.mark.parametrize("image", ["dflash_qkv_b8", "dflash_qkv_b16"])
def test_image_census(image: str):
    r = _res_usage(paths()[image])
    assert r["STACK"] == 0 and r["LOCAL"] == 0, (image, r)
    assert r["REG"] <= 255, (image, r)


@pytest.mark.parametrize("block", [8, 16])
def test_qkv_block(block: int):
    s = T.shim()
    d = layer.Dense(paths())
    fakes = T.Fakes()
    v = {n: fakes() for n in QKV_PTRS}
    s.shim_reset()
    launch = d.qkv(block, pack_only=True, hidden=H, eps=EPS, **v)
    image = f"dflash_qkv_b{block}"
    prefix = "_ZN2mk6dflash9draft_qkvI"
    blob, slots, enc = T.check_block(launch, paths()[image], prefix, v, list(QKV_PTRS))
    m, n = 2 * block, QKV_N
    info = launch.info()
    assert (info.grid, info.cluster, info.block) == (80, 4, 256), image
    assert info.smem + T.static_smem(paths()[image], prefix) <= T.OPTIN_SMEM, (
        image,
        info.smem,
    )
    maps = [
        (1, v["w"], T.BF16, (H, n), (64, 64), H * 2, T.SW128),
        (2, v["x"], T.BF16, (H, m), (64, m), H * 2, T.SW128),
        (3, v["out"], T.BF16, (n, m), (64, m // 4), n * 2, T.SWNONE),
    ]
    for param, base, dtype, dims, box, stride, swz in maps:
        T.expect_map(
            T.map_at(blob, slots, enc, param), base, dtype, dims, box, (stride,), swz
        )
    assert len(enc) == len(maps), (image, len(enc))


@pytest.mark.parametrize(
    "bad",
    [dict(block=12), dict(hidden=8192 + 64), dict(w=0)],
    ids=["block", "k_quarters", "null_w"],
)
def test_qkv_bind_refuses(bad: dict):
    T.shim()
    d = layer.Dense(paths())
    fakes = T.Fakes()
    args = dict(block=8, hidden=H, eps=EPS, **{n: fakes() for n in QKV_PTRS})
    args.update(bad)
    with pytest.raises(RuntimeError):
        d.lib.bind(
            d.lib.dll.mk_dflash_qkv_bind, None, host.make("MkDflashQkvBind", **args)
        )


# the tensor-map shim and the block checks

T.shim()  # ahead of the host library, so its tensor-map encodes bind to the shim

CUT, BAND = 0.99999, 2.0**-6
# positions P of the block: the clamped low edge (< 2047, rows straddling it at 2030 / 2040), the first unclamped ones,
# a bar context, the ring's wrap (4085 + B crosses 4096; 4196 / 6000 walk across slot 0)
POSITIONS = [0, 1, 7, 100, 2000, 2030, 2040, 2047, 2048, 2111, 4096, 4085, 4196, 6000]
BLOCKS = artifacts.DFLASH_BLOCKS
WORLDS = artifacts.WORLDS


@cache
def C(block: int, world: str = "tp8"):
    return layer.attn_constants(host.Library(paths(world)["libmk_dflash_attn"]), block)


@pytest.mark.parametrize("world", WORLDS)
def test_constants(world):
    ranks = int(world[2:])
    for b in BLOCKS:
        c = C(b, world)
        assert c.window == R.DraftConfig().window_left
        assert (c.q_heads, c.kv_heads) == (64 // ranks, 8 // ranks)
        # the key range [lo_w, lo_w + 64 blocks) spans at most B + window + 2 x 63 keys: inside the ring, one block per
        # CTA of each kv head's 36
        group = c.attn_ctas // c.kv_heads
        assert c.attn_ctas == 36 * c.kv_heads
        assert b + c.window + 2 * (c.block_keys - 1) <= c.ring_rows
        assert -(-(b + c.window + c.block_keys - 1) // c.block_keys) <= group
        # the combine: two rows per CTA, or four where two would take more than 16 clusters
        rows = 8 * b * c.kv_heads
        assert c.combine_ctas == (rows // 2 if rows <= 128 else 64)


@pytest.mark.parametrize("block", BLOCKS)
def test_window_cut(block):
    """[lo_w, lo_w + 64 blocks) holds every row's visible keys, starts at a visible key's block and ends at len's."""
    c = C(block)
    for pos in sorted(set(range(0, 9000, 37)) | set(POSITIONS) | {128000}):
        length = pos + block
        lo_w, blocks = R.attn_key_window(length, block, c)
        # row 0's first visible key: the lowest of any row
        first = max(0, pos - c.window)
        assert lo_w % c.block_keys == 0 and lo_w <= first < lo_w + c.block_keys, pos
        assert (
            lo_w + c.block_keys * (blocks - 1) < length <= lo_w + c.block_keys * blocks
        ), pos
        assert blocks <= c.attn_ctas, pos


@pytest.mark.parametrize("pos", POSITIONS + [128000])
def test_ring_holds_the_window(pos):
    c = C(16)
    length = pos + 16
    rows = torch.arange(length, dtype=torch.float32)[:, None].expand(length, 2)
    ring = R.attn_to_ring(rows, length, c.ring_rows)
    lo_w, blocks = R.attn_key_window(length, 16, c)
    walked = torch.arange(lo_w, min(length, lo_w + c.block_keys * blocks))
    # every walked position, at its slot, newest
    assert torch.equal(ring[walked % c.ring_rows, 0], walked.float())


def inputs(block: int, pos: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = pos + block
    q = torch.randn(block, 8, 128, generator=g).bfloat16()
    return (
        q,
        torch.randn(n, 128, generator=g).bfloat16(),
        torch.randn(n, 128, generator=g).bfloat16(),
    )


def band(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    a, b = x.double().flatten(), ref.double().flatten()
    return (
        float(a @ b / (a.norm() * b.norm())),
        float((a - b).abs().max()),
        float(b.abs().max()),
    )


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("pos", POSITIONS + [128000])
def test_emulation_band(block, pos):
    q, k, v = inputs(block, pos, seed=pos + block)
    c = C(block)
    x, ml = R.attn_emulate(
        q,
        R.attn_to_ring(k, pos + block, c.ring_rows),
        R.attn_to_ring(v, pos + block, c.ring_rows),
        pos + block,
        c,
    )
    assert ml.shape == (R.attn_key_window(pos + block, block, c)[1], 8 * block, 2)
    cos, mx, ref_max = band(x, R.attn_reference(q, k, v, pos, R.DraftConfig()))
    assert cos >= CUT and mx <= BAND * ref_max, (cos, mx, ref_max)


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("pos", [2040, 4096, 4196])
def test_emulation_edges_and_poison(block, pos):
    """Beacon keys on both sides of every row's low edge (the window's per-row cut decides the output); masked keys of
    the walked blocks at +-1e4 and every other slot NaN move nothing."""
    q, k, v = inputs(block, pos, seed=3 * pos + block)
    c = C(block)
    length = pos + block
    u = torch.nn.functional.normalize(
        torch.randn(128, generator=torch.Generator().manual_seed(pos)), dim=0
    )
    q = (q.float() + 8 * u).bfloat16()
    idx = torch.tensor(
        sorted(
            {
                p
                for t in range(block)
                for p in (pos + t - c.window - 1, pos + t - c.window)
                if p >= 0
            }
        )
    )
    k[idx] = (8 * u).bfloat16()
    v[idx] = (4 * torch.randn(len(idx), 128)).bfloat16()
    k_ring, v_ring = R.attn_to_ring(k, length, c.ring_rows), R.attn_to_ring(
        v, length, c.ring_rows
    )
    x, _ = R.attn_emulate(q, k_ring, v_ring, length, c)
    cos, mx, ref_max = band(x, R.attn_reference(q, k, v, pos, R.DraftConfig()))
    assert cos >= CUT and mx <= BAND * ref_max, (cos, mx, ref_max)
    # the negative control (where no row's low edge clamps at 0): a window one key wider or narrower leaves the band
    for wl in (c.window - 1, c.window + 1) if pos > c.window + 1 else ():
        lo = max(0, pos - wl)
        shifted = R.attention(
            q,
            k[lo:length, None],
            v[lo:length, None],
            pos + torch.arange(block),
            torch.arange(lo, length),
            wl,
        )
        cos, mx, ref_max = band(x, shifted)
        assert not (cos >= CUT and mx <= BAND * ref_max), (wl, cos, mx, ref_max)
    lo_w, blocks = R.attn_key_window(length, block, c)
    walked = torch.arange(lo_w, lo_w + c.block_keys * blocks)
    masked = walked[(walked < max(0, pos - c.window)) | (walked >= length)]
    outside = torch.ones(c.ring_rows, dtype=torch.bool)
    outside[walked % c.ring_rows] = False
    for ring in (k_ring, v_ring):
        ring[outside] = float("nan")
        ring[masked % c.ring_rows] = 1e4
    assert torch.equal(R.attn_emulate(q, k_ring, v_ring, length, c)[0], x)


# the shared CPU-check helpers: the tensor-map shim, cuobjdump readers

G, TAPS = 512, 2  # H: CFG.hidden (8192)
SMEM_CEIL = 231428  # kernels/framework.cuh cfg::kSmemCeil
IMAGES = artifacts.DFLASH_LAYER_IMAGES


def dims(world: str, b: int = 8) -> dict[str, int]:
    """The draft's per-rank widths at a world (kernels/dflash/common.cuh) and the attention / combine geometry
    (kernels/dflash/layer.cu Shape) at block b."""
    r = int(world[2:])
    q, kv = 64 // r, 8 // r
    rows = 8 * b * kv  # the combine's rows: every kv-head group's live rows
    combine = rows // (rows // 64 if rows > 128 else 2)
    return dict(
        ranks=r,
        o_k=q * 128,
        kv=kv,
        inter=28672 // r,
        qkv=(q + 2 * kv) * 128,
        attn_ctas=36 * kv,
        combine_ctas=combine,
        combine_entries=combine // 4,
    )


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _bf16(g, *shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(BF16)


def _conv(g, scale=0.02):
    base = _bf16(g, 2, TAPS, H, scale=0.05)
    base[:, 0] += 1  # tap 0 near identity, as the checkpoint's
    return R.ConvWeights(base, _bf16(g, 2 * TAPS * G, H, scale=scale))


# ---- layouts ----


def test_coeff_order_is_the_packet_layout():
    order = dweights.coeff_order(TAPS, G)
    assert sorted(order.tolist()) == list(range(TAPS * G))
    for j in (0, 1, 77, 127):
        packet = order[8 * j : 8 * j + 8].tolist()
        assert packet == [t * G + 4 * j + q for t in range(TAPS) for q in range(4)]


@pytest.mark.parametrize("b", [8, 16])
def test_pack_unpack_round_trip(b):
    d = _bf16(_gen(b), b, TAPS, G)
    assert torch.equal(dweights.unpack_coeff(dweights.pack_coeff(d), TAPS), d)


def test_conv_proj_shards_tile_the_packed_rows():
    g = _gen(1)
    proj = _bf16(g, 2 * TAPS * G, 64)
    packed = dweights.pack_conv_proj_packets(proj, TAPS)
    side = TAPS * G
    shards = [dweights.conv_proj_shard(packed, r) for r in range(8)]
    assert torch.equal(shards[1], shards[5])  # the two nodes compute the same rows
    for s in range(2):
        rows = torch.cat([sh[s * side // 4 : (s + 1) * side // 4] for sh in shards[:4]])
        assert torch.equal(rows, packed[s * side : (s + 1) * side])


def test_packed_conv_proj_is_the_permuted_projection():
    g = _gen(2)
    proj, h = _bf16(g, 2 * TAPS * G, H, scale=0.02), _bf16(g, 4, H)
    delta = R._mm(h, proj, BF16).view(4, 2, TAPS, G)
    out = R._mm(h, dweights.pack_conv_proj_packets(proj, TAPS), BF16).view(
        4, 2, TAPS * G
    )
    for s in range(2):
        assert torch.equal(dweights.unpack_coeff(out[:, s], TAPS), delta[:, s])


def test_side1_packet_order_is_the_finish_packet():
    g = _gen(3)
    proj = _bf16(g, 2 * TAPS * G, 64)
    q = dweights.side1_packet_order(proj, TAPS)
    # side 0 keeps the entry's prepare order
    assert torch.equal(q[: TAPS * G], proj[: TAPS * G])
    assert torch.equal(
        q[TAPS * G :], dweights.pack_conv_proj_packets(proj, TAPS)[TAPS * G :]
    )


# ---- the kernels' indexing, transcribed per thread ----


def _finish_as_the_kernel(partial, packed, base1, b):
    """dflash/layer.cu finish_tile for every tile and thread: channel ch = 64 tile + 16 quad + lane reads packet `tile` of
    each row, element quad (tap 0) and 4 + quad (tap 1)."""
    out = torch.empty(b, H, dtype=BF16)
    for tile in range(H // 64):
        pk = packed[:, 8 * tile : 8 * tile + 8].float()
        for quad in range(4):
            ch = slice(64 * tile + 16 * quad, 64 * tile + 16 * quad + 16)
            b0, b1 = base1[0, ch].float(), base1[1, ch].float()
            for n in range(b):
                o = (b0 + pk[n, quad]) * partial[n, ch]
                if n > 0:
                    o = o + (b1 + pk[n, 4 + quad]) * partial[n - 1, ch]
                out[n, ch] = o.to(BF16)
    return out


@pytest.mark.parametrize("b", [8, 16])
def test_finish_tile_indexing_is_the_reference(b):
    """The conv finish on one rank's fp32 partial, rounded once (the reference's 'pre_ar' sublayer order)."""
    g = _gen(10 + b)
    conv = _conv(g)
    partial = torch.randn(b, H, generator=g)
    delta1 = _bf16(g, b, TAPS, G, scale=0.05)
    want = R.grouped_conv(partial, delta1, conv.base[1], BF16)
    assert torch.equal(
        _finish_as_the_kernel(partial, dweights.pack_coeff(delta1), conv.base[1], b),
        want,
    )


def _prep_as_the_kernel(h, packed, base0, b):
    """dflash/layer.cu prep_row for every row and item-warp thread t: channels 32 t ..; the even thread's packet t / 2,
    word (x, z) for even t and (y, w) for odd t (groups 2 t, 2 t + 1 at tap 0 / tap 1).
    """
    out = torch.empty(b, H, dtype=BF16)
    for n in range(b):
        for t in range(256):
            pk = packed[n, 8 * (t // 2) : 8 * (t // 2) + 8].float()
            hi = t & 1
            d0, d1 = pk[2 * hi : 2 * hi + 2], pk[4 + 2 * hi : 6 + 2 * hi]
            ch = slice(32 * t, 32 * t + 32)
            c0 = torch.cat([d0[0].expand(16), d0[1].expand(16)])
            c1 = torch.cat([d1[0].expand(16), d1[1].expand(16)])
            o = (base0[0, ch].float() + c0) * h[n, ch].float()
            if n > 0:
                o = o + (base0[1, ch].float() + c1) * h[n - 1, ch].float()
            out[n, ch] = o.to(BF16)
    return out


@pytest.mark.parametrize("b", [8, 16])
def test_prep_row_indexing_is_the_reference(b):
    g = _gen(20 + b)
    conv = _conv(g)
    h = _bf16(g, b, H)
    delta0 = _bf16(g, b, TAPS, G, scale=0.05)
    want = R.grouped_conv(h, delta0, conv.base[0], BF16)
    assert torch.equal(
        _prep_as_the_kernel(h, dweights.pack_coeff(delta0), conv.base[0], b), want
    )
    # row 0 reads its delta
    assert not torch.equal(want[0], (conv.base[0][0].float() * h[0].float()).to(BF16))


def _prep_tile_as_the_kernel(h, packed, base0, b):
    """dflash/layer.cu prep_tile for every W block blk of the full side 0 and epilogue thread (quad, lane j) of every CTA q:
    row 4 q + quad, group 32 blk + j, tap 0 at tile column 8 (j / 4) + j % 4 and tap 1 four columns on; the block's
    tile column c is packed coefficient 64 blk + c."""
    out = torch.empty(b, H, dtype=BF16)
    for blk in range(16):
        tile = packed[:, 64 * blk : 64 * blk + 64].float()
        for n in range(b):
            for j in range(32):
                ch = slice((32 * blk + j) * 16, (32 * blk + j) * 16 + 16)
                d0, d1 = (
                    tile[n, 8 * (j // 4) + j % 4],
                    tile[n, 8 * (j // 4) + j % 4 + 4],
                )
                o = (base0[0, ch].float() + d0) * h[n, ch].float()
                if n > 0:
                    o = o + (base0[1, ch].float() + d1) * h[n - 1, ch].float()
                out[n, ch] = o.to(BF16)
    return out


@pytest.mark.parametrize("b", [8, 16])
def test_prep_tile_is_the_prep_row_and_the_reference(b):
    """A prepare claim's epilogue computes prep_row's values (those of the standalone prepare row unit)."""
    g = _gen(30 + b)
    conv = _conv(g)
    h = _bf16(g, b, H)
    delta0 = _bf16(g, b, TAPS, G, scale=0.05)
    packed = dweights.pack_coeff(delta0)
    got = _prep_tile_as_the_kernel(h, packed, conv.base[0], b)
    assert torch.equal(got, R.grouped_conv(h, delta0, conv.base[0], BF16))
    assert torch.equal(got, _prep_as_the_kernel(h, packed, conv.base[0], b))


def _tree_as_the_kernel(s_row):
    """dflash/layer.cu ar_norm_row's sum of squares for one row, thread by thread: (vw, lane) sums its 8 squares left
    to right; the xor butterfly over the lanes; the vw partials staged in vw order and butterflied again.
    """
    s = s_row.view(32, 32, 8)
    acc = torch.zeros(32, 32, dtype=F32)
    for e in range(8):
        acc = acc + s[:, :, e] * s[:, :, e]
    for vw in range(32):
        lanes = acc[vw].clone()
        for off in (16, 8, 4, 2, 1):
            lanes = torch.stack([lanes[i] + lanes[i ^ off] for i in range(32)])
        acc[vw] = lanes
    part = acc[:, 0].clone()
    for off in (16, 8, 4, 2, 1):
        part = torch.stack([part[i] + part[i ^ off] for i in range(32)])
    return part[0]


def test_norm_tree_is_the_kernel_tree():
    g = _gen(3)
    s = torch.randn(3, H, generator=g)
    got = R.rms_tree_sum(s)
    for n in range(3):
        assert torch.equal(got[n], _tree_as_the_kernel(s[n]))
    exact = s.double().square().sum(-1)
    assert ((got.double() - exact).abs() / exact).max() < 1e-6


# ---- reordered arithmetic ----


@pytest.mark.parametrize("b", [8, 16])
def test_add_rms_norm_tree_matches_the_plain_norm_within_an_ulp(b):
    g = _gen(30 + b)
    x, r, w = _bf16(g, b, H), _bf16(g, b, H), _bf16(g, H, scale=0.1) + 1
    h_tree, r_tree = R.add_rms_norm_tree(x, r, w, 1e-5)
    h_ref, r_ref = R.add_rms_norm(x, r, w, 1e-5, BF16)
    assert torch.equal(r_tree, r_ref)
    ulp = (h_tree.view(torch.int16).int() - h_ref.view(torch.int16).int()).abs()
    assert ulp.max() <= 1


def test_finish_is_linear_so_pre_ar_is_the_same_math():
    """With values fp32 adds and multiplies hold exactly, finishing each partial then summing equals summing then
    finishing: the two sublayer orders differ only by rounding."""
    g = _gen(4)
    b = 8
    # coefficients in halves up to 2, partials in {-1, 0, 1}: every intermediate fits bf16's 8 mantissa bits
    base1 = (torch.randint(-2, 3, (2, H), generator=g) / 2).to(BF16)
    delta1 = (torch.randint(-2, 3, (b, TAPS, G), generator=g) / 2).to(BF16)
    partials = [torch.randint(-1, 2, (b, H), generator=g).float() for _ in range(8)]
    pre = sum(R.grouped_conv(p, delta1, base1, BF16).double() for p in partials)
    post = R.grouped_conv(sum(partials), delta1, base1, BF16).double()
    assert torch.equal(pre, post)


# ---- builds ----


LAYER_PREFIX = "_ZN2mk6dflash11draft_layerI"


def image(name: str, b: int, world: str = "tp8") -> layer.Image:
    return layer.Image(paths(world), name, b, module=False)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("name", IMAGES)
@pytest.mark.parametrize("b", BLOCKS)
def test_layer_image_resources(name, b, world):
    T.shim()
    img = image(name, b, world)
    cubin = paths(world)[f"dflash_layer_{name}_b{b}"]
    use = _res_usage(cubin)
    # no spill; the stack is the timed-out wait's report (its stack frame)
    assert use["STACK"] <= 32 and use["LOCAL"] == 0, (name, b, use)
    assert use["REG"] <= 168, (name, b, use)  # 384 threads, one CTA per SM
    c = img.c
    assert c.smem + T.static_smem(cubin, LAYER_PREFIX) <= SMEM_CEIL, (name, b, c.smem)
    assert (c.grid, c.cluster, c.block, c.rows) == (144, 4, 384, b)
    assert (c.ranks, c.inter) == (dims(world)["ranks"], dims(world)["inter"])


def bind_fields(c, fakes, world: str = "tp8") -> dict:
    """A complete binding of an image (its MkDflashLayerConstants c): every slot the program reads, fake pointers, the
    world's widths."""
    d = dims(world)
    v = layer.empty_bind()
    for g in range(len(c.gemm_m)):
        if not (c.gemm_mask >> g) & 1:
            continue
        k = d["o_k"] if g == 0 else d["inter"] if g == 4 else 8192
        # conv coefficients: the rank's 512-row shard, side 1 past side 0; a prepare slot: the side's full 1024 rows
        rows = {0: 8192, 3: 2 * d["inter"], 4: 8192, 6: d["qkv"], 8: d["qkv"]}.get(
            g,
            (
                64 * c.gemm_blocks[g]
                if c.gemm_blocks[g] == 16
                else 2 * 64 * c.gemm_blocks[g]
            ),
        )
        v["gemm_w"][g], v["gemm_x"][g] = fakes(), fakes()
        v["gemm_w_rows"][g], v["gemm_x_rows"][g], v["gemm_k"][g] = rows, 32, k
        v["gemm_block0"][g] = c.gemm_blocks[g] if g in (2, 7) else 0
        v["gemm_gate"][g], v["gemm_gate_arrivals"][g] = fakes(), c.row_arrivals
    for slot in range(2):
        if (c.fin_mask >> slot) & 1:
            for f in ("fin_bases", "fin_coef", "fin_base1"):
                v[f][slot] = fakes()
        if (c.ar_mask >> slot) & 1:
            for f in (
                "ar_bases",
                "ar_r_in",
                "ar_r_out",
                "ar_w",
                "ar_h",
                "ar_ready",
                "ar_rin_gate",
            ):
                v[f][slot] = fakes()
            v["ar_eps"][slot], v["ar_rin_arrivals"][slot] = 1e-5, c.row_arrivals
        if (c.prep_mask >> slot) & 1:
            # a prepare unit's epilogue prepare reads neither coefficient rows nor an h gate of its own
            prep = c.has_attn and slot in (0, 1)
            for f in ("prep_h", "prep_base0", "prep_x", "prep_ready") + (
                () if prep else ("prep_coef", "prep_h_gate")
            ):
                v[f][slot] = fakes()
            v["prep_h_arrivals"][slot] = 0 if prep else c.row_arrivals
    for slot in range(4):
        if (c.gat_mask >> slot) & 1:
            v["gat_dst"][4 * slot : 4 * slot + 4] = [fakes() for _ in range(4)]
            v["gat_block0"][slot] = 0
    if c.swiglu:
        v["act"], v["act_ready"] = fakes(), fakes()
    if c.qkv:
        for f in (
            "qkv_out",
            "qkv_tickets",
            "qkv_cos_sin",
            "qkv_slot",
            "qkv_commit",
            "qkv_q_norm",
            "qkv_k_norm",
            "qkv_k_ring",
            "qkv_v_ring",
        ):
            v[f] = fakes()
        v["qkv_eps"] = 1e-5
    if c.has_attn:
        for f in (
            "attn_q",
            "attn_k_ring",
            "attn_v_ring",
            "draft_kv_len",
            "attn_partials",
            "attn_word",
        ):
            v[f] = fakes()
        v["attn_q_stride"] = 2 * d["qkv"]
        v["attn_x_ready"] = v["gemm_gate"][0]
        v["gemm_gate_arrivals"][0] = c.combine_ctas
    v["launch_ctr"], v["rank"], v["phase_offset"], v["diag"] = (
        fakes(),
        d["ranks"] - 3,
        3,
        fakes(),
    )
    if (
        c.has_attn and not c.qkv
    ):  # the last layer: its final norm writes the draft head's xq rows
        v["xq"] = fakes()
    return v


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("name", IMAGES)
@pytest.mark.parametrize("b", BLOCKS)
def test_image_block_and_maps(name, b, world):
    """The packed block is the cubin's (one parameter, its size), every pointer the program reads reaches it, and
    each GEMM slot's maps are the W / x geometry the units stream."""
    s = T.shim()
    img = image(name, b, world)
    c, d = img.c, dims(world, b)
    fakes = T.Fakes()
    v = bind_fields(c, fakes, world)
    s.shim_reset()
    launch = img.bind(**v)
    blob, slots = launch.params()
    size, kslots = T.params_of(paths(world)[f"dflash_layer_{name}_b{b}"], LAYER_PREFIX)
    assert len(blob) == size and slots == kslots == [(0, size)], (
        name,
        b,
        len(blob),
        size,
        slots,
    )
    words = {
        int.from_bytes(blob[i : i + 8], "little") for i in range(0, len(blob) - 7, 8)
    }
    encodes = [s.shim_get(i).contents for i in range(s.shim_count())]
    bases = {e.base for e in encodes}
    for field, val in v.items():
        for p in val if isinstance(val, list) else [val]:
            if isinstance(p, int) and p >= 0x7F00_0000_0000:
                assert (
                    p in words or p in bases
                ), f"{name} b{b}: {field} = {p:#x} never reaches the block"
    n = 0
    for g in range(len(c.gemm_m)):
        if not (c.gemm_mask >> g) & 1:
            continue
        w, x = encodes[n], encodes[n + 1]
        n += 2
        k = v["gemm_k"][g]
        T.expect_map(
            w,
            v["gemm_w"][g],
            T.BF16,
            (k, v["gemm_w_rows"][g]),
            (64, 64),
            (2 * k,),
            T.SW128,
        )
        # x as [k / 64][rows][64]: one box per ring slot, a slot's k-blocks (four 16-row or three 32-row stages)
        pack = 4 if c.xm == 16 or c.gemm_m[g] == 16 else 3
        T.expect_map(
            x,
            v["gemm_x"][g],
            T.BF16,
            (64, 32, k // 64),
            (64, c.gemm_m[g], pack),
            (2 * k, 128),
            T.SW128,
        )
    if (
        c.has_attn
    ):  # the attention's K / V rings (64 x 64 bf16 boxes) and its partials (32 fp32 x 8 B rows)
        row = 128 * d["kv"]  # a ring row: the rank's kv heads side by side
        for ring in ("attn_k_ring", "attn_v_ring"):
            T.expect_map(
                encodes[n], v[ring], T.BF16, (row, 4096), (64, 64), (2 * row,), T.SW128
            )
            n += 1
        T.expect_map(
            encodes[n],
            v["attn_partials"],
            7,
            (64, d["attn_ctas"] * 132),
            (32, 8 * b),
            (256,),
            T.SW128,
        )
        n += 1
        # the q rows: [B tokens][the rank's qkv heads][128], one [B][8 heads][64] box per kv-head group and head half
        T.expect_map(
            encodes[n],
            v["attn_q"],
            T.BF16,
            (128, d["qkv"] // 128, b),
            (64, 8, b),
            (256, v["attn_q_stride"]),
            T.SW128,
        )
        n += 1
    assert n == s.shim_count()


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("b", BLOCKS)
def test_programs(b, world):
    """The queue lengths and slot use of each program (layer.cu)."""
    T.shim()
    d = dims(world, b)
    gu, qe, combine = 2 * d["inter"] // 64 // 4, d["qkv"] // 64, d["combine_entries"]
    # each conv's side 0 as a prepare unit (16 entries of a 4-CTA K split, the prepare in their epilogue), side 1 a
    # node quarter (4)
    queue = 32 + b + 16 + 4 + gu + 32 + b + 16 + qe + 4
    want = {
        # the combine's cluster entries lead the layer's queue; the last layer ends at the MLP rows
        "layer": (combine + queue, 0xFF),
        "tail": (combine + 32 + b + 16 + 4 + gu + 32 + b, 0b11111),
    }
    for name, (length, mask) in want.items():
        c = image(name, b, world).c
        assert (c.queue_len, c.gemm_mask) == (length, mask), name
        assert (c.has_attn, c.combine_ctas) == (1, d["combine_ctas"]), name
        assert (c.head_qkv, c.head_qkv_arrivals) == (0, 0), name
    c = image("layer", b, world).c
    assert (
        c.gemm_m[6] == 2 * b
        and c.xm == max(16, 2 * b)
        and c.gemm_blocks[3] == 2 * d["inter"] // 64
        and c.gemm_blocks[6] == qe
    )
    assert c.gemm_blocks[1] == c.gemm_blocks[5] == 16 and c.gemm_blocks[2] == 4
    assert c.prep_arrivals == 64 and c.prep_mask == 0b11 and c.gat_mask == 0b1010


def attn_schedule(
    queue_len: int, combine: int, clusters: int = 36, attn: int = 9
) -> dict:
    """layer.cu's claim schedule of an attention image: physical cluster p runs virtual cluster c = (p + q) % 36, q the
    queue's clusters; claim k of c -> its entry; the attention's virtual clusters (the last ``attn``) skip round 0;
    rounds >= 1 at round_pos(c) (the combine's ``combine`` clusters first).
    -> {virtual cluster: [entries in claim order]}"""
    q = clusters - attn
    out = {}
    for p in range(clusters):
        c = (p + q) % clusters
        pos = c if c < combine else combine + c - q if c >= q else c + attn
        got, k = [], 0
        while True:
            r = k + 1 if c >= q else k
            e = c if r == 0 else q + (r - 1) * clusters + pos
            if e >= queue_len:
                break
            got.append(e)
            k += 1
        out[c] = got
    return out


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("b", BLOCKS)
@pytest.mark.parametrize("name", ["layer", "tail"])
def test_attn_schedule(b, name, world):
    """Every entry is claimed exactly once and each cluster claims ascending (the queue's deadlock freedom: a unit
    waits only on earlier entries); the attention's clusters take nothing in round 0, and the entries the combine waits
    on land on queue clusters."""
    d = dims(world, b)
    n = image(name, b, world).c.queue_len
    combine, attn = d["combine_entries"], d["attn_ctas"] // 4
    q = 36 - attn
    sched = attn_schedule(n, combine, attn=attn)
    every = sorted(e for v in sched.values() for e in v)
    assert every == list(range(n))
    assert all(v == sorted(v) for v in sched.values())
    assert all(not v or v[0] >= q for c, v in sched.items() if c >= q)
    owner = {e: c for c, v in sched.items() for e in v}
    assert all(owner[e] < q for e in range(combine))
    # no cluster streams two o_proj tiles (the combine's entries, then o_proj's 32)
    o_proj = range(combine, combine + 32)
    assert all(sum(e in o_proj for e in v) <= 1 for v in sched.values())
