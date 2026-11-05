"""CPU checks, no GPU: every cubin and host library builds; every packed parameter block is its kernel's (size =
EIATTR_CBANK_PARAM_SIZE, each parameter at the cubin's offset and size); every pointer a bind carries reaches the
block; the tensor maps (base, dims, box, stride, swizzle) are the ones the kernels' TMA loads expect, in the kernel's
parameter order, the MoE variant's slot by slot; the layer's dynamic smem layout fits next to the kernel's static
smem; the code-prefetch patch (the host's cubin edit that lets a layer launch load its own instructions into L2).

Tensor maps need a CUDA context a CPU-only host does not have: tmap_shim.c stands in for the encoder.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from functools import cache
from pathlib import Path

from sgl_muse_mega.common.jit import CACHE, toolkit
from sgl_muse_mega.muse_spark.config import Config
from sgl_muse_mega.muse_spark.sm103 import artifacts
from sgl_muse_mega.muse_spark.sm103.verify import LayerLibrary

HERE = Path(__file__).resolve().parent
CUOBJDUMP = toolkit("cuobjdump")
# CUtensorMapDataType (FP4 = 16U4_ALIGN8B)
BF16, TF32, U8, U32, U64, FP4 = (
    9,
    11,
    0,
    2,
    4,
    13,
)
SW128, SWNONE, L2_NONE, L2_256B = 3, 0, 0, 3
TAG = 0x7E45A9C3D2B10000
OPTIN_SMEM = 232448  # sm_103 per-block opt-in
SMEM_RESERVE = 1024  # cuobjdump's SHARED includes the 1 KB system reserve


def layer_zero(c) -> set[str]:
    """The MkLayerBind fields an image binds as 0: the q temperature, a bf16 image's unread MoE fields, and the
    pooled-ready words where the image has none (TP8)."""
    return (
        {"temp_scale"}
        | (BF16_UNREAD if c.bf16 else set())
        | (set() if c.pooled_ready_words else {"pooled_ready"})
    )


# the bind fields a bf16 image never reads (its MoE variant has no scale factors and no row-window exponents)
BF16_UNREAD = {
    "sf",
    "x_scale",
    "w1_sf_blk",
    "w2_sf",
    "g1_alphas",
    "g1_scale_c",
    "g2_alphas",
    "z_ex",
    "zex_peer",
}


class Encode(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("dtype", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("swizzle", ctypes.c_int32),
        ("l2", ctypes.c_int32),
        ("dims", ctypes.c_uint64 * 5),
        ("strides", ctypes.c_uint64 * 4),
        ("box", ctypes.c_uint32 * 5),
    ]


@cache
def shim() -> ctypes.CDLL:
    out = CACHE / "tmap_shim.so"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcc", "-O1", "-shared", "-fPIC", "-o", str(out), str(HERE / "tmap_shim.c")],
        check=True,
    )
    # ahead of every host library: their encodes bind here
    lib = ctypes.CDLL(str(out), mode=os.RTLD_GLOBAL)
    lib.shim_get.restype = ctypes.POINTER(Encode)
    return lib


@cache
def paths(world: str = "tp8") -> dict[str, Path]:
    return artifacts.build(m8=True, world=world)


def world_config(world: str) -> Config:
    return Config(tp=int(world.removeprefix("tp")))


def host():
    shim()
    from sgl_muse_mega.common import host as h

    return h


def kernel_params(cubin: Path) -> dict[str, tuple[int, list[tuple[int, int]]]]:
    """kernel -> (EIATTR_CBANK_PARAM_SIZE, [(offset, size)] by ordinal), from cuobjdump -elf."""
    text = subprocess.run(
        [CUOBJDUMP, "-elf", str(cubin)], check=True, capture_output=True, text=True
    ).stdout
    out, cur, attr = {}, None, None
    for line in text.splitlines():
        if line.startswith(".nv.info."):
            cur, attr = line[len(".nv.info.") :].strip(), None
            out[cur] = [None, {}]
            continue
        if line and not line[0].isspace():
            cur = None
        if cur is None:
            continue
        m = re.search(r"Attribute:\s+(\w+)", line)
        if m:
            attr = m[1]
        elif "Value:" in line and attr == "EIATTR_CBANK_PARAM_SIZE":
            out[cur][0] = int(re.search(r"Value:\s+(0x[0-9a-f]+)", line)[1], 16)
        elif "Value:" in line and attr in (
            "EIATTR_KPARAM_INFO",
            "EIATTR_KPARAM_INFO_V2",
        ):
            o, off, size = re.search(
                r"Ordinal\s*:\s*(0x\w+)\s+Offset\s*:\s*(0x\w+)\s+Size\s*:\s*(0x\w+)",
                line,
            ).groups()
            out[cur][1][int(o, 16)] = (int(off, 16), int(size, 16))
    return {
        k: (size, [p[i] for i in sorted(p)])
        for k, (size, p) in out.items()
        if size is not None
    }


def static_smem(cubin: Path, prefix: str) -> int:
    text = subprocess.run(
        [CUOBJDUMP, "-res-usage", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    hits = re.findall(
        r"Function (" + re.escape(prefix) + r"\S*):\n.*?SHARED:(\d+)", text
    )
    assert len(hits) == 1, (prefix, hits)
    return int(hits[0][1]) - SMEM_RESERVE


def params_of(cubin: Path, prefix: str):
    found = [(k, v) for k, v in kernel_params(cubin).items() if k.startswith(prefix)]
    assert len(found) == 1, (prefix, [k for k, _ in found])
    return found[0][1]


class Fakes:
    """Distinct fake device pointers (64-B aligned, far from small integers)."""

    def __init__(self):
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return 0x7F00_0000_0000 + self.n * 0x10000


def fill(
    struct: str, fakes: Fakes, ints: dict, zero: set, ranks: int | None = None
) -> dict:
    """Every field of a bind struct: the given ints, 0 for ``zero``, else distinct fake pointers (a peer array: one per
    rank of the world, ``ranks``, default its whole length)."""
    h = host()
    values = {}
    for name, ftype in h.STRUCTS[struct]._fields_:
        if name in ints:
            values[name] = ints[name]
        elif issubclass(ftype, ctypes.Array):
            n = ranks or ftype._length_
            values[name] = [0] * n if name in zero else [fakes() for _ in range(n)]
        else:
            values[name] = 0 if name in zero else fakes()
    return values


def check_block(
    launch, cubin: Path, prefix: str, values: dict, pointer_fields
) -> tuple[bytes, list, list]:
    """The packed block against the cubin, and every nonzero pointer of the bind inside it (or under a tensor map)."""
    blob, slots = launch.params()
    size, kslots = params_of(cubin, prefix)
    assert len(blob) == size, (prefix, len(blob), size)
    assert slots == kslots, (prefix, slots, kslots)
    s = shim()
    encodes = [s.shim_get(i).contents for i in range(s.shim_count())]
    words = {
        int.from_bytes(blob[i : i + 8], "little") for i in range(0, len(blob) - 7, 8)
    }
    bases = {e.base for e in encodes}
    for name in pointer_fields:
        v = values[name]
        for p in v if isinstance(v, list) else [v]:
            assert (
                p == 0 or p in words or p in bases
            ), f"{prefix}: {name} = {p:#x} never reaches the block"
    return blob, slots, encodes


def map_at(blob: bytes, slots, encodes, param: int) -> Encode:
    word = int.from_bytes(blob[slots[param][0] : slots[param][0] + 8], "little")
    assert word & ~0xFFFF == TAG, f"parameter {param} holds no encoded map"
    return encodes[word & 0xFFFF]


def expect_map(e: Encode, base, dtype, dims, box, stride, swizzle) -> None:
    got = (
        e.base,
        e.dtype,
        tuple(e.dims[: e.rank]),
        tuple(e.box[: e.rank]),
        tuple(e.strides[: e.rank - 1]),
        e.swizzle,
    )
    want = (base, dtype, dims, box, stride, swizzle)
    assert got == want and e.l2 == L2_256B, (got, want, e.l2)


# ---- checks ----


def test_images():
    cfg = Config()
    cfg.validate()
    images = [cfg.image(L) for L in range(cfg.layers)]
    assert set(images) == set(artifacts.LAYER_IMAGES)
    assert [images.count(i) for i in artifacts.LAYER_IMAGES] == [45, 15, 1, 1]


def test_layer_blocks():
    """Every layer image (and its 8-row window's) of every world: its host library's block against its cubin."""
    h, s = host(), shim()
    for world in artifacts.WORLDS:
        cfg = world_config(world)
        for image in artifacts.LAYER_IMAGES:
            check_layer(cfg, h, s, image, 16, world)
            check_layer(cfg, h, s, f"{image}_m8", 8, world)


def check_layer(cfg, h, s, tag: str, rows: int, world: str) -> None:
    lib = LayerLibrary(paths(world)[f"libmk_layer_{tag}"])
    c, cubin = lib.constants, paths(world)[f"layer_{tag}"]
    shape = (c.grid, c.cluster, c.rows, c.hidden, c.oproj_k, c.qkv_row)
    assert shape == (144, 4, rows, 8192, cfg.oproj_k, cfg.qkv_row), (tag, shape)
    assert c.smem + static_smem(cubin, lib.kernel) <= OPTIN_SMEM, (tag, c.smem)
    zero = layer_zero(c)
    ints = dict(
        rank=3,
        plane_stride=1572864,
        num_local=cfg.local_experts,
        zb_par_slots=c.moe_slots,
        kv_rows=4096,
        page_stride=131088,
        nk_page_stride=131088,
        nk_splits=c.nextkv_splits,
        nk_window=cfg.swa_window,
        b_live=c.rows,
        prefetch_experts=int(not c.bf16),
        router_mult=1.0,
        route_scale=1.0,
        eps=1e-5,
        q_scale=cfg.q_scale,
        qkv_eps=1e-5,
    )
    v = fill("MkLayerBind", Fakes(), ints, zero, cfg.tp)
    s.shim_reset()
    launch = lib.bind(lib.dll.mk_layer_bind, None, h.make("MkLayerBind", **v))
    ptrs = [n for n, _ in h.STRUCTS["MkLayerBind"]._fields_ if n not in ints]
    blob, slots, enc = check_block(launch, cubin, lib.kernel, v, ptrs)
    assert len(slots) == 23, (tag, len(slots))
    H, mh, m, ok, n2 = c.hidden, c.moe_hidden, c.rows, c.oproj_k, c.qkv_row
    ch = 14  # the next-layer phase's first tensor map
    # (parameter, base, dtype, dims, box, row stride bytes, swizzle): the maps the kernels' TMA loads expect
    maps = [
        (1, v["w_o"], BF16, (ok, H), (64, 64), ok * 2, SW128),
        (2, v["attn_arena"] + c.off_x_attn, BF16, (ok, m), (64, m), ok * 2, SW128),
        (3, v["w_p"], BF16, (H, mh), (64, 64), H * 2, SW128),
        (4, v["h3"], BF16, (H, m), (64, m), H * 2, SW128),
        # only descriptor-prefetched
        (
            5,
            v["h3"],
            BF16,
            (mh, m),
            (64, m),
            mh * 2,
            SWNONE,
        ),
        (8, v["w_r"], TF32, (H, c.experts), (32, 64), H * 4, SW128),
        (9, v["h3f"], TF32, (H, m), (32, m), H * 4, SW128),
        (ch, v["w_up"], BF16, (mh, H), (64, 64), mh * 2, SW128),
        (ch + 1, v["y"], BF16, (mh, m), (64, m), mh * 2, SW128),
        (ch + 2, v["w_qkv"], BF16, (H, n2), (64, 64), H * 2, SW128),
        (ch + 3, v["h3q"], BF16, (H, m), (64, m), H * 2, SW128),
        (ch + 4, v["qkv_out"], BF16, (n2, m), (64, m), n2 * 2, SWNONE),
    ]
    for param, base, dtype, dims, box, stride, swz in maps:
        expect_map(
            map_at(blob, slots, enc, param), base, dtype, dims, box, (stride,), swz
        )
    kv = sorted(
        (e.base, tuple(e.box[:2]))
        for e in enc
        if e.base in (v["k_cache"], v["v_cache"])
    )
    # per-row maps and their 128-row block variants over this layer's pool
    if c.global_attn:
        want = [(v[p], b) for p in ("k_cache", "v_cache") for b in ((64, 1), (64, 128))]
        assert kv == sorted(want), (tag, kv)
    else:
        assert kv == [], (tag, kv)
    moe = moe_maps(cfg, v, bool(c.bf16))
    assert len(enc) == len(maps) + len(kv) + len(moe), (tag, len(enc))
    la = slots[10][0]  # LayerArgs opens with its TmapSet: one 128 B map per slot
    for where, base, dtype, dims, box, stride, swz in moe:
        if where == "sfab":
            e = next(e for e in enc if e.base == base)
        else:
            word = int.from_bytes(
                blob[la + 128 * where : la + 128 * where + 8], "little"
            )
            assert word & ~0xFFFF == TAG, (tag, where)
            e = enc[word & 0xFFFF]
        got = (
            e.base,
            e.dtype,
            tuple(e.dims[: e.rank]),
            tuple(e.box[: e.rank]),
            tuple(e.strides[: e.rank - 1]),
        )
        assert (
            got == (base, dtype, dims, box, stride)
            and e.swizzle == swz
            and e.l2 == L2_NONE
        ), (tag, where, got)
    # the slots no device code reads stay zero: TM_G1_SFA (the A side's SF comes from the blocked image), and every
    # scale-factor slot of the bf16 variant
    for unread in (1, 3, 5, 7) if c.bf16 else (1,):
        assert blob[la + 128 * unread : la + 128 * (unread + 1)] == bytes(128), (
            tag,
            unread,
        )
    print(
        f"{tag:30s} {len(blob)} B in {len(slots)} parameters, {len(enc)} tensor maps: matches the cubin"
    )


def moe_maps(cfg: Config, v: dict, bf16: bool) -> list:
    """The MoE variant's tensor maps at this world and window (host/runtime.cu impl_host_init_tmaps), as (TmapSet slot
    or "sfab", base, dtype, dims, box, row strides in bytes, swizzle); dims and box innermost first.
    """
    E, H, I, R = cfg.experts, cfg.moe_hidden, cfg.moe_inter, cfg.tp
    rows, slots_ = (
        64,
        64 * cfg.topk,
    )  # the MoE phase's capacity (cfg::kMoeM) and its route slots
    feed, a_rows, fb_full, hb_pr = (
        128,
        256,
        cfg.inter_rank // 128,
        cfg.moe_hidden_rank // 128,
    )
    if bf16:
        bk = 64
        return [
            (
                0,
                v["w1"],
                BF16,
                (bk, H * 2 // feed * a_rows, E * fb_full),
                (bk, a_rows, 1),
                (feed, a_rows * H * 2),
                SW128,
            ),
            (2, v["q"], BF16, (H, rows), (bk, 1), (H * 2,), SW128),
            (
                4,
                v["w2"],
                BF16,
                (bk, hb_pr * (I * 2 // feed) * 128, E),
                (bk, 128, 1),
                (feed, (H // R) * I * 2),
                SW128,
            ),
            (6, v["z"], BF16, (I, slots_), (bk, 32), (I * 2,), SW128),
        ]
    bk, stages = 256, I // 256
    sf_row = 256  # the g1 SF block's bytes per row: one chunk of the cell's 16 K stages (moe.cuh g1::SF_ROW)
    sfa_block = a_rows * sf_row
    return [
        (
            0,
            v["w1"],
            FP4,
            (bk, H // 2 // feed * a_rows, E * fb_full),
            (bk, a_rows, 1),
            (feed, a_rows * H // 2),
            SW128,
        ),
        (
            "sfab",
            v["w1_sf_blk"],
            U64,
            (256, sfa_block // 2048, E * fb_full),
            (256, sfa_block // 2048, 1),
            (2048, sfa_block),
            SWNONE,
        ),
        (2, v["q"], FP4, (H, rows), (bk, 1), (H // 2,), SW128),
        (3, v["sf"], U32, (H // 64, rows), (sf_row // 4, 1), (H // 16,), SWNONE),
        (
            4,
            v["w2"],
            FP4,
            (bk, hb_pr * (I // 2 // feed) * 128, E),
            (bk, 128, 1),
            (feed, (H // R) * I // 2),
            SW128,
        ),
        (
            5,
            v["w2_sf"],
            U32,
            (I // 64, H // R, E),
            (I // 64, 128, 1),
            (I // 16, (H // R) * I // 16),
            SWNONE,
        ),
        (6, v["z"], FP4, (I, slots_), (bk, 32), (I // 2,), SW128),
        # the row window's stage-major SF and exponent planes: [stage][slot][16 B], a box = 32 slots x every stage
        (
            7,
            v["z_sf"],
            U32,
            (slots_ * 4, stages),
            (32 * 4, stages),
            (slots_ * 16,),
            SWNONE,
        ),
        (
            8,
            v["z_ex"],
            U32,
            (slots_ * 4, stages),
            (32 * 4, stages),
            (slots_ * 16,),
            SWNONE,
        ),
    ]


def test_aux_blocks():
    """The other kernels' blocks: the 16-row library, and the 8-row window's (its own qkv head and lm_head cubins),
    of every world."""
    for world in artifacts.WORLDS:
        for suffix, m in (("", 16), ("_m8", 8)):
            check_aux(suffix, m, world)


def check_aux(suffix: str, m: int, world: str) -> None:
    cfg, h, s = world_config(world), host(), shim()
    a = h.Library(paths(world)[f"libmk_aux{suffix}"])
    p = paths(world)
    H, n2 = cfg.hidden, cfg.qkv_row
    cases = [
        (
            "mk_feeds_bind",
            "MkFeedsBind",
            "window_feeds",
            "_ZN2mk12window_feedsILj256E",
            dict(
                cos_sin_rows=131088,
                r2t_rows=1,
                r2t_width=131088,
                swa_map_len=131088,
                rows=m,
                head_dim=64,
                kv_row_bytes=cfg.kv_row_bytes,
            ),
        ),
        (
            "mk_entry_bind",
            "MkEntryBind",
            "entry_fused",
            "_ZN2mk11entry_fusedI",
            dict(
                shard_rows=cfg.vocab_rank,
                rows=m,
                hidden=H,
                eps_embed=1e-5,
                eps_pre=1e-5,
            ),
        ),
        (
            "mk_qkv_bind",
            "MkQkvBind",
            f"qkv_head{suffix}",
            "_ZN2mk10dense_gemmI",
            dict(temp_scale=0, q_scale=cfg.q_scale, eps=1e-5, hidden=H),
        ),
        (
            "mk_lm_head_bind",
            "MkLmHeadBind",
            f"lm_head{suffix}",
            "_ZN2mk20dense_persist_gatherI",
            dict(
                arrive_off=m * cfg.vocab * 4,
                v_total=cfg.vocab,
                col0=cfg.vocab_rank * 3,
                n_out=cfg.vocab_rank,
                n_pad=cfg.lm_head_pad,
                grid=132,
                hidden=H,
            ),
        ),
        (
            "mk_softcap_bind",
            "MkSoftcapBind",
            "softcap",
            "_ZN2mk16softcap_gatheredI",
            dict(numel=m * cfg.vocab, scale=1.0, cap=cfg.softcap, per_launch=cfg.tp),
        ),
    ]
    for fn, struct, cubin, prefix, ints in cases:
        values = fill(struct, Fakes(), ints, set(), cfg.tp)
        s.shim_reset()
        launch = a.bind(a.dll[fn], None, h.make(struct, **values))
        ptrs = [n for n, t in h.STRUCTS[struct]._fields_ if n not in ints]
        blob, slots, enc = check_block(launch, p[cubin], prefix, values, ptrs)
        v = values
        if fn == "mk_qkv_bind":
            expect_map(
                map_at(blob, slots, enc, 1),
                v["w"],
                BF16,
                (H, n2),
                (64, 64),
                (H * 2,),
                SW128,
            )
            expect_map(
                map_at(blob, slots, enc, 2),
                v["x"],
                BF16,
                (H, m),
                (64, m),
                (H * 2,),
                SW128,
            )
            expect_map(
                map_at(blob, slots, enc, 3),
                v["out"],
                BF16,
                (n2, m),
                (64, m),
                (n2 * 2,),
                SWNONE,
            )
        if fn == "mk_lm_head_bind":
            expect_map(
                map_at(blob, slots, enc, 1),
                v["w"],
                BF16,
                (H, cfg.vocab_rank),
                (64, 64),
                (H * 2,),
                SW128,
            )
            expect_map(
                map_at(blob, slots, enc, 2),
                v["x"],
                BF16,
                (H, m),
                (64, m),
                (H * 2,),
                SW128,
            )
        print(
            f"{prefix:44s} m{m:<3d} {len(blob)} B in {len(slots)} parameters: matches the cubin"
        )


def _section_sizes(data: bytes) -> dict[str, int]:
    """ELF64 section name -> size."""
    shoff = int.from_bytes(data[0x28:0x30], "little")
    entsize, num, strndx = (
        int.from_bytes(data[o : o + 2], "little") for o in (0x3A, 0x3C, 0x3E)
    )

    def header(k: int) -> tuple[int, int, int]:
        h = data[shoff + k * entsize : shoff + (k + 1) * entsize]
        return (
            int.from_bytes(h[0:4], "little"),
            int.from_bytes(h[24:32], "little"),
            int.from_bytes(h[32:40], "little"),
        )

    _, str_off, _ = header(strndx)
    out = {}
    for k in range(num):
        name, _, size = header(k)
        end = data.index(b"\0", str_off + name)
        out[data[str_off + name : end].decode()] = size
    return out


def test_code_prefetch_patch():
    """The C++ in-memory patch on every layer image: a few bytes change (the native object's relocation and value)
    and the text size it reports is the kernel's ``.text`` section's."""
    host()
    for world, image in (
        (w, i)
        for w in artifacts.WORLDS
        for i in (*artifacts.LAYER_IMAGES, *(f"{i}_m8" for i in artifacts.LAYER_IMAGES))
    ):
        lib = LayerLibrary(paths(world)[f"libmk_layer_{image}"])
        data = paths(world)[f"layer_{image}"].read_bytes()
        mine = ctypes.create_string_buffer(data, len(data))
        text = ctypes.c_uint64()
        lib.dll.mk_code_prefetch_patch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.check(
            lib.dll.mk_code_prefetch_patch(
                mine, len(data), lib.kernel.encode(), ctypes.byref(text)
            ),
            "patch",
        )
        sizes = {
            n: v
            for n, v in _section_sizes(data).items()
            if n.startswith(".text." + lib.kernel)
        }
        assert len(sizes) == 1 and text.value == next(iter(sizes.values())), (
            image,
            sizes,
            text.value,
        )
        changed = sum(a != b for a, b in zip(mine.raw, data))
        assert 1 <= changed <= 9, (image, changed)
        print(
            f"{world} {image:19s} code-prefetch patch: {changed} bytes changed, text {text.value} B"
        )


def test_imports():
    import importlib

    for mod in (
        "plugin",
        "common.jit",
        "common.host",
        "muse_spark.config",
        "muse_spark.sm103.artifacts",
        "muse_spark.sm103.buffers",
        "muse_spark.sm103.verify",
        "muse_spark.sm103.weights",
        "muse_spark.sm103.dflash.iteration",
    ):
        importlib.import_module(f"sgl_muse_mega.{mod}")
    print("all modules import")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
