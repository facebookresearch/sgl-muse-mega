"""Every image the stack loads, built with ``common.jit``: the device cubins and the host libraries that bind them.

One image per build: ``-DMK_IMAGE=<image>`` over ``kernels/layer.cu`` (the four layer images) and over
``kernels/head.cu`` (the rest of the verify graph), one host library per layer image (the same defines) plus one for the
other kernels, every library from ``host/runtime.cu`` at its ``-DMK_HOST=<library>``.
Every artifact builds for one tensor-parallel world, ``-DMK_WORLD=tp8|tp4`` (``kernels/framework.cuh``'s world table):
a world's set carries the same names, so one ``build`` call returns one world's. ``m8`` adds the 8-row window's set
(suffix ``_m8``): the layer images, the qkv head, the lm_head and their libraries at ``-DMK_WINDOW_M=8``; the feeds,
entry and softcap take their rows at bind; the layer images' attention runs at the window's 8 rows (attention.cuh: the
sliding-window attention's 2-head row tiles, the global attention's M = 64 tiles). ``dflash`` adds the DFlash images
(``kernels/dflash/``, ``dflash_artifacts``: with the sampled step's head and accept, ``-DMK_DFLASH_SAMPLE=1``, suffix
``_sample``, and the draft-pool sync) and the 8-row set (the B = 8 verify); ``tree`` the token-tree window's images
(``-DMK_DFLASH_TREE=1``, suffix ``_tree``: the head, verify_accept with tree_compact, the four layer images; they bind
through the same host libraries).

    python -m sgl_muse_mega.muse_spark.sm103.artifacts [--world tp8|tp4] [--m8] [--dflash] [--tree]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ...common.jit import Artifact, build_artifacts
from . import HOST, KERNELS, ROOT

# the tensor-parallel worlds a build selects (kernels/framework.cuh)
WORLDS = ("tp8", "tp4")
# the four one-launch layer images: attention type x MoE variant (x the last layer)
LAYER_IMAGES = ("swa", "global", "swa_bf16", "global_bf16_tail")
# the rest of the verify graph: image -> its translation unit
AUX_IMAGES = ("window_feeds", "qkv_head", "lm_head", "softcap", "entry_fused")
# DFlash: the block sizes (a compile pin of the entry, qkv and layer images) and the draft-layer images
DFLASH_BLOCKS = (8, 16)
DFLASH_LAYER_IMAGES = ("layer", "tail")
# every host library is this one source at its -DMK_HOST=<library>
RUNTIME = (HOST / "runtime.cu",)


def _at_world(world: str, todo: list[Artifact]) -> list[Artifact]:
    assert world in WORLDS, world
    return [replace(a, defines=(f"MK_WORLD={world}", *a.defines)) for a in todo]


def artifacts(m8: bool = False, world: str = "tp8") -> list[Artifact]:
    """Everything a verify step loads at ``world``: a cubin and a host library per layer image, the other kernels'
    cubins and the aux host library (with ``m8``, also the 8-row window's set)."""
    windows = ((), ("MK_WINDOW_M=8",)) if m8 else ((),)
    out = []
    for window in windows:
        sfx = "_m8" if window else ""
        for image in LAYER_IMAGES:
            d = (f"MK_IMAGE={image}", *window)
            out += [
                Artifact(
                    f"layer_{image}{sfx}",
                    "cubin",
                    (KERNELS / "layer.cu",),
                    d,
                    compress_none=True,
                ),
                Artifact(
                    f"libmk_layer_{image}{sfx}", "so", RUNTIME, ("MK_HOST=layer", *d)
                ),
            ]
    out += [
        Artifact(image, "cubin", (KERNELS / "head.cu",), (f"MK_IMAGE={image}",))
        for image in AUX_IMAGES
    ]
    out.append(Artifact("libmk_aux", "so", RUNTIME, ("MK_HOST=aux",)))
    if m8:
        out += [
            Artifact(
                f"{image}_m8",
                "cubin",
                (KERNELS / "head.cu",),
                (f"MK_IMAGE={image}", *windows[1]),
            )
            for image in ("qkv_head", "lm_head")
        ]
        out.append(
            Artifact("libmk_aux_m8", "so", RUNTIME, ("MK_HOST=aux", *windows[1]))
        )
    return _at_world(world, out)


def dflash_artifacts(world: str = "tp8", tree: bool = False) -> list[Artifact]:
    """Every DFlash image and host library a DFlash iteration loads at ``world`` (each cubin's host library shares its
    defines); ``tree``: also the token-tree window's images."""
    k = KERNELS / "dflash"
    out = []
    for b in DFLASH_BLOCKS:
        d = (f"MK_DFLASH_B={b}",)
        out += [
            Artifact(
                f"dflash_entry_b{b}", "cubin", (k / "head.cu",), ("MK_IMAGE=entry", *d)
            ),
            Artifact(
                f"libmk_dflash_entry_b{b}", "so", RUNTIME, ("MK_HOST=dflash_entry", *d)
            ),
            Artifact(
                f"dflash_qkv_b{b}", "cubin", (k / "layer.cu",), ("MK_IMAGE=qkv", *d)
            ),
        ]
        for image in DFLASH_LAYER_IMAGES:
            di = (f"MK_IMAGE={image}", *d)
            out += [
                Artifact(f"dflash_layer_{image}_b{b}", "cubin", (k / "layer.cu",), di),
                Artifact(
                    f"libmk_dflash_layer_{image}_b{b}",
                    "so",
                    RUNTIME,
                    ("MK_HOST=dflash_layer", *di),
                ),
            ]
    out += [
        Artifact("libmk_dflash_attn", "so", RUNTIME, ("MK_HOST=dflash_attn",)),
        Artifact("libmk_dflash_dense", "so", RUNTIME, ("MK_HOST=dflash_dense",)),
        Artifact(
            "dflash_draft_head", "cubin", (k / "head.cu",), ("MK_IMAGE=draft_head",)
        ),
        Artifact(
            "libmk_dflash_draft_head", "so", RUNTIME, ("MK_HOST=dflash_draft_head",)
        ),
        Artifact("dflash_verify_accept", "cubin", (k / "verify_accept.cu",)),
        Artifact(
            "libmk_dflash_verify_accept",
            "so",
            RUNTIME,
            ("MK_HOST=dflash_verify_accept",),
        ),
        Artifact(
            "dflash_draft_head_sample",
            "cubin",
            (k / "head.cu",),
            ("MK_IMAGE=draft_head", "MK_DFLASH_SAMPLE=1"),
        ),
        Artifact(
            "dflash_verify_accept_sample",
            "cubin",
            (k / "verify_accept.cu",),
            ("MK_DFLASH_SAMPLE=1",),
        ),
        Artifact("dflash_kv_sync", "cubin", (k / "kv_sync.cu",)),
    ]
    if tree:
        t = ("MK_DFLASH_TREE=1",)
        out += [
            Artifact(
                "dflash_draft_head_tree",
                "cubin",
                (k / "head.cu",),
                ("MK_IMAGE=draft_head", *t),
            ),
            Artifact(
                "dflash_verify_accept_tree", "cubin", (k / "verify_accept.cu",), t
            ),
        ]
        out += [
            Artifact(
                f"layer_{image}_tree",
                "cubin",
                (KERNELS / "layer.cu",),
                (f"MK_IMAGE={image}", *t),
                compress_none=True,
            )
            for image in LAYER_IMAGES
        ]
    return _at_world(world, out)


def build_set(todo: list[Artifact]) -> dict[str, Path]:
    """Builds (or finds in the cache) the given artifacts of this tree; returns name -> path."""
    return build_artifacts(todo, ROOT, KERNELS)


def build(
    dflash: bool = False, m8: bool = False, world: str = "tp8", tree: bool = False
) -> dict[str, Path]:
    """Builds (or finds in the cache) every artifact of ``world``, (``m8`` or ``dflash``) the 8-row set and
    (``dflash``) the DFlash images, with ``tree`` the token-tree window's; returns name -> path.
    """
    todo = artifacts(m8 or dflash, world) + (
        dflash_artifacts(world, tree) if dflash else []
    )
    return build_set(todo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m8", action="store_true", help="also the 8-row window's set")
    ap.add_argument(
        "--dflash",
        action="store_true",
        help="also the DFlash images (kernels/dflash/) and the 8-row set",
    )
    ap.add_argument("--world", choices=WORLDS, default="tp8")
    ap.add_argument(
        "--tree",
        action="store_true",
        help="with --dflash, also the token-tree window's images",
    )
    args = ap.parse_args()
    for name, path in build(
        dflash=args.dflash, m8=args.m8, world=args.world, tree=args.tree
    ).items():
        print(f"{name:32s} {path}")


if __name__ == "__main__":
    main()
