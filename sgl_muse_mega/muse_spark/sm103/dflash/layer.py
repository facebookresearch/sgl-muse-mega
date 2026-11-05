"""A draft layer from python (kernels/dflash/layer.cu): the layer images' host libraries, the cross-rank state they
push into and one launch record per binding; the layer's qkv as its own launch on the dense executor; the attention's
geometry (host/runtime.cu). The kernels' operation order in torch: test/reference/dflash2.py.

Images (``-DMK_IMAGE=<image> -DMK_DFLASH_B=8|16``, each with its host library):
  qkv     a draft layer's qkv (+ hd128 rowops) on the dense executor (layer 0's own launch)
  layer   a draft layer in one launch: the attention, its split-KV combine, then the layer's queue
  tail    the last draft layer (its rows end at the final norm)

The attention (36 CTAs per kv head of the rank, one 64-key block of the window each, a partial per CTA) and its combine
(partials -> x_attn) run inside the layer images.

Cross-rank state (torch symmetric memory over the TP group, filled at bind; only the tests' between-runs reset,
``Iteration.reset``, refills it):
  planes   2 sites (attention, MLP) x 2 phases x the ranks x one [B, H] bf16 slot per rank, pushed through the
           multicast address; a site's bases are [the rank bases, multicast] (host/abi.h fin_bases)
  coef     the four coefficient slots [B, coef_row] bf16, prefilled 0xFFFF (the empty mark: never a rounded value).
           The coefficient tiles land in the node's 4 ranks' rows; every packet has one reader, which re-marks it empty.

A wait that outlives its budget writes a host-pinned hang record ({code, block, thread, observed, target}, readable
after the trap has killed the context) and traps.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from ....common import host
from ..buffers import Symmetric
from .weights import NODE_RANKS, DraftConfig

# ---- the attention's geometry ----


def attn_constants(lib: host.Library, block: int) -> ctypes.Structure:
    c = host.STRUCTS["MkDflashAttnConstants"]()
    lib.dll.mk_dflash_attn_constants.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(type(c)),
    ]
    lib.check(lib.dll.mk_dflash_attn_constants(block, ctypes.byref(c)), "constants")
    return c


# ---- the qkv launch ----

QUARTERS = 4  # the qkv's token quarters (one per CTA of a width-4 cluster)


def qkv_heads(cfg: DraftConfig, ranks: int) -> tuple[int, int]:
    """(q heads, kv heads) of one rank's qkv rows."""
    return cfg.q_heads // ranks, cfg.kv_heads // ranks


def qkv_cols(cfg: DraftConfig, ranks: int) -> int:
    """One rank's qkv row: its q heads, then its k and v heads, head_dim each."""
    q, kv = qkv_heads(cfg, ranks)
    return (q + 2 * kv) * cfg.head_dim


def qkv_tickets(cfg: DraftConfig, ranks: int) -> int:
    """u32 words the qkv's PAIR fixup counts on: one per (head, token quarter)."""
    q, kv = qkv_heads(cfg, ranks)
    return (q + 2 * kv) * QUARTERS


class Dense:
    """The host library and the two images' modules (loaded on first use; a device context must be current)."""

    def __init__(self, paths: dict[str, Path]):
        self.paths = paths
        self.lib = host.Library(paths["libmk_dflash_dense"])
        self._modules: dict[str, host.Module] = {}

    def module(self, image: str) -> host.Module:
        if image not in self._modules:
            self._modules[image] = self.lib.load(self.paths[image])
        return self._modules[image]

    def qkv(self, block: int, pack_only: bool = False, **tensors) -> host.Launch:
        """A draft layer's qkv launch: the MkDflashQkvBind fields by name (block given here)."""
        b = host.make("MkDflashQkvBind", block=block, **tensors)
        mod = None if pack_only else self.module(f"dflash_qkv_b{block}")
        return self.lib.bind(self.lib.dll.mk_dflash_qkv_bind, mod, b)


# ---- the layer images ----

# the coefficient slots: attn_conv_proj side 0 / 1, mlp_conv_proj side 0 / 1
COEF_SLOTS = ("att0", "att1", "mlp0", "mlp1")
SITES = ("att", "mlp")


def empty_bind() -> dict:
    """Every MkDflashLayerBind field at its absent value (0 / zero arrays): a binding fills the slots its program
    reads."""
    out = {}
    for name, ftype in host.STRUCTS["MkDflashLayerBind"]._fields_:
        out[name] = [0] * ftype._length_ if issubclass(ftype, ctypes.Array) else 0
    return out


class Image:
    """A loaded (image, B): its host library, its compile-time facts (MkDflashLayerConstants) and its module."""

    def __init__(self, paths: dict[str, Path], image: str, b: int, module: bool = True):
        self.name, self.b = image, b
        self.lib = host.Library(paths[f"libmk_dflash_layer_{image}_b{b}"])
        d = self.lib.dll
        self.c = host.STRUCTS["MkDflashLayerConstants"]()
        d.mk_dflash_layer_constants.argtypes = [ctypes.POINTER(type(self.c))]
        self.lib.check(d.mk_dflash_layer_constants(ctypes.byref(self.c)), "constants")
        self.module = (
            self.lib.load(paths[f"dflash_layer_{image}_b{b}"]) if module else None
        )

    def bind(self, **fields) -> host.Launch:
        return self.lib.bind(
            self.lib.dll.mk_dflash_layer_bind,
            self.module,
            host.make("MkDflashLayerBind", **fields),
        )


class State:
    """The queue's cross-rank state for block size b on one rank (module docstring)."""

    def __init__(
        self, b: int, rank: int, device, group, ranks: int, hidden: int, coef_row: int
    ):
        self.b, self.rank, self.ranks, self.hidden, self.coef_row = (
            b,
            rank,
            ranks,
            hidden,
            coef_row,
        )
        self.slot = b * hidden
        self.site_bytes = 2 * ranks * self.slot * 2
        self.planes = Symmetric(len(SITES) * self.site_bytes, device, group)
        assert (
            self.planes.mc != 0
        ), "the lamport push stores through the TP group's multicast address"
        self.site_bases = {
            s: torch.tensor(
                [p + i * self.site_bytes for p in self.planes.ptrs]
                + [self.planes.mc + i * self.site_bytes],
                dtype=torch.int64,
                device=device,
            )
            for i, s in enumerate(SITES)
        }
        self.coef_bytes = b * coef_row * 2
        self.coef = Symmetric(len(COEF_SLOTS) * self.coef_bytes, device, group)
        self.coef.local.view(torch.int16).fill_(-1)
        torch.cuda.synchronize(device)

    def rows(self, slot: str) -> torch.Tensor:
        """This rank's coefficient rows of a slot, [b, coef_row] bf16."""
        i = COEF_SLOTS.index(slot)
        return (
            self.coef.local[i * self.coef_bytes : (i + 1) * self.coef_bytes]
            .view(torch.bfloat16)
            .view(self.b, self.coef_row)
        )

    def gather_dst(self, slot: str) -> list[int]:
        """A coefficient gather's destinations: the slot's rows in each of the node's 4 ranks (device addresses)."""
        node0 = self.rank - self.rank % NODE_RANKS
        return [
            self.coef.ptrs[node0 + i] + COEF_SLOTS.index(slot) * self.coef_bytes
            for i in range(NODE_RANKS)
        ]


class Words:
    """One binding's u32 words, zero at bind: [launch ticket per CTA | counts]."""

    COUNTS = ("row0", "row1", "prep0", "prep1", "act")

    def __init__(self, grid: int, device):
        self.t = torch.zeros(grid + len(self.COUNTS), dtype=torch.int32, device=device)
        self.grid = grid

    @property
    def launch_ctr(self) -> torch.Tensor:
        return self.t[: self.grid]

    def count(self, name: str) -> int:
        return self.t[self.grid + self.COUNTS.index(name) :].data_ptr()


def diag_words() -> torch.Tensor:
    """The host-pinned hang record every layer binding shares (kernels/dflash/layer.cu hang)."""
    return torch.zeros(8, dtype=torch.int64, pin_memory=True)
