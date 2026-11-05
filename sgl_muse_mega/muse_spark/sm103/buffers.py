"""Device state of the step, allocated once before capture. Every protocol word (tickets, epochs, arrival counts)
starts at zero: the kernels advance them across warm-ups and replays, every rank must run the same launches in
the same order, and only ``Buffers.reset`` (below: between the tests' and the benchmark's runs, every rank quiescent;
a server never resets) puts them back.

Cross-rank surfaces are torch symmetric memory over the TP group (8 ranks on two nodes of one NVLink domain, or 4 on
one node):
- the post-attention all-reduce workspace: 2 lamport phases x the ranks x one slot per rank; its multicast address
  follows the peer bases. Inside each slot, past the all-reduce message, the K-window variant's fp32 partial planes (+0
  = empty, like the message's packets), then the pooler and unpooler gather rows of the node's 4 ranks; they start
  all-ones (0xFFFF is never a rounded value, so it marks "not yet written").
- one MoE gather region per MoE variant (NVFP4 / bf16 experts): the gathered h rows (+ scale planes) and the zb parity
  ring of partial outputs, both prefilled with their lamport sentinels at bind and at every reset, and the anchor words.
- the vocab shards (the fused entry reads the owner's row) and the gathered logits row (multicast-written).

``Buffers.reset`` puts every word and workspace back to its bind-time value between runs, so every run starts as a
fresh bind does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..config import Config

# one rank's all-reduce slot per phase: 3 MiB (it holds the message and both gather rows: host/runtime.cu
# plane_stride_min)
AR_SLOT_BYTES = 3 << 20
REGION_ALIGN = 4096
# a token-tree window's table (kernels/dflash/tree.cuh kTableWords), after the feeds' kv_len word
TREE_WORDS = 96


def _zeros(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    return torch.zeros(int(n), dtype=dtype, device=device)


class Symmetric:
    """A zeroed symmetric allocation: this rank's bytes, every rank's base address and the multicast address."""

    def __init__(self, nbytes: int, device, group: dist.ProcessGroup):
        self.local = symm_mem.empty(int(nbytes), dtype=torch.uint8, device=device)
        self.handle = symm_mem.rendezvous(self.local, group)
        self.local.zero_()
        self.ptrs = [int(p) for p in self.handle.buffer_ptrs]
        self.mc = int(self.handle.multicast_ptr)


def published(group: dist.ProcessGroup, device) -> None:
    """Every rank's initial bytes are written before any peer's first launch can touch them."""
    torch.cuda.synchronize(device)
    dist.barrier(group=group)


class AllReduce:
    def __init__(self, c, cfg: Config, device, group):
        world = cfg.tp
        self.plane_stride = AR_SLOT_BYTES // 2  # bf16 elements
        assert self.plane_stride >= c.plane_stride_min, (
            self.plane_stride,
            c.plane_stride_min,
        )
        self.mem = Symmetric(2 * world * AR_SLOT_BYTES, device, group)
        assert (
            self.mem.mc != 0
        ), "the all-reduce pushes through the TP group's multicast address"
        self.peer_bases = torch.tensor(
            [*self.mem.ptrs, self.mem.mc], dtype=torch.int64, device=device
        )
        self.rows = (
            (c.unpool_ag_off, c.unpool_ag_off + c.rows * c.hidden),
            (c.pool_ag_off, c.pool_ag_off + c.rows * c.moe_hidden),
        )
        self.reset()

    def reset(self) -> None:
        self.mem.local.zero_()
        rows16 = self.mem.local.view(torch.int16)
        for lo, hi in self.rows:
            rows16[lo:hi].fill_(-1)


class MoeVariant:
    """One MoE variant (NVFP4 or bf16 experts): its symmetric gather region and the MoE phase's device-local state,
    shared by every layer of that variant (safe: consecutive layers' uses are ordered by the sequence of launches).
    """

    def __init__(self, c, device, group):
        slots, tiles, rows, mh = (
            c.moe_slots,
            c.moe_tiles,
            c.moe_rows,
            c.moe_hidden,
        )
        sizes = dict(
            z=slots * c.z_row_bytes,
            z_sf=slots * c.zsf_row_bytes,
            z_ex=slots * c.zsf_row_bytes,
            zb=c.zb_ring * slots * mh * 2,
            arrive=c.arrive_words * 4,
        )
        off, total = {}, 0
        for name, n in sizes.items():
            off[name] = total
            total += (n + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN
        self.mem = Symmetric(total, device, group)
        self.peers = {name: [p + off[name] for p in self.mem.ptrs] for name in sizes}
        self.views = {
            name: self.mem.local[off[name] : off[name] + n] for name, n in sizes.items()
        }
        # the cross-rank gathers' empty elements (kernels/moe.cuh): z / z_sf / z_ex bytes 0x80 on the NVFP4 variant,
        # z and zb all ones; the views this variant's MoE phase reads
        self.sentinels = [
            (self.views["zb"], 0xFF),
            (self.views["z"], 0xFF if c.bf16 else 0x80),
        ]
        if not c.bf16:
            self.sentinels += [(self.views["z_sf"], 0x80), (self.views["z_ex"], 0x80)]
        self.fill_sentinels()
        self.zb_par_slots = slots
        i32 = torch.int32
        self.counts, self.offsets = _zeros(c.experts, i32, device), _zeros(
            c.experts + 1, i32, device
        )
        self.slot_tok, self.slot_k, self.tok_slot = (
            _zeros(slots, i32, device) for _ in range(3)
        )
        self.tile_e, self.tile_r0 = _zeros(tiles, i32, device), _zeros(
            tiles, i32, device
        )
        # the tile count, then the MoE phase's pre-wait sentinel
        self.n_tiles = _zeros(2, i32, device)
        self.gsync = _zeros(c.gsync_words, torch.uint32, device)
        self.epoch = _zeros(1, torch.uint32, device)
        self.y = torch.zeros(rows, mh, dtype=torch.bfloat16, device=device)
        if c.bf16:
            # the normed rows
            self.q = torch.zeros(rows, mh, dtype=torch.bfloat16, device=device)
            self.sf = self.x_scale = None
        else:
            self.q = torch.zeros(rows, mh // 2, dtype=torch.uint8, device=device)
            self.sf = torch.zeros(rows, mh // 16, dtype=torch.uint8, device=device)
            self.x_scale = _zeros(rows, torch.float32, device)

    def fill_sentinels(self) -> None:
        for t, byte in self.sentinels:
            t.fill_(byte)

    def state(self) -> list[torch.Tensor]:
        """The words a reset zeroes (then ``fill_sentinels``)."""
        return [
            self.mem.local,
            self.counts,
            self.offsets,
            self.slot_tok,
            self.slot_k,
            self.tok_slot,
            self.tile_e,
            self.tile_r0,
            self.n_tiles,
            self.gsync,
            self.epoch,
            self.y,
            self.q,
            *([] if self.sf is None else [self.sf, self.x_scale]),
        ]


class AttnType:
    """An attention type's (sliding-window or global) arena (x_attn, the qkv row, the split-KV workspace) and the
    next-layer phase's outputs into that type's layers (the next layer's normed row and residual).
    """

    def __init__(self, c, cfg: Config, device):
        self.arena = _zeros(c.arena_bytes, torch.uint8, device)
        self.x_attn = self.arena[
            c.off_x_attn : c.off_x_attn + c.rows * c.oproj_k * 2
        ].view(torch.bfloat16)
        self.qkv_out = self.arena[
            c.off_qkv_out : c.off_qkv_out + c.rows * c.qkv_row * 2
        ].view(torch.bfloat16)
        self.h3 = torch.zeros(c.rows, c.hidden, dtype=torch.bfloat16, device=device)
        self.stream_out = torch.zeros(
            c.rows, c.hidden, dtype=torch.float32, device=device
        )
        # a NoPE successor's identity RoPE rows {cos 1 | sin 0}
        half = cfg.head_dim // 2
        self.nope_cs = torch.cat(
            [torch.ones(c.rows, half), torch.zeros(c.rows, half)], 1
        ).to(device)

    def state(self) -> list[torch.Tensor]:
        return [self.arena, self.h3, self.stream_out]


class LayerState:
    """One layer launch's own outputs and protocol words."""

    def __init__(self, c, device):
        m, H = c.rows, c.hidden
        self.stream_out = torch.zeros(m, H, dtype=torch.float32, device=device)
        self.h3 = torch.zeros(m, H, dtype=torch.bfloat16, device=device)
        self.h3f = torch.zeros(m, H, dtype=torch.float32, device=device)
        self.ws = _zeros(c.ws_floats, torch.float32, device)
        self.counters = _zeros(c.counters, torch.int32, device)
        self.logits = torch.zeros(m, c.experts, dtype=torch.float32, device=device)
        self.idx = torch.zeros(m, c.topk, dtype=torch.int64, device=device)
        self.local = torch.zeros(m, c.topk, dtype=torch.int64, device=device)
        self.idx32 = torch.zeros(m, c.topk, dtype=torch.int32, device=device)
        self.w = torch.zeros(m, c.topk, dtype=torch.float32, device=device)
        self.words = _zeros(c.words, torch.uint32, device)
        # the K-window variant's per-tile pooled-ready words (None at TP8: the image has none)
        self.pooled_ready = (
            _zeros(c.pooled_ready_words, torch.uint64, device)
            if c.pooled_ready_words
            else None
        )
        self.fctr = _zeros(c.fctr_words, torch.int32, device)
        self.sm_epoch = _zeros(c.sm_epoch_words, torch.int64, device)
        self.next_layer_words = _zeros(c.next_layer_words, torch.uint32, device)
        self.tail = bool(c.tail)
        if c.tail:
            # the last layer's next-layer phase ends at the final norm: its outputs, and placeholders for the unused
            # qkv GEMM
            self.final_h3 = torch.zeros(m, H, dtype=torch.bfloat16, device=device)
            self.final_stream = torch.zeros(m, H, dtype=torch.float32, device=device)
            self.final_qkv_out = torch.zeros(
                m, c.qkv_row, dtype=torch.bfloat16, device=device
            )
            self.final_cos_sin = torch.zeros(m, 64, dtype=torch.float32, device=device)
            self.final_loc = _zeros(m, torch.int32, device)
            self.final_kv = torch.zeros(
                1, c.kv_row_bytes // 2, dtype=torch.bfloat16, device=device
            )
            self.final_pt = torch.zeros(1, 1, dtype=torch.int64, device=device)
            self.final_kv_len = _zeros(1, torch.int32, device)

    def state(self) -> list[torch.Tensor]:
        own = [
            self.stream_out,
            self.h3,
            self.h3f,
            self.ws,
            self.counters,
            self.logits,
            self.idx,
            self.local,
            self.idx32,
            self.w,
            self.words,
            self.fctr,
            self.sm_epoch,
            self.next_layer_words,
            *([] if self.pooled_ready is None else [self.pooled_ready]),
        ]
        if self.tail:
            own += [
                self.final_h3,
                self.final_stream,
                self.final_qkv_out,
            ]
        return own


class Feeds:
    """The step's inputs and the feeds launch's outputs. The inputs: the window's positions and cache slots (the full
    pool's and the SWA pool's), the committed length, the request's slot (req_pool_indices), and three tables: the
    request slot tables (req_to_token, int32 [requests, width]), the full -> SWA slot map (int64, a negative index
    wraps: the trailing -1 sentinel) and the fp32 rope rows {cos | sin} per position. The page tables are one request's
    slot-table row as byte offsets (slot x row bytes). ``tree``: the kv_len word is followed by a token-tree window's
    table (``tree``), which the tree images' attention CTAs read through the kv_len pointer.
    """

    def __init__(
        self,
        cfg: Config,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        device,
        tree: bool = False,
    ):
        m, d = cfg.rows, cfg.head_dim
        assert req_to_token.dtype == torch.int32 and req_to_token.is_contiguous()
        assert full_to_swa.dtype == torch.int64 and cos_sin_cache.dtype == torch.float32
        assert cos_sin_cache.shape[1] == d
        self.req_to_token, self.full_to_swa = req_to_token, full_to_swa
        self.cos_sin_cache = cos_sin_cache
        self.width = req_to_token.shape[1]
        self.positions = _zeros(m, torch.int64, device)
        self.out_cache_loc = _zeros(m, torch.int64, device)
        self.swa_out_cache_loc = _zeros(m, torch.int64, device)
        self.seq_lens = _zeros(1, torch.int64, device)
        self.req_pool_indices = _zeros(1, torch.int64, device)
        self.cos_sin = torch.zeros(m, d, dtype=torch.float32, device=device)
        self.loc_full, self.loc_swa = _zeros(m, torch.int32, device), _zeros(
            m, torch.int32, device
        )
        kv = _zeros(1 + (TREE_WORDS if tree else 0), torch.int32, device)
        self.kv_len, self.tree = kv[:1], kv[1:] if tree else None
        self.pt_full = torch.zeros(1, self.width, dtype=torch.int64, device=device)
        self.pt_swa = torch.zeros(1, self.width, dtype=torch.int64, device=device)
        self.kv_row_bytes = cfg.kv_row_bytes

    @property
    def page(self) -> dict:
        return {"swa": self.pt_swa, "global": self.pt_full}

    @property
    def loc(self) -> dict:
        return {"swa": self.loc_swa, "global": self.loc_full}


@dataclass
class Logits:
    row: Symmetric  # [rows, vocab] fp32, then the s32 arrival word
    row_bytes: int
    out: torch.Tensor  # [rows, vocab] fp32
    state: torch.Tensor  # softcap's [generation, done]
    grid_done: torch.Tensor


class Buffers:
    def __init__(
        self,
        cfg: Config,
        consts: dict,
        device,
        group,
    ):
        """``consts``: image -> MkLayerConstants of the built layer libraries."""
        cfg.validate()
        c_fp4, c_bf16 = consts["swa"], consts["swa_bf16"]
        self.ar = AllReduce(c_fp4, cfg, device, group)
        self.moe_variants = {
            "fp4": MoeVariant(c_fp4, device, group),
            "bf16": MoeVariant(c_bf16, device, group),
        }
        self.attn_types = {
            "swa": AttnType(consts["swa"], cfg, device),
            "global": AttnType(consts["global"], cfg, device),
        }
        self.layers = [
            LayerState(consts[cfg.image(L)], device) for L in range(cfg.layers)
        ]
        # the entry: the window's token ids (identical on every rank), every rank's vocab shard, layer 0's inputs
        self.ids = _zeros(cfg.rows, torch.int64, device)
        self.embed = Symmetric(cfg.vocab_rank * cfg.hidden * 2, device, group)
        self.residual = torch.zeros(
            cfg.rows, cfg.hidden, dtype=torch.float32, device=device
        )
        self.normed = torch.zeros(
            cfg.rows, cfg.hidden, dtype=torch.bfloat16, device=device
        )
        row_bytes = cfg.rows * cfg.vocab * 4
        row = Symmetric(row_bytes + 16, device, group)
        assert (
            row.mc != 0
        ), "the lm_head gather writes through the TP group's multicast address"
        self.logits = Logits(
            row=row,
            row_bytes=row_bytes,
            out=torch.zeros(cfg.rows, cfg.vocab, dtype=torch.float32, device=device),
            state=_zeros(2, torch.int32, device),
            grid_done=_zeros(1, torch.int32, device),
        )
        self.device = device
        # reset's zero set as byte views (one foreach)
        self.state_bytes = [
            t.view(-1).view(torch.uint8)
            for t in (
                *(t for m in self.moe_variants.values() for t in m.state()),
                *(t for c in self.attn_types.values() for t in c.state()),
                *(t for st in self.layers for t in st.state()),
                self.logits.row.local,
                self.logits.out,
                self.logits.state,
                self.logits.grid_done,
            )
        ]
        published(group, device)

    def reset(self, group) -> None:
        """Every word and workspace back to its bind-time value; the inputs (weights, KV, feeds, ids, the vocab shards)
        keep theirs. Every rank is quiescent across it (synchronized, then the group's barrier, on both sides).
        """
        published(group, self.device)
        self.ar.reset()
        torch._foreach_zero_(self.state_bytes)
        for m in self.moe_variants.values():
            m.fill_sentinels()
        published(group, self.device)

    def embed_shard(self, cfg: Config) -> torch.Tensor:
        return self.embed.local.view(torch.bfloat16).view(cfg.vocab_rank, cfg.hidden)
