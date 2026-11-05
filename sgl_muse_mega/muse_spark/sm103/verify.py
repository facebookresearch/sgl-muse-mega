"""The verify step on the stack's kernels, bound once and captured as one CUDA graph:

    feeds | fused entry -> layer 0's qkv head -> 62 layers -> lm_head -> softcap

Every launch after the entry carries the PDL attribute: each kernel's prologue (weight streams, tensor-map prefetch,
the code prefetch, the expert prefetch, old-KV preloads) runs under its predecessor's tail and waits
(griddepcontrol.wait) only before reading what the predecessor wrote. The entry waits for the feeds to complete
(nothing before layer 0's qkv head reads the feeds, and the entry has no grid wait of its own).

A window is two bind values of a DFlash verify (``Window``): its live rows (8 for a block of 8 on the 16-row images:
rows 8-15 are pads) and the layers whose output residual the draft reads (``Stack.cap``); or, for a block of 8, the
8-row images (``M8Stack``); or a token tree (the ``_tree`` layer images, ``Stack.load_tree``).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ...common import host
from ..config import Config
from .artifacts import AUX_IMAGES, LAYER_IMAGES
from .buffers import Buffers, Feeds
from .weights import ModelWeights


@dataclass(frozen=True)
class Window:
    """The verify window's bind values: ``b_live`` live rows, the ``capture`` layers, and (a block of 8) ``native``:
    the 8-row images instead of the 16-row images at b_live 8, or ``tree``: the 16 rows are a token tree (the tree
    images read its table after the feeds' kv_len word)."""

    b_live: int = Config().rows
    capture: tuple[int, ...] = ()
    native: bool = False
    tree: bool = False


class LayerLibrary(host.Library):
    """A layer image's host library: its constants and the kernel's mangled-name prefix."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.dll.mk_layer_constants.argtypes = [
            ctypes.POINTER(host.STRUCTS["MkLayerConstants"])
        ]
        self.dll.mk_layer_kernel.restype = ctypes.c_char_p
        self.constants = host.STRUCTS["MkLayerConstants"]()
        self.check(
            self.dll.mk_layer_constants(ctypes.byref(self.constants)), "constants"
        )
        self.kernel = self.dll.mk_layer_kernel().decode()


def layer_libraries(paths: dict[str, Path], sfx: str = "") -> dict[str, LayerLibrary]:
    return {
        image: LayerLibrary(paths[f"libmk_layer_{image}{sfx}"])
        for image in LAYER_IMAGES
    }


def persist_grid(blocks: int, sms: int) -> int:
    """The balanced persistent grid: the largest divisor of the block count that fits the SMs (every CTA walks the
    same number of blocks)."""
    return max(d for d in range(1, sms + 1) if blocks % d == 0)


@dataclass(frozen=True)
class Aux:
    """The verify graph's launches around the layers at one window height: the fused entry, layer 0's qkv head, the
    lm_head gather and its softcap."""

    entry: host.Launch
    qkv: host.Launch
    lm_head: host.Launch
    softcap: host.Launch


class Stack:
    def __init__(
        self,
        cfg: Config,
        paths: dict[str, Path],
        libs: dict[str, LayerLibrary],
        bufs: Buffers,
        weights: ModelWeights,
        rank: int,
        device,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        feeds: Feeds,
    ):
        """``kv``: every layer's (K, V) pool, [slots, kv heads x 64] bf16 rows (a layer's slots in its attention
        type's pool); ``feeds``: the step's inputs and the feeds' outputs."""
        self.cfg, self.bufs, self.w, self.rank, self.libs = (
            cfg,
            bufs,
            weights,
            rank,
            libs,
        )
        self.kv, self.f = kv, feeds
        self.mods = {
            image: lib.load(paths[f"layer_{image}"], lib.kernel)
            for image, lib in libs.items()
        }
        self.tree_mods: dict[str, host.Module] = {}
        lib = host.Library(paths["libmk_aux"])
        mods = {n: lib.load(paths[n]) for n in AUX_IMAGES}
        self.feeds = self._bind_feeds(lib, mods["window_feeds"])
        self.aux = self._bind_aux(lib, mods, cfg, device)
        self._layers: dict[tuple, list[host.Launch]] = {}
        self._caps: dict[int, torch.Tensor] = {}

    # ---- binding ----

    def _bind_feeds(self, a: host.Library, module: host.Module) -> host.Launch:
        cfg, f = self.cfg, self.f
        return a.bind(
            a.dll.mk_feeds_bind,
            module,
            host.make(
                "MkFeedsBind",
                positions=f.positions,
                cos_sin_cache=f.cos_sin_cache,
                cos_sin_rows=f.cos_sin_cache.shape[0],
                out_cache_loc=f.out_cache_loc,
                swa_out_cache_loc=f.swa_out_cache_loc,
                seq_lens=f.seq_lens,
                req_pool_indices=f.req_pool_indices,
                req_to_token=f.req_to_token,
                r2t_rows=f.req_to_token.shape[0],
                r2t_width=f.width,
                full_to_swa=f.full_to_swa,
                swa_map_len=f.full_to_swa.shape[0],
                cos_sin=f.cos_sin,
                loc_full=f.loc_full,
                loc_swa=f.loc_swa,
                kv_len=f.kv_len,
                pt_full=f.pt_full,
                pt_swa=f.pt_swa,
                rows=cfg.rows,
                head_dim=cfg.head_dim,
                kv_row_bytes=f.kv_row_bytes,
            ),
        )

    def _bind_aux(self, a: host.Library, mods: dict, cfg: Config, device) -> Aux:
        """The launches around the layers at ``cfg.rows`` rows (``a`` and ``mods``: that height's library and
        images)."""
        b, f = self.bufs, self.f
        entry = a.bind(
            a.dll.mk_entry_bind,
            mods["entry_fused"],
            host.make(
                "MkEntryBind",
                ids=b.ids,
                embed_peer=b.embed.ptrs,
                shard_rows=cfg.vocab_rank,
                pre_w=self.w.entry_pre_w,
                residual=b.residual,
                normed=b.normed,
                rows=cfg.rows,
                hidden=cfg.hidden,
                eps_embed=cfg.rms_eps,
                eps_pre=cfg.rms_eps,
            ),
        )
        c0 = cfg.attn_type(0)
        k0, v0 = self.kv[0]
        qkv = a.bind(
            a.dll.mk_qkv_bind,
            mods["qkv_head"],
            host.make(
                "MkQkvBind",
                w=self.w.layers[0].qkv,
                x=b.normed,
                out=b.attn_types[c0].qkv_out,
                cos_sin=f.cos_sin if c0 == "swa" else b.attn_types[c0].nope_cs,
                cache_loc=f.loc[c0],
                k_cache=k0,
                v_cache=v0,
                temp_scale=None,
                q_scale=cfg.q_scale,
                eps=cfg.rms_eps,
                hidden=cfg.hidden,
            ),
        )
        lg, last = b.logits, b.layers[cfg.layers - 1]
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        lm_head = a.bind(
            a.dll.mk_lm_head_bind,
            mods["lm_head"],
            host.make(
                "MkLmHeadBind",
                w=self.w.lm_head,
                x=last.final_h3,
                mc_base=lg.row.mc,
                arrive_off=lg.row_bytes,
                grid_done=lg.grid_done,
                v_total=cfg.vocab,
                col0=cfg.vocab_rank * self.rank,
                n_out=cfg.vocab_rank,
                n_pad=cfg.lm_head_pad,
                grid=persist_grid(cfg.lm_head_pad // 64, sms),
                hidden=cfg.hidden,
            ),
        )
        softcap = a.bind(
            a.dll.mk_softcap_bind,
            mods["softcap"],
            host.make(
                "MkSoftcapBind",
                x=lg.row.local,
                out=lg.out,
                numel=cfg.rows * cfg.vocab,
                scale=cfg.output_multiplier,
                cap=cfg.softcap,
                arrive=lg.row.local.data_ptr() + lg.row_bytes,
                state=lg.state,
                per_launch=cfg.tp,
            ),
        )
        return Aux(entry, qkv, lm_head, softcap)

    def cap(self, L: int) -> torch.Tensor:
        """Capture layer L's output residual (fp32 [rows, hidden], allocated at its first bind): a capturing window
        binds it as L's output (its next-layer phase writes it) and L + 1's residual input in place of its attention
        type's shared buffer, so it still holds L's output after the step, when a later layer of the same attention type
        has overwritten the shared buffer.
        """
        assert (
            0 <= L < self.cfg.layers - 1
        ), f"capture layer {L}: the last layer ends at the final norm"
        if L not in self._caps:
            self._caps[L] = torch.zeros(
                self.cfg.rows,
                self.cfg.hidden,
                dtype=torch.float32,
                device=self.bufs.residual.device,
            )
        return self._caps[L]

    def layer_bind(self, L: int, window: Window = Window()):
        """Layer L's operands (an MkLayerBind): its own weights, KV pool and words; its attention type's arena and
        residual (or a capture buffer); its MoE variant's state; the successor's qkv weight, pool and attention-type
        buffers (the next-layer phase).
        """
        cfg, b, f = self.cfg, self.bufs, self.f
        c = self.libs[cfg.image(L)].constants
        cls, bf16 = cfg.attn_type(L), cfg.is_bf16(L)
        m, st, lw = (
            b.moe_variants["bf16" if bf16 else "fp4"],
            b.layers[L],
            self.w.layers[L],
        )
        k, v = self.kv[L]
        if c.tail:
            next_layer = dict(
                w_qkv=lw.qkv,
                h3q=st.final_h3,
                qkv_out=st.final_qkv_out,
                cos_sin=st.final_cos_sin,
                cache_loc=st.final_loc,
                next_k_cache=st.final_kv,
                next_v_cache=st.final_kv,
                q_scale=1.0,
                c_stream_out=st.final_stream,
                nk_page_table=st.final_pt,
                nk_kv_len=st.final_kv_len,
                nk_page_stride=1,
                nk_splits=0,
            )
        else:
            sc = cfg.attn_type(L + 1)
            nk, nv = self.kv[L + 1]
            succ = b.attn_types[sc]
            next_layer = dict(
                w_qkv=self.w.layers[L + 1].qkv,
                h3q=succ.h3,
                qkv_out=succ.qkv_out,
                cos_sin=f.cos_sin if sc == "swa" else succ.nope_cs,
                cache_loc=f.loc[sc],
                next_k_cache=nk,
                next_v_cache=nv,
                q_scale=cfg.q_scale,
                c_stream_out=self.cap(L) if L in window.capture else succ.stream_out,
                nk_page_table=f.pt_swa,
                nk_kv_len=f.kv_len,
                nk_page_stride=f.width,
                nk_splits=c.nextkv_splits if sc == "swa" else 0,
            )
        no_peers = [0] * cfg.tp
        if L == 0:
            stream_in = b.residual
        elif L - 1 in window.capture:
            stream_in = self.cap(L - 1)
        else:
            stream_in = b.attn_types[cls].stream_out
        return host.make(
            "MkLayerBind",
            w_o=lw.o_proj,
            attn_arena=b.attn_types[cls].arena,
            peer_bases=b.ar.peer_bases,
            rank=self.rank,
            plane_stride=b.ar.plane_stride,
            stream_in=stream_in,
            alpha=lw.post_alpha,
            beta=lw.post_beta,
            prenorm_w=lw.post_pre_w,
            stream_out=st.stream_out,
            h3=st.h3,
            h3f=st.h3f,
            w_p=lw.pooler,
            w_r=lw.router,
            logits=st.logits,
            ws=st.ws,
            counters=st.counters,
            router_mult=cfg.output_multiplier,
            expert_bias=lw.expert_bias,
            num_local=cfg.local_experts,
            topk_idx=st.idx,
            topk_idx_local=st.local,
            topk_w=st.w,
            topk_idx32=st.idx32,
            route_scale=cfg.route_scale,
            gamma=lw.pre_gamma,
            eps=cfg.rms_eps,
            q=m.q,
            sf=m.sf,
            x_scale=m.x_scale,
            rt_counts=m.counts,
            rt_offsets=m.offsets,
            rt_slot_tok=m.slot_tok,
            rt_slot_k=m.slot_k,
            rt_tok_slot=m.tok_slot,
            rt_tile_e=m.tile_e,
            rt_tile_r0=m.tile_r0,
            rt_n_tiles=m.n_tiles,
            words=st.words,
            pooled_ready=st.pooled_ready,
            w1=lw.w1,
            w1_sf_blk=lw.w1_sf_blk,
            w2=lw.w2,
            w2_sf=lw.w2_sf,
            g1_alphas=lw.g1_alphas,
            g1_scale_c=lw.g1_scale_c,
            g2_alphas=lw.g2_alphas,
            gamma_p1=lw.gamma_p1,
            z=m.views["z"],
            z_sf=m.views["z_sf"],
            z_ex=None if bf16 else m.views["z_ex"],
            zb=m.views["zb"],
            zb_par_slots=m.zb_par_slots,
            arrive=m.views["arrive"],
            z_peer=m.peers["z"],
            zsf_peer=m.peers["z_sf"],
            zex_peer=no_peers if bf16 else m.peers["z_ex"],
            zb_peer=m.peers["zb"],
            arrive_peer=m.peers["arrive"],
            gsync=m.gsync,
            epoch=m.epoch,
            y=m.y,
            attn_fctr=st.fctr,
            attn_sm_epoch=st.sm_epoch,
            k_cache=k,
            v_cache=v,
            kv_rows=k.shape[0],
            page_table=f.page[cls],
            kv_len=f.kv_len,
            page_stride=f.width,
            prefetch_experts=int(not bf16),
            w_up=lw.unpooler,
            qkv_eps=cfg.rms_eps,
            temp_scale=None,
            c_alpha=lw.cross_alpha,
            c_beta=lw.cross_beta,
            c_pre_w=lw.cross_pre_w,
            c_post_w=lw.cross_post_w,
            next_layer_words=st.next_layer_words,
            nk_window=cfg.swa_window,
            b_live=window.b_live,
            **next_layer,
        )

    def load_tree(self, paths: dict[str, Path]) -> None:
        """The token-tree window's layer images (``artifacts.dflash_artifacts(tree=True)``), bound through the same
        libraries."""
        assert self.f.tree is not None, "the feeds hold no tree table"
        self.tree_mods = {
            image: lib.load(paths[f"layer_{image}_tree"], lib.kernel)
            for image, lib in self.libs.items()
        }

    def layers(self, window: Window) -> list[host.Launch]:
        key = (window.b_live, window.capture, window.tree)
        if key not in self._layers:
            assert not window.tree or (
                self.tree_mods and window.b_live == self.cfg.rows
            ), window
            mods = self.tree_mods if window.tree else self.mods
            out = []
            for L in range(self.cfg.layers):
                image = self.cfg.image(L)
                lib = self.libs[image]
                out.append(
                    lib.bind(
                        lib.dll.mk_layer_bind,
                        mods[image],
                        self.layer_bind(L, window),
                    )
                )
            self._layers[key] = out
        return self._layers[key]

    # ---- running ----

    def aux_for(self, window: Window) -> Aux:
        return self.aux

    def forward(self, window: Window, main: torch.cuda.Stream) -> None:
        """One step's launches on ``main``: eager, or recorded under capture."""
        s = main.cuda_stream
        layers, aux = self.layers(window), self.aux_for(window)
        self.feeds(s, True)
        aux.entry(s, False)
        aux.qkv(s, True)
        for launch in layers:
            launch(s, True)
        aux.lm_head(s, True)
        aux.softcap(s, True)

    def capture(self, window: Window, main: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
        self.layers(window)  # bind outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main):
            self.forward(window, main)
        return g


# ---- the 8-row window (a DFlash block of 8) ----

LIVE = 8  # the native window's rows
# the aux kernels with an 8-row image of their own; the feeds, entry and softcap take their rows at bind
AUX8 = ("qkv_head", "lm_head")


class M8Stack(Stack):
    """The stack plus the 8-row image set, which a ``Window(native=True)`` selects."""

    def __init__(self, cfg, paths, libs, bufs, weights, rank, device, kv, feeds):
        super().__init__(cfg, paths, libs, bufs, weights, rank, device, kv, feeds)
        self.libs8 = layer_libraries(paths, "_m8")
        self.mods8 = {
            image: lib.load(paths[f"layer_{image}_m8"], lib.kernel)
            for image, lib in self.libs8.items()
        }
        for image, lib in self.libs8.items():
            c = lib.constants
            assert c.rows == LIVE, (image, c.rows)
            assert c.nextkv_splits == libs[image].constants.nextkv_splits
        # the launches around the layers once more at 8 rows; the feeds stay the 16-row layout's
        lib = host.Library(paths["libmk_aux_m8"])
        mods = {
            n: lib.load(paths[f"{n}_m8" if n in AUX8 else n])
            for n in AUX_IMAGES
            if n != "window_feeds"
        }
        self.aux8 = self._bind_aux(lib, mods, replace(cfg, draft=LIVE - 1), device)

    def aux_for(self, window: Window) -> Aux:
        return self.aux8 if window.native else self.aux

    def layers(self, window: Window) -> list[host.Launch]:
        if not window.native:
            return super().layers(window)
        assert window.b_live == LIVE, window
        key = ("native", window.capture)
        if key not in self._layers:
            out = []
            for L in range(self.cfg.layers):
                image = self.cfg.image(L)
                lib, mod = self.libs8[image], self.mods8[image]
                out.append(
                    lib.bind(lib.dll.mk_layer_bind, mod, self.layer_bind(L, window))
                )
            self._layers[key] = out
        return self._layers[key]


def share_gather_rows(bufs: Buffers, c16, c8) -> None:
    """The 8-row images' gathered rows join the rows the reset prefills: at their own offsets in the all-reduce slot
    (TP8), disjoint from the 16-row images' gathered rows and from either set's all-reduce message; or (the K-window
    variant, TP4) inside the 16-row images' rows, which the reset prefills already.
    """
    rows8 = (
        (c8.unpool_ag_off, c8.unpool_ag_off + c8.rows * c8.hidden),
        (c8.pool_ag_off, c8.pool_ag_off + c8.rows * c8.moe_hidden),
    )
    message = (0, c16.rows * c16.hidden)
    new = []
    for lo, hi in rows8:
        if any(a <= lo and hi <= b for a, b in bufs.ar.rows):
            continue
        for a, b in (*bufs.ar.rows, message):
            assert hi <= a or b <= lo, ((lo, hi), (a, b))
        assert hi <= bufs.ar.plane_stride, (hi, bufs.ar.plane_stride)
        new.append((lo, hi))
    bufs.ar.rows = (*bufs.ar.rows, *new)
    bufs.ar.reset()
