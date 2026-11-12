"""One DFlash2 iteration at bs1 on the stack, captured as ONE CUDA graph:

    entry -> qkv -> layer x 4 -> tail -> draft_head -> fused entry -> qkv_head -> 62 verify layers -> verify_accept

Every launch carries PDL and every per-iteration value is a device word. verify_accept writes the step words (P, the
anchor, the accept, the commit), the next verify window's feeds and the draft's (rope rows, ring slots, the attention's
kv_len); the head writes the verify window's ids; the verify's capture layers leave the draft's features in place
(``Window.capture``). The host only replays.

The draft is one launch per draft layer (layer images layer / tail: the attention on the first-dispatched 9 clusters,
its split-KV combine leading the layer's queue; each conv's side 0 a prepare unit: a GEMM unit with the conv's prepare
in its epilogue), after the entry and layer 0's own qkv launch (the dense qkv image), then the draft head (draft_head:
an e4m3 pre-score of the shard, exact bf16 rescoring of the survivors that cover each rank's pre-score top-PRESCORE_K).
The verify is the stack with b_live = B and the capture pointers; at B = 8 it runs on the 8-row images (``verify``;
``verify_padded()``: the 16-row images at b_live 8, whose live rows are bitwise the same).

``Context`` holds one rank's target stack, draft weights and loaded images; ``Iteration`` binds one block size (the
draft's buffers, its launches, the verify and its accept) and starts a run (``start``).

A token-tree iteration (``tree``, block 16; kernels/dflash/tree.cuh) verifies a tree of 15 drafts grown from the head's
lattice instead of a linear chain: the head's tree image writes the window's ids, the tree table and the rows' rope rows
(P + depth), the verify runs on the tree layer images, verify_accept's tree image accepts the matching path, and
tree_compact moves the path's K / V and captured rows to the slots of rows 1 .. accept (one launch more).

A sampled step (``sample``: the launches of ``launches(sample=True)``, kernels/dflash/sample.cuh) runs the same
iteration with the sampled head image, and ends the verify with the verify graph's lm_head and softcap and then
sample_accept instead of verify_accept: the sampling words (``sampling``) carry the request's temperature, top-k, top-p
and seed.

``pool_sync`` binds draft_kv_sync (kernels/dflash/kv_sync.cu) over a draft KV pool: the rows of the committed positions
from the draft's rings into the pool, as the iteration's last launch or after the draft alone.
"""

from __future__ import annotations

import dataclasses

import torch

from ....common import host
from ...config import Config
from ..artifacts import DFLASH_BLOCKS, DFLASH_LAYER_IMAGES, build
from ..buffers import published
from ..verify import LIVE, Stack, Window
from . import head, layer
from .head import N_SAMP, SAMP, WORDS
from .verify_accept import SampleAccept, VerifyAccept
from .weights import (
    NODE_RANKS,
    ConvWeights,
    DraftConfig,
    DraftWeights,
    conv_proj_shard,
    pack_conv_proj0_tiles,
    pack_conv_proj_packets,
    pack_gate_up,
    pack_qkv,
    pack_tiles,
    rope_cos_sin,
    side1_packet_order,
)

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
PRESCORE_K = 32  # the survivor top-K per rank


def capture_layers(cfg: Config, layers) -> tuple[int, ...]:
    """A capture list as a bind value: ascending, distinct, each layer's successor reads it (never the last layer)."""
    t = tuple(int(L) for L in layers)
    assert list(t) == sorted(set(t)), f"capture layers must ascend without repeats: {t}"
    assert all(
        0 <= L < cfg.layers - 1 for L in t
    ), f"capture layers in [0, {cfg.layers - 2}]: {t}"
    return t


@dataclasses.dataclass
class Layer:
    """One draft layer on the device in the kernels' layouts (``weights``)."""

    qkv: torch.Tensor  # [qkv cols, H] (pack_qkv)
    o: torch.Tensor  # [H, q heads x D]: the rank's input columns
    gate_up: torch.Tensor  # [2 I, H] (pack_gate_up)
    down: torch.Tensor  # [H, I]
    att_base: torch.Tensor  # [2 sides, 2 taps, H]
    mlp_base: torch.Tensor
    # the rank's conv coefficient-projection rows (conv_proj_shard of pack_conv_proj_packets)
    att_w: torch.Tensor
    mlp_w: torch.Tensor
    # side 0's full 16 blocks [coef_row, H] (the prepare units of the images with the attention)
    att_w0: torch.Tensor
    mlp_w0: torch.Tensor
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor


class DraftModel:
    """The draft on this rank: device copies of ``ref`` (the rank's shard, ``weights.load_shard``) in the kernels'
    layouts (``ref`` itself is not kept)."""

    def __init__(self, ref: DraftWeights, rank: int, device, ranks: int):
        cfg = self.cfg = DraftConfig()
        dev = device
        self.layers = []
        for ls in ref.layers:

            def conv_proj(conv: ConvWeights) -> tuple[torch.Tensor, torch.Tensor]:
                packed = pack_conv_proj_packets(conv.proj, cfg.conv_taps)
                side0 = packed[: packed.shape[0] // 2]
                return (
                    conv_proj_shard(packed, rank).contiguous().to(dev),
                    side0.contiguous().to(dev),
                )

            att_w, att_w0 = conv_proj(ls.attn_conv)
            mlp_w, mlp_w0 = conv_proj(ls.mlp_conv)
            self.layers.append(
                Layer(
                    qkv=pack_qkv(ls.qkv, cfg, ranks).to(dev),
                    o=ls.o.contiguous().to(dev),
                    gate_up=pack_gate_up(ls.gate_up).to(dev),
                    down=ls.down.contiguous().to(dev),
                    att_base=ls.attn_conv.base.contiguous().to(dev),
                    mlp_base=ls.mlp_conv.base.contiguous().to(dev),
                    att_w=att_w,
                    mlp_w=mlp_w,
                    att_w0=att_w0,
                    mlp_w0=mlp_w0,
                    input_norm=ls.input_norm.to(dev),
                    post_norm=ls.post_norm.to(dev),
                    q_norm=ls.q_norm.to(dev),
                    k_norm=ls.k_norm.to(dev),
                )
            )
        self.norm = ref.norm.to(dev)
        self.pred, self.succ = ref.pred_codebook.to(dev), ref.succ_codebook.to(dev)
        # the rank's W_P band of the head weight
        hp = cfg.selector_rank // ranks
        self.wp_rows = ref.hidden_proj[rank * hp : (rank + 1) * hp].to(dev)
        # the entry: fc output-sharded; layer 0's attention conv with side 1 in the queue's packet order
        proj0q = side1_packet_order(ref.layers[0].attn_conv.proj, cfg.conv_taps)
        self.w_fc = pack_tiles(ref.fc).to(dev)
        self.w_conv_proj = pack_conv_proj0_tiles(proj0q, rank, ranks).to(dev)
        self.entry_w = head.EntryWeights(
            hidden_norm=ref.hidden_norm.to(dev),
            in_norm=self.layers[0].input_norm,
            conv_base=self.layers[0].att_base,
        )


# ---- one rank's stack ----


def head_shape(
    cfg: Config, dcfg: DraftConfig, rank: int, b: int, device
) -> head.HeadShape:
    """The draft head's shape on this rank at block size b (the vocab shard, the W_P band, the persistent grid)."""
    return head.HeadShape(
        hidden=cfg.hidden,
        vocab_rank=cfg.vocab_rank,
        rank=rank,
        ranks=cfg.tp,
        gamma=b - 1,
        sms=torch.cuda.get_device_properties(device).multi_processor_count,
        mult=dcfg.output_multiplier,
        cap=dcfg.softcap,
    )


def draft_rope_cache(rows: int, cfg: DraftConfig) -> torch.Tensor:
    """The draft's fp32 rope rows {cos[D / 2] | sin[D / 2]} per position (theta 5e5), as the accept gathers them."""
    cos, sin = rope_cos_sin(torch.arange(rows), cfg.head_dim, cfg.rope_theta)
    return torch.cat([cos, sin], 1).contiguous()


class DraftDevice:
    """The draft's device tensors on one rank, allocated with the target's weights: the draft in the kernels' layouts
    (``DraftModel``), the head weight W = [the target lm_head's vocab shard | the draft's W_P band | zero pad to the
    256-row schedule] (the head reads both in one pass) and the pre-score's e4m3 copy of the vocab shard.
    """

    def __init__(
        self, draft: DraftWeights, lm_head: torch.Tensor, cfg: Config, rank: int, device
    ):
        self.model = DraftModel(draft, rank, device, cfg.tp)
        # the W_P band and the pad rows do not depend on the block size
        s = head_shape(cfg, self.model.cfg, rank, DFLASH_BLOCKS[0], device)
        self.hw = torch.zeros(s.n_pad, cfg.hidden, dtype=BF16, device=device)
        self.hw[: cfg.vocab_rank] = lm_head
        self.hw[cfg.vocab_rank : s.shard_cols] = self.model.wp_rows
        self.wq, self.w_scale = head.quantize_rows(self.hw[: cfg.vocab_rank])


class Context:
    """One rank: the target's verify stack (bound to its KV pools and feeds), the draft's device tensors
    (``DraftDevice``) and every DFlash image loaded (at the stack's world); ``tree``: also the token-tree window's.
    """

    def __init__(self, stack: Stack, draft: DraftDevice, group, tree: bool = False):
        self.stack, self.group = stack, group
        self.cfg = cfg = stack.cfg
        self.dcfg = DraftConfig()
        assert (
            self.dcfg.hidden == cfg.hidden
        ), "the draft borrows the target's embedding and lm_head"
        self.bufs, self.weights, self.feeds = stack.bufs, stack.w, stack.f
        self.rank, self.device = stack.rank, stack.bufs.device
        self.paths = paths = build(dflash=True, world=cfg.world, tree=tree)
        self.model, self.hw = draft.model, draft.hw
        self.wq, self.w_scale = draft.wq, draft.w_scale
        self.shape = {
            b: head_shape(cfg, self.dcfg, self.rank, b, self.device)
            for b in DFLASH_BLOCKS
        }
        self.capture = capture_layers(cfg, self.dcfg.capture)
        # the draft's rope rows for every position a slot-table row reaches
        self.dcache = draft_rope_cache(self.feeds.width, self.dcfg).to(self.device)
        self.dense = layer.Dense(paths)
        # the draft attention's geometry (the layer images' attention)
        attn_lib = host.Library(paths["libmk_dflash_attn"])
        self.attn_c = {b: layer.attn_constants(attn_lib, b) for b in DFLASH_BLOCKS}
        self.ring = self.attn_c[8].ring_rows
        self.layer_images = {
            b: {n: layer.Image(paths, n, b) for n in DFLASH_LAYER_IMAGES}
            for b in DFLASH_BLOCKS
        }
        self.accept_lib = host.Library(paths["libmk_dflash_verify_accept"])
        self.accept_mod = self.accept_lib.load(paths["dflash_verify_accept"])
        self.accept_sample_mod = self.accept_lib.load(
            paths["dflash_verify_accept_sample"]
        )
        self.kv_sync_mod = self.accept_lib.load(paths["dflash_kv_sync"])
        self.draft_head_lib = head.DraftHeadLib(paths["libmk_dflash_draft_head"])
        self.draft_head_mod = self.draft_head_lib.load(paths["dflash_draft_head"])
        self.draft_head_sample_mod = self.draft_head_lib.load(
            paths["dflash_draft_head_sample"]
        )
        self.tree = tree
        if tree:
            stack.load_tree(paths)
            self.draft_head_tree_mod = self.draft_head_lib.load(
                paths["dflash_draft_head_tree"]
            )
            self.accept_tree_mod = self.accept_lib.load(
                paths["dflash_verify_accept_tree"]
            )
        self.diag = layer.diag_words()
        self.main = torch.cuda.Stream(self.device)

    def published(self) -> None:
        published(self.group, self.device)


# ---- one block size ----


@dataclasses.dataclass
class LayerBuffers:
    """One draft layer's activations and its KV (the residual carries over: r_c of layer l is r_a of layer l + 1)."""

    # [B, H] the residual in (layer 0: the block's embedding rows, the entry's)
    r_a: torch.Tensor
    r_b: torch.Tensor  # after the attention sublayer
    r_c: torch.Tensor  # after the MLP sublayer
    h2: torch.Tensor  # [16, H] the post-attention norm (mlp_conv_proj's x)
    x_gu: torch.Tensor  # [16, H] gate_up's x (the MLP conv's prepared rows)
    act: torch.Tensor  # [16, I] SwiGLU
    # [16, H] the next layer's input norm; the last layer: y [17, H] (the head reads rows 1..)
    h: torch.Tensor
    x_attn: (
        torch.Tensor
    )  # [16, q heads x D] the attention combine's output (o_proj's x)
    # [2B, qkv cols] this layer's qkv rows (the attention's q: rows < B)
    qout: torch.Tensor
    # [ring, kv heads x D] the draft KV ring (slot = position mod ring)
    k_ring: torch.Tensor
    v_ring: torch.Tensor
    ws: torch.Tensor  # the attention's partials


class Iteration:
    """Block size B on one rank: the draft's state, its launches (``draft``), the verify with verify_accept
    (``verify``) and the run state (``start``; ``reset`` between the tests' runs). ``override`` (int32 [n];
    verify_accept's one test hook, for the tests and the benchmark on synthetic target weights, which never accept on
    their own; a server passes one only for SGLANG_SIMULATE_ACC_LEN): verify_accept takes the accept from it, indexed by
    the iteration word mod n, instead of the greedy prefix rule (on a tree, the accepted leaf row). ``tree``: the
    token-tree iteration (block 16, the Context's tree images). ``sample``: also the sampled step's launches (a linear
    chain; ``block``: SGLang's block verification, else token-wise).
    """

    def __init__(
        self,
        b: int,
        cx: Context,
        override: torch.Tensor | None = None,
        tree: bool = False,
        sample: bool = False,
        block: bool = True,
    ):
        assert b in DFLASH_BLOCKS
        assert not tree or (
            cx.tree and b == cx.cfg.rows
        ), "a tree iteration: block 16, a tree Context"
        assert not (tree and sample), "a sampled step walks the linear chain"
        self.b, self.cx, self.tree, self.sample = b, cx, tree, sample
        dev, rank, cfg, dcfg = cx.device, cx.rank, cx.cfg, cx.dcfg
        hidden, inter, hd = dcfg.hidden, dcfg.inter // cfg.tp, dcfg.head_dim
        q_heads, kv_heads = layer.qkv_heads(dcfg, cfg.tp)
        self.qkv_cols, self.tickets = layer.qkv_cols(dcfg, cfg.tp), layer.qkv_tickets(
            dcfg, cfg.tp
        )
        # every protocol word and region with its bind-time fill (reset's byte views of the zero-filled ones)
        self.proto: list[tuple[torch.Tensor, int]] = []
        self._zero_bytes: list[torch.Tensor] = []

        def zb(*s):
            return torch.zeros(*s, dtype=BF16, device=dev)

        # the step words and the draft's feeds: verify_accept writes them, the next iteration reads them
        self.words = torch.zeros(head.N_WORDS, dtype=torch.int64, device=dev)
        self.samp = torch.zeros(N_SAMP, dtype=torch.int64, device=dev)
        # int32 view of the commit word's low half (little endian)
        self.commit = self.words.data_ptr() + 8 * WORDS["COMMIT"]
        self.dcos_sin = torch.zeros(2 * b, hd, dtype=F32, device=dev)
        self.dslot = torch.zeros(2 * b, dtype=I32, device=dev)
        self.dkv_len = torch.zeros(1, dtype=I32, device=dev)
        self.override = None if override is None else override.to(dev)
        coef_row = dcfg.conv_taps * dcfg.conv_groups
        self.st = layer.State(b, rank, dev, cx.group, cfg.tp, hidden, coef_row)
        assert all(img.c.coef_row == coef_row for img in cx.layer_images[b].values())
        # every qkv's x: [block rows | context rows (the entry's ctx_n)]
        self.x_qkv = zb(max(16, 2 * b), hidden)
        self.y = zb(head.ROWS + 1, hidden)
        # the draft head's x rows: y rows 1 .. B - 1 in e4m3, written by the last draft layer's final norm
        self.xq = torch.zeros(b - 1, hidden, dtype=torch.uint8, device=dev)
        ci = cx.attn_c[b]
        self.lb: list[LayerBuffers] = []
        r_a = zb(b, hidden)
        for i in range(dcfg.layers):
            r_c = zb(b, hidden)
            self.lb.append(
                LayerBuffers(
                    r_a=r_a,
                    r_b=zb(b, hidden),
                    r_c=r_c,
                    h2=zb(16, hidden),
                    x_gu=zb(16, hidden),
                    act=zb(16, inter),
                    h=self.y if i == dcfg.layers - 1 else zb(16, hidden),
                    x_attn=zb(16, q_heads * hd),
                    qout=zb(2 * b, self.qkv_cols),
                    k_ring=zb(cx.ring, kv_heads * hd),
                    v_ring=zb(cx.ring, kv_heads * hd),
                    ws=torch.zeros(ci.attn_ctas * ci.ws_row, dtype=F32, device=dev),
                )
            )
            r_a = r_c
        self.head_out = head.HeadOut.alloc(b - 1, dev)
        self.head_out.verify_ids = cx.bufs.ids
        self._proto_tensor(self.st.planes.local, 0)
        self._proto_tensor(self.st.coef.local, 0xFF)
        self._bind_shared()
        self._bind_draft()
        self._bind_verify(block)

    # ---- protocol state ----

    def _proto_tensor(self, t: torch.Tensor, fill: int) -> torch.Tensor:
        self.proto.append((t, fill))
        return t

    def _zeros(self, *shape, dtype=I32) -> torch.Tensor:
        return self._proto_tensor(
            torch.zeros(*shape, dtype=dtype, device=self.cx.device), 0
        )

    def _words(self, image: layer.Image) -> layer.Words:
        w = layer.Words(image.c.grid, self.cx.device)
        self._proto_tensor(w.t, 0)
        return w

    def reset(self) -> None:
        """Every protocol word and cross-rank region back to its bind-time value (between the tests' runs; the caller
        syncs)."""
        if not self._zero_bytes:
            self._zero_bytes = [
                t.view(-1).view(torch.uint8) for t, f in self.proto if f == 0
            ]
        torch._foreach_zero_(self._zero_bytes)
        for t, fill in self.proto:
            if fill != 0:
                t.view(torch.uint8).fill_(fill)

    # ---- the entry and layer 0's qkv ----

    def _bind_shared(self) -> None:
        cx, b, m, lb0 = self.cx, self.b, self.cx.model, self.lb[0]
        self.entry = head.Entry(
            b,
            cx.paths,
            m.w_fc,
            m.w_conv_proj,
            m.entry_w,
            [cx.stack.cap(L) for L in cx.capture],
            self.words[WORDS["ANCHOR"] : WORDS["ANCHOR"] + 1],
            cx.bufs.embed.ptrs,
            cx.cfg.vocab_rank,
            cx.rank,
            cx.group,
            cx.dcfg.rms_eps,
            out=dict(
                ctx_n=self.x_qkv[b : 2 * b],
                xq=self.x_qkv[:b],
                resid=lb0.r_a,
                c_att1=self.st.rows("att1"),
            ),
        )
        self._proto_tensor(self.entry.sym.local, 0xFF)
        self._proto_tensor(self.entry.ws, 0)
        self._proto_tensor(self.entry.ctr, 0)
        self.qkv0 = self._qkv(0, self._zeros(self.tickets))

    def bind_constants(self) -> None:
        """The entry's mask rows (two rank-synchronized launches with anchor = mask); once, before the first run."""

        def step():
            self.cx.published()
            self.entry(self.cx.main.cuda_stream, False)
            self.cx.published()

        self.entry.bind_constants(self.cx.dcfg.mask_id, step)

    def _qkv(self, i: int, tickets: torch.Tensor) -> host.Launch:
        """Draft layer i's qkv + hd128 rowops as its own launch (the dense qkv image)."""
        lw, lb = self.cx.model.layers[i], self.lb[i]
        return self.cx.dense.qkv(
            self.b,
            w=lw.qkv,
            x=self.x_qkv,
            out=lb.qout,
            tickets=tickets,
            cos_sin=self.dcos_sin,
            slot=self.dslot,
            commit=self.commit,
            q_norm=lw.q_norm,
            k_norm=lw.k_norm,
            k_ring=lb.k_ring,
            v_ring=lb.v_ring,
            hidden=self.cx.dcfg.hidden,
            eps=self.cx.dcfg.rms_eps,
        )

    def _gathers(self, v: dict, image: layer.Image, slots: tuple[str, ...]) -> None:
        """The coefficient gathers' destination rows (the node's 4 ranks) and this rank's block of each side (the
        slots the image's program gathers; slot 2, side 1 of mlp_conv_proj, covers one node quarter in every such
        program).
        """
        for s, slot in enumerate(slots):
            if (image.c.gat_mask >> s) & 1:
                v["gat_dst"][4 * s : 4 * s + 4] = self.st.gather_dst(slot)
                v["gat_block0"][s] = (self.cx.rank % NODE_RANKS) * image.c.gemm_blocks[
                    2
                ]

    # ---- the draft: one launch per draft layer ----

    def _layer(self, i: int) -> host.Launch:
        """Draft layer i as one launch."""
        cx, b, m, st = self.cx, self.b, self.cx.model, self.st
        tail = i == cx.dcfg.layers - 1
        name = "tail" if tail else "layer"
        img = cx.layer_images[b][name]
        c, v, words = img.c, layer.empty_bind(), self._words(img)
        lw, lb = m.layers[i], self.lb[i]
        nx = None if tail else m.layers[i + 1]
        x_ready, word = self._zeros(1), self._zeros(1)
        row0, row1 = (words.count("row0"), c.row_arrivals), (
            words.count("row1"),
            c.row_arrivals,
        )

        def gemm(g, w, x, gate):
            v["gemm_w"][g], v["gemm_w_rows"][g], v["gemm_x"][g], v["gemm_x_rows"][g] = (
                w,
                w.shape[0],
                x,
                x.shape[0],
            )
            v["gemm_k"][g] = w.shape[1]
            v["gemm_gate"][g], v["gemm_gate_arrivals"][g] = gate

        gemm(0, lw.o, lb.x_attn, (x_ready, c.combine_ctas))
        # the prepare unit: side 0 over its full 16 blocks, prepared in the epilogue (prep 0 / 1), x ready at
        # prep_arrivals
        gemm(1, lw.mlp_w0, lb.h2, row0)
        gemm(2, lw.mlp_w, lb.h2, row0)
        v["gemm_block0"][2] = c.gemm_blocks[2]
        gemm(3, lw.gate_up, lb.x_gu, (words.count("prep0"), c.prep_arrivals))
        gemm(4, lw.down, lb.act, (words.count("act"), c.act_arrivals))
        for s, (site, slot, base) in enumerate(
            (("att", "att1", lw.att_base), ("mlp", "mlp1", lw.mlp_base))
        ):
            bases = st.site_bases[site]
            v["fin_bases"][s], v["fin_coef"][s], v["fin_base1"][s] = (
                bases,
                st.rows(slot),
                base[1],
            )
            v["ar_bases"][s], v["ar_eps"][s] = bases, cx.dcfg.rms_eps
        v["ar_r_in"][0], v["ar_r_out"][0], v["ar_w"][0], v["ar_h"][0] = (
            lb.r_a,
            lb.r_b,
            lw.post_norm,
            lb.h2,
        )
        v["ar_ready"][0] = words.count("row0")
        v["ar_r_in"][1], v["ar_r_out"][1] = lb.r_b, lb.r_c
        v["ar_w"][1], v["ar_h"][1] = (m.norm, self.y) if tail else (nx.input_norm, lb.h)
        if tail:
            v["xq"] = self.xq
        v["ar_ready"][1] = words.count("row1")
        v["ar_rin_gate"][1], v["ar_rin_arrivals"][1] = row0
        self._gathers(
            v, img, ("mlp0", "mlp1") if tail else ("mlp0", "mlp1", "att0", "att1")
        )
        v["prep_h"][0], v["prep_base0"][0], v["prep_x"][0] = (
            lb.h2,
            lw.mlp_base[0],
            lb.x_gu,
        )
        v["prep_ready"][0] = words.count("prep0")
        v.update(act=lb.act, act_ready=words.count("act"))
        if not tail:
            gemm(5, nx.att_w0, lb.h, row1)
            gemm(7, nx.att_w, lb.h, row1)
            v["gemm_block0"][7] = c.gemm_blocks[7]
            gemm(
                6,
                nx.qkv,
                self.x_qkv,
                (words.count("prep1"), c.prep_arrivals),
            )
            v["prep_h"][1], v["prep_base0"][1], v["prep_x"][1] = (
                lb.h,
                nx.att_base[0],
                self.x_qkv,
            )
            v["prep_ready"][1] = words.count("prep1")
            nb = self.lb[i + 1]
            v.update(
                qkv_out=nb.qout,
                qkv_tickets=self._zeros(self.tickets),
                qkv_cos_sin=self.dcos_sin,
                qkv_slot=self.dslot,
                qkv_commit=self.commit,
                qkv_q_norm=nx.q_norm,
                qkv_k_norm=nx.k_norm,
                qkv_k_ring=nb.k_ring,
                qkv_v_ring=nb.v_ring,
                qkv_eps=cx.dcfg.rms_eps,
            )
        v.update(
            attn_q=lb.qout,
            attn_q_stride=self.qkv_cols * 2,
            attn_k_ring=lb.k_ring,
            attn_v_ring=lb.v_ring,
            draft_kv_len=self.dkv_len,
            attn_partials=lb.ws,
            attn_word=word,
            attn_x_ready=x_ready,
            launch_ctr=words.launch_ctr,
            rank=cx.rank,
            phase_offset=i,
            diag=cx.diag,
        )
        return img.bind(**v)

    def _bind_draft(self) -> None:
        """The draft: the entry, layer 0's qkv, one launch per draft layer, the draft head."""
        cx, b, m, f = self.cx, self.b, self.cx.model, self.cx.feeds
        self.draft_head = head.DraftHead(
            cx.draft_head_lib,
            cx.draft_head_tree_mod if self.tree else cx.draft_head_mod,
            cx.shape[b],
            PRESCORE_K,
            cx.hw[: cx.shape[b].shard_cols],
            cx.wq,
            cx.w_scale,
            self.y,
            m.pred,
            m.succ,
            self.words,
            cx.group,
            out=self.head_out,
            xq=self.xq,
            tree=(f.tree, f.cos_sin, f.cos_sin_cache) if self.tree else None,
            sample=(cx.draft_head_sample_mod, self.samp) if self.sample else None,
        )
        for t in self.draft_head.protocol():
            self._proto_tensor(t, 0)
        layers = [self._layer(i) for i in range(cx.dcfg.layers)]
        self.draft = [self.entry, self.qkv0, *layers, self.draft_head.launch]
        self.draft_sample = [*self.draft[:-1], self.draft_head.sample_launch]

    # ---- the verify and its accept ----

    def _bind_verify(self, block: bool) -> None:
        cx, cfg, b = self.cx, self.cx.cfg, self.b
        f = cx.feeds
        dev = cx.device
        self.accept_buf = head.AcceptBuffers(
            top1=torch.zeros(cfg.rows, dtype=I32, device=dev),
            out_ring=torch.zeros(cx.ring, dtype=I32, device=dev),
            cos_sin=f.cos_sin,
            loc_full=f.loc_full,
            loc_swa=f.loc_swa,
            kv_len=f.kv_len,
            dcos_sin=self.dcos_sin,
            dslot=self.dslot,
            dkv_len=self.dkv_len,
        )
        self.verify_accept = VerifyAccept(
            self,
            cx.accept_lib,
            cx.accept_tree_mod if self.tree else cx.accept_mod,
            tree=self.tree,
        )
        st = cx.stack
        # the 16-row images at b_live = B (a tree: their tree images); a block of 8 verifies on the 8-row set
        self.window = Window(b_live=b, capture=cx.capture, tree=self.tree)
        window = (
            Window(b_live=b, capture=cx.capture, native=True)
            if b == LIVE
            else self.window
        )
        aux = st.aux_for(window)
        self.verify = [
            aux.entry,
            aux.qkv,
            *st.layers(window),
            self.verify_accept.launch,
        ]
        if self.tree:
            self.verify.append(self.verify_accept.compact)
        if self.sample:
            self.sample_accept = SampleAccept(
                self, cx.accept_lib, cx.accept_sample_mod, block
            )
            self.verify_sample = [
                *self.verify[:-1],
                aux.lm_head,
                aux.softcap,
                self.sample_accept.launch,
            ]

    def verify_padded(self) -> list:
        """The verify on the 16-row images at b_live = B (at B = 8 the padded window, whose live rows are bitwise the
        8-row set's), bound on first use."""
        assert not self.tree, "a tree iteration verifies on its own launches"
        st = self.cx.stack
        return [
            st.aux.entry,
            st.aux.qkv,
            *st.layers(self.window),
            self.verify_accept.launch,
        ]

    def launches(
        self, part: str = "iter", padded: bool = False, sample: bool = False
    ) -> list:
        """The iteration's launch list: ``iter`` the whole iteration, ``draft`` its draft, ``verify`` the verify and
        verify_accept; ``padded``: the verify on the 16-row images; ``sample``: the sampled step's.
        """
        assert not sample or (self.sample and not padded)
        draft = self.draft_sample if sample else self.draft
        if sample:
            verify = self.verify_sample
        elif padded:
            verify = self.verify_padded()
        else:
            verify = self.verify
        return {"iter": draft + verify, "draft": draft, "verify": verify}[part]

    def sampling(
        self, seed: int, stream: int, top_k: int, temperature: float, top_p: float
    ) -> None:
        """The sampled step's words: the request's 64-bit seed (signed), the run's start position (keeps a restarted
        run's uniforms apart), top_k (>= the vocabulary: none), temperature (> 0) and top_p (0, 1].
        """
        assert temperature > 0 and 0 < top_p <= 1 and top_k >= 1
        tp = torch.tensor([temperature, top_p], dtype=torch.float32).view(torch.int32)
        words = torch.zeros(N_SAMP, dtype=torch.int64)
        words[SAMP["SEED"]] = seed
        words[SAMP["STREAM"]] = stream
        words[SAMP["TOP_K"]] = min(top_k, self.cx.cfg.vocab)
        words[SAMP["TEMP_TOP_P"]] = (int(tp[1]) << 32) | (int(tp[0]) & 0xFFFFFFFF)
        self.samp.copy_(words.pin_memory(), non_blocking=True)

    def pool_sync(
        self, pool: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[host.Launch, host.Launch]:
        """draft_kv_sync over a draft KV pool ([slots, row] K / V buffers per draft layer, the rings' rows): (the
        iteration's last launch, the launch after a draft alone). ``sync_state[0]``, the position up to which the pool
        holds the run's context, is the caller's to set at a run's start."""
        cx, dev = self.cx, self.cx.device
        assert len(pool) == len(self.lb)
        rings, bufs = [], []
        for lb, (k, v) in zip(self.lb, pool):
            for r, t in ((lb.k_ring, k), (lb.v_ring, v)):
                assert t.dtype == r.dtype and t.shape[1:] == r.shape[1:], (
                    t.shape,
                    r.shape,
                )
                rings.append(r.data_ptr())
                bufs.append(t.data_ptr())
        self.sync_state = torch.zeros(1 + cx.cfg.rows, dtype=torch.int64, device=dev)
        self._sync_ptrs = torch.tensor([rings, bufs], dtype=torch.int64, device=dev)
        lib = cx.accept_lib

        def bind(lag: int) -> host.Launch:
            return lib.bind(
                lib.dll.mk_dflash_kv_sync_bind,
                cx.kv_sync_mod,
                host.make(
                    "MkDflashKvSyncBind",
                    words=self.words,
                    window=cx.feeds.out_cache_loc,
                    state=self.sync_state,
                    ring=self._sync_ptrs[0],
                    pool=self._sync_ptrs[1],
                    n_buf=len(rings),
                    row_bytes=self.lb[0].k_ring[0].numel()
                    * self.lb[0].k_ring.element_size(),
                    ring_rows=cx.ring,
                    lag=lag,
                ),
            )

        return bind(1), bind(0)

    # ---- a run's start ----

    def start(self, p0: int, commit: int) -> None:
        """The step words and the draft's feeds of a run from committed length P0 after a commit of ``commit`` tokens;
        the anchor word and the draft's context (the ring, the previous verify's features) are the caller's.
        """
        cx, b = self.cx, self.b
        assert 1 <= commit <= b
        words = torch.zeros(head.N_WORDS, dtype=torch.int64)
        words[WORDS["P"]] = p0
        words[WORDS["ACCEPT"]], words[WORDS["COMMIT"]] = commit - 1, commit
        self.words.copy_(words.pin_memory(), non_blocking=True)
        d = head.draft_feeds(cx.dcache, p0, commit, b, cx.ring)
        self.dcos_sin.copy_(d["dcos_sin"])
        self.dslot.copy_(d["dslot"].pin_memory(), non_blocking=True)
        self.dkv_len.copy_(d["dkv_len"].pin_memory(), non_blocking=True)
