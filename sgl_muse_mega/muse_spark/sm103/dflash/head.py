"""The draft step's entry and the draft head from python (kernels/dflash/head.cu), and the head pieces the accept
shares (verify_accept): the head's shape, its outputs, the step words, and the accept's outputs and start feeds.

The entry turns the last verify's captured features and the next anchor into every input the draft's layer 0 needs,
identically on every rank:

    ctx_n   [B, H]        hidden_norm(fc(F)), F = the 5 captured residuals (fp32, rounded to bf16) of the window rows;
                          the ctx rows of every draft layer's K / V (rows >= the commit count are never committed)
    xq      [B, H]        layer 0's conv-prepared block rows (its qkv x): rows 0-1 per step, rows >= 2 a bind constant
    resid   [B, H]        layer 0's residual = the block's embedding rows [E[a'], E[mask] x (B - 1)]
    c_att1  [B, 2, 512]   layer 0's attention-conv side-1 coefficients (its finish), row 0 per step
    h_row   [H]           rms(E[a']) * in_norm (read by the bind passes)

fc is output-sharded over the ranks (TP8 1024 columns per rank, 83.9 MB; TP4 2048, 167.8 MB) and gathered by
multicast to every rank; conv_proj0 (layer 0's attention-conv projection) runs for the anchor row only, 2048 / R of its
outputs per rank, through the same gather. Its torch math and elementwise order: test/reference/dflash2.py.

The draft head reads W = [the rank's vocab rows of the target lm_head | its rows of W_P] (one tensor map, one pass over
the blocks: the band fills the shard's last block's pad rows) and the draft block's final rows 1 .. B-1. It streams an
e4m3 copy of the vocab shard (one fp32 scale per row: half the bytes of the bf16 shard), keeps per CTA and x row the
top-16 of this e4m3 pre-score, and rescores in bf16 only the survivors: every list entry at or above the ``ksurv``-th
largest list head. Its candidates are the exact top-16 of the bf16 logits computed in the rescoring's dot order whenever
every global candidate survives on its rank (the pre-score check: test/kernels/test_dflash_iteration.py);
``quantize_rows`` builds the pre-score's weight copy. Every rank ends with the same candidates, lattice and drafts (the
head's cross-rank merge is exact), so nothing is broadcast. verify_accept runs the accept and writes the next
iteration's step words and feeds. The sampled head (kernels/dflash/sample.cuh) walks the lattice by sampling instead,
with the same uniforms on every rank, and keeps each slot's probabilities (``q``) and draw (``path``) for the accept.
"""

from __future__ import annotations

import ctypes
import dataclasses
from pathlib import Path

import torch

from ....common import host
from ..buffers import Symmetric
from ..verify import persist_grid

TOPK = 16  # candidates per slot
RANK = 256  # selector rank
ROWS = 16  # the head's persist tile rows / the verify window's rows
ID_BITS = 18


def inbox_words(ranks: int) -> int:
    """The exchange's u64 words per (half, source rank, row): 16 candidates, then the rank's W_P y band (256 / ranks
    rows) as bf16 pairs (kernels/dflash/common.cuh kInboxWords at the image's world)."""
    return TOPK + 256 // ranks // 2


WORDS = {"P": 0, "ANCHOR": 1, "ACCEPT": 2, "COMMIT": 3, "ITER": 4, "OUT_COUNT": 5}
N_WORDS = 8
# the sampled step's words (kernels/dflash/sample.cuh): the request's seed, the run's start position (the uniforms'
# stream), top_k, and the fp32 temperature and top_p bits
SAMP = {"SEED": 0, "STREAM": 1, "TOP_K": 2, "TEMP_TOP_P": 3}
N_SAMP = 4


def draft_feeds(
    dcos_sin_cache: torch.Tensor, pos: int, commit: int, b: int, dring: int
) -> dict[str, torch.Tensor]:
    """The draft's rows in the order of its qkv's x: [0, b) the block at P' .., [b, 2b) the committed context rows at
    P' - c' .. (then masked: slot -1, rope row P'); slot = position mod the ring; the attention's kv_len P' + b.
    """
    q = torch.arange(2 * b)
    live = (q < b) | (q - b < commit)
    p = torch.where(q < b, pos + q, pos - commit + q - b)
    rows = torch.where(live, p, torch.full_like(p, pos))
    return {
        "dcos_sin": dcos_sin_cache[
            (rows.pin_memory() if dcos_sin_cache.is_cuda else rows).to(
                dcos_sin_cache.device, non_blocking=True
            )
        ],
        "dslot": torch.where(live, p % dring, torch.full_like(p, -1)).int(),
        "dkv_len": torch.tensor([pos + b], dtype=torch.int32),
    }


@dataclasses.dataclass
class HeadOut:
    cand: torch.Tensor  # i32 [gamma, 16]
    unary: torch.Tensor  # f32 [gamma, 16]
    hhat: torch.Tensor  # bf16 [gamma, 256]
    scores: torch.Tensor  # f32 [gamma, 16, 16]
    drafts: torch.Tensor  # i64 [gamma]
    verify_ids: torch.Tensor  # i64 [16]
    # f32 [gamma, 16] the sampled walk's probabilities over each slot's candidates
    q: torch.Tensor
    path: torch.Tensor  # i32 [gamma] the candidate index each slot drew

    @classmethod
    def alloc(cls, gamma: int, device) -> HeadOut:
        def z(*s, dtype):
            return torch.zeros(*s, dtype=dtype, device=device)

        return cls(
            z(gamma, TOPK, dtype=torch.int32),
            z(gamma, TOPK, dtype=torch.float32),
            z(gamma, RANK, dtype=torch.bfloat16),
            z(gamma, TOPK, TOPK, dtype=torch.float32),
            z(gamma, dtype=torch.int64),
            z(ROWS, dtype=torch.int64),
            z(gamma, TOPK, dtype=torch.float32),
            z(gamma, dtype=torch.int32),
        )

    def bind_fields(self) -> dict:
        return dict(
            cand=self.cand,
            unary=self.unary,
            hhat=self.hhat,
            scores=self.scores,
            drafts=self.drafts,
            verify_ids=self.verify_ids,
        )


@dataclasses.dataclass(frozen=True)
class HeadShape:
    hidden: int
    vocab_rank: int
    rank: int
    ranks: int
    gamma: int
    sms: int
    mult: float
    cap: float

    @property
    def hproj(self) -> int:
        return RANK // self.ranks

    @property
    def shard_cols(self) -> int:
        return self.vocab_rank + self.hproj

    @property
    def n_pad(self) -> int:
        # the lm_head's 256-row schedule
        return (self.shard_cols + 255) // 256 * 256

    @property
    def grid(self) -> int:
        return persist_grid(self.n_pad // 64, self.sms)


@dataclasses.dataclass
class AcceptBuffers:
    """The accept's outputs: the per-row argmax, the out ring and the feeds the next iteration's launches read."""

    top1: torch.Tensor
    out_ring: torch.Tensor
    cos_sin: torch.Tensor
    loc_full: torch.Tensor
    loc_swa: torch.Tensor
    kv_len: torch.Tensor
    dcos_sin: torch.Tensor
    dslot: torch.Tensor
    dkv_len: torch.Tensor


# ---- the draft head ----

STATE_WORDS = 3 + ROWS
# kernels/dflash/tree.cuh kFan: the successors a tree head keeps per lattice row
TREE_FAN = 4


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-score's weight copy: e4m3 rows with one fp32 scale per row (amax / 448), round to nearest even. Returns
    (uint8 [rows, H], fp32 [rows])."""
    s = w.float().abs().amax(1) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = (w.float() / s[:, None]).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), s


class DraftHeadLib(host.Library):
    def __init__(self, path):
        super().__init__(path)
        d = self.dll
        d.mk_dflash_draft_head_state_words.restype = ctypes.c_int64
        d.mk_dflash_draft_head_max_survivors.restype = ctypes.c_int64


class DraftHead:
    """The draft head. ``wb``: bf16 [vocab_rank + hproj, H] (the shard, then the W_P band); ``wq`` / ``w_scale``:
    ``quantize_rows`` of its vocab rows; ``y``: the block's final rows (>= gamma + 1 rows; rows 1 .. gamma read);
    ``xq``: u8 [gamma, H], y's rows 1 .. gamma in e4m3 with a scale per row (the last draft layer's final norm writes
    them); ``out``: the head's outputs. ``tree``: (the tree table, the window's rope rows, the rope table) for a tree
    head image (kernels/dflash/tree.cuh), which grows the token tree in place of the linear chain. ``sample``: (the
    sampled head image, the sampling words) for ``sample_launch``, the head that samples the walk into ``out.q`` /
    ``out.path`` (kernels/dflash/sample.cuh), bound on the same buffers and words as ``launch``.
    """

    def __init__(
        self,
        lib: DraftHeadLib,
        mod: host.Module,
        s: HeadShape,
        ksurv: int,
        wb: torch.Tensor,
        wq: torch.Tensor,
        w_scale: torch.Tensor,
        y: torch.Tensor,
        pred_cb: torch.Tensor,
        succ_cb: torch.Tensor,
        words: torch.Tensor,
        group,
        out: HeadOut,
        xq: torch.Tensor,
        tree: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        sample: tuple[host.Module, torch.Tensor] | None = None,
    ):
        assert wb.shape == (s.shard_cols, s.hidden) and wq.shape == (
            s.vocab_rank,
            s.hidden,
        )
        assert w_scale.shape == (s.vocab_rank,) and y.shape[0] >= s.gamma + 1
        device = wb.device
        self.inbox = Symmetric(
            2 * s.ranks * ROWS * inbox_words(s.ranks) * 8, device, group
        )
        self.state = torch.zeros(STATE_WORDS, dtype=torch.int32, device=device)
        self.part = torch.zeros(s.grid * ROWS * TOPK, dtype=torch.int64, device=device)
        self.max_surv = int(lib.dll.mk_dflash_draft_head_max_survivors())
        self.resc = torch.zeros(ROWS, self.max_surv, dtype=torch.int64, device=device)
        self.hloc = torch.zeros(ROWS, s.hproj, dtype=torch.bfloat16, device=device)
        self.xq = xq
        assert xq.shape == (s.gamma, s.hidden) and xq.dtype == torch.uint8
        self.pick = torch.zeros(s.gamma, TOPK, dtype=torch.int32, device=device)
        self.out = out
        fan = (s.gamma, TOPK, TREE_FAN)
        self.fan_idx = (
            None
            if tree is None
            else torch.zeros(*fan, dtype=torch.int32, device=device)
        )
        self.fan_lp = (
            None
            if tree is None
            else torch.zeros(*fan, dtype=torch.float32, device=device)
        )
        assert int(lib.dll.mk_dflash_draft_head_state_words()) == STATE_WORDS

        def bind(module, tree, samp) -> host.Launch:
            table, cos_sin, cos_sin_cache = tree or (None, None, None)
            return lib.bind(
                lib.dll.mk_dflash_draft_head_bind,
                module,
                host.make(
                    "MkDflashDraftHeadBind",
                    wq=wq,
                    w_scale=w_scale,
                    wb=wb,
                    y=y,
                    xq=self.xq,
                    hidden=s.hidden,
                    vocab_rank=s.vocab_rank,
                    hproj=s.hproj,
                    n_pad=s.n_pad,
                    grid=s.grid,
                    rank=s.rank,
                    ranks=s.ranks,
                    gamma=s.gamma,
                    ksurv=ksurv,
                    inbox_peer=self.inbox.ptrs,
                    pred_codebook=pred_cb,
                    succ_codebook=succ_cb,
                    words=words,
                    state=self.state,
                    part=self.part,
                    resc=self.resc,
                    hloc=self.hloc,
                    pick=self.pick,
                    mult=s.mult,
                    cap=s.cap,
                    **self.out.bind_fields(),
                    tree=table,
                    fan_idx=self.fan_idx,
                    fan_lp=self.fan_lp,
                    cos_sin=cos_sin,
                    cos_sin_cache=cos_sin_cache,
                    cos_sin_rows=0 if tree is None else cos_sin_cache.shape[0],
                    d=0 if tree is None else cos_sin_cache.shape[1],
                    samp=samp,
                    walk_q=None if samp is None else self.out.q,
                    walk_path=None if samp is None else self.out.path,
                ),
            )

        self.launch = bind(mod, tree, None)
        self.sample_launch = (
            None if sample is None else bind(sample[0], None, sample[1])
        )

    def protocol(self) -> list[torch.Tensor]:
        """The words and regions a run resets to zero (the caller syncs every rank)."""
        return [self.inbox.local, self.state, self.part]


# ---- the entry ----


def entry_constants(lib: host.Library) -> ctypes.Structure:
    c = host.STRUCTS["MkDflashEntryConstants"]()
    lib.dll.mk_dflash_entry_constants.argtypes = [ctypes.POINTER(type(c))]
    lib.check(lib.dll.mk_dflash_entry_constants(ctypes.byref(c)), "constants")
    return c


@dataclasses.dataclass
class EntryWeights:
    """The draft weights the entry reads beside its packed tiles (fc and layer 0's attention-conv projection)."""

    hidden_norm: torch.Tensor  # [H]
    in_norm: torch.Tensor  # layers.0.input_layernorm [H]
    conv_base: torch.Tensor  # layers.0.attention_conv.base_kernel [2 sides, taps, H]


class Entry:
    """One bound entry: its symmetric region, partials, protocol words, outputs and bind constants."""

    def __init__(
        self,
        rows: int,
        paths: dict[str, Path],
        w_fc: torch.Tensor,
        w_conv_proj: torch.Tensor,
        w: EntryWeights,
        cap: list[torch.Tensor],
        anchor: torch.Tensor,
        embed_peers: list[int],
        shard_rows: int,
        rank: int,
        group,
        eps: float,
        out: dict[str, torch.Tensor],
    ):
        """``out``: the caller-owned outputs by name (ctx_n, xq, resid, c_att1): views of the draft layer 0's qkv x
        rows, its residual and its finish's coefficient rows."""
        self.lib = host.Library(paths[f"libmk_dflash_entry_b{rows}"])
        self.c = c = entry_constants(self.lib)
        assert c.rows == rows
        module = self.lib.load(paths[f"dflash_entry_b{rows}"])
        dev = w_fc.device
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.sym = Symmetric(c.sym_bytes, dev, group)
        assert (
            self.sym.mc != 0
        ), "the gather writes through the TP group's multicast address"
        # every gathered packet starts unlanded (bf16 0xFFFF)
        self.sym.local.fill_(0xFF)
        self.ws = torch.zeros(c.ws_floats, dtype=torch.float32, device=dev)
        self.ctr = torch.zeros(c.ctr_words, dtype=torch.int32, device=dev)
        H = c.hidden

        def output(name: str, *shape) -> torch.Tensor:
            t = out[name]
            assert (
                t.shape == shape and t.dtype == torch.bfloat16 and t.is_contiguous()
            ), name
            return t

        assert set(out) == {"ctx_n", "xq", "resid", "c_att1"}, sorted(out)
        self.ctx_n, self.xq = output("ctx_n", rows, H), output("xq", rows, H)
        self.resid, self.c_att1 = output("resid", rows, H), output(
            "c_att1", rows, c.coeffs // 2
        )
        self.h_row = torch.zeros(H, **bf)
        self.e_mask, self.h_mask, self.xq_const = (
            torch.zeros(H, **bf) for _ in range(3)
        )
        self.d_mask, self.delta = (torch.zeros(c.coeffs, **bf) for _ in range(2))
        self.anchor = anchor
        self.launch = self.lib.bind(
            self.lib.dll.mk_dflash_entry_bind,
            module,
            host.make(
                "MkDflashEntryBind",
                cap=cap,
                anchor=anchor,
                embed_peer=embed_peers,
                shard_rows=shard_rows,
                w_fc=w_fc,
                w_conv_proj=w_conv_proj,
                in_norm=w.in_norm,
                hid_norm=w.hidden_norm,
                conv_base=w.conv_base[0],
                e_mask=self.e_mask,
                h_mask=self.h_mask,
                d_mask=self.d_mask,
                xq_const=self.xq_const,
                sym_local=self.sym.ptrs[rank],
                sym_mc=self.sym.mc,
                rank=rank,
                ctx_n=self.ctx_n,
                xq=self.xq,
                resid=self.resid,
                c_att1=self.c_att1,
                h_row=self.h_row,
                delta=self.delta,
                ws=self.ws,
                ctr=self.ctr,
                eps=eps,
            ),
        )

    def __call__(self, stream: int, pdl: bool = False) -> None:
        self.launch(stream, pdl)

    def bind_constants(self, mask_id: int, step) -> None:
        """The mask rows, from two launches with anchor = mask (``step`` = one rank-synchronized launch): the first
        yields E[mask], rms(E[mask]) and its coefficients, the second the prepared row of two mask rows (row 1 with
        h0 = hm). Every rank runs both."""
        saved = self.anchor.clone()
        self.anchor.fill_(mask_id)
        step()
        self.e_mask.copy_(self.resid[0])
        self.h_mask.copy_(self.h_row)
        self.d_mask.copy_(self.delta)
        step()
        self.xq_const.copy_(self.xq[1])
        self.anchor.copy_(saved)
