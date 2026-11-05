"""The DFlash2 iteration's checks on 4 ranks (one GB300 node, TP4) or 8 (two nodes, TP8), synthetic target weights
and the published draft (or a synthetic one):

    torchrun --nproc-per-node 4 test/kernels/test_dflash_iteration.py --out DIR [--draft DIR]

Checks per block size B, from one start state (P0 = --ctx, the anchor, the commit, seeded features), ``--iters``
iterations under a forced accept (``FORCED``: the synthetic target never accepts on its own, so verify_accept takes
every accept count in turn), on the iteration (``iteration.Iteration``: one launch per draft layer, the draft head,
the verify with verify_accept; at B = 8 the verify on the 8-row images):
  reference (the iteration run eagerly, PDL on every launch, synced per iteration; every check feeds the reference the
             kernel's own inputs)
    stages   every draft unit of every iteration vs dflash2 fed that unit's kernel inputs (each unit's output
             stays in its layer's buffer): the entry (its residual rows equal to the embedding rows, ctx_n), the
             attention + its split-KV combine, the o_proj sublayer (r', h2), the MLP conv's prepare, gate_up + SwiGLU,
             the down sublayer (r', h), every qkv (q, block K / V); per row cos >= 0.9999, rel-L2 <= 1e-2
    layers   every draft layer of every iteration vs the reference layer fed the kernel's input to it (its normed
             input and residual, the kernel's KV history and context rows): r' and h, the unit band (layer 0: its own,
             LAYER0_BAND)
    head     where every global candidate survived some rank's e4m3 pre-score, the head's candidates = the exact
             top-16 of the bf16 logits in the rescoring's order (reference.dflash2.exact_logits over every rank's
             shard), and its unary = their softcap
    tokens   the head rerun on the kernel's own values: the fp64 lattice of its own candidates / unary / W_P y = its
             scores, the walk of its scores = its drafts (bitwise). The whole head rerun in fp64 on the kernel's y is
             counted (a top-16 boundary or walk step within an ulp may round the other way)
    accept   the step words (P', anchor, accept, commit) = the greedy accept of verify_accept's per-row argmax and the
             window ids under the forced accept (the prefix rule itself: test/unit/test_dflash_head.py); the window
             ids = the anchor and the drafts, the pads repeating them
    verify   the last iteration's per-row argmax = that of a plain verify step's logits on the same window, ids,
             b_live and capture
    ranks    y, the drafts, the words and the features bitwise equal on every rank
  end to end (recorded, not checked: the per-unit and per-layer checks localize it) y and every draft KV row the run
             wrote vs the reference run end to end on its own KV, in the kernel's sublayer order ('pre_ar'), next to
             the reference's own spread between that order and SGLang's ('post_ar'); and per iteration the forced
             suffixes: y vs the reference run from layer l's kernel input to the end, l = 0 .. 4
  graph      the iteration as one CUDA graph: ``--iters`` replays bitwise the eager run, a second set bitwise the first
  padded     at B = 8, the iteration on the 8-row images bitwise the same iteration on the padded 16-row window (every
             output; with --long-ctx also from that start state)
  bind       on plain verify steps: the capture pointers leave the logits and every layer's outputs bitwise; b_live 8
             leaves rows 0-7 of the logits and of every capture buffer bitwise those of the 16-row window
``--tree``: B = 16 on the token-tree window (kernels/dflash/tree.cuh), the forced accept naming the leaf row; the
linear chain's tokens / accept / window-ids / verify checks become:
  tree       the head's tree (ids, drafts, the table's depth / mask / parent) bitwise the CPU reference
             (reference.dflash2.token_tree) on the head's own lattice
  accept     the step words, the table's path and the out ring's committed tokens = reference.dflash2.tree_accept
  compact    every iteration (eager): after tree_compact, each layer's K / V rows and the captured rows of rows
             1 .. accept are the path's rows from before it, every other window row unchanged
  verify     the last iteration's per-row argmax = that of a verify step on the same tree (the tree layer images, the
             table, the rope rows at P + depth)
  rows       each tree row vs the linear window that holds its path (the anchor .. the row at rows 0 .. depth): a row on
             the main path (rank-1 rows: the same keys at the same slots) bitwise in every layer and the logits; a branch row
             (its ancestors at other key slots: another accumulation order) within TREE_REL of its path's row in layer
             0's output, and TREE_SEPARATION x closer to it than to the window's row read as a linear chain; the negative
             control: with one branch row's parent bit cleared from its mask, that row's layer-0 output leaves its
             path's by more than TREE_SEPARATION x its own difference (the check sees a mask error). Recorded: the
             first layer where a branch row leaves TREE_REL (on synthetic weights an ulp-level difference grows
             through routing near-ties, so later layers and the logits are not comparable), the logits, the argmax,
             and a second linear window with another tail (the linear chain's own row spread)
  coverage   the run moved rows (an accepted path off the main path) and the tree has branch rows
Per-rank raws: DIR/dflash_rank<r>_<label>.json; rank 0: DIR/dflash_<label>.json and the summary lines.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parents[1]),
    str(HERE.parent),
]  # the package, test/ (the references)

import synthetic  # noqa: E402
from reference import dflash2 as R  # noqa: E402
from sgl_muse_mega.muse_spark.config import Config  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash import iteration  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash.head import WORDS  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash.weights import (  # noqa: E402
    DRAFT_FILE,
    DraftWeights,
    load_shard,
)
from sgl_muse_mega.muse_spark.sm103.verify import LIVE, Window  # noqa: E402
from synthetic import begin, forced_accepts, hang_report, reset_all  # noqa: E402
from test_verify_step import layer_digests  # noqa: E402

BF16, F32 = torch.bfloat16, torch.float32
# the end-to-end band (recorded): the kernel's draft against the reference in the kernel's sublayer order ('pre_ar')
# is held to the reference's own spread between that order and SGLang's ('post_ar'): SPREAD x it, at least FLOOR (per
# row rel-L2)
ORDERS = ("pre_ar", "post_ar")
SPREAD, FLOOR = 1.5, 2e-2
# the unit checks' band (per row)
COS_MIN, REL_MAX = 0.9999, 1e-2
# the layer check's band (cos_min, rel_max per row): the unit band, except layer 0, whose reference output moves 1.0 %
# when one unit's output carries a 2e-3 relative jitter (layers 1-4: <= 0.26 %; the published DFlash2 weights at
# these inputs' statistics): 2.5x that
LAYER0_BAND = (0.9995, 2.5e-2)
# the tree table's words (kernels/dflash/tree.cuh)
T_DEPTH, T_MASK, T_PARENT, T_PATH, T_SLOT_FULL, T_SLOT_SWA = 0, 16, 32, 48, 64, 80
# a branch tree row against its path's linear row in layer 0's output: the unit band; and TREE_SEPARATION x closer to
# it than to the window's row read as a linear chain (other positions and keys)
TREE_REL, TREE_SEPARATION = REL_MAX, 10.0


# ---- cross-rank reads and comparisons ----


class GlooComm:
    """dflash2's Comm over the TP group: all_gather of bytes (gloo moves no bf16), rank order."""

    def __init__(self, rank: int, ranks: int):
        self.rank, self.ranks = rank, ranks

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        h = x.detach().contiguous()
        raw = h.view(-1).view(torch.uint8)
        parts = [torch.empty_like(raw) for _ in range(self.ranks)]
        dist.all_gather(parts, raw)
        return torch.cat([p.view(h.dtype).view(h.shape) for p in parts], dim)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(-1).view(torch.uint8),
            b.contiguous().view(-1).view(torch.uint8),
        )
    )


def rows_rel(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    g, r = got.double().flatten(1), ref.double().flatten(1)
    return (g - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-300)


def row_band(
    got: torch.Tensor,
    ref: torch.Tensor,
    cos_min: float = COS_MIN,
    rel_max: float = REL_MAX,
) -> dict:
    g, r = got.double().flatten(1), ref.double().flatten(1)
    cos = (g * r).sum(1) / (g.norm(dim=1) * r.norm(dim=1)).clamp_min(1e-300)
    rel = rows_rel(got, ref)
    return dict(
        ok=bool(cos.min() >= cos_min and rel.max() <= rel_max),
        cos_min=float(cos.min()),
        rel_max=float(rel.max()),
        frac_diff=float((got != ref.to(got.dtype)).float().mean()),
        tol=dict(cos=cos_min, rel=rel_max),
    )


def sha(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(
            t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()
        )
    return h.hexdigest()[:16]


def synced(cx: iteration.Context, what: str) -> None:
    """A device sync that names a trapped wait (the layer's hang record; the other units print theirs) instead of a
    bare launch failure."""
    try:
        torch.cuda.synchronize(cx.device)
    except RuntimeError as e:
        raise RuntimeError(f"{what} rank {cx.rank}: {hang_report(cx.diag) or e}") from e


# ---- runs ----


class Graphs:
    """One CUDA graph per (block, part, padded), captured on first use."""

    def __init__(self, cx: iteration.Context):
        self.cx, self.g = cx, {}

    def get(
        self, it: iteration.Iteration, part: str, padded: bool = False
    ) -> torch.cuda.CUDAGraph:
        key = (it.b, part, padded)
        if key not in self.g:
            cx, s = self.cx, self.cx.main.cuda_stream
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=cx.main):
                for launch in it.launches(part, padded):
                    launch(s, True)
            self.g[key] = g
        return self.g[key]


def record(cx: iteration.Context, it: iteration.Iteration, words_in, caps_in) -> dict:
    """One iteration's outputs on the host (its inputs given; a tree iteration's table and out ring too)."""
    b, h = it.b, it.head_out
    tree = {}
    if it.tree:
        # the table's path past the commit holds earlier iterations' rows (never read): masked
        table = cx.feeds.tree.cpu()
        table[T_PATH + int(it.words[WORDS["COMMIT"]]) : T_PATH + 16] = -1
        tree = dict(table=table, out_ring=it.accept_buf.out_ring.cpu())
    return dict(
        **tree,
        words_in=words_in,
        caps_in=caps_in,
        y=it.y[:b].cpu(),
        cand=h.cand.cpu(),
        unary=h.unary.cpu(),
        hhat=h.hhat.cpu(),
        scores=h.scores.cpu(),
        drafts=h.drafts.cpu(),
        ids=cx.bufs.ids.cpu(),
        top1=it.accept_buf.top1[:b].cpu(),
        words=it.words.cpu(),
        resc=it.draft_head.resc.cpu(),
    )


def caps_now(cx: iteration.Context, b: int) -> torch.Tensor:
    return torch.stack([cx.stack.cap(L)[:b] for L in cx.capture]).cpu()


def window_rows(cx: iteration.Context) -> dict:
    """A tree window's rows after verify_accept: every layer's K / V rows at the slots verify_accept recorded in the table (its class's
    pool) and the captured rows."""
    t = cx.feeds.tree.cpu()
    slots = {
        "global": t[T_SLOT_FULL : T_SLOT_FULL + 16].long(),
        "swa": t[T_SLOT_SWA : T_SLOT_SWA + 16].long(),
    }
    kv = []
    for L, (k, v) in enumerate(cx.stack.kv):
        sl = slots[cx.cfg.attn_type(L)].to(k.device)
        kv.append(torch.stack([k[sl], v[sl]]).cpu())
    return dict(kv=torch.stack(kv), caps=caps_now(cx, cx.cfg.rows))


def compact_check(rec: dict, pre: dict, post: dict) -> dict:
    """tree_compact: row i of 1 .. accept holds row path[i] from before it, every other row is unchanged."""
    commit = int(rec["words"][WORDS["COMMIT"]])
    path = rec["table"][T_PATH + 1 : T_PATH + commit].tolist()
    src = list(range(16))
    for i, r in enumerate(path, 1):
        src[i] = r
    ok = bits_equal(post["kv"], pre["kv"][:, :, src]) and bits_equal(
        post["caps"], pre["caps"][:, src]
    )
    return dict(
        ok=ok, accept=commit - 1, moved=sum(r != i for i, r in enumerate(path, 1))
    )


def layer_record(it: iteration.Iteration) -> dict:
    """Every draft layer's own buffers after an iteration (each unit's output stays in its layer's buffer), and the
    entry's ctx rows (the qkv x's rows [B, 2B), which no later unit writes)."""
    b = it.b
    layers = [
        dict(
            r_a=lb.r_a.cpu(),
            x_attn=lb.x_attn[:b].cpu(),
            r_b=lb.r_b.cpu(),
            h2=lb.h2[:b].cpu(),
            x_gu=lb.x_gu[:b].cpu(),
            act=lb.act[:b].cpu(),
            r_c=lb.r_c.cpu(),
            h=lb.h[:b].cpu(),
            qout=lb.qout[:b].cpu(),
            k_ring=lb.k_ring.cpu(),
            v_ring=lb.v_ring.cpu(),
        )
        for lb in it.lb
    ]
    return dict(layers=layers, ctx_n=it.x_qkv[b : 2 * b].cpu())


def run(
    cx: iteration.Context,
    its: dict,
    it: iteration.Iteration,
    n: int,
    p0: int,
    graphs,
    layers: bool = False,
    padded: bool = False,
) -> dict:
    """``n`` iterations from the start state, eager (graphs None: every launch PDL) or as graph replays; per iteration
    its record (``layers``: and every draft layer's buffers), then the draft rings and the features. ``padded``: the
    verify on the 16-row images.
    """
    begin(cx, its, it, p0)
    s = cx.main.cuda_stream
    recs = []
    for i in range(n):
        words_in, caps_in = it.words.cpu(), caps_now(cx, it.b)
        pre = None
        if graphs is None:
            with torch.cuda.stream(cx.main):
                it.draft_head.resc.zero_()  # the head check's survivor lists (the head writes each row's first n)
            launches = it.launches("iter", padded)
            if it.tree:  # the window's rows before tree_compact, its last launch
                for launch in launches[:-1]:
                    launch(s, True)
                synced(cx, f"b{it.b} iteration {i} before tree_compact")
                pre, launches = window_rows(cx), launches[-1:]
            for launch in launches:
                launch(s, True)
        else:
            with torch.cuda.stream(cx.main):
                graphs.get(it, "iter", padded).replay()
        synced(cx, f"b{it.b} iteration {i}")
        recs.append(record(cx, it, words_in, caps_in))
        if pre is not None:
            recs[-1]["compact"] = compact_check(recs[-1], pre, window_rows(cx))
        if layers:
            recs[-1].update(layer_record(it))
    rings = torch.stack([torch.stack([lb.k_ring, lb.v_ring]) for lb in it.lb]).cpu()
    return dict(recs=recs, rings=rings, caps=caps_now(cx, it.b))


# the outputs a run's records compare (verify_accept leaves no logits row: the accept's per-row argmax stands for them)
KEYS = ("y", "cand", "unary", "hhat", "scores", "drafts", "ids", "top1", "words")


def same_run(a: dict, b: dict, keys=KEYS) -> dict:
    """Bitwise per output over the iterations (the first iteration that differs), the rings and the features."""
    out = {}
    for k in keys:
        bad = [
            i
            for i, (x, y) in enumerate(zip(a["recs"], b["recs"]))
            if not bits_equal(x[k], y[k])
        ]
        out[k] = dict(ok=not bad, first_diff=bad[0] if bad else None)
    out["rings"] = dict(ok=bits_equal(a["rings"], b["rings"]))
    out["caps"] = dict(ok=bits_equal(a["caps"], b["caps"]))
    return out


# ---- the reference checks ----


def spread_band(got: torch.Tensor, ref: torch.Tensor, other: torch.Tensor) -> dict:
    """got vs ref (the reference in the kernel's order) held to the reference's own spread: per row rel-L2 within
    max(FLOOR, SPREAD x) the largest row rel-L2 between ref and other (the same math in SGLang's order).
    """
    b, s = row_band(got, ref), row_band(other, ref)
    tol = max(FLOOR, SPREAD * s["rel_max"])
    return dict(
        ok=b["rel_max"] <= tol,
        rel_max=b["rel_max"],
        cos_min=b["cos_min"],
        tol=tol,
        orders_rel_max=s["rel_max"],
        orders_cos_min=s["cos_min"],
        frac_diff=b["frac_diff"],
    )


def history_kv(cx: iteration.Context, layers: list[dict], p: int, b: int) -> R.DraftKV:
    """A reference KV holding the kernel's ring rows at positions [p - window_left, p) of every layer (its history and
    this iteration's context rows, exactly what the kernel's attention read); the block rows are the reference's own.
    """
    kv = R.DraftKV.empty(cx.dcfg, p + b, cx.cfg.tp, BF16)
    pos = torch.arange(max(0, p - cx.dcfg.window_left), p)
    for i, L in enumerate(layers):
        kv.k[i][pos] = L["k_ring"][pos % cx.ring].view(kv.k[i][pos].shape)
        kv.v[i][pos] = L["v_ring"][pos % cx.ring].view(kv.v[i][pos].shape)
    return kv


def forced_suffixes(cx: iteration.Context, w: DraftWeights, rec: dict, comm) -> dict:
    """The reference run from draft layer l's kernel input to the final norm, l = 0 .. 4, in the kernel's sublayer
    order: the layer's normed input and residual as the kernel computed them (layer 0: the entry's embedding rows,
    normed by the reference: no rounding sits between), the kernel's KV history. Records y vs the kernel's y
    (``y_rel``: max over rows; ``y_rows``: per row) and checks that run's first layer's outputs r' and h against the
    kernel's (the layer check)."""
    dcfg, b = cx.dcfg, rec["y"].shape[0]
    p = int(rec["words_in"][WORDS["P"]])
    L = rec["layers"]
    out = dict(y_rel=[], y_rows=[], layer=[])
    for first in range(dcfg.layers):
        kv = history_kv(cx, L, p, b)

        def args(i: int) -> tuple:
            return kv.k[i], kv.v[i], p, comm, BF16, "pre_ar"

        lw = w.layers[first]
        if first == 0:
            x, res = R.draft_layer(dcfg, lw, L[0]["r_a"], None, *args(0))
        else:
            h_in, r_in = L[first - 1]["h"], L[first]["r_a"]
            x, res = R.draft_layer_normed(dcfg, lw, h_in, r_in, *args(first))
        nxt = w.layers[first + 1].input_norm if first + 1 < dcfg.layers else w.norm
        h, r_c = R.add_rms_norm(x, res, nxt, dcfg.rms_eps, BF16)
        band = LAYER0_BAND if first == 0 else (COS_MIN, REL_MAX)
        out["layer"].append(
            dict(
                r=row_band(L[first]["r_c"], r_c, *band),
                h=row_band(L[first]["h"], h, *band),
            )
        )
        for i in range(first + 1, dcfg.layers):
            x, res = R.draft_layer(dcfg, w.layers[i], x, res, *args(i))
        y = R.add_rms_norm(x, res, w.norm, dcfg.rms_eps, BF16)[0]
        rel = rows_rel(rec["y"], y)
        out["y_rel"].append(float(rel.max()))
        out["y_rows"].append([round(float(v), 5) for v in rel])
    out["ok"] = all(g["r"]["ok"] and g["h"]["ok"] for g in out["layer"])
    return out


def reference_check(
    cx: iteration.Context,
    w: DraftWeights,
    it: iteration.Iteration,
    eager: dict,
    rings0: torch.Tensor,
    p0: int,
    comm,
) -> dict:
    """dflash2 on the kernel's own inputs, iteration by iteration (module docstring): the tokens, the
    accept, the forced suffixes (the layer check), and the end-to-end run in both sublayer orders (recorded).
    """
    cfg, dcfg, b = cx.cfg, cx.dcfg, it.b
    t = R.TargetShared(cx.bufs.embed_shard(cfg).cpu(), cx.hw[: cfg.vocab_rank].cpu())
    recs = eager["recs"]
    p_end = int(recs[-1]["words"][WORDS["P"]])
    kvs = {o: R.DraftKV.empty(dcfg, p_end + b + 16, cfg.tp, BF16) for o in ORDERS}
    lo = max(0, p0 - cx.ring)
    pos = torch.arange(lo, p0)
    for kv in kvs.values():
        for i in range(dcfg.layers):
            kv.k[i][lo:p0] = rings0[i, 0][pos % cx.ring].view(kv.k[i][lo:p0].shape)
            kv.v[i][lo:p0] = rings0[i, 1][pos % cx.ring].view(kv.v[i][lo:p0].shape)
    override = it.override.cpu()
    per = []
    for rec in recs:
        wi = rec["words_in"]
        P, anchor, commit, itn = (
            int(wi[WORDS[k]]) for k in ("P", "ANCHOR", "COMMIT", "ITER")
        )
        feats = torch.cat(
            [rec["caps_in"][k].to(BF16) for k in range(len(cx.capture))], 1
        )
        st = R.DecodeState(P, anchor, feats, commit)
        ds = {
            o: R.draft_step(dcfg, w, t, kvs[o], st, b, comm, BF16, order=o)
            for o in ORDERS
        }
        d = ds["pre_ar"]
        g = {
            "y_end_to_end": dict(
                info=True,
                **spread_band(rec["y"], d.y, ds["post_ar"].y),
                rows_rel=[round(float(x), 5) for x in rows_rel(rec["y"], d.y)],
            ),
            "y_vs_sglang_order": dict(info=True, **row_band(rec["y"], ds["post_ar"].y)),
            "forced": forced_suffixes(cx, w, rec, comm),
        }
        yk, cand_k = rec["y"][1:], rec["cand"].long()
        # the head rerun on its own values: the fp64 lattice of its own candidates / unary / W_P y, the walk of its own
        # scores: each bitwise the kernel's
        own = R.lattice_hhat(
            w.pred_codebook, w.succ_codebook, rec["hhat"], cand_k, rec["unary"], anchor
        )
        g["head_scores"] = dict(ok=bits_equal(rec["scores"], own))
        forced = int(override[itn % override.numel()])
        if it.tree:
            g.update(tree_checks(rec, anchor, cand_k, P, forced))
            per.append(g)
            continue
        walk, _ = R.selector_walk(cand_k, rec["scores"])
        g["head_drafts"] = dict(ok=bool(torch.equal(walk, rec["drafts"])))
        # the whole head rerun in fp64 on the kernel's y: the logits round once from fp64, so a top-16 boundary or a
        # walk step within an ulp may go the other way (counted)
        cand, unary = R.candidates(dcfg, t, yk, comm, BF16)
        drafts, _ = R.selector_walk(cand, R.lattice(w, yk, cand, unary, anchor, BF16))
        g["rerun_y"] = dict(
            info=True,
            cand_equal=bool(torch.equal(cand_k, cand)),
            drafts_equal=bool(torch.equal(rec["drafts"], drafts)),
            kernel=rec["drafts"].tolist(),
            rerun=drafts.tolist(),
        )
        g["drafts_vs_reference_y"] = dict(
            info=True,
            **{o: bool(torch.equal(ds[o].tokens[1:], rec["drafts"])) for o in ORDERS},
            orders_equal=bool(torch.equal(ds["pre_ar"].tokens, ds["post_ar"].tokens)),
        )
        a = R.accept_ref(rec["ids"][:b], rec["top1"], P, forced=forced)
        wo = rec["words"]
        got = tuple(int(wo[WORDS[k]]) for k in ("P", "ANCHOR", "ACCEPT", "COMMIT"))
        g["accept"] = dict(
            ok=got == (a.pos, a.bonus, a.accept, a.commit),
            got=got,
            ref=(a.pos, a.bonus, a.accept, a.commit),
        )
        g["window_ids"] = dict(
            ok=bool(torch.equal(rec["ids"], R.verify_ids(anchor, rec["drafts"])))
        )
        per.append(g)
    # the rings: every position the run wrote, its last writer's row; the last draft wrote its block at
    # [P_last, P_last + B) and no draft follows the last accept
    k0 = int(recs[0]["words_in"][WORDS["COMMIT"]])
    p_last = int(recs[-1]["words_in"][WORDS["P"]])
    span = torch.arange(p0 - k0, p_last + b)
    rings = eager["rings"]
    ring = {}
    for i in range(dcfg.layers):
        for j, name in enumerate(("k", "v")):
            ref = [
                (kvs[o].k if j == 0 else kvs[o].v)[i][span].flatten(1) for o in ORDERS
            ]
            ring[f"L{i}_{name}"] = dict(
                info=True, **spread_band(rings[i, j][span % cx.ring], *ref)
            )
    ok = all(v.get("ok", True) for g in per for v in g.values() if not v.get("info"))
    return dict(
        ok=ok,
        iterations=per,
        ring_end_to_end=ring,
        positions=[int(span[0]), int(span[-1])],
    )


def tree_checks(
    rec: dict, anchor: int, cand: torch.Tensor, p: int, forced: int
) -> dict:
    """A tree iteration's head and accept against the CPU reference (module docstring, tree / accept / compact)."""
    tr = R.token_tree(anchor, cand, rec["scores"])
    tab = rec["table"]
    tree_ok = (
        rec["ids"].tolist() == tr.tokens
        and rec["drafts"].tolist() == tr.tokens[1:]
        and tab[T_DEPTH : T_DEPTH + 16].tolist() == tr.depth
        and tab[T_MASK : T_MASK + 16].tolist() == tr.mask
        and tab[T_PARENT : T_PARENT + 16].tolist() == tr.parent
    )
    a, path = R.tree_accept(tr, rec["top1"], p, forced=forced)
    wo, ring = rec["words"], rec["out_ring"]
    got = tuple(int(wo[WORDS[k]]) for k in ("P", "ANCHOR", "ACCEPT", "COMMIT"))
    cnt = int(rec["words_in"][WORDS["OUT_COUNT"]])
    committed = ring[(cnt + torch.arange(a.commit)) % ring.numel()].tolist()
    accept_ok = (
        got == (a.pos, a.bonus, a.accept, a.commit)
        and tab[T_PATH + 1 : T_PATH + 1 + a.accept].tolist() == path
        and committed == a.committed.tolist()
    )
    return {
        "tree": dict(ok=tree_ok),
        "accept": dict(ok=accept_ok, got=got, ref=(a.pos, a.bonus, a.accept, a.commit)),
        "compact": rec["compact"],
    }


def draft_head_check(
    cx: iteration.Context, it: iteration.Iteration, eager: dict, comm
) -> dict:
    """The draft head's candidates per iteration (module docstring, head): the exact bf16 logits of every rank's
    shard in the rescoring's order, gathered, and their top-16 (ties to the lower id); where each of those 16 survived
    on some rank (the survivor keys of an eager run, zeroed before each iteration), the head's candidates = them and its
    unary = their softcap."""
    cfg, dcfg, b = cx.cfg, cx.dcfg, it.b
    w = cx.hw[: cfg.vocab_rank]
    per = []
    for rec in eager["recs"]:
        mine = R.exact_logits(w, rec["y"][1:].to(w.device)).cpu()
        vals = comm.all_gather(mine, 1).float()
        resc = comm.all_gather(rec["resc"][None], 0)
        cand = rec["cand"].long()
        tv, tid = R.topk_low_id(
            vals, torch.arange(cfg.vocab).expand_as(vals), dcfg.top_k
        )
        alive = R.survived(resc[:, : b - 1], tid)
        du = (
            (R.softcap(tv, dcfg.output_multiplier, dcfg.softcap) - rec["unary"])
            .abs()
            .amax(1)
        )
        ok = [
            bool(torch.equal(cand[e], tid[e]))
            and float(du[e]) <= 1e-6  # CPU tanh vs CUDA tanhf
            for e in range(b - 1)
            if alive[e]
        ]
        per.append(dict(survived=int(alive.sum()), slots=b - 1, ok=all(ok)))
    return dict(ok=all(g["ok"] for g in per), per_iteration=per)


def stage_check(
    cx: iteration.Context, w: DraftWeights, it: iteration.Iteration, eager: dict, comm
) -> dict:
    """Every draft unit of every iteration against dflash2 fed the kernel's own inputs to that unit, the
    sublayers in the kernel's order (module docstring, stages)."""
    dcfg, b, m = cx.dcfg, it.b, w
    eps, wl, hd = dcfg.rms_eps, dcfg.window_left, dcfg.head_dim
    t = R.TargetShared(
        cx.bufs.embed_shard(cx.cfg).cpu(), cx.hw[: cx.cfg.vocab_rank].cpu()
    )
    bands = {}

    def put(
        name: str, got: torch.Tensor, ref: torch.Tensor, i: int, layer_i: int
    ) -> None:
        r = row_band(got, ref)
        key = f"L{layer_i}_{name}"
        worst = bands.get(key)
        if worst is None or r["rel_max"] > worst["rel_max"]:
            bands[key] = dict(
                r,
                iteration=i,
                rows_rel=[round(float(x), 5) for x in rows_rel(got, ref)],
            )

    def qkv_rows(lw, x_prep, pos):
        qh, kvh = R._heads(dcfg, lw)
        q, k, v = R._mm(x_prep, lw.qkv, BF16).split([qh * hd, kvh * hd, kvh * hd], 1)
        cos, sin = R.rope_cos_sin(pos + torch.arange(b), hd, dcfg.rope_theta)
        qn = R.head_norm_rope(
            q.reshape(b, qh, hd), lw.q_norm, eps, cos, sin, BF16, True
        )
        kn = R.head_norm_rope(
            k.reshape(b, kvh, hd), lw.k_norm, eps, cos, sin, BF16, True
        )
        return qn.reshape(b, qh * hd), kn.reshape(b, kvh * hd), v.reshape(b, kvh * hd)

    entry_resid = []
    for i, rec in enumerate(eager["recs"]):
        P, anchor = (int(rec["words_in"][WORDS[k]]) for k in ("P", "ANCHOR"))
        L = rec["layers"]
        lo = max(0, P - wl)
        pos_k = torch.arange(lo, P + b)
        slots = pos_k % cx.ring
        block = torch.arange(P, P + b) % cx.ring
        # the entry: layer 0's residual = the block's embedding rows (a copy: equal values; the reference's
        # all-reduce of one non-zero partial turns a -0 into +0), ctx_n = hidden_norm(fc(F))
        ids = torch.full((b,), dcfg.mask_id, dtype=torch.int64)
        ids[0] = anchor
        entry_resid.append(torch.equal(L[0]["r_a"], R.embed(ids, t.embed, comm, BF16)))
        feats = torch.cat(
            [rec["caps_in"][k].to(BF16) for k in range(len(cx.capture))], 1
        )
        ctx = R.rms_norm(
            comm.all_gather(R._mm(feats, m.fc, BF16), 1), m.hidden_norm, eps, BF16
        )
        put("entry_ctx_n", rec["ctx_n"], ctx, i, 0)
        # layer 0's input norm
        h_in = R.rms_norm(L[0]["r_a"], m.layers[0].input_norm, eps, BF16)
        x0, _ = R.conv_prepare(m.layers[0].attn_conv, h_in, BF16)
        q0, k0, v0 = qkv_rows(m.layers[0], x0, P)
        put("qkv_q", L[0]["qout"][:, : q0.shape[1]], q0, i, 0)
        put("qkv_k", L[0]["k_ring"][block], k0, i, 0)
        put("qkv_v", L[0]["v_ring"][block], v0, i, 0)
        for li, lw in enumerate(m.layers):
            K = L[li]
            qh = lw.o.shape[1] // hd
            kvh = K["k_ring"].shape[1] // hd
            att = R.attention(
                K["qout"][:, : qh * hd].reshape(b, qh, hd),
                K["k_ring"][slots].view(-1, kvh, hd),
                K["v_ring"][slots].view(-1, kvh, hd),
                P + torch.arange(b),
                pos_k,
                wl,
            )
            put("attn_combine", K["x_attn"], att, i, li)
            _, c_att = R.conv_prepare(lw.attn_conv, h_in, BF16)
            a = R.sublayer_out(
                lw.attn_conv, K["x_attn"], lw.o, c_att, comm, BF16, "pre_ar"
            )
            h2, r_b = R.add_rms_norm(a, K["r_a"], lw.post_norm, eps, BF16)
            put("o_sublayer_r", K["r_b"], r_b, i, li)
            put("o_sublayer_h2", K["h2"], h2, i, li)
            x_gu, c_mlp = R.conv_prepare(lw.mlp_conv, K["h2"], BF16)
            put("mlp_prep", K["x_gu"], x_gu, i, li)
            gate, up = R._mm(K["x_gu"], lw.gate_up, BF16).chunk(2, 1)
            put("gate_up_swiglu", K["act"], R.silu_mul(gate, up, BF16), i, li)
            nx = m.layers[li + 1] if li + 1 < len(m.layers) else None
            d = R.sublayer_out(
                lw.mlp_conv, K["act"], lw.down, c_mlp, comm, BF16, "pre_ar"
            )
            h, r_c = R.add_rms_norm(
                d, K["r_b"], nx.input_norm if nx else m.norm, eps, BF16
            )
            put("down_sublayer_r", K["r_c"], r_c, i, li)
            put("down_sublayer_h", K["h"], h, i, li)
            if nx:
                xq, _ = R.conv_prepare(nx.attn_conv, K["h"], BF16)
                qn, kn, vn = qkv_rows(nx, xq, P)
                N = L[li + 1]
                put("qkv_q", N["qout"][:, : qn.shape[1]], qn, i, li + 1)
                put("qkv_k", N["k_ring"][block], kn, i, li + 1)
                put("qkv_v", N["v_ring"][block], vn, i, li + 1)
                h_in = K["h"]
    bands["entry_resid_equal"] = dict(ok=all(entry_resid), per_iteration=entry_resid)
    return dict(ok=all(v["ok"] for v in bands.values()), worst=bands)


def rank_digests(cx: iteration.Context, run_: dict) -> dict:
    mine = [
        sha(r["y"], r["drafts"], r["words"], r["caps_in"]) for r in run_["recs"]
    ] + [sha(run_["caps"])]
    every = [None] * cx.cfg.tp
    dist.all_gather_object(every, mine)
    return dict(ok=all(e == every[0] for e in every), digests=every[0])


def tree_forward(cx: iteration.Context, window: Window, p: int) -> None:
    """The verify step on a tree window: Stack.forward's launches with the window's rope rows at P + depth (the tree
    table's; an iteration's tree head writes them) after the feeds."""
    st, f, s = cx.stack, cx.feeds, cx.main.cuda_stream
    aux = st.aux_for(window)
    st.feeds(s, True)
    with torch.cuda.stream(cx.main):
        f.cos_sin.copy_(f.cos_sin_cache[p + f.tree[T_DEPTH : T_DEPTH + 16].long()])
    aux.entry(s, False)
    aux.qkv(s, True)
    for launch in st.layers(window):
        launch(s, True)
    aux.lm_head(s, True)
    aux.softcap(s, True)


def plain_step(cx: iteration.Context, its: dict, window: Window, p: int) -> dict:
    """One eager verify step at window position p from the bind-time words (the inputs keep theirs: the weights, the
    target KV, the ids, the routing, a tree window's table): its logits, the capture buffers and every layer's outputs
    (layer_digests).
    """
    reset_all(cx, its)
    synthetic.target_feeds(cx, p)
    cx.published()
    if window.tree:
        tree_forward(cx, window, p)
    else:
        cx.stack.forward(window, cx.main)
    synced(cx, f"plain step {window}")
    return dict(
        logits=cx.bufs.logits.out.cpu(),
        caps=caps_now(cx, cx.cfg.rows),
        digests=layer_digests(cx.cfg, cx.bufs),
    )


def verify_replay(
    cx: iteration.Context, its: dict, it: iteration.Iteration, eager: dict
) -> dict:
    """The last eager iteration's per-row argmax (verify_accept's) = the argmax of a plain verify step's logits on the
    same window: its P, the ids the head wrote, b_live = B, the capture pointers, the target KV the iteration left (the
    replay rewrites the window's rows with the same values) and its routing."""
    last = eager["recs"][-1]
    p = int(last["words_in"][WORDS["P"]])
    got = plain_step(cx, its, it.window, p)
    top1 = R.first_argmax(got["logits"][: it.b])
    return dict(
        ok=bool(torch.equal(top1.to(last["top1"].dtype), last["top1"])), position=p
    )


def tree_rows(
    cx: iteration.Context, its: dict, it: iteration.Iteration, eager: dict
) -> dict:
    """Each row of the last iteration's tree (its ids and table, at its P) against the linear window holding its path
    (module docstring, rows)."""
    p = int(eager["recs"][-1]["words_in"][WORDS["P"]])
    ids = cx.bufs.ids
    tok, tab = ids.cpu(), cx.feeds.tree.cpu()
    depth, mask = (
        tab[T_DEPTH : T_DEPTH + 16].tolist(),
        tab[T_MASK : T_MASK + 16].tolist(),
    )
    parents = set(tab[T_PARENT + 1 : T_PARENT + 16].tolist())
    linear = Window(capture=cx.capture)

    def step(window: Window) -> tuple[torch.Tensor, torch.Tensor]:
        """A verify step's logits and every layer's own post-attention normed rows (h3) [layers, 16, H]."""
        lg = plain_step(cx, its, window, p)["logits"]
        return lg, torch.stack([st.h3 for st in cx.bufs.layers]).float().cpu()

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(rows_rel(a[None], b[None]))

    tree_lg, tree_h3 = step(it.window)
    _, linear_h3 = step(linear)  # the window's ids read as a linear chain
    main_path = [
        r for r in range(16) if depth[r] == r
    ]  # rows 0 .. n of a linear chain sit at depth 0 .. n
    main_path_bitwise, off, spread, argmax_equal = True, {}, 0.0, 0
    path_row0 = {}  # a branch row's path row in layer 0's output
    for leaf in (r for r in range(16) if r not in parents):
        path = [0] + [r for r in range(1, 16) if (mask[leaf] >> r) & 1]
        rest = [r for r in range(16) if r not in path]
        runs = []
        for tail in (
            tok[rest],
            tok[:1].expand(len(rest)),
        ):  # two linear windows with other tails
            ids.copy_(torch.cat([tok[path], tail]).to(ids.device))
            runs.append(step(linear))
        (lg, h3), (lg2, _) = runs
        for d, r in enumerate(path):
            spread = max(spread, rel(lg2[d], lg[d]))
            argmax_equal += int(R.first_argmax(tree_lg[r]) == R.first_argmax(lg[d]))
            if r in main_path:
                main_path_bitwise &= bits_equal(tree_lg[r], lg[d]) and bits_equal(
                    tree_h3[:, r], h3[:, d]
                )
            elif r not in off:
                by_layer = [rel(tree_h3[L, r], h3[L, d]) for L in range(h3.shape[0])]
                off[r] = dict(
                    layer0=by_layer[0],
                    layer0_vs_linear=rel(tree_h3[0, r], linear_h3[0, r]),
                    first_layer_out=next(
                        (L for L, x in enumerate(by_layer) if x > TREE_REL), None
                    ),
                    logits=rel(tree_lg[r], lg[d]),
                )
                path_row0[r] = h3[0, d]
    ids.copy_(tok.to(ids.device))
    control = None
    if off:  # the negative control on the branch row closest to its path
        r = min(off, key=lambda x: off[x]["layer0"])
        table = cx.feeds.tree
        saved = table.clone()
        table[T_MASK + r] = mask[r] & ~(1 << int(tab[T_PARENT + r]))
        moved_rel = rel(step(it.window)[1][0, r], path_row0[r])
        table.copy_(saved)
        control = dict(
            row=r,
            layer0=moved_rel,
            detected=moved_rel > 0 and moved_rel > TREE_SEPARATION * off[r]["layer0"],
        )
    near = all(o["layer0"] <= TREE_REL for o in off.values())
    apart = all(
        o["layer0_vs_linear"] >= TREE_SEPARATION * o["layer0"] for o in off.values()
    )
    return dict(
        ok=main_path_bitwise
        and near
        and apart
        and control is not None
        and control["detected"],
        main_path_rows=len(main_path),
        main_path_bitwise=main_path_bitwise,
        branch_rows={
            r: {k: (round(v, 7) if isinstance(v, float) else v) for k, v in o.items()}
            for r, o in off.items()
        },
        off_layer0_max=max((o["layer0"] for o in off.values()), default=None),
        off_layer0_vs_linear_min=min(
            (o["layer0_vs_linear"] for o in off.values()), default=None
        ),
        control=control,
        linear_spread_max=spread,
        argmax_equal=argmax_equal,
        position=p,
        tol=dict(rel=TREE_REL, separation=TREE_SEPARATION),
    )


def bind_check(cx: iteration.Context, its: dict, p: int) -> dict:
    """The verify's two DFlash bind values on plain steps from the bind-time words (module docstring, bind)."""
    live, cap = LIVE, cx.capture
    plain = plain_step(cx, its, Window(), p)
    with_cap = plain_step(cx, its, Window(capture=cap), p)
    live8 = plain_step(cx, its, Window(b_live=live, capture=cap), p)
    return dict(
        capture_logits=dict(ok=bits_equal(with_cap["logits"], plain["logits"])),
        capture_layers=dict(ok=with_cap["digests"] == plain["digests"]),
        b_live_logits=dict(
            ok=bits_equal(live8["logits"][:live], with_cap["logits"][:live])
        ),
        b_live_caps=dict(
            ok=bits_equal(live8["caps"][:, :live], with_cap["caps"][:, :live])
        ),
        b_live_pads_finite=dict(ok=bool(torch.isfinite(live8["caps"][:, live:]).all())),
    )


def checks(
    cx: iteration.Context,
    w: DraftWeights,
    its: dict,
    b: int,
    n: int,
    graphs: Graphs,
    p0: int,
    p_long: int = 0,
) -> dict:
    it = its[b]
    rings0 = torch.stack([torch.stack([lb.k_ring, lb.v_ring]) for lb in it.lb]).cpu()
    eager = run(cx, its, it, n, p0, None, layers=True)
    comm = GlooComm(cx.rank, cx.cfg.tp)
    res = {
        "reference": reference_check(cx, w, it, eager, rings0, p0, comm),
        "stages": stage_check(cx, w, it, eager, comm),
        "draft_head": draft_head_check(cx, it, eager, comm),
        "ranks": rank_digests(cx, eager),
        "verify": verify_replay(cx, its, it, eager),
    }
    if it.tree:
        res["rows"] = tree_rows(cx, its, it, eager)
        moved = sum(r["compact"]["moved"] for r in eager["recs"])
        res["coverage"] = dict(
            ok=moved > 0 and len(res["rows"]["branch_rows"]) > 0,
            moved=moved,
            branch_rows=len(res["rows"]["branch_rows"]),
        )
    for r in eager["recs"]:
        del r["layers"], r["ctx_n"]
    keys = KEYS + (("table", "out_ring") if it.tree else ())
    g1 = run(cx, its, it, n, p0, graphs)
    g2 = run(cx, its, it, n, p0, graphs)
    res["graph_vs_eager"] = same_run(g1, eager, keys)
    res["graph_two_runs"] = same_run(g2, g1, keys)
    if b == LIVE:
        res["vs_padded"] = same_run(g1, run(cx, its, it, n, p0, graphs, padded=True))
        if p_long:
            # from a long context too (a short context's global attention takes the mma.sync loop, a long one's the
            # tcgen05 path), after every short run: the draft rings carry over between runs, and a long run's ring slots
            # would stand in a later short run's history
            res["vs_padded_long"] = same_run(
                run(cx, its, it, n, p_long, graphs),
                run(cx, its, it, n, p_long, graphs, padded=True),
            )
    res["trace"] = [
        dict(
            words=r["words"].tolist(),
            drafts=r["drafts"].tolist(),
            accept=int(r["words"][WORDS["ACCEPT"]]),
        )
        for r in eager["recs"]
    ]
    return res


def checks_pass(res: dict) -> bool:
    """Every check's ``ok`` (recorded values, ``info``, and the trace are not checks)."""

    def ok(d) -> bool:
        if isinstance(d, dict):
            if d.get("info"):
                return True
            if "ok" in d and not isinstance(d["ok"], dict):
                return bool(d["ok"])
            return all(ok(v) for k, v in d.items() if k != "trace")
        if isinstance(d, list):
            return all(ok(v) for v in d)
        return True

    return ok({k: v for k, v in res.items() if k != "trace"})


# ---- the report ----


def reference_lines(b: int, g: dict) -> list[str]:
    """The reference checks of one block size, then the end-to-end record and its localization."""
    ref, st, sc = g["reference"], g["stages"], g["draft_head"]
    its = ref["iterations"]
    n = len(its)

    def count(key: str, f: str = "ok") -> int:
        return sum(bool(i[key][f]) for i in its)

    worst_stage = max(
        (v["rel_max"], k) for k, v in st["worst"].items() if "rel_max" in v
    )
    layer = [
        (max(x["r"]["rel_max"], x["h"]["rel_max"]), i, li)
        for i, it in enumerate(its)
        for li, x in enumerate(it["forced"]["layer"])
    ]
    e2e = [(it["y_end_to_end"]["rel_max"], i) for i, it in enumerate(its)]
    rel, worst = max(e2e)
    f = its[worst]["forced"]
    fails = [i for i, it in enumerate(its) if not it["y_end_to_end"]["ok"]]
    ring = ref["ring_end_to_end"]
    alive = sum(x["survived"] for x in sc["per_iteration"])
    slots = sum(x["slots"] for x in sc["per_iteration"])
    if "tree" in its[0]:
        tr = g["rows"]
        head = (
            f"head lattice {count('head_scores')}/{n}, tree vs the CPU reference {count('tree')}/{n}; accept"
            f" {count('accept')}/{n}; compact {count('compact')}/{n} (moved rows"
            f" {[i['compact']['moved'] for i in its]}); verify replay {g['verify']['ok']}; rows"
            f" {'PASS' if tr['ok'] else 'FAIL'} (main path {tr['main_path_rows']} rows bitwise {tr['main_path_bitwise']}; branch"
            f" {len(tr['branch_rows'])} rows: layer 0 rel vs path max {tr['off_layer0_max']}, vs the linear reading min"
            f" {tr['off_layer0_vs_linear_min']}; negative control {tr['control']}; linear spread"
            f" {tr['linear_spread_max']:.2e}; argmax equal {tr['argmax_equal']})"
        )
    else:
        head = (
            f"head rerun scores {count('head_scores')}/{n} drafts {count('head_drafts')}/{n}; fp64 rerun drafts"
            f" equal {count('rerun_y', 'drafts_equal')}/{n}; accept {count('accept')}/{n}; window ids"
            f" {count('window_ids')}/{n}; verify replay {g['verify']['ok']}"
        )
    return [
        f"b{b} reference {'PASS' if ref['ok'] and st['ok'] and sc['ok'] else 'FAIL'}: units worst rel"
        f" {worst_stage[0]:.2e} ({worst_stage[1]}), entry resid equal {st['worst']['entry_resid_equal']['ok']}; layers"
        f" worst rel {max(layer)[0]:.2e} (iteration {max(layer)[1]}, layer {max(layer)[2]}); head {sc['ok']}"
        f" ({alive}/{slots} slots survived); {head}",
        f"b{b} end to end (recorded): y within the order band at {n - len(fails)}/{n} (outside: {fails}); worst rel"
        f" {rel:.2e} at iteration {worst} (order spread {its[worst]['y_end_to_end']['orders_rel_max']:.2e}); forced from"
        f" layer 0..4 there: {[round(x, 4) for x in f['y_rel']]}; ring within the order band"
        f" {sum(v['ok'] for v in ring.values())}/{len(ring)}",
    ]


# ---- main ----


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--draft",
        default="",
        help="the DFlash2 draft's directory; omitted: synthetic draft weights",
    )
    ap.add_argument("--blocks", default="8,16")
    ap.add_argument(
        "--tree",
        action="store_true",
        help="the token-tree window (block 16; module docstring)",
    )
    ap.add_argument("--iters", type=int, default=6, help="checked iterations per run")
    ap.add_argument("--ctx", type=int, default=4096, help="the start state's P0")
    ap.add_argument(
        "--long-ctx",
        type=int,
        default=0,
        help="at B = 8 the padded-window check also from this P0 (0: off)",
    )
    ap.add_argument(
        "--routing-scale",
        type=float,
        default=0.0,
        help="expert_bias = scale x the popularity profile (synthetic.set_routing)",
    )
    ap.add_argument("--label", default="dflash")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    blocks = (16,) if args.tree else tuple(int(x) for x in args.blocks.split(","))
    assert (
        not args.draft or (Path(args.draft) / DRAFT_FILE).is_file()
    ), f"--draft {args.draft}: no {DRAFT_FILE}"

    dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=15))
    group, rank, world = dist.group.WORLD, dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.zeros(1, device=device)
    torch.set_num_threads(
        max(4, (os.cpu_count() or 16) // int(os.environ.get("LOCAL_WORLD_SIZE", 4)))
    )
    cfg = Config(tp=world)
    draft = (
        load_shard(str(Path(args.draft) / DRAFT_FILE), rank, world)
        if args.draft
        else synthetic.draft(rank, world)
    )
    source = args.draft or "synthetic"
    max_ctx = max(args.ctx, args.long_ctx) + (2 * args.iters + 4) * cfg.rows
    cx = synthetic.context(draft, max_ctx, rank, device, group, cfg, args.tree)
    # the forced accept: every count in turn (on a tree, every leaf row)
    its = {
        b: iteration.Iteration(b, cx, forced_accepts(b), tree=args.tree) for b in blocks
    }
    for it in its.values():
        synthetic.seed_rings(it)
    graphs = Graphs(cx)
    args.out.mkdir(parents=True, exist_ok=True)
    launches = {
        f"b{b}": dict(draft=len(its[b].draft), iteration=len(its[b].launches()))
        for b in blocks
    }
    mine = dict(
        label=args.label,
        rank=rank,
        host=socket.gethostname(),
        draft=source,
        launches=launches,
        routing_scale=args.routing_scale,
    )
    raw = args.out / f"dflash_rank{rank}_{args.label}.json"

    def bank() -> None:
        raw.write_text(json.dumps(mine, indent=1, default=str))

    # the target's warm-ups (plain verify steps), then the entry's bind constants
    synthetic.set_routing(cfg, cx.weights, args.routing_scale, device)
    synthetic.target_feeds(cx, args.ctx)
    cx.published()
    for _ in range(2):
        cx.stack.forward(Window(), cx.main)
    cx.published()
    for b in blocks:
        begin(cx, its, its[b], args.ctx)
        its[b].bind_constants()
    bank()

    for b in blocks:
        try:
            res = checks(cx, draft, its, b, args.iters, graphs, args.ctx, args.long_ctx)
        except Exception:  # recorded as a FAIL with its traceback
            res = dict(error=traceback.format_exc())
            print(f"### CHECKS b{b} rank {rank} raised:\n{res['error']}")
        mine.setdefault("checks", {})[f"b{b}"] = res
        mine.setdefault("checks_pass", {})[f"b{b}"] = (
            "error" not in res and checks_pass(res)
        )
        bank()
    mine["checks"]["bind"] = bind_check(cx, its, args.ctx)
    mine["checks_pass"]["bind"] = checks_pass(mine["checks"]["bind"])
    bank()
    every_checks = [None] * world
    dist.all_gather_object(every_checks, mine["checks_pass"])
    all_ok = all(all(e.values()) for e in every_checks)
    if rank == 0:
        lines = [
            f"dflash {args.label} draft {source} routing scale {args.routing_scale}"
        ]
        for k, v in launches.items():
            lines.append(
                f"launches {k}: draft {v['draft']}, iteration {v['iteration']}"
            )
        for b in blocks:
            g = mine["checks"][f"b{b}"]
            if "error" in g:
                lines.append(f"b{b} checks raised: {g['error'].splitlines()[-1]}")
                continue
            lines += reference_lines(b, g)
            for key in (
                "ranks",
                "rows",
                "coverage",
                "graph_vs_eager",
                "graph_two_runs",
                "vs_padded",
                "vs_padded_long",
            ):
                if key not in g:
                    continue
                body = json.dumps(
                    {k: v for k, v in g[key].items() if k != "digests"}, default=str
                )
                lines.append(
                    f"b{b} {key}: {'PASS' if checks_pass({key: g[key]}) else 'FAIL'} {body}"
                )
            lines.append(f"b{b} accepts {[t['accept'] for t in g['trace']]}")
        bind = mine["checks"]["bind"]
        lines.append(
            f"bind: {'PASS' if checks_pass(bind) else 'FAIL'} {json.dumps(bind)}"
        )
        lines.append(f"DFLASH CHECKS {'PASS' if all_ok else 'FAIL'} (all ranks)")
        print("\n".join(lines))
        (args.out / f"dflash_{args.label}.json").write_text(
            json.dumps(
                dict(
                    label=args.label,
                    draft=source,
                    launches=launches,
                    checks_pass_per_rank=every_checks,
                    checks_rank0=mine["checks"],
                ),
                indent=1,
                default=str,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    if not all_ok:
        raise SystemExit("DFLASH CHECKS FAIL")


def test_dflash_iteration(tmp_path):
    """This file under torchrun on one node's 4 GPUs (TP4), the synthetic draft unless
    ``SGL_MUSE_MEGA_DRAFT_PATH`` names the published draft's directory."""
    import pytest

    if torch.cuda.device_count() < 4:
        pytest.skip("needs one node of 4 GB300 GPUs")
    draft = os.environ.get("SGL_MUSE_MEGA_DRAFT_PATH")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4"]
        + [__file__, "--out", str(tmp_path), *(["--draft", draft] if draft else [])],
        check=True,
    )


if __name__ == "__main__":
    main()
