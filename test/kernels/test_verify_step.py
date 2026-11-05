"""The verify step's correctness against the torch reference (``reference.muse_spark``), stage by stage, and its
determinism, on 4 ranks (one GB300 node, TP4) or 8 (two nodes, TP8), synthetic weights:

    torchrun --nproc-per-node 4 test/kernels/test_verify_step.py --ctx 4096 --layers 0,1,2,4,5,60,61 --out DIR

1. One eager step with a device sync after every launch: each checked layer's inputs are read before its launch and
   its stored values after (x_attn, the post-attention stream / h3, the router partials, idx32 / topk_w, the expert input,
   y, the next-layer phase's stream / h3 / successor qkv row and KV rows). Every reference stage is fed the kernel's own
   inputs to that stage (``reference.ref_layer(forced=...)``), so a stage is held to its own rounding, not to an
   upstream quantizer's amplification. The entry, layer 0's head, the feeds and the logits are checked the same way.
2. The captured graph replayed twice: every layer's outputs bitwise equal across replays and equal to the eager step.
``--m8``: the 8-row window on its own images (the attention at 8 rows too), every stage on rows 0-7 against the
reference's 8-row window (positions ctx .. ctx + 7); the feeds stay the 16-row layout's. 3. Then the padded 16-row
window (b_live 8, the pad rows repeating the live ids) from fresh words, eager and replayed: the live rows of every
layer's outputs (stream, h3, h3f, idx32, topk_w), the final rows and the logits bitwise the 8-row images'.
Per-rank results go to DIR/check_rank<r>_<label>.json; rank 0 writes the table and check_<label>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE.parents[1]),
    str(HERE.parent),
]  # the package, test/ (the references)

import synthetic  # noqa: E402
from reference import muse_spark as R  # noqa: E402
from sgl_muse_mega.muse_spark.config import Config  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.artifacts import LAYER_IMAGES, build  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.buffers import Buffers  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.verify import (  # noqa: E402
    LIVE,
    M8Stack,
    Stack,
    Window,
    layer_libraries,
    share_gather_rows,
)
from sgl_muse_mega.muse_spark.sm103.weights import ModelWeights  # noqa: E402

# the expert input's NVFP4 codes may differ from the reference's only where the pooled row (computed in another fp32
# order, and not readable) put a value across a rounding boundary: at most this fraction of elements
FLIP_FRAC = 0.01


def boundary_flips(kq, ksf, rq, rsf) -> dict:
    """The kernel's quantized expert input against the reference's quantization of its own row: every scale byte within
    one e4m3 step, and where a group's scale agrees every code within one e2m1 magnitude step (same sign, or a zero on
    either side); max_rel = the fraction of differing elements."""
    sf_step = (ksf.long() - rsf.long()).abs()
    kc, rc = R.unpack_nibbles(kq).long(), R.unpack_nibbles(rq).long()
    same_sf = (sf_step == 0).repeat_interleave(R.NVFP4_GROUP, dim=-1)
    km, rm = kc & 7, rc & 7
    adjacent = ((km - rm).abs() <= 1) & (((kc ^ rc) & 8 == 0) | (km == 0) | (rm == 0))
    frac = float((kc != rc).float().mean())
    ok = (
        int(sf_step.max()) <= 1
        and bool((~same_sf | adjacent).all())
        and frac <= FLIP_FRAC
    )
    return dict(max_abs=float(sf_step.max()), max_rel=frac, ok=ok, tol=FLIP_FRAC)


def ref_weights(cfg: Config, w: ModelWeights, L: int) -> R.LayerWeights:
    lw = w.layers[L]
    if cfg.is_bf16(L):
        experts = R.Bf16Experts(lw.w1, lw.w2)
    else:
        experts = R.Fp4Experts(
            lw.w1,
            lw.w1_sf_blk,
            lw.w2,
            lw.w2_sf,
            lw.g1_alphas,
            lw.g1_scale_c,
            lw.g2_alphas,
        )
    return R.LayerWeights(
        w_o=lw.o_proj,
        post_alpha=lw.post_alpha,
        post_beta=lw.post_beta,
        post_pre_w=lw.post_pre_w,
        w_pool=lw.pooler,
        w_router=lw.router,
        expert_bias=lw.expert_bias,
        router_mult=cfg.output_multiplier,
        route_scale=cfg.route_scale,
        pre_gamma=lw.pre_gamma,
        pre_eps=cfg.rms_eps,
        experts=experts,
        gamma_p1=lw.gamma_p1,
        w_unpool=lw.unpooler,
        x_alpha=lw.cross_alpha,
        x_beta=lw.cross_beta,
        x_pre_w=lw.cross_pre_w,
        x_post_w=lw.cross_post_w,
    )


def router_logits(
    ws: torch.Tensor, split: int, rows: int, experts: int
) -> torch.Tensor:
    """The router's logits from its K-slice partials, ws[((slice * 2 + n-block) * rows + t) * 128 + col], summed in
    slice order in fp32 as the top-k row sums them (postattn.cuh)."""
    n = experts // 128
    p = ws[: split * n * rows * 128].view(split, n, rows, 128)
    acc = p[0].clone()
    for s in range(1, split):
        acc = acc + p[s]
    return acc.permute(1, 0, 2).reshape(rows, experts)


class Checked:
    """One step's launches (``window``'s: the 16-row window, or the 8-row one's own images), eager, a sync after each;
    the checked layers' inputs and stored values (the window's ``rows``) read around their launches.
    """

    def __init__(
        self, cfg: Config, stack: Stack, bufs: Buffers, ctx: int, layers, window: Window
    ):
        self.cfg, self.stack, self.b, self.ctx, self.check = (
            cfg,
            stack,
            bufs,
            ctx,
            set(layers),
        )
        self.window = window
        self.rows = LIVE if window.native else cfg.rows
        self.inputs: dict[int, dict] = {}
        self.outputs: dict[int, dict] = {}

    def _kv_rows(self, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.stack.kv[L]
        w = slice(self.ctx, self.ctx + self.rows)
        return k[w].clone(), v[w].clone()

    def _qkv(self, cls: str) -> torch.Tensor:
        q = self.b.attn_types[cls].qkv_out.view(self.cfg.rows, self.cfg.qkv_row)
        return q[: self.rows].clone()

    def _readout(self, L: int, consts) -> dict:
        cfg, b = self.cfg, self.b
        st, cls, m = b.layers[L], cfg.attn_type(L), self.rows
        mem = b.moe_variants["bf16" if cfg.is_bf16(L) else "fp4"]
        o = dict(
            x_attn=b.attn_types[cls].x_attn.view(cfg.rows, cfg.oproj_k)[:m].clone(),
            stream_mid=st.stream_out[:m].clone(),
            h3=st.h3[:m].clone(),
            h3f=st.h3f[:m].clone(),
            # the K-window variant's ws holds this rank's partials only: its logits row is the all-reduced sum
            logits=(
                st.logits[:m].clone()
                if consts.pooled_ready_words
                else router_logits(st.ws, consts.router_split, m, cfg.experts)
            ),
            ids=st.idx32[:m].clone(),
            w=st.w[:m].clone(),
            y=mem.y[:m].clone(),
        )
        if cfg.is_bf16(L):
            o["x"] = mem.q[:m].clone()
        else:
            o.update(
                xq=mem.q[:m].clone(),
                x_sf=mem.sf[:m].clone(),
                x_scale=mem.x_scale[:m].clone(),
            )
        if L == cfg.layers - 1:
            o.update(
                stream_out=st.final_stream[:m].clone(), h3_next=st.final_h3[:m].clone()
            )
        else:
            sc = cfg.attn_type(L + 1)
            o.update(
                stream_out=b.attn_types[sc].stream_out[:m].clone(),
                h3_next=b.attn_types[sc].h3[:m].clone(),
                qkv_next=self._qkv(sc),
                kv_next=self._kv_rows(L + 1),
            )
        return o

    def run(self) -> None:
        s, cfg, st, m = (
            torch.cuda.current_stream().cuda_stream,
            self.cfg,
            self.stack,
            self.rows,
        )
        sync = torch.cuda.synchronize
        # the 8-row window's entry, qkv head, lm_head and softcap are its own (bound at 8 rows); the feeds are the
        # 16-row layout's
        aux = st.aux_for(self.window)
        entry, qkv, lm_head, softcap = aux.entry, aux.qkv, aux.lm_head, aux.softcap
        st.feeds(s, True)
        entry(s, False)
        sync()
        self.entry = dict(
            residual=self.b.residual[:m].clone(), normed=self.b.normed[:m].clone()
        )
        qkv(s, True)
        sync()
        self.head = dict(qkv=self._qkv(cfg.attn_type(0)), kv=self._kv_rows(0))
        layers = st.layers(self.window)
        for L, launch in enumerate(layers):
            if L in self.check:
                cls = cfg.attn_type(L)
                self.inputs[L] = dict(
                    qkv=self._qkv(cls),
                    stream_in=(
                        self.b.residual if L == 0 else self.b.attn_types[cls].stream_out
                    )[:m].clone(),
                )
            launch(s, True)
            sync()
            if L in self.check:
                self.outputs[L] = self._readout(L, st.libs[cfg.image(L)].constants)
        lm_head(s, True)
        softcap(s, True)
        sync()
        self.logits = self.b.logits.out[:m].clone()


def live_digests(cfg: Config, bufs: Buffers, rows: int) -> list[str]:
    """Every layer's live-row outputs (stream, h3, h3f, idx32, topk_w), then the final rows and the logits."""

    def sha(*ts) -> str:
        h = hashlib.sha256()
        for t in ts:
            h.update(t[:rows].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    out = [sha(st.stream_out, st.h3, st.h3f, st.idx32, st.w) for st in bufs.layers]
    last = bufs.layers[cfg.layers - 1]
    return out + [sha(last.final_stream, last.final_h3), sha(bufs.logits.out)]


def layer_digests(cfg: Config, bufs: Buffers) -> list[str]:
    """Every layer's own stored outputs (stream, h3, h3f, idx32, topk_w), then the step's final rows and logits."""

    def sha(*ts) -> str:
        h = hashlib.sha256()
        for t in ts:
            h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    out = [sha(st.stream_out, st.h3, st.h3f, st.idx32, st.w) for st in bufs.layers]
    last = bufs.layers[cfg.layers - 1]
    return out + [sha(last.final_stream, last.final_h3), sha(bufs.logits.out)]


class Results:
    """``put(name, got, ref, tol=None)``: one comparison into ``res`` (reference.compare; ``TOLERANCES[name]`` by
    default, integers exact)."""

    def __init__(self, res: dict):
        self.res = res

    def __call__(self, name, got, ref, tol=None) -> None:
        a, r, ok = R.compare(got, ref, name, tol)
        self.res[name] = dict(
            max_abs=a,
            max_rel=r,
            ok=ok,
            tol=R.TOLERANCES.get(name, 0.0) if tol is None else tol,
        )


def compare_layer(cfg, w, chk: Checked, L: int, comm, feeds) -> dict:
    """The reference stages of layer L fed the kernel's own stage inputs -> {value: (max_abs, max_rel, ok, tol)}."""
    i, k = chk.inputs[L], chk.outputs[L]
    ctx, cls = chk.ctx, cfg.attn_type(L)
    kv_ctx = [
        t[:ctx].view(ctx, cfg.kv_heads_rank, cfg.head_dim) for t in chk.stack.kv[L]
    ]
    inputs = R.LayerInputs(
        qkv=i["qkv"],
        k_ctx=kv_ctx[0],
        v_ctx=kv_ctx[1],
        stream_in=i["stream_in"],
        window=cfg.swa_window if cls == "swa" else None,
    )
    nxt = None
    if L < cfg.layers - 1:
        nxt = R.NextHead(
            w_qkv=w.layers[L + 1].qkv,
            q_scale=cfg.q_scale,
            eps=cfg.rms_eps,
            cos_sin=(
                feeds.cos_sin[: chk.rows] if cfg.attn_type(L + 1) == "swa" else None
            ),
            q_heads=cfg.q_heads_rank,
            kv_heads=cfg.kv_heads_rank,
        )
    forced = {n: k[n] for n in R.LAYER_VALUES if n in k}
    o = R.ref_layer(ref_weights(cfg, w, L), inputs, comm, nxt, forced=forced)
    res = {}

    put = Results(res)

    for name in (
        "x_attn",
        "stream_mid",
        "h3",
        "logits",
        "ids",
        "w",
        "stream_out",
        "h3_next",
    ):
        put(name, k[name], o[name])
    put("h3f", k["h3f"], k["h3"].float(), 0.0)  # the router's fp32 copy of h3, exact
    if cfg.is_bf16(L):
        put("x", k["x"], o["x"])
    else:
        put("x_scale", k["x_scale"], o["x_scale"])
        for name in ("xq", "x_sf"):
            put(name, k[name], o[name])
        put(
            "x_deq",
            R.dequant_nvfp4(k["xq"], k["x_sf"]) * k["x_scale"][:, None],
            R.dequant_nvfp4(o["xq"], o["x_sf"]) * o["x_scale"][:, None],
            R.TOLERANCES["x"],
        )
        res["x_codes"] = boundary_flips(k["xq"], k["x_sf"], o["xq"], o["x_sf"])
    # y crosses the MoE phase's h requant (zb is not readable): zb's bound
    put("y", k["y"], o["y"], R.TOLERANCES["zb"])
    if nxt is not None:
        put("qkv_next", k["qkv_next"], o["qkv_next"])
        kv_cols = cfg.kv_heads_rank * cfg.head_dim
        qk = cfg.q_heads_rank * cfg.head_dim
        # the appended KV rows are the qkv row's k / v columns
        for j, name in enumerate(("k_rows", "v_rows")):
            col = qk + j * kv_cols
            put(name, k["kv_next"][j], k["qkv_next"][:, col : col + kv_cols], 0.0)
    # the quantizer's input is not readable: x_codes checks it
    for name in (
        "xq",
        "x_sf",
        "x_deq",
    ):
        if name in res:
            res[name]["gating"] = False
    return res


def check_edges(cfg, w, chk: Checked, comm, bufs: Buffers) -> dict:
    """The feeds (integer gathers, the 16-row layout's), the fused entry, layer 0's head and its KV rows, the logits
    tail (the window's rows)."""
    res, f, ctx, m = {}, chk.stack.f, chk.ctx, cfg.rows

    put = Results(res)

    pos = torch.arange(ctx, ctx + m, device=f.cos_sin.device)
    put("feeds_cos_sin", f.cos_sin, f.cos_sin_cache[pos], 0.0)
    put("feeds_loc_full", f.loc_full.long(), pos)
    put("feeds_loc_swa", f.loc_swa.long(), pos)
    put("feeds_kv_len", f.kv_len.long(), torch.tensor([ctx + m], device=pos.device))
    span = torch.arange(ctx + m, device=pos.device) * f.kv_row_bytes
    put("feeds_pt_full", f.pt_full[0, : ctx + m], span)
    put("feeds_pt_swa", f.pt_swa[0, : ctx + m], span)
    rows = chk.rows
    e = R.ref_entry(
        bufs.ids[:rows],
        bufs.embed_shard(cfg),
        w.entry_pre_w,
        comm,
        cfg.rms_eps,
        cfg.rms_eps,
    )
    put("residual", chk.entry["residual"], e["residual"])
    put("normed", chk.entry["normed"], e["normed"])
    head = R.qkv_head(
        chk.entry["normed"],
        w.layers[0].qkv,
        cfg.q_heads_rank,
        cfg.kv_heads_rank,
        cfg.q_scale,
        cfg.rms_eps,
        f.cos_sin[:rows] if cfg.attn_type(0) == "swa" else None,
    )
    put("head_qkv", chk.head["qkv"], head, R.TOLERANCES["qkv_next"])
    kv_cols, qk = cfg.kv_heads_rank * cfg.head_dim, cfg.q_heads_rank * cfg.head_dim
    for j, name in enumerate(("head_k_rows", "head_v_rows")):
        col = qk + j * kv_cols
        put(name, chk.head["kv"][j], chk.head["qkv"][:, col : col + kv_cols], 0.0)
    last = bufs.layers[cfg.layers - 1]
    put(
        "logits_out",
        chk.logits,
        R.ref_logits(
            last.final_h3[:rows], w.lm_head, comm, cfg.output_multiplier, cfg.softcap
        ),
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="held context (positions of the window's first row)",
    )
    ap.add_argument(
        "--layers",
        default="0,1,2,4,5,60,61",
        help="comma list of checked layers, or 'all'",
    )
    ap.add_argument(
        "--routing-scale",
        type=float,
        default=0.0,
        help="expert_bias = scale x the popularity profile",
    )
    ap.add_argument(
        "--m8",
        action="store_true",
        help="the 8-row window on its own images",
    )
    ap.add_argument("--label", default="check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    window = Window(b_live=LIVE, native=True) if args.m8 else Window()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = Config(tp=world)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.zeros(1, device=device)
    layers = (
        range(cfg.layers)
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )

    paths = build(m8=args.m8, world=cfg.world)
    libs = layer_libraries(paths)
    consts = {image: libs[image].constants for image in LAYER_IMAGES}
    bufs = Buffers(cfg, consts, device, dist.group.WORLD)
    synthetic.seed_ids(bufs, cfg)
    weights = synthetic.model_weights(cfg, rank, device, bufs.embed_shard(cfg))
    kv = synthetic.kv_fill(cfg, rank, (args.ctx + cfg.rows + 127) // 128 * 128, device)
    feeds = synthetic.identity_feeds(cfg, device)
    synthetic.set_context(feeds, args.ctx, cfg.rows)
    if args.m8:
        bufs.ids[LIVE:] = bufs.ids[
            :LIVE
        ]  # the padded window's pads repeat the live ids
        stack = M8Stack(cfg, paths, libs, bufs, weights, rank, device, kv, feeds)
        share_gather_rows(bufs, consts["swa"], stack.libs8["swa"].constants)
    else:
        stack = Stack(cfg, paths, libs, bufs, weights, rank, device, kv, feeds)
    synthetic.set_routing(cfg, weights, args.routing_scale, device)
    comm = R.DistComm(cfg.node_ranks)
    main_s = torch.cuda.Stream(device)

    torch.cuda.synchronize()
    dist.barrier()
    # one plain eager step first: the checked step is the second
    stack.forward(window, main_s)
    torch.cuda.synchronize()
    dist.barrier()
    chk = Checked(cfg, stack, bufs, args.ctx, layers, window)
    with torch.cuda.stream(main_s):
        chk.run()
    eager = layer_digests(cfg, bufs)

    g = stack.capture(window, main_s)
    dist.barrier()
    replays = []
    for _ in range(2):
        with torch.cuda.stream(main_s):
            g.replay()
        torch.cuda.synchronize()
        replays.append(layer_digests(cfg, bufs))

    results = {"edges": check_edges(cfg, weights, chk, comm, bufs)}
    for L in layers:
        results[f"L{L}"] = compare_layer(cfg, weights, chk, L, comm, feeds)
    distinct = {L: int(torch.unique(chk.outputs[L]["ids"]).numel()) for L in layers}
    det = dict(
        replays_equal=replays[0] == replays[1],
        eager_equals_graph=eager == replays[0],
        first_diff_layer=next(
            (i for i, (a, b) in enumerate(zip(eager, replays[0])) if a != b), None
        ),
        digests_replay=replays[0],
    )
    if args.m8:
        native = live_digests(cfg, bufs, LIVE)
        padded = Window(b_live=LIVE)
        torch.cuda.synchronize()
        dist.barrier()
        bufs.reset(dist.group.WORLD)
        stack.forward(padded, main_s)
        gp = stack.capture(padded, main_s)
        dist.barrier()
        with torch.cuda.stream(main_s):
            gp.replay()
        torch.cuda.synchronize()
        det["padded_live_equal"] = live_digests(cfg, bufs, LIVE) == native
    args.out.mkdir(parents=True, exist_ok=True)
    mine = dict(
        label=args.label,
        rank=rank,
        host=socket.gethostname(),
        ctx=args.ctx,
        window=str(window),
        routing_scale=args.routing_scale,
        distinct_experts=distinct,
        determinism=det,
        results=results,
    )
    (args.out / f"check_rank{rank}_{args.label}.json").write_text(
        json.dumps(mine, indent=1)
    )
    every = [None] * world
    dist.all_gather_object(every, mine)
    if rank == 0:
        table, ok_all = [], True
        for key in results:
            for name, r in results[key].items():
                worst = max(e["results"][key][name]["max_rel"] for e in every)
                ok = all(e["results"][key][name]["ok"] for e in every)
                gating = r.get("gating", True)
                ok_all &= ok or not gating
                table.append((key, name, worst, r["tol"], ok, gating))
        det_all = all(e["determinism"]["replays_equal"] for e in every)
        eq_all = all(e["determinism"]["eager_equals_graph"] for e in every)
        pad_all = all(e["determinism"].get("padded_live_equal", True) for e in every)
        lines = [
            f"check {args.label} ctx {args.ctx} rows {chk.rows} {window}: stage values vs reference (max_rel = worst rank)"
        ]
        lines += [
            f"{k:6s} {n:14s} max_rel {w:10.3e} tol {t:8.1e} {'ok' if o else 'FAIL'}{'' if gt else ' (info)'}"
            for k, n, w, t, o, gt in table
        ]
        lines.append(f"distinct experts per checked layer (rank 0): {distinct}")
        lines.append(
            f"replays bitwise equal on all ranks: {det_all}; eager step == graph replay on all ranks: {eq_all}"
        )
        if args.m8:
            lines.append(
                f"live rows bitwise the padded 16-row window's on all ranks: {pad_all}"
            )
        passed = ok_all and det_all and eq_all and pad_all
        lines.append(f"CHECK {'PASS' if passed else 'FAIL'}")
        print("\n".join(lines))
        (args.out / f"check_{args.label}.json").write_text(
            json.dumps(
                dict(
                    ctx=args.ctx,
                    passed=passed,
                    replays_equal=det_all,
                    eager_equals_graph=eq_all,
                    table=table,
                ),
                indent=1,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and not passed:
        raise SystemExit("CHECK FAIL")


def test_verify_step(tmp_path):
    """This file under torchrun on one node's 4 GPUs (TP4): the 16-row window and the 8-row images."""
    import pytest

    if torch.cuda.device_count() < 4:
        pytest.skip("needs one node of 4 GB300 GPUs")
    for extra in ([], ["--m8"]):
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4"]
            + [__file__, "--out", str(tmp_path), *extra],
            check=True,
        )


if __name__ == "__main__":
    main()
