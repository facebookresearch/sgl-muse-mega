"""The static step table: the verify step and the DFlash2 iteration on synthetic target weights, timed as CUDA graphs.

    torchrun --nproc-per-node 4 benchmark/bench_verify.py --out DIR [--draft DIR] [--ctxs 4096,128000]

One process holds every row's graph (sharing the weights, the KV and every buffer). Rows: ``verify16`` the 16-row
verify step, ``verify8`` the 8-row images (a block of 8), ``iteration_b<B>`` / ``draft_b<B>`` the DFlash2 iteration
and its draft alone at a block of B (verify_accept forced through every accept count in turn: the synthetic target
never accepts on its own), ``iteration_b<B>_sample`` the sampled iteration (temperature 1, top-k 64: its accept is the
sampled one's on the synthetic target). Each visit starts from bind-time words and the row's start state and replays
the graph ``--steps`` times, each replay between two CUDA events; a row's time is the median replay of a visit, the
median over ``--passes`` visits (the rows visited in turn), then the mean over the ranks. The routing of the synthetic
weights is an expert_bias scale (``--routing-scale``) or the scale bisected to ``--distinct`` distinct experts per
NVFP4 layer over the window's live rows; the table reports the count each row routed to.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "test" / "kernels")]

import synthetic  # noqa: E402
from sgl_muse_mega.muse_spark.config import Config  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash import iteration  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash.weights import (  # noqa: E402
    DRAFT_FILE,
    load_shard,
)
from sgl_muse_mega.muse_spark.sm103.verify import LIVE, Window  # noqa: E402
from synthetic import begin, forced_accepts, reset_all, target_feeds  # noqa: E402

BLOCKS = (8, 16)


def rows_of(name: str) -> int:
    return LIVE if name.endswith(("8", "_b8", "_b8_sample")) else 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--draft", default="", help="the DFlash2 draft's directory; omitted: synthetic"
    )
    ap.add_argument("--ctxs", default="4096", help="comma-separated P0 of the visits")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=32, help="replays per visit")
    ap.add_argument("--routing-scale", type=float, default=0.0)
    ap.add_argument(
        "--distinct",
        type=float,
        default=0.0,
        help="bisect the scale to this count (0: off)",
    )
    ap.add_argument("--label", default="bench")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctxs.split(",")]

    dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=15))
    group, rank, world = dist.group.WORLD, dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.zeros(1, device=device)
    cfg = Config(tp=world)
    draft = (
        load_shard(str(Path(args.draft) / DRAFT_FILE), rank, world)
        if args.draft
        else synthetic.draft(rank, world)
    )
    max_ctx = max(ctxs) + (args.steps + 4) * cfg.rows
    cx = synthetic.context(draft, max_ctx, rank, device, group, cfg)
    its = {
        b: iteration.Iteration(b, cx, forced_accepts(b), sample=True) for b in BLOCKS
    }
    for it in its.values():
        synthetic.seed_rings(it)
    st, main_s = cx.stack, cx.main

    def plain(window: Window):
        return lambda: st.forward(window, main_s)

    synthetic.set_routing(cfg, cx.weights, args.routing_scale, device)
    target_feeds(cx, ctxs[0])
    cx.published()
    for _ in range(2):
        st.forward(Window(), main_s)
    cx.published()
    scale = args.routing_scale
    if args.distinct:
        scale, _ = synthetic.calibrate_routing(
            cfg, cx.weights, cx.bufs, plain(Window()), args.distinct, 12, 2.0, group
        )
    synthetic.set_routing(cfg, cx.weights, scale, device)
    for b in BLOCKS:
        begin(cx, its, its[b], ctxs[0])
        its[b].bind_constants()

    def graph(launches) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main_s):
            for launch in launches:
                launch(main_s.cuda_stream, True)
        return g

    graphs = {
        "verify16": st.capture(Window(), main_s),
        "verify8": st.capture(Window(b_live=LIVE, native=True), main_s),
    }
    for b in BLOCKS:
        graphs[f"iteration_b{b}"] = graph(its[b].launches("iter"))
        graphs[f"iteration_b{b}_sample"] = graph(its[b].launches("iter", sample=True))
        graphs[f"draft_b{b}"] = graph(its[b].launches("draft"))

    def visit(name: str, p0: int) -> float:
        if name.startswith("verify"):
            reset_all(cx, its)
            target_feeds(cx, p0)
            cx.published()
        else:
            it = its[int(name.split("_b")[1].removesuffix("_sample"))]
            begin(cx, its, it, p0)
            if name.endswith("_sample"):
                it.sampling(1, p0, 64, 1.0, 1.0)
                torch.cuda.synchronize(device)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * args.steps)]
        with torch.cuda.stream(main_s):
            for k in range(args.steps):
                ev[2 * k].record(main_s)
                graphs[name].replay()
                ev[2 * k + 1].record(main_s)
        torch.cuda.synchronize(device)
        return statistics.median(
            1e3 * ev[2 * k].elapsed_time(ev[2 * k + 1]) for k in range(args.steps)
        )

    mine = dict(
        label=args.label, rank=rank, host=socket.gethostname(), scale=scale, rows={}
    )
    for p0 in ctxs:
        for name in graphs:  # replayed once outside the passes
            visit(name, p0)
        meds = {name: [] for name in graphs}
        for _ in range(args.passes):
            for name in graphs:
                meds[name].append(visit(name, p0))
        distinct = {}
        for name in graphs:
            if not name.startswith("draft"):
                visit(name, p0)
                distinct[name] = synthetic.distinct_experts(cfg, cx.bufs, rows_of(name))
        mine["rows"][p0] = {
            n: dict(us=statistics.median(v), visits=v, distinct=distinct.get(n))
            for n, v in meds.items()
        }
    every = [None] * world
    dist.all_gather_object(every, mine)
    if rank == 0:
        lines = [
            f"bench {args.label} TP{world} draft {args.draft or 'synthetic'} routing scale {scale:.4g}"
        ]
        for p0 in ctxs:
            lines.append(
                f"P0 {p0}: us per replay (median over visits, mean over {world} ranks)"
            )
            for name in graphs:
                us = statistics.mean(e["rows"][p0][name]["us"] for e in every)
                d = mine["rows"][p0][name]["distinct"]
                routed = (
                    f"  {d:.1f} distinct experts over {rows_of(name)} rows" if d else ""
                )
                lines.append(f"  {name:14s} {us:9.1f}{routed}")
        print("\n".join(lines))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"bench_{args.label}.json").write_text(
            json.dumps(every, indent=1, default=str)
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
