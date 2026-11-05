"""The sampled DFlash2 step and the draft-pool sync on 4 ranks (one GB300 node, TP4) or 8 (two nodes, TP8), synthetic
target weights and the published draft (or a synthetic one), SGLang importable:

    torchrun --nproc-per-node 4 test/kernels/test_dflash_sample.py --out DIR [--draft DIR]

  accept   sample_accept alone on crafted inputs (synthetic.sampled_case, written in place), per sampling
           configuration (synthetic.SAMPLING_CONFIGS), block verification and token-wise, B = 16 and 8: each trial's
           accept and bonus = SGLang's chain sampling on the same logits, q and uniforms
           (build_speculative_verify_target_probs or the temperature softmax, chain_speculative_sampling_triton) and
           = reference.dflash2.sampled_accept (fp64), but where the bonus's uniform lands within BOUNDARY of the mass
           between the two tokens in the fp64 CDF (two fp32 CDF sums over a flat row part at a token boundary)
  walk     the sampled draft (the draft launches with the sampled head) at three temperatures: its drafts and draws =
           SGLang's selector_walk_triton on the head's own scores and candidates with the same uniforms, its q within
           1e-6
  graph    ITERS sampled iterations from one start state as one CUDA graph: bitwise the eager run (words, out ring,
           drafts per iteration)
  ranks    those iterations' words, out ring and drafts bitwise equal on every rank
  sync     a draft pool beside the rings (identity slots): after the iterations (draft_kv_sync as each one's last
           launch) and a flush (the draft alone, then draft_kv_sync at lag 0), the pool's row of every committed
           position = the ring's, and no other row written
Per-rank raws: DIR/sample_rank<r>_<label>.json; rank 0: DIR/sample_<label>.json and the summary lines.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

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
from sgl_muse_mega.muse_spark.sm103.dflash.verify_accept import (  # noqa: E402
    SampleAccept,
)
from sgl_muse_mega.muse_spark.sm103.dflash.weights import (  # noqa: E402
    DRAFT_FILE,
    load_shard,
)
from sgl_muse_mega.muse_spark.sm103.verify import Window  # noqa: E402
from synthetic import ANCHOR, TOP_K_ALL, begin, sampled_case  # noqa: E402
from test_dflash_iteration import bits_equal, checks_pass, sha, synced  # noqa: E402

TRIALS = 48
# a bonus split is a CDF boundary when the fp64 CDF between the two tokens passes the target within this fraction of
# the residual's mass: two fp32 sums of up to 202,048 terms differ by about 1e-6 of it
BOUNDARY = 1e-5
WALK_TEMPS = (0.5, 1.0, 1.5)
ITERS = 8


def sglang_accept(
    dev, logits, tokens, cand, q, u, temperature, top_k, top_p, block
) -> tuple[int, int]:
    """SGLang's DFlash2 sampled accept (dspark_accept._accept_sampling_core) on these inputs and uniforms."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )
    from sglang.srt.speculative.dflash_utils import (
        _get_or_create_chain_verify_buffers,
        build_speculative_verify_target_probs,
    )

    logits, tokens, cand, q, u = (t.to(dev) for t in (logits, tokens, cand, q, u))
    b = tokens.shape[0]
    info = SimpleNamespace(
        temperatures=torch.full((1, 1), temperature, dtype=torch.float32, device=dev),
        top_ks=torch.full((1,), top_k, dtype=torch.int32, device=dev),
        top_ps=torch.full((1,), top_p, dtype=torch.float32, device=dev),
        need_top_k_sampling=top_k != TOP_K_ALL,
        need_top_p_sampling=top_p != 1.0,
    )
    if not info.need_top_k_sampling and not info.need_top_p_sampling:
        probs = SoftmaxTemp.execute(
            logits=logits, temperatures=info.temperatures, rows_per_request=b
        ).view(1, b, -1)
    else:
        probs = build_speculative_verify_target_probs(
            next_token_logits=logits,
            sampling_info=info,
            draft_token_num=b,
            bs=1,
            max_top_k=top_k,
            uniform_top_k_value=top_k,
        )
    draft = torch.zeros(1, b - 1, logits.shape[1], dtype=torch.float32, device=dev)
    draft[0].scatter_(1, cand.long(), q)
    ri, rnt, rns, pred, acc_idx, acc_num = _get_or_create_chain_verify_buffers(
        bs=1, draft_token_num=b, device=dev
    )
    chain_speculative_sampling_triton(
        predicts=pred,
        accept_index=acc_idx,
        accept_token_num=acc_num,
        candidates=tokens[None].long().contiguous(),
        retrive_index=ri,
        retrive_next_token=rnt,
        retrive_next_sibling=rns,
        uniform_samples=u[None, : b - 1].contiguous(),
        uniform_samples_for_final_sampling=u[R.FINAL_UNIFORM :].contiguous(),
        target_probs=probs,
        draft_probs=draft,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
        block_verification=block,
    )
    a = int(acc_num[0])
    return a, int(pred[int(acc_idx[0, a])])


def boundary_margin(probs, cand, q, ref: R.SampledAccept, gamma: int, u, ids) -> float:
    """|the fp64 CDF of the bonus's distribution at the boundary between two ids - its target| / its mass."""
    p = probs[ref.accept_len].double()
    if ref.accept_len < gamma:
        dense = torch.zeros(p.shape[0], dtype=torch.float64)
        dense[cand[ref.accept_len].long()] = q[ref.accept_len].double()
        p = torch.clamp(ref.scale * p - dense, min=0.0)
    cdf = torch.cumsum(p, 0)
    return abs(float(cdf[min(ids)]) - float(u[R.FINAL_UNIFORM]) * ref.norm) / ref.norm


def accept_check(
    cx, it: iteration.Iteration, sa: SampleAccept, block: bool, p0: int
) -> dict:
    """sample_accept alone on crafted inputs vs SGLang's chain sampling (module docstring)."""
    b, h, dev = it.b, it.head_out, cx.device
    g = torch.Generator(device="cpu").manual_seed(17 + b + 2 * block)
    out = {}
    for ci, (name, cfg) in enumerate(synthetic.SAMPLING_CONFIGS.items()):
        temperature, top_k, top_p = cfg
        rows = []
        for t in range(TRIALS):
            cand, q, path, tokens, logits = sampled_case(g, b, cx.cfg.vocab, ANCHOR)
            seed = 1000 * ci + t
            cx.bufs.logits.out[:b].copy_(logits)
            h.cand.copy_(cand)
            h.q.copy_(q)
            h.path.copy_(path.int())
            cx.bufs.ids[:b].copy_(tokens)
            words = torch.zeros_like(it.words)
            words[WORDS["P"]], words[WORDS["ITER"]] = p0, t
            it.words.copy_(words)
            it.sampling(seed, p0, top_k, temperature, top_p)
            torch.cuda.synchronize(dev)
            sa.launch(cx.main.cuda_stream, False)
            synced(cx, "sample_accept")
            got = (int(it.words[WORDS["ACCEPT"]]), int(it.words[WORDS["ANCHOR"]]))
            u = R.uniforms(seed, p0, t, R.PURPOSE_ACCEPT)
            want = sglang_accept(dev, logits, tokens, cand, q, u, *cfg, block)
            probs = R.target_probs(logits, *cfg)
            ref = R.sampled_accept(tokens, probs, cand, q, u, block)
            row = dict(
                got=got, sglang=want, ref=(ref.accept_len, ref.bonus), margins=[]
            )
            for other in {want, row["ref"]} - {got}:
                row["margins"].append(
                    boundary_margin(probs, cand, q, ref, b - 1, u, (got[1], other[1]))
                    if other[0] == got[0]
                    else float("inf")
                )
            rows.append(row)
        accepts = [r["got"][0] for r in rows]
        splits = [m for r in rows for m in r["margins"]]
        out[name] = dict(
            # identical uniforms: the same accept, and the same bonus but at a CDF boundary
            ok=all(m <= BOUNDARY for m in splits),
            max_margin=max(splits, default=0.0),
            trials=TRIALS,
            vs_sglang_differ=sum(r["got"] != r["sglang"] for r in rows),
            vs_reference_differ=sum(r["got"] != r["ref"] for r in rows),
            accept_mean=sum(accepts) / len(accepts),
            accept_max=max(accepts),
            rows=rows,
        )
    return out


def walk_check(cx, it: iteration.Iteration, p0: int) -> dict:
    """The sampled draft's walk vs SGLang's selector_walk_triton (module docstring)."""
    from sglang.kernels.ops.speculative.dflash import selector_walk_triton

    b, h, dev, s = it.b, it.head_out, cx.device, cx.main.cuda_stream
    trials = []
    for i, temperature in enumerate(WALK_TEMPS):
        seed, it_word = 77 + i, 3 + i
        words = it.words.clone()
        words[WORDS["ITER"]] = it_word
        it.words.copy_(words)
        it.sampling(seed, p0, TOP_K_ALL, temperature, 1.0)
        torch.cuda.synchronize(dev)
        for launch in it.launches("draft", sample=True):
            launch(s, True)
        synced(cx, "sampled draft")
        u = R.uniforms(seed, p0, it_word, R.PURPOSE_WALK).to(dev)
        tokens, q_rows = selector_walk_triton(
            candidate_ids=h.cand[None].long(),
            scores=h.scores[None],
            uniforms=u[None, : b - 1].contiguous(),
            temperatures=torch.tensor(
                [max(temperature, 1e-5)], dtype=torch.float32, device=dev
            ),
            greedy_mask=torch.zeros(1, dtype=torch.bool, device=dev),
        )
        drafts = h.drafts.cpu()
        anchor = torch.tensor([int(it.words[WORDS["ANCHOR"]])])
        trials.append(
            dict(
                T=temperature,
                drafts_ok=torch.equal(tokens[0].cpu(), drafts),
                path_ok=torch.equal(
                    h.cand.long().gather(1, h.path.long()[:, None])[:, 0].cpu(), drafts
                ),
                ids_ok=torch.equal(cx.bufs.ids[:b].cpu(), torch.cat([anchor, drafts])),
                q_err=float((q_rows[0] - h.q).abs().max()),
                drafts=drafts.tolist(),
                sglang=tokens[0].tolist(),
            )
        )
    ok = all(t["drafts_ok"] and t["path_ok"] and t["ids_ok"] for t in trials)
    return dict(ok=ok and all(t["q_err"] <= 1e-6 for t in trials), trials=trials)


def run(
    cx,
    its,
    it: iteration.Iteration,
    p0: int,
    graph: torch.cuda.CUDAGraph | None,
    sync=None,
) -> list[dict]:
    """ITERS sampled iterations from the start state (eager, or replays of ``graph``); before each, the window's
    identity slots (the runtime's window_slots); with ``sync``, draft_kv_sync after each (eager) and a flush at the
    end."""
    begin(cx, its, it, p0)
    it.sampling(4321, p0, 64, 1.0, 1.0)
    s, f = cx.main.cuda_stream, cx.feeds
    offs = torch.arange(16, dtype=torch.int64, device=cx.device)
    if sync is not None:
        lo = p0 - synthetic.COMMIT0
        it.sync_state.copy_(
            torch.cat([torch.tensor([lo]), torch.arange(lo, lo + 16)]).to(cx.device)
        )
    torch.cuda.synchronize(cx.device)
    recs = []
    with torch.cuda.stream(cx.main):
        for _ in range(ITERS):
            torch.add(it.words[WORDS["P"]], offs, out=f.out_cache_loc)
            if graph is None:
                for launch in it.launches("iter", sample=True):
                    launch(s, True)
                if sync is not None:
                    sync[0](s, True)
            else:
                graph.replay()
            synced(cx, "sampled iteration")
            recs.append(
                dict(
                    words=it.words.cpu(),
                    out_ring=it.accept_buf.out_ring.cpu(),
                    drafts=it.head_out.drafts.cpu(),
                    q=it.head_out.q.cpu(),
                )
            )
        if sync is not None:
            for launch in it.launches("draft"):
                launch(s, True)
            sync[1](s, True)
            synced(cx, "flush")
    return recs


def sync_check(cx, it: iteration.Iteration, pool, p0: int, final_p: int) -> dict:
    """The pool's rows of the committed positions [p0 - COMMIT0, final P) = the rings'; every other row still zero."""
    lo, hi = p0 - synthetic.COMMIT0, final_p
    pos = torch.arange(lo, hi, device=cx.device)
    bad, untouched = [], True
    for i, (lb, (k, v)) in enumerate(zip(it.lb, pool)):
        for name, ring, buf in (("k", lb.k_ring, k), ("v", lb.v_ring, v)):
            if not bits_equal(buf[pos], ring[pos % cx.ring]):
                bad.append(f"layer {i} {name}")
            rest = torch.ones(buf.shape[0], dtype=torch.bool, device=cx.device)
            rest[pos] = False
            untouched &= bool((buf[rest] == 0).all())
    return dict(
        ok=not bad and untouched and hi - lo >= synthetic.COMMIT0 + ITERS,
        positions=[lo, hi],
        bad=bad,
        untouched=untouched,
    )


def brief(v):
    """A check's summary: no per-trial ids or digests, a traceback's last line."""
    if isinstance(v, str):
        return v.splitlines()[-1] if "\n" in v else v
    if isinstance(v, dict):
        drop = ("rows", "digests", "drafts", "sglang")
        return {k: brief(x) for k, x in v.items() if k not in drop}
    return [brief(x) for x in v] if isinstance(v, list) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--draft",
        default="",
        help="the DFlash2 draft's directory; omitted: synthetic draft weights",
    )
    ap.add_argument("--ctx", type=int, default=4096, help="the start state's P0")
    ap.add_argument("--routing-scale", type=float, default=0.0)
    ap.add_argument("--label", default="sample")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert (
        not args.draft or (Path(args.draft) / DRAFT_FILE).is_file()
    ), f"--draft {args.draft}: no {DRAFT_FILE}"

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
    max_ctx = args.ctx + (2 * ITERS + 4) * cfg.rows
    cx = synthetic.context(draft, max_ctx, rank, device, group, cfg)
    its = {b: iteration.Iteration(b, cx, sample=True, block=True) for b in (16, 8)}
    for it in its.values():
        synthetic.seed_rings(it)
    token_wise = {
        b: SampleAccept(it, cx.accept_lib, cx.accept_sample_mod, block=False)
        for b, it in its.items()
    }
    synthetic.set_routing(cfg, cx.weights, args.routing_scale, device)
    synthetic.target_feeds(cx, args.ctx)
    cx.published()
    for _ in range(2):
        cx.stack.forward(Window(), cx.main)
    cx.published()
    for it in its.values():
        begin(cx, its, it, args.ctx)
        it.bind_constants()
    args.out.mkdir(parents=True, exist_ok=True)
    mine = dict(
        label=args.label,
        rank=rank,
        host=socket.gethostname(),
        draft=args.draft or "synthetic",
        checks={},
    )
    raw = args.out / f"sample_rank{rank}_{args.label}.json"

    def guarded(name, fn):
        try:
            mine["checks"][name] = fn()
        except Exception:
            mine["checks"][name] = dict(ok=False, error=traceback.format_exc())
            print(f"### {name} rank {rank} raised:\n{mine['checks'][name]['error']}")
        raw.write_text(json.dumps(mine, indent=1, default=str))

    for b, it in its.items():
        begin(cx, its, it, args.ctx)
        for kind, sa in (("block", it.sample_accept), ("token", token_wise[b])):
            guarded(
                f"accept_{kind}_b{b}",
                lambda: accept_check(cx, it, sa, kind == "block", args.ctx),
            )
        begin(cx, its, it, args.ctx)
        guarded(f"walk_b{b}", lambda: walk_check(cx, it, args.ctx))

    it = its[16]
    n = max_ctx + cfg.rows
    pool = [
        tuple(lb.k_ring.new_zeros(n, lb.k_ring.shape[1]) for _ in range(2))
        for lb in it.lb
    ]
    sync = it.pool_sync(pool)

    def graph_and_sync():
        eager = run(cx, its, it, args.ctx, None, sync)
        final_p = int(eager[-1]["words"][WORDS["P"]])
        res = dict(sync=sync_check(cx, it, pool, args.ctx, final_p))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cx.main):
            for launch in it.launches("iter", sample=True):
                launch(cx.main.cuda_stream, True)
        replay = run(cx, its, it, args.ctx, g)
        same = [all(bits_equal(e[k], r[k]) for k in e) for e, r in zip(eager, replay)]
        res["graph"] = dict(
            ok=all(same),
            per_iteration=same,
            commits=[int(e["words"][WORDS["COMMIT"]]) for e in eager],
        )
        digests = [sha(e["words"], e["out_ring"], e["drafts"]) for e in eager]
        every = [None] * world
        dist.all_gather_object(every, digests)
        res["ranks"] = dict(ok=all(d == digests for d in every), digests=digests)
        return res

    guarded("iteration", graph_and_sync)

    oks = {k: checks_pass(v) for k, v in mine["checks"].items()}
    every_ok = [None] * world
    dist.all_gather_object(every_ok, oks)
    all_ok = all(all(e.values()) for e in every_ok)
    if rank == 0:
        lines = [
            f"sample {args.label} draft {mine['draft']} world {world} ctx {args.ctx}"
        ]

        def report(name: str, c: dict) -> None:
            if "ok" in c:
                body = json.dumps(brief(c), default=str)
                lines.append(f"{name}: {'PASS' if c['ok'] else 'FAIL'} {body}")
            else:
                for k, v in c.items():
                    report(f"{name} {k}", v)

        for name, c in mine["checks"].items():
            report(name, c)
        lines.append(f"SAMPLE CHECKS {'PASS' if all_ok else 'FAIL'} (all ranks)")
        print("\n".join(lines))
        (args.out / f"sample_{args.label}.json").write_text(
            json.dumps(
                dict(label=args.label, per_rank=every_ok, rank0=mine),
                indent=1,
                default=str,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    if not all_ok:
        raise SystemExit("SAMPLE CHECKS FAIL")


def test_dflash_sample(tmp_path):
    """This file under torchrun on one node's 4 GPUs (TP4), the synthetic draft unless ``SGL_MUSE_MEGA_DRAFT_PATH``
    names the published draft's directory."""
    import pytest

    pytest.importorskip("sglang")
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
