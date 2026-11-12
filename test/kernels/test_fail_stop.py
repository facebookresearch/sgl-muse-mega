"""Fail-stop: the bounded cross-rank waits against a planted missing rank, on 4 ranks (one node, TP4) or 8 (TP8).

    torchrun --nproc-per-node 4 test/kernels/test_fail_stop.py --missing 3 --wait 75

Every rank binds the stack. Then the missing rank launches nothing and sleeps past --wait while the others run one
eager step: each live rank's first all-reduce consume faults after its 30 s budget with the named report (this rank,
the packet, the mask of ranks whose packets landed). Each rank prints one CONTROL line, then leaves through os._exit
(a faulted context cannot tear down; a hung one is torn down by the driver at exit). A faulted rank leaves its CUDA
context unusable, so the pytest entry runs the case in its own torchrun.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1])]  # the package

import synthetic  # noqa: E402
from sgl_muse_mega.muse_spark.config import Config  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.artifacts import LAYER_IMAGES, build  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.buffers import Buffers  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.verify import (  # noqa: E402
    Stack,
    Window,
    layer_libraries,
)

# a live rank's fault arrives within the all-reduce's 30 s budget plus the step's start
FAULT_S = (30.0, 60.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, default=4096, help="held context")
    ap.add_argument(
        "--missing",
        type=int,
        default=-1,
        help="the rank that launches nothing (default: the last)",
    )
    ap.add_argument(
        "--wait",
        type=float,
        default=75.0,
        help="seconds a live rank waits for its step",
    )
    args = ap.parse_args()

    dist.init_process_group("gloo")
    group, rank, world = dist.group.WORLD, dist.get_rank(), dist.get_world_size()
    cfg = Config(tp=world)
    missing = args.missing % world
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    # the primary context, current for the driver-API launches
    torch.zeros(1, device=device)

    paths = build(world=cfg.world)
    libs = layer_libraries(paths)
    consts = {image: libs[image].constants for image in LAYER_IMAGES}
    bufs = Buffers(cfg, consts, device, group)
    synthetic.seed_ids(bufs, cfg)
    weights = synthetic.model_weights(cfg, rank, device, bufs.embed_shard(cfg))
    kv = synthetic.kv_fill(cfg, rank, (args.ctx + cfg.rows + 127) // 128 * 128, device)
    feeds = synthetic.identity_feeds(cfg, args.ctx, device)
    synthetic.set_context(feeds, args.ctx, cfg.rows)
    stack = Stack(cfg, paths, libs, bufs, weights, rank, device, kv, feeds)
    window = Window()
    stack.layers(window)  # bind on every rank before the step
    main_s = torch.cuda.Stream(device)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == missing:
        # alive (its symmetric memory mapped) until the live ranks have reported
        time.sleep(args.wait + 15.0)
        print(f"CONTROL rank {rank}: missing (launched nothing)", flush=True)
        os._exit(0)
    t0 = time.monotonic()
    stack.forward(window, main_s)
    done = torch.cuda.Event()
    done.record(main_s)
    try:
        while not done.query():
            if time.monotonic() - t0 > args.wait:
                print(
                    f"CONTROL rank {rank}: step still running after {args.wait:.0f} s",
                    flush=True,
                )
                os._exit(0)
            time.sleep(0.05)
        print(
            f"CONTROL rank {rank}: step returned after {time.monotonic() - t0:.1f} s",
            flush=True,
        )
    except RuntimeError as e:
        dt = time.monotonic() - t0
        # the device printf buffer flushes at a sync, which raises the same fault
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        print(
            f"CONTROL rank {rank}: step faulted after {dt:.1f} s: {str(e).splitlines()[0]}",
            flush=True,
        )
    os._exit(0)


def test_fail_stop():
    """One node (TP4), the last rank missing: every live rank faults inside FAULT_S."""
    import pytest

    if torch.cuda.device_count() < 4:
        pytest.skip("needs one node of 4 GB300 GPUs")
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4"]
        + [__file__, "--missing", "3"],
        capture_output=True,
        text=True,
    )
    faults = [
        float(t)
        for t in re.findall(r"CONTROL rank \d: step faulted after ([\d.]+) s", r.stdout)
    ]
    assert len(faults) == 3 and all(FAULT_S[0] <= t <= FAULT_S[1] for t in faults), (
        r.stdout + r.stderr
    )


if __name__ == "__main__":
    main()
