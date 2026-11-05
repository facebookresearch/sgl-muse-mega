"""The DFlash2 draft's spec. It covers no step of the draft model: its build puts the rank's shard of the draft
checkpoint in the kernels' layouts beside the target's weights (before SGLang sizes its KV pools), for the fused
iteration that replaces the DFlash2 worker's batch-1 greedy and sampled steps (``runtime.iteration``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.runtime_context import get_parallel, get_spec

from .runtime import build as runtime_build
from .sm103.dflash.iteration import DraftDevice
from .sm103.dflash.weights import DRAFT_FILE, DraftConfig, load_shard

# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "SPECULATIVE_DECODING",
        "TOLERANCE",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("rank_in_group", "world_size"),
    "srt/arg_groups/fields/spec.py": ("speculative_draft_model_path",),
}


class Draft:
    widths = ()
    draft_sampler = None

    def summary(self) -> str:
        return "DFlash2 draft in the kernels' layouts"


def model_disable_reasons(model, hf_config) -> list[str]:
    try:
        cfg = DraftConfig.from_hf(hf_config.to_dict())
    except (ValueError, KeyError) as e:
        return [f"not a DFlash2 draft the kernels implement: {type(e).__name__}: {e}"]
    want = DraftConfig()
    reasons = [
        f"{f.name} {getattr(cfg, f.name)!r} is not {getattr(want, f.name)!r}"
        for f in dataclasses.fields(DraftConfig)
        if getattr(cfg, f.name) != getattr(want, f.name)
    ]
    if runtime_build.TARGET is None:
        reasons.append("the target's megakernel is not built in this process")
    return reasons


def build(model, hf_config, ctx) -> Draft:
    target, tp = runtime_build.TARGET, get_parallel().tp_group
    path = Path(get_spec().speculative_draft_model_path) / DRAFT_FILE
    weights = load_shard(str(path), tp.rank_in_group, tp.world_size)
    target.draft = DraftDevice(
        weights, target.weights.lm_head, target.cfg, tp.rank_in_group, target.device
    )
    torch.cuda.synchronize(target.device)
    torch.cuda.empty_cache()
    return Draft()


SPEC = MegakernelSpec(
    name="DFlash2 draft for Muse Spark Open",
    requires=Requirements(
        isas=frozenset({"sm_103"}),
        tp_sizes=frozenset({4, 8}),
        dtypes=frozenset({"bfloat16"}),
        decode_graph_backends=frozenset({"full"}),
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # each draft unit against the draft's math in a fixed order (the kernel tests' band)
    numerics=Numerics(kind=NumericsKind.TOLERANCE, tolerances={"draft_unit.rel": 2e-2}),
    max_rows=0,
    build=build,
    api_version=1,
    model_disable_reasons=model_disable_reasons,
)
