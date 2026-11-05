"""Muse Spark Open's megakernel spec: the batch-1 DFlash2 verify step (16 rows at a block of 16, 8 at a block of 8)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import torch.distributed as dist

from sglang.kernels.plugin_kernels.nvidia.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.kernel_plugins.nvidia.binding import deadline_s
from sglang.srt.runtime_context import get_model, get_parallel

from ..common.jit import toolchain_reasons
from .config import Config
from .runtime.build import INDEX, TREE_ENV, alias_reasons, build, tree_enabled
from .runtime.executor import rope_rows
from .sm103.dflash.weights import DRAFT_FILE

# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "SPECULATIVE_DECODING",
        "MULTI_NODE_TP",
        "EXPERT_PARALLEL",
        "TOLERANCE",
    ),
    "srt/kernel_plugins/nvidia/binding.py": ("deadline_s",),
    "srt/runtime_context.py": ("get_model", "get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("rank_in_group", "world_size", "cpu_group"),
    "srt/environ.py": ("SGLANG_USE_HND_KVCACHE",),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "speculative_draft_window_size",
        "speculative_draft_model_path",
        "speculative_draft_kv_cache_dtype",
        "enable_unified_memory",
        "disable_hybrid_swa_memory",
        "moe_a2a_backend",
        "startup_weight_load_mode",
        "context_length",
        "tp_size",
        "nnodes",
        "watchdog_timeout",
        "load_format",
        "model_path",
    ),
}

# the longest bounded wait of the kernels, seconds: kernels/dflash/common.cuh kSpinBudgetNs (the DFlash2 entry, head and
# verify_accept), kernels/head.cu kGatherSpinBudgetNs and kernels/framework.cuh kRowPollBudgetNs
LONGEST_WAIT_S = 60.0


def draft_reasons(path) -> list[str]:
    """The fused iteration's draft: a local DFlash2 directory with its one safetensors file (the draft spec's build
    reads it)."""
    d = Path(path or "")
    try:
        arch = json.loads((d / "config.json").read_text())["architectures"]
    except (OSError, ValueError, KeyError) as e:
        return [
            f"--speculative-draft-model-path {path!r} has no local config.json naming its architecture "
            f"({type(e).__name__})"
        ]
    reasons = []
    if arch != ["DFlash2DraftModel"]:
        reasons.append(
            f"the draft's architectures {arch} are not ['DFlash2DraftModel']"
        )
    if not (d / DRAFT_FILE).is_file():
        reasons.append(
            f"the draft build reads {DRAFT_FILE} under --speculative-draft-model-path (one local file)"
        )
    return reasons


def config_disable_reasons(cfg, hf_config) -> list[str]:
    model = Config()
    reasons = model.hf_reasons(hf_config)
    if cfg.context_length is None or cfg.context_length > model.context_length:
        reasons.append(
            f"--context-length {cfg.context_length} is unset or above {model.context_length}"
        )
    if cfg.tp_size != model.node_ranks * cfg.nnodes:
        reasons.append(
            f"tp_size {cfg.tp_size} over {cfg.nnodes} node(s): the kernels gather within groups of "
            f"{model.node_ranks} ranks on one node"
        )
    if deadline_s(cfg.watchdog_timeout) < LONGEST_WAIT_S:
        reasons.append(
            f"--watchdog-timeout {cfg.watchdog_timeout} leaves a wait deadline under the kernels' {LONGEST_WAIT_S:g} s"
        )
    if tree_enabled() and cfg.speculative_num_draft_tokens != model.rows:
        reasons.append(
            f"{TREE_ENV}=1: the token tree verifies a block of {model.rows} (--speculative-num-draft-tokens)"
        )
    if cfg.load_format == "dummy" or not (Path(cfg.model_path) / INDEX).is_file():
        reasons.append(
            f"the build reads the experts from {INDEX} under --model-path, a local checkpoint"
        )
    return (
        reasons + draft_reasons(cfg.speculative_draft_model_path) + toolchain_reasons()
    )


def model_disable_reasons(model, hf_config) -> list[str]:
    tp = get_parallel().tp_group
    cfg = Config(tp=tp.world_size)
    reasons = alias_reasons(model, cfg, tp.rank_in_group)
    # verify_accept stages the rope rows of the next window: up to position P + 31 at a block of 16,
    # P < --context-length
    if (
        not reasons
        and rope_rows(model).shape[0] < get_model().context_length + 2 * cfg.rows
    ):
        reasons.append(f"the rope rows do not reach --context-length + {2 * cfg.rows}")
    hosts = [None] * tp.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=tp.cpu_group)
    if any(hosts[r] != hosts[r - r % cfg.node_ranks] for r in range(tp.world_size)):
        reasons.append(
            f"the TP ranks are not node-major in groups of {cfg.node_ranks}: {hosts}"
        )
    return reasons


SPEC = MegakernelSpec(
    name="Muse Spark Open",
    requires=Requirements(
        isas=frozenset({"sm_103"}),
        tp_sizes=frozenset({4, 8}),
        # the layer launch: 36 clusters of 4 CTAs, one CTA per SM
        resident_blocks=144,
        dtypes=frozenset({"bfloat16"}),
        decode_graph_backends=frozenset({"full"}),
        server_args={
            # the kernels attend over and write a bf16 pool
            "kv_cache_dtype": frozenset({"bf16", "bfloat16"}),
            "speculative_algorithm": frozenset({"DFLASH"}),
            "speculative_num_draft_tokens": frozenset({8, 16}),
            # the draft ring copies SGLang's draft K / V rows as they are
            "speculative_draft_kv_cache_dtype": frozenset(
                {None, "auto", "bf16", "bfloat16"}
            ),
            # the draft reads the target's slot tables
            "speculative_draft_window_size": frozenset({None}),
            # slot ids are physical rows of the pools
            "enable_unified_memory": frozenset({False}),
            # the sliding-window layers' own pool and its full -> SWA slot map
            "disable_hybrid_swa_memory": frozenset({False}),
            "moe_a2a_backend": frozenset({"none"}),
            # the build aliases the loaded weights in place
            "startup_weight_load_mode": frozenset({"serial"}),
        },
        hf_config={"model_type": "muse_spark"},
        # the kernels read the pools' token-major rows
        env={"SGLANG_USE_HND_KVCACHE": False},
    ),
    features=frozenset(
        {Feature.SPECULATIVE_DECODING, Feature.MULTI_NODE_TP, Feature.EXPERT_PARALLEL}
    ),
    # relative errors against SGLang's path, all in the same precision class: split-KV and all-reduce reduction
    # orders, the TF32 router, the experts' two NVFP4 quantizers (a code flips where its input moves by one bf16
    # step). Each is the largest measured on the public weights against a torch reference in SGLang's math, rounded
    # up; logits.rel was measured (0.136) when the logits were rounded to bf16, which adds at most 2^-9 relative, so it
    # stands for the fp32 logits (not re-measured)
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        tolerances={
            "attention.rel": 6e-3,
            "post_attention.rel": 5e-3,
            "router_logits.rel": 5e-3,
            "moe_output.rel": 0.15,
            "layer_output.rel": 5e-2,
            "logits.rel": 0.14,
        },
    ),
    max_rows=16,
    build=build,
    api_version=1,
    model_disable_reasons=model_disable_reasons,
    config_disable_reasons=config_disable_reasons,
)
