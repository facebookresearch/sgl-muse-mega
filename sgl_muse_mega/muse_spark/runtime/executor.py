"""The built megakernel: plan() covers the batch-1 DFlash2 verify step under decode-graph capture, forward() runs it
on the stack in place of the model's forward.

The stack binds SGLang's KV pools, slot tables and rope rows at its first covered step (the pools exist only after
the build). Each step copies the window's inputs from SGLang's forward batch into the feeds' static buffers, runs
the verify graph's launches (``sm103.verify``), and returns what Muse Spark's forward returns: the full-vocabulary
fp32 logits (softcapped) and the DFlash2 capture layers' output streams packed in capture order (bf16).
"""

from __future__ import annotations

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import BuiltMegakernel, Step
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import get_req_to_token_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_model

from ..config import Config
from ..sm103.buffers import Buffers, Feeds
from ..sm103.dflash.iteration import DraftDevice, capture_layers
from ..sm103.dflash.weights import rope_cos_sin
from ..sm103.verify import M8Stack, Stack, Window
from ..sm103.weights import ModelWeights

# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("BuiltMegakernel", "Step", "view"),
    "srt/kernel_plugins/nvidia/binding.py": ("kv_pool", "pool"),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    "srt/model_executor/forward_context.py": ("get_req_to_token_pool",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "batch_size",
        "seq_lens",
        "out_cache_loc",
        "req_pool_indices",
        "is_target_verify",
    ),
    "srt/mem_cache/memory_pool.py": ("req_to_token",),
    "srt/mem_cache/swa_memory_pool.py": (
        "get_key_buffer",
        "get_value_buffer",
        "full_to_swa_index_mapping",
    ),
    "srt/models/muse_spark.py": (
        "model",
        "language_model",
        "layers_to_capture",
        "layers",
        "self_attn",
        "rotary_emb",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
    # Muse Spark's prefill calls its forward again with input_ids None (plan declines it)
    "srt/managers/mm_utils.py": ("general_mm_embed_routine",),
    "srt/runtime_context.py": ("get_model",),
    "srt/arg_groups/fields/*.py": ("context_length",),
}
# SGLang's megakernel contract has the DFlash iteration slot (and the pool view's layer buffers); an SGLang without
# it gets the fused iteration through hooks on its DFlash2 worker (runtime.iteration.install)
SLOT = "dflash_iteration" in BuiltMegakernel.__annotations__


def rope_rows(model) -> torch.Tensor:
    """SGLang's fp32 rope rows {cos | sin} per position (a sliding layer's: the global layers have none)."""
    return model.model.language_model.layers[0].self_attn.rotary_emb.cos_sin_cache


def rope_table(rows: torch.Tensor, need: int, cfg: Config) -> torch.Tensor:
    """SGLang's rope rows, continued to ``need`` rows as SGLang computes them where they end sooner: verify_accept
    stages rows up to P + 31 (< --context-length + 32), and rows past the context reach no committed token.
    """
    if rows.shape[0] >= need:
        return rows
    cos, sin = rope_cos_sin(
        torch.arange(rows.shape[0], need), cfg.head_dim, cfg.rope_theta
    )
    return torch.cat([rows, torch.cat([cos, sin], 1).to(rows.device)])


def window_slots(
    f: Feeds, offsets: torch.Tensor, out_cache_loc: torch.Tensor | None = None
) -> None:
    """The window's positions (the committed length + 0..15, the 16-row layout) and their slots in both pools: the
    block's rows at the slots SGLang allocated (``out_cache_loc``), or every row from the request's slot-table row.
    """
    torch.add(f.seq_lens, offsets, out=f.positions)
    if out_cache_loc is None:
        flat = f.req_pool_indices * f.width + f.positions
        f.out_cache_loc.copy_(f.req_to_token.view(-1)[flat])
    else:
        f.out_cache_loc[: out_cache_loc.shape[0]].copy_(out_cache_loc)
    torch.index_select(f.full_to_swa, 0, f.out_cache_loc, out=f.swa_out_cache_loc)


def kv_buffers(pool, layers: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every layer's (K, V) buffer of SGLang's pool."""
    return [(pool.get_key_buffer(L), pool.get_value_buffer(L)) for L in range(layers)]


def _addresses(kv) -> list[tuple]:
    return [(k.data_ptr(), tuple(k.shape), v.data_ptr(), tuple(v.shape)) for k, v in kv]


class Verify:
    draft_sampler = None

    def __init__(
        self,
        cfg: Config,
        block: int,
        paths: dict,
        libs: dict,
        bufs: Buffers,
        weights: ModelWeights,
        model,
        rank: int,
        device,
        group,
        summary: str,
        tree: bool = False,
    ):
        self.cfg, self.block, self.paths, self.libs = cfg, block, paths, libs
        # the fused iteration verifies a token tree (runtime.build.tree_enabled)
        self.tree = tree
        self.bufs, self.weights, self.model, self.rank, self.device = (
            bufs,
            weights,
            model,
            rank,
            device,
        )
        self.group, self.widths, self._summary = group, (block,), summary
        self.stack: Stack | None = None
        # the draft spec's device tensors and the fused iteration (runtime.iteration), when DFlash2 builds them
        self.draft: DraftDevice | None = None
        self.fused = None
        # the 16-row layout's positions are the committed length + 0..15 for either block
        self.offsets = torch.arange(cfg.rows, dtype=torch.int64, device=device)
        self.window: Window | None = None
        self.caps: list[torch.Tensor] = []
        self.aux: torch.Tensor | None = None
        # the pool the stack bound and its KV buffers' (address, shape) per layer: the graphs keep the addresses
        self.kv_pool = None
        self.kv_bound: list[tuple] = []

    def summary(self) -> str:
        return self._summary

    def plan(self, input_ids, forward_batch, *args, **kwargs) -> Step | None:
        # Muse Spark's prefill re-enters its forward with input_ids None (the multimodal embedding routine)
        if input_ids is None or not get_is_capture_mode():
            return None
        if (
            not forward_batch.forward_mode.is_target_verify()
            or int(forward_batch.batch_size) != 1
        ):
            return None
        if int(input_ids.shape[0]) != self.block:
            return None
        return Step(mode="target_verify", rows=self.block)

    def _bind(self, step: Step) -> None:
        """The stack on SGLang's KV pools (each layer's K / V rows in its attention type's pool), slot tables and rope rows."""
        cfg = self.cfg
        pool = self.kv_pool = step.view.kv_pool if SLOT else step.view.kv_pool.pool
        kv = []
        for layer, (k, v) in enumerate(kv_buffers(pool, cfg.layers)):
            assert (
                k.dtype == v.dtype == torch.bfloat16
                and k.is_contiguous()
                and v.is_contiguous()
            ), layer
            kv.append((k.view(k.shape[0], -1), v.view(v.shape[0], -1)))
            assert kv[-1][0].shape[1] * 2 == cfg.kv_row_bytes, (layer, tuple(k.shape))
        self.kv_bound = _addresses(kv_buffers(pool, cfg.layers))
        feeds = Feeds(
            cfg,
            get_req_to_token_pool().req_to_token,
            pool.full_to_swa_index_mapping,
            rope_table(
                rope_rows(self.model), get_model().context_length + 2 * cfg.rows, cfg
            ),
            self.device,
            tree=self.tree,
        )
        cls = M8Stack if self.block < cfg.rows else Stack
        self.stack = cls(
            cfg,
            self.paths,
            self.libs,
            self.bufs,
            self.weights,
            self.rank,
            self.device,
            kv,
            feeds,
        )
        lm = self.model.model.language_model
        self.window = Window(
            b_live=self.block,
            capture=capture_layers(cfg, lm.layers_to_capture),
            native=self.block < cfg.rows,
        )
        assert self.window.capture, "DFlash2 has not set the target's capture layers"
        self.stack.layers(self.window)
        self.caps = [self.stack.cap(L) for L in self.window.capture]
        self.aux = torch.empty(
            self.block,
            len(self.caps) * cfg.hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def check_kv(self, pool) -> None:
        """Raises unless SGLang's KV buffers are still the ones the graphs were bound to."""
        now = _addresses(kv_buffers(pool, self.cfg.layers))
        if now != self.kv_bound:
            bad = [L for L, (x, y) in enumerate(zip(now, self.kv_bound)) if x != y]
            raise RuntimeError(
                f"SGLang's KV buffers moved after the megakernel bound them: layers {bad}"
            )

    def dflash_iteration(self, *, block_size: int, draft_model, draft_kv_pool):
        """The fused DFlash2 iteration (``runtime.iteration``), bound once SGLang's graphs are captured: called by
        SGLang's DFlash iteration slot, or by the plugin's hook on an older SGLang."""
        from .iteration import Fused  # iteration imports this module

        assert block_size == self.block, (block_size, self.block)
        assert (
            self.stack is not None
        ), "the target's batch-1 verify graph was not captured"
        # the target spec refuses a draft other than DFlash2 (spec.config_disable_reasons)
        assert self.draft is not None, "the DFlash2 draft's spec did not build"
        self.check_kv(self.kv_pool)
        return Fused(self, draft_model, draft_kv_pool)

    def forward(
        self,
        step: Step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
    ):
        if self.stack is None:
            self._bind(step)
        b, f = self.block, self.stack.f
        f.seq_lens.copy_(forward_batch.seq_lens[:1])
        f.req_pool_indices.copy_(forward_batch.req_pool_indices[:1])
        window_slots(f, self.offsets, forward_batch.out_cache_loc)
        self.bufs.ids[:b].copy_(input_ids)
        self.stack.forward(self.window, torch.cuda.current_stream())
        torch.cat([c[:b] for c in self.caps], 1, out=self.aux)
        return LogitsProcessorOutput(
            next_token_logits=self.bufs.logits.out[:b], hidden_states=self.aux
        )
