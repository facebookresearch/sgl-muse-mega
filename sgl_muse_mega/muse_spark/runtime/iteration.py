"""The fused DFlash2 iteration inside SGLang's DFlash worker: for a batch-1 decode step, one CUDA graph runs the draft,
the target verify and the accept (``sm103.dflash.iteration``) in place of the worker's step, and returns what the
worker returns. A greedy request takes the greedy graph; a sampled one (temperature, top-k, top-p) the sampled graph,
which samples the draft's walk and accepts as SGLang's DFlash2 step does at temperature > 0. Every other step
(prefill, requests with logprobs, grammar, penalties, logit bias, min-p, a custom logit processor or a sampling mask,
larger batches) runs SGLang's own.

SGLang's DFlash2 worker runs it through its megakernel DFlash iteration slot where SGLang has one, else through two
hooks on ``DFlashWorkerV2`` (SGLang's HookRegistry; ``install``, called by the target spec's build, so only a server
with ``--enable-nvidia-megakernel`` has them):
- once ``init_cuda_graphs`` has captured SGLang's graphs (the target's verify stack bound to the pools): the iteration
  is bound on the target stack and the draft's device tensors, and captured twice (its outputs alternate between two
  sets, so a step's results stay intact while the overlap scheduler launches the next step);
- at ``forward_batch_generation``: an eligible step replays the graph.

A graph reads the window from SGLang's state: the committed length and the request's slot (copied into the feeds'
static inputs before each replay) and the slot table SGLang filled for the window, so its target KV goes to the slots
SGLang allocated. The draft keeps its context K / V in its own ring; at a request's first fused step (or after a step
SGLang ran) the ring is filled from SGLang's draft KV pool, which holds the context of every committed position up to
then, and the iteration starts from SGLang's committed length and bonus token. The fused steps keep that pool in
step: a committed position's draft K / V reaches the ring one iteration after its commit, and each graph's last launch
copies the rows that arrived into the pool at the positions' slots; the last commit of a run (its final fused step) is
flushed before the next step of any request runs (the draft alone, then the copy). So a request that resumes on a
prefix-cache hit, or falls back to SGLang's step, drafts from the context the fused steps made.

With the token tree on (``SGLANG_MUSE_MEGA_TREE=1``, block 16) the graph verifies a tree of drafts and moves the
accepted path's K / V into the slots of the committed positions (and its captured rows into the draft's context) before
the export, so SGLang receives the commit and the tokens exactly as from the linear chain.

With SGLang's ``SGLANG_SIMULATE_ACC_LEN`` set (and its ``_METHOD`` / ``_TOKEN_MODE``), verify_accept forces each step's
commit length from a table of the lengths SGLang's simulated step draws (``simulated_accepts``), and a ``fixed`` token
mode exports SGLang's fixed token id; sampled steps then run SGLang's step, which simulates them itself.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math

import torch

from sglang.srt.kernel_plugins.nvidia.binding import built_megakernel
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
)

from ..sm103.dflash.head import WORDS
from ..sm103.dflash.iteration import Context, Iteration
from .executor import SLOT, Verify, window_slots

# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py": ("built_megakernel",),
    "srt/plugins/hook_registry.py": (
        "HookRegistry",
        "HookType",
        "register",
        "apply_hooks",
        "AFTER",
        "AROUND",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "next_token_logits"),
    "srt/managers/utils.py": (
        "GenerationBatchResult",
        "logits_output",
        "next_token_ids",
        "accept_lens",
        "can_run_cuda_graph",
        "next_draft_input",
        "speculative_num_draft_tokens",
        "new_seq_lens",
    ),
    "srt/speculative/dflash_info_v2.py": ("DFlashDraftInputV2", "bonus_tokens"),
    "srt/speculative/dflash_utils.py": (
        "apply_dflash_simulated_acceptance",
        "apply_dflash_simulated_acceptance(fixed_token_id)",
    ),
    "srt/speculative/spec_utils.py": (
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_METHOD",
        "SIMULATE_ACC_TOKEN_MODE",
        '"match-expected"',
        '"multinomial"',
        '"fixed"',
    ),
    "srt/speculative/draft_worker_common.py": (
        "make_draft_input_v2",
        "make_draft_input_v2(bonus_tokens)",
        "make_draft_input_v2(new_seq_lens)",
    ),
    "srt/speculative/*.py": (
        "target_worker",
        "draft_model_runner",
        "draft_model",
        "_can_skip_verify_logits",
        "forward_batch_generation(on_publish)",
    ),
    "srt/model_executor/model_runner.py": ("token_to_kv_pool", "model"),
    "srt/managers/schedule_batch.py": (
        "reqs",
        "rid",
        "seq_lens",
        "req_pool_indices",
        "spec_info",
        "sampling_info",
        "is_extend_in_batch",
        "forward_mode",
        "has_grammar",
        "return_logprob",
        "sampling_params",
    ),
    "srt/sampling/sampling_batch_info.py": (
        "need_min_p_sampling",
        "has_custom_logit_processor",
        "penalizer_orchestrator",
        "grammar_mask",
        "logit_bias",
        "sampling_mask_batch_indices",
    ),
    "srt/sampling/sampling_params.py": (
        "temperature",
        "top_p",
        "top_k",
        "sampling_seed",
    ),
    "srt/sampling/penaltylib/orchestrator.py": ("is_required",),
    "srt/runtime_context.py": ("get_spec",),
    "srt/arg_groups/fields/spec.py": ("speculative_use_block_verification",),
    "srt/model_executor/forward_batch_info.py": ("is_extend", "is_idle"),
    "srt/models/dflash.py": ("layers", "self_attn", "attn"),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_value_buffer"),
}

WORKER = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2"
logger = logging.getLogger(__name__)
# verify_accept reads its forced-accept table at the iteration word mod the length
SIMULATED_STEPS = 4096
GOLDEN = (math.sqrt(5) - 1) / 2


def simulated_accepts(b: int) -> torch.Tensor | None:
    """SGLANG_SIMULATE_ACC_LEN's forced accepts (commit length - 1) for verify_accept, or None when it is off: the
    distribution of SGLang's sample_simulated_acc_len over [1, b] at golden-ratio quantiles instead of random draws, so
    any run of consecutive steps meets the expected total within a few commits; the same table on every rank.
    """
    if SIMULATE_ACC_LEN <= 0:
        return None
    u = torch.remainder(
        0.5 + GOLDEN * torch.arange(SIMULATED_STEPS, dtype=torch.float64), 1.0
    )
    if SIMULATE_ACC_METHOD == "match-expected":
        x = min(max(SIMULATE_ACC_LEN, 1.0), b)
        commit = math.floor(x) + (u < x - math.floor(x)).long()
    elif SIMULATE_ACC_METHOD == "multinomial":
        commit = torch.round(SIMULATE_ACC_LEN + torch.special.ndtri(u)).clamp(1, b)
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")
    return (commit - 1).to(torch.int32)


def fixed_token() -> int | None:
    """The token id SGLang's simulated step commits in the "fixed" token mode (SGLang's DFlash2 worker checked the mode
    at its start), or None (real tokens)."""
    if SIMULATE_ACC_LEN <= 0 or SIMULATE_ACC_TOKEN_MODE != "fixed":
        return None
    sig = inspect.signature(apply_dflash_simulated_acceptance)
    return sig.parameters["fixed_token_id"].default


def _seed(req) -> int:
    """The request's 64-bit seed (signed): its sampling_seed, else one derived from its id (the same on every rank)."""
    seed = req.sampling_params.sampling_seed
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(req.rid.encode()).digest()[:8], "little")
    return (seed + 2**63) % 2**64 - 2**63


def _plain(batch) -> bool:
    """A sampled batch the sampled graph serves exactly: no logprobs, grammar, penalties, logit bias, min-p, custom
    logit processor or sampling mask (SGLang's step applies those to its logits)."""
    info = batch.sampling_info
    penalizer = info.penalizer_orchestrator
    return not (
        batch.has_grammar
        or batch.return_logprob
        or info.need_min_p_sampling
        or info.has_custom_logit_processor
        or (penalizer is not None and penalizer.is_required)
        or info.grammar_mask is not None
        or info.logit_bias is not None
        or info.sampling_mask_batch_indices is not None
    )


class Fused:
    """One process's fused iteration of the target's block size: the greedy graph, the sampled graph (not with the
    token tree or a simulated accept length) and the flush graph (the draft alone, then the pool copy of the run's last
    commit).
    """

    def __init__(self, target: Verify, draft_model, draft_kv_pool):
        cx = Context(target.stack, target.draft, target.group, tree=target.tree)
        self.cx, self.b, self.f = cx, target.block, target.stack.f
        self.sample = not target.tree and SIMULATE_ACC_LEN <= 0
        self.fixed = fixed_token()
        self.it = it = Iteration(
            self.b,
            cx,
            override=simulated_accepts(self.b),
            tree=target.tree,
            sample=self.sample,
            block=bool(get_spec().speculative_use_block_verification),
        )
        it.bind_constants()
        dev = cx.device
        pool = draft_kv_pool
        layers = [layer.self_attn.attn.layer_id for layer in draft_model.layers]
        assert len(layers) == len(it.lb)
        # the ring fill copies SGLang's draft K / V rows, indexed by the target's slot ids, as they are
        full_slots = target.stack.kv[cx.cfg.layers - 1][0].shape[0]
        row = it.lb[0].k_ring.shape[1]
        self.draft_kv = []
        for layer_id in layers:
            kv = (pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id))
            for t in kv:
                assert (
                    t.dtype == torch.bfloat16
                    and t.is_contiguous()
                    and t[0].numel() == row
                    and t.shape[0] >= full_slots
                ), (layer_id, t.dtype, tuple(t.shape), row, full_slots)
            self.draft_kv.append(tuple(t.view(t.shape[0], -1) for t in kv))
        sync, flush_sync = it.pool_sync(self.draft_kv)
        self.r2t = self.f.req_to_token
        self.offsets = target.offsets
        self.cols = torch.arange(self.b, dtype=torch.int64, device=dev)
        self.outs = [self._outputs() for _ in range(2)]
        modes = ("greedy", "sample") if self.sample else ("greedy",)
        self.graphs = {
            m: [self._capture(out, m, sync) for out in self.outs] for m in modes
        }
        self.flush_graph = self._capture_flush(flush_sync)
        self.parity = 0
        self.rid = None
        # the run's last commit is not in SGLang's draft pool yet
        self.pending = False
        logger.info(
            "fused DFlash2 iteration: block %d%s, %d launches in one graph%s, through %s%s",
            self.b,
            " (token tree)" if target.tree else "",
            len(it.launches("iter")) + 2,
            (
                " (sampled: %d)" % (len(it.launches("iter", sample=True)) + 2)
                if self.sample
                else ""
            ),
            "SGLang's iteration slot" if SLOT else "hooks",
            (
                f"; simulated accept length {SIMULATE_ACC_LEN:g} ({SIMULATE_ACC_METHOD}, "
                f"{SIMULATE_ACC_TOKEN_MODE} tokens; sampled steps run SGLang's)"
                if SIMULATE_ACC_LEN > 0
                else ""
            ),
        )

    def _outputs(self) -> dict:
        dev = self.cx.device
        return dict(
            tokens=torch.zeros(1, self.b, dtype=torch.int64, device=dev),
            commit=torch.zeros(1, dtype=torch.int32, device=dev),
            bonus=torch.zeros(1, dtype=torch.int64, device=dev),
            new_seq_lens=torch.zeros(1, dtype=torch.int64, device=dev),
        )

    def _export(self, out: dict) -> None:
        """The step's results in the worker's form: the committed tokens (the accepted drafts, then the bonus; zeros
        past the commit), the commit length, the bonus and the new committed length. In the fixed token mode, as
        SGLang's simulated step: every token and the bonus are the fixed id, which the next draft also starts from.
        """
        w, ring = self.it.words, self.it.accept_buf.out_ring
        commit = w[WORDS["COMMIT"]]
        if self.fixed is None:
            idx = torch.remainder(
                w[WORDS["OUT_COUNT"]] - commit + self.cols, ring.shape[0]
            )
            out["tokens"][0].copy_(torch.where(self.cols < commit, ring[idx].long(), 0))
        else:
            w[WORDS["ANCHOR"]].fill_(self.fixed)
            out["tokens"].fill_(self.fixed)
        out["commit"].copy_(commit)
        out["bonus"].copy_(w[WORDS["ANCHOR"]])
        out["new_seq_lens"].copy_(w[WORDS["P"]])

    def _capture(self, out: dict, mode: str, sync) -> torch.cuda.CUDAGraph:
        cx, s = self.cx, self.cx.main.cuda_stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cx.main):
            window_slots(self.f, self.offsets)
            self.cx.stack.feeds(s, True)
            for launch in self.it.launches("iter", sample=mode == "sample"):
                launch(s, True)
            sync(s, True)
            self._export(out)
        return g

    def _capture_flush(self, flush_sync) -> torch.cuda.CUDAGraph:
        """The draft alone (its context fill writes the last commit's rows into the ring), then their pool copy."""
        cx, s = self.cx, self.cx.main.cuda_stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cx.main):
            for launch in self.it.launches("draft"):
                launch(s, True)
            flush_sync(s, True)
        return g

    def mode(self, batch, needs_logits: bool) -> str | None:
        """The graph of a step: "greedy" (a batch-1 greedy decode step that needs no logits), "sample" (a batch-1
        sampled decode step with temperature, top-k and top-p only), or None (SGLang's step).
        """
        mode = batch.forward_mode
        if mode.is_extend() or batch.is_extend_in_batch or mode.is_idle():
            return None
        if len(batch.reqs) != 1 or not isinstance(batch.spec_info, DFlashDraftInputV2):
            return None
        # per request, as SGLang's draft does: the batch's is_all_greedy is not recomputed when a sampled request leaves
        if batch.sampling_info is None or batch.reqs[0].sampling_params.top_k <= 1:
            return None if needs_logits else "greedy"
        return "sample" if self.sample and _plain(batch) else None

    def __call__(
        self, batch, *, needs_logits: bool, on_publish
    ) -> GenerationBatchResult | None:
        """A step of SGLang's DFlash2 worker: the fused iteration for an eligible batch (``mode``), else None (SGLang's
        step; the request's next fused step restarts it). Before any step but the run's next fused one, the run's last
        commit goes into SGLang's draft pool."""
        mode = self.mode(batch, needs_logits)
        if self.pending and (mode is None or batch.reqs[0].rid != self.rid):
            self.flush_graph.replay()
            self.pending = False
        if mode is None:
            self.rid = None
            return None
        return self.step(batch, mode, on_publish)

    def _start(self, batch, mode: str) -> None:
        """The request's state from SGLang's: the committed length P0 and the bonus token as the anchor, commit 1, the
        draft ring filled from SGLang's draft pool with the context K / V of the positions the draft window reaches,
        the context row masked (its K / V is the pool's), the pool held up to P0, and (sampled) its sampling words.
        """
        it, cx, b = self.it, self.cx, self.b
        p0 = int(batch.seq_lens[0])
        # the forced-accept table continues across runs instead of restarting at its first entry
        steps = None if it.override is None else it.words[WORDS["ITER"]].clone()
        it.start(p0, 1)
        if steps is not None:
            it.words[WORDS["ITER"]].copy_(steps)
        it.words[WORDS["ANCHOR"]].copy_(batch.spec_info.bonus_tokens[0])
        it.dslot[b].fill_(-1)
        it.sync_state[0].fill_(p0)
        lo = max(0, p0 - (cx.ring - 2 * b))
        pos = torch.arange(lo, p0, device=cx.device)
        slots = self.r2t[batch.req_pool_indices[0], lo:p0].long()
        rows = pos % cx.ring
        for lb, (k, v) in zip(it.lb, self.draft_kv):
            lb.k_ring[rows] = k[slots]
            lb.v_ring[rows] = v[slots]
        if mode == "sample":
            req = batch.reqs[0]
            sp = req.sampling_params
            it.sampling(_seed(req), p0, sp.top_k, sp.temperature, sp.top_p)

    def step(self, batch, mode: str, on_publish) -> GenerationBatchResult:
        batch.seq_lens.record_stream(torch.cuda.current_stream())
        rid = batch.reqs[0].rid
        if rid != self.rid:
            self._start(batch, mode)
            self.rid = rid
            if self.cx.rank == 0:
                logger.info(
                    "fused DFlash2 iteration: request %s starts (%s)", rid, mode
                )
        self.f.seq_lens.copy_(batch.seq_lens)
        self.f.req_pool_indices.copy_(batch.req_pool_indices)
        self.graphs[mode][self.parity].replay()
        self.pending = True
        out = self.outs[self.parity]
        self.parity ^= 1
        if on_publish is not None:
            on_publish(out["new_seq_lens"])
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=out["tokens"].view(-1),
            accept_lens=out["commit"],
            can_run_cuda_graph=True,
            next_draft_input=make_draft_input_v2(
                bonus_tokens=out["bonus"], new_seq_lens=out["new_seq_lens"]
            ),
            speculative_num_draft_tokens=self.b,
            new_seq_lens=out["new_seq_lens"],
        )


def _target(worker) -> Verify | None:
    built = built_megakernel(worker.target_worker.model_runner.model)
    return built if isinstance(built, Verify) else None


def after_init_cuda_graphs(result, worker, *args, **kwargs):
    """Bind and capture the fused iteration once SGLang's graphs exist (a server with the target's spec built)."""
    target = _target(worker)
    if target is not None:
        target.fused = target.dflash_iteration(
            block_size=target.block,
            draft_model=worker.draft_model,
            draft_kv_pool=worker.draft_model_runner.token_to_kv_pool,
        )
    return result


def around_forward_batch_generation(
    original, worker, batch, *args, on_publish=None, **kwargs
):
    """The fused iteration for an eligible step, else SGLang's step."""
    target = _target(worker)
    out = None
    if target is not None and target.fused is not None:
        out = target.fused(
            batch,
            needs_logits=not worker._can_skip_verify_logits(batch, batch.sampling_info),
            on_publish=on_publish,
        )
    if out is None:
        return original(worker, batch, *args, on_publish=on_publish, **kwargs)
    return out


def install() -> None:
    """The two hooks on SGLang's DFlash2 worker where SGLang has no DFlash iteration slot (the target spec's build
    calls this once per process, before SGLang builds the worker)."""
    if SLOT:
        return
    HookRegistry.register(
        f"{WORKER}.init_cuda_graphs", after_init_cuda_graphs, HookType.AFTER
    )
    HookRegistry.register(
        f"{WORKER}.forward_batch_generation",
        around_forward_batch_generation,
        HookType.AROUND,
    )
    HookRegistry.apply_hooks()
