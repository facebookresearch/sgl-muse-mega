"""The SGLang-facing runtime on CPU, with stub batches, a stub worker, a meta-tensor model and a tiny checkpoint: which
steps the verify covers (``Verify.plan``), which steps the fused iteration takes and with which graph (``Fused.mode``),
the flush of a run's last commit into SGLang's draft pool, a request's seed, what the hook does on the others, the
fused step's exported tokens, the window's slots, the KV re-check, the aliased tensors' refusals and the expert shards
read from a checkpoint (``layer_experts``, ``sm103.weights.shard_*``).
"""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from safetensors.torch import save_file  # noqa: E402
from sgl_muse_mega.muse_spark.config import Config  # noqa: E402
from sgl_muse_mega.muse_spark.runtime import build, executor, iteration  # noqa: E402
from sgl_muse_mega.muse_spark.sm103 import weights  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash.head import N_WORDS, WORDS  # noqa: E402

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2  # noqa: E402
from sglang.srt.speculative.spec_utils import sample_simulated_acc_len  # noqa: E402

# ---- the verify's coverage ----


def verify(block: int = 16) -> executor.Verify:
    cfg = Config(tp=4)
    return executor.Verify(cfg, block, {}, {}, None, None, None, 0, "cpu", None, "")


def forward_batch(verify_mode: bool = True, bs: int = 1) -> SimpleNamespace:
    mode = SimpleNamespace(is_target_verify=lambda: verify_mode)
    return SimpleNamespace(forward_mode=mode, batch_size=bs)


@pytest.mark.parametrize(
    "capture, ids, fb, covered",
    [
        (True, 16, forward_batch(), True),
        (False, 16, forward_batch(), False),
        (True, None, forward_batch(), False),
        (True, 16, forward_batch(bs=2), False),
        (True, 16, forward_batch(verify_mode=False), False),
        (True, 8, forward_batch(), False),
    ],
    ids=["verify_bs1", "eager", "omni_reentry", "bs2", "decode", "other_block"],
)
def test_the_verify_covers_the_batch1_verify_under_capture(
    monkeypatch, capture, ids, fb, covered
):
    monkeypatch.setattr(executor, "get_is_capture_mode", lambda: capture)
    input_ids = None if ids is None else torch.zeros(ids, dtype=torch.int64)
    step = verify(16).plan(input_ids, fb)
    assert (step is not None) == covered
    if covered:
        assert (step.mode, step.rows) == ("target_verify", 16)


# ---- the fused iteration's eligibility and the hook ----


def sampling(**kw) -> SimpleNamespace:
    """A sampling_info with nothing but temperature / top-k / top-p; ``kw`` overrides a field."""
    s = dict(
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=None,
        sampling_mask_batch_indices=None,
    )
    s.update(kw)
    return SimpleNamespace(**s)


def req(rid: str = "a", top_k: int = 1, seed: int | None = None) -> SimpleNamespace:
    """A request: greedy (SGLang normalizes greedy to top_k 1), else sampled."""
    return SimpleNamespace(
        rid=rid, sampling_params=SimpleNamespace(top_k=top_k, sampling_seed=seed)
    )


def batch(**kw) -> SimpleNamespace:
    """A batch-1 greedy DFlash2 decode step; ``kw`` overrides a field."""
    b = dict(
        extend=False,
        idle=False,
        is_extend_in_batch=False,
        reqs=[req()],
        spec_info=object.__new__(DFlashDraftInputV2),
        sampling_info=sampling(),
        skip_logits=True,
        has_grammar=False,
        return_logprob=False,
    )
    b.update(kw)
    mode = SimpleNamespace(is_extend=lambda: b["extend"], is_idle=lambda: b["idle"])
    return SimpleNamespace(forward_mode=mode, **b)


def worker() -> SimpleNamespace:
    """SGLang's own rule for the steps whose logits nobody reads (grammar, logprobs, penalties, ...)."""
    return SimpleNamespace(_can_skip_verify_logits=lambda b, s: b.skip_logits)


SAMPLED = dict(reqs=[req(top_k=64)], skip_logits=False)


def sampled(**kw) -> dict:
    """A batch-1 sampled step's fields; ``kw`` overrides a sampling_info field."""
    return {**SAMPLED, "sampling_info": sampling(**kw)}


@pytest.mark.parametrize(
    "kw, mode",
    [
        ({}, "greedy"),
        (dict(sampling_info=None), "greedy"),
        (dict(extend=True), None),
        (dict(idle=True), None),
        (dict(is_extend_in_batch=True), None),
        (dict(reqs=[req("a"), req("b")]), None),
        (dict(spec_info=object()), None),
        (dict(skip_logits=False), None),
        (SAMPLED, "sample"),
        # the batch flag stays False after a sampled request leaves the batch
        (dict(sampling_info=sampling(is_all_greedy=False)), "greedy"),
        ({**SAMPLED, "has_grammar": True}, None),
        ({**SAMPLED, "return_logprob": True}, None),
        (sampled(need_min_p_sampling=True), None),
        (sampled(has_custom_logit_processor=True), None),
        (sampled(penalizer_orchestrator=SimpleNamespace(is_required=True)), None),
        (sampled(penalizer_orchestrator=SimpleNamespace(is_required=False)), "sample"),
        (sampled(grammar_mask=object()), None),
        (sampled(logit_bias=torch.zeros(1)), None),
        (sampled(sampling_mask_batch_indices=torch.zeros(1)), None),
        ({**SAMPLED, "reqs": [req("a", 64), req("b", 64)]}, None),
    ],
    ids=[
        "greedy_bs1",
        "no_sampling_info",
        "extend",
        "idle",
        "extend_in_batch",
        "bs2",
        "another_spec_input",
        "greedy_logprobs_grammar_or_penalties",
        "sampled_bs1",
        "greedy_after_a_sampled_request_left",
        "sampled_grammar",
        "sampled_logprobs",
        "sampled_min_p",
        "sampled_custom_processor",
        "sampled_penalties",
        "sampled_penalizer_idle",
        "sampled_grammar_mask",
        "sampled_logit_bias",
        "sampled_sampling_mask",
        "sampled_bs2",
    ],
)
def test_the_fused_iteration_takes_batch1_greedy_and_plainly_sampled_decode(kw, mode):
    b, fused = batch(**kw), FusedStub()
    out = fused(b, needs_logits=not b.skip_logits, on_publish=None)
    assert out == (mode and f"fused {mode}")


def test_with_the_token_tree_sampled_steps_are_sglangs():
    fused = FusedStub(sample=False)
    assert fused(batch(**SAMPLED), needs_logits=True, on_publish=None) is None
    assert fused(batch(), needs_logits=False, on_publish=None) == "fused greedy"


class FusedStub:
    """The fused iteration's step decision (``Fused.__call__``, ``Fused.mode``) over a recording step and flush."""

    __call__ = iteration.Fused.__call__
    mode = iteration.Fused.mode

    def __init__(self, sample: bool = True):
        self.rid, self.steps, self.sample, self.pending = "a", [], sample, False
        self.flushes = 0
        self.flush_graph = SimpleNamespace(replay=self._flush)

    def _flush(self):
        self.flushes += 1

    def step(self, batch, mode, on_publish):
        self.steps.append((batch, on_publish))
        self.rid, self.pending = batch.reqs[0].rid, True
        return f"fused {mode}"


def test_the_runs_last_commit_is_flushed_before_any_other_step():
    """After a fused step, the next step of the same request runs fused (its own draft carries the commit to the
    pool); any other step (SGLang's, or another request's fused one) first replays the flush, once.
    """
    fused = FusedStub()
    fused.rid = None
    a, extend = batch(), batch(extend=True)
    # (the step, its result, the flushes and the run's request after it)
    for b, want, flushes, rid in [
        (a, "fused greedy", 0, "a"),
        (a, "fused greedy", 0, "a"),
        (extend, None, 1, None),
        (extend, None, 1, None),
        (a, "fused greedy", 1, "a"),
        (batch(reqs=[req("b")]), "fused greedy", 2, "b"),
        (batch(**{**SAMPLED, "reqs": [req("b", 64)]}), "fused sample", 2, "b"),
    ]:
        out = fused(b, needs_logits=not b.skip_logits, on_publish=None)
        assert (out, fused.flushes, fused.rid) == (want, flushes, rid)


def test_a_requests_seed_is_its_sampling_seed_or_its_ids_hash():
    def seed(rid, s):
        return iteration._seed(req(rid, seed=s))

    assert (seed("x", 7), seed("x", 2**63 + 5), seed("x", -3)) == (7, -(2**63) + 5, -3)
    a = seed("x", None)
    assert a == seed("x", None) and a != seed("y", None) and -(2**63) <= a < 2**63


def test_an_ineligible_step_runs_sglangs_and_restarts_the_request(monkeypatch):
    fused = FusedStub()
    monkeypatch.setattr(iteration, "_target", lambda w: SimpleNamespace(fused=fused))
    calls = []

    def original(w, b, *args, on_publish=None, **kw):
        calls.append(on_publish)
        return "sglang"

    publish = object()
    w = worker()
    out = iteration.around_forward_batch_generation(
        original, w, batch(skip_logits=False), on_publish=publish
    )
    assert out == "sglang" and calls == [publish] and fused.rid is None
    out = iteration.around_forward_batch_generation(
        original, w, batch(), on_publish=publish
    )
    assert out == "fused greedy" and fused.steps[-1][1] is publish and len(calls) == 1


def test_without_the_target_built_every_step_is_sglangs(monkeypatch):
    monkeypatch.setattr(iteration, "_target", lambda w: None)
    out = iteration.around_forward_batch_generation(
        lambda w, b, on_publish=None: "sglang", worker(), batch()
    )
    assert out == "sglang"


def export(fixed: int | None = None) -> tuple[dict, torch.Tensor]:
    """A step's export at OUT_COUNT 18, commit 3, anchor 77, P 4099 over an out ring of 100 + row (block 8); the
    outputs and the step words after it."""
    f = object.__new__(iteration.Fused)
    b, ring = 8, 16
    f.b, f.cols, f.fixed = b, torch.arange(b), fixed
    words = torch.zeros(N_WORDS, dtype=torch.int64)
    words[WORDS["OUT_COUNT"]], words[WORDS["COMMIT"]] = 18, 3
    words[WORDS["ANCHOR"]], words[WORDS["P"]] = 77, 4099
    out_ring = torch.arange(100, 100 + ring, dtype=torch.int32)
    f.it = SimpleNamespace(words=words, accept_buf=SimpleNamespace(out_ring=out_ring))
    out = dict(
        tokens=torch.zeros(1, b, dtype=torch.int64),
        commit=torch.zeros(1, dtype=torch.int32),
        bonus=torch.zeros(1, dtype=torch.int64),
        new_seq_lens=torch.zeros(1, dtype=torch.int64),
    )
    f._export(out)
    return out, words


def test_the_export_reads_the_committed_tokens_from_the_ring():
    """The step's tokens: the accepted drafts, then the bonus, from the out ring ending at OUT_COUNT; zeros past the
    commit; the commit length, the bonus and the new committed length as the worker returns them.
    """
    out, _ = export()
    # OUT_COUNT 18 - commit 3 = 15: ring rows 15, 0, 1
    assert out["tokens"].tolist() == [[115, 100, 101, 0, 0, 0, 0, 0]]
    assert (out["commit"].item(), out["bonus"].item(), out["new_seq_lens"].item()) == (
        3,
        77,
        4099,
    )


def test_the_fixed_token_mode_exports_sglangs_fixed_id():
    """SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed, as SGLang's simulated step: every token and the bonus are the fixed id,
    and the next draft starts from it; the commit and the new committed length are the step's.
    """
    out, words = export(fixed=100)
    assert out["tokens"].tolist() == [[100] * 8]
    assert (out["commit"].item(), out["bonus"].item(), out["new_seq_lens"].item()) == (
        3,
        100,
        4099,
    )
    assert words[WORDS["ANCHOR"]].item() == 100


def commit_lengths(length: float, method: str, b: int) -> dict[int, float]:
    """SGLang's sample_simulated_acc_len distribution over [1, b]: {commit length: probability}."""
    if method == "match-expected":
        x = min(max(length, 1.0), b)
        lo, up = math.floor(x), x - math.floor(x)
        return {lo: 1 - up, lo + 1: up} if up else {lo: 1.0}
    cdf = (
        [0.0]
        + [0.5 * math.erfc((length - k - 0.5) / math.sqrt(2)) for k in range(1, b)]
        + [1.0]
    )
    return {k: cdf[k] - cdf[k - 1] for k in range(1, b + 1)}


@pytest.mark.parametrize(
    "length, method, b, window",
    [
        (5.5, "match-expected", 16, 3),
        (3.25, "match-expected", 8, 3),
        (6.1, "match-expected", 16, 3),
        (8.0, "match-expected", 8, 0),
        (20.0, "match-expected", 16, 0),
        (0.5, "match-expected", 16, 0),
        (5.5, "multinomial", 16, 12),
        (2.0, "multinomial", 8, 12),
        (7.7, "multinomial", 8, 12),
    ],
    ids=[
        "half_b16",
        "quarter_b8",
        "tenth_b16",
        "block_b8",
        "past_block",
        "under_one",
        "multinomial_b16",
        "multinomial_low_clamp",
        "multinomial_high_clamp",
    ],
)
def test_a_simulated_length_forces_sglangs_commit_lengths(
    monkeypatch, length, method, b, window
):
    """verify_accept's forced accepts (commit - 1) follow SGLang's distribution of simulated commit lengths, the same
    table on every rank; any run of consecutive steps (from any entry, across the wrap) totals within ``window``
    commits of the expected."""
    monkeypatch.setattr(iteration, "SIMULATE_ACC_LEN", length)
    monkeypatch.setattr(iteration, "SIMULATE_ACC_METHOD", method)
    t = iteration.simulated_accepts(b)
    assert t.dtype == torch.int32 and t.shape == (iteration.SIMULATED_STEPS,)
    assert torch.equal(t, iteration.simulated_accepts(b))
    commit = t.double() + 1
    want = commit_lengths(length, method, b)
    assert set(commit.long().tolist()) <= set(want)
    for k, p in want.items():
        assert abs((commit == k).double().mean().item() - p) < 1e-3, (k, p)
    expect = sum(k * p for k, p in want.items())
    n = torch.arange(1, commit.numel() + 1, dtype=torch.float64)
    for start in range(0, commit.numel(), 97):
        run = torch.roll(commit, -start).cumsum(0)
        assert (run - n * expect).abs().max().item() <= window + 1e-9, start


@pytest.mark.parametrize(
    "length, method, b",
    [
        (5.5, "match-expected", 16),
        (3.25, "match-expected", 8),
        (5.5, "multinomial", 16),
        (2.0, "multinomial", 8),
    ],
)
def test_the_tables_distribution_is_sglangs_sampler(length, method, b):
    """The distribution the table is built to (``commit_lengths``) against SGLang's own sample_simulated_acc_len."""
    torch.manual_seed(0)
    draws = torch.tensor(
        [sample_simulated_acc_len(length, method, b) for _ in range(20000)]
    )
    want = commit_lengths(length, method, b)
    assert set(draws.tolist()) <= set(want)
    for k, p in want.items():
        assert abs((draws == k).double().mean().item() - p) < 0.015, (k, p)


@pytest.mark.parametrize("override", [None, torch.zeros(4, dtype=torch.int32)])
def test_a_simulated_run_continues_the_forced_accept_table(override):
    """A request's start resets the step words from SGLang's state; with a forced-accept table the iteration word
    (its index) carries over, so a new run does not restart the table at its first entry.
    """
    f = object.__new__(iteration.Fused)
    b, ring, row = 8, 64, 4
    words = torch.zeros(N_WORDS, dtype=torch.int64)
    words[WORDS["ITER"]] = 37

    def start(p0, commit):
        words.zero_()
        words[WORDS["P"]], words[WORDS["COMMIT"]] = p0, commit

    rings = SimpleNamespace(
        k_ring=torch.zeros(ring, row), v_ring=torch.zeros(ring, row)
    )
    f.b, f.cx = b, SimpleNamespace(ring=ring, device="cpu")
    f.it = SimpleNamespace(
        words=words,
        start=start,
        override=override,
        dslot=torch.zeros(2 * b, dtype=torch.int32),
        sync_state=torch.zeros(2, dtype=torch.int64),
        lb=[rings],
    )
    f.r2t = torch.arange(400).view(4, 100)
    f.draft_kv = [(torch.ones(400, row), torch.ones(400, row))]
    b1 = SimpleNamespace(
        seq_lens=torch.tensor([20]),
        req_pool_indices=torch.tensor([1]),
        spec_info=SimpleNamespace(bonus_tokens=torch.tensor([9])),
    )
    f._start(b1, "greedy")
    assert words[WORDS["ITER"]].item() == (0 if override is None else 37)
    assert (words[WORDS["P"]].item(), words[WORDS["ANCHOR"]].item()) == (20, 9)


def test_without_a_simulated_length_the_accept_is_the_kernels(monkeypatch):
    monkeypatch.setattr(iteration, "SIMULATE_ACC_LEN", -1.0)
    assert iteration.simulated_accepts(16) is None and iteration.fixed_token() is None


def test_an_unknown_simulation_method_is_refused(monkeypatch):
    monkeypatch.setattr(iteration, "SIMULATE_ACC_LEN", 4.0)
    monkeypatch.setattr(iteration, "SIMULATE_ACC_METHOD", "uniform")
    with pytest.raises(ValueError, match="simulate_acc_method"):
        iteration.simulated_accepts(16)


@pytest.mark.parametrize("mode, token", [("fixed", 100), ("real-draft-token", None)])
def test_the_token_mode_picks_sglangs_fixed_id_or_the_real_tokens(
    monkeypatch, mode, token
):
    monkeypatch.setattr(iteration, "SIMULATE_ACC_LEN", 4.0)
    monkeypatch.setattr(iteration, "SIMULATE_ACC_TOKEN_MODE", mode)
    assert iteration.fixed_token() == token


# ---- the window's slots and the KV binding ----


def feeds(r2t: torch.Tensor) -> SimpleNamespace:
    n = Config().rows
    z = lambda: torch.zeros(n, dtype=torch.int64)  # noqa: E731
    return SimpleNamespace(
        seq_lens=torch.tensor([5]),
        req_pool_indices=torch.tensor([1]),
        positions=z(),
        out_cache_loc=z(),
        swa_out_cache_loc=z(),
        req_to_token=r2t,
        width=r2t.shape[1],
        full_to_swa=torch.arange(1000, 1000 + r2t.numel() + 1),
    )


def test_the_window_slots_come_from_sglang_or_the_slot_table():
    r2t = torch.arange(2 * 64, dtype=torch.int32).view(2, 64)
    offsets = torch.arange(Config().rows)
    f = feeds(r2t)
    executor.window_slots(f, offsets)
    assert f.positions.tolist() == list(range(5, 21))
    assert f.out_cache_loc.tolist() == list(range(64 + 5, 64 + 21))
    assert torch.equal(f.swa_out_cache_loc, f.out_cache_loc + 1000)
    f = feeds(r2t)
    executor.window_slots(f, offsets, torch.tensor([7, 9]))
    assert f.out_cache_loc[:3].tolist() == [7, 9, 0]
    assert f.swa_out_cache_loc[:2].tolist() == [1007, 1009]


def test_a_moved_kv_buffer_is_named():
    layers = Config().layers
    bufs = [torch.zeros(4, 2) for _ in range(2 * layers)]
    pool = SimpleNamespace(
        get_key_buffer=lambda L: bufs[2 * L], get_value_buffer=lambda L: bufs[2 * L + 1]
    )
    v = verify()
    v.kv_bound = executor._addresses(executor.kv_buffers(pool, layers))
    v.check_kv(pool)
    bufs[2 * 7 + 1] = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match=r"layers \[7\]"):
        v.check_kv(pool)


# ---- the aliased tensors ----


def meta(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device="meta")


def model(cfg: Config, lm_head_rows: int | None = None) -> SimpleNamespace:
    """A Muse Spark model as the build reads it: meta tensors of the shapes and dtypes SGLang loads at cfg's TP."""
    h, mh, e = cfg.hidden, cfg.moe_hidden, cfg.experts

    def norm(n):
        return SimpleNamespace(_frozen_weight=meta(n))

    def gate():
        return SimpleNamespace(
            _alpha=meta(1, h, dtype=torch.float32),
            _beta=meta(1, h, dtype=torch.float32),
        )

    rope = SimpleNamespace(
        cos_sin_cache=meta(cfg.context_length + 32, 64, dtype=torch.float32)
    )
    layers = [
        SimpleNamespace(
            input_layernorm=norm(h),
            pre_feedforward_layernorm=norm(h),
            post_feedforward_layernorm=norm(h),
            post_attention_residual_gate=gate(),
            post_feedforward_residual_gate=gate(),
            self_attn=SimpleNamespace(
                qkv_proj=SimpleNamespace(weight=meta(cfg.qkv_row, h)),
                o_proj=SimpleNamespace(weight=meta(h, cfg.oproj_k)),
                rotary_emb=rope,
            ),
            mlp=SimpleNamespace(
                pre_expert_proj=SimpleNamespace(weight=meta(mh, h)),
                post_expert_proj=SimpleNamespace(weight=meta(h, mh)),
                gate=SimpleNamespace(
                    weight=meta(e, h, dtype=torch.float32),
                    e_score_correction_bias=meta(e, dtype=torch.float32),
                ),
                pre_expert_norm=norm(mh),
                experts=SimpleNamespace(post_expert_norm=norm(mh)),
            ),
        )
        for _ in range(cfg.layers)
    ]

    def vocab(rows):
        return SimpleNamespace(
            weight=meta(rows, h),
            shard_indices=SimpleNamespace(org_vocab_start_index=0),
        )

    lm = SimpleNamespace(
        layers=layers, norm=norm(h), embed_tokens=vocab(cfg.vocab_rank)
    )
    return SimpleNamespace(
        model=SimpleNamespace(language_model=lm),
        lm_head=vocab(lm_head_rows or cfg.vocab_rank),
    )


@pytest.mark.parametrize("tp", [4, 8])
def test_sglangs_tensors_alias_in_place(tp):
    cfg = Config(tp=tp)
    assert build.alias_reasons(model(cfg), cfg, 0) == []


def test_a_wrong_tensor_is_named():
    cfg = Config(tp=8)
    m = model(cfg)
    m.model.language_model.layers[3].mlp.gate.weight = meta(cfg.experts, cfg.hidden)
    (reason,) = build.alias_reasons(m, cfg, 0)
    assert reason.startswith("layer 3 router is (256, 8192) torch.bfloat16"), reason


def test_a_replicated_lm_head_is_refused():
    cfg = Config(tp=8)
    reasons = build.alias_reasons(model(cfg, lm_head_rows=cfg.vocab), cfg, 0)
    assert reasons == ["the lm_head is replicated (full vocabulary on every rank)"]


# ---- the experts read from a checkpoint ----

# the smallest dims the expert packers take at TP4: a 128-row pair tile per rank, 128 output rows per rank
TINY = replace(
    Config(tp=4), experts=2, moe_inter=512, moe_hidden=512, bf16_moe_layers=(0,)
)


def tiny_checkpoint(path) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    e, i, h = TINY.experts, TINY.moe_inter, TINY.moe_hidden

    def u8(*s):
        return torch.randint(0, 256, s, generator=g, dtype=torch.uint8)

    p0, p1 = (f"model.language_model.layers.{L}.mlp.experts." for L in (0, 1))
    t = {
        p0 + "gate_up_proj": torch.randn(e, 2 * i, h, generator=g).bfloat16(),
        p0 + "down_proj": torch.randn(e, h, i, generator=g).bfloat16(),
        p1 + "gate_up_proj": u8(e, 2 * i, h // 2),
        p1
        + "gate_up_proj_weight_scale": u8(e, 2 * i, h // 16).view(torch.float8_e4m3fn),
        p1 + "gate_up_proj_weight_scale_2": torch.rand(e, 2, generator=g) + 0.5,
        p1 + "down_proj": u8(e, h, i // 2),
        p1 + "down_proj_weight_scale": u8(e, h, i // 16).view(torch.float8_e4m3fn),
        p1 + "down_proj_weight_scale_2": torch.rand(e, generator=g) + 0.5,
    }
    save_file(t, str(path / "a.safetensors"))
    (path / build.INDEX).write_text(
        json.dumps({"weight_map": dict.fromkeys(t, "a.safetensors")})
    )
    return t


@pytest.mark.parametrize("rank", range(4))
def test_the_experts_are_the_ranks_rows_of_the_checkpoint(tmp_path, rank):
    t = tiny_checkpoint(tmp_path)
    ck = build.Checkpoint(tmp_path)
    i, i_s, h_s = TINY.moe_inter, TINY.inter_rank, TINY.moe_hidden_rank
    gate, up = slice(rank * i_s, (rank + 1) * i_s), slice(
        i + rank * i_s, i + (rank + 1) * i_s
    )
    out = slice(rank * h_s, (rank + 1) * h_s)
    p = "model.language_model.layers.{}.mlp.experts."

    bf = build.layer_experts(ck, TINY, 0, rank, "cpu")
    g_, u_ = weights.unpack_w1_bf16(bf["w1"])
    w13 = t[p.format(0) + "gate_up_proj"]
    assert torch.equal(g_, w13[:, gate]) and torch.equal(u_, w13[:, up])
    assert torch.equal(
        weights.unpack_w2_bf16(bf["w2"]), t[p.format(0) + "down_proj"][:, out]
    )

    fp = build.layer_experts(ck, TINY, 1, rank, "cpu")
    q = p.format(1)
    codes, sf = t[q + "gate_up_proj"], t[q + "gate_up_proj_weight_scale"].view(
        torch.uint8
    )
    gq, gs, uq, us = weights.unpack_w1_fp4(fp["w1"], fp["w1_sf_blk"])
    assert torch.equal(gq, codes[:, gate]) and torch.equal(uq, codes[:, up])
    assert torch.equal(gs, sf[:, gate]) and torch.equal(us, sf[:, up])
    w2, w2_sf = weights.unpack_w2_fp4(fp["w2"], fp["w2_sf"])
    assert torch.equal(w2, t[q + "down_proj"][:, out])
    assert torch.equal(w2_sf, t[q + "down_proj_weight_scale"].view(torch.uint8)[:, out])
    s2 = t[q + "gate_up_proj_weight_scale_2"]
    assert torch.equal(fp["g1_alphas"], s2[:, 0]) and torch.equal(
        fp["g1_scale_c"], s2[:, 1] / s2[:, 0]
    )
    assert torch.equal(fp["g2_alphas"], t[q + "down_proj_weight_scale_2"])
