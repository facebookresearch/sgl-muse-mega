"""The plugin's entry point, what ``register`` registers with SGLang (the specs, by path), the hooks the target's build
installs, and the specs' refusals on configurations the kernels do not serve (CPU).
"""

import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sglang = pytest.importorskip("sglang")

from sgl_muse_mega import plugin  # noqa: E402
from sgl_muse_mega.muse_spark import draft, spec  # noqa: E402
from sgl_muse_mega.muse_spark.config import POST_EPS, Config  # noqa: E402
from sgl_muse_mega.muse_spark.runtime import build as runtime_build  # noqa: E402
from sgl_muse_mega.muse_spark.runtime import iteration  # noqa: E402
from sgl_muse_mega.muse_spark.runtime.build import INDEX  # noqa: E402
from sgl_muse_mega.muse_spark.sm103.dflash.weights import (  # noqa: E402
    DRAFT_FILE,
    DraftConfig,
)

from sglang.kernels.plugin_kernels.nvidia import spec as megakernels  # noqa: E402
from sglang.srt.plugins.hook_registry import HookRegistry  # noqa: E402

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_the_entry_point_is_the_one_sglang_loads():
    import tomllib

    eps = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"][
        "sglang.srt.plugins"
    ]
    assert eps == {megakernels.ENTRY_POINT: "sgl_muse_mega.plugin:register"}


def test_register_adds_the_specs():
    plugin.register()
    for architecture in plugin.SPECS:
        spec, why = megakernels.find_spec(architectures=[architecture], isa="sm_103")
        assert spec is not None, why
        assert spec.api_version == megakernels.MEGAKERNEL_API_VERSION


def test_register_alone_imports_no_model_code_and_installs_no_hook():
    """Every SGLang process runs register: without the flag nothing of the model's package loads and SGLang's DFlash2
    worker keeps its own methods."""
    code = (
        "import sys\n"
        "from sglang.srt.plugins.hook_registry import HookRegistry\n"
        "from sgl_muse_mega import plugin\n"
        "plugin.register()\n"
        "print(sorted(m for m in sys.modules if m.startswith('sgl_muse_mega.muse_spark')))\n"
        "print(sorted(HookRegistry._hooks))\n"
    )
    root = str(Path(plugin.__file__).resolve().parents[1])
    path = os.pathsep.join(filter(None, (root, os.environ.get("PYTHONPATH"))))
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": path},
    ).stdout.splitlines()
    assert out[-2:] == ["[]", "[]"], out


@pytest.mark.parametrize("slot", [False, True], ids=["hooks", "slot"])
def test_the_build_installs_the_hooks_unless_sglang_has_the_slot(monkeypatch, slot):
    monkeypatch.setattr(HookRegistry, "_hooks", collections.defaultdict(list))
    monkeypatch.setattr(iteration, "SLOT", slot)
    applied = []
    monkeypatch.setattr(HookRegistry, "apply_hooks", lambda: applied.append(True))
    iteration.install()
    hooks = {
        f"{iteration.WORKER}.init_cuda_graphs",
        f"{iteration.WORKER}.forward_batch_generation",
    }
    assert dict(HookRegistry._hooks).keys() == (set() if slot else hooks)
    assert applied == ([] if slot else [True])


def test_the_specs_serve_only_their_isa():
    plugin.register()
    spec, why = megakernels.find_spec(
        architectures=list(plugin.SPECS)[:1], isa="sm_100"
    )
    assert spec is None and "sm_103" in why


def muse_config(exclude=(), **text) -> SimpleNamespace:
    """A Muse Spark Open config.json as SGLang hands it to a spec, with ``text`` overriding text_config fields and
    ``exclude`` joining the quantization's excluded modules."""
    c = Config()
    layers = range(c.layers)
    tc = dict(
        hidden_size=c.hidden,
        num_hidden_layers=c.layers,
        num_attention_heads=c.q_heads,
        num_key_value_heads=c.kv_heads,
        head_dim=c.head_dim,
        vocab_size=c.vocab,
        num_local_experts=c.experts,
        num_experts_per_tok=c.topk,
        num_shared_experts=0,
        moe_hidden_size=c.moe_hidden,
        moe_intermediate_size=c.moe_inter,
        sliding_window=c.swa_window + 1,
        rms_norm_eps=c.rms_eps,
        post_norm_eps=POST_EPS,
        final_logit_softcapping=c.softcap,
        routed_scaling_factor=c.route_scale,
        layer_types=[
            "full_attention" if c.is_global(i) else "sliding_attention" for i in layers
        ],
        layer_rope_theta=[0 if c.is_global(i) else c.rope_theta for i in layers],
        mlp_layer_types=["sparse"] * c.layers,
        qk_scale_factor=c.q_scale,
        output_multiplier=c.output_multiplier,
    )
    tc.update(text)
    quant = dict(
        quant_algo="NVFP4",
        group_size=16,
        exclude_modules=["layers.0.", "layers.61.", "self_attn", "lm_head", *exclude],
    )
    return SimpleNamespace(text_config=SimpleNamespace(**tc), quantization_config=quant)


def draft_dir(
    tmp_path, architectures=("DFlash2DraftModel",), weights=True, name="draft"
) -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": list(architectures)}))
    if weights:
        (d / DRAFT_FILE).write_bytes(b"")
    return d


def server(tmp_path, **kw) -> SimpleNamespace:
    (tmp_path / INDEX).write_text("{}")
    args = dict(
        context_length=131072,
        tp_size=8,
        nnodes=2,
        watchdog_timeout=300,
        load_format="auto",
        model_path=str(tmp_path),
        speculative_draft_model_path=str(draft_dir(tmp_path)),
    )
    args.update(kw)
    return SimpleNamespace(**args)


def test_the_public_config_is_served(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "toolchain_reasons", lambda: [])
    assert spec.config_disable_reasons(server(tmp_path), muse_config()) == []
    assert (
        spec.config_disable_reasons(
            server(tmp_path, tp_size=4, nnodes=1), muse_config()
        )
        == []
    )


@pytest.mark.parametrize(
    "text, args, reason",
    [
        (dict(head_dim=128), {}, "text_config.head_dim"),
        (dict(layer_rope_theta=[1e6] * 62), {}, "text_config.layer_rope_theta"),
        (dict(output_multiplier=0.2), {}, "text_config.output_multiplier"),
        ({}, dict(tp_size=8, nnodes=1), "groups of 4"),
        ({}, dict(context_length=None), "--context-length"),
        ({}, dict(watchdog_timeout=60), "--watchdog-timeout"),
        ({}, dict(load_format="dummy"), "the build reads the experts"),
    ],
    ids=[
        "head_dim",
        "rope",
        "multiplier",
        "tp8_one_node",
        "no_context",
        "watchdog",
        "dummy",
    ],
)
def test_refusals_name_the_reason(tmp_path, monkeypatch, text, args, reason):
    monkeypatch.setattr(spec, "toolchain_reasons", lambda: [])
    reasons = spec.config_disable_reasons(server(tmp_path, **args), muse_config(**text))
    assert any(reason in r for r in reasons), reasons


@pytest.mark.parametrize(
    "tree, length, refused", [("1", 4.0, True), ("1", -1.0, False), ("0", 4.0, False)]
)
def test_the_token_tree_refuses_a_simulated_length(
    tmp_path, monkeypatch, tree, length, refused
):
    """The tree's forced accept names a leaf row, which forces no length: the megakernel refuses the pair."""
    monkeypatch.setattr(spec, "toolchain_reasons", lambda: [])
    monkeypatch.setattr(spec, "SIMULATE_ACC_LEN", length)
    monkeypatch.setenv(spec.TREE_ENV, tree)
    cfg = server(tmp_path, speculative_num_draft_tokens=16)
    reasons = spec.config_disable_reasons(cfg, muse_config())
    assert any("SGLANG_SIMULATE_ACC_LEN" in r for r in reasons) == refused, reasons


def test_another_draft_architecture_is_refused_before_the_build(tmp_path, monkeypatch):
    """The fused iteration needs the DFlash2 draft's spec: any other draft is refused at resolution, before the
    expert copy (SGLang's v1 DFlash draft, a hub id, a sharded draft)."""
    monkeypatch.setattr(spec, "toolchain_reasons", lambda: [])
    for d, reason in (
        (draft_dir(tmp_path, ["DFlashDraftModel"], name="v1"), "architectures"),
        (draft_dir(tmp_path, weights=False, name="sharded"), DRAFT_FILE),
        (tmp_path / "missing", "config.json"),
    ):
        cfg = server(tmp_path)
        cfg.speculative_draft_model_path = str(d)
        reasons = spec.config_disable_reasons(cfg, muse_config())
        assert any(reason in r for r in reasons), reasons


def test_the_draft_pool_must_hold_bf16_rows():
    allowed = spec.SPEC.requires.server_args["speculative_draft_kv_cache_dtype"]
    assert None in allowed and "bf16" in allowed and "fp8_e4m3" not in allowed


def test_the_bf16_expert_layers_follow_the_expert_paths():
    """An excluded module elsewhere (a vision tower's layer 5) does not make layer 5's experts bf16."""
    assert Config().hf_reasons(muse_config(["model.vision_tower.layers.5."])) == []
    reasons = Config().hf_reasons(
        muse_config(["model.language_model.layers.5.mlp.experts"])
    )
    assert any("bf16-expert layers (0, 5, 61)" in r for r in reasons), reasons


def draft_config(**dflash) -> SimpleNamespace:
    d = DraftConfig()
    cfg = dict(
        architectures=["DFlash2DraftModel"],
        is_causal=False,
        layer_types=["sliding_attention"] * d.layers,
        hidden_act="silu",
        attention_bias=False,
        rope_parameters=dict(rope_type="default", rope_theta=d.rope_theta),
        hidden_size=d.hidden,
        num_hidden_layers=d.layers,
        num_attention_heads=d.q_heads,
        num_key_value_heads=d.kv_heads,
        head_dim=d.head_dim,
        intermediate_size=d.inter,
        vocab_size=d.vocab,
        sliding_window=d.sliding_window,
        rms_norm_eps=d.rms_eps,
        dflash_config=dict(
            conv_kernel_size=d.conv_taps,
            conv_group_size=d.conv_group,
            target_layer_ids=list(d.capture),
            mask_token_id=d.mask_id,
            selector_rank=d.selector_rank,
            selector_top_k=d.top_k,
            output_multiplier=d.output_multiplier,
            final_logit_softcapping=d.softcap,
        )
        | dflash,
    )
    return SimpleNamespace(to_dict=lambda: cfg)


@pytest.mark.parametrize(
    "context, refused", [(131072, False), (1048576, False), (1048577, True)]
)
def test_the_rope_rows_bound_the_context(monkeypatch, context, refused):
    # SGLang's rope rows for the published checkpoint: its max_position_embeddings
    tp = SimpleNamespace(world_size=4, rank_in_group=0, cpu_group=None)
    monkeypatch.setattr(spec, "get_parallel", lambda: SimpleNamespace(tp_group=tp))
    monkeypatch.setattr(spec, "alias_reasons", lambda *a: [])
    monkeypatch.setattr(
        spec, "rope_rows", lambda m: torch.empty(1048576, 64, device="meta")
    )
    monkeypatch.setattr(
        spec, "get_model", lambda: SimpleNamespace(context_length=context)
    )
    monkeypatch.setattr(
        spec.dist,
        "all_gather_object",
        lambda out, obj, group: out.__setitem__(slice(None), [obj] * len(out)),
    )
    reasons = spec.model_disable_reasons(None, None)
    assert reasons == (
        [f"--context-length {context} is above SGLang's 1048576 rope rows"]
        if refused
        else []
    )


def test_the_published_draft_is_served_beside_a_built_target(monkeypatch):
    monkeypatch.setattr(runtime_build, "TARGET", object())
    assert draft.model_disable_reasons(None, draft_config()) == []


def test_the_draft_needs_the_target_built(monkeypatch):
    monkeypatch.setattr(runtime_build, "TARGET", None)
    assert draft.model_disable_reasons(None, draft_config()) == [
        "the target's megakernel is not built in this process"
    ]


def test_another_draft_is_refused(monkeypatch):
    monkeypatch.setattr(runtime_build, "TARGET", object())
    reasons = draft.model_disable_reasons(
        None, draft_config(target_layer_ids=[2, 17, 31, 45, 60])
    )
    assert reasons == [f"capture (2, 17, 31, 45, 60) is not {DraftConfig().capture!r}"]


def test_a_draft_variant_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(runtime_build, "TARGET", object())
    cfg = draft_config()
    cfg.to_dict()["hidden_act"] = "gelu"
    (reason,) = draft.model_disable_reasons(None, cfg)
    assert "ValueError" in reason and "silu" in reason
