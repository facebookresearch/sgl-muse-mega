"""SGLang-facing: the build that runs after the model loads and before SGLang sizes its KV pools: the kernel images
(built or found in the cache), the cross-rank buffers on SGLang's TP group, and the layer kernels' per-rank weights.

Every dense weight, norm weight, residual coefficient, the router, the lm_head and the rope rows are SGLang's own
tensors (no copy): SGLang already stores them in the kernels' layouts (q / k rows in rotary pairs, norm weights with
their gain center added, the residual gates' (alpha, beta) precomputed). The vocab shard is copied into the
caller's symmetric buffer.

The experts are read from the checkpoint's safetensors. SGLang holds them expert-parallel (each rank a quarter or an
eighth of the experts, in FlashInfer's layouts); the layer kernels run them tensor-parallel: every expert, the rank's
slice of its intermediate (W1: gate and up rows) and of its output rows (W2). So each rank reads its rows of every
expert from the checkpoint and packs them for the kernels (``sm103.weights``): codes and scale bytes move unchanged,
and the per-expert global scales fold into the epilogue constants (g1_alphas = s2_gate, g1_scale_c = s2_up / s2_gate,
g2_alphas = s2_down).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.runtime_context import get_model, get_parallel

from ..config import Config
from ..sm103.artifacts import LAYER_IMAGES
from ..sm103.artifacts import build as build_images
from ..sm103.buffers import Buffers
from ..sm103.verify import LayerLibrary, layer_libraries, share_gather_rows
from ..sm103.weights import (
    LayerWeights,
    ModelWeights,
    pack_w1_bf16,
    pack_w1_fp4,
    pack_w2_bf16,
    pack_w2_fp4,
    shard_w2,
    shard_w13,
)
from . import iteration
from .executor import Verify

# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_model", "get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": (
        "rank_in_group",
        "world_size",
        "device_group",
    ),
    "srt/arg_groups/fields/model.py": ("model_path",),
    "srt/models/muse_spark.py": (
        "model",
        "language_model",
        "layers",
        "lm_head",
        "norm",
        "embed_tokens",
        "self_attn",
        "mlp",
        "qkv_proj",
        "o_proj",
        "rotary_emb",
        "gate",
        '"e_score_correction_bias"',
        "pre_expert_proj",
        "post_expert_proj",
        "pre_expert_norm",
        "experts",
        "post_expert_norm",
        "input_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_attention_residual_gate",
        "post_feedforward_residual_gate",
    ),
    "srt/layers/muse_spark/muse_spark_layers.py": (
        '"_frozen_weight"',
        "_alpha",
        "_beta",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
    ),
}

INDEX = "model.safetensors.index.json"
# opt-in: the fused DFlash2 iteration verifies a token tree of the draft's 15 likeliest continuations instead of its
# linear chain (block 16; sm103/kernels/dflash/tree.cuh). Lossless under greedy decoding, not bitwise the linear chain's tokens.
TREE_ENV = "SGLANG_MUSE_MEGA_TREE"


def tree_enabled() -> bool:
    return os.environ.get(TREE_ENV, "0") == "1"


class Checkpoint:
    """A safetensors checkpoint directory; a tensor slice reads only its own rows."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.weight_map = json.loads((self.path / INDEX).read_text())["weight_map"]
        self._files: dict[str, object] = {}

    def slice(self, name: str):
        f = self.weight_map[name]
        if f not in self._files:
            self._files[f] = safe_open(str(self.path / f), framework="pt")
        return self._files[f].get_slice(name)


def layer_experts(ck: Checkpoint, cfg: Config, layer: int, rank: int, device) -> dict:
    """Layer ``layer``'s expert fields of ``LayerWeights`` for ``rank``, on ``device``."""
    p = f"model.language_model.layers.{layer}.mlp.experts."

    def w13(name: str) -> list[torch.Tensor]:
        rows = shard_w13(ck.slice(p + name), rank, cfg.tp, cfg.moe_inter)
        return [t.to(device) for t in rows]

    def w2(name: str) -> torch.Tensor:
        return shard_w2(ck.slice(p + name), rank, cfg.tp, cfg.moe_hidden).to(device)

    if cfg.is_bf16(layer):
        return dict(
            w1=pack_w1_bf16(*w13("gate_up_proj")), w2=pack_w2_bf16(w2("down_proj"))
        )
    (gate_q, up_q), (gate_sf, up_sf) = w13("gate_up_proj"), w13(
        "gate_up_proj_weight_scale"
    )
    w1, w1_sf = pack_w1_fp4(gate_q, gate_sf, up_q, up_sf)
    w2_q, w2_sf = pack_w2_fp4(w2("down_proj"), w2("down_proj_weight_scale"))
    s2 = ck.slice(p + "gate_up_proj_weight_scale_2")[:].to(device)
    return dict(
        w1=w1,
        w1_sf_blk=w1_sf,
        w2=w2_q,
        w2_sf=w2_sf,
        g1_alphas=s2[:, 0].contiguous(),
        g1_scale_c=(s2[:, 1] / s2[:, 0]).contiguous(),
        g2_alphas=ck.slice(p + "down_proj_weight_scale_2")[:].to(device),
    )


BF16, F32 = torch.bfloat16, torch.float32


def _alias_specs(model, cfg: Config):
    """(layer, field) -> (tensor, shape, dtype) of every aliased tensor; layer None: model-level."""
    lm = model.model.language_model
    h, mh = cfg.hidden, cfg.moe_hidden
    out = {
        (None, "entry_pre_w"): (
            lm.layers[0].input_layernorm._frozen_weight,
            (h,),
            BF16,
        ),
        (None, "lm_head"): (
            model.lm_head.weight[: cfg.vocab_rank],
            (cfg.vocab_rank, h),
            BF16,
        ),
        (None, "cos_sin_cache"): (
            lm.layers[0].self_attn.rotary_emb.cos_sin_cache,
            None,
            F32,
        ),
    }
    for i, layer in enumerate(lm.layers):
        att, mlp = layer.self_attn, layer.mlp
        nxt = lm.layers[i + 1].input_layernorm if i + 1 < cfg.layers else lm.norm
        fields = dict(
            qkv=(att.qkv_proj.weight, (cfg.qkv_row, h), BF16),
            o_proj=(att.o_proj.weight, (h, cfg.oproj_k), BF16),
            pooler=(mlp.pre_expert_proj.weight, (mh, h), BF16),
            router=(mlp.gate.weight, (cfg.experts, h), F32),
            unpooler=(mlp.post_expert_proj.weight, (h, mh), BF16),
            expert_bias=(mlp.gate.e_score_correction_bias, (cfg.experts,), F32),
            post_alpha=(layer.post_attention_residual_gate._alpha.view(-1), (h,), F32),
            post_beta=(layer.post_attention_residual_gate._beta.view(-1), (h,), F32),
            post_pre_w=(layer.pre_feedforward_layernorm._frozen_weight, (h,), BF16),
            cross_alpha=(
                layer.post_feedforward_residual_gate._alpha.view(-1),
                (h,),
                F32,
            ),
            cross_beta=(layer.post_feedforward_residual_gate._beta.view(-1), (h,), F32),
            cross_pre_w=(nxt._frozen_weight, (h,), BF16),
            cross_post_w=(layer.post_feedforward_layernorm._frozen_weight, (h,), BF16),
            pre_gamma=(mlp.pre_expert_norm._frozen_weight, (mh,), BF16),
            gamma_p1=(mlp.experts.post_expert_norm._frozen_weight, (mh,), BF16),
        )
        out.update({(i, k): v for k, v in fields.items()})
    return out


def alias_reasons(model, cfg: Config, rank: int) -> list[str]:
    """Every aliased tensor that is missing or not in the kernels' shape, dtype and (row-major) layout, and a vocab
    shard that is not rank ``rank``'s rows [rank V / tp, (rank + 1) V / tp)."""
    try:
        specs = _alias_specs(model, cfg)
    except (AttributeError, IndexError, TypeError) as e:
        return [
            f"the loaded model lacks a tensor the kernels read: {type(e).__name__}: {e}"
        ]
    reasons = []
    for (layer, name), (t, shape, dtype) in specs.items():
        where = name if layer is None else f"layer {layer} {name}"
        if t is None:
            reasons.append(f"{where} is not set")
        elif (
            (shape is not None and tuple(t.shape) != shape)
            or t.dtype != dtype
            or not t.is_contiguous()
        ):
            reasons.append(
                f"{where} is {tuple(t.shape)} {t.dtype}, want {shape} {dtype} contiguous"
            )
    lm = model.model.language_model
    if lm.layers[0].self_attn.rotary_emb.cos_sin_cache.shape[1:] != (cfg.head_dim,):
        reasons.append(f"the rope rows are not {cfg.head_dim} wide (cos | sin)")
    for name, mod in (("embed_tokens", lm.embed_tokens), ("lm_head", model.lm_head)):
        start = mod.shard_indices.org_vocab_start_index
        if mod.weight.shape[0] != cfg.vocab and start != rank * cfg.vocab_rank:
            reasons.append(
                f"{name}'s shard starts at row {start}, not {rank * cfg.vocab_rank}"
            )
    # the lm_head aliases rows [0, V / tp) of its weight: a sharded weight's own rows
    if model.lm_head.weight.shape[0] == cfg.vocab and cfg.tp > 1:
        reasons.append("the lm_head is replicated (full vocabulary on every rank)")
    return reasons


def model_weights(
    model, cfg: Config, rank: int, device, embed_out: torch.Tensor, checkpoint
) -> ModelWeights:
    """Rank ``rank``'s ``ModelWeights``: SGLang's tensors, the vocab shard copied into ``embed_out`` and the experts
    read from ``checkpoint`` (a directory)."""
    specs = _alias_specs(model, cfg)
    embed = model.model.language_model.embed_tokens.weight
    lo = rank * cfg.vocab_rank
    if embed.shape[0] == cfg.vocab:
        embed = embed[lo : lo + cfg.vocab_rank]
    assert embed.shape == embed_out.shape, (tuple(embed.shape), tuple(embed_out.shape))
    embed_out.copy_(embed)
    ck = Checkpoint(checkpoint)
    layers = []
    for i in range(cfg.layers):
        fields = {
            k: specs[(i, k)][0]
            for k in LayerWeights.__dataclass_fields__
            if (i, k) in specs
        }
        layers.append(LayerWeights(**fields, **layer_experts(ck, cfg, i, rank, device)))
    return ModelWeights(
        embed=embed_out,
        entry_pre_w=specs[(None, "entry_pre_w")][0],
        lm_head=specs[(None, "lm_head")][0],
        layers=layers,
    )


# this process's target build (one target per process); the DFlash2 draft spec's build adds the draft's tensors to it
TARGET: Verify | None = None


def build(model, hf_config, ctx) -> Verify:
    """The verify step of a DFlash2 block (``ctx.tokens_per_request`` rows) on this rank: the kernel images built or
    found in the cache, the cross-rank buffers on SGLang's TP group, the weights (SGLang's tensors, the vocab shard,
    the experts read from the checkpoint)."""
    t0 = time.perf_counter()
    tp = get_parallel().tp_group
    rank, device = tp.rank_in_group, torch.device("cuda", torch.cuda.current_device())
    block = ctx.tokens_per_request
    cfg = Config(tp=tp.world_size)
    cfg.validate()
    m8 = block < cfg.rows
    paths = build_images(m8=m8, world=cfg.world)
    libs = layer_libraries(paths)
    consts = {image: libs[image].constants for image in LAYER_IMAGES}
    bufs = Buffers(cfg, consts, device, tp.device_group)
    if m8:
        share_gather_rows(
            bufs, consts["swa"], LayerLibrary(paths["libmk_layer_swa_m8"]).constants
        )
    t1 = time.perf_counter()
    weights = model_weights(
        model, cfg, rank, device, bufs.embed_shard(cfg), get_model().model_path
    )
    torch.cuda.synchronize(device)
    # the expert packing's staging goes back to the driver before SGLang sizes its pools
    torch.cuda.empty_cache()
    summary = (
        f"TP{cfg.tp}, block {block}, {len(paths)} images; experts read in {time.perf_counter() - t1:.0f} s, "
        f"build {time.perf_counter() - t0:.0f} s"
    )
    tree = tree_enabled()
    if tree:
        summary += ", token tree"
    global TARGET
    TARGET = Verify(
        cfg,
        block,
        paths,
        libs,
        bufs,
        weights,
        model,
        rank,
        device,
        tp.device_group,
        summary,
        tree,
    )
    # the flag is on: the fused DFlash2 iteration joins SGLang's DFlash2 worker, which is built after the target
    iteration.install()
    return TARGET
