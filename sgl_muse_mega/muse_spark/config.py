"""The model (Muse Spark Open) and the deployment it is cut for: TP8 over two 4-GPU nodes of one NVLink
domain (or TP4 on one node: ``Config(tp=4)``, the kernels' ``-DMK_WORLD=tp4``), one sequence verifying 15 draft tokens
(16 rows). Whole-model dims are fields; per-rank shapes are derived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# the post norms' eps (the attention output's and the experts'), a constant of the kernels
POST_EPS = 1e-8


@dataclass(frozen=True)
class Config:
    # the model
    hidden: int = 8192
    layers: int = 62
    q_heads: int = 128
    kv_heads: int = 16
    head_dim: int = 64
    vocab: int = 202048
    experts: int = 256
    topk: int = 8
    moe_hidden: int = 4096  # the experts' width (the pooler's output)
    moe_inter: int = 4096  # expert intermediate
    # layer i is global (NoPE, unbounded window) when i % 4 == 1, else sliding window
    global_every: int = 4
    swa_window: int = 2047
    # experts excluded from NVFP4 in the checkpoint
    bf16_moe_layers: tuple[int, ...] = (0, 61)
    rms_eps: float = 1e-5
    # the rope rows' base (the window's rope rows are cos / sin of position x 500000^(-2i/64))
    rope_theta: float = 500000.0
    # the model's output_multiplier, 2^-2.5 = 1 / sqrt(32): it scales the router's logits before the sigmoid and the
    # lm_head's logits before the softcap
    output_multiplier: float = 32**-0.5
    softcap: float = 20.0
    route_scale: float = 1.0
    context_length: int = 131072
    # the deployment: the tensor-parallel world (8, or 4 on one node)
    tp: int = 8
    # ranks per node (node-major); the pooler / unpooler gathers stay inside a node
    node_ranks: int = 4
    bs: int = 1
    draft: int = 15

    @property
    def q_scale(self) -> float:
        # the checkpoint's attention scale_query_by (43.7840518911) over sqrt(head_dim): applied to q after the
        # weightless qk-norm
        return 43.7840518911 / math.sqrt(self.head_dim)

    @property
    def world(self) -> str:
        """The kernels' world row (build.WORLDS, -DMK_WORLD)."""
        return f"tp{self.tp}"

    @property
    def rows(self) -> int:
        return self.bs * (1 + self.draft)

    @property
    def q_heads_rank(self) -> int:
        return self.q_heads // self.tp

    @property
    def kv_heads_rank(self) -> int:
        return self.kv_heads // self.tp

    @property
    def oproj_k(self) -> int:
        return self.q_heads_rank * self.head_dim

    @property
    def qkv_row(self) -> int:
        # [q | k | v | o-gate]: the o-gate is q-width
        return 2 * (self.q_heads_rank + self.kv_heads_rank) * self.head_dim

    @property
    def kv_row_bytes(self) -> int:
        return self.kv_heads_rank * self.head_dim * 2

    @property
    def inter_rank(self) -> int:
        # W1: the rank's slice of every expert's intermediate
        return self.moe_inter // self.tp

    @property
    def moe_hidden_rank(self) -> int:
        # W2: the rank's slice of every expert's output rows
        return self.moe_hidden // self.tp

    @property
    def vocab_rank(self) -> int:
        return self.vocab // self.tp

    @property
    def lm_head_pad(self) -> int:
        # the schedule's 256-row pair tiles; the weight stays raw
        return (self.vocab_rank + 255) // 256 * 256

    @property
    def local_experts(self) -> int:
        return self.experts // self.tp

    def is_global(self, layer: int) -> bool:
        return layer % self.global_every == 1

    def attn_type(self, layer: int) -> str:
        return "global" if self.is_global(layer) else "swa"

    def is_bf16(self, layer: int) -> bool:
        return layer in self.bf16_moe_layers

    def image(self, layer: int) -> str:
        """The layer's kernel image (build.LAYER_IMAGES): attention type x MoE variant x last layer."""
        if layer == self.layers - 1:
            assert self.is_bf16(layer) and self.is_global(
                layer
            ), "the tail image is the global attention type's bf16 variant"
            return "global_bf16_tail"
        return self.attn_type(layer) + ("_bf16" if self.is_bf16(layer) else "")

    def validate(self) -> None:
        assert self.tp in (4, 8), f"the kernels serve TP8 and TP4, not TP{self.tp}"
        assert (
            self.vocab % self.tp == 0
            and self.q_heads % self.tp == 0
            and self.kv_heads % self.tp == 0
        )
        assert (
            self.moe_inter % self.tp == 0
            and self.moe_hidden % self.tp == 0
            and self.tp % self.node_ranks == 0
        )
        assert self.rows == 16, "the kernels are cut for 16 window rows"

    def hf_reasons(self, hf_config) -> list[str]:
        """Every way a checkpoint's config (Muse Spark's: text_config, quantization_config) differs from the model the
        kernels are built for."""
        tc = hf_config.text_config
        want = dict(
            hidden_size=self.hidden,
            num_hidden_layers=self.layers,
            num_attention_heads=self.q_heads,
            num_key_value_heads=self.kv_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab,
            num_local_experts=self.experts,
            num_experts_per_tok=self.topk,
            num_shared_experts=0,
            moe_hidden_size=self.moe_hidden,
            moe_intermediate_size=self.moe_inter,
            sliding_window=self.swa_window + 1,
            rms_norm_eps=self.rms_eps,
            post_norm_eps=POST_EPS,
            final_logit_softcapping=self.softcap,
            routed_scaling_factor=self.route_scale,
            layer_types=[
                "full_attention" if self.is_global(i) else "sliding_attention"
                for i in range(self.layers)
            ],
            layer_rope_theta=[
                0 if self.is_global(i) else self.rope_theta for i in range(self.layers)
            ],
            mlp_layer_types=["sparse"] * self.layers,
        )
        reasons = [
            f"text_config.{k} {getattr(tc, k, None)!r} is not {v!r}"
            for k, v in want.items()
            if getattr(tc, k, None) != v
        ]
        qk = getattr(tc, "qk_scale_factor", None)
        if qk is None or abs(qk - self.q_scale) > 1e-12:
            reasons.append(
                f"text_config.qk_scale_factor {qk!r} is not {self.q_scale!r}"
            )
        # the multiplier is an fp32 kernel argument
        mult = getattr(tc, "output_multiplier", None)
        if mult is None or float(torch.tensor(mult, dtype=torch.float32)) != float(
            torch.tensor(self.output_multiplier, dtype=torch.float32)
        ):
            reasons.append(
                f"text_config.output_multiplier {mult!r} is not {self.output_multiplier!r}"
            )
        quant = getattr(hf_config, "quantization_config", None) or {}
        if (quant.get("quant_algo"), quant.get("group_size")) != ("NVFP4", 16):
            reasons.append(
                "the experts are not NVFP4 with 16-element scale groups (quantization_config)"
            )
        # an entry covers a layer's experts as SGLang's quantization check reads it: a part of the experts' module
        # path, or a path inside it
        excluded = quant.get("exclude_modules", [])
        experts = [
            f"model.language_model.layers.{i}.mlp.experts" for i in range(self.layers)
        ]
        bf16 = tuple(
            i
            for i, path in enumerate(experts)
            if any(x in path or x.startswith(path) for x in excluded)
        )
        if bf16 != self.bf16_moe_layers:
            reasons.append(
                f"the bf16-expert layers {bf16} are not {self.bf16_moe_layers} "
                "(quantization_config.exclude_modules)"
            )
        return reasons
