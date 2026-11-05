"""The DFlash2 draft's configuration and weights: one rank's shard of the published checkpoint, read from its
safetensors, the rope rows the draft gathers, and the kernels' weight and coefficient layouts. A layout takes a rank's
shard (``shard_layer``) to the bytes a kernel streams; the kernels read nothing else, so these functions are the one
source of their layouts:

  pack_qkv / unpack_qkv          a draft layer's qkv rows (the dense qkv image)
  pack_gate_up / unpack_gate_up  gate and up rows interleaved 8 + 8 (the SwiGLU hook's operand pairs)
  pack_tiles / unpack_tiles      128 x 128 tiles with 16-B chunks XOR-swizzled by row (the entry's ldmatrix reads)
  pack_conv_proj0_tiles          layer 0's attention-conv projection, the rank's output rows as tiles (the entry)
  coeff_order / pack_conv_proj_packets / conv_proj_shard / pack_coeff / unpack_coeff
                                 a conv's coefficients in the finish tile's 16-B packet order (the layer's
                                 coefficient-projection units)
  side1_packet_order             layer 0's projection with side 1 in packet order (the entry's gathered side-1 row is
                                 then the layer's finish coefficients)
"""

from __future__ import annotations

import dataclasses

import torch
from safetensors import safe_open

from ...config import Config


@dataclasses.dataclass(frozen=True)
class DraftConfig:
    """The draft's constants. Defaults = the public DFlash2 config.json (RadixArk/Muse-Spark-Open-DFlash2)."""

    hidden: int = 8192
    layers: int = 5
    q_heads: int = 64
    kv_heads: int = 8
    head_dim: int = 128
    inter: int = 28672
    vocab: int = 202048
    sliding_window: int = 2048  # HF convention: counts the query's own position
    rope_theta: float = 500000.0
    # every norm: input / post / final / hidden, q / k head norms
    rms_eps: float = 1e-5
    conv_taps: int = 2
    conv_group: int = 16  # channels sharing one dynamic coefficient
    # the target layers whose output residual the draft reads, concatenated per row into fc's input
    capture: tuple[int, ...] = (1, 16, 30, 44, 59)
    mask_id: int = 200002
    selector_rank: int = 256
    top_k: int = 16
    output_multiplier: float = 0.1767766952966369
    softcap: float = 20.0

    @property
    def window_left(self) -> int:
        """Keys a query sees on each side."""
        return self.sliding_window - 1

    @property
    def conv_groups(self) -> int:
        return self.hidden // self.conv_group

    @property
    def features(self) -> int:
        return len(self.capture)

    @classmethod
    def from_hf(cls, cfg: dict) -> DraftConfig:
        """From a DFlash2 config.json; raises ValueError on the variants this file does not implement."""
        d = cfg["dflash_config"]
        for what, ok in (
            (
                f"architectures {cfg['architectures']}",
                cfg["architectures"] == ["DFlash2DraftModel"],
            ),
            (
                "a causal or non-sliding layer",
                cfg["is_causal"] is False
                and set(cfg["layer_types"]) == {"sliding_attention"},
            ),
            (
                "an activation other than silu, or attention bias",
                cfg["hidden_act"] == "silu" and not cfg["attention_bias"],
            ),
            (
                "a rope type other than default",
                cfg["rope_parameters"]["rope_type"] == "default",
            ),
        ):
            if not ok:
                raise ValueError(what)
        return cls(
            hidden=cfg["hidden_size"],
            layers=cfg["num_hidden_layers"],
            q_heads=cfg["num_attention_heads"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            inter=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
            sliding_window=cfg["sliding_window"],
            rope_theta=float(cfg["rope_parameters"]["rope_theta"]),
            rms_eps=cfg["rms_norm_eps"],
            conv_taps=d["conv_kernel_size"],
            conv_group=d["conv_group_size"],
            capture=tuple(d["target_layer_ids"]),
            mask_id=d["mask_token_id"],
            selector_rank=d["selector_rank"],
            top_k=d["selector_top_k"],
            output_multiplier=d["output_multiplier"],
            softcap=float(d["final_logit_softcapping"]),
        )


@dataclasses.dataclass
class ConvWeights:
    """One DFlashGroupedConv. base [2, taps, H]: side 0 convolves the sublayer's input, side 1 its
    output. proj [2 taps groups, H] = kernel_projection, output n = (side, tap, group) row-major.
    """

    base: torch.Tensor
    proj: torch.Tensor


@dataclasses.dataclass
class LayerWeights:
    """qkv = the rank's q_proj, k_proj, v_proj rows (QKVParallelLinear); gate_up = its gate_proj rows, then its
    up_proj rows (MergedColumnParallelLinear); o / down = its input columns (RowParallelLinear).
    """

    input_norm: torch.Tensor  # [H]
    post_norm: torch.Tensor  # [H]
    q_norm: torch.Tensor  # [D]
    k_norm: torch.Tensor  # [D]
    qkv: torch.Tensor  # [(q + 2 kv) D, H]
    o: torch.Tensor  # [H, q D]
    gate_up: torch.Tensor  # [2 I, H]
    down: torch.Tensor  # [H, I]
    attn_conv: ConvWeights
    mlp_conv: ConvWeights


@dataclasses.dataclass
class DraftWeights:
    layers: list[LayerWeights]
    fc: torch.Tensor  # [H, features H] (the rank's output rows)
    hidden_norm: torch.Tensor  # [H]
    norm: torch.Tensor  # [H]
    pred_codebook: torch.Tensor  # [V, R] predecessor codebook A
    succ_codebook: torch.Tensor  # [V, R] successor codebook S
    hidden_proj: torch.Tensor  # [R, H] W_P


def block(x: torch.Tensor, rank: int, ranks: int, dim: int = 0) -> torch.Tensor:
    """The rank's equal contiguous block of x along dim."""
    assert x.shape[dim] % ranks == 0, (tuple(x.shape), dim, ranks)
    n = x.shape[dim] // ranks
    return x.narrow(dim, rank * n, n)


def shard_layer(
    cfg: DraftConfig, lw: LayerWeights, rank: int, ranks: int
) -> LayerWeights:
    """An unsharded layer -> rank's shard. Rank r owns q heads [r q/ranks, ...) and kv heads [r kv/ranks, ...), so its
    q heads read only its own kv heads (GQA group = q / kv)."""
    assert cfg.q_heads % ranks == 0 and cfg.kv_heads % ranks == 0
    d = cfg.head_dim
    q, k, v = lw.qkv.split([cfg.q_heads * d, cfg.kv_heads * d, cfg.kv_heads * d])
    gate, up = lw.gate_up.chunk(2)
    return dataclasses.replace(
        lw,
        qkv=torch.cat([block(x, rank, ranks) for x in (q, k, v)]),
        o=block(lw.o, rank, ranks, 1),
        gate_up=torch.cat([block(gate, rank, ranks), block(up, rank, ranks)]),
        down=block(lw.down, rank, ranks, 1),
    )


def read_safetensors(path: str, names: list[str]) -> dict[str, torch.Tensor]:
    """The named bf16 tensors of a safetensors file."""
    with safe_open(path, framework="pt") as f:
        out = {name: f.get_tensor(name) for name in names}
    for name, t in out.items():
        assert t.dtype == torch.bfloat16, (name, t.dtype)
    return out


def checkpoint_layer(path: str, i: int) -> LayerWeights:
    """Draft layer i of a DFlash2 checkpoint, unsharded; q / k / v and gate / up fused as sglang loads them
    ."""
    p = f"layers.{i}."
    convs = [
        f"{p}{c}.{n}"
        for c in ("attention_conv", "mlp_conv")
        for n in ("base_kernel", "kernel_projection.weight")
    ]
    names = [f"{p}self_attn.{n}_proj.weight" for n in "qkvo"] + [
        f"{p}mlp.{n}_proj.weight" for n in ("gate", "up", "down")
    ]
    norms = [
        f"{p}{n}.weight"
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        )
    ]
    s = read_safetensors(path, names + norms + convs)
    a = f"{p}self_attn."
    return LayerWeights(
        *(s[n] for n in norms),
        qkv=torch.cat([s[f"{a}{n}_proj.weight"] for n in "qkv"]),
        o=s[f"{a}o_proj.weight"],
        gate_up=torch.cat([s[f"{p}mlp.gate_proj.weight"], s[f"{p}mlp.up_proj.weight"]]),
        down=s[f"{p}mlp.down_proj.weight"],
        attn_conv=ConvWeights(*(s[n] for n in convs[:2])),
        mlp_conv=ConvWeights(*(s[n] for n in convs[2:])),
    )


# the published draft's one safetensors file in its directory
DRAFT_FILE = "model.safetensors"


def load_shard(path: str, rank: int, ranks: int) -> DraftWeights:
    """The rank's shard of a DFlash2 checkpoint file (the directory's ``DRAFT_FILE``)."""
    cfg, layers = DraftConfig(), []
    for i in range(cfg.layers):
        ls = shard_layer(cfg, checkpoint_layer(path, i), rank, ranks)
        # the row-parallel shards are column views: copy them so the unsharded tensors are freed
        layers.append(
            dataclasses.replace(ls, o=ls.o.contiguous(), down=ls.down.contiguous())
        )
    sel = "candidate_selector."
    names = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    names += [f"{sel}{n}" for n in ("predecessor_codebook", "successor_codebook")]
    names.append(f"{sel}hidden_projection.weight")
    g = read_safetensors(path, names)
    return DraftWeights(
        layers,
        fc=block(g[names[0]], rank, ranks).clone(),
        hidden_norm=g[names[1]],
        norm=g[names[2]],
        pred_codebook=g[names[3]],
        succ_codebook=g[names[4]],
        hidden_proj=g[names[5]],
    )


def rope_cos_sin(
    pos: torch.Tensor, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows of sglang's fp32 neox cos / sin cache at pos -> [n, dim / 2]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = pos.to(torch.float32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


# ---- the kernels' layouts ----

CHUNK = 8  # bf16 elements per 16-B chunk
TILE_N, STAGE_K = 128, 128  # the entry's packed W tile: rows x K columns
# the coefficient gathers' in-node group: the target's node-major groups
NODE_RANKS = Config.node_ranks


# ---- the dense images' W ----


def pack_qkv(qkv: torch.Tensor, cfg: DraftConfig, ranks: int) -> torch.Tensor:
    """The reference's per-rank qkv rows ([q heads | k head | v head] x head_dim) -> the kernel's W: the same head
    order (two 64-row W blocks per head), contiguous."""
    n = (cfg.q_heads + 2 * cfg.kv_heads) // ranks * cfg.head_dim
    assert qkv.shape[0] == n and qkv.dtype == torch.bfloat16, (
        tuple(qkv.shape),
        qkv.dtype,
    )
    return qkv.contiguous()


def unpack_qkv(
    w: torch.Tensor, cfg: DraftConfig, ranks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's qkv W -> (q [q heads x head_dim, H], k [kv heads x head_dim, H], v)."""
    q, kv = cfg.q_heads // ranks * cfg.head_dim, cfg.kv_heads // ranks * cfg.head_dim
    return tuple(w.split([q, kv, kv]))


def pack_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """The reference's per-rank gate_up rows (gate rows, then up rows) -> the kernel's W: in 16-row group g of 64-row
    block b, gate row 32 b + 8 g + l at W row 64 b + 16 g + l and its up row at 64 b + 16 g + 8 + l (l < 8), so the
    SwiGLU operands of one output sit in lanes l and l + 8 of one epilogue warp. The intermediate keeps its natural
    column order (the down projection reads it unpermuted)."""
    gate, up = gate_up.chunk(2)
    inter, h = gate.shape
    assert inter % 32 == 0, inter
    pair = torch.stack(
        [gate.view(inter // 32, 4, 8, h), up.view(inter // 32, 4, 8, h)], 2
    )
    return pair.reshape(2 * inter, h).contiguous()


def unpack_gate_up(w: torch.Tensor) -> torch.Tensor:
    """The kernel's gate_up W (or its GEMM's output columns, transposed) -> gate rows, then up rows."""
    n, h = w.shape
    pair = w.reshape(n // 64, 4, 2, 8, h)
    return torch.cat(
        [pair[:, :, 0].reshape(n // 2, h), pair[:, :, 1].reshape(n // 2, h)]
    )


# ---- the entry's swizzled tiles ----


def _tile_index(device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(TILE_N, device=device)[:, None]
    return rows, torch.arange(STAGE_K // CHUNK, device=device)[None, :] ^ (rows & 7)


def pack_tiles(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> [N / 128 blocks, K / 128 stages, 128, 128]: each 128 x 128 tile row-major, its 16-B chunk c of row r
    stored at chunk c ^ (r & 7) (the kernel's ldmatrix reads 8 rows of one chunk from 8 distinct bank groups).
    """
    n, k = w.shape
    assert n % TILE_N == 0 and k % STAGE_K == 0, (n, k)
    t = w.reshape(n // TILE_N, TILE_N, k // STAGE_K, STAGE_K // CHUNK, CHUNK).permute(
        0, 2, 1, 3, 4
    )
    rows, src = _tile_index(w.device)
    return (
        t[:, :, rows, src]
        .reshape(n // TILE_N, k // STAGE_K, TILE_N, STAGE_K)
        .contiguous()
    )


def unpack_tiles(p: torch.Tensor) -> torch.Tensor:
    blocks, stages = p.shape[:2]
    rows, src = _tile_index(p.device)
    # the XOR is an involution
    t = p.reshape(blocks, stages, TILE_N, STAGE_K // CHUNK, CHUNK)[:, :, rows, src]
    return t.permute(0, 2, 1, 3, 4).reshape(blocks * TILE_N, stages * STAGE_K)


def pack_conv_proj0_tiles(proj: torch.Tensor, rank: int, ranks: int) -> torch.Tensor:
    """Layer 0's attention_conv.kernel_projection [2 taps groups, H] -> the rank's packed output rows (the entry runs
    the anchor row's coefficients, an output shard per rank)."""
    return pack_tiles(block(proj, rank, ranks))


# ---- the layer's coefficient packets ----


def coeff_order(taps: int, groups: int) -> torch.Tensor:
    """packed row -> (tap x groups + group) of one side: packed row 8 j + 4 tap + q = (tap, group 4 j + q), so the 8
    coefficients one finish tile (64 channels = 4 groups) needs for one block row are one 16-B packet.
    """
    assert taps == 2 and groups % 4 == 0, (taps, groups)
    j, t, q = torch.meshgrid(
        torch.arange(groups // 4), torch.arange(taps), torch.arange(4), indexing="ij"
    )
    return (t * groups + 4 * j + q).reshape(-1)


def pack_conv_proj_packets(proj: torch.Tensor, taps: int) -> torch.Tensor:
    """kernel_projection [2 sides x taps x groups, H] -> both sides' rows in packet order, side 0 first."""
    n = proj.shape[0] // 2
    order = coeff_order(taps, n // taps)
    return torch.cat([proj[:n][order], proj[n:][order]])


def conv_proj_shard(packed: torch.Tensor, rank: int) -> torch.Tensor:
    """The packed rows a rank streams: its node quarter of each side ([side 0 part; side 1 part]); the coefficient unit
    gathers the quarters into the node's 4 ranks."""
    side = packed.shape[0] // 2
    q, i = side // NODE_RANKS, rank % NODE_RANKS
    return torch.cat(
        [packed[i * q : (i + 1) * q], packed[side + i * q : side + (i + 1) * q]]
    )


def unpack_coeff(rows: torch.Tensor, taps: int) -> torch.Tensor:
    """Packed coefficient rows [n, taps x groups] -> delta [n, taps, groups] (the reference's coefficient layout)."""
    n, w = rows.shape
    out = torch.empty_like(rows)
    out[:, coeff_order(taps, w // taps)] = rows
    return out.view(n, taps, w // taps)


def pack_coeff(delta: torch.Tensor) -> torch.Tensor:
    """delta [n, taps, groups] -> packed rows [n, taps x groups]."""
    n, taps, groups = delta.shape
    return delta.reshape(n, taps * groups)[:, coeff_order(taps, groups)]


def side1_packet_order(proj: torch.Tensor, taps: int) -> torch.Tensor:
    """A conv's kernel_projection [2 sides x taps x groups, H] with side 1's rows in packet order: the entry's gathered
    side-1 row is then layer 0's finish coefficients exactly as the o_proj tile polls them (one 16-B packet per tile
    and block row), and side 0 keeps the order the entry's prepare reads."""
    n = proj.shape[0] // 2
    return torch.cat([proj[:n], proj[n:][coeff_order(taps, n // taps)]])
