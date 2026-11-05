"""The model's per-rank tensors in the kernels' layouts, and the MoE weight layouts the layer kernels stream (their
ABI): per rank, an expert's W1 is the rank's I slice of its gate and up rows (I_s = I / tp features each) and its W2
the rank's H slice of output rows over the full I. ``shard_*`` cut a checkpoint's expert stack to the rank's rows,
``pack_*`` build the kernel layouts from the shards, ``unpack_*`` invert them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerWeights:
    """One decoder layer's per-rank tensors in kernel layout (None where the layer's MoE variant reads nothing)."""

    # [qkv_row, H] bf16: [q | k | v | o-gate] rows of the rank's heads
    qkv: torch.Tensor
    o_proj: torch.Tensor  # [H, oproj_k] bf16
    pooler: torch.Tensor  # [moe_hidden, H] bf16, replicated
    router: torch.Tensor  # [E, H] f32, replicated
    unpooler: torch.Tensor  # [H, moe_hidden] bf16, replicated
    expert_bias: torch.Tensor  # [E] f32
    # post-attention norm: residual gate (f32 [H] x2) + pre-norm weight (bf16 [H])
    post_alpha: torch.Tensor
    post_beta: torch.Tensor
    post_pre_w: torch.Tensor
    # cross-layer (last layer: final) norm: + its input-scale post weight
    cross_alpha: torch.Tensor
    cross_beta: torch.Tensor
    cross_pre_w: torch.Tensor
    cross_post_w: torch.Tensor
    pre_gamma: torch.Tensor  # pre-expert norm weight, bf16 [moe_hidden]
    gamma_p1: torch.Tensor  # post-expert norm weight + 1, bf16 [moe_hidden]
    w1: torch.Tensor
    w2: torch.Tensor
    w1_sf_blk: torch.Tensor | None = None
    w2_sf: torch.Tensor | None = None
    g1_alphas: torch.Tensor | None = None
    g1_scale_c: torch.Tensor | None = None
    g2_alphas: torch.Tensor | None = None


@dataclass
class ModelWeights:
    # [vocab_rank, H] bf16: this rank's vocab shard (the caller places it in symmetric memory)
    embed: torch.Tensor
    entry_pre_w: torch.Tensor  # layer 0's pre-attention norm weight, bf16 [H]
    lm_head: torch.Tensor  # [vocab_rank, H] bf16
    layers: list[LayerWeights]


# ---- the MoE layouts ----


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """e4m3 scale factors as their bytes (the packers move codes and scale bytes unchanged)."""
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def shard_w13(
    w13, rank: int, ranks: int, inter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank's (gate, up) rows of a checkpoint w13 [E, 2 inter, ...] (gate rows first; a tensor or a safetensors
    slice, which reads only these rows): rows [r I_s, (r + 1) I_s) of each half."""
    i_s = inter // ranks
    return tuple(
        _bytes(w13[:, h * inter + rank * i_s : h * inter + (rank + 1) * i_s])
        for h in (0, 1)
    )


def shard_w2(w2, rank: int, ranks: int, hidden: int) -> torch.Tensor:
    """The rank's output rows of a checkpoint w2 [E, hidden, ...] (a tensor or a safetensors slice)."""
    h_s = hidden // ranks
    return _bytes(w2[:, rank * h_s : (rank + 1) * h_s])


FBW = 128  # W1 pair-tile width: tile rows 0..127 = gate features fb*128.., rows 128..255 = the same up features
_SF_TILE = 512  # one SF image tile: 4 row groups x 32 rows x 4 bytes (one K64 MMA cell of 128 rows)


def _pair_tiles(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(gate, up) [E, I_s, X] -> pair tiles [E, I_s/128, 256, X]."""
    e, i_s, x = gate.shape
    fb = i_s // FBW
    return torch.cat((gate.reshape(e, fb, FBW, x), up.reshape(e, fb, FBW, x)), dim=2)


def _split_pair_tiles(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, fb, _, x = t.shape
    return t[:, :, :FBW].reshape(e, fb * FBW, x), t[:, :, FBW:].reshape(e, fb * FBW, x)


def pack_w1_fp4(
    gate_q: torch.Tensor, gate_sf: torch.Tensor, up_q: torch.Tensor, up_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank's NVFP4 W1 (gate / up codes [E, I_s, H/2] u8, scales [E, I_s, H/16] u8) -> the kernel's
    (w1 [E, I_s/128, H/256, 256, 128] u8: pair tiles K-blocked in 128-B rows; w1_sf [E I_s/128, (H/64) 1024] u8:
    the g1 SF copy-route image, tile byte n*16 + g*4 + k for tile row 32g + n of K64 cell c; host/runtime.cu
    impl_host_init_tmaps maps both).
    """
    rows = _pair_tiles(gate_q, up_q)
    e, fb, _, kb = rows.shape
    w1 = rows.view(e, fb, 2 * FBW, kb // 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    sf = _pair_tiles(gate_sf, up_sf)
    ksf = sf.shape[-1]
    # (e, fb, half, g, n, cell, k) -> (e, fb, cell, half, n, g, k)
    img = sf.view(e, fb, 2, 4, 32, ksf // 4, 4).permute(0, 1, 5, 2, 4, 3, 6)
    return w1, img.reshape(e * fb, (ksf // 4) * 2 * _SF_TILE)


def unpack_w1_fp4(w1: torch.Tensor, w1_sf: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Inverse of ``pack_w1_fp4`` -> (gate_q, gate_sf, up_q, up_sf)."""
    e, fb, ks, rows, b = w1.shape
    assert rows == 2 * FBW and w1_sf.shape[0] == e * fb, (w1.shape, w1_sf.shape)
    gate_q, up_q = _split_pair_tiles(
        w1.permute(0, 1, 3, 2, 4).reshape(e, fb, rows, ks * b)
    )
    cells = w1_sf.shape[1] // (2 * _SF_TILE)
    sf = (
        w1_sf.view(e, fb, cells, 2, 32, 4, 4)
        .permute(0, 1, 3, 5, 4, 2, 6)
        .reshape(e, fb, rows, 4 * cells)
    )
    gate_sf, up_sf = _split_pair_tiles(sf)
    return gate_q, gate_sf, up_q, up_sf


def pack_w2_fp4(
    w2_q: torch.Tensor, w2_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank's NVFP4 W2 (codes [E, H_s, I/2] u8, scales [E, H_s, I/16] u8) -> (w2 [E, H_s/128, I/256, 128, 128]
    u8: 128x128-byte blocks; w2_sf: the scales as they are).
    """
    e, h_s, kb = w2_q.shape
    w2 = (
        w2_q.reshape(e, h_s // 128, 128, kb // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return w2, w2_sf.contiguous()


def unpack_w2_fp4(
    w2: torch.Tensor, w2_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    e, hb, ks, rows, b = w2.shape
    return w2.permute(0, 1, 3, 2, 4).reshape(e, hb * rows, ks * b), w2_sf


def pack_w1_bf16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The rank's bf16 W1 (gate / up [E, I_s, H]) -> [E, I_s/128, H/64, 256, 64]: pair tiles K-blocked in 128-B
    rows."""
    t = _pair_tiles(gate, up)
    e, fb, rows, h = t.shape
    return t.view(e, fb, rows, h // 64, 64).permute(0, 1, 3, 2, 4).contiguous()


def unpack_w1_bf16(w1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, fb, ks, rows, b = w1.shape
    return _split_pair_tiles(w1.permute(0, 1, 3, 2, 4).reshape(e, fb, rows, ks * b))


def pack_w2_bf16(w2: torch.Tensor) -> torch.Tensor:
    """The rank's bf16 W2 ([E, H_s, I]) -> [E, H_s/128, I/64, 128, 64]."""
    e, h_s, i = w2.shape
    return (
        w2.reshape(e, h_s // 128, 128, i // 64, 64).permute(0, 1, 3, 2, 4).contiguous()
    )


def unpack_w2_bf16(w2: torch.Tensor) -> torch.Tensor:
    e, hb, ks, rows, b = w2.shape
    return w2.permute(0, 1, 3, 2, 4).reshape(e, hb * rows, ks * b)
