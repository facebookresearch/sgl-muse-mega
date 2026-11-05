"""CPU checks of the DFlash draft head, the accept and the verify side (sm103/dflash/head.py,
sm103/dflash/verify_accept.py, kernels/dflash/{head,verify_accept}.cu, kernels/dflash/common.cuh), no GPU.

The head's shared pieces and the accept:
- the kernels' candidate-key order is the reference top-k's (NaN first, -0 == +0, ties to the smaller id);
- each algorithm, modeled step for step in torch / numpy, equals the reference on the same inputs: the per-tile list
  updates + the heads-threshold merge + the cross-rank merge (= dflash2.candidates), the compensated lattice
  (bitwise = the fp64 lattice), the accept and the draft feeds.

The draft head:
- the rescoring dot's order, modeled lane by lane, is ``warp_dot`` (bitwise) and within fp32 rounding of the fp64 dot;
- the activation quantizer keeps every value within half an e4m3 step and maps a zero row to zeros;
- the kernel's selection, modeled step for step (per-CTA e4m3 pre-score lists over the persistent block loop, the
  survivors at or above the ksurv-th list head, exact rescoring, the rank and global top-16): the survivors hold each
  rank's pre-score top-ksurv, and the candidates are the exact global top-16 whenever every global candidate survived;
- the walk over per-predecessor argmax tables is the reference walk (ties and NaNs included);
- the bind packs the kernel's parameter block (the cubin's layout), its e4m3 tensor maps, and its smem fits (the
  token-tree image with its tree arguments after the post phase's, the sampled image with the walk's).

The sampled step (kernels/dflash/{sample.cuh,kv_sync.cu}; on the device against SGLang's own chain sampling:
test/kernels/test_dflash_sample.py):
- the uniforms: Philox4x32-10's known answers, and sample.cuh's philox / uniform (built for the host) = the reference's;
- the reference's target rows (top-k, then top-p within it, as SGLang builds them), and that its chain sampling is
  lossless: chained steps over a Markov target emit that chain's transitions (block verification and token-wise);
- sample_accept's algorithm, modeled step for step (the per-row column slices and chunk masses, the residual's
  candidate terms, the CDF walk over chunks then over one chunk's elements), = the reference's decisions on the same
  uniforms; the sampling and statistics words are sample.cuh's.

verify_accept and the iteration's launch lists (dflash.stack): verify_accept's argmax (per tile, per CTA, per rank,
across ranks on the fp32 softcapped logit) modeled step for step against torch.argmax of the softcapped fp32 row, ties
and NaN included; its exchanged word and tag; its parameter block against the cubin (the tree image's with its table,
and tree_compact's), sample_accept's at 8 and 16 rows and draft_kv_sync's; the iteration is the draft then the verify,
which ends at verify_accept (sampled: the sampled head, then the lm_head, the softcap and sample_accept).

The verify side (the layer images with two bind values, verify.Window):
- b_live is a bind value: a layer's packed block differs between b_live 16 and b < 16 in exactly the three int fields
  the front item, the MoE phase and the combine item read, and nowhere else (the cubin and its parameter layout are
  untouched); out-of-range values are refused at bind.
- capture lists are validated (sm103/dflash/iteration.py capture_layers).
- the reference with b_live: the live rows of a layer are bitwise those of the full window, the pads' y is exactly 0,
  and large-finite pad inputs leave the live rows bitwise (TP8 emulated, reduced dims, both MoE variants, both
  attention types).
"""

from __future__ import annotations

import ctypes
import math
import re
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import test_param_blocks as T
import torch
from reference import dflash2 as R
from sgl_muse_mega.common import host
from sgl_muse_mega.common.jit import NVCC
from sgl_muse_mega.muse_spark.config import Config
from sgl_muse_mega.muse_spark.sm103 import KERNELS, artifacts
from sgl_muse_mega.muse_spark.sm103.dflash import head, iteration, verify_accept
from sgl_muse_mega.muse_spark.sm103.dflash.weights import DraftConfig
from sgl_muse_mega.muse_spark.sm103.verify import LayerLibrary
from synthetic import SAMPLING_CONFIGS, sampled_case
from test_reference import CASES, Model

BF = torch.bfloat16


def bf16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int16).view(BF)


def rand_logits(g: torch.Generator, shape, kind: str) -> torch.Tensor:
    """bf16 logits: 'normal', 'ties' (a handful of distinct values), 'equal' (one value), 'nan' (normal + NaNs),
    'ascending' (sorted by id: the adversarial order for a scan)."""
    if kind == "ties":
        return (torch.randint(0, 5, shape, generator=g).float() * 0.25).to(BF)
    if kind == "equal":
        return torch.full(shape, 1.5, dtype=BF)
    x = torch.randn(shape, generator=g) * 3
    if kind == "ascending":
        x = x.sort(-1).values
    if kind == "nan":
        x.view(-1)[torch.randperm(x.numel(), generator=g)[:7]] = float("nan")
    return x.to(BF)


# ---- the key order -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["normal", "ties", "equal", "nan"])
def test_key_order_is_reference_topk(kind):
    g = torch.Generator().manual_seed(1)
    vals = rand_logits(g, (32, 300), kind)
    specials = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), 1e-40, -1e-40], dtype=BF
    )
    vals[:, :6] = specials
    ids = torch.arange(300).expand_as(vals) * 3 + 7
    _, ref_ids = R.topk_low_id(vals.float(), ids, 16)
    assert torch.equal(R.key_ids(R.topk_keys(vals, ids)), ref_ids)


def test_key_bits_round_trip():
    bits = torch.arange(0, 1 << 16)
    o = R.ord16(bits)
    finite = (bits & 0x7FFF) <= 0x7F80
    order = torch.argsort(o[finite], stable=True)
    v = bf16_from_bits(bits[finite]).float()[order]
    # ord16 is monotone in the value (NaNs excluded, -0 == +0)
    assert torch.all(v[1:] >= v[:-1])
    assert int(o[bits == 0x8000]) == int(o[bits == 0])


# ---- the fused head: per-tile lists, the CTA merge, the rank merge -------------------------------------------------


def tile_update(lst: list[int], new: list[int]) -> list[int]:
    """The sink's per-row update: stage the list and the keys above its 16th, place each by its rank."""
    thr = lst[15]
    stage = lst + [k for k in new if k > thr]
    out = list(lst)
    for k in stage:
        r = sum(1 for j in stage if j > k)
        if r < 16:
            out[r] = k
    return out


def merge_lists(lists: list[list[int]]) -> list[int]:
    """merge_lists: only entries >= the 16th largest head reach the ranking (at most 256 of them)."""
    heads = [lst[0] for lst in lists]
    thr = sorted(heads, reverse=True)[15] if len(lists) >= 16 else 0
    stage = [k for lst in lists for k in lst if k >= thr]
    assert len(stage) <= 256
    return sorted(stage, reverse=True)[:16]


def fused_rank_lists(
    logits: torch.Tensor, rank: int, grid: int, blocks: int
) -> list[list[int]]:
    """One rank's top-16 per row as the draft head computes it: CTA c walks blocks c, c + grid, ..; tiles in order."""
    rows, n = logits.shape
    k = R.keys(R.bf16_bits(logits), torch.arange(n) + rank * n)
    out = []
    for t in range(rows):
        ctal = []
        for c in range(grid):
            lst = [0] * 16
            for b in range(c, blocks, grid):
                cols = range(b * 64, min(b * 64 + 64, n))
                lst = tile_update(lst, [int(k[t, j]) for j in cols])
            ctal.append(lst)
        out.append(merge_lists(ctal))
    return out


@pytest.mark.parametrize("kind", ["normal", "ties", "equal", "nan", "ascending"])
@pytest.mark.parametrize(
    "vocab_rank,grid,ranks", [(1000, 5, 2), (2512, 13, 4), (640, 3, 8)]
)
def test_fused_candidates_model(kind, vocab_rank, grid, ranks):
    g = torch.Generator().manual_seed(vocab_rank + grid)
    rows = 3
    full = rand_logits(g, (rows, vocab_rank * ranks), kind)
    blocks = (vocab_rank + 32 + 63) // 64
    per_rank = [
        fused_rank_lists(
            full[:, r * vocab_rank : (r + 1) * vocab_rank], r, grid, blocks
        )
        for r in range(ranks)
    ]
    for t in range(rows):
        got = sorted([k for r in range(ranks) for k in per_rank[r][t]], reverse=True)[
            :16
        ]
        ref_vals, ref_ids = R.topk_low_id(
            full[t : t + 1].float(), torch.arange(full.shape[1])[None], 16
        )
        assert [int(x) for x in R.key_ids(torch.tensor(got))] == ref_ids[0].tolist()


# ---- the lattice ---------------------------------------------------------------------------------------------------


def two_sum(s, c, x):
    """TwoSum on float32 arrays (numpy rounds every float32 op to nearest, as the kernel's __fadd_rn / __fsub_rn)."""
    t = s + x
    bp = t - s
    return t, c + ((s - (t - bp)) + (x - bp))


def lattice_model(pred_cb, succ_cb, hhat, cand, unary, anchor) -> torch.Tensor:
    """lattice_slot in numpy fp32: prod = bf16(A * hh) (exact fp32 product, one rounding), the 256-term dot in four
    compensated chains over r mod 4, combined, rounded fp32 -> bf16, plus the unary score. (A plain sequential fp32 dot
    missed the reference's bf16 in 6 of 92,160 scores on these inputs; compensated, none.)
    """
    gamma = cand.shape[0]
    out = np.zeros((gamma, 16, 16), dtype=np.float32)
    for e in range(gamma):
        pred = [anchor] * 16 if e == 0 else cand[e - 1].tolist()
        # [16 p, 256 r]
        prod = (pred_cb[pred].float() * hhat[e].float()[None]).to(BF).float().numpy()
        s_rows = succ_cb[cand[e]].float().numpy()  # [16 c, 256 r]
        terms = prod[:, None, :] * s_rows[None, :, :]  # exact: bf16 x bf16 fits fp32
        s = np.zeros((4, 16, 16), np.float32)
        er = np.zeros((4, 16, 16), np.float32)
        for r in range(0, 256, 4):
            s, er = two_sum(s, er, np.moveaxis(terms[:, :, r : r + 4], 2, 0))
        tot, err = s[0], (er[0] + er[1]) + (er[2] + er[3])
        for q in (1, 2, 3):
            tot, err = two_sum(tot, err, s[q])
        out[e] = tot + err
    return unary[:, None, :] + torch.from_numpy(out).to(BF).float()


@pytest.mark.parametrize("scale", [0.05, 1.0])
def test_lattice_compensated_is_reference(scale):
    g = torch.Generator().manual_seed(11)
    vocab, gamma = 2000, 15
    pred_cb = (torch.randn(vocab, 256, generator=g) * scale).to(BF)
    succ_cb = (torch.randn(vocab, 256, generator=g) * scale).to(BF)
    hhat = (torch.randn(gamma, 256, generator=g) * 2).to(BF)
    cand = torch.randint(0, vocab, (gamma, 16), generator=g)
    unary = torch.randn(gamma, 16, generator=g) * 5
    ref = R.lattice_hhat(pred_cb, succ_cb, hhat, cand, unary, 42)
    got = lattice_model(pred_cb, succ_cb, hhat, cand, unary, 42)
    assert torch.equal(got, ref)


def test_lattice_hhat_is_reference_lattice():
    g = torch.Generator().manual_seed(3)
    vocab, hidden, gamma = 300, 64, 4
    w = R.DraftWeights(
        layers=[],
        fc=None,
        hidden_norm=None,
        norm=None,
        pred_codebook=torch.randn(vocab, 256, generator=g).to(BF),
        succ_codebook=torch.randn(vocab, 256, generator=g).to(BF),
        hidden_proj=(torch.randn(256, hidden, generator=g) / 8).to(BF),
    )
    y = torch.randn(gamma, hidden, generator=g).to(BF)
    cand = torch.randint(0, vocab, (gamma, 16), generator=g)
    unary = torch.randn(gamma, 16, generator=g)
    hhat = R._mm(y, w.hidden_proj, BF)
    ref = R.lattice(w, y, cand, unary, 9, BF)
    assert torch.equal(
        R.lattice_hhat(w.pred_codebook, w.succ_codebook, hhat, cand, unary, 9), ref
    )


def test_verify_ids_and_walk():
    g = torch.Generator().manual_seed(5)
    for gamma in (7, 15):
        cand = torch.randint(0, 1000, (gamma, 16), generator=g)
        scores = torch.randn(gamma, 16, 16, generator=g)
        scores[0, 0, 3] = scores[
            0, 0
        ].max()  # an exact tie in slot 0: the first index wins
        drafts, _ = R.selector_walk(cand, scores)
        ids = R.verify_ids(77, drafts)
        b = gamma + 1
        assert (
            ids[0] == 77
            and torch.equal(ids[1:b], drafts)
            and torch.equal(ids, ids[torch.arange(16) % b])
        )


# ---- the accept ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("b_live", [8, 16])
def test_accept_ref_every_count(b_live):
    g = torch.Generator().manual_seed(100 + b_live)
    vocab = 4096
    for acc in range(b_live):
        x = torch.randn(b_live, vocab, generator=g)
        t = R.first_argmax(x)
        tokens = torch.randint(0, vocab, (b_live,), generator=g)
        tokens[1 : acc + 1] = t[:acc]
        if acc + 1 < b_live:
            tokens[acc + 1] = (t[acc] + 1) % vocab
        a = R.accept_ref(tokens, t, 1000)
        assert (a.accept, a.commit, a.bonus, a.pos) == (
            acc,
            acc + 1,
            int(t[acc]),
            1000 + acc + 1,
        )
        assert a.committed.tolist() == tokens[1 : acc + 1].tolist() + [int(t[acc])]
        f = R.accept_ref(tokens, t, 1000, forced=b_live - 1)
        assert f.accept == b_live - 1 and f.bonus == int(t[b_live - 1])


def test_draft_feeds_ref():
    cache = torch.arange(100 * 4, dtype=torch.float32).view(100, 4)
    f = head.draft_feeds(cache, pos=40, commit=3, b=8, dring=16)
    assert (
        f["dslot"].tolist()
        == [(40 + i) % 16 for i in range(8)]
        + [
            37 % 16,
            38 % 16,
            39 % 16,
        ]
        + [-1] * 5
    )
    assert torch.equal(f["dcos_sin"][8], cache[37]) and torch.equal(
        f["dcos_sin"][8 + 3], cache[40]
    )
    assert torch.equal(f["dcos_sin"][7], cache[47])
    assert f["dkv_len"].tolist() == [48]


@pytest.mark.parametrize("hidden", [256, 768])
def test_warp_dot_order(hidden):
    g = torch.Generator().manual_seed(hidden)
    w = (torch.randn(5, hidden, generator=g) * 0.05).to(BF)
    y = (torch.randn(hidden, generator=g) * 2).to(BF)
    got = R.warp_dot(w, y)
    wf, yf = w.float().numpy(), y.float().numpy()
    for r in range(w.shape[0]):
        lanes = np.zeros(32, dtype=np.float32)
        for lane in range(32):
            acc = np.float32(0)
            for j in range(lane, hidden // 8, 32):
                for u in range(8):
                    acc = np.float32(acc + np.float32(wf[r, 8 * j + u] * yf[8 * j + u]))
            lanes[lane] = acc
        for o in (16, 8, 4, 2, 1):
            lanes = (lanes + lanes[np.arange(32) ^ o]).astype(np.float32)
        assert np.float32(got[r]) == lanes[0]
        exact = float((w[r].double() * y.double()).sum())
        assert (
            abs(float(got[r]) - exact)
            <= 1e-5 * (w[r].double().abs() * y.double().abs()).sum()
        )


def test_quantize_y():
    g = torch.Generator().manual_seed(3)
    y = (torch.randn(4, 1024, generator=g) * 3).to(BF)
    y[2] = 0
    q = R.quantize_y(y).view(torch.float8_e4m3fn).float()
    a = y.float().abs().amax(1, keepdim=True)
    deq = q * torch.where(a > 0, a / 448.0, torch.ones_like(a))
    rel = (deq - y.float()).abs() / y.float().abs().clamp_min(1e-30)
    big = (
        y.float().abs() > a * 2**-6
    )  # e4m3 normals: 3 mantissa bits, half a step <= 2^-4 relative
    assert float(rel[big].max()) <= 2**-4 + 1e-6
    assert torch.all(q[2] == 0)


def score_keys(s: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """score_key: the fp32 order image << 18 | (2^18 - 1 - id)."""
    b = s.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    o = torch.where(b & 0x80000000 != 0, ~b & 0xFFFFFFFF, b | 0x80000000)
    return (o << head.ID_BITS) | ((1 << head.ID_BITS) - 1 - ids)


def cta_lists(keys: torch.Tensor, grid: int) -> list[list[int]]:
    """One x row: CTA c walks blocks c, c + grid, .. of 64 columns, keeping its top-16 (descending, 0 = empty)."""
    n = keys.numel()
    out = []
    for c in range(grid):
        cols = [
            j
            for b in range(c, (n + 63) // 64, grid)
            for j in range(b * 64, min(b * 64 + 64, n))
        ]
        top = sorted((int(keys[j]) for j in cols), reverse=True)[:16]
        out.append(top + [0] * (16 - len(top)))
    return out


def survivors(lists: list[list[int]], ksurv: int) -> list[int]:
    """The kernel's survivors: entries >= the ksurv-th largest list head, lists in CTA order, each list in order."""
    heads = [lst[0] for lst in lists]
    thr = sorted(heads, reverse=True)[ksurv - 1] if len(lists) >= ksurv else 0
    return [
        k
        for lst in lists
        if lst[0] >= thr and lst[0] != 0
        for k in lst
        if k >= thr and k != 0
    ]


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.5])
@pytest.mark.parametrize(
    "vocab_rank,grid,ranks,ksurv", [(640, 5, 4, 5), (1000, 7, 2, 7), (2048, 16, 8, 16)]
)
def test_prescore_model(noise, vocab_rank, grid, ranks, ksurv):
    g = torch.Generator().manual_seed(vocab_rank + grid)
    rows = 3
    logits = (torch.randn(rows, vocab_rank * ranks, generator=g) * 3).to(BF)
    prescore = logits.float() + torch.randn(logits.shape, generator=g) * noise
    ids = torch.arange(vocab_rank * ranks)
    exact = R.keys(R.bf16_bits(logits), ids.expand_as(logits))
    for t in range(rows):
        per_rank, surv_ids = [], set()
        for r in range(ranks):
            sl = slice(r * vocab_rank, (r + 1) * vocab_rank)
            sk = score_keys(prescore[t, sl], ids[sl])
            surv = survivors(cta_lists(sk, grid), ksurv)
            got = {int(k) for k in R.key_ids(torch.tensor(surv))}
            # the survivors hold the rank's pre-score top-ksurv
            assert set(R.key_ids(sk.topk(ksurv).values).tolist()) <= got
            surv_ids |= got
            ek = exact[t, sorted(got)]
            per_rank += ek.topk(min(16, ek.numel())).values.tolist()
        cand = R.key_ids(torch.tensor(sorted(per_rank, reverse=True)[:16]))
        ref = R.key_ids(exact[t].topk(16).values)
        if set(ref.tolist()) <= surv_ids:
            assert torch.equal(cand, ref)
        if noise == 0.0:
            assert (
                set(ref.tolist()) <= surv_ids
            )  # an exact pre-score always keeps the top-16


@pytest.mark.parametrize("kind", ["normal", "ties", "nan"])
def test_walk_tables(kind):
    g = torch.Generator().manual_seed(11)
    gamma = 15
    for _ in range(20):
        if kind == "ties":
            sc = torch.randint(0, 3, (gamma, 16, 16), generator=g).float()
        else:
            sc = torch.randn(gamma, 16, 16, generator=g)
        if kind == "nan":
            sc.view(-1)[torch.randperm(sc.numel(), generator=g)[:9]] = float("nan")
        cand = torch.randint(0, 200000, (gamma, 16), generator=g)
        ref, _ = R.selector_walk(cand, sc)
        pick = R.first_argmax(
            sc
        )  # [gamma, 16]: slot e's argmax successor per predecessor
        prev, toks = 0, []
        for e in range(gamma):
            prev = int(pick[e, prev])
            toks.append(int(cand[e, prev]))
        assert toks == ref.tolist()


# ---- the sampled step: the uniforms, the reference's sampling, sample_accept's algorithm ----------------------------

M32 = 0xFFFFFFFF


@pytest.mark.parametrize(
    "counter, key, want",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((M32,) * 4, (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
    ids=["zero", "ones", "pi"],
)
def test_philox_known_answers(counter, key, want):
    assert R.philox(counter, key) == want


def test_the_kernels_uniforms_are_the_references(tmp_path):
    """sample.cuh's philox and uniform, compiled for the host, against reference.dflash2.uniforms: bitwise."""
    seeds = [(0, 0, 0), (2**64 - 1, 131071, 7), (0x0123456789ABCDEF, 4096, 2**31 + 5)]
    calls = "".join(
        f"  {{ const int64_t s[4] = {{{(seed + 2**63) % 2**64 - 2**63}LL, {stream}, 0, 0}};\n"
        "    for (unsigned p = 0; p < 2; ++p) for (unsigned i = 0; i < 16; ++i)\n"
        f'      std::printf("%a\\n", mk::dflash::sample::uniform(s, {it}ull, p, i)); }}\n'
        for seed, stream, it in seeds
    )
    src, exe = tmp_path / "u.cu", tmp_path / "u"
    src.write_text(
        f'#include "dflash/sample.cuh"\n#include <cstdio>\nint main() {{\n{calls}}}\n'
    )
    arch = "-gencode=arch=compute_103a,code=sm_103a"
    subprocess.run(
        [NVCC, arch, "-std=c++20", "-DMK_WORLD=tp8", f"-I{KERNELS}", "-o", exe, src],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    got = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    want = [
        float(x)
        for seed, stream, it in seeds
        for p in (R.PURPOSE_WALK, R.PURPOSE_ACCEPT)
        for x in R.uniforms(seed, stream, it, p)
    ]
    assert [float.fromhex(x) for x in got.split()] == want


def test_uniforms_are_uniform_and_apart():
    u = torch.stack([R.uniforms(5, 4096, it, p) for it in range(400) for p in (0, 1)])
    assert 0.0 <= float(u.min()) and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.01
    # another stream (a restarted run) or purpose draws other numbers
    assert not torch.equal(R.uniforms(5, 4096, 3, 0), R.uniforms(5, 4097, 3, 0))
    assert not torch.equal(R.uniforms(5, 4096, 3, 0), R.uniforms(5, 4096, 3, 1))


def test_top_p_keeps_the_values_whose_mass_above_is_below_top_p():
    """flashinfer's top_p_renorm_probs example (top_p 0.3)."""
    p = torch.tensor(
        [
            [0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255],
        ]
    )
    got = R.target_probs(p.log(), 1.0, 5, 0.3)
    for r, k in enumerate([[1, 3], [3], [0, 2], [1]]):
        want = torch.zeros(5, dtype=torch.float64)
        want[k] = p[r, k].double() / p[r, k].double().sum()
        torch.testing.assert_close(got[r], want, rtol=1e-6, atol=1e-6)


def test_top_k_keeps_boundary_ties_then_top_p_within():
    logits = torch.tensor([[3.0, 1.0, 2.0, 2.0, 0.0, -1.0]])
    p = R.target_probs(logits, 1.0, 3, 1.0)
    assert (p[0] > 0).tolist() == [True, False, True, True, False, False]
    e = torch.tensor([3.0, 2.0, 2.0], dtype=torch.float64).exp()
    # top-p 0.6 within the top-3: mass above the 2.0s is e^3 / (e^3 + 2 e^2) = 0.576 < 0.6: all three stay
    torch.testing.assert_close(
        R.target_probs(logits, 1.0, 3, 0.6)[0, [0, 2, 3]], e / e.sum()
    )
    # top-p 0.5: only the 3.0 (0.576 >= 0.5 above the 2.0s)
    assert (R.target_probs(logits, 1.0, 3, 0.5)[0] > 0).tolist() == [True] + [False] * 5
    # temperature scales the logits before the softmax
    t = R.target_probs(logits, 0.5, 6, 1.0)[0]
    torch.testing.assert_close(t, torch.softmax(logits[0].double() / 0.5, 0))


@pytest.mark.parametrize("block", [True, False], ids=["block", "token"])
def test_chain_sampling_is_lossless(block):
    """Chained steps over a Markov target (row j of a verify = M[the window's token j]) with a draft that proposes from
    its own lattice-like q (16 candidates per slot, q depending on the previous draw): the emitted tokens' transitions
    match M (chi-square per frequent row)."""
    v, b, k, steps = 20, 4, 16, 20000
    g = torch.Generator().manual_seed(11)
    m = torch.softmax(torch.randn(v, v, generator=g, dtype=torch.float64), 1)
    cand = torch.stack([torch.randperm(v, generator=g)[:k] for _ in range(b - 1)])
    # the draft's q per slot and previous draw: [slot, prev, c]
    qlat = torch.softmax(torch.randn(b - 1, k, k, generator=g) * 2.0, -1)
    counts = torch.zeros(v, v, dtype=torch.float64)
    anchor = 0
    for _ in range(steps):
        u = torch.rand(16, generator=g)
        walk = torch.rand(b - 1, generator=g)
        toks, qrows, prev = [anchor], [], 0
        for e in range(b - 1):
            qe = qlat[e, prev]
            prev = min(int((walk[e] >= torch.cumsum(qe, 0)).sum()), k - 1)
            toks.append(int(cand[e, prev]))
            qrows.append(qe)
        tokens = torch.tensor(toks)
        a = R.sampled_accept(tokens, m[tokens], cand, torch.stack(qrows), u, block)
        for x in toks[1 : 1 + a.accept_len] + [a.bonus]:
            counts[anchor, x] += 1
            anchor = x
    tested = 0
    for row in range(v):
        n = counts[row].sum()
        exp = m[row] * n
        cells = exp >= 5
        if n < 1000 or int(cells.sum()) < 5:
            continue
        chi = float(((counts[row][cells] - exp[cells]) ** 2 / exp[cells]).sum())
        df = int(cells.sum()) - 1
        # the chi-square quantile at p = 1e-4 (Wilson-Hilferty)
        crit = df * (1 - 2 / (9 * df) + 3.719 * math.sqrt(2 / (9 * df))) ** 3
        assert chi < crit, (row, chi, crit, int(n))
        tested += 1
    assert tested >= 5, tested


CTAS, CHUNK = 8, 1024  # kernels/dflash/sample.cuh kCtas, kChunk


def sample_accept_model(logits, temperature, top_k, top_p, cand, q, path, u, block):
    """kernels/dflash/sample.cuh rows + decide in numpy fp32: each row in kCtas column slices of kChunk chunks, the
    keep threshold, the normalizer and the chunk masses; then the decision from the candidates' probabilities and the
    chunk masses, the residual's chunk masses with the candidates' terms, the CDF over chunks, then over one chunk.
    """
    rows, vocab = logits.shape
    gamma = rows - 1
    sl = ((vocab + CTAS - 1) // CTAS + 3) // 4 * 4
    cpc = (sl + CHUNK - 1) // CHUNK

    def span(k):  # chunk k's columns: CTA k // cpc's slice, its chunk k % cpc
        j, cc = divmod(k, cpc)
        return j * sl + cc * CHUNK, min(j * sl + min((cc + 1) * CHUNK, sl), vocab)

    y = (logits.float() / torch.tensor(temperature, dtype=torch.float32)).numpy()
    stats, pc = [], np.zeros((rows, 16), np.float32)
    chunk = np.zeros((rows, CTAS * cpc), np.float32)
    for r in range(rows):
        yr = y[r].astype(np.float64)
        m = yr.max()
        keep = np.ones(vocab, bool)
        if top_k < vocab:
            keep &= yr >= np.sort(yr)[-top_k]
        if top_p < 1.0:
            w = np.where(keep, np.floor(np.exp(yr - m) * 2.0**40), 0)
            order = np.argsort(-yr, kind="stable")
            cum = np.cumsum(w[order])
            keep &= yr >= yr[order[np.searchsorted(cum, math.ceil(top_p * w.sum()))]]
        z = np.exp(yr[keep] - m).sum()
        p = np.where(keep, np.exp(yr - m) / z, 0.0).astype(np.float32)
        stats.append(p)
        for k in range(CTAS * cpc):
            lo, hi = span(k)
            chunk[r, k] = p[lo:hi].sum()
        if r < gamma:
            pc[r] = p[cand[r].numpy()]
    tot = chunk.sum(1)
    qn = q.numpy()
    pv, qv = pc[np.arange(gamma), path.numpy()], qn[np.arange(gamma), path.numpy()]

    def residual(r, s):
        return (
            s * max(tot[r] - pc[r].sum(), 0.0)
            + np.maximum(s * pc[r] - qn[r], 0.0).sum()
        )

    pps, pp = [1.0], 1.0
    for e in range(gamma):
        pp = min(pp * pv[e] / qv[e], 1.0) if 0.0 < qv[e] <= 1.0 else 0.0
        pps.append(pp)
    acc, scale, norm = 0, 1.0, -1.0
    if block:
        for e in range(gamma, 0, -1):
            coin = float(u[e - 1])
            if coin < pps[e]:
                ap, rm = pps[e], -1.0
                if e < gamma and 0.0 < pps[e] < 1.0:
                    rm = residual(e, pps[e])
                    ap = rm / (rm + 1.0 - pps[e])
                if coin < ap:
                    acc, scale, norm = e, pps[e], rm
                    break
    else:
        while (
            acc < gamma and 0.0 < qv[acc] <= 1.0 and float(u[acc]) * qv[acc] < pv[acc]
        ):
            acc += 1
    full = acc == gamma
    if norm < 0.0:
        norm = tot[acc] if full else residual(acc, scale)
    target = float(u[R.FINAL_UNIFORM]) * norm
    mk = chunk[acc].astype(np.float64)
    vals = stats[acc].astype(np.float64)
    if not full:
        vals = scale * vals
        pin, rin = np.zeros_like(mk), np.zeros_like(mk)
        for c, idx in enumerate(cand[acc].numpy()):
            k = (idx // sl) * cpc + (idx % sl) // CHUNK
            vals[idx] = max(scale * stats[acc][idx] - qn[acc, c], 0.0)
            pin[k] += stats[acc][idx]
            rin[k] += vals[idx]
        mk = scale * np.maximum(mk - pin, 0.0) + rin
    incl = np.cumsum(mk)
    hit = np.nonzero(incl > target)[0]
    if not hit.size:
        return acc, vocab - 1
    k = int(hit[0])
    lo, hi = span(k)
    live = vals[lo:hi] > 0
    pos = np.nonzero((incl[k] - mk[k] + np.cumsum(vals[lo:hi]) > target) & live)[0]
    return acc, lo + int(pos[0] if pos.size else np.nonzero(live)[0][-1])


@pytest.mark.parametrize(
    "cfg", list(SAMPLING_CONFIGS.values()), ids=list(SAMPLING_CONFIGS)
)
@pytest.mark.parametrize("block", [True, False], ids=["block", "token"])
def test_the_accepts_algorithm_is_the_references(cfg, block):
    """Random rows over a 20,000-id vocabulary (3 chunks per slice, the last partial) with the drafts' candidates
    favored, random q and draws: the modeled kernel's accept and bonus = the reference's, but at a boundary within the
    rounding of fp32 against fp64 (at most 2 of the cases)."""
    g = torch.Generator().manual_seed(3)
    off = 0
    for case in range(120):
        cand, q, path, tokens, logits = sampled_case(g, 16 if case % 2 else 8, 20000, 7)
        u = torch.rand(16, generator=g)
        want = R.sampled_accept(tokens, R.target_probs(logits, *cfg), cand, q, u, block)
        got = sample_accept_model(logits, *cfg, cand, q, path, u, block)
        off += got != (want.accept_len, want.bonus)
    assert off <= 2, off


def test_the_sampling_and_statistics_words_are_the_kernels():
    src = (KERNELS / "dflash/sample.cuh").read_text()
    k = {n: int(v) for n, v in re.findall(r"(k\w+) = (\d+)[,;]", src)}
    names = dict(
        SEED="kSampSeed",
        STREAM="kSampStream",
        TOP_K="kSampTopK",
        TEMP_TOP_P="kSampTemp",
    )
    assert {w: k[n] for w, n in names.items()} == head.SAMP
    assert (k["kSampWords"], k["kStatWords"]) == (head.N_SAMP, verify_accept.STATS)


# ---- the bind: parameter block against the cubin --------------------------------------------------------------------


@pytest.fixture(scope="module")
def built():
    import test_param_blocks as T

    T.shim()
    return T, artifacts.build_set(artifacts.dflash_artifacts(tree=True))


HEAD_TREE_ONLY = {"tree", "fan_idx", "fan_lp", "cos_sin", "cos_sin_cache"}
HEAD_SAMPLE_ONLY = {"samp", "walk_q", "walk_path"}


# the token-tree head image (its tree arguments after the post phase's) and the sampled one (the walk's)
@pytest.mark.parametrize(
    "gamma,ksurv,image",
    [(7, 32, "greedy"), (15, 64, "greedy"), (15, 32, "tree"), (15, 32, "sample")],
)
def test_draft_head_block(built, gamma, ksurv, image):
    T, paths = built
    tree = image == "tree"
    lib = head.DraftHeadLib(paths["libmk_dflash_draft_head"])
    s = head.HeadShape(
        hidden=8192,
        vocab_rank=25256,
        rank=3,
        ranks=8,
        gamma=gamma,
        sms=148,
        mult=0.17677,
        cap=20.0,
    )
    ints = dict(
        hidden=s.hidden,
        vocab_rank=s.vocab_rank,
        hproj=s.hproj,
        n_pad=s.n_pad,
        grid=s.grid,
        rank=s.rank,
        ranks=s.ranks,
        gamma=gamma,
        ksurv=ksurv,
        mult=s.mult,
        cap=s.cap,
        cos_sin_rows=131088 if tree else 0,
        d=64 if tree else 0,
    )
    zero = {"tree": HEAD_SAMPLE_ONLY, "sample": HEAD_TREE_ONLY}.get(
        image, HEAD_SAMPLE_ONLY | HEAD_TREE_ONLY
    )
    v = T.fill("MkDflashDraftHeadBind", T.Fakes(), ints, zero)
    sh = T.shim()
    sh.shim_reset()
    launch = lib.bind(
        lib.dll.mk_dflash_draft_head_bind,
        None,
        host.make("MkDflashDraftHeadBind", **v),
    )
    prefix = "_ZN2mk6dflash4head10draft_head"
    cubin = paths["dflash_draft_head" + ("" if image == "greedy" else f"_{image}")]
    ptrs = [f for f in v if f not in ints and f not in zero]
    blob, slots, enc = T.check_block(launch, cubin, prefix, v, ptrs)
    T.expect_map(
        T.map_at(blob, slots, enc, 1),
        v["wq"],
        T.U8,
        (8192, 25256),
        (128, 64),
        (8192,),
        T.SW128,
    )
    T.expect_map(
        T.map_at(blob, slots, enc, 2),
        v["xq"],
        T.U8,
        (8192, gamma),
        (128, 16),
        (8192,),
        T.SW128,
    )
    i = launch.info()
    assert (i.grid, i.block, s.n_pad) == (132, 256, 25344)
    assert i.smem + T.static_smem(cubin, prefix) <= T.OPTIN_SMEM
    assert int(lib.dll.mk_dflash_draft_head_state_words()) == head.STATE_WORDS


ID_BITS, TAG_BITS = (
    18,
    14,
)  # kernels/dflash/common.cuh kIdBits, dflash/verify_accept.cu kAcceptTagBits


def softcap(v: torch.Tensor, scale: float, cap: float) -> torch.Tensor:
    """head.cu's expression in fp32: cap x tanh(v x scale / cap), op by op."""
    v = v.float()
    return torch.tanh((v * scale) / cap) * cap


def beats(av, ai, bv, bi) -> bool:
    """torch.argmax's order: NaN first, then the larger value, then the lower id."""
    an, bn = av != av, bv != bv
    if an != bn:
        return an
    if not an and av != bv:
        return av > bv
    return ai < bi


def take(best, cand):
    return cand if beats(cand[0], cand[1], best[0], best[1]) else best


def accept_model(
    rows: torch.Tensor, ranks: int, grid: int, scale: float, cap: float
) -> list[int]:
    """verify_accept's reduction tree over one fp32 row [ranks x shard] (kernels/dflash/verify_accept.cu): per 64-column
    tile the 8 packets' bests (columns 4p .. 4p + 3 and 32 + 4p .. 32 + 4p + 3), the CTA's running best (blocks cta,
    cta + grid, ..), the last CTA over the CTAs, then the rank merge on the exchanged softcapped value; returns each
    row's argmax.
    """
    n_rows, vocab = rows.shape
    shard = vocab // ranks
    ident = (float("-inf"), 2**31 - 1)
    out = []
    for r in range(n_rows):
        best = ident
        for q in range(ranks):
            cta_best = [ident] * grid
            for blk in range((shard + 63) // 64):
                for pk in range(8):
                    pbest = ident
                    for c in (blk * 64 + pk * 4, blk * 64 + 32 + pk * 4):
                        for col in range(c, c + 4):
                            if col < shard:
                                v = float(softcap(rows[r, q * shard + col], scale, cap))
                                pbest = take(pbest, (v, q * shard + col))
                    cta_best[blk % grid] = take(cta_best[blk % grid], pbest)
            rank_best = ident
            for cb in cta_best:
                rank_best = take(rank_best, cb)
            _, value, i = unword(word(*rank_best, tag=5))
            best = take(best, (value, i))
        out.append(best[1])
    return out


def word(value: float, i: int, tag: int) -> int:
    """accept_word: {fp32 bits << 32 | id << 14 | tag}."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return bits << 32 | i << TAG_BITS | tag


def unword(w: int) -> tuple[int, float, int]:
    """(tag, value, id) of an exchanged word, as the merge reads it."""
    value = struct.unpack("<f", struct.pack("<I", w >> 32))[0]
    return w & ((1 << TAG_BITS) - 1), value, (w & 0xFFFFFFFF) >> TAG_BITS


def accept_tag(epoch: int) -> int:
    return epoch % ((1 << TAG_BITS) - 1) + 1


@pytest.mark.parametrize("kind", ["normal", "saturated", "ties", "nan", "bf16_ties"])
def test_accept_argmax_model_is_torch_argmax(kind):
    g = torch.Generator().manual_seed(7)
    ranks, shard, grid, scale, cap = (
        4,
        200,
        3,
        1.0,
        20.0,
    )  # a 4-block shard, a partial last tile, 3 CTAs
    x = torch.randn(4, ranks * shard, generator=g) * 3
    if (
        kind == "saturated"
    ):  # distinct fp32 logits that softcap to one fp32 value: the lower id must win
        x[:, ::37] = 400.0
        x[:, 5::91] = 600.0
    elif kind == "ties":
        x = torch.randint(0, 3, x.shape, generator=g).float()
    elif kind == "nan":
        x[1, 333] = float("nan")
        x[2, 17] = float("nan")
        x[2, 701] = float("nan")
    elif kind == "bf16_ties":
        # two fp32 logits of one bf16 value (6.0): the larger one (the higher id) must win, not the lower id
        x = 5.75 - torch.rand(x.shape, generator=g) * 0.25
        x[:, 7] = 6.0 - 2**-12
        x[:, 650] = 6.0 - 2**-14
        assert torch.equal(x.to(BF)[:, 7], x.to(BF)[:, 650])
    want = torch.argmax(softcap(x, scale, cap), dim=1).tolist()
    assert accept_model(x, ranks, grid, scale, cap) == want
    if kind == "bf16_ties":
        assert want == [650] * 4


def test_accept_word_and_tag():
    for value, i in (
        (-7.25, 0),
        (19.75, 2**ID_BITS - 1),
        (float("-inf"), 202047),
        (0.0, 12345),
    ):
        tag = accept_tag(123456)
        assert unword(word(value, i, tag)) == (tag, value, i)
    # a slot read by launch e holds 0, launch e - 2's word or e's (uint32 epochs)
    for e in [*range(0, 40000), *range(2**32 - 40000, 2**32)]:
        t = accept_tag(e)
        assert 1 <= t < 2**TAG_BITS and t != accept_tag((e - 2) % 2**32)


# tree: the token-tree verify_accept image (16 live rows), its table after the post phase's arguments
@pytest.mark.parametrize("b_live,tree", [(8, False), (16, False), (16, True)])
def test_verify_accept_block(built, b_live, tree):
    T, paths = built
    lib = host.Library(paths["libmk_dflash_verify_accept"])
    ints = dict(
        hidden=8192,
        n_out=25256,
        n_pad=25344,
        grid=132,
        rank=5,
        ranks=8,
        n_override=4,
        ring=4096,
        cos_sin_rows=131088,
        d=64,
        dcos_sin_rows=131088,
        dd=128,
        dring=4096,
        b_live=b_live,
        scale=1.0,
        cap=20.0,
    )
    zero = set() if tree else {"tree"}
    v = T.fill("MkDflashVerifyAcceptBind", T.Fakes(), ints, zero)
    sh = T.shim()
    sh.shim_reset()
    launch = lib.bind(
        lib.dll.mk_dflash_verify_accept_bind,
        None,
        host.make("MkDflashVerifyAcceptBind", **v),
    )
    prefix = "_ZN2mk6dflash13verify_accept"
    cubin = paths["dflash_verify_accept_tree" if tree else "dflash_verify_accept"]
    ptrs = [f for f in v if f not in ints and f not in zero]
    blob, slots, enc = T.check_block(launch, cubin, prefix, v, ptrs)
    # the lm_head's operands: W [n_out][hidden] and the 16 normed rows, 64 x 64 / 64 x 16 boxes
    T.expect_map(
        T.map_at(blob, slots, enc, 1),
        v["w"],
        T.BF16,
        (8192, 25256),
        (64, 64),
        (8192 * 2,),
        T.SW128,
    )
    T.expect_map(
        T.map_at(blob, slots, enc, 2),
        v["x"],
        T.BF16,
        (8192, 16),
        (64, 16),
        (8192 * 2,),
        T.SW128,
    )
    i = launch.info()
    assert (i.grid, i.block) == (132, 256)
    assert i.smem + T.static_smem(cubin, prefix) <= T.OPTIN_SMEM


@pytest.mark.parametrize("b_live", [8, 16])
def test_sample_accept_block(built, b_live):
    T, paths = built
    lib = host.Library(paths["libmk_dflash_verify_accept"])
    ints = dict(
        vocab=202048,
        block_verification=1,
        ring=4096,
        cos_sin_rows=131088,
        d=64,
        dcos_sin_rows=131088,
        dd=128,
        dring=4096,
        b_live=b_live,
    )
    v = T.fill("MkDflashSampleAcceptBind", T.Fakes(), ints, set())
    launch = lib.bind(
        lib.dll.mk_dflash_sample_accept_bind,
        None,
        host.make("MkDflashSampleAcceptBind", **v),
    )
    prefix = "_ZN2mk6dflash13sample_accept"
    cubin = paths["dflash_verify_accept_sample"]
    T.check_block(launch, cubin, prefix, v, [f for f in v if f not in ints])
    i = launch.info()
    assert (i.grid, i.block, i.cluster) == (8 * b_live, 256, 8)
    assert i.smem >= 202048 // 8 * 4
    assert i.smem + T.static_smem(cubin, prefix) <= T.OPTIN_SMEM
    lib.dll.mk_dflash_sample_chunks.restype = ctypes.c_int64
    assert int(lib.dll.mk_dflash_sample_chunks(ctypes.c_int64(202048))) == 8 * 25


# (ints, cubin, symbol prefix, grid); tree_compact at TP8: 45 + 1 sliding and 15 + 1 global layers' K and V pools, 5
# capture buffers of 8192 fp32
AUX = {
    "tree_compact": (
        dict(n_swa=92, n_full=32, n_cap=5, kv_row_bytes=256, cap_row_bytes=32768),
        "dflash_verify_accept_tree",
        "_ZN2mk6dflash4tree12tree_compactI",
        92 + 32 + 5 * 4,
    ),
    "kv_sync": (
        dict(n_buf=10, row_bytes=256, ring_rows=4096, lag=1),
        "dflash_kv_sync",
        "_ZN2mk6dflash13draft_kv_sync",
        1,
    ),
}


@pytest.mark.parametrize("name", AUX)
def test_aux_block(built, name):
    T, paths = built
    ints, cubin, prefix, grid = AUX[name]
    lib = host.Library(paths["libmk_dflash_verify_accept"])
    struct_name = "MkDflash" + name.title().replace("_", "") + "Bind"
    v = T.fill(struct_name, T.Fakes(), ints, set())
    T.shim().shim_reset()
    bind = getattr(lib.dll, f"mk_dflash_{name}_bind")
    launch = lib.bind(bind, None, host.make(struct_name, **v))
    T.check_block(launch, paths[cubin], prefix, v, [f for f in v if f not in ints])
    i = launch.info()
    assert (i.grid, i.block, i.cluster) == (grid, 256, 1)


# sample: the sampled head, then the verify ending at the lm_head, the softcap and sample_accept
@pytest.mark.parametrize(
    "padded, sample", [(False, False), (True, False), (False, True)]
)
def test_iteration_launch_lists(padded, sample):
    padded_verify = [
        "entry_f",
        "qkv_head",
        *[f"V{i}" for i in range(62)],
        "verify_accept",
    ]
    it = SimpleNamespace(
        sample=True,
        draft=["entry", "qkv0", "L0", "L1", "L2", "L3", "tail", "head"],
        draft_sample=["entry", "qkv0", "L0", "L1", "L2", "L3", "tail", "head_sample"],
        verify=["entry8", "qkv_head8", *[f"N{i}" for i in range(62)], "verify_accept"],
        verify_padded=lambda: padded_verify,
        verify_sample=[
            "entry16",
            "qkv_head",
            *[f"N{i}" for i in range(62)],
            "lm_head",
            "softcap",
            "sample_accept",
        ],
    )
    draft = it.draft_sample if sample else it.draft
    verify = it.verify_sample if sample else padded_verify if padded else it.verify
    got = [
        iteration.Iteration.launches(it, part, padded, sample)
        for part in ("iter", "draft", "verify")
    ]
    assert got == [draft + verify, draft, verify]
    assert len(iteration.Iteration.launches(it)) == 73


ROWS = Config().rows
# the pads' poison: large but finite in bf16 (a NaN pad V row would reach the live rows through the masked P x V MMA)
POISON = 1.0e4


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise (tells -0 from +0, keeps NaN payloads)."""
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def _layer_values(c, cfg) -> dict:
    zero = T.layer_zero(c)
    ints = dict(
        rank=3,
        plane_stride=1572864,
        num_local=cfg.local_experts,
        zb_par_slots=c.moe_slots,
        kv_rows=4096,
        page_stride=131088,
        nk_page_stride=131088,
        nk_splits=c.nextkv_splits,
        nk_window=cfg.swa_window,
        b_live=c.rows,
        prefetch_experts=int(not c.bf16),
        router_mult=1.0,
        route_scale=1.0,
        eps=1e-5,
        q_scale=cfg.q_scale,
        qkv_eps=1e-5,
    )
    return T.fill("MkLayerBind", T.Fakes(), ints, zero)


def _pack(lib, h, values: dict):
    T.shim().shim_reset()
    return lib.bind(
        lib.dll.mk_layer_bind, None, h.make("MkLayerBind", **values)
    ).params()


@pytest.mark.parametrize("image", artifacts.LAYER_IMAGES)
@pytest.mark.parametrize("b_live", [1, 8, 15])
def test_b_live_is_three_bind_fields(image, b_live):
    cfg, h = Config(), T.host()
    lib = LayerLibrary(T.paths()[f"libmk_layer_{image}"])
    c = lib.constants
    v = _layer_values(c, cfg)
    full, slots = _pack(lib, h, v)
    part, slots_b = _pack(lib, h, {**v, "b_live": b_live})
    assert slots_b == slots and len(part) == len(full)
    # bytes no parameter field covers carry host stack / heap residue: the gaps between parameters and the tail
    # padding of the last one (NextKvPrefetch: 4 pointers + 3 u32)
    nk_off, nk_size = slots[-1]
    fields = {i for o, n in slots[:-1] for i in range(o, o + n)} | set(
        range(nk_off, nk_off + 4 * 8 + 3 * 4)
    )
    assert nk_size == 48
    words = sorted({i // 4 * 4 for i in fields if full[i] != part[i]})
    got = [int.from_bytes(part[w : w + 4], "little") for w in words]
    was = [int.from_bytes(full[w : w + 4], "little") for w in words]
    owner = [next(i for i, (o, n) in enumerate(slots) if o <= w < o + n) for w in words]
    # the front item's args, the MoE phase's LayerArgs, the combine item's args (host/runtime.cu pack order; the
    # expert prefetch's parameter sits before the next-layer phase's)
    assert owner == [7, 10, 21], (image, owner)
    assert got == [b_live] * 3 and was == [c.rows] * 3, (image, got, was)


@pytest.mark.parametrize("b_live", [0, ROWS + 1])
def test_b_live_refused(b_live):
    cfg, h = Config(), T.host()
    lib = LayerLibrary(T.paths()["libmk_layer_swa"])
    v = _layer_values(lib.constants, cfg)
    with pytest.raises(RuntimeError, match="live rows"):
        _pack(lib, h, {**v, "b_live": b_live})


def test_capture_layers():
    cfg = Config()
    assert iteration.capture_layers(cfg, DraftConfig().capture) == (1, 16, 30, 44, 59)
    assert iteration.capture_layers(cfg, [0, cfg.layers - 2]) == (0, cfg.layers - 2)
    for bad in ([16, 1], [1, 1], [cfg.layers - 1], [-1]):
        with pytest.raises(AssertionError):
            iteration.capture_layers(cfg, bad)


# ---- the reference with b_live (reduced dims, 8 emulated ranks) ----

REF_VALUES = (
    "x_attn",
    "branch",
    "stream_mid",
    "h3",
    "pooled",
    "logits",
    "ids",
    "w",
    "zb",
    "y",
    "ffn_out",
    "stream_out",
    "h3_next",
    "qkv_next",
)


def _poisoned(model: Model, b_live: int) -> Model:
    """The same layer with its pad rows' inputs (residual and qkv row) replaced by +-POISON."""
    gen = torch.Generator().manual_seed(4)
    pads = ROWS - b_live

    def sign(n):
        return torch.randint(0, 2, (pads, n), generator=gen).float() * 2 - 1

    model.stream = model.stream.clone()
    model.stream[b_live:] = sign(model.stream.shape[1]) * POISON
    # qkv_in parts are stored transposed: [cols, rows]
    model.qkv_in = [p.clone() for p in model.qkv_in]
    for p in model.qkv_in:
        p[:, b_live:] = (sign(p.shape[0]).T * POISON).bfloat16()
    return model


@pytest.mark.parametrize(
    "moe,window,nxt", CASES, ids=["fp4-swa", "fp4-global", "bf16-swa", "bf16-tail"]
)
@pytest.mark.parametrize("b_live", [1, 8, 15])
def test_reference_b_live(moe, window, nxt, b_live):
    full = Model(moe, window, nxt, seed=21).run(8, 4)
    part = Model(moe, window, nxt, seed=21).run(8, 4, b_live=b_live)
    pois = _poisoned(Model(moe, window, nxt, seed=21), b_live).run(8, 4, b_live=b_live)
    for r in range(8):
        for name in REF_VALUES:
            if name not in full[r]:
                continue
            assert same_bits(part[r][name][:b_live], full[r][name][:b_live]), (
                name,
                r,
            )
            assert same_bits(pois[r][name][:b_live], part[r][name][:b_live]), (
                "poison",
                name,
                r,
            )
        assert not part[r]["y"][b_live:].any() and not part[r]["zb"][b_live:].any(), r
        assert full[r]["y"][b_live:].any(), r  # the mask is what zeroes the pads
        assert torch.isfinite(pois[r]["stream_out"].float()).all(), r
