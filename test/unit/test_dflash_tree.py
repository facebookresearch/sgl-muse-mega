"""CPU checks of the token-tree window's reference (reference.dflash2 token_tree / tree_accept) and of its constants
against kernels/dflash/tree.cuh and the attention CTAs' table words (kernels/attention.cuh), no GPU.
"""

import random
import re

import pytest
import torch
from reference import dflash2 as R
from sgl_muse_mega.muse_spark.sm103 import KERNELS
from sgl_muse_mega.muse_spark.sm103.buffers import TREE_WORDS
from sgl_muse_mega.muse_spark.sm103.dflash.head import TREE_FAN

GAMMA, ROWS = 15, R.TREE_ROWS


def lattice(seed: int, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    cand = torch.randperm(1000, generator=g)[: GAMMA * 16].view(GAMMA, 16) + 5000
    return cand, torch.randn(GAMMA, 16, 16, generator=g) * temp


def candidate_index(cand: torch.Tensor, tree: R.TokenTree) -> list[int]:
    """Each row's candidate index in its slot (the anchor: 0, slot 0's predecessor row)."""
    return [0] + [
        cand[tree.depth[r] - 1].tolist().index(tree.tokens[r]) for r in range(1, ROWS)
    ]


def path_lp(cand, scores, tree: R.TokenTree, r: int) -> float:
    """A row's path log-probability, recomputed from its ancestors."""
    chain = []
    while r:
        chain.append(r)
        r = tree.parent[r]
    idx, lp, p = candidate_index(cand, tree), 0.0, 0
    for d, row in enumerate(reversed(chain)):
        order = torch.argsort(scores[d, p], descending=True, stable=True)[: R.TREE_FAN]
        k = order.tolist().index(idx[row])
        step = R.tree_fan_lp(scores[d, p][order].tolist())[k]
        lp = R._f32(lp + step) if d else step
        p = idx[row]
    return lp


@pytest.mark.parametrize(
    "seed,temp", [(s, t) for s in range(4) for t in (0.3, 1.0, 3.0)]
)
def test_the_tree_is_a_preorder_tree(seed, temp):
    cand, scores = lattice(seed, temp)
    tr = R.token_tree(7, cand, scores)
    assert (
        tr.parent[0] == -1
        and tr.depth[0] == 0
        and tr.mask[0] == 1
        and tr.tokens[0] == 7
    )
    for r in range(1, ROWS):
        p = tr.parent[r]
        assert 0 <= p < r and tr.depth[r] == tr.depth[p] + 1 <= GAMMA
        assert tr.mask[r] == tr.mask[p] | (1 << r)
        assert tr.tokens[r] in cand[tr.depth[r] - 1].tolist()
    idx = candidate_index(cand, tr)
    for p in range(ROWS):
        kids = [r for r in range(1, ROWS) if tr.parent[r] == p]
        # siblings carry distinct tokens, ascend by rank, and each precedes the next sibling's subtree
        assert len({tr.tokens[r] for r in kids}) == len(kids)
        if kids:
            order = torch.argsort(
                scores[tr.depth[kids[0]] - 1, idx[p]], descending=True, stable=True
            ).tolist()
            ranks = [order.index(idx[r]) for r in kids]
            assert ranks == sorted(ranks)
            subtree = [[c for c in range(ROWS) if (tr.mask[c] >> r) & 1] for r in kids]
            assert all(kids[j + 1] > max(subtree[j]) for j in range(len(kids) - 1))


@pytest.mark.parametrize("seed", range(4))
def test_the_tree_takes_the_most_probable_nodes(seed):
    cand, scores = lattice(seed, 1.0)
    tr = R.token_tree(7, cand, scores)
    worst = min(path_lp(cand, scores, tr, r) for r in range(1, ROWS))
    idx = candidate_index(cand, tr)
    # every frontier node (a fan entry of a chosen row not chosen itself) is at most as probable as every chosen one
    for r in range(ROWS):
        if tr.depth[r] >= GAMMA:
            continue
        row = scores[tr.depth[r], idx[r]]
        order = torch.argsort(row, descending=True, stable=True)[: R.TREE_FAN]
        lps = R.tree_fan_lp(row[order].tolist())
        base = 0.0 if r == 0 else path_lp(cand, scores, tr, r)
        chosen = {tr.tokens[c] for c in range(1, ROWS) if tr.parent[c] == r}
        for k, c in enumerate(order.tolist()):
            if int(cand[tr.depth[r], c]) not in chosen:
                assert R._f32(base + lps[k]) <= worst + 1e-6


def test_a_peaked_lattice_is_linear():
    cand, _ = lattice(0, 1.0)
    scores = torch.full((GAMMA, 16, 16), -80.0)
    scores[..., :4] = torch.tensor([0.0, -50.0, -60.0, -70.0])
    tr = R.token_tree(7, cand, scores)
    assert tr.parent == [-1] + list(range(ROWS - 1)) and tr.depth == list(range(ROWS))
    assert tr.tokens[1:] == R.selector_walk(cand, scores)[0].tolist()


@pytest.mark.parametrize("seed", range(6))
def test_the_accept_follows_the_matching_path(seed):
    cand, scores = lattice(seed, 1.0)
    tr = R.token_tree(7, cand, scores)
    rnd = random.Random(seed)
    leaf = rnd.randrange(ROWS)
    want = [r for r in range(1, ROWS) if (tr.mask[leaf] >> r) & 1]
    # the target's argmax: the path's next token at each path row, a token no child carries at the leaf
    t1 = [rnd.randrange(10**6, 2 * 10**6) for _ in range(ROWS)]
    node = 0
    for r in want:
        t1[node] = tr.tokens[r]
        node = r
    a, path = R.tree_accept(tr, torch.tensor(t1), 100)
    assert path == want and [tr.depth[r] for r in path] == list(range(1, len(want) + 1))
    assert (a.accept, a.commit, a.bonus, a.pos) == (
        len(want),
        len(want) + 1,
        t1[leaf],
        100 + len(want) + 1,
    )
    assert a.committed.tolist() == [tr.tokens[r] for r in want] + [t1[leaf]]
    # a forced leaf names the path whatever the argmax
    assert R.tree_accept(tr, torch.tensor(t1), 100, forced=leaf)[1] == want


def test_the_constants_are_the_kernels():
    src = (KERNELS / "dflash/tree.cuh").read_text()
    c = {
        k: int(v)
        for k, v in re.findall(r"inline constexpr uint32_t (k\w+) = (\d+);", src)
    }
    assert (c["kRows"], c["kFan"], c["kTableWords"]) == (
        ROWS,
        R.TREE_FAN,
        TREE_WORDS,
    ) and TREE_FAN == R.TREE_FAN
    attn = (KERNELS / "attention.cuh").read_text()
    words = {
        k: int(v)
        for k, v in re.findall(r"constexpr uint32_t (kTree\w+Word) = (\d+);", attn)
    }
    # the attention CTAs read the table one word past kv_len
    assert words == {"kTreeDepthWord": 1 + c["kDepth"], "kTreeMaskWord": 1 + c["kMask"]}
