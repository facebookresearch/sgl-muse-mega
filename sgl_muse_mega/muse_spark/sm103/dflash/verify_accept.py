"""The verify's lm_head with the greedy accept in its epilogue (kernels/dflash/verify_accept.cu), one launch after the
verify's last layer. Each rank sends one tagged word per live row (its argmax over the shard, on the fp32 softcapped
logit) to every rank; no logits row exists, so the iteration's outputs carry the accept's per-row argmax (``top1``).
On a token-tree window the accept follows the tree's matching path and ``compact`` (tree_compact, one more launch)
moves the path's K / V rows of every layer and captured rows to the slots of rows 1 .. accept. A sampled step runs the
verify graph's lm_head and softcap instead and then ``SampleAccept`` (kernels/dflash/sample.cuh): the sampled accept on
the gathered logits, with the same commit and feeds.
"""

from __future__ import annotations

import ctypes

import torch

from ....common import host
from ..buffers import Symmetric
from ..verify import persist_grid
from .head import TOPK

STATS = 3  # per-row statistics words of sample_accept (sample.cuh kStatWords)


class VerifyAccept:
    """verify_accept for one Iteration (block size b): the accept's operands and outputs (the iteration's accept
    buffers, the next verify's feeds in place), the lm_head's weight, rows and persistent grid, and its own inbox and
    words. ``tree``: a tree verify_accept image (the feeds' tree table) and its ``compact`` launch.
    """

    def __init__(self, it, lib: host.Library, module: host.Module, tree: bool = False):
        cx = it.cx
        cfg, dev, ab, f = cx.cfg, cx.device, it.accept_buf, cx.feeds
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        grid = persist_grid(cfg.lm_head_pad // 64, sms)
        # two halves by launch parity x ranks x 16 rows of one 8-B word
        self.inbox = Symmetric(2 * cfg.tp * 16 * 8, dev, cx.group)
        self.part = it._zeros(grid * 16, dtype=torch.int64)
        self.state = it._zeros(2)
        it._proto_tensor(self.inbox.local, 0)
        self.launch = lib.bind(
            lib.dll.mk_dflash_verify_accept_bind,
            module,
            host.make(
                "MkDflashVerifyAcceptBind",
                w=cx.weights.lm_head,
                x=cx.bufs.layers[cfg.layers - 1].final_h3,
                hidden=cfg.hidden,
                n_out=cfg.vocab_rank,
                n_pad=cfg.lm_head_pad,
                grid=grid,
                rank=cx.rank,
                ranks=cfg.tp,
                inbox_peer=self.inbox.ptrs,
                part=self.part,
                state=self.state,
                tokens=cx.bufs.ids,
                words=it.words,
                top1=ab.top1,
                accept_override=it.override,
                n_override=0 if it.override is None else it.override.numel(),
                out_ring=ab.out_ring,
                ring=cx.ring,
                cos_sin_cache=f.cos_sin_cache,
                cos_sin_rows=f.cos_sin_cache.shape[0],
                d=f.cos_sin_cache.shape[1],
                cos_sin=ab.cos_sin,
                loc_full=ab.loc_full,
                loc_swa=ab.loc_swa,
                kv_len=ab.kv_len,
                dcos_sin_cache=cx.dcache,
                dcos_sin_rows=cx.dcache.shape[0],
                dd=cx.dcache.shape[1],
                dcos_sin=ab.dcos_sin,
                dslot=ab.dslot,
                dkv_len=ab.dkv_len,
                dring=cx.ring,
                b_live=it.b,
                scale=cfg.output_multiplier,
                cap=cfg.softcap,
                tree=f.tree if tree else None,
            ),
        )
        self.compact = None
        if tree:
            st = cx.stack
            pools = {"swa": [], "global": []}
            for L, kv in enumerate(st.kv):
                pools[cfg.attn_type(L)] += kv
            caps = [st.cap(L) for L in cx.capture]
            for t in pools["swa"] + pools["global"] + caps:
                assert t.is_contiguous(), tuple(t.shape)
            self.rows = torch.tensor(
                [t.data_ptr() for t in pools["swa"] + pools["global"] + caps],
                dtype=torch.int64,
                device=dev,
            )
            self.compact = lib.bind(
                lib.dll.mk_dflash_tree_compact_bind,
                module,
                host.make(
                    "MkDflashTreeCompactBind",
                    words=it.words,
                    tree=f.tree,
                    rows=self.rows,
                    n_swa=len(pools["swa"]),
                    n_full=len(pools["global"]),
                    n_cap=len(caps),
                    kv_row_bytes=cfg.kv_row_bytes,
                    cap_row_bytes=caps[0][0].numel() * caps[0].element_size(),
                ),
            )


class SampleAccept:
    """sample_accept for one Iteration (block size b) of a sampled step: after the verify graph's lm_head and softcap,
    the sampled accept on their fp32 softcapped logits (every rank's copy), then verify_accept's commit and the next
    iteration's feeds. Its per-row statistics, candidate probabilities and chunk masses, and its words. ``block``:
    SGLang's block verification (--speculative-use-block-verification), else token-wise rejection.
    """

    def __init__(self, it, lib: host.Library, module: host.Module, block: bool):
        cx = it.cx
        cfg, dev, ab, f = cx.cfg, cx.device, it.accept_buf, cx.feeds
        lib.dll.mk_dflash_sample_chunks.argtypes = [ctypes.c_int64]
        lib.dll.mk_dflash_sample_chunks.restype = ctypes.c_int64
        rows = cfg.rows
        self.stats = torch.zeros(rows, STATS, dtype=torch.float32, device=dev)
        self.pc = torch.zeros(rows, TOPK, dtype=torch.float32, device=dev)
        self.chunk = torch.zeros(
            rows,
            int(lib.dll.mk_dflash_sample_chunks(cfg.vocab)),
            dtype=torch.float32,
            device=dev,
        )
        self.state = it._zeros(2)
        h = it.head_out
        self.launch = lib.bind(
            lib.dll.mk_dflash_sample_accept_bind,
            module,
            host.make(
                "MkDflashSampleAcceptBind",
                logits=cx.bufs.logits.out,
                vocab=cfg.vocab,
                samp=it.samp,
                cand=h.cand,
                walk_q=h.q,
                walk_path=h.path,
                stats=self.stats,
                pc=self.pc,
                chunk=self.chunk,
                state=self.state,
                block_verification=int(block),
                tokens=cx.bufs.ids,
                words=it.words,
                out_ring=ab.out_ring,
                ring=cx.ring,
                cos_sin_cache=f.cos_sin_cache,
                cos_sin_rows=f.cos_sin_cache.shape[0],
                d=f.cos_sin_cache.shape[1],
                cos_sin=ab.cos_sin,
                loc_full=ab.loc_full,
                loc_swa=ab.loc_swa,
                kv_len=ab.kv_len,
                dcos_sin_cache=cx.dcache,
                dcos_sin_rows=cx.dcache.shape[0],
                dd=cx.dcache.shape[1],
                dcos_sin=ab.dcos_sin,
                dslot=ab.dslot,
                dkv_len=ab.dkv_len,
                dring=cx.ring,
                b_live=it.b,
            ),
        )
