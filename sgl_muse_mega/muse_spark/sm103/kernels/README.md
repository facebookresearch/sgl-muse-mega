# Muse Spark Open kernels (SM103)

The device code of the batch-1 verify window (16 rows, or 8 for a DFlash2 block of 8): one persistent launch per decoder
layer plus the head and tail kernels, and the DFlash2 draft. Every build is one image: a TU (`layer.cu`, `head.cu`,
`dflash/{layer,head,verify_accept,kv_sync}.cu`) at `-DMK_IMAGE=<image>`, which instantiates that image's kernel. Every image
builds for one tensor-parallel world (`-DMK_WORLD=tp8|tp4`: a row of the world table in `framework.cuh`). The host
library (`../host/runtime.cu`) includes the same TU under `MK_HOST` for its argument structs and constants (no kernel is
instantiated there) and binds the arguments, tensor maps and launch shapes named in each image's launch comment.

| file | what |
|---|---|
| `ptx.cuh` | the PTX wrappers, by purpose: every PTX instruction of the kernels but three ("Order-dependent placement") |
| `framework.cuh` | the typed configuration (`mk::cfg`), the twelve-warp block contract and role maps, the carve descriptors, the ticketed counts, the row ops, the dense executor and its qkv rowops epilogue |
| `attention.cuh` | the attention phase: its constants, the sliding-window attention's mma.sync loop, the global attention (tcgen05), the split-KV combine, the global layers' policy |
| `moe.cuh` | the tensor-parallel MoE phase (`moe_tp`), the combine rows and the expert-weight L2 prefetch |
| `postattn.cuh` | the post-attention phase |
| `next_layer.cuh` | the next-layer phase |
| `layer.cu` | the layer images: the image table, the code prefetch, the phase program and its kernel |
| `head.cu` | the verify graph's head and tail images |
| `dflash/common.cuh` | the DFlash2 pieces the draft's TUs share |
| `dflash/layer.cu`, `dflash/head.cu`, `dflash/verify_accept.cu` | the DFlash2 draft layer, its entry and head, and the verify's lm_head with the accept |
| `dflash/tree.cuh` | the opt-in token-tree window (`-DMK_DFLASH_TREE=1`): the head's fan and tree walk, the tree accept, `tree_compact` |
| `dflash/sample.cuh` | the sampled step (`-DMK_DFLASH_SAMPLE=1`): the uniforms (Philox4x32-10), the head's sampled walk, `sample_accept`'s per-row statistics and decision |
| `dflash/kv_sync.cu` | `draft_kv_sync`: the committed positions' draft K / V from the rings into SGLang's draft KV pool |

Names: the layer, head and tail code lives in `namespace mk` (the DFlash2 draft in `mk::dflash`), the PTX wrappers in
`ptx` and the MoE phase in `moe_tp`. Only the C ABI of the host libraries keeps the `mk_` / `Mk` prefix (C has no
namespaces).

## Images

| image | TU | kernel | layers |
|---|---|---|---|
| `swa` | `layer.cu` | `decoder_layer<4>`: sliding-window attention, NVFP4 experts | 45 |
| `global` | `layer.cu` | same kernel: global attention, NVFP4 experts, row window | 15 |
| `swa_bf16` | `layer.cu` | same kernel: sliding-window attention, bf16 experts | layer 0 |
| `global_bf16_tail` | `layer.cu` | same kernel: global attention, bf16 experts, next-layer phase ends at the final norm | layer 61 |
| `window_feeds` | `head.cu` | the window's feeds; its arguments are SGLang's pool tensors (`req_to_token`, `req_pool_indices`, `full_to_swa`, `out_cache_loc`) | head |
| `entry_fused` | `head.cu` | `entry_fused`: vocab-shard lookup + embed norm + layer 0's pre-norm in one launch | head |
| `qkv_head` | `head.cu` | layer 0's qkv: the dense executor with the rowops epilogue | head |
| `lm_head` | `head.cu` | the gathered lm_head GEMM | tail |
| `softcap` | `head.cu` | the softcap into fp32 | tail |

The 8-row window (a DFlash2 block of 8) builds the same images with `-DMK_WINDOW_M=8` (the layer images, the qkv head,
the lm_head): x8 TMEM drains, the half packet round of the all-reduce push, the pooled gather and the vocab gather, 8 B
token pairs in the next-layer phase's K-quarter reductions, a token pair per rowops warp, 8 AR rows / 2 top-k units /
2 nq rows in the post-attention queue. The carve keeps the 16-row layout (`postattn::kMc`) and the attention the 16-row
split grid (`attn_cfg::kFeedRows`: kv_len = P + 16, its chunks, splits and blocks), but computes only the 8 live rows
(`attn_cfg::kM`, the window's rows): the sliding-window attention's 64-row tiles hold two q heads x 8 tokens per
mma.sync A tile on 4 warps; the global attention runs M = 64 tcgen05 MMAs (S / P / O in the first 16 TMEM lanes of each
quadrant, drained by the `.16x32bx2` shapes, two threads per row); 64-row partials, the combine at 2 rows per CTA. Its
window ends at P + 8 on the 16-row split grid, so its rows equal the 16-row window's first 8 bitwise (each row's
arithmetic is its own).

Flags: `-gencode=arch=compute_103a,code=sm_103a -std=c++20 -O1 --ptxas-options=--register-usage-level=10
--diag-suppress=39,161,174,177,186,940 --expt-relaxed-constexpr --expt-extended-lambda -I kernels`, and
`--compress-mode=none` on the layer images (the host's code-prefetch patch edits the cubin's ELF before loading it,
which needs the uncompressed image).

## The layer launch (`layer.cu`)

A compile-time phase program on 144 CTAs (36 width-4 clusters, 384 threads, one CTA per SM):

| phase | file | what it does |
|---|---|---|
| attention | `attention.cuh` (the attention's constants and arguments, the shared machinery, the global attention, the head phase: the sliding-window mma.sync loop and the combine, the global layers' policy) | split-KV decode attention and its combine (gate-norm) into x_attn |
| post-attention | `postattn.cuh` (the all-reduce push epilogue, top-k, the front item, the layout / unit queue / operands, the phase) and the all-reduce + norm rows of `framework.cuh` | o_proj, lamport all-reduce + norm, pooler and router GEMMs, top-k, the nq rows (norm + NVFP4 quant), the route tables |
| MoE | `moe.cuh` (the shard pack's layouts, the row window's byte math, the cross-rank gathers, the front, g1, g2, the MoE phase's entry) | the tensor-parallel MoE: g1 (gate/up + SwiGLU + requant, z gathered on every rank), g2 (full-K down projection, zb gathered) |
| next-layer | `next_layer.cuh` (the norm item, the combine item over `moe.cuh`'s combine rows, the phase body) and the qkv rowops epilogue of `framework.cuh` | combine into y, unpooler GEMM, cross-layer norm, next layer's qkv GEMM with the rowops epilogue, next layer's KV prefetch |

Shared pieces: `framework.cuh` (the role maps, the dense executor (the qkv head's and the lm_head's body; the
post-attention and next-layer phases share its conventions), the row ops, the ticketed counts and item barrier, the
carve descriptors), the expert L2 prefetch (`moe.cuh`, over the shard pack's layouts), the code prefetch (`layer.cu`)
and `ptx.cuh` (PTX wrappers by purpose: TMA, mbarrier, tcgen05 and the MMA descriptors, sync (fences, named barriers,
warp / lane / elect, PDL), cluster (peer-CTA st.async / atom), row loads / stores by coherence, global atomics,
multimem, the spins (arrival counts and the bounded spins), cvt, math). No host code: the tensor maps the host encodes
are in `../host/runtime.cu`.

Text size: a layer's 200-320 KB of text is fetched again at every launch, and only about the first 64 KB arrives with
the launch (the code prefetch loads the rest into L2), so loops that carry no per-element register state stay rolled
(`#pragma unroll 1`; the MoE phase's z / SF pushes to the peers at `unroll 4`): the same per-thread sequence of copies,
issues and waits, so the arithmetic is bitwise the unrolled loop's, in fewer bytes. Rolled: the post-attention W / x
deliveries, its MMA warp's ring drain and the router reduction's bulk copies (`postattn.cuh`); g1's peer pushes, its SF
copy issue and the g1 / g2 MMA issue loops, g2's producer stage loop (`moe.cuh`); the next-layer phase's W2 tail
prefetch (`next_layer.cuh`).

Kept for the schedule: the layer images carry unread constructs because removing them changes ptxas's schedule of the
images, not the work, and measured 1-2 % slower per step at TP8 (neutral or faster at TP4): the argument slots
`PostAttnArgs::stamps`, `NormQuantArgs::{pooled, pooled_ready, arrivals}`, `NextLayerArgs::out1`,
`NextLayerNormItem::Args::launch_ctr`, `LayerArgs::{w1_sf, clk, n_sm, sf_img}`, `AttnArgs::{prof_buf, attn_window}`,
`RouterArgs::{num_local, topk_idx, topk_local}` (the int64 top-k ids and their local view), the qkv epilogue's
`temp_scale` (always null), the zero `TM_G1_SFA` map and the `tmap_pooled` parameter (only descriptor-prefetched); the
global attention's uncalled 64-row drain, g1's resident-row scan, the front's `[j_lo, j_hi)` bounds, `combine_rows`'s
single-trip loop, the front item's third operand, the next-layer prologue's `QkvCoords` stores, the MoE phase's epoch
store, `LayerShared::entry_rdy`, the post-attention row's copied operand arrays and its two empty branches, the gathered
nq variant's vote, and the top-k row's int64 stores (TP8 ahead of its route arrival, TP4 behind it:
`World::route_early`).

## Configuration

All typed constants:

| where | what |
|---|---|
| `framework.cuh` (`mk::cfg`) | the model dims per rank (the world table), the window's rows, the smem ceiling and the dense ring; the window's rows are the only build define besides the image and the world (`-DMK_WINDOW_M=8`) |
| `layer.cu` (the image table) | the layer images as a table of `LayerImage{global_attn, bf16, tail}` rows, `kImage = image::MK_IMAGE`, and what follows (`kHRow`): the one kernel reads the image through `if constexpr` and `std::conditional_t` (the attention policy) |
| `attention.cuh` (`mk::attn_cfg`) | the attention phase's constants, derived from the image and the rows |

The head / tail images read `mk::cfg` alone. A layer launch triggers PDL once, at the next-layer phase's hand-over.

Every cross-CTA wait is bounded (it traps instead of hanging every rank), in seven kinds:

| wait | where | bound, then |
|---|---|---|
| `ptx::bounded_spin` / `ptx::ticket_wait` (`ptx.cuh`) | the all-reduce's lamport consume / every ticketed count | 30 s of global timer, a named report / 2^28 polls (~38 s); trap |
| the row polls (`framework.cuh`: the all-gather rows, the fp32 all-reduce consume) and the softcap's arrival wait (`head.cu`) | the gathered and fp32 all-reduced rows' packets; the lm_head gather's arrivals | 60 s of global timer; trap (the softcap prints a report first) |
| `moe_tp::lam_spin` (`moe.cuh`) | the MoE phase's two cross-rank gathers (the g2 z-tile gate, the combine's zb rows) and the anchor | 10 s of global timer (ahead of the waits queued behind them), a named report; trap |
| the global attention's `bar_wait` (`attention.cuh`) | its mbarrier waits | 2^24 polls; `ptx::device_assert` |
| `bar_wait` / `word_spin` (`dflash/layer.cu`, the attention phase) | the DFlash2 attention's waits | 2^26 polls; `ptx::device_assert` |
| `Budget` + `hang` (`dflash/layer.cu`, the row pieces) | the DFlash2 layer's row, gate and lamport waits | 30 s; the wait's record in host-pinned words, trap |
| `dflash::bounded_spin` + `spin_report` (`dflash/common.cuh`) | the DFlash2 entry's and head's waits | 60 s; a printed report, trap |

The DFlash2 layer keeps its own waits (`FinArgs::tap`, `GatherArgs::n_dst` / `::tap` fixed at bind): the shared ones
reschedule the composed layer image and measured slower.

## DFlash2 (`dflash/`)

The DFlash2 draft and accept: three TUs, each image picked with `-DMK_IMAGE=<image>`, plus `-DMK_DFLASH_B=8|16` where
the draft block B is a compile pin. The draft's kernels are `draft_<unit>` (and `draft_head`); the accept kernel is
`verify_accept`.

| image | TU | kernel | launch |
|---|---|---|---|
| `entry` (per B) | `head.cu` | `draft_entry<B>`: fc over the 5 captured features + hidden_norm (8-way output shard, multicast gather), the anchor's embedding row, layer 0's norm, conv coefficients and prepared rows | grid 144, 256 threads, `entry::Shape<B>::kSmem`, PDL |
| `qkv` (per B) | `layer.cu` | `draft_qkv<B>`: a draft layer's qkv on the dense executor, 4-CTA K split, the pair fixup's per-head norm + RoPE (the second CTA of a pair to arrive finishes the rows), the block and committed context K / V into the ring | grid 80, cluster 4, 256 threads |
| `layer`, `tail` (per B) | `layer.cu` | `draft_layer<prog>`: a draft layer in one launch: the attention phase on 9 clusters, then one unit queue (the split-KV combine, o_proj + conv finish + lamport all-reduce + norm, the MLP conv, gate_up + SwiGLU, down + all-reduce + norm, the next layer's conv and qkv); each conv's side 0 is a prepare unit (all 16 W blocks, the prepare in its epilogue), rounds >= 1 put no two o_proj tiles on one cluster; `tail` ends at the final norm, which also quantizes y rows 1 .. B - 1 to e4m3 for the draft head | grid 144 = 36 width-4 clusters, 384 threads, PDL |
| `draft_head` | `head.cu` | `draft_head`: the lm_head shard pre-scored in e4m3 (a scale per row) against the tail's e4m3 rows, the survivors of each rank's pre-score top-32 rescored exactly in bf16, then the exchange, the selector lattice and walk | 132 CTAs, 256 threads |
| (one) | `verify_accept.cu` | `verify_accept`: the verify's lm_head persist body with an argmax epilogue, one tagged word per row and rank, then the accept and the next iteration's feeds: no logits row | the lm_head's persistent grid, 256 threads |
| `draft_head_sample` | `head.cu` | `draft_head` with the sampled walk (`-DMK_DFLASH_SAMPLE=1`): slot e draws from softmax(score row / T) with one uniform, keeps its q row and draw | as `draft_head` |
| `verify_accept_sample` | `verify_accept.cu` | `sample_accept` (`-DMK_DFLASH_SAMPLE=1`), after the verify graph's lm_head and softcap: per live row a cluster holds the row's scaled logits (a column slice per CTA) and computes its max, the exact top-k and top-p thresholds (radix selection: counts, then masses in 2^-40 fixed point), the normalizer, its candidates' probabilities and per-chunk masses; the last CTA runs the chain's accept (block or token-wise), samples the bonus (chunk masses, then one chunk's elements) and writes verify_accept's commit and feeds | 8 CTAs per live row in clusters of 8, 256 threads, PDL |
| `kv_sync` | `kv_sync.cu` | `draft_kv_sync`: copies the ring rows of the positions committed since the last copy into SGLang's draft KV pool at their slots (an iteration's last launch; or after the draft alone, for a request's last commit) | grid 1, 256 threads, PDL |

`verify_accept`'s `accept_override` is the one test hook in a shipping image: a table of forced accepts, one per
iteration, so the GPU tests on synthetic target weights (which never accept a draft on their own) drive every accept
length through the shipped `verify_accept` binary; the server binds it null.

The token tree (`-DMK_DFLASH_TREE=1`, block 16, `dflash/tree.cuh`) builds tree images of `draft_head`, `verify_accept` and the
four layer images, bound through the same host libraries. The head keeps each lattice row's top 4 with their softmax
and grows the 15 rows of highest path probability, laid out in preorder with the rank-1 child first; it writes the
window's ids, the tree table (depth, ancestor mask, parent) after the feeds' kv_len word and each row's rope row at
P + depth. The layer images' attention CTAs read the table through the kv_len pointer and mask the window's keys by ancestor.
verify_accept accepts the deepest row whose path matches the target's argmax (its forced accept names the leaf row), and
`tree_compact` (one launch after it: a CTA per K / V pool, four per capture buffer) moves the path's rows to the slots
of rows 1 .. accept. Without the define the TUs are unchanged, token for token.

The sampled images (`-DMK_DFLASH_SAMPLE=1`, `dflash/sample.cuh`) bind through the same host libraries; every rank
holds the same gathered logits and draws the same uniforms (counter {step, run start, purpose, i / 4}, the request's
seed), and every reduction runs in a fixed order or in integers, so the ranks decide the same tokens without an
exchange. Without the define the TUs are unchanged, token for token.

Shared pieces: `common.cuh` (the entry's and the head's bounded wait, the draft's dims, the qkv images' pins, the two
GEMM epilogues: SwiGLU and the qkv's hd128 rowops, the head's candidate keys and step words); `layer.cu` also holds the
attention phase, its split-KV combine and the layer's row units (the prepare unit's epilogue, the protocol waits),
`verify_accept.cu` the accept's operands.

## Order-dependent placement

Two pieces stay where they are: the code prefetch ahead of the phase headers in `layer.cu`, and the DFlash2 bounded
wait ahead of the framework in `dflash/common.cuh`. nvcc numbers a TU's string constants in the order it emits the
functions that hold them, and their constant-bank slots follow that numbering: moved behind the framework, the trap
reports' strings take the earlier numbers and the same code reads other slots (a different cubin).

Both pieces sit ahead of `ptx.cuh`'s include, so their three PTX instructions stay inline: the code prefetch's
evict-last line load (`layer.cu`) and the bounded wait's global-timer read and trap (`dflash/common.cuh`). Including
`ptx.cuh` ahead of either changes the cubins (every layer image: the same instructions in a reordered image; the DFlash2
entry, head and verify_accept images: the strings' numbering).
