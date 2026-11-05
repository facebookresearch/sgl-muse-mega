// C ABI of the stack's host launcher. A cubin is loaded with the driver API; each kernel launch is bound ONCE: its
// __grid_constant__ parameters (structs + tensor maps) are packed into a persistent blob in the kernel's device-side
// layout; mk_launch(handle, stream, pdl) is then the only per-step call (and graph capture freezes it anyway).
//
// Every bind struct holds only uint64_t / int64_t / double fields, one declaration per line (device pointers as
// uint64_t, 0 = absent; arrays sized by the #defines below): sgl_muse_mega/common/host.py builds its ctypes mirrors by
// parsing this file.
//
// Every int-returning function returns 0 on success and -1 on failure, with the reason in mk_last_error().
#pragma once

#include <stdint.h>

// the peer arrays' capacity: the largest tensor-parallel world (the kernels' mk::cfg::kRanks <= it)
#define MKH_RANKS 8

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MkModule MkModule;
typedef struct MkLaunch MkLaunch;

const char* mk_last_error(void);

// Loads a cubin (a linked sm_103a ELF) into the current context. code_prefetch_prefix: the mangled-name prefix of the
// kernel whose text the image prefetches into L2 (the code prefetch of kernels/layer.cu), or NULL. With a prefix the
// image is patched in memory before the load (mk_code_prefetch_patch) and the prefetch range is bound to the kernel's
// text range; without, the range stays zero and the prefetch loads nothing.
int mk_module_load(const void* image, uint64_t bytes, const char* code_prefetch_prefix, MkModule** out);
// The code prefetch needs the kernel's own text address in device data, which no API returns. nvcc emits
// g_code_prefetch_native = {&kernel, 0} with an R_CUDA_DESCRIPTOR relocation; the patch retypes it to R_CUDA_64 (the
// loader then writes the text pc) and writes the text size beside it. Edits `image` in place (4 bytes) and returns
// the kernel's text size.
int mk_code_prefetch_patch(void* image, uint64_t bytes, const char* prefix, uint64_t* text_bytes);
void mk_module_free(MkModule* module);

// Every mk_*_bind(module, ...) returns a launch record; with module == NULL the record is pack-only (no device call,
// cannot launch): the CPU checks read its parameter block.
int mk_launch(const MkLaunch* launch, uint64_t stream, int pdl);
void mk_launch_free(MkLaunch* launch);

typedef struct MkLaunchInfo {
    int64_t grid;
    int64_t block;
    int64_t cluster;
    int64_t smem;
    int64_t params;
    int64_t param_bytes;
} MkLaunchInfo;
int mk_launch_info(const MkLaunch* launch, MkLaunchInfo* out);
// The parameter block and each parameter's (offset, size).
int mk_launch_params(const MkLaunch* launch, void* blob, uint64_t cap_bytes, uint64_t* offsets, uint64_t* sizes,
                     int64_t cap_params);

// ---- the one-launch decoder layer (built once per image: libmk_layer_<image>.so) ----

// Compile-time facts of the image the library was built for: buffer extents the caller allocates and the launch
// geometry. Read once; nothing in python hard-codes a kernel constant.
typedef struct MkLayerConstants {
    int64_t global_attn;         // 1: global attention, 0: sliding-window attention
    int64_t bf16;                // 1: bf16 MoE variant, 0: NVFP4
    int64_t tail;                // 1: the last layer (its next-layer phase ends at the final norm)
    int64_t grid;                // persistent grid: every CTA co-resident
    int64_t cluster;             // cluster width
    int64_t block;               // threads per CTA
    int64_t smem;                // dynamic shared memory per CTA
    int64_t rows;                // the verify window's rows (bs x (1 + draft))
    int64_t hidden;              // model width
    int64_t moe_hidden;          // the experts' width
    int64_t oproj_k;             // o_proj K = q heads x 64
    int64_t qkv_row;             // [q | k | v | o-gate] row width
    int64_t kv_row_bytes;        // one K (or V) pool row: kv heads x 64 bf16
    int64_t experts;             // routed experts
    int64_t topk;                // experts per row
    int64_t words;               // u32 [h3_ready | route_count | front_done | nq_done | launch ticket per CTA]
    int64_t next_layer_words;    // u32 [ffn_ready | h3_ready | y_ready | launch ticket per CTA]
    int64_t counters;            // i32 router C1 pool + row arrivals
    int64_t ws_floats;           // router K-split partials
    int64_t router_split;        // K slices of the router (the logits sum them in slice order)
    int64_t fctr_words;          // attention counters (i32)
    int64_t sm_epoch_words;      // attention epochs (i64)
    int64_t arena_bytes;         // attention-type arena: x_attn | qkv_out | split-KV workspace
    int64_t off_x_attn;          // arena offset of x_attn
    int64_t off_qkv_out;         // arena offset of the qkv row
    int64_t plane_stride_min;    // AR workspace slot (bf16 elements): the message, partial planes and gathers fit
    int64_t pooled_ready_words;  // u64 per-tile pooled-ready words (the K-window variant: 1 + the pooler tiles; else 0)
    int64_t pool_ag_off;         // pooler gather row, bf16 elements from a slot base (prefilled 0xFFFF)
    int64_t unpool_ag_off;       // unpooler gather row
    int64_t moe_rows;            // MoE variant capacity (token surfaces' rows)
    int64_t moe_slots;           // route slots (rows x topk)
    int64_t moe_tiles;           // route tiles
    int64_t arrive_words;        // u32 cross-rank anchor words
    int64_t gsync_words;         // u32 MoE-phase fan-in / cursor words
    int64_t zb_ring;             // zb parity planes
    int64_t inter;               // full expert intermediate (the gathered h row)
    int64_t z_row_bytes;         // gathered h row bytes per slot (fp4: I/2, bf16: 2I)
    int64_t zsf_row_bytes;       // its scale factors per slot (I/16)
    int64_t nextkv_splits;       // the sliding-window successor's split count (0: no next-layer KV prefetch)
} MkLayerConstants;
int mk_layer_constants(MkLayerConstants* out);
// The layer kernel's mangled-name prefix (the code-prefetch entry for mk_module_load).
const char* mk_layer_kernel(void);

// One layer launch's operands. Field groups follow the kernel's phases.
typedef struct MkLayerBind {
    // post-attention: o_proj + lamport all-reduce + norm
    uint64_t w_o;
    uint64_t attn_arena;  // this layer's attention-type arena (x_attn lives at off_x_attn)
    uint64_t peer_bases;  // device i64 [ranks + 1]: every rank's AR workspace base, then the multicast alias
    int64_t rank;
    int64_t plane_stride;  // bf16 elements per AR slot
    uint64_t stream_in;
    uint64_t alpha;
    uint64_t beta;
    uint64_t prenorm_w;
    uint64_t stream_out;
    uint64_t h3;
    uint64_t h3f;
    // pooler || router -> top-k -> pre-expert norm + quant -> route tables (the front item)
    uint64_t w_p;
    uint64_t w_r;
    uint64_t logits;
    uint64_t ws;
    uint64_t counters;
    double router_mult;
    uint64_t expert_bias;
    int64_t num_local;  // the routing's local view (topk_idx_local) shard size: written, read by nothing here
    uint64_t topk_idx;
    uint64_t topk_idx_local;
    uint64_t topk_w;
    uint64_t topk_idx32;
    double route_scale;
    uint64_t gamma;
    double eps;
    uint64_t q;  // MoE token surfaces: fp4 rows | scale factors | per-token scale (bf16: q only)
    uint64_t sf;
    uint64_t x_scale;
    uint64_t rt_counts;
    uint64_t rt_offsets;
    uint64_t rt_slot_tok;
    uint64_t rt_slot_k;
    uint64_t rt_tok_slot;
    uint64_t rt_tile_e;
    uint64_t rt_tile_r0;
    uint64_t rt_n_tiles;
    uint64_t words;
    uint64_t pooled_ready;  // u64 [pooled_ready_words] (0 when the image has none)
    // MoE phase
    uint64_t w1;
    uint64_t w1_sf_blk;  // NVFP4: the blocked W1 SF image (the only W1 SF storage)
    uint64_t w2;
    uint64_t w2_sf;
    uint64_t g1_alphas;
    uint64_t g1_scale_c;
    uint64_t g2_alphas;
    uint64_t gamma_p1;
    uint64_t z;  // symmetric region views (this rank's)
    uint64_t z_sf;
    uint64_t z_ex;
    uint64_t zb;
    int64_t zb_par_slots;
    uint64_t arrive;
    uint64_t z_peer[MKH_RANKS];  // every rank's copy of the same views
    uint64_t zsf_peer[MKH_RANKS];
    uint64_t zex_peer[MKH_RANKS];
    uint64_t zb_peer[MKH_RANKS];
    uint64_t arrive_peer[MKH_RANKS];
    uint64_t gsync;
    uint64_t epoch;
    uint64_t y;
    // attention phase
    uint64_t attn_fctr;
    uint64_t attn_sm_epoch;
    uint64_t k_cache;  // this layer's pool, [kv_rows, kv heads x 64] bf16
    uint64_t v_cache;
    int64_t kv_rows;
    uint64_t page_table;  // i64 [1, page_stride] byte offsets, position-ordered
    uint64_t kv_len;      // i32 [1]
    int64_t page_stride;
    // expert prefetch: 1 = L2-prefetch the experts this layer routed to last step (NVFP4 images only)
    int64_t prefetch_experts;
    // next-layer phase into the successor (the tail: the final norm; the qkv GEMM's operands are placeholders)
    uint64_t w_up;
    uint64_t w_qkv;
    uint64_t h3q;
    uint64_t qkv_out;
    uint64_t cos_sin;
    uint64_t cache_loc;
    uint64_t next_k_cache;
    uint64_t next_v_cache;
    double q_scale;
    double qkv_eps;
    uint64_t temp_scale;
    uint64_t c_alpha;
    uint64_t c_beta;
    uint64_t c_pre_w;
    uint64_t c_post_w;
    uint64_t c_stream_out;
    uint64_t next_layer_words;
    // next-layer KV prefetch (the successor's sliding-window pool; splits 0 = off)
    uint64_t nk_page_table;
    uint64_t nk_kv_len;
    int64_t nk_page_stride;
    int64_t nk_splits;
    int64_t nk_window;
    // live window rows (1..rows): the route tables and the combine skip rows >= b_live, whose MoE output is 0. A DFlash
    // block of 8 verifies as rows 0-7 of the 16-row window, the pads 8-15 repeating the live ids (positions P+8..P+15)
    int64_t b_live;
} MkLayerBind;
int mk_layer_bind(MkModule* module, const MkLayerBind* bind, MkLaunch** out);

// ---- the rest of the verify graph (libmk_aux.so) ----

// Window feeds: the step's RoPE rows, cache-write slots, kv_len and page-table byte offsets.
typedef struct MkFeedsBind {
    uint64_t positions;
    uint64_t cos_sin_cache;
    int64_t cos_sin_rows;
    uint64_t out_cache_loc;
    uint64_t swa_out_cache_loc;
    uint64_t seq_lens;
    uint64_t req_pool_indices;
    uint64_t req_to_token;
    int64_t r2t_rows;
    int64_t r2t_width;
    uint64_t full_to_swa;
    int64_t swa_map_len;
    uint64_t cos_sin;
    uint64_t loc_full;
    uint64_t loc_swa;
    uint64_t kv_len;
    uint64_t pt_full;
    uint64_t pt_swa;
    int64_t rows;
    int64_t head_dim;
    int64_t kv_row_bytes;
} MkFeedsBind;
int mk_feeds_bind(MkModule* module, const MkFeedsBind* bind, MkLaunch** out);

// The fused entry: vocab-shard embedding gather from the owning rank, embed RMSNorm, layer-0 pre-attention norm.
typedef struct MkEntryBind {
    uint64_t ids;
    uint64_t embed_peer[MKH_RANKS];  // every rank's vocab shard (symmetric memory)
    int64_t shard_rows;
    uint64_t pre_w;
    uint64_t residual;
    uint64_t normed;
    int64_t rows;
    int64_t hidden;
    double eps_embed;
    double eps_pre;
} MkEntryBind;
int mk_entry_bind(MkModule* module, const MkEntryBind* bind, MkLaunch** out);

// Layer 0's head: the qkv GEMM + qk-norm + RoPE + K/V append (the next-layer phase's qkv GEMM as a standalone launch).
typedef struct MkQkvBind {
    uint64_t w;
    uint64_t x;
    uint64_t out;
    uint64_t cos_sin;
    uint64_t cache_loc;
    uint64_t k_cache;
    uint64_t v_cache;
    uint64_t temp_scale;
    double q_scale;
    double eps;
    int64_t hidden;
} MkQkvBind;
int mk_qkv_bind(MkModule* module, const MkQkvBind* bind, MkLaunch** out);

// lm_head: the rank's vocab shard, every tile multicast into every rank's gathered logits row.
typedef struct MkLmHeadBind {
    uint64_t w;
    uint64_t x;
    uint64_t mc_base;
    int64_t arrive_off;
    uint64_t grid_done;
    int64_t v_total;
    int64_t col0;
    int64_t n_out;
    int64_t n_pad;
    int64_t grid;
    int64_t hidden;
} MkLmHeadBind;
int mk_lm_head_bind(MkModule* module, const MkLmHeadBind* bind, MkLaunch** out);

// Softcap over the gathered row once every rank's lm_head arrived.
typedef struct MkSoftcapBind {
    uint64_t x;
    uint64_t out;
    int64_t numel;
    double scale;
    double cap;
    uint64_t arrive;
    uint64_t state;
    int64_t per_launch;
} MkSoftcapBind;
int mk_softcap_bind(MkModule* module, const MkSoftcapBind* bind, MkLaunch** out);

// ---- DFlash2: the draft entry (libmk_dflash_entry_b<B>.so, built with -DMK_DFLASH_B) ----

// The image's compile-time facts: launch geometry, the caller's buffer extents, the packed weight geometry.
typedef struct MkDflashEntryConstants {
    int64_t rows;  // the draft block B
    int64_t grid;
    int64_t block;
    int64_t smem;
    int64_t hidden;
    int64_t features;  // captured target layers (fc K = features x hidden)
    int64_t ranks;
    int64_t fc_n;         // fc output columns per rank
    int64_t conv_proj_n;  // conv_proj0 outputs per rank
    int64_t coeffs;       // conv coefficients per row (side, tap, group)
    int64_t groups;
    int64_t sym_bytes;  // the symmetric region: ctx_raw | coefficient row (0xFFFF between uses)
    int64_t ws_floats;  // fp32 K-slice partials
    int64_t ctr_words;  // u32 per-block slice counts (zeroed once, +split per launch)
} MkDflashEntryConstants;
int mk_dflash_entry_constants(MkDflashEntryConstants* out);

typedef struct MkDflashEntryBind {
    uint64_t cap[5];
    uint64_t anchor;
    uint64_t embed_peer[8];
    int64_t shard_rows;
    uint64_t w_fc;
    uint64_t w_conv_proj;
    uint64_t in_norm;
    uint64_t hid_norm;
    uint64_t conv_base;
    uint64_t e_mask;
    uint64_t h_mask;
    uint64_t d_mask;
    uint64_t xq_const;
    uint64_t sym_local;
    uint64_t sym_mc;
    int64_t rank;
    uint64_t ctx_n;
    uint64_t xq;
    uint64_t resid;
    uint64_t c_att1;
    uint64_t h_row;
    uint64_t delta;
    uint64_t ws;
    uint64_t ctr;
    double eps;
} MkDflashEntryBind;
int mk_dflash_entry_bind(MkModule* module, const MkDflashEntryBind* bind, MkLaunch** out);

// ---- DFlash2: a draft layer's qkv (+ hd128 rowops) on the dense executor (libmk_dflash_dense.so) ----

typedef struct MkDflashQkvBind {
    uint64_t w;        // [1280, hidden] bf16: the rank's 8 q heads, then its k and v head (dflash/weights.py pack_qkv)
    uint64_t x;        // [2 block, hidden] bf16: the block rows, then the committed context rows
    uint64_t out;      // [2 block, 1280] bf16: the block rows' q heads (the attention's Q)
    uint64_t tickets;  // u32 [40] per (head, token quarter), zeroed once, never reset
    uint64_t cos_sin;  // [2 block, 128] fp32 per row {cos[64] | sin[64]}
    uint64_t slot;     // [2 block] int32 ring slots
    uint64_t commit;   // int32 c': context rows j >= c' store nothing
    uint64_t q_norm;   // [128] bf16
    uint64_t k_norm;   // [128] bf16
    uint64_t k_ring;   // [4096, 128] bf16
    uint64_t v_ring;   // [4096, 128] bf16
    int64_t block;     // B: 8 or 16, the module's MK_DFLASH_B
    int64_t hidden;    // K
    double eps;
} MkDflashQkvBind;
int mk_dflash_qkv_bind(MkModule* module, const MkDflashQkvBind* b, MkLaunch** out);

// ---- DFlash2: the draft attention's geometry for either block size: the ring, the window, the partials and the
// attention / combine shapes (libmk_dflash_attn.so) ----

// The kernels' compile-time facts at block size `block`: the extents the caller allocates and the launch shapes.
typedef struct MkDflashAttnConstants {
    int64_t ring_rows;     // draft KV ring rows per K / V ring (slot = position mod ring_rows)
    int64_t head_dim;      // one kv head of a K / V ring row: head_dim bf16
    int64_t q_heads;       // the rank's query heads (q row = q_heads x head_dim bf16)
    int64_t kv_heads;      // the rank's kv heads (a ring row = kv_heads x head_dim bf16; a group of 8 q heads each)
    int64_t window;        // keys on each side of a query
    int64_t block_keys;    // keys per attention block (CTA j walks block j of the window's aligned range)
    int64_t attn_ctas;     // the launched attention's grid: one (kv head, key block) per CTA, kv head major
    int64_t combine_ctas;  // the launched combine's grid
    int64_t ws_row;        // one CTA's partial in fp32 words: O [128 rows][head_dim / 2 bf16 pairs], then (m, l)
} MkDflashAttnConstants;
int mk_dflash_attn_constants(int64_t block, MkDflashAttnConstants* out);

// ---- DFlash2: the draft layer, one launch record per binding (libmk_dflash_layer_<image>_b<B>.so) ----
// Slot numbering follows the draft layer's program: GEMM 0 o_proj, 1 / 2 mlp_conv_proj side 0 / 1, 3 gate_up, 4 down, 5
// attn_conv_proj side 0, 6 qkv, 7 attn_conv_proj side 1, 8 the first draft layer's own qkv (image layer0); on the
// attention images slots 1 and 5 are prepare units (the side's full 16 blocks, prepared through prep 0 / 1); fin 0 / 1
// = the attention / MLP site; gather 0..3 = mlp_conv_proj side 0 / 1, attn_conv_proj side 0 / 1; rows 0 / 1 and prep
// 0 / 1 = the attention / MLP sublayer's. An image binds the slots its program reads (MkDflashLayerConstants masks).

typedef struct MkDflashLayerConstants {
    int64_t rows;           // B: the block rows
    int64_t xm;             // the x tile's rows: every x buffer holds at least the rows its unit's MMA reads
    int64_t grid;           // persistent grid: every CTA co-resident
    int64_t cluster;        // cluster width
    int64_t block;          // threads per CTA
    int64_t smem;           // dynamic shared memory per CTA
    int64_t hidden;         // the residual width
    int64_t inter;          // the rank's MLP intermediate
    int64_t coef_row;       // one side's packed coefficients per block row
    int64_t slot;           // one rank's lamport slot (bf16 elements); a site's planes are 2 x ranks x slot
    int64_t ranks;          // TP world of the all-reduce
    int64_t row_arrivals;   // the rows' ready count per launch
    int64_t prep_arrivals;  // a prepare unit's prepared-x ready count per launch (its slot's prep_ready)
    int64_t act_arrivals;   // the SwiGLU tiles' ready count per launch
    int64_t queue_len;      // the program's entries
    int64_t gemm_mask;      // bit g: the program reads GEMM slot g
    int64_t fin_mask;
    int64_t gat_mask;
    int64_t ar_mask;
    int64_t prep_mask;
    int64_t swiglu;             // 1: the program runs the SwiGLU hook
    int64_t qkv;                // 1: the program runs the qkv hook
    int64_t gemm_m[9];          // per GEMM slot: the MMA's rows (0: unread)
    int64_t gemm_blocks[9];     // W blocks the program walks from the slot's block0
    int64_t has_attn;           // 1: an attention image (the attention and its combine lead the launch)
    int64_t combine_ctas;       // the combine's CTAs (its x_ready arrivals per launch); 0 without the attention
    int64_t head_qkv;           // 1: the first draft layer's own qkv leads the queue (GEMM 8, qkv0_*, attn_q_ready)
    int64_t head_qkv_arrivals;  // its q_ready arrivals per launch
} MkDflashLayerConstants;
int mk_dflash_layer_constants(MkDflashLayerConstants* out);

typedef struct MkDflashLayerBind {
    uint64_t gemm_w[9];  // W [w_rows, k] bf16
    int64_t gemm_w_rows[9];
    uint64_t gemm_x[9];  // x [x_rows, k] bf16
    int64_t gemm_x_rows[9];
    int64_t gemm_k[9];
    int64_t gemm_block0[9];
    uint64_t gemm_gate[9];  // u32 count: x ready at arrivals x (ticket + 1); 0 = x is the previous launch's
    int64_t gemm_gate_arrivals[9];
    uint64_t fin_bases[2];  // device i64 [ranks + 1]: every rank's site plane base, then the site's multicast address
    uint64_t fin_coef[2];   // the side-1 coefficient rows [rows][coef_row] bf16 (this rank's)
    uint64_t fin_base1[2];  // [2 taps][hidden] bf16
    uint64_t gat_dst[16];   // gather slot s: destinations 4 s .. 4 s + 3 (each node rank's side row, device address)
    int64_t gat_block0[4];  // the slot's first 64-coefficient block in the row
    uint64_t ar_bases[2];   // = the site's fin_bases
    uint64_t ar_r_in[2];
    uint64_t ar_r_out[2];
    uint64_t ar_w[2];
    uint64_t ar_h[2];
    double ar_eps[2];
    uint64_t ar_ready[2];
    uint64_t ar_rin_gate[2];  // 0 = r_in is the previous launch's
    int64_t ar_rin_arrivals[2];
    uint64_t prep_coef[2];  // the side-0 coefficient rows (this rank's)
    uint64_t prep_h[2];
    uint64_t prep_base0[2];  // [2 taps][hidden] bf16
    uint64_t prep_x[2];
    uint64_t prep_h_gate[2];  // 0 = h is the previous launch's
    int64_t prep_h_arrivals[2];
    uint64_t prep_ready[2];
    uint64_t act;          // [rows][inter] bf16: the SwiGLU hook's output
    uint64_t act_ready;    // u32
    uint64_t qkv_out;      // [2 rows][1280] bf16: the qkv rows, the block rows' q heads after the rowops
    uint64_t qkv_tickets;  // u32 [40]: the pair fixup's per-(head, token quarter) tickets, zeroed once
    uint64_t qkv_cos_sin;  // [2 rows][128] fp32 {cos | sin} per row
    uint64_t qkv_slot;     // [2 rows] int32 ring slots
    uint64_t qkv_commit;   // int32 c'
    uint64_t qkv_q_norm;   // [128] bf16
    uint64_t qkv_k_norm;   // [128] bf16
    uint64_t qkv_k_ring;   // [4096][128] bf16
    uint64_t qkv_v_ring;   // [4096][128] bf16
    double qkv_eps;
    uint64_t launch_ctr;  // u32 [grid], zeroed once
    int64_t rank;
    int64_t phase_offset;  // the draft layer's index: its planes' phase offset
    uint64_t diag;         // host-pinned u64 [8] hang record (kernels/dflash/layer.cu), or 0
    uint64_t xq;           // the last layer: e4m3 [B - 1][hidden], y rows 1 .. B - 1 for the draft head
    // an attention image's attention (MkDflashLayerConstants.has_attn); the combine writes gemm_x[0] (x_attn) and
    // counts attn_x_ready, which o_proj's gate (gemm_gate[0], combine_ctas arrivals) waits for
    uint64_t attn_q;  // bf16 [B][1024]: this layer's q rows (the previous launch's qkv), attn_q_stride bytes apart
    int64_t attn_q_stride;
    uint64_t attn_k_ring;    // [4096][128] bf16: this layer's draft KV
    uint64_t attn_v_ring;    // [4096][128] bf16
    uint64_t draft_kv_len;   // i32: P + B
    uint64_t attn_partials;  // fp32 [36][ws_row] (MkDflashAttnConstants): the partials
    uint64_t attn_word;      // u32, zeroed once
    uint64_t attn_x_ready;   // u32, zeroed once
    // the first draft layer's own qkv (MkDflashLayerConstants.head_qkv): its rows land in attn_q and the K / V rings,
    // then the attention's gate attn_q_ready (u32, zeroed once); rope rows, slots, commit and eps are the qkv_* ones
    uint64_t qkv0_tickets;  // u32 [40], zeroed once
    uint64_t qkv0_q_norm;
    uint64_t qkv0_k_norm;
    uint64_t attn_q_ready;
} MkDflashLayerBind;
int mk_dflash_layer_bind(MkModule* module, const MkDflashLayerBind* bind, MkLaunch** out);

// ---- DFlash2: the draft head (libmk_dflash_draft_head.so) ----

// kernels/dflash/head.cu: an e4m3 pre-score of the vocab shard, exact rescoring of its survivors, then the cross-rank
// exchange, the selector lattice and walk.
typedef struct MkDflashDraftHeadBind {
    uint64_t wq;       // e4m3 [vocab_rank][hidden]: the vocab shard, one scale per row
    uint64_t w_scale;  // f32 [vocab_rank]
    uint64_t wb;       // bf16 [vocab_rank + hproj][hidden]: the vocab shard, then the W_P shard
    uint64_t y;        // bf16 [>= gamma + 1][hidden]: the block's final rows (rows 1 .. gamma are read)
    uint64_t xq;  // e4m3 [gamma][hidden]: y rows 1 .. gamma, written by the last draft layer (MkDflashLayerBind xq)
    int64_t hidden;
    int64_t vocab_rank;
    int64_t hproj;
    int64_t n_pad;  // the fp8 shard's padded rows (64-row blocks)
    int64_t grid;
    int64_t rank;
    int64_t ranks;
    int64_t gamma;
    int64_t ksurv;           // survivors cover the rank's pre-score top-ksurv (16 .. 64)
    uint64_t inbox_peer[8];  // every rank's inbox (symmetric), u64 [2][ranks][16][16 + hproj / 2]
    uint64_t pred_codebook;
    uint64_t succ_codebook;
    uint64_t words;
    uint64_t state;  // u32 [mk_dflash_draft_head_state_words()], zeroed once
    uint64_t part;   // u64 [grid][16][16]
    uint64_t resc;   // u64 [16][mk_dflash_draft_head_max_survivors()]
    uint64_t hloc;   // bf16 [16][hproj]
    uint64_t cand;
    uint64_t unary;
    uint64_t hhat;
    uint64_t scores;
    uint64_t pick;  // i32 [gamma][16]
    uint64_t drafts;
    uint64_t verify_ids;
    double mult;
    double cap;
    // a token-tree head (an MK_DFLASH_TREE image, kernels/dflash/tree.cuh), or tree = 0: the linear chain
    uint64_t tree;           // i32 [96] the tree table (the target feeds' kv_len buffer + 1)
    uint64_t fan_idx;        // i32 [gamma][16][4]
    uint64_t fan_lp;         // f32 [gamma][16][4]
    uint64_t cos_sin;        // f32 [16][d] the verify window's rope rows
    uint64_t cos_sin_cache;  // f32 [cos_sin_rows][d]
    int64_t cos_sin_rows;
    int64_t d;
    // a sampled head (an MK_DFLASH_SAMPLE image, kernels/dflash/sample.cuh), or samp = 0: the greedy walk
    uint64_t samp;       // i64 [4] the sampling words
    uint64_t walk_q;     // f32 [gamma][16] each slot's probabilities over its candidates
    uint64_t walk_path;  // i32 [gamma] the candidate index each slot drew
} MkDflashDraftHeadBind;
int64_t mk_dflash_draft_head_state_words(void);
int64_t mk_dflash_draft_head_max_survivors(void);
int mk_dflash_draft_head_bind(MkModule* module, const MkDflashDraftHeadBind* bind, MkLaunch** out);

// ---- DFlash2: verify_accept, the verify's lm_head + the greedy accept (libmk_dflash_verify_accept.so) ----

// The verify's lm_head with the greedy accept in its epilogue (kernels/dflash/verify_accept.cu): the lm_head's
// operands, the cross-rank exchange of each live row's argmax, the accept's step words and committed tokens, and the
// next iteration's target and draft feeds.
typedef struct MkDflashVerifyAcceptBind {
    uint64_t w;  // bf16 [n_out][hidden]: the rank's vocab shard
    uint64_t x;  // bf16 [16][hidden]: the window's final normed rows
    int64_t hidden;
    int64_t n_out;  // the shard's rows (vocab ids rank x n_out ..)
    int64_t n_pad;  // the shard's padded rows (64-row blocks)
    int64_t grid;
    int64_t rank;
    int64_t ranks;
    uint64_t inbox_peer[8];  // every rank's inbox (symmetric), u64 [2][ranks][16]
    uint64_t part;           // u64 [grid][16]
    uint64_t state;          // u32 [2], zeroed once
    uint64_t tokens;         // i64 [b_live] the window's ids
    uint64_t words;
    uint64_t top1;             // i32 [16]
    uint64_t accept_override;  // i32 [n_override] or 0
    int64_t n_override;
    uint64_t out_ring;  // i32 [ring]
    int64_t ring;
    uint64_t cos_sin_cache;  // f32 [cos_sin_rows][d]
    int64_t cos_sin_rows;
    int64_t d;
    uint64_t cos_sin;   // f32 [16][d]
    uint64_t loc_full;  // i32 [16]
    uint64_t loc_swa;
    uint64_t kv_len;          // i32 [1]
    uint64_t dcos_sin_cache;  // f32 [dcos_sin_rows][dd]
    int64_t dcos_sin_rows;
    int64_t dd;
    uint64_t dcos_sin;  // f32 [2 b_live][dd]
    uint64_t dslot;     // i32 [2 b_live]
    uint64_t dkv_len;   // i32 [1]: P' + b_live
    int64_t dring;
    int64_t b_live;
    double scale;
    double cap;
    // a token-tree verify_accept (an MK_DFLASH_TREE image, kernels/dflash/tree.cuh), or tree = 0: the linear chain
    uint64_t tree;  // i32 [96] the tree table
} MkDflashVerifyAcceptBind;
int mk_dflash_verify_accept_bind(MkModule* module, const MkDflashVerifyAcceptBind* bind, MkLaunch** out);

// tree_compact (kernels/dflash/tree.cuh, the MK_DFLASH_TREE verify_accept image): after the tree accept, the accepted
// path's rows to the slots of rows 1 .. accept in every listed buffer (rows: u64 [n_swa + n_full + n_cap] base
// pointers, the SWA layers' K / V pools, the full layers', then the capture buffers (row = window row)).
typedef struct MkDflashTreeCompactBind {
    uint64_t words;
    uint64_t tree;
    uint64_t rows;
    int64_t n_swa;
    int64_t n_full;
    int64_t n_cap;
    int64_t kv_row_bytes;
    int64_t cap_row_bytes;
} MkDflashTreeCompactBind;
int mk_dflash_tree_compact_bind(MkModule* module, const MkDflashTreeCompactBind* bind, MkLaunch** out);

// sample_accept (kernels/dflash/sample.cuh, the MK_DFLASH_SAMPLE verify_accept image): the sampled accept on the
// verify graph's gathered softcapped logits (after its lm_head and softcap), then verify_accept's commit and feeds.
typedef struct MkDflashSampleAcceptBind {
    uint64_t logits;  // f32 [16][vocab]: every rank's copy
    int64_t vocab;
    uint64_t samp;       // i64 [4] the sampling words
    uint64_t cand;       // i32 [gamma][16] the head's candidates
    uint64_t walk_q;     // f32 [gamma][16]
    uint64_t walk_path;  // i32 [gamma]
    uint64_t stats;      // f32 [16][3]
    uint64_t pc;         // f32 [16][16]
    uint64_t chunk;      // f32 [16][mk_dflash_sample_chunks(vocab)]
    uint64_t state;      // u32 [2], zeroed once
    int64_t block_verification;
    // the commit and the next iteration's feeds, as MkDflashVerifyAcceptBind's
    uint64_t tokens;
    uint64_t words;
    uint64_t out_ring;
    int64_t ring;
    uint64_t cos_sin_cache;
    int64_t cos_sin_rows;
    int64_t d;
    uint64_t cos_sin;
    uint64_t loc_full;
    uint64_t loc_swa;
    uint64_t kv_len;
    uint64_t dcos_sin_cache;
    int64_t dcos_sin_rows;
    int64_t dd;
    uint64_t dcos_sin;
    uint64_t dslot;
    uint64_t dkv_len;
    int64_t dring;
    int64_t b_live;
} MkDflashSampleAcceptBind;
int64_t mk_dflash_sample_chunks(int64_t vocab);
int mk_dflash_sample_accept_bind(MkModule* module, const MkDflashSampleAcceptBind* bind, MkLaunch** out);

// draft_kv_sync (kernels/dflash/kv_sync.cu): the committed positions' draft K / V from the draft's rings into SGLang's
// draft KV pool (ring / pool: u64 [n_buf] base pointers, the K and V buffer of each draft layer in the same order).
typedef struct MkDflashKvSyncBind {
    uint64_t words;
    uint64_t window;  // i64 [16] the window's slots
    uint64_t state;   // i64 [17]
    uint64_t ring;
    uint64_t pool;
    int64_t n_buf;
    int64_t row_bytes;
    int64_t ring_rows;
    int64_t lag;  // 1: an iteration's last launch; 0: after a draft alone
} MkDflashKvSyncBind;
int mk_dflash_kv_sync_bind(MkModule* module, const MkDflashKvSyncBind* bind, MkLaunch** out);

#ifdef __cplusplus
}
#endif
