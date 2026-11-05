// The host libraries' one source: the launcher core every library shares (cubin loading with the code-prefetch patch,
// the parameter blob, launch records, tensor maps; driver API only) and each library's binds. -DMK_HOST=<library>
// selects the library (artifacts.py): layer (one per layer image, with its -DMK_IMAGE), aux, and the DFlash2 libraries.
// The device files are included for their argument structs and constants: under MK_HOST they instantiate no kernel, so
// a parameter struct is the kernel's own and its layout is the kernel's by construction.
#define MK_HOST_layer 1
#define MK_HOST_aux 2
#define MK_HOST_dflash_entry 3
#define MK_HOST_dflash_dense 4
#define MK_HOST_dflash_attn 5
#define MK_HOST_dflash_layer 6
#define MK_HOST_dflash_draft_head 7
#define MK_HOST_dflash_verify_accept 8
#define MK_HOST_CAT_(x) MK_HOST_##x
#define MK_HOST_CAT(x) MK_HOST_CAT_(x)
#define MK_HOST_LIB MK_HOST_CAT(MK_HOST)

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "abi.h"

// HOST side: cuTensorMapEncodeTiled for tiled 2D / 3D boxes (the MoE phase's maps, the core).
// Coordinates follow the driver: dim 0 is the innermost, stride-1 axis (columns); strides from dim 1 up are in bytes.
// A descriptor the driver rejects throws.

namespace tmap {

inline CUtensorMap encode(CUtensorMapDataType dtype, uint32_t rank, void* global_ptr, const cuuint64_t* global_dim,
                          const cuuint64_t* global_strides, const cuuint32_t* box_dim,
                          const cuuint32_t* element_strides, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promo) {
    CUtensorMap m{};
    const CUresult r =
        cuTensorMapEncodeTiled(&m, dtype, rank, global_ptr, global_dim, global_strides, box_dim, element_strides,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char* what = nullptr;
        cuGetErrorString(r, &what);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled rejected a rank-") + std::to_string(rank) +
                                 " descriptor: " + (what ? what : "unknown error"));
    }
    return m;
}

inline CUtensorMap encode_tiled_2d(void* global_ptr, CUtensorMapDataType dtype, uint64_t global_rows,
                                   uint64_t global_cols, uint64_t row_stride_bytes, uint32_t box_rows,
                                   uint32_t box_cols, CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    const cuuint64_t global_dim[2] = {global_cols, global_rows};
    const cuuint64_t global_strides[1] = {row_stride_bytes};
    const cuuint32_t box_dim[2] = {box_cols, box_rows};
    const cuuint32_t element_strides[2] = {1, 1};
    return encode(dtype, 2, global_ptr, global_dim, global_strides, box_dim, element_strides, swizzle, promo);
}

// dim 2 = the outermost axis (depth, e.g. the expert index); one row group per box (box_dim[2] = 1)
inline CUtensorMap encode_tiled_3d(void* global_ptr, CUtensorMapDataType dtype, uint64_t global_depth,
                                   uint64_t global_rows, uint64_t global_cols, uint64_t row_stride_bytes,
                                   uint64_t depth_stride_bytes, uint32_t box_rows, uint32_t box_cols,
                                   CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    const cuuint64_t global_dim[3] = {global_cols, global_rows, global_depth};
    const cuuint64_t global_strides[2] = {row_stride_bytes, depth_stride_bytes};
    const cuuint32_t box_dim[3] = {box_cols, box_rows, 1u};
    const cuuint32_t element_strides[3] = {1, 1, 1};
    return encode(dtype, 3, global_ptr, global_dim, global_strides, box_dim, element_strides, swizzle,
                  CU_TENSOR_MAP_L2_PROMOTION_NONE);
}

}  // namespace tmap

// The launcher core shared by every host library: cubin loading (with the code-prefetch patch), the parameter blob,
// launch records. Driver API only.

#define MKH_API extern "C" __attribute__((visibility("default")))

namespace mkh {

[[noreturn]] void fail(const std::string& what);
void check(CUresult r, const char* what);

#define MKH_REQUIRE(cond, msg)                      \
    do {                                            \
        if (!(cond)) ::mkh::fail(std::string(msg)); \
    } while (0)

// 64-B aligned bytes (CUtensorMap's alignment)
struct AlignedBytes {
    struct Free {
        void operator()(uint8_t* p) const { std::free(p); }
    };
    std::unique_ptr<uint8_t, Free> p;
    static AlignedBytes make(size_t n) {
        AlignedBytes b{
            std::unique_ptr<uint8_t, Free>(static_cast<uint8_t*>(std::aligned_alloc(64, (n + 63) / 64 * 64)))};
        if (!b.p) throw std::bad_alloc();
        return b;
    }
    uint8_t* get() const { return p.get(); }
};

// A kernel's parameter block in its device-side layout: every value at its own alignment, in declaration order (the
// layout cuFuncGetParamInfo reports; make_launch checks the two agree).
class Params {
   public:
    template <class T>
    void add(const T& v) {
        const size_t off = (bytes_ + alignof(T) - 1) / alignof(T) * alignof(T);
        MKH_REQUIRE(off + sizeof(T) <= kCap, "parameter block overflow");
        std::memcpy(blob_.get() + off, &v, sizeof(T));
        slots_.emplace_back(off, sizeof(T));
        bytes_ = off + sizeof(T);
    }
    size_t bytes() const { return bytes_; }
    const uint8_t* data() const { return blob_.get(); }
    const std::vector<std::pair<size_t, size_t>>& slots() const { return slots_; }

   private:
    static constexpr size_t kCap = 32764;  // the CUDA >= 12.1 kernel parameter limit
    AlignedBytes blob_{AlignedBytes::make(32768)};
    size_t bytes_ = 0;
    std::vector<std::pair<size_t, size_t>> slots_;
};

struct Geometry {
    unsigned grid, block, cluster, smem;
    bool persistent;  // every CTA must be co-resident: grid == cluster x the
                      // resident cluster count
};

// Builds the launch record of the module's unique kernel whose mangled name starts with `prefix`. A null module gives a
// pack-only record (no device call): the CPU checks read its parameter block.
MkLaunch* make_launch(MkModule* module, const char* prefix, const Params& params, const Geometry& g);

// Runs f, mapping any exception to -1 + mk_last_error().
int guard(const std::function<void()>& f);

template <class T>
T* ptr(uint64_t v) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(v));
}

// A 2D tensor map over a row-major [outer][inner] array of `elem`-byte elements: box_inner x box_outer elements, the
// 128 B swizzle or none, L2 promotion 256 B (tmap::encode_tiled_2d encodes it; a rejected descriptor throws).
inline CUtensorMap tma_2d(uint64_t base, CUtensorMapDataType dtype, int elem, int inner, int outer, int box_inner,
                          int box_outer, bool swizzle) {
    MKH_REQUIRE(base != 0, "tensor map over a null base");
    return tmap::encode_tiled_2d(ptr<void>(base), dtype, uint64_t(outer), uint64_t(inner), uint64_t(inner) * elem,
                                 uint32_t(box_outer), uint32_t(box_inner),
                                 swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
}
// ... of bf16 rows, 64-element (one 128 B swizzle atom) boxes
inline CUtensorMap bf16_map(uint64_t base, int inner, int outer, int box_outer, bool swizzle = true) {
    return tma_2d(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, inner, outer, 64, box_outer, swizzle);
}

}  // namespace mkh

struct MkModule {
    CUmodule module = nullptr;
    std::vector<std::string> kernels;  // entry kernels (mangled), from the image's symbol table
};

struct MkLaunch {
    CUfunction fn = nullptr;
    mkh::Geometry g{};
    mkh::AlignedBytes blob;
    size_t bytes = 0;
    std::vector<void*> args;                       // one pointer per parameter, into blob
    std::vector<std::pair<size_t, size_t>> slots;  // (offset, size) per parameter
};

namespace mkh {

namespace {

thread_local std::string g_error;

// ---- the cubin's ELF: just enough to name its kernels and patch the code-prefetch object ----

constexpr uint16_t kEmCuda = 190;
constexpr uint32_t kShtSymtab = 2, kShtRela = 4;
constexpr uint32_t kRCuda64 = 2, kRCudaDescriptor = 35;
constexpr uint8_t kSttFunc = 2, kStoCudaEntry = 0x10;
const char* const kPrefetchNative = "_ZN2mk22g_code_prefetch_nativeE";
const char* const kPrefetchRange = "_ZN2mk15g_code_prefetchE";

struct Section {
    std::string name;
    uint32_t name_off, type, link, info;
    uint64_t offset, size, entsize;
};

struct Symbol {
    std::string name;
    uint8_t info, other;
    uint16_t shndx;
    uint64_t value, size;
};

template <class T>
T rd(const std::vector<uint8_t>& d, uint64_t off) {
    MKH_REQUIRE(off + sizeof(T) <= d.size(), "truncated ELF");
    T v;
    std::memcpy(&v, d.data() + off, sizeof(T));
    return v;
}

std::string cstr(const std::vector<uint8_t>& d, uint64_t off) {
    MKH_REQUIRE(off < d.size(), "ELF string offset");
    const char* s = reinterpret_cast<const char*>(d.data() + off);
    return std::string(s, strnlen(s, d.size() - off));
}

struct Elf {
    std::vector<Section> sections;
    std::vector<Symbol> symbols;
    uint32_t symtab = 0;

    explicit Elf(const std::vector<uint8_t>& d) {
        MKH_REQUIRE(d.size() >= 64 && std::memcmp(d.data(),
                                                  "\x7f"
                                                  "ELF\x02\x01",
                                                  6) == 0,
                    "not an uncompressed little-endian ELF64 cubin");
        MKH_REQUIRE(rd<uint16_t>(d, 18) == kEmCuda, "not a CUDA image");
        const uint64_t shoff = rd<uint64_t>(d, 40);
        const uint16_t shentsize = rd<uint16_t>(d, 58), shnum = rd<uint16_t>(d, 60), shstrndx = rd<uint16_t>(d, 62);
        MKH_REQUIRE(shentsize == 64 && shstrndx < shnum, "section table");
        for (uint16_t i = 0; i < shnum; ++i) {
            const uint64_t h = shoff + uint64_t(i) * 64;
            sections.push_back({"", rd<uint32_t>(d, h), rd<uint32_t>(d, h + 4), rd<uint32_t>(d, h + 40),
                                rd<uint32_t>(d, h + 44), rd<uint64_t>(d, h + 24), rd<uint64_t>(d, h + 32),
                                rd<uint64_t>(d, h + 56)});
        }
        for (auto& s : sections)
            s.name = cstr(d, sections[shstrndx].offset + s.name_off);
        for (uint32_t i = 0; i < sections.size(); ++i)
            if (sections[i].type == kShtSymtab) symtab = i;
        MKH_REQUIRE(symtab != 0 && sections[symtab].entsize == 24, "symbol table");
        const Section& st = sections[symtab];
        const uint64_t strs = sections.at(st.link).offset;
        for (uint64_t o = st.offset; o < st.offset + st.size; o += 24)
            symbols.push_back({cstr(d, strs + rd<uint32_t>(d, o)), rd<uint8_t>(d, o + 4), rd<uint8_t>(d, o + 5),
                               rd<uint16_t>(d, o + 6), rd<uint64_t>(d, o + 8), rd<uint64_t>(d, o + 16)});
    }

    uint32_t section(const std::string& name) const {
        uint32_t found = 0, n = 0;
        for (uint32_t i = 0; i < sections.size(); ++i)
            if (sections[i].name == name) found = i, ++n;
        MKH_REQUIRE(n == 1, "missing or ambiguous section " + name);
        return found;
    }

    bool entry(const Symbol& s) const { return (s.info & 15) == kSttFunc && (s.other & kStoCudaEntry); }

    uint32_t unique_symbol(const std::string& prefix, bool kernel) const {
        uint32_t found = 0, n = 0;
        for (uint32_t i = 0; i < symbols.size(); ++i)
            if (symbols[i].name.rfind(prefix, 0) == 0 && entry(symbols[i]) == kernel) found = i, ++n;
        MKH_REQUIRE(n == 1, "expected exactly one symbol matching " + prefix + ", found " + std::to_string(n));
        return found;
    }
};

// The code prefetch needs the kernel's own text address in device data, which no API returns. nvcc emits
// mk::g_code_prefetch_native = {&kernel, 0} with an R_CUDA_DESCRIPTOR relocation against the kernel; retyping it to
// R_CUDA_64 makes the loader write the kernel's text pc instead, and the text size goes into the second word. Only the
// relocation record and the 8 size bytes change: the code is untouched.
uint64_t patch_code_prefetch(std::vector<uint8_t>& d, const std::string& prefix) {
    const Elf e(d);
    const uint32_t native = e.unique_symbol(kPrefetchNative, false);
    const uint32_t kernel = e.unique_symbol(prefix, true);
    const uint32_t init = e.section(".nv.global.init");
    const Symbol& ns = e.symbols[native];
    const Section& is = e.sections[init];
    MKH_REQUIRE(ns.shndx == init && ns.size == 16 && ns.value % 8 == 0 && ns.value + 16 <= is.size,
                "the code-prefetch object must be an initialized 16-byte global");
    const Symbol& ks = e.symbols[kernel];
    const Section& text = e.sections[e.section(".text." + ks.name)];
    MKH_REQUIRE(ks.value == 0 && ks.size == text.size && text.size > 0, "kernel symbol and text extent disagree");
    const Section& rela = e.sections[e.section(".rela.nv.global.init")];
    MKH_REQUIRE(rela.type == kShtRela && rela.info == init && rela.link == e.symtab && rela.size % 24 == 0,
                "initializer relocations");
    int64_t hit = -1;
    for (uint64_t o = rela.offset; o < rela.offset + rela.size; o += 24)
        if (rd<uint64_t>(d, o) == ns.value) {
            MKH_REQUIRE(hit < 0, "the code-prefetch object has more than one relocation");
            hit = int64_t(o);
        }
    MKH_REQUIRE(hit >= 0, "the code-prefetch object has no relocation");
    const uint64_t info = rd<uint64_t>(d, hit + 8);
    MKH_REQUIRE((info >> 32) == kernel && (info & 0xffffffffu) == kRCudaDescriptor && rd<int64_t>(d, hit + 16) == 0,
                "the code-prefetch relocation must be a descriptor of the prefetched kernel");
    for (int i = 0; i < 16; ++i)
        MKH_REQUIRE(d[is.offset + ns.value + i] == 0, "the code-prefetch object is not zero");
    const uint64_t retyped = (uint64_t(kernel) << 32) | kRCuda64;
    std::memcpy(d.data() + hit + 8, &retyped, 8);
    std::memcpy(d.data() + is.offset + ns.value + 8, &text.size, 8);
    return text.size;
}

int current_device_attr(CUdevice_attribute a) {
    CUdevice dev;
    check(cuCtxGetDevice(&dev), "cuCtxGetDevice");
    int v = 0;
    check(cuDeviceGetAttribute(&v, a, dev), "cuDeviceGetAttribute");
    return v;
}

}  // namespace

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error(what);
}

void check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    fail(std::string(what) + " failed: " + (name ? name : std::to_string(int(r))));
}

int guard(const std::function<void()>& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown exception";
    }
    return -1;
}

MkLaunch* make_launch(MkModule* m, const char* prefix, const Params& p, const Geometry& g) {
    auto l = std::make_unique<MkLaunch>();
    l->g = g;
    l->blob = AlignedBytes::make(p.bytes());
    l->bytes = p.bytes();
    std::memcpy(l->blob.get(), p.data(), p.bytes());
    for (const auto& s : p.slots()) {
        l->args.push_back(l->blob.get() + s.first);
        l->slots.push_back(s);
    }
    if (m == nullptr) return l.release();

    std::string name;
    for (const auto& k : m->kernels)
        if (k.rfind(prefix, 0) == 0) {
            MKH_REQUIRE(name.empty(), std::string("more than one kernel matches ") + prefix);
            name = k;
        }
    MKH_REQUIRE(!name.empty(), std::string("no kernel matches ") + prefix);
    check(cuModuleGetFunction(&l->fn, m->module, name.c_str()), "cuModuleGetFunction");

    // the packed block must be the kernel's own layout, parameter by parameter
    const auto& slots = p.slots();
    for (size_t i = 0;; ++i) {
        size_t off = 0, size = 0;
        const CUresult r = cuFuncGetParamInfo(l->fn, i, &off, &size);
        if (r == CUDA_ERROR_INVALID_VALUE) {
            MKH_REQUIRE(i == slots.size(), "the kernel takes " + std::to_string(i) + " parameters, packed " +
                                               std::to_string(slots.size()));
            break;
        }
        check(r, "cuFuncGetParamInfo");
        MKH_REQUIRE(i < slots.size() && slots[i].first == off && slots[i].second == size,
                    "parameter " + std::to_string(i) + " is at " + std::to_string(off) + " (" + std::to_string(size) +
                        " B) in the kernel, packed elsewhere");
    }

    int stat = 0;
    check(cuFuncGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, l->fn), "cuFuncGetAttribute");
    const int optin = current_device_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    MKH_REQUIRE(int(g.smem) <= optin - stat, "dynamic smem " + std::to_string(g.smem) + " B exceeds the opt-in " +
                                                 std::to_string(optin) + " B minus the static " + std::to_string(stat));
    check(cuFuncSetAttribute(l->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(g.smem)),
          "cuFuncSetAttribute(max dynamic smem)");
    if (g.persistent) {
        // the persistent grid spins on ticketed waits: every CTA must be resident at once
        CUlaunchAttribute a{};
        a.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        a.value.clusterDim.x = g.cluster;
        a.value.clusterDim.y = a.value.clusterDim.z = 1;
        CUlaunchConfig cfg{};
        cfg.gridDimX = g.cluster;
        cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = g.block;
        cfg.blockDimY = cfg.blockDimZ = 1;
        cfg.sharedMemBytes = g.smem;
        cfg.attrs = &a;
        cfg.numAttrs = 1;
        int clusters = 0;
        check(cuOccupancyMaxActiveClusters(&clusters, l->fn, &cfg), "cuOccupancyMaxActiveClusters");
        MKH_REQUIRE(unsigned(clusters) * g.cluster == g.grid,
                    "the persistent grid " + std::to_string(g.grid) + " is not every resident CTA (" +
                        std::to_string(clusters) + " clusters of " + std::to_string(g.cluster) + ")");
    }
    return l.release();
}

}  // namespace mkh

using mkh::guard;

MKH_API const char* mk_last_error(void) {
    return mkh::g_error.c_str();
}

MKH_API int mk_code_prefetch_patch(void* image, uint64_t bytes, const char* prefix, uint64_t* text_bytes) {
    return guard([&] {
        std::vector<uint8_t> d(static_cast<uint8_t*>(image), static_cast<uint8_t*>(image) + bytes);
        *text_bytes = mkh::patch_code_prefetch(d, prefix);
        std::memcpy(image, d.data(), bytes);
    });
}

MKH_API int mk_module_load(const void* image, uint64_t bytes, const char* code_prefetch_prefix, MkModule** out) {
    return guard([&] {
        std::vector<uint8_t> d(static_cast<const uint8_t*>(image), static_cast<const uint8_t*>(image) + bytes);
        auto m = std::make_unique<MkModule>();
        uint64_t text = 0;
        if (code_prefetch_prefix != nullptr) text = mkh::patch_code_prefetch(d, code_prefetch_prefix);
        const mkh::Elf e(d);
        for (const auto& s : e.symbols)
            if (e.entry(s)) m->kernels.push_back(s.name);
        mkh::check(cuInit(0), "cuInit");
        CUcontext ctx = nullptr;
        mkh::check(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
        MKH_REQUIRE(ctx != nullptr, "no current CUDA context (make the device current first)");
        mkh::check(cuModuleLoadData(&m->module, d.data()), "cuModuleLoadData");
        if (code_prefetch_prefix != nullptr) {
            // load the kernel before reading the object its relocation resolves
            CUfunction fn;
            const uint32_t k = e.unique_symbol(code_prefetch_prefix, true);
            mkh::check(cuModuleGetFunction(&fn, m->module, e.symbols[k].name.c_str()), "cuModuleGetFunction");
            CUdeviceptr native = 0, range_dev = 0;
            size_t nb = 0, sb = 0;
            mkh::check(cuModuleGetGlobal(&native, &nb, m->module, mkh::kPrefetchNative), "cuModuleGetGlobal(native)");
            mkh::check(cuModuleGetGlobal(&range_dev, &sb, m->module, mkh::kPrefetchRange), "cuModuleGetGlobal(range)");
            MKH_REQUIRE(nb == 16 && sb == 16, "code-prefetch globals");
            unsigned long long range[2];
            mkh::check(cuMemcpyDtoH(range, native, 16), "cuMemcpyDtoH(code-prefetch range)");
            // the prefetch reads one word per 128-B line from the base: an unaligned base would skip the last line
            MKH_REQUIRE(range[0] != 0 && range[0] % 128 == 0 && range[1] == text,
                        "the loader did not resolve the kernel's text range");
            mkh::check(cuMemcpyHtoD(range_dev, range, 16), "cuMemcpyHtoD(code-prefetch range)");
        }
        *out = m.release();
    });
}

MKH_API void mk_module_free(MkModule* m) {
    if (m == nullptr) return;
    if (m->module != nullptr) cuModuleUnload(m->module);
    delete m;
}

MKH_API int mk_launch(const MkLaunch* l, uint64_t stream, int pdl) {
    return guard([&] {
        MKH_REQUIRE(l->fn != nullptr, "a pack-only record cannot launch");
        CUlaunchAttribute a[2]{};
        unsigned n = 0;
        if (l->g.cluster > 1) {
            a[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            a[n].value.clusterDim.x = l->g.cluster;
            a[n].value.clusterDim.y = a[n].value.clusterDim.z = 1;
            ++n;
        }
        if (pdl) {
            a[n].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            a[n].value.programmaticStreamSerializationAllowed = 1;
            ++n;
        }
        CUlaunchConfig cfg{};
        cfg.gridDimX = l->g.grid;
        cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = l->g.block;
        cfg.blockDimY = cfg.blockDimZ = 1;
        cfg.sharedMemBytes = l->g.smem;
        cfg.hStream = reinterpret_cast<CUstream>(static_cast<uintptr_t>(stream));
        cfg.attrs = a;
        cfg.numAttrs = n;
        mkh::check(cuLaunchKernelEx(&cfg, l->fn, const_cast<void**>(l->args.data()), nullptr), "cuLaunchKernelEx");
    });
}

MKH_API void mk_launch_free(MkLaunch* l) {
    delete l;
}

MKH_API int mk_launch_info(const MkLaunch* l, MkLaunchInfo* out) {
    return guard([&] {
        *out = {int64_t(l->g.grid), int64_t(l->g.block),     int64_t(l->g.cluster),
                int64_t(l->g.smem), int64_t(l->args.size()), int64_t(l->bytes)};
    });
}

MKH_API int mk_launch_params(const MkLaunch* l, void* blob, uint64_t cap_bytes, uint64_t* offsets, uint64_t* sizes,
                             int64_t cap_params) {
    return guard([&] {
        MKH_REQUIRE(l->bytes <= cap_bytes && int64_t(l->slots.size()) <= cap_params, "output buffers too small");
        std::memcpy(blob, l->blob.get(), l->bytes);
        for (size_t i = 0; i < l->slots.size(); ++i) {
            offsets[i] = l->slots[i].first;
            sizes[i] = l->slots[i].second;
        }
    });
}

#if MK_HOST_LIB == MK_HOST_layer

// The one-launch decoder layer's host side: packs one layer's operands into the kernel's parameter block (5 structs of
// the post-attention phase, the MoE variant's LayerArgs with its own tensor maps, the attention's, the expert
// prefetch's and the next-layer phase's) and binds the launch. Built once per image with that image's pins
// (-DMK_IMAGE): the parameter structs are the device headers' own, so their layout is the kernel's by construction.
#include <algorithm>

#include "layer.cu"

// The MoE phase's tensor maps (moe.cuh's stage and box constants).

namespace moe_tp {

// The tensor-map encode for one launch's LayerArgs: the reference the host side reproduces (the maps' geometry is
// the kernel's contract). kHRow: the B-side SF block and the exponent plane come off the stage-major z_sf / z_ex
// as 2D boxes of 512 B x STAGES.
inline void impl_host_encode_hrow_b_tmaps(LayerArgs& a) {
    if constexpr (kHRow) {
        // z_sf / z_ex are stage-major (zsf_off): a 2D map of STAGES rows x kMaxSlots*16 B; the box = 32 rows' 16 B for
        // every stage (128 u32 x STAGES), landing [stage][32 rows][16 B] = g2's pieces
        a.tmaps.tm[TM_G2_SFB] = tmap::encode_tiled_2d(a.z_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, /*rows=*/g2::STAGES,
                                                      /*cols=*/uint64_t(kMaxSlots) * 4,
                                                      /*row_stride=*/uint64_t(kMaxSlots) * 16, /*box_rows=*/g2::STAGES,
                                                      /*box_cols=*/kTokBox * 4, CU_TENSOR_MAP_SWIZZLE_NONE);
        a.tmaps.tm[TM_G2_EXB] = tmap::encode_tiled_2d(a.z_ex, CU_TENSOR_MAP_DATA_TYPE_UINT32, /*rows=*/g2::STAGES,
                                                      /*cols=*/uint64_t(kMaxSlots) * 4,
                                                      /*row_stride=*/uint64_t(kMaxSlots) * 16, /*box_rows=*/g2::STAGES,
                                                      /*box_cols=*/kTokBox * 4, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
}
inline void impl_host_init_tmaps(LayerArgs& a) {
    // g1 A: SWIZZLE_128B box (kARows rows, kBK elements). K-blocked: the rows axis is [ks][kARows] and a stage
    // reads the contiguous run at row ks x kARows; the depth indexes the shard pack (e x kFbPr + local block)
    if constexpr (kBf16) {
        // kBf16: the same pack layouts, rows and row strides as the fp4 maps (a K stage is one 128 B row either
        // way), bf16 elements: a box column count of kBK = 64; no scale-factor maps (their slots stay unencoded and
        // unread)
        a.tmaps.tm[TM_G1_A] = tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w1_fp4), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                    uint64_t(kE) * kFbFull, size_t(kW1KSlices) * kARows, kBK,
                                                    /*row_stride=*/kFeedBytes, /*depth_stride=*/kW1TileBytes,
                                                    /*box_rows=*/kARows, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
        a.tmaps.tm[TM_G1_B] = tmap::encode_tiled_2d(const_cast<uint8_t*>(a.x_fp4), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                    kBMax, kH, /*row_stride=*/size_t(kH) * 2,
                                                    /*box_rows=*/1, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
        a.tmaps.tm[TM_G2_A] = tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w2_fp4), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                    kE, size_t(kHbPr) * kW2KSlices * 128, kBK,
                                                    /*row_stride=*/kFeedBytes, /*depth_stride=*/size_t(kHpr) * kI * 2,
                                                    /*box_rows=*/128, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
        a.tmaps.tm[TM_G2_B] = tmap::encode_tiled_2d(a.z_fp4, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kMaxSlots, kI,
                                                    /*row_stride=*/size_t(kI) * 2,
                                                    /*box_rows=*/kTokBox, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
        return;
    }
    constexpr uint32_t kSfBoxU32 = g1::SFU32;
    constexpr CUtensorMapSwizzle kSfSwz = CU_TENSOR_MAP_SWIZZLE_NONE;
    a.tmaps.tm[TM_G1_A] = tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w1_fp4), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                                                uint64_t(kE) * kFbFull, size_t(kW1KSlices) * kARows, kBK,
                                                /*row_stride=*/kFeedBytes, /*depth_stride=*/kW1TileBytes,
                                                /*box_rows=*/kARows, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
    // the A side's SF comes from the blocked image (tm_sfab): the TM_G1_SFA slot stays zero (descriptor-prefetched
    // only)
    a.tm_sfab = tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w1_sf_blk), CU_TENSOR_MAP_DATA_TYPE_UINT64,
                                      uint64_t(kE) * kFbPr * g1::NCHUNK, /*rows=*/g1::SFA_BLOCK / 2048,
                                      /*cols=*/256, /*row_stride=*/2048, /*depth_stride=*/g1::SFA_BLOCK,
                                      /*box_rows=*/g1::SFA_BLOCK / 2048, /*box_cols=*/256, CU_TENSOR_MAP_SWIZZLE_NONE);
    // g1 B/SFB gather4: one token row per request, OOB rows zero-fill the pads.
    a.tmaps.tm[TM_G1_B] = tmap::encode_tiled_2d(const_cast<uint8_t*>(a.x_fp4), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                                                kBMax, kH, /*row_stride=*/kH / 2,
                                                /*box_rows=*/1, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
    a.tmaps.tm[TM_G1_SFB] = tmap::encode_tiled_2d(const_cast<uint8_t*>(a.x_sf), CU_TENSOR_MAP_DATA_TYPE_UINT32, kBMax,
                                                  kSfU32H, /*row_stride=*/kH / kSfVec,
                                                  /*box_rows=*/1, /*box_cols=*/kSfBoxU32, kSfSwz);
    // g2 A: box (128 output cols, kBK of the FULL K); rows axis =
    // [hb<kHbPr][ks][128] over the SHARD pack; depth = expert.
    a.tmaps.tm[TM_G2_A] = tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w2_fp4), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                                                kE, size_t(kHbPr) * kW2KSlices * 128, kBK,
                                                /*row_stride=*/kFeedBytes, /*depth_stride=*/size_t(kHpr) * (kI / 2),
                                                /*box_rows=*/128, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
    constexpr uint32_t kSf2BoxU32 = kSfU32I;
    constexpr CUtensorMapSwizzle kSf2Swz = CU_TENSOR_MAP_SWIZZLE_NONE;
    a.tmaps.tm[TM_G2_SFA] =
        tmap::encode_tiled_3d(const_cast<uint8_t*>(a.w2_sf), CU_TENSOR_MAP_DATA_TYPE_UINT32, kE, kHpr, kSfU32I,
                              /*row_stride=*/kI / kSfVec, /*depth_stride=*/size_t(kHpr) * (kI / kSfVec),
                              /*box_rows=*/128, /*box_cols=*/kSf2BoxU32, kSf2Swz);
    // g2 B: the gathered z. The box is kTokBox rows, not the tile capacity: a wider box re-reads neighbouring
    // experts' z every stage
    a.tmaps.tm[TM_G2_B] =
        tmap::encode_tiled_2d(a.z_fp4, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, kMaxSlots, kI, /*row_stride=*/kI / 2,
                              /*box_rows=*/kTokBox, /*box_cols=*/kBK, CU_TENSOR_MAP_SWIZZLE_128B);
    if constexpr (kHRow) {
        impl_host_encode_hrow_b_tmaps(a);  // the stage-major B-side maps
    } else {
        a.tmaps.tm[TM_G2_SFB] = tmap::encode_tiled_2d(a.z_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, kMaxSlots, kSfU32I,
                                                      /*row_stride=*/kI / kSfVec, /*box_rows=*/kTokBox,
                                                      /*box_cols=*/kSf2BoxU32, kSf2Swz);
    }
}

}  // namespace moe_tp

namespace {

namespace acfg = mk::attn_cfg;
using mkh::bf16_map;
using mkh::ptr;
using mkh::tma_2d;

constexpr int kW = mk::cfg::kW;
constexpr int kM = int(mk::postattn::kM);
constexpr int kHidden = int(mk::postattn::kHidden);
constexpr int kGrid = int(mk::AttnPolicy::kGrid);
constexpr int kSmem = kLayerSmem;
constexpr int kOprojK = 64 * int(mk::postattn::kOprojStages);
constexpr int kPooledN = int(mk::postattn::kPoolerBlocks * mk::postattn::BN);
constexpr int kExperts = mk::cfg::kExperts;
constexpr int kRSplit = mk::cfg::kRouterSplit;
constexpr int kRouterTiles = kExperts / (2 * 128);  // the router's 256-wide n-tiles (one CTA pair of 128 each)
constexpr int kKvCols = int(acfg::kNumKvHeads) * 64;
constexpr int kQkvRow = 2 * (mk::cfg::kQHeads + mk::cfg::kKvHeads) * 64;
constexpr int kWords = 4;           // h3_ready | route_count | front_done | nq_done
constexpr int kNextLayerWords = 3;  // ffn_ready | h3_ready | y_ready
constexpr int kPostAttnClusters =
    int(mk::postattn::kOprojBlocks) / kW;  // the post-attention queue: the clusters of the o_proj tiles
constexpr int kGatherRanks = 4;  // the pooler / unpooler gather group: a node's ranks (next_layer.cuh ag_rank % 4)
constexpr int kBlocks1 = kHidden / 64 / kGatherRanks;  // the rank's unpooler column blocks
constexpr int kNextKvSplits = kGrid / int(mk::kNextKvUnits);
constexpr bool kGlobalAttn = mk::kImage.global_attn;
constexpr bool kBf16 = moe_tp::kBf16;
constexpr bool kTail = mk::kImage.tail;
// the K-window variant's per-tile pooled-ready words (u64; word 1 + t = pooler tile t's launches; word 0 is unused)
constexpr int kPooledReadyWords = mk::postattn::kKWindow ? 1 + int(mk::postattn::kPoolerBlocks) : 0;
constexpr float kPostNormEps = 1e-8f;  // the per-expert post-norm's eps (the model's post_norm_eps)
constexpr const char* kKernelPrefix = "_ZN2mk13decoder_layerILj4EE";

static_assert(MKH_RANKS >= mk::cfg::kRanks, "the bind's peer arrays hold the TP world's");
static_assert(kW == 4 && kGrid == 144 && kSmem <= mk::cfg::kSmemOptin,
              "the launch geometry: 36 width-4 clusters at the 216 KB carve");
static_assert(mk::cfg::kStages == int(mk::postattn::kSlots * mk::postattn::kSlotStages),
              "the ring: 20 pack-4 stages as 5 slots");
static_assert(kBlocks1 * (mk::cfg::kWorld.unpool_ksplit ? 4 : 1) <= kGrid &&
                  mk::cfg::kStages <= kHidden / 64 / int(mk::kNextQkvCk),
              "the unpooler GEMM: one column block per width-4 cluster (K in quarters) or per CTA; "
              "the qkv GEMM: the ring fits the CTA's K loop");
static_assert(kPostAttnClusters > int(mk::postattn::kFrontCluster) && kPostAttnClusters >= int(mk::postattn::kQRouter),
              "the post-attention queue holds the front cluster and one wave of router claims");
static_assert(kM <= 64 && kGrid >= int(mk::CombineItem::kOwnerCtas), "one combine round");
static_assert(kGrid == int(mk::kNextQkvCk) * (kQkvRow / 64), "the qkv GEMM's split grid is the launch's");
static_assert((uint64_t(1) << 32) % kBlocks1 == 0, "the ffn_ready count must not wrap");
static_assert(moe_tp::kHRow == !kBf16, "the NVFP4 variant gathers h as the row window");

int64_t plane_stride_min() {
    return std::max<int64_t>({int64_t(kM) * kHidden,
                              int64_t(mk::postattn::kPoolAgOff) + int64_t(kM) * mk::postattn::kPooledRow,
                              int64_t(mk::postattn::kUnpoolAgOff) + int64_t(kM) * kHidden});
}

void pack(const MkLayerBind& b, mkh::Params& p) {
    MKH_REQUIRE(
        b.rank >= 0 && b.rank < mk::cfg::kRanks && b.plane_stride % 8 == 0 && b.plane_stride >= plane_stride_min(),
        "rank / AR plane stride");
    MKH_REQUIRE(b.kv_rows > 0 && b.page_stride > 0 && b.nk_page_stride > 0, "KV extents");
    MKH_REQUIRE(b.nk_splits == 0 || (b.nk_splits == kNextKvSplits && !kTail), "next-layer KV splits");
    MKH_REQUIRE(b.nk_window > 16, "the next-layer KV window must reach past the window's own rows");
    MKH_REQUIRE(!b.prefetch_experts || !kBf16, "the expert prefetch reads the NVFP4 packs");
    MKH_REQUIRE(b.zb_par_slots >= int64_t(moe_tp::kMaxSlots), "zb parity planes hold the MoE variant's slots");
    MKH_REQUIRE(b.b_live >= 1 && b.b_live <= kM, "live rows: 1 <= b_live <= the window's rows");
    MKH_REQUIRE((b.pooled_ready != 0) == (kPooledReadyWords > 0), "the pooled-ready words (the K-window variant's)");
    auto* const wd = ptr<uint32_t>(b.words);
    auto* const cw = ptr<uint32_t>(b.next_layer_words);
    const uint64_t x_attn = b.attn_arena + acfg::kOff_x_attn;

    // ---- post-attention: o_proj + lamport all-reduce + norm ----
    const mk::PostAttnArgs args = {
        .n_clusters = uint32_t(kPostAttnClusters),
        .queue_len = mk::postattn::Queue::kLen,
        .h3_ready = wd,
        .route_count = wd + 1,
        .launch_ctr = wd + kWords,
        .stamps = nullptr,
        .pooled_ready = ptr<uint64_t>(b.pooled_ready),
        .peer_bases = ptr<const uint64_t>(b.peer_bases),
        .plane_stride = b.plane_stride,
        .rank = uint32_t(b.rank),
        .stream_in = ptr<const float>(b.stream_in),
        .alpha = ptr<const float>(b.alpha),
        .beta = ptr<const float>(b.beta),
        .prenorm_w = ptr<const nv_bfloat16>(b.prenorm_w),
        .stream_out = ptr<float>(b.stream_out),
        .h3 = ptr<nv_bfloat16>(b.h3),
        .h3f = ptr<float>(b.h3f),
        .front_done = wd + 2,
        .nq_done = wd + 3,
    };
    p.add(args);
    p.add(bf16_map(b.w_o, kOprojK, kHidden, 64));    // tmap_wo
    p.add(bf16_map(x_attn, kOprojK, kM, kM));        // tmap_xa
    p.add(bf16_map(b.w_p, kHidden, kPooledN, 64));   // tmap_wp
    p.add(bf16_map(b.h3, kHidden, kM, kM));          // tmap_xh
    p.add(bf16_map(b.h3, kPooledN, kM, kM, false));  // tmap_pooled: only descriptor-prefetched, over any bf16 extent

    // ---- pooler || router -> top-k -> pre-expert norm + quant -> front ----
    auto* const ctr = ptr<uint32_t>(b.counters);
    const mk::RouterArgs ra = {
        .ws = ptr<float>(b.ws),
        .counters = ctr,
        .arrival = ctr + 2 * kRouterTiles,
        .logits = ptr<float>(b.logits),
        .split = uint32_t(kRSplit),
        .k_blocks = mk::postattn::kRouterKBlocks,
        .w_hint = ptx::kEvictFirst,  // the router W is read once per step
        .bias = ptr<const float>(b.expert_bias),
        .router_mult = float(b.router_mult),
        .route_scale = float(b.route_scale),
        .num_local = uint32_t(b.num_local),  // the local view's shard size (written, read by nothing here)
        .topk_idx = ptr<int64_t>(b.topk_idx),
        .topk_local = ptr<int64_t>(b.topk_idx_local),
        .topk_w = ptr<float>(b.topk_w),
        .topk_idx32 = ptr<int32_t>(b.topk_idx32),
        .nq =
            {
                .pooled = nullptr,
                .gamma = ptr<void>(b.gamma),
                .eps = float(b.eps),
                .q = ptr<uint8_t>(b.q),
                .sf = ptr<uint8_t>(b.sf),
                .x_scale = ptr<float>(b.x_scale),
                .pooled_ready = nullptr,
                .arrivals = 0,
            },
    };
    p.add(ra);
    const moe_tp::RouteTables route{ptr<int32_t>(b.rt_counts),   ptr<int32_t>(b.rt_offsets),
                                    ptr<int32_t>(b.rt_slot_tok), ptr<int32_t>(b.rt_slot_k),
                                    ptr<int32_t>(b.rt_tok_slot), ptr<int32_t>(b.rt_tile_e),
                                    ptr<int32_t>(b.rt_tile_r0),  ptr<int32_t>(b.rt_n_tiles)};
    const mk::MoeFrontItem<true>::Args fa = {
        .topk_idx = ptr<const int32_t>(b.topk_idx32),
        .route = route,
        .B = kM,
        .b_live = int(b.b_live),
    };
    p.add(fa);
    // the router: W (tf32 boxes of 32 x the router rows) and x = h3f
    p.add(tma_2d(b.w_r, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kHidden, kExperts, 32, int(mk::postattn::kRouterRows),
                 true));
    p.add(tma_2d(b.h3f, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 4, kHidden, kM, 32, kM, true));

    // ---- the MoE phase (this image's MoE variant) ----
    moe_tp::LayerArgs la;
    std::memset(&la, 0, sizeof(la));
    la.topk_idx = ptr<const int32_t>(b.topk_idx32);
    la.topk_w = ptr<const float>(b.topk_w);
    la.x_fp4 = ptr<const uint8_t>(b.q);
    la.x_sf = ptr<const uint8_t>(b.sf);
    la.x_scale = ptr<const float>(b.x_scale);
    la.b_live = int(b.b_live);
    la.B = kM;
    la.w1_fp4 = ptr<const uint8_t>(b.w1);
    la.w1_sf = nullptr;
    la.w1_sf_blk = ptr<const uint8_t>(b.w1_sf_blk);
    la.w2_fp4 = ptr<const uint8_t>(b.w2);
    la.w2_sf = ptr<const uint8_t>(b.w2_sf);
    la.g1_alphas = ptr<const float>(b.g1_alphas);
    la.g1_scale_c = ptr<const float>(b.g1_scale_c);
    la.g2_alphas = ptr<const float>(b.g2_alphas);
    la.route = route;
    la.z_fp4 = ptr<uint8_t>(b.z);
    la.z_sf = ptr<uint8_t>(b.z_sf);
    la.zb = ptr<__nv_bfloat16>(b.zb);
    la.zb_par_slots = int(b.zb_par_slots);
    for (int r = 0; r < MKH_RANKS; ++r)
        MKH_REQUIRE((b.z_peer[r] != 0) == (r < mk::cfg::kRanks) && (b.arrive_peer[r] != 0) == (r < mk::cfg::kRanks),
                    "the peers are the image's world's: one per rank of -DMK_WORLD, zero past it");
    for (int r = 0; r < mk::cfg::kRanks; ++r) {
        la.z_peer[r] = ptr<uint8_t>(b.z_peer[r]);
        la.zsf_peer[r] = ptr<uint8_t>(b.zsf_peer[r]);
        la.zb_peer[r] = ptr<__nv_bfloat16>(b.zb_peer[r]);
        la.arrive_peer[r] = ptr<uint32_t>(b.arrive_peer[r]);
        if constexpr (mk::kHRow) la.zex_peer[r] = ptr<uint8_t>(b.zex_peer[r]);
    }
    if constexpr (mk::kHRow) la.z_ex = ptr<uint8_t>(b.z_ex);
    la.arrive_self = ptr<uint32_t>(b.arrive);
    la.gsync = ptr<uint32_t>(b.gsync);
    la.epoch_dev = ptr<uint32_t>(b.epoch);
    la.clk = nullptr;
    la.rank = int(b.rank);
    la.n_sm = 0;
    la.sf_img = nullptr;
    la.norm_gamma_p1 = ptr<const __nv_bfloat16>(b.gamma_p1);
    la.norm_eps = kPostNormEps;
    la.y = ptr<__nv_bfloat16>(b.y);
    moe_tp::impl_host_init_tmaps(la);
    p.add(la);

    // ---- the attention phase ----
    const mk::AttnArgs attention{
        .arena = ptr<uint8_t>(b.attn_arena),
        .fctr = ptr<uint32_t>(b.attn_fctr),
        .sm_epoch = ptr<uint64_t>(b.attn_sm_epoch),
        .prof_buf = nullptr,
        .k_cache = ptr<uint8_t>(b.k_cache),
        .v_cache = ptr<uint8_t>(b.v_cache),
        .attn_page_table = ptr<const int64_t>(b.page_table),
        .attn_kv_len = ptr<const int32_t>(b.kv_len),
        .attn_page_stride = uint32_t(b.page_stride),
        .attn_window = acfg::kAttnWindow,
    };
    if constexpr (kGlobalAttn) {
        // the global attention streams KV through TMA: per-row maps and their block-grain variants (the block
        // zero-filled first: the maps' 64-B alignment leaves padding behind the attention args)
        const int rows = int(b.kv_rows), box = int(mk::attn::kGlobalTcN);
        mk::GlobalAttention::Args attn_args{};
        attn_args.a = attention;
        attn_args.tk = bf16_map(b.k_cache, kKvCols, rows, 1);
        attn_args.tv = bf16_map(b.v_cache, kKvCols, rows, 1);
        attn_args.tkb = bf16_map(b.k_cache, kKvCols, rows, box);
        attn_args.tvb = bf16_map(b.v_cache, kKvCols, rows, box);
        p.add(attn_args);
    } else {
        p.add(attention);
    }

    // ---- the expert prefetch: last step's routing of this layer over this layer's packs ----
    mk::expert_prefetch::Args prefetch{};
    if (b.prefetch_experts) {
        prefetch.ids = ptr<const int32_t>(b.topk_idx32);
        prefetch.route_count = wd + 1;
        prefetch.weights[0] = ptr<const uint8_t>(b.w1);
        prefetch.weights[1] = ptr<const uint8_t>(b.w1_sf_blk);
        prefetch.weights[2] = ptr<const uint8_t>(b.w2);
        prefetch.weights[3] = ptr<const uint8_t>(b.w2_sf);
    }
    p.add(prefetch);

    // ---- the next-layer phase: combine, unpooler GEMM, cross-layer norm, the successor's qkv GEMM + rowops ----
    constexpr int kK1 = int(moe_tp::kH), kN1 = kHidden, kK2 = kHidden;
    const mk::NextLayerArgs cargs = {
        .stages1 = uint32_t(kK1 / 64),
        .blocks1 = uint32_t(kBlocks1),
        .stages2 = uint32_t(kK2 / 64),
        .n1 = uint32_t(kN1),
        .out1 = nullptr,
        .ffn_ready = cw,
        .launch_ctr = cw + kNextLayerWords,
        .out2 = ptr<nv_bfloat16>(b.qkv_out),
        .n2 = uint32_t(kQkvRow),
        .ag_bases = ptr<const uint64_t>(b.peer_bases),
        .ag_rank = uint32_t(b.rank),
        .ag_off = mk::postattn::kUnpoolAgOff,
    };
    p.add(cargs);
    p.add(bf16_map(b.w_up, kK1, kN1, 64));               // tmap_w1
    p.add(bf16_map(b.y, kK1, kM, kM));                   // tmap_x1: y's first kM rows
    p.add(bf16_map(b.w_qkv, kK2, kQkvRow, 64));          // tmap_w2
    p.add(bf16_map(b.h3q, kK2, kM, kM));                 // tmap_x2
    p.add(bf16_map(b.qkv_out, kQkvRow, kM, kM, false));  // tmap_out2
    const mk::NextLayerTileEpi::Args tepi = {
        .cos_sin = ptr<const float>(b.cos_sin),
        .cache_loc = ptr<const int32_t>(b.cache_loc),
        .k_cache = ptr<uint8_t>(b.next_k_cache),
        .v_cache = ptr<uint8_t>(b.next_v_cache),
        .temp_scale = ptr<const float>(b.temp_scale),
        .q_scale = float(b.q_scale),
        .eps = float(b.qkv_eps),
    };
    p.add(tepi);
    const mk::NextLayerNormItem::Args citem = {
        .branch = nullptr,                            // the next-layer phase points it at the gathered row
        .stream_in = ptr<const float>(b.stream_out),  // this layer's post-attention residual
        .alpha = ptr<const float>(b.c_alpha),
        .beta = ptr<const float>(b.c_beta),
        .pre_w = ptr<const nv_bfloat16>(b.c_pre_w),
        .post_w = ptr<const nv_bfloat16>(b.c_post_w),
        .stream_out = ptr<float>(b.c_stream_out),
        .h3 = ptr<nv_bfloat16>(b.h3q),
        .h3_ready = cw + 1,
        .launch_ctr = cw + kNextLayerWords,
    };
    p.add(citem);
    const mk::CombineItem::Args comb = {
        .tok_slot = route.tok_slot,
        .topk_w = la.topk_w,
        .zb = la.zb,
        .zb_par_slots = uint32_t(la.zb_par_slots),
        .epoch_dev = la.epoch_dev,
        .gamma_p1 = la.norm_gamma_p1,
        .y = la.y,
        .y_ready = cw + 2,
        .gsync = la.gsync,
        .B = kM,
        .b_live = int(b.b_live),
        .eps = kPostNormEps,
    };
    p.add(comb);
    const mk::NextKvPrefetch nextkv = {
        .page_table = ptr<const int64_t>(b.nk_page_table),
        .kv_len = ptr<const int32_t>(b.nk_kv_len),
        .k_cache = ptr<const uint8_t>(b.next_k_cache),
        .v_cache = ptr<const uint8_t>(b.next_v_cache),
        .page_stride = uint32_t(b.nk_page_stride),
        .splits = uint32_t(b.nk_splits),
        .window = uint32_t(b.nk_window),
    };
    p.add(nextkv);
}

}  // namespace

MKH_API int mk_layer_constants(MkLayerConstants* out) {
    return mkh::guard([&] {
        MkLayerConstants c{};
        c.global_attn = kGlobalAttn;
        c.bf16 = kBf16;
        c.tail = kTail;
        c.grid = kGrid;
        c.cluster = kW;
        c.block = mk::block::kThreads;
        c.smem = kSmem;
        c.rows = kM;
        c.hidden = kHidden;
        c.moe_hidden = moe_tp::kH;
        c.oproj_k = kOprojK;
        c.qkv_row = kQkvRow;
        c.kv_row_bytes = 2 * kKvCols;
        c.experts = kExperts;
        c.topk = moe_tp::kTopK;
        c.words = kWords + kGrid;
        c.next_layer_words = kNextLayerWords + kGrid;
        c.counters = 2 * kRouterTiles + kM;
        c.ws_floats = int64_t(kRouterTiles) * kRSplit * 2 * kM * 128;
        c.router_split = kRSplit;
        c.fctr_words = mk::kCtrAttnParts + mk::kCtrAttnPartsWords;
        c.sm_epoch_words = mk::AttnPolicy::kGrid;
        c.arena_bytes = acfg::kOff_attn_ws + int64_t(mk::AttnPolicy::kGrid) * mk::attn::kWsRow * 4;
        c.off_x_attn = acfg::kOff_x_attn;
        c.off_qkv_out = acfg::kOff_qkv_out;
        c.plane_stride_min = plane_stride_min();
        c.pooled_ready_words = kPooledReadyWords;
        c.pool_ag_off = mk::postattn::kPoolAgOff;
        c.unpool_ag_off = mk::postattn::kUnpoolAgOff;
        c.moe_rows = moe_tp::kBMax;
        c.moe_slots = moe_tp::kMaxSlots;
        c.moe_tiles = moe_tp::kMaxTiles;
        c.arrive_words = moe_tp::kArriveWords;
        c.gsync_words = moe_tp::kGsyncWords;
        c.zb_ring = moe_tp::kZbRing;
        c.inter = moe_tp::kI;
        c.z_row_bytes = kBf16 ? 2 * moe_tp::kI : moe_tp::kI / 2;
        c.zsf_row_bytes = moe_tp::kI / moe_tp::kSfVec;
        c.nextkv_splits = kTail ? 0 : kNextKvSplits;
        *out = c;
    });
}

MKH_API const char* mk_layer_kernel(void) {
    return kKernelPrefix;
}

MKH_API int mk_layer_bind(MkModule* module, const MkLayerBind* bind, MkLaunch** out) {
    return mkh::guard([&] {
        mkh::Params p;
        pack(*bind, p);
        *out = mkh::make_launch(
            module, kKernelPrefix, p,
            {unsigned(kGrid), mk::block::kThreads, unsigned(kW), unsigned(kSmem), /*persistent=*/true});
    });
}

#elif MK_HOST_LIB == MK_HOST_aux

// Host side of the verify graph's other kernels: the window feeds, the fused entry, layer 0's qkv head, the lm_head
// gather and the gathered softcap.
#include "head.cu"

namespace {

using mkh::bf16_map;
using mkh::ptr;

constexpr int kM = mk::cfg::kWindowM, kStages = mk::cfg::kStages, kPack = mk::cfg::kPack, kCk = mk::cfg::kQkvCk;
constexpr int kQkvRow = 2 * (mk::cfg::kQHeads + mk::cfg::kKvHeads) * 64;
using QkvEpi = mk::TileEpiQkvRowops<mk::cfg::kQHeads, mk::cfg::kKvHeads, kM>;
constexpr unsigned kQkvSmem = mk::dense_smem_layout(kM, kPack, kStages, kCk).size;
constexpr unsigned kPersistSmem = mk::dense_persist_smem_layout(kM, kPack, kStages).size;
constexpr unsigned kThreads = 256;
static_assert(MKH_RANKS >= int(mk::kEntryRanks), "the bind's shard array holds the TP world's");
constexpr uint32_t kEntryU4 = 4;  // 16-B chunks per entry thread (entry_fused<256, 4, kEntryRanks>)

}  // namespace

MKH_API int mk_feeds_bind(MkModule* module, const MkFeedsBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->rows > 0 && b->rows <= 16 && b->r2t_width > 0 && b->kv_row_bytes > 0, "feeds extents");
        mk::WindowFeedsArgs a{};
        a.positions = ptr<const int64_t>(b->positions);
        a.cos_sin_cache = ptr<const float>(b->cos_sin_cache);
        a.out_cache_loc = ptr<const int64_t>(b->out_cache_loc);
        a.swa_out_cache_loc = ptr<const int64_t>(b->swa_out_cache_loc);
        a.seq_lens = ptr<const int64_t>(b->seq_lens);
        a.req_pool_indices = ptr<const int64_t>(b->req_pool_indices);
        a.req_to_token = ptr<const int32_t>(b->req_to_token);
        a.full_to_swa = ptr<const int64_t>(b->full_to_swa);
        a.cos_sin = ptr<float>(b->cos_sin);
        a.loc_full = ptr<int32_t>(b->loc_full);
        a.loc_swa = ptr<int32_t>(b->loc_swa);
        a.kv_len = ptr<int32_t>(b->kv_len);
        a.pt_full = ptr<int64_t>(b->pt_full);
        a.pt_swa = ptr<int64_t>(b->pt_swa);
        a.m = uint32_t(b->rows);
        a.d = uint32_t(b->head_dim);
        a.w = uint32_t(b->r2t_width);
        a.kv_row_bytes = uint32_t(b->kv_row_bytes);
        a.r2t_stride = b->r2t_width;
        a.r2t_rows = b->r2t_rows;
        a.cos_sin_rows = b->cos_sin_rows;
        a.swa_map_len = b->swa_map_len;
        mkh::Params p;
        p.add(a);
        const unsigned grid = unsigned((b->r2t_width + kThreads - 1) / kThreads);
        *out = mkh::make_launch(module, "_ZN2mk12window_feedsILj256E", p, {grid, kThreads, 1, 0, false});
    });
}

MKH_API int mk_entry_bind(MkModule* module, const MkEntryBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->hidden == kThreads * kEntryU4 * 8 && b->rows > 0 && b->shard_rows > 0, "entry extents");
        mk::EntryArgs a{};
        a.ids = ptr<const int64_t>(b->ids);
        for (int r = 0; r < int(mk::kEntryRanks); ++r) {
            MKH_REQUIRE(b->embed_peer[r] != 0, "every rank's vocab shard");
            a.embed[r] = ptr<const uint4>(b->embed_peer[r]);
        }
        a.shard_rows = b->shard_rows;
        a.pre_w = ptr<const nv_bfloat16>(b->pre_w);
        a.residual = ptr<float>(b->residual);
        a.normed = ptr<nv_bfloat16>(b->normed);
        a.eps_embed = float(b->eps_embed);
        a.eps_pre = float(b->eps_pre);
        mkh::Params p;
        p.add(a);
        *out = mkh::make_launch(module, "_ZN2mk11entry_fusedI", p, {unsigned(b->rows), kThreads, 1, 0, false});
    });
}

MKH_API int mk_qkv_bind(MkModule* module, const MkQkvBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->hidden % 64 == 0 && kStages <= b->hidden / 64 / kCk, "qkv K loop");
        const int k = int(b->hidden);
        mkh::Params p;
        p.add(mk::DenseArgs{.stages_total = uint32_t(k / 64)});
        p.add(bf16_map(b->w, k, kQkvRow, 64));
        p.add(bf16_map(b->x, k, kM, kM));
        p.add(bf16_map(b->out, kQkvRow, kM, kM, false));
        p.add(QkvEpi::Args{
            .cos_sin = ptr<const float>(b->cos_sin),
            .cache_loc = ptr<const int32_t>(b->cache_loc),
            .k_cache = ptr<uint8_t>(b->k_cache),
            .v_cache = ptr<uint8_t>(b->v_cache),
            .temp_scale = ptr<const float>(b->temp_scale),
            .q_scale = float(b->q_scale),
            .eps = float(b->eps),
        });
        // one 64-row W block per CTA pair, K split across the pair's cluster
        *out = mkh::make_launch(module, "_ZN2mk10dense_gemmI", p,
                                {unsigned(kCk * kQkvRow / 64), kThreads, unsigned(kCk), kQkvSmem, false});
    });
}

MKH_API int mk_lm_head_bind(MkModule* module, const MkLmHeadBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        const int k = int(b->hidden), n = int(b->n_out);
        MKH_REQUIRE(k % 64 == 0 && kStages <= k / 64 && b->n_pad >= n && b->n_pad % 64 == 0 && n % 8 == 0,
                    "lm_head shape grain");
        MKH_REQUIRE(b->v_total % n == 0 && b->col0 % n == 0 && b->col0 % 8 == 0 && b->col0 + n <= b->v_total,
                    "the gathered row is every rank's shard, in 16-B packets");
        MKH_REQUIRE(b->arrive_off % 4 == 0 && b->arrive_off >= int64_t(kM) * b->v_total * int64_t(sizeof(float)),
                    "the arrival word sits past the row");
        MKH_REQUIRE(b->mc_base != 0 && b->mc_base % 16 == 0, "the rows' multicast base");
        const int blocks = int(b->n_pad / 64);
        MKH_REQUIRE(b->grid >= 1 && b->grid <= blocks, "persistent grid vs the block loop");
        mkh::Params p;
        p.add(mk::DensePersistArgs{
            .stages_total = uint32_t(k / 64), .num_blocks = uint32_t(blocks), .n_out = uint32_t(n)});
        p.add(bf16_map(b->w, k, n, 64));
        p.add(bf16_map(b->x, k, kM, kM));
        p.add(mk::PersistGather::Args{
            .mc_base = b->mc_base,
            .arrive_off = uint64_t(b->arrive_off),
            .grid_done = ptr<uint32_t>(b->grid_done),
            .v_total = uint32_t(b->v_total),
            .col0 = uint32_t(b->col0),
        });
        *out = mkh::make_launch(module, "_ZN2mk20dense_persist_gatherI", p,
                                {unsigned(b->grid), kThreads, 1, kPersistSmem, false});
    });
}

MKH_API int mk_softcap_bind(MkModule* module, const MkSoftcapBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->numel > 0 && b->numel % 8 == 0 && b->cap > 0.0 && b->per_launch > 0, "softcap operands");
        const uint32_t num_c8 = uint32_t(b->numel / 8);
        mkh::Params p;
        p.add(ptr<const float4>(b->x));
        p.add(ptr<float4>(b->out));
        p.add(num_c8);
        p.add(float(b->scale));
        p.add(float(b->cap));
        p.add(ptr<const int>(b->arrive));
        p.add(ptr<uint32_t>(b->state));
        p.add(uint32_t(b->per_launch));
        *out = mkh::make_launch(module, "_ZN2mk16softcap_gatheredI", p,
                                {(num_c8 + kThreads - 1) / kThreads, kThreads, 1, 0, false});
    });
}

#else  // the DFlash2 libraries

// The DFlash host TUs' tensor-map encoders (cuTensorMapEncodeTiled, L2 256-B promotion, no OOB fill): a 2D bf16 operand
// in 64-element boxes, the draft KV ring and the attention's partials.

namespace mkh::dflash {

// A row-major bf16 [outer][row_elems] operand, its first `inner` columns as 64 x box_outer boxes.
inline CUtensorMap bf16_map(uint64_t base, uint64_t inner, uint64_t outer, uint32_t box_outer, uint64_t row_elems,
                            bool swizzle = true) {
    MKH_REQUIRE(base != 0 && base % 16 == 0, "tensor map over a null or unaligned base");
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {row_elems * 2};
    const cuuint32_t box[2] = {64, box_outer};
    const cuuint32_t es[2] = {1, 1};
    CUtensorMap m;
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr<void>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled");
    return m;
}

// A GEMM unit's x [rows][k] bf16 as [k / 64 k-blocks][rows][64]: one box of `pack` k-blocks x m rows lands as `pack`
// consecutive [m][64] SWIZZLE_128B tiles (one ring slot's x stages).
inline CUtensorMap x_map(uint64_t base, uint64_t k, uint64_t rows, uint32_t m, uint32_t pack) {
    MKH_REQUIRE(base != 0 && base % 16 == 0 && k % 64 == 0, "a GEMM unit's x");
    const cuuint64_t dims[3] = {64, rows, k / 64};
    const cuuint64_t strides[2] = {k * 2, 128};
    const cuuint32_t box[3] = {64, m, pack};
    const cuuint32_t es[3] = {1, 1, 1};
    CUtensorMap mp;
    check(cuTensorMapEncodeTiled(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr<void>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(x)");
    return mp;
}

}  // namespace mkh::dflash

#if MK_HOST_LIB == MK_HOST_dflash_entry

// Host side of the DFlash draft entry: one image per draft block (-DMK_DFLASH_B=8|16).
#include "dflash/head.cu"

namespace {

using mkh::ptr;
using S = mk::dflash::entry::Shape<MK_DFLASH_B>;

}  // namespace

MKH_API int mk_dflash_entry_constants(MkDflashEntryConstants* out) {
    return mkh::guard([&] {
        *out = MkDflashEntryConstants{
            .rows = MK_DFLASH_B,
            .grid = S::kGrid,
            .block = S::kThreads,
            .smem = S::kSmem,
            .hidden = S::kHidden,
            .features = S::kFeatures,
            .ranks = S::kRanks,
            .fc_n = S::kFcN,
            .conv_proj_n = S::kConvProjN,
            .coeffs = S::kCoeffs,
            .groups = S::kGroups,
            .sym_bytes = S::kSymBytes,
            .ws_floats = S::kWsFloats,
            .ctr_words = S::kCtrWords,
        };
    });
}

MKH_API int mk_dflash_entry_bind(MkModule* module, const MkDflashEntryBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->rank >= 0 && b->rank < S::kRanks && b->shard_rows > 0 && b->eps > 0.0, "entry scalars");
        MKH_REQUIRE(b->sym_local % 16 == 0 && b->sym_mc != 0 && b->sym_mc % 16 == 0, "the symmetric region");
        mk::dflash::entry::Args a{};
        for (int k = 0; k < int(S::kFeatures); ++k) {
            MKH_REQUIRE(b->cap[k] != 0 && b->cap[k] % 16 == 0, "every captured feature, 16-B aligned");
            a.cap[k] = ptr<const float>(b->cap[k]);
        }
        for (int r = 0; r < int(S::kRanks); ++r) {
            MKH_REQUIRE(b->embed_peer[r] != 0, "every rank's vocab shard");
            a.embed[r] = ptr<const uint4>(b->embed_peer[r]);
        }
        // the anchor is one int64 load (it may be a word of the step's words); the rest are 16-B rows or packets
        MKH_REQUIRE(b->anchor != 0 && b->anchor % 8 == 0, "the anchor word, 8-B aligned");
        const uint64_t need[] = {b->w_fc,   b->w_conv_proj, b->in_norm,  b->hid_norm, b->conv_base, b->e_mask,
                                 b->h_mask, b->d_mask,      b->xq_const, b->ctx_n,    b->xq,        b->resid,
                                 b->c_att1, b->h_row,       b->delta,    b->ws,       b->ctr};
        for (const uint64_t p : need)
            MKH_REQUIRE(p != 0 && p % 16 == 0, "a null or unaligned entry operand");
        a.anchor = ptr<const int64_t>(b->anchor);
        a.shard_rows = b->shard_rows;
        a.w_fc = ptr<const nv_bfloat16>(b->w_fc);
        a.w_conv_proj = ptr<const nv_bfloat16>(b->w_conv_proj);
        a.in_norm = ptr<const nv_bfloat16>(b->in_norm);
        a.hid_norm = ptr<const nv_bfloat16>(b->hid_norm);
        a.conv_base = ptr<const nv_bfloat16>(b->conv_base);
        a.e_mask = ptr<const nv_bfloat16>(b->e_mask);
        a.h_mask = ptr<const nv_bfloat16>(b->h_mask);
        a.d_mask = ptr<const nv_bfloat16>(b->d_mask);
        a.xq_const = ptr<const nv_bfloat16>(b->xq_const);
        a.sym = ptr<uint8_t>(b->sym_local);
        a.sym_mc = b->sym_mc;
        a.rank = uint32_t(b->rank);
        a.ctx_n = ptr<nv_bfloat16>(b->ctx_n);
        a.xq = ptr<nv_bfloat16>(b->xq);
        a.resid = ptr<nv_bfloat16>(b->resid);
        a.c_att1 = ptr<nv_bfloat16>(b->c_att1);
        a.h_row = ptr<nv_bfloat16>(b->h_row);
        a.delta = ptr<nv_bfloat16>(b->delta);
        a.ws = ptr<float>(b->ws);
        a.ctr = ptr<uint32_t>(b->ctr);
        a.eps = float(b->eps);
        if (module != nullptr) {
            // the block counts and the gather polls spin on other CTAs: every CTA resident at once (one per SM by smem)
            CUdevice dev;
            mkh::check(cuCtxGetDevice(&dev), "cuCtxGetDevice");
            int sms = 0, smem_sm = 0;
            mkh::check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SM count");
            mkh::check(cuDeviceGetAttribute(&smem_sm, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev),
                       "smem per SM");
            MKH_REQUIRE(int(S::kGrid) <= sms && 2 * int(S::kSmem) > smem_sm,
                        "the entry needs " + std::to_string(S::kGrid) + " co-resident CTAs, one per SM");
        }
        mkh::Params p;
        p.add(a);
        *out = mkh::make_launch(module, "_ZN2mk6dflash11draft_entryI", p, {S::kGrid, S::kThreads, 1, S::kSmem, false});
    });
}

#elif MK_HOST_LIB == MK_HOST_dflash_dense

// Host side of the DFlash draft's dense-executor qkv launch (kernels/dflash/layer.cu): a draft layer's qkv.
#include "dflash/common.cuh"

namespace {

using mkh::ptr;
using mkh::dflash::bf16_map;
namespace df = mk::dflash;

constexpr unsigned kThreads = 256;

template <uint32_t kB>
MkLaunch* qkv_launch(MkModule* module, const MkDflashQkvBind* b) {
    using Q = df::QkvPins<kB>;
    constexpr uint64_t kM = 2 * kB, kN = df::kQkvN;
    const uint64_t k = uint64_t(b->hidden);
    mkh::Params p;
    p.add(mk::DenseArgs{.stages_total = uint32_t(k / 64)});
    // one 64-row W block per 4-CTA cluster, K split over the cluster; the executor's out map (kM / 4 token rows per
    // CTA) is bound, the PAIR fixup stores by generic writes
    p.add(bf16_map(b->w, k, kN, 64, k));
    p.add(bf16_map(b->x, k, kM, kM, k));
    p.add(bf16_map(b->out, kN, kM, kM / 4, kN, false));
    p.add(df::DflashQkvArgs{
        .out = ptr<nv_bfloat16>(b->out),
        .out_stride = df::kQkvN,
        .rows = uint32_t(kM),
        .tickets = ptr<uint32_t>(b->tickets),
        .cos_sin = ptr<const float>(b->cos_sin),
        .slot = ptr<const int32_t>(b->slot),
        .commit = ptr<const int32_t>(b->commit),
        .q_norm = ptr<const nv_bfloat16>(b->q_norm),
        .k_norm = ptr<const nv_bfloat16>(b->k_norm),
        .k_ring = ptr<nv_bfloat16>(b->k_ring),
        .v_ring = ptr<nv_bfloat16>(b->v_ring),
        .eps = float(b->eps),
    });
    return mkh::make_launch(
        module, "_ZN2mk6dflash9draft_qkvI", p,
        {unsigned(4 * kN / 64), kThreads, 4, mk::dense_smem_layout(Q::kM, Q::kPack, Q::kStages, 4).size, false});
}

}  // namespace

MKH_API int mk_dflash_qkv_bind(MkModule* module, const MkDflashQkvBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->block == 8 || b->block == 16, "the draft block is 8 or 16 rows");
        MKH_REQUIRE(b->hidden > 0 && b->hidden % 256 == 0, "K: whole 64-element stages in four equal quarters");
        MKH_REQUIRE(b->w && b->x && b->out && b->tickets && b->cos_sin && b->slot && b->commit && b->q_norm &&
                        b->k_norm && b->k_ring && b->v_ring,
                    "qkv operands");
        *out = b->block == 8 ? qkv_launch<8>(module, b) : qkv_launch<16>(module, b);
    });
}

#elif MK_HOST_LIB == MK_HOST_dflash_attn

// The DFlash draft attention's geometry (kernels/dflash/layer.cu): the draft KV ring, the window, the
// partials and the attention / combine shapes, for either block size.
#include "dflash/layer.cu"

namespace {

namespace dfl = mk::dflash;

unsigned combine_ctas(int64_t block) {
    return block == 8 ? dfl::Shape<8>::kCombineCtas : dfl::Shape<16>::kCombineCtas;
}

}  // namespace

MKH_API int mk_dflash_attn_constants(int64_t block, MkDflashAttnConstants* out) {
    return mkh::guard([&] {
        MKH_REQUIRE(block == 8 || block == 16, "block must be 8 or 16");
        *out = MkDflashAttnConstants{
            .ring_rows = dfl::kRing,
            .head_dim = dfl::kD,
            .q_heads = dfl::kQHeads,
            .kv_heads = dfl::kKvHeads,
            .window = dfl::kWindow,
            .block_keys = dfl::kBlk,
            .attn_ctas = dfl::kAttnCtas,
            .combine_ctas = combine_ctas(block),
            .ws_row = dfl::kWsRow,
        };
    });
}

#elif MK_HOST_LIB == MK_HOST_dflash_layer

// Host side of the draft-layer images (kernels/dflash/layer.cu): packs one launch's Params. Built per image
// with the device TU's defines (-DMK_IMAGE -DMK_DFLASH_B).
#include "dflash/layer.cu"

// the attention's maps
namespace mkh::dflash {

// A draft KV ring [kRing][kRingRow] bf16 as 64 x 64 SWIZZLE_128B boxes: one attention operand chunk per box (a kv
// head's chunks at columns kv head x kD).
inline CUtensorMap ring_map(uint64_t base) {
    namespace dfl = mk::dflash;
    return bf16_map(base, dfl::kRingRow, dfl::kRing, dfl::kBlk, dfl::kRingRow);
}

// The attention's q rows: [block tokens][the rank's qkv heads][128] bf16 (row stride `stride` bytes; the q heads lead
// each row) as one [block][8 heads][64 dims] SWIZZLE_128B box per 64-dim half of a kv-head group (row = token x 8 +
// head).
inline CUtensorMap q_map(uint64_t base, uint64_t stride, uint32_t block) {
    namespace dfl = mk::dflash;
    MKH_REQUIRE(base != 0 && base % 16 == 0 && stride % 16 == 0, "the attention's q rows");
    const cuuint64_t dims[3] = {dfl::kHd, dfl::kQkvHeads, block};
    const cuuint64_t strides[2] = {dfl::kHd * 2, stride};
    const cuuint32_t box[3] = {64, dfl::kGqa, block};
    const cuuint32_t es[3] = {1, 1, 1};
    CUtensorMap m;
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr<void>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(q)");
    return m;
}

// The attention's partials as 64-word fp32 rows, stored in [32 words x 8 B rows] boxes (one O half of a CTA's
// stage).
inline CUtensorMap ws_map(uint64_t base, uint32_t block) {
    namespace dfl = mk::dflash;
    MKH_REQUIRE(base != 0 && base % 16 == 0, "the attention's partials");
    const cuuint64_t dims[2] = {64, cuuint64_t(dfl::kAttnCtas) * dfl::kWsImageRows};
    const cuuint64_t strides[1] = {64 * 4};
    const cuuint32_t box[2] = {32, 8 * block};
    const cuuint32_t es[2] = {1, 1};
    CUtensorMap m;
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr<void>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(ws)");
    return m;
}

}  // namespace mkh::dflash

// The draft layer's packer (kernels/dflash/layer.cu Params), included once by each image's host library after the
// image's table and programs, so a binding is checked against the cubin's own program.

namespace {

using mkh::ptr;
using namespace mk::dflash;
using namespace mk::dflash::layer;
using mkh::dflash::bf16_map;
using Prog = Program<MK_DFLASH_PROG>;
constexpr unsigned kLayerThreads = 384;

// what the program reads: slot masks, each GEMM slot's MMA rows, 4-CTA K split (ck4) and blocks
struct Facts {
    uint32_t gemm = 0, fin = 0, gat = 0, ar = 0, prep = 0, prep_rows = 0;  // prep_rows: the slots a prepare row reads
    bool swiglu = false, qkv = false, qkv0 = false;
    uint32_t m[kGemms] = {}, blocks[kGemms] = {};
    bool ck4[kGemms] = {};
    uint32_t gat_gemm[4] = {};  // the GEMM slot each gather slot finishes
};
constexpr Facts facts() {
    Facts f;
    for (uint32_t u = 0; u < Prog::kUnits; ++u) {
        const Unit x = Prog::unit(u);
        if (x.kind < kArRow) {
            f.gemm |= 1u << x.arg;
            f.m[x.arg] = x.m;
            f.ck4[x.arg] = x.kind == kCk4Gather || x.kind == kCk4Qkv || x.kind == kCk4Prep;
            f.blocks[x.arg] = f.ck4[x.arg] ? x.entries : x.entries * kW;
        }
        if (x.kind == kTileFin) f.fin |= 1u << x.epi;
        if (x.kind == kCk4Gather) {
            f.gat |= 1u << x.epi;
            f.gat_gemm[x.epi] = x.arg;
        }
        if (x.kind == kArRow) f.ar |= 1u << x.arg;
        if (x.kind == kPrepRow) {
            f.prep |= 1u << x.arg;
            f.prep_rows |= 1u << x.arg;
        }
        if (x.kind == kCk4Prep) f.prep |= 1u << x.epi;  // its epilogue prepares through the slot (h, base0, x, ready)
        f.swiglu |= x.kind == kTileSwiglu;
        f.qkv |= x.kind == kCk4Qkv && x.epi == 0;
        f.qkv0 |= x.kind == kCk4Qkv && x.epi == 1;
    }
    return f;
}
constexpr Facts kFacts = facts();

bool has(uint32_t mask, uint32_t i) {
    return (mask >> i) & 1u;
}

}  // namespace

MKH_API int mk_dflash_layer_constants(MkDflashLayerConstants* out) {
    return mkh::guard([&] {
        *out = MkDflashLayerConstants{};
        out->rows = kB;
        out->xm = kXM;
        out->grid = kGrid;
        out->cluster = kW;
        out->block = kLayerThreads;
        out->smem = kSmemBytes;
        out->hidden = kH;
        out->inter = kI;
        out->coef_row = kCoefRow;
        out->slot = kSlot;
        out->ranks = kRanks;
        out->row_arrivals = kRowArrivals;
        out->prep_arrivals = kPrepArrivals;
        out->act_arrivals = kActArrivals;
        out->queue_len = queue_len<Prog>();
        out->gemm_mask = kFacts.gemm;
        out->fin_mask = kFacts.fin;
        out->gat_mask = kFacts.gat;
        out->ar_mask = kFacts.ar;
        out->prep_mask = kFacts.prep;
        out->swiglu = kFacts.swiglu;
        out->qkv = kFacts.qkv;
        for (uint32_t g = 0; g < kGemms; ++g) {
            out->gemm_m[g] = kFacts.m[g];
            out->gemm_blocks[g] = kFacts.blocks[g];
        }
        out->has_attn = kAttnImage;
        if constexpr (kAttnImage) {
            out->combine_ctas = Shape<kB>::kCombineCtas;
            out->head_qkv = kFacts.qkv0;
            out->head_qkv_arrivals = kFacts.qkv0 ? kQkv0Arrivals : 0;
        }
    });
}

namespace {

// Prm is the image's parameter block (Params): a template parameter, so the branches an image discards may name the
// fields only the images with the attention declare
template <class Prm>
void pack(const MkDflashLayerBind* b, Prm& p) {
    for (uint32_t g = 0; g < kGemms; ++g) {
        if (!has(kFacts.gemm, g)) continue;
        const int64_t k = b->gemm_k[g], step = 64;  // K per ring stage
        MKH_REQUIRE(k > 0 && k % step == 0 && (!kFacts.ck4[g] || k % (step * kW) == 0), "GEMM K loop");
        MKH_REQUIRE(b->gemm_x_rows[g] >= kFacts.m[g], "an x buffer holds every row its MMA reads");
        MKH_REQUIRE(b->gemm_block0[g] >= 0 && (b->gemm_block0[g] + kFacts.blocks[g]) * BN <= b->gemm_w_rows[g],
                    "the slot's W blocks lie inside its weight");
        // the world's widths: the program's blocks cover gate_up and the qkv rows whole, and down / o_proj's K is the
        // rank's intermediate / q heads (a weight cut for another world fails here, not in the numbers)
        if (kFacts.swiglu && g == 3)
            MKH_REQUIRE(b->gemm_w_rows[g] == int64_t(2 * kI) && kFacts.blocks[g] * BN == 2 * kI,
                        "gate_up: the rank's 2 x kI rows");
        if (kFacts.swiglu && g == 4) MKH_REQUIRE(b->gemm_k[g] == int64_t(kI), "down: K = the rank's kI");
        if ((kFacts.qkv && g == 6) || (kFacts.qkv0 && g == 8))
            MKH_REQUIRE(b->gemm_w_rows[g] == int64_t(kQkvN) && kFacts.blocks[g] * BN == kQkvN,
                        "qkv: the rank's q, k and v heads");
        MKH_REQUIRE((b->gemm_gate[g] == 0) == (b->gemm_gate_arrivals[g] == 0), "a gate carries its arrivals");
        p.maps[2 * g] = bf16_map(b->gemm_w[g], uint64_t(k), uint64_t(b->gemm_w_rows[g]), BN, uint64_t(k));
        p.maps[2 * g + 1] = mkh::dflash::x_map(b->gemm_x[g], uint64_t(k), uint64_t(b->gemm_x_rows[g]), kFacts.m[g],
                                               kXM == 16 ? stage_pack(16) : stage_pack(kFacts.m[g]));
        p.gemm[g] = GemmArgs{uint32_t(k / step), uint32_t(b->gemm_block0[g]), ptr<const uint32_t>(b->gemm_gate[g]),
                             uint32_t(b->gemm_gate_arrivals[g])};
    }
    for (uint32_t s = 0; s < 2; ++s) {
        if (has(kFacts.fin, s)) {
            MKH_REQUIRE(b->fin_bases[s] != 0 && b->fin_coef[s] != 0 && b->fin_base1[s] != 0, "a finish site");
            p.fin[s] = FinArgs{ptr<const uint64_t>(b->fin_bases[s]), ptr<const nv_bfloat16>(b->fin_coef[s]),
                               ptr<const nv_bfloat16>(b->fin_base1[s]), nullptr};
        }
        if (has(kFacts.ar, s)) {
            MKH_REQUIRE(b->ar_bases[s] != 0 && b->ar_r_in[s] != 0 && b->ar_r_out[s] != 0 && b->ar_w[s] != 0 &&
                            b->ar_h[s] != 0 && b->ar_ready[s] != 0 && b->ar_eps[s] > 0.0,
                        "an all-reduce row slot");
            MKH_REQUIRE((b->ar_rin_gate[s] == 0) == (b->ar_rin_arrivals[s] == 0), "a gate carries its arrivals");
            p.ar[s] = ArArgs{ptr<const uint64_t>(b->ar_bases[s]), ptr<const nv_bfloat16>(b->ar_r_in[s]),
                             ptr<nv_bfloat16>(b->ar_r_out[s]),    ptr<const nv_bfloat16>(b->ar_w[s]),
                             ptr<nv_bfloat16>(b->ar_h[s]),        float(b->ar_eps[s]),
                             ptr<uint32_t>(b->ar_ready[s]),       ptr<const uint32_t>(b->ar_rin_gate[s]),
                             uint32_t(b->ar_rin_arrivals[s])};
        }
        if (has(kFacts.prep, s)) {
            MKH_REQUIRE((b->prep_coef[s] != 0 || !has(kFacts.prep_rows, s)) && b->prep_h[s] != 0 &&
                            b->prep_base0[s] != 0 && b->prep_x[s] != 0 && b->prep_ready[s] != 0,
                        "a prepare slot (the coefficient rows: a prepare row's)");
            MKH_REQUIRE((b->prep_h_gate[s] == 0) == (b->prep_h_arrivals[s] == 0), "a gate carries its arrivals");
            p.prep[s] = PrepArgs{ptr<nv_bfloat16>(b->prep_coef[s]),        ptr<const nv_bfloat16>(b->prep_h[s]),
                                 ptr<const nv_bfloat16>(b->prep_base0[s]), ptr<nv_bfloat16>(b->prep_x[s]),
                                 ptr<const uint32_t>(b->prep_h_gate[s]),   uint32_t(b->prep_h_arrivals[s]),
                                 ptr<uint32_t>(b->prep_ready[s])};
        }
    }
    for (uint32_t s = 0; s < 4; ++s) {
        if (!has(kFacts.gat, s)) continue;
        GatherArgs& a = p.gat[s];
        for (uint32_t d = 0; d < 4; ++d) {
            MKH_REQUIRE(b->gat_dst[4 * s + d] != 0 && b->gat_dst[4 * s + d] % 16 == 0, "a gather destination");
            a.dst[d] = b->gat_dst[4 * s + d];
        }
        a.n_dst = 4;
        a.dst_block0 = uint32_t(b->gat_block0[s]);
        a.tap = nullptr;
        MKH_REQUIRE(b->gat_block0[s] >= 0 && (b->gat_block0[s] + kFacts.blocks[kFacts.gat_gemm[s]]) * BN <= kCoefRow,
                    "the gathered blocks lie inside the row");
    }
    if (kFacts.swiglu) {
        MKH_REQUIRE(b->act != 0 && b->act_ready != 0, "the SwiGLU hook's operands");
        p.swiglu = {ptr<nv_bfloat16>(b->act), kI, kB};
        p.act_ready = ptr<uint32_t>(b->act_ready);
    }
    if (kFacts.qkv) {
        MKH_REQUIRE(b->qkv_out != 0 && b->qkv_tickets != 0 && b->qkv_cos_sin != 0 && b->qkv_slot != 0 &&
                        b->qkv_commit != 0 && b->qkv_q_norm != 0 && b->qkv_k_norm != 0 && b->qkv_k_ring != 0 &&
                        b->qkv_v_ring != 0 && b->qkv_eps > 0.0,
                    "the qkv hook's operands");
        p.qkv = DflashQkvArgs{ptr<nv_bfloat16>(b->qkv_out),
                              kQkvN,
                              2 * kB,
                              ptr<uint32_t>(b->qkv_tickets),
                              ptr<const float>(b->qkv_cos_sin),
                              ptr<const int32_t>(b->qkv_slot),
                              ptr<const int32_t>(b->qkv_commit),
                              ptr<const nv_bfloat16>(b->qkv_q_norm),
                              ptr<const nv_bfloat16>(b->qkv_k_norm),
                              ptr<nv_bfloat16>(b->qkv_k_ring),
                              ptr<nv_bfloat16>(b->qkv_v_ring),
                              float(b->qkv_eps)};
    }
    MKH_REQUIRE(b->launch_ctr != 0 && b->rank >= 0 && b->rank < kRanks && b->phase_offset >= 0, "the launch");
    p.launch_ctr = ptr<uint32_t>(b->launch_ctr);
    p.rank = uint32_t(b->rank);
    p.phase_offset = uint32_t(b->phase_offset);
    p.diag = ptr<uint64_t>(b->diag);
    // the last layer (its rows end at the final norm) writes the draft head's xq rows
    MKH_REQUIRE(MK_DFLASH_PROG != kProgTail || (b->xq != 0 && b->xq % 16 == 0), "the last layer's xq rows");
    p.xq = ptr<uint8_t>(b->xq);
    if constexpr (kAttnImage) {
        constexpr uint32_t kCombine = Shape<kB>::kCombineCtas;
        MKH_REQUIRE(b->attn_q != 0 && b->attn_q_stride >= int64_t(kXRow) * 2 && b->attn_q_stride % 16 == 0,
                    "the attention's q rows");
        MKH_REQUIRE(b->draft_kv_len != 0 && b->attn_word != 0 && b->attn_x_ready != 0, "the attention's words");
        MKH_REQUIRE(b->gemm_gate[0] == b->attn_x_ready && b->gemm_gate_arrivals[0] == int64_t(kCombine),
                    "o_proj's x gate is the attention combine's count");
        MKH_REQUIRE(b->gemm_k[0] == int64_t(kXRow), "o_proj: K = the rank's q heads x head_dim");
        p.attn = DraftAttnArgs{ptr<const int32_t>(b->draft_kv_len), ptr<float>(b->attn_partials),
                               ptr<uint32_t>(b->attn_word)};
        p.x_attn = ptr<nv_bfloat16>(b->gemm_x[0]);
        p.x_ready = ptr<uint32_t>(b->attn_x_ready);
        p.attn_maps[0] = mkh::dflash::ring_map(b->attn_k_ring);
        p.attn_maps[1] = mkh::dflash::ring_map(b->attn_v_ring);
        p.attn_maps[2] = mkh::dflash::ws_map(b->attn_partials, kB);
        p.attn_maps[3] = mkh::dflash::q_map(b->attn_q, uint64_t(b->attn_q_stride), kB);
        if (kFacts.qkv0) {
            // the attention reads this launch's qkv0 rows: its q rows are qkv0's output, its rings qkv0's
            MKH_REQUIRE(b->qkv0_tickets != 0 && b->qkv0_q_norm != 0 && b->qkv0_k_norm != 0 && b->attn_q_ready != 0 &&
                            b->qkv_cos_sin != 0 && b->qkv_slot != 0 && b->qkv_commit != 0 && b->qkv_eps > 0.0,
                        "the first layer's qkv operands");
            MKH_REQUIRE(b->attn_q_stride == int64_t(kQkvN) * 2, "qkv0 writes the attention's q rows");
            p.qkv0 = DflashQkvArgs{ptr<nv_bfloat16>(b->attn_q),
                                   kQkvN,
                                   2 * kB,
                                   ptr<uint32_t>(b->qkv0_tickets),
                                   ptr<const float>(b->qkv_cos_sin),
                                   ptr<const int32_t>(b->qkv_slot),
                                   ptr<const int32_t>(b->qkv_commit),
                                   ptr<const nv_bfloat16>(b->qkv0_q_norm),
                                   ptr<const nv_bfloat16>(b->qkv0_k_norm),
                                   ptr<nv_bfloat16>(b->attn_k_ring),
                                   ptr<nv_bfloat16>(b->attn_v_ring),
                                   float(b->qkv_eps)};
            p.q_ready = ptr<uint32_t>(b->attn_q_ready);
        }
    }
}

}  // namespace

MKH_API int mk_dflash_layer_bind(MkModule* module, const MkDflashLayerBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        Params p{};
        pack(b, p);
        mkh::Params pp;
        pp.add(p);
        *out =
            mkh::make_launch(module, "_ZN2mk6dflash11draft_layerI", pp, {kGrid, kLayerThreads, kW, kSmemBytes, true});
    });
}

#elif MK_HOST_LIB == MK_HOST_dflash_draft_head

// Host side of the draft head (kernels/dflash/head.cu; dflash/tree.cuh: its tree variant).
#include "dflash/head.cu"
#include "dflash/tree.cuh"

namespace {

using mkh::check;
using mkh::ptr;
namespace dfl = mk::dflash;
namespace dh = mk::dflash::head;

constexpr unsigned kSmem = mk::dense_persist_smem_layout(dh::kRows, mk::cfg::kPack, mk::cfg::kStages, false).size;
static_assert(sizeof(dh::PostSmem) <= kSmem, "the post phase's smem fits the body's carve");
static_assert((sizeof(dh::PostSmem) + 15) / 16 * 16 + sizeof(dfl::tree::WalkSmem) <= kSmem,
              "the tree walk's smem fits the carve past the post phase's");
static_assert((sizeof(dh::PostSmem) + 15) / 16 * 16 + (dfl::kTopK - 1) * dfl::kTopK * dfl::kTopK * sizeof(float) <=
                  kSmem,
              "the sampled walk's score rows fit the carve past the post phase's");

// A row-major e4m3 [outer][hidden] operand as 128 x box_outer SWIZZLE_128B boxes (one 128-B K row per box row).
CUtensorMap fp8_map(uint64_t base, uint64_t hidden, uint64_t outer, uint32_t box_outer) {
    MKH_REQUIRE(base != 0 && base % 16 == 0, "tensor map over a null or unaligned base");
    const cuuint64_t dims[2] = {hidden, outer};
    const cuuint64_t strides[1] = {hidden};
    const cuuint32_t box[2] = {dh::kKStage, box_outer};
    const cuuint32_t es[2] = {1, 1};
    CUtensorMap m;
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr<void>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "cuTensorMapEncodeTiled(fp8)");
    return m;
}

}  // namespace

MKH_API int64_t mk_dflash_draft_head_state_words(void) {
    return dh::kStateWords;
}

MKH_API int64_t mk_dflash_draft_head_max_survivors(void) {
    return dh::kMaxSurv;
}

MKH_API int mk_dflash_draft_head_bind(MkModule* module, const MkDflashDraftHeadBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->gamma >= 1 && b->gamma < int64_t(dfl::kTopK), "draft slots 1 .. 15");
        MKH_REQUIRE(b->ranks == mk::cfg::kRanks && b->ranks * b->hproj == dfl::kRank && b->hproj % 2 == 0,
                    "the image's world, its W_P band the selector rank's share (the inbox's W_P words)");
        MKH_REQUIRE(b->vocab_rank % 8 == 0 && b->vocab_rank * b->ranks < (int64_t(1) << dfl::kIdBits),
                    "vocab ids in 18 bits");
        MKH_REQUIRE(b->hidden % 256 == 0 && b->hidden <= int64_t(dh::kMaxHidden) &&
                        mk::cfg::kStages <= b->hidden / int64_t(dh::kKStage),
                    "whole 16-B chunks per lane of a warp dot; the K loop fills the ring");
        MKH_REQUIRE(b->n_pad % 64 == 0 && b->n_pad >= b->vocab_rank, "64-row blocks cover the shard");
        const int64_t group = b->grid / b->gamma;
        MKH_REQUIRE(b->grid <= b->n_pad / 64 && b->grid <= int64_t(dfl::kThreads) && group >= 1,
                    "a block per CTA at least, one head per thread, a CTA per slot");
        MKH_REQUIRE(b->ksurv >= int64_t(dfl::kTopK) && b->ksurv <= int64_t(dh::kMaxSurvLists) && b->ksurv <= b->grid,
                    "survivor lists 16 .. 64, at most one per CTA");
        MKH_REQUIRE(b->rank >= 0 && b->rank < b->ranks, "rank");
        for (int r = 0; r < b->ranks; ++r)
            MKH_REQUIRE(b->inbox_peer[r] != 0, "every rank's inbox");
        for (uint64_t p : {b->w_scale, b->wb, b->y, b->pred_codebook, b->succ_codebook, b->words, b->state, b->part,
                           b->resc, b->hloc, b->cand, b->unary, b->hhat, b->scores, b->pick, b->drafts, b->verify_ids})
            MKH_REQUIRE(p != 0, "draft_head operand");
        MKH_REQUIRE(b->wb % 16 == 0 && b->y % 16 == 0 && b->xq % 16 == 0 && b->hloc % 4 == 0,
                    "16-B rows for the warp dots and the x boxes");
        const auto k = uint64_t(b->hidden);
        mkh::Params p;
        p.add(dh::HeadArgs{
            .w_scale = ptr<const float>(b->w_scale),
            .stages_total = uint32_t(k / dh::kKStage),
            .num_blocks = uint32_t(b->n_pad / 64),
            .vocab_rank = uint32_t(b->vocab_rank),
            .id0 = uint32_t(b->rank * b->vocab_rank),
        });
        p.add(fp8_map(b->wq, k, uint64_t(b->vocab_rank), 64));
        // the box's x rows past gamma read as zeros (TMA out-of-bounds fill)
        p.add(fp8_map(b->xq, k, uint64_t(b->gamma), dh::kRows));
        dh::PostArgs post{};
        for (int r = 0; r < 8; ++r)
            post.inbox_peer[r] = r < b->ranks ? ptr<uint64_t>(b->inbox_peer[r]) : nullptr;
        post.part = ptr<uint64_t>(b->part);
        post.resc = ptr<uint64_t>(b->resc);
        post.hloc = ptr<__nv_bfloat16>(b->hloc);
        post.state = ptr<uint32_t>(b->state);
        post.wb = ptr<const __nv_bfloat16>(b->wb);
        post.y = ptr<const __nv_bfloat16>(b->y);
        post.pred_cb = ptr<const __nv_bfloat16>(b->pred_codebook);
        post.succ_cb = ptr<const __nv_bfloat16>(b->succ_codebook);
        post.words = ptr<const int64_t>(b->words);
        post.cand = ptr<int32_t>(b->cand);
        post.unary = ptr<float>(b->unary);
        post.hhat = ptr<__nv_bfloat16>(b->hhat);
        post.scores = ptr<float>(b->scores);
        post.pick = ptr<int32_t>(b->pick);
        post.drafts = ptr<int64_t>(b->drafts);
        post.verify_ids = ptr<int64_t>(b->verify_ids);
        post.rank = uint32_t(b->rank);
        post.ranks = uint32_t(b->ranks);
        post.gamma = uint32_t(b->gamma);
        post.hproj = uint32_t(b->hproj);
        post.vocab_rank = uint32_t(b->vocab_rank);
        post.hidden = uint32_t(k);
        post.group = uint32_t(group);
        post.ksurv = uint32_t(b->ksurv);
        post.mult = float(b->mult);
        post.cap = float(b->cap);
        p.add(post);
        if (b->tree != 0) {
            MKH_REQUIRE(b->gamma + 1 == int64_t(dfl::tree::kRows), "a tree window has 16 rows");
            MKH_REQUIRE(b->fan_idx != 0 && b->fan_lp != 0 && b->cos_sin != 0 && b->cos_sin_cache != 0 &&
                            b->cos_sin % 16 == 0 && b->cos_sin_cache % 16 == 0 && b->d > 0 && b->d % 4 == 0 &&
                            b->cos_sin_rows >= int64_t(dfl::tree::kRows),
                        "the tree walk's operands: the fan, 16-B aligned rope rows");
            p.add(dfl::tree::TreeArgs{.table = ptr<int32_t>(b->tree),
                                      .fan_idx = ptr<int32_t>(b->fan_idx),
                                      .fan_lp = ptr<float>(b->fan_lp),
                                      .cos_sin = ptr<float>(b->cos_sin),
                                      .cos_sin_cache = ptr<const float>(b->cos_sin_cache),
                                      .cos_sin_rows = b->cos_sin_rows,
                                      .d = uint32_t(b->d)});
        }
        if (b->samp != 0) {
            MKH_REQUIRE(b->tree == 0 && b->walk_q != 0 && b->walk_path != 0,
                        "a sampled head walks the linear chain into its q rows and draws");
            p.add(dfl::sample::WalkArgs{
                .samp = ptr<const int64_t>(b->samp), .q = ptr<float>(b->walk_q), .path = ptr<int32_t>(b->walk_path)});
        }
        *out = mkh::make_launch(module, "_ZN2mk6dflash4head10draft_head", p,
                                {unsigned(b->grid), dfl::kThreads, 1, kSmem, false});
    });
}

#elif MK_HOST_LIB == MK_HOST_dflash_verify_accept

// Host side of verify_accept (kernels/dflash/verify_accept.cu): the verify's lm_head with the accept in its epilogue,
// and (dflash/tree.cuh) its tree variant's tree_compact.
#include "dflash/verify_accept.cu"
#include "dflash/tree.cuh"
#include "dflash/kv_sync.cu"

namespace {

using mkh::ptr;
using mkh::dflash::bf16_map;
namespace dfl = mk::dflash;
namespace smp = mk::dflash::sample;

// the commit's and the feeds' operands (verify_accept's and sample_accept's binds name them alike)
template <class Bind>
void fill_accept(const Bind* b, dfl::AcceptArgs& a) {
    MKH_REQUIRE(b->b_live >= 1 && b->b_live <= int64_t(dfl::kAcceptRows), "live rows 1 .. 16");
    MKH_REQUIRE(b->ring > 0, "the out ring");
    MKH_REQUIRE(b->d > 0 && b->d % 4 == 0 && b->d <= int64_t(dfl::kMaxD), "target rope rows: float4 chunks, <= 64");
    MKH_REQUIRE(b->cos_sin_cache % 16 == 0 && b->cos_sin % 16 == 0, "16-B aligned target rope rows");
    MKH_REQUIRE(b->cos_sin_rows >= int64_t(dfl::kWindowRows), "target rope rows");
    MKH_REQUIRE(b->dd > 0 && b->dd % 4 == 0 && b->dd <= int64_t(dfl::kMaxDD) && b->dring > 0 &&
                    (b->dring & (b->dring - 1)) == 0 && b->dcos_sin % 16 == 0 && b->dcos_sin_cache % 16 == 0 &&
                    b->dcos_sin_rows > 0,
                "the draft feeds: 16-B aligned rope rows <= 128 wide, a power-of-two ring");
    for (uint64_t p : {b->tokens, b->words, b->out_ring, b->cos_sin_cache, b->cos_sin, b->loc_full, b->loc_swa,
                       b->kv_len, b->dcos_sin_cache, b->dcos_sin, b->dslot, b->dkv_len})
        MKH_REQUIRE(p != 0, "the commit's and the feeds' operands");
    a.tokens = ptr<const int64_t>(b->tokens);
    a.words = ptr<int64_t>(b->words);
    a.out_ring = ptr<int32_t>(b->out_ring);
    a.cos_sin_cache = ptr<const float>(b->cos_sin_cache);
    a.cos_sin = ptr<float>(b->cos_sin);
    a.loc_full = ptr<int32_t>(b->loc_full);
    a.loc_swa = ptr<int32_t>(b->loc_swa);
    a.kv_len = ptr<int32_t>(b->kv_len);
    a.dcos_sin_cache = ptr<const float>(b->dcos_sin_cache);
    a.dcos_sin = ptr<float>(b->dcos_sin);
    a.dslot = ptr<int32_t>(b->dslot);
    a.dkv_len = ptr<int32_t>(b->dkv_len);
    a.cos_sin_rows = b->cos_sin_rows;
    a.dcos_sin_rows = b->dcos_sin_rows;
    a.b_live = uint32_t(b->b_live);
    a.ring = uint32_t(b->ring);
    a.d = uint32_t(b->d);
    a.dd = uint32_t(b->dd);
    a.dring = uint32_t(b->dring);
}

constexpr unsigned kPersistSmem =
    mk::dense_persist_smem_layout(mk::cfg::kWindowM, mk::cfg::kPack, mk::cfg::kStages).size;
static_assert((dfl::kTgtPer + dfl::kDrfPer) * dfl::kThreads * 16 <= kPersistSmem, "the accept's staging fits the ring");
static_assert((dfl::kTgtPer + dfl::kDrfPer) * dfl::kThreads * 16 + (dfl::tree::kPath + 1) * 4 <= kPersistSmem,
              "the tree accept's table fits past the staging");

}  // namespace

MKH_API int mk_dflash_verify_accept_bind(MkModule* module, const MkDflashVerifyAcceptBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        const int k = int(b->hidden), n = int(b->n_out);
        MKH_REQUIRE(k % 64 == 0 && mk::cfg::kStages <= k / 64, "the K loop fills the ring");
        MKH_REQUIRE(b->n_pad % 64 == 0 && b->n_pad >= n && n % 8 == 0, "64-row blocks cover the shard");
        MKH_REQUIRE(b->grid >= 1 && b->grid <= b->n_pad / 64, "a block per CTA at least");
        MKH_REQUIRE(b->ranks >= 1 && b->ranks <= 8 && b->rank >= 0 && b->rank < b->ranks, "rank");
        MKH_REQUIRE(int64_t(n) * b->ranks < (int64_t(1) << dfl::kIdBits), "vocab ids in the word's 18 bits");
        MKH_REQUIRE(b->cap > 0.0, "verify_accept operands");
        MKH_REQUIRE((b->accept_override == 0) == (b->n_override == 0), "an override table and its length");
        for (int r = 0; r < b->ranks; ++r)
            MKH_REQUIRE(b->inbox_peer[r] != 0, "every rank's inbox");
        for (uint64_t p : {b->w, b->x, b->part, b->state, b->top1})
            MKH_REQUIRE(p != 0, "verify_accept operand");
        mkh::Params p;
        p.add(mk::DensePersistArgs{
            .stages_total = uint32_t(k / 64), .num_blocks = uint32_t(b->n_pad / 64), .n_out = uint32_t(n)});
        p.add(bf16_map(b->w, uint64_t(k), uint64_t(n), 64, uint64_t(k)));
        p.add(bf16_map(b->x, uint64_t(k), mk::cfg::kWindowM, mk::cfg::kWindowM, uint64_t(k)));
        p.add(dfl::AcceptSink::Args{.id0 = uint32_t(b->rank * n),
                                    .b_live = uint32_t(b->b_live),
                                    .scale = float(b->scale),
                                    .cap = float(b->cap)});
        dfl::AcceptPostArgs post{};
        for (int r = 0; r < 8; ++r)
            post.inbox_peer[r] = r < b->ranks ? ptr<uint64_t>(b->inbox_peer[r]) : nullptr;
        post.part = ptr<uint64_t>(b->part);
        post.state = ptr<uint32_t>(b->state);
        post.rank = uint32_t(b->rank);
        post.ranks = uint32_t(b->ranks);
        dfl::AcceptArgs& a = post.acc;
        fill_accept(b, a);
        a.top1 = ptr<int32_t>(b->top1);
        a.accept_override = ptr<const int32_t>(b->accept_override);
        a.n_override = uint32_t(b->n_override);
        p.add(post);
        if (b->tree != 0) {
            MKH_REQUIRE(b->b_live == int64_t(dfl::tree::kRows), "a tree window: 16 live rows");
            p.add(ptr<int32_t>(b->tree));
        }
        *out = mkh::make_launch(module, "_ZN2mk6dflash13verify_acceptI", p,
                                {unsigned(b->grid), dfl::kThreads, 1, kPersistSmem, false});
    });
}

MKH_API int mk_dflash_tree_compact_bind(MkModule* module, const MkDflashTreeCompactBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        namespace tr = dfl::tree;
        MKH_REQUIRE(b->words != 0 && b->tree != 0 && b->rows != 0 && b->n_swa >= 0 && b->n_full >= 0 && b->n_cap >= 0 &&
                        b->n_swa + b->n_full + b->n_cap >= 1,
                    "tree_compact operands");
        MKH_REQUIRE(b->kv_row_bytes % 16 == 0 && b->cap_row_bytes % (16 * tr::kCapCtas) == 0,
                    "16-B chunks, a capture row in kCapCtas equal parts");
        mkh::Params p;
        p.add(tr::CompactArgs{.words = ptr<const int64_t>(b->words),
                              .table = ptr<const int32_t>(b->tree),
                              .rows = ptr<const uint64_t>(b->rows),
                              .n_swa = uint32_t(b->n_swa),
                              .n_full = uint32_t(b->n_full),
                              .n_cap = uint32_t(b->n_cap),
                              .kv_row_bytes = uint32_t(b->kv_row_bytes),
                              .cap_row_bytes = uint32_t(b->cap_row_bytes)});
        *out = mkh::make_launch(module, "_ZN2mk6dflash4tree12tree_compactI", p,
                                {unsigned(b->n_swa + b->n_full + b->n_cap * tr::kCapCtas), 256, 1, 0, false});
    });
}

MKH_API int64_t mk_dflash_sample_chunks(int64_t vocab) {
    return int64_t(smp::kCtas) * smp::chunks_per_cta(smp::slice_of(uint32_t(vocab)));
}

MKH_API int mk_dflash_sample_accept_bind(MkModule* module, const MkDflashSampleAcceptBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        const auto vocab = uint32_t(b->vocab), slice = smp::slice_of(vocab);
        MKH_REQUIRE(
            b->vocab > 0 && b->vocab % 4 == 0 && b->vocab < (int64_t(1) << 24) && (smp::kCtas - 1) * slice < vocab,
            "float4 rows; every CTA of a row's cluster holds columns");
        MKH_REQUIRE(b->b_live >= 2 && b->block_verification >= 0 && b->block_verification <= 1,
                    "a block of drafts; block verification 0 / 1");
        MKH_REQUIRE(smp::kChunk == 4 * dfl::kThreads && smp::kCtas * smp::chunks_per_cta(slice) <= dfl::kThreads,
                    "a chunk is 4 elements per thread; a row's chunks one per thread");
        for (uint64_t p : {b->logits, b->samp, b->cand, b->walk_q, b->walk_path, b->stats, b->pc, b->chunk, b->state})
            MKH_REQUIRE(p != 0, "sample_accept operand");
        MKH_REQUIRE(b->logits % 16 == 0, "16-B aligned logits rows");
        const unsigned smem = (slice * 4 + 15) / 16 * 16 + sizeof(smp::RadixSmem);
        MKH_REQUIRE(smem >= (dfl::kTgtPer + dfl::kDrfPer) * dfl::kThreads * 16 +
                                smp::kRows * smp::kCtas * smp::chunks_per_cta(slice) * sizeof(float),
                    "the feeds' staging and the decision's row masses fit");
        mkh::Params p;
        p.add(smp::Args{.logits = ptr<const float>(b->logits),
                        .samp = ptr<const int64_t>(b->samp),
                        .cand = ptr<const int32_t>(b->cand),
                        .q = ptr<const float>(b->walk_q),
                        .path = ptr<const int32_t>(b->walk_path),
                        .stats = ptr<float>(b->stats),
                        .pc = ptr<float>(b->pc),
                        .chunk = ptr<float>(b->chunk),
                        .vocab = vocab,
                        .slice = slice,
                        .gamma = uint32_t(b->b_live - 1),
                        .block = uint32_t(b->block_verification)});
        dfl::AcceptPostArgs post{};
        post.state = ptr<uint32_t>(b->state);
        fill_accept(b, post.acc);
        p.add(post);
        *out = mkh::make_launch(module, "_ZN2mk6dflash13sample_accept", p,
                                {unsigned(b->b_live) * smp::kCtas, dfl::kThreads, smp::kCtas, smem, false});
    });
}

MKH_API int mk_dflash_kv_sync_bind(MkModule* module, const MkDflashKvSyncBind* b, MkLaunch** out) {
    return mkh::guard([&] {
        MKH_REQUIRE(b->words != 0 && b->window != 0 && b->state != 0 && b->ring != 0 && b->pool != 0 && b->n_buf >= 1,
                    "draft_kv_sync operands");
        MKH_REQUIRE(b->row_bytes > 0 && b->row_bytes % 16 == 0 && b->ring_rows > 0 &&
                        (b->ring_rows & (b->ring_rows - 1)) == 0 && (b->lag == 0 || b->lag == 1),
                    "16-B rows, a power-of-two ring, lag 0 / 1");
        mkh::Params p;
        p.add(dfl::KvSyncArgs{.words = ptr<const int64_t>(b->words),
                              .window = ptr<const int64_t>(b->window),
                              .state = ptr<int64_t>(b->state),
                              .ring = ptr<const uint64_t>(b->ring),
                              .pool = ptr<const uint64_t>(b->pool),
                              .n_buf = uint32_t(b->n_buf),
                              .row_bytes = uint32_t(b->row_bytes),
                              .ring_rows = uint32_t(b->ring_rows),
                              .lag = uint32_t(b->lag)});
        *out = mkh::make_launch(module, "_ZN2mk6dflash13draft_kv_sync", p, {1, dfl::kThreads, 1, 0, false});
    });
}

#else
#error \
    "runtime.cu: -DMK_HOST=layer|aux|dflash_entry|dflash_dense|dflash_attn|dflash_layer|dflash_draft_head|dflash_verify_accept"
#endif
#endif
