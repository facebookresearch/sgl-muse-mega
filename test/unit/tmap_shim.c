// Stands in for the driver's cuTensorMapEncodeTiled in the CPU checks (no GPU, no context; loaded RTLD_GLOBAL ahead of
// a host library so its references bind here): records every encode's arguments and stamps the map's first word with
// the encode's index, so each tensor-map parameter of a packed block names the encode that produced it.
#include <stdint.h>
#include <string.h>

#define TAG 0x7e45a9c3d2b10000ull

typedef struct {
    uint64_t base;
    int32_t dtype, rank, swizzle, l2;
    uint64_t dims[5], strides[4];
    uint32_t box[5];
} Encode;

static Encode g_encodes[1024];
static int g_count;

int cuTensorMapEncodeTiled(void* map, int dtype, uint32_t rank, void* base, const uint64_t* dims,
                           const uint64_t* strides, const uint32_t* box, const uint32_t* elem_strides, int interleave,
                           int swizzle, int l2, int oob) {
    (void)elem_strides;
    (void)interleave;
    (void)oob;
    if (g_count >= 1024 || rank < 1 || rank > 5) return 1;
    Encode* e = &g_encodes[g_count];
    memset(e, 0, sizeof(*e));
    e->base = (uint64_t)(uintptr_t)base;
    e->dtype = dtype;
    e->rank = (int32_t)rank;
    e->swizzle = swizzle;
    e->l2 = l2;
    for (uint32_t i = 0; i < rank; ++i) {
        e->dims[i] = dims[i];
        e->box[i] = box[i];
        if (i + 1 < rank) e->strides[i] = strides[i];
    }
    memset(map, 0, 128);
    const uint64_t tag = TAG | (uint64_t)g_count;
    memcpy(map, &tag, sizeof(tag));
    ++g_count;
    return 0;
}

int shim_count(void) {
    return g_count;
}
void shim_reset(void) {
    g_count = 0;
}
const Encode* shim_get(int i) {
    return &g_encodes[i];
}
