#include "pf.h"

#include "gguf_loader.h"

#include <cstdlib>
#include <cstring>
#include <string>

struct pf_ctx {
    pf::model_file file;
    std::string    device;
    int            n_threads = 0;
    int32_t        max_forward_tokens = 4096;
    std::string    error;
};

int pf_abi_version(void) { return PF_ABI_VERSION; }

pf_ctx * pf_load(const char * gguf_path, const char * device, int n_threads) {
    if (!gguf_path) return nullptr;
    auto * ctx = new pf_ctx();
    ctx->device    = device ? device : "cpu";
    ctx->n_threads = n_threads;
    // P0: metadata only. Weight realization (CPU zero-copy / Vulkan upload)
    // lands with the forward pass.
    if (!ctx->file.open(gguf_path, /*with_data =*/ false)) {
        ctx->error = ctx->file.error;
    }
    return ctx;
}

void pf_free(pf_ctx * ctx) { delete ctx; }

const char * pf_last_error(const pf_ctx * ctx) {
    if (!ctx) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

void pf_set_window(pf_ctx * ctx, int32_t max_forward_tokens) {
    if (ctx) ctx->max_forward_tokens = max_forward_tokens;
}

static int not_implemented(pf_ctx * ctx) {
    if (ctx) ctx->error = "not implemented yet";
    return -1;
}

int pf_classify(pf_ctx * ctx, const char *, size_t, float, pf_entity ** out, size_t * n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    return not_implemented(ctx);
}

void pf_entities_free(pf_entity * ents, size_t) { free(ents); }

int pf_tokenize(pf_ctx * ctx, const char *, size_t, int32_t ** ids, int32_t ** offsets, size_t * n) {
    if (ids) *ids = nullptr;
    if (offsets) *offsets = nullptr;
    if (n) *n = 0;
    return not_implemented(ctx);
}

int pf_logits(pf_ctx * ctx, const int32_t *, size_t, float ** logits) {
    if (logits) *logits = nullptr;
    return not_implemented(ctx);
}

void pf_buf_free(void * buf) { free(buf); }
