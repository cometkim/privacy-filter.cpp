#include "gguf_loader.h"

#include <cstdio>
#include <cstring>

namespace pf {

static const char * ARCH = "openai-privacy-filter";

namespace {

// KV readers: set err once on first failure, no-op afterwards (lets the
// caller batch reads and check a single error string).
struct kv {
    const gguf_context * g;
    std::string &        err;

    int64_t find(const char * key, bool required) const {
        int64_t id = gguf_find_key(g, key);
        if (id < 0 && required && err.empty()) {
            err = std::string("missing GGUF key: ") + key;
        }
        return id;
    }
    void u32(const char * key, int32_t & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = (int32_t) gguf_get_val_u32(g, id);
    }
    void f32(const char * key, float & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_f32(g, id);
    }
    void boolean(const char * key, bool & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id >= 0) out = gguf_get_val_bool(g, id);
    }
    std::string str(const char * key, bool required = true) const {
        int64_t id = find(key, required);
        return id >= 0 ? gguf_get_val_str(g, id) : "";
    }
    void strings(const char * key, std::vector<std::string> & out, bool required = true) const {
        int64_t id = find(key, required);
        if (id < 0) return;
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
            if (err.empty()) err = std::string(key) + " is not a string array";
            return;
        }
        const size_t n = gguf_get_arr_n(g, id);
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            out[i] = gguf_get_arr_str(g, id, i);
        }
    }
};

std::string akey(const char * suffix) {
    return std::string(ARCH) + "." + suffix;
}

} // namespace

// Structural pre-validation in front of gguf_init_from_file: ggml is
// vendored unpatched and its parser hard-asserts on hostile metadata (e.g.
// gguf.cpp GGML_ASSERT(!key.empty()) — found by fuzz_gguf), so walk the KV
// stream with bounded seeks (no allocation) and reject malformed files
// before handing them over. Not a security boundary — a hardening layer
// that shrinks the abort surface.
static bool plausible_gguf(const std::string & path, std::string & error) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = "cannot open: " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const int64_t fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    bool ok = true;
    auto fail = [&](const char * why) {
        if (ok) error = std::string("malformed GGUF (") + why + "): " + path;
        ok = false;
        return false;
    };
    auto rd = [&](void * dst, size_t n) {
        return ok && std::fread(dst, 1, n, f) == n ? true : fail("truncated");
    };
    auto skip = [&](uint64_t n) {
        if (!ok) return false;
        if (n > (uint64_t) fsize) return fail("length exceeds file");
        std::fseek(f, (long) n, SEEK_CUR);
        return std::ftell(f) <= fsize ? true : fail("seek past end");
    };
    // scalar sizes by gguf type id; 8=string, 9=array handled separately
    static const int TYPE_SIZE[13] = { 1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8 };
    char key_buf[64];
    auto rd_str_hdr = [&](bool key) {
        key_buf[0] = 0;
        uint64_t len = 0;
        if (!rd(&len, 8)) return false;
        if (key && (len == 0 || len > 4096)) return fail("bad key length");
        if (key && len < sizeof(key_buf)) {
            if (!rd(key_buf, len)) return false;
            key_buf[len] = 0;
            return true;
        }
        return skip(len);
    };

    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        error = "not a GGUF file: " + path;
        std::fclose(f);
        return false;
    }
    rd(&version, 4);
    rd(&n_tensors, 8);
    rd(&n_kv, 8);
    if (ok && (version < 2 || version > 3 ||
               n_tensors > (uint64_t) fsize / 24 || n_kv > (uint64_t) fsize / 16)) {
        fail("implausible header counts");
    }

    for (uint64_t i = 0; ok && i < n_kv; i++) {
        if (!rd_str_hdr(/*key =*/ true)) break;
        uint32_t type = 0;
        if (!rd(&type, 4)) break;
        if (std::strcmp(key_buf, "general.alignment") == 0) {
            // ggml takes offset % alignment: zero or non-power-of-2 is an FPE
            uint32_t a = 0;
            if (type != 4 || !rd(&a, 4)) { fail("bad alignment kv"); break; }
            if (a == 0 || (a & (a - 1)) != 0) { fail("bad alignment value"); break; }
            continue;
        }
        if (type == 8) {
            rd_str_hdr(false);
        } else if (type == 9) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) break;
            if (et > 12 || et == 9 || n > (uint64_t) fsize) { fail("bad array"); break; }
            if (et == 8) {
                for (uint64_t k = 0; ok && k < n; k++) rd_str_hdr(false);
            } else {
                skip(n * TYPE_SIZE[et]);
            }
        } else if (type <= 12) {
            skip(TYPE_SIZE[type]);
        } else {
            fail("bad kv type");
        }
    }
    for (uint64_t i = 0; ok && i < n_tensors; i++) {
        if (!rd_str_hdr(/*key =*/ true)) break;  // tensor names: same bounds
        uint32_t n_dims = 0;
        if (!rd(&n_dims, 4)) break;
        if (n_dims > 4) { fail("bad tensor rank"); break; }
        uint64_t ne[4] = { 1, 1, 1, 1 };
        for (uint32_t d = 0; ok && d < n_dims; d++) rd(&ne[d], 8);
        uint32_t type = 0;
        if (!rd(&type, 4)) break;
        // deprecated/removed type slots have blck_size 0: ggml divides by it
        // when sizing the tensor (FPE found by fuzz_gguf)
        if (type >= GGML_TYPE_COUNT || ggml_blck_size((ggml_type) type) <= 0) {
            fail("bad tensor type");
            break;
        }
        // zero dims FPE in gguf.cpp:681's overflow check (INT64_MAX/ne[1]
        // with ne[1]==0 — zero passes its <0 validation); found by fuzz_gguf
        const uint64_t blck = (uint64_t) ggml_blck_size((ggml_type) type);
        if (ne[0] == 0 || ne[1] == 0 || ne[2] == 0 || ne[3] == 0 ||
            ne[0] % blck != 0 || ne[0] > (uint64_t) fsize * 8 ||
            ne[1] > (uint64_t) fsize || ne[2] > (uint64_t) fsize || ne[3] > (uint64_t) fsize) {
            fail("bad tensor shape");
            break;
        }
        skip(8);  // data offset
    }
    std::fclose(f);
    return ok;
}

bool model_file::open(const std::string & path, bool with_data) {
    close();

    if (!plausible_gguf(path, error)) {
        return false;
    }

    gguf_init_params params = { /*no_alloc =*/ !with_data, /*ctx =*/ &ctx };
    guf = gguf_init_from_file(path.c_str(), params);
    if (!guf) {
        error = "failed to read GGUF: " + path;
        return false;
    }

    kv r{guf, error};

    const std::string arch = r.str("general.architecture");
    if (error.empty() && arch != ARCH) {
        error = "unsupported architecture '" + arch + "' (want " + ARCH + ")";
    }

    r.u32(akey("block_count").c_str(),                          hp.n_layer);
    r.u32(akey("embedding_length").c_str(),                     hp.n_embd);
    r.u32(akey("attention.head_count").c_str(),                 hp.n_head);
    r.u32(akey("attention.head_count_kv").c_str(),              hp.n_head_kv);
    r.u32(akey("attention.key_length").c_str(),                 hp.n_rot);
    r.u32(akey("expert_feed_forward_length").c_str(),           hp.n_ff_exp);
    r.u32(akey("expert_count").c_str(),                         hp.n_expert);
    r.u32(akey("expert_used_count").c_str(),                    hp.n_expert_used);
    r.u32(akey("attention.sliding_window").c_str(),             hp.swa_radius);
    r.u32(akey("context_length").c_str(),                       hp.n_ctx_train);
    r.u32(akey("rope.scaling.original_context_length").c_str(), hp.n_ctx_orig);
    r.u32(akey("embedding_length_out").c_str(),                 hp.n_cls);
    r.f32(akey("attention.layer_norm_rms_epsilon").c_str(),     hp.rms_eps);
    r.f32(akey("rope.freq_base").c_str(),                       hp.rope_freq_base);
    r.f32(akey("rope.scaling.factor").c_str(),                  hp.yarn_factor);
    r.f32(akey("rope.scaling.yarn_beta_fast").c_str(),          hp.yarn_beta_fast);
    r.f32(akey("rope.scaling.yarn_beta_slow").c_str(),          hp.yarn_beta_slow);
    r.boolean(akey("rope.scaling.yarn_truncate").c_str(),       hp.yarn_truncate,
              /*required =*/ false);

    r.strings(akey("classifier.output_labels").c_str(), labels);
    if (error.empty() && (int32_t) labels.size() != hp.n_cls) {
        error = "label count " + std::to_string(labels.size()) +
                " != embedding_length_out " + std::to_string(hp.n_cls);
    }

    if (!error.empty()) {
        close();
        return false;
    }
    return true;
}

void model_file::close() {
    if (ctx) { ggml_free(ctx);  ctx = nullptr; }
    if (guf) { gguf_free(guf);  guf = nullptr; }
    hp = hparams{};
    labels.clear();
    error.clear();
}

ggml_tensor * model_file::tensor(const char * name) const {
    return ctx ? ggml_get_tensor(ctx, name) : nullptr;
}

ggml_tensor * model_file::require(const char * name) {
    ggml_tensor * t = tensor(name);
    if (!t && error.empty()) {
        error = std::string("missing tensor: ") + name;
    }
    return t;
}

} // namespace pf
