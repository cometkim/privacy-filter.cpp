#include "gguf_loader.h"

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

bool model_file::open(const std::string & path, bool with_data) {
    close();

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
