// GGUF model file: metadata, hyperparameters, labels, tensor access.
// Format-compatible with the llama.cpp-fork converter output for
// arch "openai-privacy-filter" (the same .gguf loads in both engines).
#pragma once

#include <ggml.h>
#include <gguf.h>

#include <string>
#include <vector>

namespace pf {

struct hparams {
    int32_t n_layer       = 0;   // block_count
    int32_t n_embd        = 0;   // embedding_length
    int32_t n_head        = 0;   // attention.head_count
    int32_t n_head_kv     = 0;   // attention.head_count_kv
    int32_t n_rot         = 0;   // attention.key_length (= head_dim)
    int32_t n_ff_exp      = 0;   // expert_feed_forward_length
    int32_t n_expert      = 0;   // expert_count
    int32_t n_expert_used = 0;   // expert_used_count
    int32_t swa_radius    = 0;   // attention.sliding_window (radius; band diameter = 2*radius)
    int32_t n_ctx_train   = 0;   // context_length
    int32_t n_ctx_orig    = 0;   // rope.scaling.original_context_length
    int32_t n_cls         = 0;   // embedding_length_out (= number of labels)
    float   rms_eps        = 0.f; // attention.layer_norm_rms_epsilon
    float   rope_freq_base = 0.f; // rope.freq_base
    float   yarn_factor    = 0.f; // rope.scaling.factor
    float   yarn_beta_fast = 0.f; // rope.scaling.yarn_beta_fast
    float   yarn_beta_slow = 0.f; // rope.scaling.yarn_beta_slow
    // Model trains YaRN with un-truncated ramp corners; default false matches
    // GGUFs that predate the rope.scaling.yarn_truncate KV (KV overrides).
    bool    yarn_truncate  = false;
};

struct model_file {
    gguf_context * guf = nullptr;
    ggml_context * ctx = nullptr;  // tensor metadata; tensor data iff opened with_data
    hparams                  hp{};
    std::vector<std::string> labels;
    std::string              error;

    model_file() = default;
    model_file(const model_file &) = delete;
    model_file & operator=(const model_file &) = delete;
    ~model_file() { close(); }

    // with_data=false reads metadata only (cheap); true maps tensor data too.
    bool open(const std::string & path, bool with_data);
    void close();

    ggml_tensor * tensor(const char * name) const;   // nullptr if absent
    ggml_tensor * require(const char * name);        // sets error if absent
};

} // namespace pf
