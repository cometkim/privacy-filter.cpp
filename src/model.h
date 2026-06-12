// Weights, YaRN frequency factors, graph build and forward pass for the
// openai-privacy-filter architecture (port of the llama.cpp-fork graph in
// src/models/openai-privacy-filter.cpp, collapsed to plain ggml).
#pragma once

#include "backend.h"
#include "gguf_loader.h"

#include <map>
#include <vector>

namespace pf {

struct layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr, * bq = nullptr;
    ggml_tensor * wk = nullptr, * bk = nullptr;
    ggml_tensor * wv = nullptr, * bv = nullptr;
    ggml_tensor * wo = nullptr, * bo = nullptr;
    ggml_tensor * sinks = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * router_w = nullptr, * router_b = nullptr;
    ggml_tensor * gate_exps = nullptr, * gate_exps_b = nullptr;
    ggml_tensor * up_exps   = nullptr, * up_exps_b   = nullptr;
    ggml_tensor * down_exps = nullptr, * down_exps_b = nullptr;
};

// transformers _compute_yarn_parameters semantics. Writes n_rot/2 factors
// ff[j] = inv_freq_base[j] / inv_freq_yarn[j] (ggml_rope_ext divides theta by
// ff, so this reproduces the exact YaRN frequencies with ext_factor=0), and
// the cos/sin scale 0.1*ln(factor)+1 as attn_factor.
void yarn_freq_factors(const hparams & hp, std::vector<float> & ff, float & attn_factor);

// Per-token taps captured during forward (name -> row-major [n_tok, D]).
struct tap_data {
    std::vector<float>   f32;
    std::vector<int32_t> i32;
    int64_t              n_rows = 0, n_cols = 0;
};
using tap_map = std::map<std::string, tap_data>;

struct model {
    model_file     file;
    engine_backend be;

    std::vector<layer_weights> layers;
    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * cls_w = nullptr, * cls_b = nullptr;

    std::vector<float> freq_factors;        // [n_rot/2]
    float              attn_factor = 1.0f;

    ggml_backend_buffer_t weights_buf = nullptr;  // CPU zero-copy wrapper or device buffer
    ggml_context *        device_ctx  = nullptr;  // device tensor mirror (non-CPU)

    std::string error;

    bool load(const std::string & gguf_path, const std::string & device, int n_threads);
    void release();
    ~model() { release(); }

    const hparams & hp() const { return file.hp; }

    // Single forward pass over ids[0..n): logits out is [n, n_cls] row-major.
    // If taps != nullptr, capture every named tap (slower: marks them as
    // graph outputs so the allocator cannot reuse their buffers).
    bool forward(const int32_t * ids, int64_t n, std::vector<float> & logits,
                 tap_map * taps = nullptr);

  private:
    bool realize_weights();
    bool map_tensors();
};

// Symmetric sliding-window mask predicate: token pair (p0, p1) is VISIBLE iff
// |p1 - p0| <= radius. Single source of truth, unit-tested at the band edge.
inline bool swa_visible(int64_t p0, int64_t p1, int64_t radius) {
    const int64_t d = p1 > p0 ? p1 - p0 : p0 - p1;
    return d <= radius;
}

// dst is [n, n] row-major: row p1 (query) x col p0 (key), 0 = visible,
// -INF = masked. Matches HF create_bidirectional_sliding_window_mask
// (|q - kv| <= config.sliding_window).
void fill_swa_mask(float * dst, int64_t n, int64_t radius);

} // namespace pf
