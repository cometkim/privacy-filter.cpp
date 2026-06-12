// pf-cli — inspect and run the privacy-filter model.
//   pf-cli --info <model.gguf>
//   pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu|vulkan]
//          [--threads N] [--dump-taps DIR] [--logits-out FILE]
#include "gguf_loader.h"
#include "model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int info(const char * path) {
    pf::model_file mf;
    if (!mf.open(path, /*with_data =*/ false)) {
        std::fprintf(stderr, "error: %s\n", mf.error.c_str());
        return 1;
    }
    const pf::hparams & h = mf.hp;
    std::printf("arch            openai-privacy-filter\n");
    std::printf("kv pairs        %lld\n", (long long) gguf_get_n_kv(mf.guf));
    std::printf("tensors         %lld\n", (long long) gguf_get_n_tensors(mf.guf));
    std::printf("layers          %d\n", h.n_layer);
    std::printf("d_model         %d\n", h.n_embd);
    std::printf("heads           %d q / %d kv, head_dim %d\n", h.n_head, h.n_head_kv, h.n_rot);
    std::printf("experts         %d, top-%d, ff %d\n", h.n_expert, h.n_expert_used, h.n_ff_exp);
    std::printf("swa radius      %d (band diameter %d)\n", h.swa_radius, 2 * h.swa_radius);
    std::printf("rope            theta %.0f, yarn x%.0f (beta %.0f/%.0f, orig_ctx %d, truncate %s)\n",
                h.rope_freq_base, h.yarn_factor, h.yarn_beta_fast, h.yarn_beta_slow,
                h.n_ctx_orig, h.yarn_truncate ? "true" : "false");
    std::printf("rms_eps         %g\n", h.rms_eps);
    std::printf("labels          %d (%s ... %s)\n", h.n_cls,
                mf.labels.front().c_str(), mf.labels.back().c_str());
    return 0;
}

static bool read_i32(const std::string & path, std::vector<int32_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / sizeof(int32_t));
    size_t got = std::fread(out.data(), sizeof(int32_t), out.size(), f);
    std::fclose(f);
    return got == out.size();
}

static bool dump_taps(const pf::tap_map & taps, int64_t n_tok, const std::string & dir) {
    std::string meta = "{\n \"n_tok\": " + std::to_string(n_tok) + ",\n \"taps\": {";
    bool first = true;
    for (const auto & [name, d] : taps) {
        const bool is_i32 = !d.i32.empty();
        const std::string path = dir + "/" + name + (is_i32 ? ".i32" : ".f32");
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            return false;
        }
        if (is_i32) std::fwrite(d.i32.data(), sizeof(int32_t), d.i32.size(), f);
        else        std::fwrite(d.f32.data(), sizeof(float),   d.f32.size(), f);
        std::fclose(f);
        meta += std::string(first ? "" : ",") + "\n  \"" + name + "\": {\"shape\": [" +
                std::to_string(d.n_rows) + ", " + std::to_string(d.n_cols) +
                "], \"dtype\": \"" + (is_i32 ? "i32" : "f32") + "\"}";
        first = false;
    }
    meta += "\n }\n}\n";
    FILE * f = std::fopen((dir + "/meta.json").c_str(), "wb");
    if (!f) return false;
    std::fwrite(meta.data(), 1, meta.size(), f);
    std::fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--info") == 0) {
        return info(argv[2]);
    }

    std::string model_path, tokens_path, taps_dir, logits_path, device = "cpu";
    int threads = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if      (a == "--model")      model_path  = argv[i + 1];
        else if (a == "--tokens")     tokens_path = argv[i + 1];
        else if (a == "--dump-taps")  taps_dir    = argv[i + 1];
        else if (a == "--logits-out") logits_path = argv[i + 1];
        else if (a == "--device")     device      = argv[i + 1];
        else if (a == "--threads")    threads     = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || tokens_path.empty()) {
        std::fprintf(stderr,
            "usage: pf-cli --info <model.gguf>\n"
            "       pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu|vulkan]\n"
            "              [--threads N] [--dump-taps DIR] [--logits-out FILE]\n");
        return 2;
    }

    std::vector<int32_t> ids;
    if (!read_i32(tokens_path, ids)) {
        std::fprintf(stderr, "cannot read %s\n", tokens_path.c_str());
        return 1;
    }

    pf::model m;
    if (!m.load(model_path, device, threads)) {
        std::fprintf(stderr, "load error: %s\n", m.error.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded %s on %s, %zu tokens\n",
                 model_path.c_str(), m.be.device.c_str(), ids.size());

    pf::tap_map taps;
    std::vector<float> logits;
    if (!m.forward(ids.data(), (int64_t) ids.size(), logits, taps_dir.empty() ? nullptr : &taps)) {
        std::fprintf(stderr, "forward error: %s\n", m.error.c_str());
        return 1;
    }

    if (!taps_dir.empty() && !dump_taps(taps, (int64_t) ids.size(), taps_dir)) {
        return 1;
    }
    if (!logits_path.empty()) {
        FILE * f = std::fopen(logits_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", logits_path.c_str()); return 1; }
        std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
    }

    // print argmax label per token as a quick eyeball
    const int n_cls = m.hp().n_cls;
    for (size_t t = 0; t < ids.size() && t < 32; t++) {
        int best = 0;
        for (int c = 1; c < n_cls; c++) {
            if (logits[t * n_cls + c] > logits[t * n_cls + best]) best = c;
        }
        std::printf("%3zu %-6d %s\n", t, ids[t], m.file.labels[best].c_str());
    }
    return 0;
}
