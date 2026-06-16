// pf-bench — tokens/s and per-stage latency at several document lengths.
//   pf-bench <model.gguf> [device] [iters] [lengths]
// Synthesizes PII-shaped text, then per length: tokenize / forward (windowed)
// / decode timings, plus RSS and cold-start (load -> first entity).
// [lengths] is an optional comma-separated list of EXACT token counts (the
// synthesized text is tokenized then truncated to each count) — use it to match
// scripts/bench_torch.py for an apples-to-apples PyTorch comparison. Omitted, it
// defaults to ~{128,512,2048,8192,32768}-token documents.
#include "model.h"
#include "ner.h"
#include "tokenizer.h"

#include <ggml.h>  // ggml_time_us

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static size_t rss_kb(const char * key) {
    FILE * f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t v = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, std::strlen(key)) == 0) {
            std::sscanf(line + std::strlen(key), "%zu", &v);
            break;
        }
    }
    std::fclose(f);
    return v;
}

static std::string make_text(int approx_tokens) {
    std::string out;
    int i = 0;
    while ((int) out.size() < approx_tokens * 4) {  // ~4 bytes/token
        out += "Case " + std::to_string(i) + ": Anna Kowalski reported an issue. "
               "Contact at anna.kowalski" + std::to_string(i) + "@mail.example.com "
               "or +48 123 456 789. Ships to 12 Elm Street, Lyon. "
               "Refund to IBAN DE89 3704 0044 0532 0130 00.\n\n";
        i++;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: pf-bench <model.gguf> [cpu|vulkan] [iters] [len1,len2,...]\n");
        return 2;
    }
    const char * device = argc > 2 ? argv[2] : "cpu";
    const int iters = argc > 3 ? std::atoi(argv[3]) : 3;

    // Optional exact token-count list (4th arg): the synthesized text per length
    // is truncated to exactly this many tokens, so the lengths match whatever is
    // passed to scripts/bench_torch.py --lengths. Empty -> the approximate
    // defaults below.
    std::vector<int> lengths;
    if (argc > 4) {
        for (char * s = std::strtok(argv[4], ","); s; s = std::strtok(nullptr, ","))
            if (int v = std::atoi(s)) lengths.push_back(v);
    }
    const bool exact = !lengths.empty();
    if (!exact) lengths = { 128, 512, 2048, 8192, 32768 };

    // PF_WINDOW: tokens per forward pass (the pf_set_window knob). Longer inputs
    // run as overlapping halo windows; larger W means fewer windows (less halo
    // recompute, faster) but a bigger compute buffer (more RAM/VRAM). Default 4096.
    const int W = std::getenv("PF_WINDOW") ? std::atoi(std::getenv("PF_WINDOW")) : 4096;

    const size_t rss0 = rss_kb("VmRSS:");
    const int64_t t_load0 = ggml_time_us();
    pf::model m;
    if (!m.load(argv[1], device, 0)) {
        std::fprintf(stderr, "load: %s\n", m.error.c_str());
        return 1;
    }
    pf::tokenizer tk;
    if (!tk.load(m.file.guf)) {
        std::fprintf(stderr, "tokenizer: %s\n", tk.error.c_str());
        return 1;
    }
    const int64_t t_load1 = ggml_time_us();
    const size_t rss1 = rss_kb("VmRSS:");

    // cold start: load + first short classification
    {
        const std::string text = make_text(32);
        const auto toks = tk.encode(text.data(), text.size());
        std::vector<int32_t> ids(toks.size());
        for (size_t i = 0; i < toks.size(); i++) ids[i] = toks[i].id;
        std::vector<pf::ner::tok_span> spans;
        std::string err;
        pf::ner::classify_tokens(m, ids.data(), (int) ids.size(), W, 0.5f, spans, err);
    }
    const int64_t t_first = ggml_time_us();

    // weights buffer: device memory (Vulkan VRAM) or the zero-copy CPU wrap.
    const double wbuf_mib = m.weights_buf ? ggml_backend_buffer_get_size(m.weights_buf) / 1048576.0 : 0;
    std::printf("device %s | load %.2fs (+%.0f MiB) | weights %.0f MiB | window %d | %d iters\n\n",
                m.be.device.c_str(), (t_load1 - t_load0) / 1e6,
                (rss1 - rss0) / 1024.0, wbuf_mib, W, iters);
    std::printf("| %8s | %11s | %8s | %9s | %9s |\n",
                "tokens", "forward ms", "tok/s", "cmp MiB", "RSS MiB");
    std::printf("|---------:|------------:|--------:|---------:|--------:|\n");

    for (const int target : lengths) {
        const std::string text = make_text(target);
        int64_t tok_us = 0, fwd_us = 0, dec_us = 0;
        size_t n_tok = 0;
        // iteration -1 is an untimed warm-up: GPU backends compile pipelines
        // lazily per shape class, which would otherwise dominate iteration 0
        for (int it = -1; it < iters; it++) {
            int64_t t0 = ggml_time_us();
            const auto toks = tk.encode(text.data(), text.size());
            int64_t t1 = ggml_time_us();
            std::vector<int32_t> ids(toks.size());
            for (size_t i = 0; i < toks.size(); i++) ids[i] = toks[i].id;
            // exact mode: truncate to the requested count (make_text overshoots)
            if (exact && (int) ids.size() > target) ids.resize(target);
            n_tok = ids.size();

            std::vector<float> emit;
            std::string err;
            if (!pf::ner::emit_logprobs(m, ids.data(), (int) ids.size(), W, emit, err)) {
                std::fprintf(stderr, "forward: %s\n", err.c_str());
                return 1;
            }
            int64_t t2 = ggml_time_us();
            const auto lt = pf::ner::build_label_table(m.file.labels);
            const auto path = pf::ner::bioes_viterbi(lt, emit, (int) n_tok, m.hp().n_cls);
            const auto spans = pf::ner::assemble_spans(lt, path, emit, m.hp().n_cls);
            int64_t t3 = ggml_time_us();
            if (it >= 0) {
                tok_us += t1 - t0;
                fwd_us += t2 - t1;
                dec_us += t3 - t2;
            }
            (void) spans;
        }
        const double fwd_ms = fwd_us / 1e3 / iters;
        // compute buffer: per-forward activation memory (Vulkan VRAM / CPU RAM),
        // sized to one window -> grows with min(n_tok, W). RSS is host resident.
        const double cmp_mib = ggml_gallocr_get_buffer_size(m.be.galloc, 0) / 1048576.0;
        std::printf("| %8zu | %11.1f | %8.0f | %9.0f | %8.0f |\n",
                    n_tok, fwd_ms, n_tok / (fwd_ms / 1e3), cmp_mib, rss_kb("VmRSS:") / 1024.0);
        (void) tok_us; (void) dec_us;
    }
    std::printf("\npeak RSS %.0f MiB | weights %.0f MiB\n",
                rss_kb("VmHWM:") / 1024.0, wbuf_mib);
    return 0;
}
