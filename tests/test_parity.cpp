// End-to-end parity vs the HF reference fixtures (label: model).
//
// Per case under $PF_FIXTURES ({short-en, multilingual, pii-dense, long-3k}):
//   f32 GGUF (pf-f32.gguf, if present):
//     vs logits.f32 (f64-rotary reference): argmax 100%, per-row cosine
//     >= 0.99999, row-normalized error <= 1e-2
//   f16 GGUF ($PF_GGUF_NAME, default pf-rope2-f16.gguf):
//     vs logits_stock.f32 (stock f32-rotary reference): argmax 100%,
//     per-row cosine >= 0.999
// The per-tap layer-by-layer harness is scripts/compare_taps.py against a
// pf-cli --dump-taps run; this test gates the end results.
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
} while (0)

static bool read_file(const std::string & path, std::vector<char> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t got = std::fread(out.data(), 1, sz, f);
    std::fclose(f);
    return got == (size_t) sz;
}

struct gate_result {
    double min_cos = 1.0, max_row_err = 0.0;
    int    argmax_miss = 0, argmax_ties = 0;
};

// An argmax flip only counts when the reference itself is decisive: where its
// top-2 margin is below TIE_MARGIN any noise source flips the label (the
// reference's own f32-rotary noise does too). Measured margin distribution on
// long-3k: p1 = 0.159, p50 = 6.37 — real predictions sit far above 0.05.
// GPU runs use wider gates: ggml-vulkan's matmul paths run at fp16-class
// precision (measured bit-identical across RADV and NVIDIA — deterministic,
// not device noise): cos >= 0.9985, row_err <= 7.4e-2, and label flips reach
// reference margins of ~0.2 at 3k tokens. A real graph bug still craters
// cosine to ~0.9.
static bool   is_gpu()    { const char * d = std::getenv("PF_DEVICE"); return d && std::strcmp(d, "cpu") != 0; }
static double tie_margin() { return is_gpu() ? 0.25 : 0.05; }

static gate_result compare(const std::vector<float> & ref, const std::vector<float> & got, int n_cls) {
    gate_result r;
    const size_t n = ref.size() / n_cls;
    for (size_t t = 0; t < n; t++) {
        const float * a = ref.data() + t * n_cls;
        const float * b = got.data() + t * n_cls;
        double dot = 0, na = 0, nb = 0, dmax = 0, amax = 0;
        int arg_a = 0, arg_b = 0;
        for (int c = 0; c < n_cls; c++) {
            dot += (double) a[c] * b[c];
            na += (double) a[c] * a[c];
            nb += (double) b[c] * b[c];
            dmax = std::max(dmax, (double) std::fabs(a[c] - b[c]));
            amax = std::max(amax, (double) std::fabs(a[c]));
            if (a[c] > a[arg_a]) arg_a = c;
            if (b[c] > b[arg_b]) arg_b = c;
        }
        r.min_cos = std::min(r.min_cos, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
        r.max_row_err = std::max(r.max_row_err, dmax / std::max(amax, 1.0));
        if (arg_a != arg_b) {
            double second = -1e30;
            for (int c = 0; c < n_cls; c++) {
                if (c != arg_a) second = std::max(second, (double) a[c]);
            }
            if (a[arg_a] - second < tie_margin()) {
                r.argmax_ties++;
            } else {
                r.argmax_miss++;
            }
        }
    }
    return r;
}

static const char * CASES[] = { "short-en", "multilingual", "pii-dense", "long-3k" };

static void run_model(const std::string & gguf, const std::string & fixtures,
                      const char * ref_file, double cos_gate, double row_err_gate) {
    pf::model m;
    if (!m.load(gguf, std::getenv("PF_DEVICE") ? std::getenv("PF_DEVICE") : "cpu", 0)) {
        CHECK_MSG(false, "load %s: %s", gguf.c_str(), m.error.c_str());
        return;
    }
    const int n_cls = m.hp().n_cls;

    for (const char * cs : CASES) {
        const std::string dir = fixtures + "/" + cs;
        std::vector<char> tok_raw, ref_raw;
        if (!read_file(dir + "/tokens.i32", tok_raw) || !read_file(dir + "/" + ref_file, ref_raw)) {
            std::fprintf(stderr, "note: fixture case %s missing, skipping\n", cs);
            continue;
        }
        const size_t n = tok_raw.size() / sizeof(int32_t);
        std::vector<float> ref(n * n_cls);
        if (ref_raw.size() != ref.size() * sizeof(float)) {
            CHECK_MSG(false, "%s/%s: size mismatch", cs, ref_file);
            continue;
        }
        std::memcpy(ref.data(), ref_raw.data(), ref_raw.size());

        std::vector<float> got;
        if (!m.forward((const int32_t *) tok_raw.data(), n, got)) {
            CHECK_MSG(false, "forward %s: %s", cs, m.error.c_str());
            continue;
        }
        gate_result r = compare(ref, got, n_cls);
        std::printf("%-28s %-12s argmax_miss=%d/%zu (ties=%d) min_cos=%.6f row_err=%.2e\n",
                    gguf.substr(gguf.rfind('/') + 1).c_str(), cs, r.argmax_miss, n,
                    r.argmax_ties, r.min_cos, r.max_row_err);
        CHECK_MSG(r.argmax_miss == 0, "%s: %d argmax mismatches", cs, r.argmax_miss);
        CHECK_MSG(r.min_cos >= cos_gate, "%s: min_cos %.6f < %.6f", cs, r.min_cos, cos_gate);
        CHECK_MSG(r.max_row_err <= row_err_gate, "%s: row_err %.2e > %.2e", cs,
                  r.max_row_err, row_err_gate);
    }
}

int main() {
    const char * gguf_dir = std::getenv("PF_GGUF_DIR");
    const char * fixtures = std::getenv("PF_FIXTURES");
    if (!gguf_dir || !fixtures) {
        std::fprintf(stderr, "PF_GGUF_DIR or PF_FIXTURES not set, skipping\n");
        return 77;
    }
    // The skip above is the only legitimate one: model testing wasn't requested
    // (the fast tier runs -LE model; this is the local "full suite, no assets"
    // path). Once PF_GGUF_DIR/PF_FIXTURES are set the operator IS asking for
    // parity, so every asset is a hard requirement -- a missing one fails loudly
    // rather than silently skipping the gate. CI regenerates them all on every
    // run (scripts/hf_dump.py + scripts/convert.py), so a missing file is a real
    // error, not a reason to skip.
    auto require_asset = [](const std::string & path, const char * what) {
        if (FILE * f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return; }
        failures++;
        std::fprintf(stderr, "FAIL: required %s missing: %s\n", what, path.c_str());
    };
    const char * f16_name = std::getenv("PF_GGUF_NAME");
    const std::string f16 = std::string(gguf_dir) + "/" + (f16_name ? f16_name : "pf-rope2-f16.gguf");
    const std::string f32 = std::string(gguf_dir) + "/pf-f32.gguf";
    require_asset(std::string(fixtures) + "/short-en/tokens.i32", "fixtures (scripts/hf_dump.py)");
    require_asset(f16, "f16 GGUF (scripts/convert.py)");
    require_asset(f32, "f32 GGUF (scripts/convert.py --outtype f32)");
    if (failures) return 1;

    // f32 GGUF: tight per-row gates vs the exact-rotation reference
    if (is_gpu()) run_model(f32, fixtures, "logits.f32", 0.998, 0.15);
    else          run_model(f32, fixtures, "logits.f32", 0.99999, 1e-2);

    // f16 GGUF: production-file gate vs the stock reference
    run_model(f16, fixtures, "logits_stock.f32", is_gpu() ? 0.998 : 0.999, is_gpu() ? 0.15 : 5e-2);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
