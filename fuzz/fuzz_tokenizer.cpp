// libFuzzer: arbitrary bytes -> tokenizer invariants.
//   - never crashes (ASan/UBSan compiled in)
//   - pretokenize ranges exactly partition [0, len)
//   - encode: valid ids, start < end, non-decreasing starts, every byte
//     covered (offsets are widened to UTF-8 char boundaries, so tokens may
//     overlap on a multibyte char but never leave gaps)
// With PF_GGUF set to a loadable GGUF the full encode path runs; unset runs
// pretokenize-only. PF_GGUF set but missing is a hard error (exit 1) — setting
// it requests full-encode fuzzing, so the file has to be there.
#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static pf::tokenizer * g_tok = nullptr;

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    const char * gguf = std::getenv("PF_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "fuzz_tokenizer: PF_GGUF unset, pretokenize-only mode\n");
        return 0;
    }
    // PF_GGUF was set, so full-encode fuzzing was requested: the GGUF is a hard
    // requirement. Exit cleanly (exit 1, not abort -> no core dump) when it's
    // missing, so CI fails loudly instead of silently fuzzing pretokenize-only.
    // CI generates it with scripts/convert.py; a missing file means misconfig.
    // A file that exists but won't load is a real bug, so that path aborts below.
    if (FILE * f = std::fopen(gguf, "rb")) {
        std::fclose(f);
    } else {
        std::fprintf(stderr, "fuzz_tokenizer: PF_GGUF set but missing: %s\n", gguf);
        std::exit(1);
    }
    ggml_context * gctx = nullptr;
    gguf_init_params params = { /*no_alloc =*/ true, &gctx };
    gguf_context * g = gguf_init_from_file(gguf, params);
    if (!g) abort();
    static pf::tokenizer tok;
    if (!tok.load(g)) abort();
    g_tok = &tok;
    gguf_free(g);
    ggml_free(gctx);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const char * text = (const char *) data;

    const auto pre = pf::pretokenize(text, size);
    size_t cursor = 0;
    for (const auto & [b, e] : pre) {
        if ((size_t) b != cursor || e <= b || (size_t) e > size) abort();
        cursor = e;
    }
    if (cursor != size) abort();

    if (!g_tok || size == 0) return 0;
    const auto toks = g_tok->encode(text, size);
    int32_t prev_start = 0;
    int32_t covered = 0;
    for (const auto & t : toks) {
        if (t.id < 0 || t.id >= 200064) abort();
        if (t.start >= t.end || t.start < prev_start || t.end > (int32_t) size) abort();
        prev_start = t.start;
        if (t.start > covered) abort();  // gap
        if (t.end > covered) covered = t.end;
    }
    if (!toks.empty() && covered != (int32_t) size) abort();
    if (toks.empty()) abort();  // non-empty input must produce tokens
    return 0;
}
