// pf_web.cpp — Emscripten bridge for the privacy-filter C API.
//
// Exposes three functions to JavaScript via EMSCRIPTEN_KEEPALIVE:
//   pf_web_load(model_path)           → 0 on success, 1 on error
//   pf_web_classify(text, threshold)  → JSON string or null
//   pf_web_error()                    → last error string
//
// The model file must be written to Emscripten's MEMFS before pf_web_load
// is called (see the JS wrapper in src/pf-worker.ts).
#include "pf.h"

#include <emscripten/emscripten.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static pf_ctx *g_ctx = nullptr;
static std::string g_error;
static std::string g_json_buf;

extern "C" {

EMSCRIPTEN_KEEPALIVE
int pf_web_load(const char *model_path) {
    if (g_ctx) { pf_free(g_ctx); g_ctx = nullptr; }
    g_error.clear();
    g_ctx = pf_load(model_path, nullptr, 0);
    if (pf_last_error(g_ctx)) {
        g_error = pf_last_error(g_ctx);
        pf_free(g_ctx);
        g_ctx = nullptr;
        return 1;
    }
    return 0;
}

EMSCRIPTEN_KEEPALIVE
const char *pf_web_classify(const char *text, float threshold) {
    if (!g_ctx) {
        g_json_buf = R"({"error":"model not loaded"})";
        return g_json_buf.c_str();
    }

    pf_entity *ents = nullptr;
    size_t n = 0;
    size_t len = std::strlen(text);

    if (pf_classify(g_ctx, text, len, threshold, &ents, &n) != 0) {
        g_error = pf_last_error(g_ctx);
        g_json_buf = R"({"error":")" + g_error + R"("})";
        pf_entities_free(ents, n);
        return g_json_buf.c_str();
    }

    // Build JSON: [{"entity_group":...,"start":N,"end":N,"score":F,"text":"..."},...]
    g_json_buf.clear();
    g_json_buf.reserve(n * 80);
    g_json_buf.push_back('[');
    char num[64];
    for (size_t i = 0; i < n; i++) {
        if (i) g_json_buf.push_back(',');
        g_json_buf += R"({"entity_group":")";
        g_json_buf += ents[i].label;
        g_json_buf += R"(","start":)";
        std::snprintf(num, sizeof(num), "%d", ents[i].start);
        g_json_buf += num;
        g_json_buf += R"(,"end":)";
        std::snprintf(num, sizeof(num), "%d", ents[i].end);
        g_json_buf += num;
        g_json_buf += R"(,"score":)";
        std::snprintf(num, sizeof(num), "%.4f", ents[i].score);
        g_json_buf += num;
        g_json_buf += R"(,"text":")";
        // Escape the matched text for JSON
        for (int j = ents[i].start; j < ents[i].end && j < (int)len; j++) {
            unsigned char c = (unsigned char)text[j];
            switch (c) {
                case '"':  g_json_buf += "\\\""; break;
                case '\\': g_json_buf += "\\\\"; break;
                case '\n': g_json_buf += "\\n";  break;
                case '\r': g_json_buf += "\\r";  break;
                case '\t': g_json_buf += "\\t";  break;
                default:
                    if (c < 0x20) {
                        std::snprintf(num, sizeof(num), "\\u%04x", c);
                        g_json_buf += num;
                    } else {
                        g_json_buf.push_back((char)c);
                    }
            }
        }
        g_json_buf += "\"}";
    }
    g_json_buf.push_back(']');
    pf_entities_free(ents, n);
    return g_json_buf.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char *pf_web_error() {
    return g_error.c_str();
}

EMSCRIPTEN_KEEPALIVE
int pf_web_abi() {
    return pf_abi_version();
}

EMSCRIPTEN_KEEPALIVE
void pf_web_free() {
    if (g_ctx) { pf_free(g_ctx); g_ctx = nullptr; }
}

}  // extern "C"
