// Asserts the known openai-privacy-filter GGUF loads with the expected
// hyperparameters, labels and tensors. Needs PF_GGUF_DIR (label: model).
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

int main() {
    const char * dir = std::getenv("PF_GGUF_DIR");
    if (!dir) {
        std::fprintf(stderr, "PF_GGUF_DIR not set, skipping\n");
        return 77;
    }

    {
        pf::model_file bad;
        CHECK(!bad.open(std::string(dir) + "/does-not-exist.gguf", false));
        CHECK(!bad.error.empty());
    }

    pf::model_file mf;
    if (!mf.open(std::string(dir) + "/pf-rope2-f16.gguf", false)) {
        std::fprintf(stderr, "open failed: %s\n", mf.error.c_str());
        return 1;
    }

    CHECK(mf.hp.n_layer       == 8);
    CHECK(mf.hp.n_embd        == 640);
    CHECK(mf.hp.n_head        == 14);
    CHECK(mf.hp.n_head_kv     == 2);
    CHECK(mf.hp.n_rot         == 64);
    CHECK(mf.hp.n_ff_exp      == 640);
    CHECK(mf.hp.n_expert      == 128);
    CHECK(mf.hp.n_expert_used == 4);
    CHECK(mf.hp.swa_radius    == 128);
    CHECK(mf.hp.n_ctx_orig    == 4096);
    CHECK(mf.hp.n_cls         == 217);
    CHECK(mf.hp.rope_freq_base == 150000.0f);
    CHECK(mf.hp.yarn_factor    == 32.0f);
    CHECK(mf.hp.yarn_beta_fast == 32.0f);
    CHECK(mf.hp.yarn_beta_slow == 1.0f);
    CHECK(mf.hp.rms_eps > 0.9e-5f && mf.hp.rms_eps < 1.1e-5f);
    CHECK(mf.hp.yarn_truncate  == false);  // legacy GGUF: KV absent, default

    CHECK(mf.labels.size() == 217);
    CHECK(mf.labels[0]   == "O");
    CHECK(mf.labels[1]   == "B-ACCOUNTNAME");
    CHECK(mf.labels[216] == "S-ZIPCODE");

    CHECK(mf.tensor("token_embd.weight")          != nullptr);
    CHECK(mf.tensor("cls.output.weight")          != nullptr);
    CHECK(mf.tensor("cls.output.bias")            != nullptr);
    CHECK(mf.tensor("blk.7.ffn_up_exps.weight")   != nullptr);
    CHECK(mf.tensor("blk.0.attn_sinks.weight")    != nullptr);
    CHECK(mf.tensor("output.weight")              == nullptr);  // no LM head

    ggml_tensor * cls = mf.tensor("cls.output.weight");
    CHECK(cls && cls->ne[0] == 640 && cls->ne[1] == 217);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
