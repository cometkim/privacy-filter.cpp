// pf-cli — inspect and (eventually) run the privacy-filter model.
//   pf-cli --info <model.gguf>
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>

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

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--info") == 0) {
        return info(argv[2]);
    }
    std::fprintf(stderr, "usage: pf-cli --info <model.gguf>\n");
    return 2;
}
