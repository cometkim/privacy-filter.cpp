// libFuzzer: attacker bytes -> model_file::open (metadata path).
// The loader pre-validates the header before gguf_init_from_file; residual
// aborts inside ggml's parser are findings to report upstream (ggml is
// vendored unpatched by design).
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    char path[] = "/tmp/pf_fuzz_gguf_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return 0;
    if (write(fd, data, size) != (ssize_t) size) {
        close(fd);
        unlink(path);
        return 0;
    }
    close(fd);

    pf::model_file mf;
    mf.open(path, /*with_data =*/ false);  // must not crash; failure is fine

    unlink(path);
    return 0;
}
