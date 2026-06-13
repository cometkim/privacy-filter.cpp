// libFuzzer harness over model_file::open under ASan/UBSan: random bytes
// straight into ggml's GGUF parser, to surface memory errors in the load
// path. GGUF files are trusted assets, not an attack surface — ggml may
// abort/FPE on hostile metadata, which is expected here, not a finding.
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
