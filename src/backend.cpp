#include "backend.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace pf {

namespace {

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// "cuda:1" -> ("cuda", 1); "vulkan" -> ("vulkan", 0); "" -> ("", 0).
void parse_device(const std::string & req, std::string & name, int & index) {
    const size_t colon = req.find(':');
    name  = lower(colon == std::string::npos ? req : req.substr(0, colon));
    index = (colon != std::string::npos) ? std::atoi(req.c_str() + colon + 1) : 0;
}

} // namespace

bool engine_backend::init(const std::string & device_req, int n_threads) {
    release();

    std::string name;
    int         want_idx = 0;
    parse_device(device_req, name, want_idx);

    if (name.empty() || name == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (n_threads <= 0) {
            // ggml's default is 4 threads; matmul-heavy work wants the
            // physical cores (SMT siblings only add contention here)
            n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        }
        ggml_backend_cpu_set_n_threads(be, n_threads);
    } else if (name == "gpu" || name == "cuda" || name == "vulkan") {
        // "gpu" picks the first GPU of whichever backend was compiled in;
        // "cuda"/"vulkan" pin a specific backend when more than one is built.
        // ":N" selects the Nth matching GPU — multi-GPU hosts often enumerate
        // an integrated GPU first.
        const std::string want_reg = (name == "gpu") ? "" : name;
        int gpu_idx = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (!want_reg.empty()) {
                const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
                if (!reg || lower(reg) != want_reg) continue;
            }
            if (gpu_idx++ != want_idx) continue;
            be = ggml_backend_dev_init(dev, nullptr);
            if (be) {
                device = ggml_backend_dev_name(dev);
                break;
            }
        }
        if (!be) {
            error = "no usable '" + name + "' device (built with PF_CUDA/PF_VULKAN? driver present?)";
            return false;
        }
    } else {
        error = "unknown device '" + device_req + "' (want cpu|gpu|cuda|vulkan)";
        return false;
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    return true;
}

void engine_backend::release() {
    if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
    if (be)     { ggml_backend_free(be);     be = nullptr; }
    device.clear();
}

bool engine_backend::is_cpu() const {
    return be && ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

} // namespace pf
