#include "backend.h"

#include <ggml-cpu.h>

#include <cstring>

namespace pf {

bool engine_backend::init(const std::string & device_req, int n_threads) {
    release();

    const bool want_vulkan = device_req == "vulkan";
    if (want_vulkan) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                be = ggml_backend_dev_init(dev, nullptr);
                if (be) {
                    device = ggml_backend_dev_name(dev);
                    break;
                }
            }
        }
        if (!be) {
            error = "no usable GPU device (built with PF_VULKAN? driver present?)";
            return false;
        }
    } else if (device_req.empty() || device_req == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (n_threads > 0) {
            ggml_backend_cpu_set_n_threads(be, n_threads);
        }
    } else {
        error = "unknown device '" + device_req + "' (want cpu|vulkan)";
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
