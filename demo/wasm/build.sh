#!/usr/bin/env bash
set -euo pipefail

# Build privacy-filter.cpp → WebAssembly with SIMD.
#
# Compiles the same C++ engine (ggml CPU backend + pf library) to WASM using
# Emscripten. The output is a single pf.js + pf.wasm module that runs the
# full inference pipeline (tokenizer + MoE transformer + NER) client-side.
#
# Usage:
#   source /path/to/emsdk/emsdk_env.sh && ./demo/wasm/build.sh
#   PF_MODEL_URL=https://huggingface.co/.../resolve/main/model.gguf \
#     ./demo/wasm/build.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/wasm-build"

source "$EMSDK/emsdk_env.sh" 2>/dev/null || true

mkdir -p "$BUILD_DIR"

# ─── Configure ───────────────────────────────────────────────────────────────
emcmake cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPF_BUILD_TOOLS=OFF \
    -DPF_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_RPC=OFF \
    -DGGML_OPENCL=OFF \
    -DGGML_OPENVINO=OFF \
    -DGGML_CANN=OFF \
    -DGGML_SYCL=OFF \
    -DGGML_HIP=OFF

# ─── Build just the pf static library ────────────────────────────────────────
cmake --build "$BUILD_DIR" -j"$(nproc)" --target pf

# ─── Link the WASM module ────────────────────────────────────────────────────
# Link the pf-web wrapper against the static archives directly. Emscripten
# handles archive extraction; order matters (pf → ggml → ggml-cpu → ggml-base).
em++ \
    "$ROOT_DIR/tools/pf-web.cpp" \
    -I"$ROOT_DIR/include" \
    -I"$ROOT_DIR/ggml/include" \
    -I"$ROOT_DIR/ggml/src" \
    -I"$BUILD_DIR/ggml/include" \
    "$BUILD_DIR/libpf.a" \
    "$BUILD_DIR/ggml/src/libggml.a" \
    "$BUILD_DIR/ggml/src/libggml-cpu.a" \
    "$BUILD_DIR/ggml/src/libggml-base.a" \
    -msimd128 \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s USE_ES6_IMPORT_META=1 \
    -s EXPORT_NAME=PfModule \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=256MB \
    -s MAXIMUM_MEMORY=4GB \
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','UTF8ToString','stringToUTF8','lengthBytesUTF8','FS']" \
    -s EXPORT_ALL=1 \
    -o "$SCRIPT_DIR/wasm/pf.js"

echo "✓ Built: $SCRIPT_DIR/wasm/pf.js + $SCRIPT_DIR/wasm/pf.wasm"
ls -lh "$SCRIPT_DIR/wasm/pf.js" "$SCRIPT_DIR/wasm/pf.wasm"
