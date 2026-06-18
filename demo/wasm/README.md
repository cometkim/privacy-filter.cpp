# privacy-filter.cpp — WebAssembly demo

A browser-only PII / NER scanner that compiles the **original privacy-filter.cpp
engine** (ggml CPU + `pf` library) directly to WebAssembly with SIMD and
pthreads. No Python, no ONNX, no Transformers.js — the same C++ code path that
runs on the Raspberry Pi 5 demo runs natively in your browser tab.

Live: **https://privacy-filter-demo.cometkim.dev**

## How it works

```
Browser ──fetch GGUF──▶ Cache API (IndexedDB) ──▶ Web Worker
                                                        │
                               pf_web_load() ◀─────────┤
                                   │                    │
                          ggml WASM SIMD + pthreads     │
                                   │                    │
                               pf_web_classify()        │
                                   │                    │
                          JSON entities ◀───────────────┤
                                   ▼                    │
                         Highlight + table              │
```

- **WASM SIMD128** — ggml's `arch/wasm/quants.c` provides hand-vectorized
  quantized matmul kernels (384 SIMD intrinsics).
- **Pthreads** — `SharedArrayBuffer` enables 2 worker threads for parallel
  matrix multiplies (requires COOP/COEP headers, served via `_headers`).
- **Streaming** — text is chunked via `Intl.Segmenter` and classified chunk by
  chunk; entities appear progressively as the scan runs.
- **Cache API** — the GGUF model (~1.6 GB) is downloaded once and cached by
  URL; subsequent loads read from disk.

## Prerequisites

- [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html) (emsdk)
- Node.js >= 22
- A Cloudflare account (for deployment; local dev needs no account)

## Build

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh          # compile pf.cpp + ggml → wasm/pf.js + wasm/pf.wasm
npm install
npm run build       # Vite bundle → dist/
```

## Local development

```sh
npm run dev         # Vite dev server on :5174 (no COOP/COEP — single-threaded)
npm run typecheck   # tsc --noEmit
```

For full pthread testing locally, use wrangler (serves the `_headers` file):

```sh
npx wrangler dev    # http://localhost:8787
```

## Deploy

```sh
npm run deploy      # vite build + wrangler deploy
```

Wrangler runs `npm run build` automatically via the `build` config in
`wrangler.jsonc`. Deployment serves `dist/` as static assets on Cloudflare's
edge with COOP/COEP headers from `_headers`.

## Project layout

```
demo/wasm/
├── build.sh          # Emscripten build: CMake + em++ → wasm/pf.{js,wasm}
├── index.html        # UI: dark theme, side-by-side output + entity table
├── public/
│   └── _headers      # COOP/COEP headers for SharedArrayBuffer
├── src/
│   ├── main.ts       # UI orchestration: DOM, highlighting, worker messages
│   └── worker.ts     # WASM module owner: model fetch/cache, streaming classify
├── vite.config.ts    # Vite + copy-static plugin for pf.{js,wasm}
├── wrangler.jsonc    # Cloudflare Workers config (static assets + custom domain)
└── wasm/             # Emscripten output (gitignored, built by build.sh)
    ├── pf.js
    └── pf.wasm
```

## Configuration

- **Model URL** — editable in the UI; persisted in `localStorage`. Any
  privacy-filter GGUF works (q8, f16, multilingual). Different URLs get
  separate Cache API entries.
- **Threshold** — per-scan slider (0–1).
- **Threads** — capped at `min(2, navigator.hardwareConcurrency)`.
- **Chunk size** — 600 chars (balance between streaming visuals and throughput).

## Architecture notes

The C++ bridge (`tools/pf-web.cpp`) exposes three `extern "C"` functions via
`EMSCRIPTEN_KEEPALIVE`:

- `pf_web_load(path, n_threads)` — loads GGUF into ggml
- `pf_web_classify(text, threshold)` — returns JSON array of entities
- `pf_web_error()` — last error string

All JSON construction happens in C++ — the worker just parses and forwards to
the main thread. Byte offsets from `pf_classify` are UTF-8, so the highlighter
uses `TextEncoder`/`TextDecoder` for correct slicing.
