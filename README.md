# privacy-filter.cpp

Minimal [GGML](https://github.com/ggml-org/ggml) inference engine for the
`openai-privacy-filter` token-classification model family
([openai/privacy-filter](https://huggingface.co/openai/privacy-filter),
[OpenMed/privacy-filter-multilingual](https://huggingface.co/OpenMed/privacy-filter-multilingual)):
PII/NER entity spans with exact UTF-8 byte offsets. Stock upstream ggml, no patches.

## Build

```sh
git clone --recursive <repo>
cmake --preset release && cmake --build --preset release -j
```

Presets: `release`, `debug` (ASan+UBSan), `profile` (RelWithDebInfo), `fuzz` (clang libFuzzer).

## Run

```sh
build/release/pf-cli --info path/to/privacy-filter.gguf
```

## Test

```sh
ctest --preset debug -LE model                      # fast suite, no assets
PF_GGUF_DIR=~/models/privacy-filter-multilingual \
  ctest --preset debug -L model                     # parity vs HF reference
```

Status: P0 (bootstrap) — loader + CLI info. Forward pass, tokenizer, NER
decode, Vulkan, bench and fuzz land in subsequent phases.
