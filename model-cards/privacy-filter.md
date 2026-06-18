---
license: apache-2.0
base_model: openai/privacy-filter
base_model_relation: quantized
pipeline_tag: token-classification
library_name: gguf
tags:
  - gguf
  - privacy-filter.cpp
  - llama-cpp
  - localai
  - token-classification
  - pii
  - ner
  - privacy
  - redaction
  - openai-privacy-filter
---

# privacy-filter — GGUF (F16 + Q8_0)

GGUF conversion of [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter),
OpenAI's bidirectional PII **token-classification** model. It labels every token with a BIOES
tag over **8 PII categories (33 classes)** in a single forward pass, then decodes coherent
spans with a constrained Viterbi procedure — so it can be served locally with **no Python** as
the encoder/NER tier of a PII redactor.

For the full model description, training, evaluation, operating points, limitations, and
citations, see the **[source model card](https://huggingface.co/openai/privacy-filter)** — this
card only covers the GGUF packaging and how to run it.

> For broader language coverage (54 categories across 16 languages), see the multilingual
> fine-tune [`privacy-filter-multilingual` GGUF](https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF).

## Runtimes

This GGUF uses a **custom architecture, `openai-privacy-filter`**, that is not (yet) part of
upstream llama.cpp. It runs on:

1. **[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp)** *(recommended)* —
   a small standalone GGML engine for exactly this model family, on **stock upstream ggml with
   no patches** (CPU / CUDA / Vulkan). This is the reference runtime.

   ```sh
   # build (see the repo README for CUDA/Vulkan)
   cmake --preset release && cmake --build --preset release -j
   # run
   echo "My name is Alice Smith" | \
     build/release/pf-cli --classify privacy-filter-f16.gguf 0.5
   ```

   It exposes a flat C API (`pf_load` / `pf_classify` → entity spans with UTF-8 byte offsets;
   `pf_tokenize` / `pf_logits`) shaped for FFI — see the repo README.

2. **[LocalAI](https://github.com/mudler/LocalAI)** — install from the model gallery; LocalAI
   serves it behind the gRPC `TokenClassify` RPC and runs the constrained BIOES Viterbi decode,
   returning entity spans. LocalAI drives it through the **`privacy-filter` backend** (which
   wraps privacy-filter.cpp). The model is **not** a chat/completion model — it is a PII
   detector that other models opt into via a `pii.detectors` list.

3. **llama.cpp — only with a patch.** Stock `llama.cpp`, `llama-cpp-python`, Ollama, and
   LM Studio will **fail to load** this file (`unknown model architecture:
   'openai-privacy-filter'`). The arch can be added with carry-patches (TOKEN_CLS pooling, the
   architecture + HF→GGUF converter, the bidirectional banded-attention graph, and an all-SWA
   no-cache mask fix; TOKEN_CLS pooling tracks the still-open
   [PR #19725](https://github.com/ggml-org/llama.cpp/pull/19725)). Until that support lands
   upstream, `privacy-filter.cpp` above is the patch-free alternative.

> **Pooling note (llama.cpp path only):** the model must be loaded with **TOKEN_CLS pooling**
> (the GGUF's default). If you drive `llama-embedding` directly for testing, do **not** pass
> `--pooling none`. privacy-filter.cpp handles this automatically.

## Files

| File | Precision | Size | Notes |
|---|---|---|---|
| `privacy-filter-f16.gguf` | F16 | 2.82 GB | Reference artifact. 156 tensors; 33 `classifier.output_labels`; `pooling_type = TOKEN_CLS`. |
| `privacy-filter-q8.gguf` | Q8_0 (experts) | ~1.6 GB | MoE expert weights → Q8_0, the rest F16. For RAM-constrained / edge use. |

`sha256 (f16): eb71312b6b9370d0fe582e576b840567bb06603c4de241c6d899205d1b04dc81`
`sha256 (q8):  80efc1803eda7c095a79741d2008c07e2e0a57b01bac8825fbeb448fd097998c`

**Q8_0 quantization — and why it isn't free.** `q8` stores the bulk of the weights (the MoE
expert matrices) as 8-bit integers instead of 16-bit floats — via
[`scripts/requant_q8.py`](https://github.com/localai-org/privacy-filter.cpp/blob/master/scripts/requant_q8.py),
with attention, embeddings and the classifier head left at F16. That roughly halves the download
(2.82 GB → ≈1.6 GB) and is usually a bit faster on CPU.

The catch: **reducing precision throws information away, and it is almost never a free lunch.**
On a mixed-PII document (1,360 tokens) q8 matched f16 on **99.7%** of token labels (average
prediction shift, KL divergence, of 1.1e-3) — close, but note it did **not** match on all of
them; a few tokens flipped. That is the point in miniature: a reassuring average still hides the
specific cases that change, and **accuracy benchmarks tend to look fine until the one that
bites.** For PII detection a missed span is a leak, so **prefer F16 when you can afford it** (it
is the reference precision) and treat **Q8_0 as a deliberate size/speed tradeoff** for
constrained hardware — ideally re-checked on your own data.

## Architecture & conversion

gpt-oss-style sparse **MoE** (8 layers, `d_model=640`, 128 experts, top-4 routing; ~1.5B total /
~50M active per token), **bidirectional banded attention** (symmetric sliding window, attention
sinks retained), **interleaved (GPT-J) RoPE** with YaRN (θ=150000, factor 32), o200k
(`o200k_base`) tokenizer, and a 33-way token-classification head (`score` → `cls.output`).
privacy-filter.cpp re-derives the YaRN `truncate=false` frequencies at load time (fed to
`ggml_rope_ext` as `freq_factors`) so the GGUF is interchangeable across runtimes.

## Label space

`O` plus `B-`/`I-`/`E-`/`S-` for each of 8 categories (1 + 8×4 = 33):
`account_number`, `private_address`, `private_date`, `private_email`, `private_person`,
`private_phone`, `private_url`, `secret`. The ordered `id2label` table is embedded in the GGUF
(`classifier.output_labels`).

## Limitations & intended use

Identical to the [source model](https://huggingface.co/openai/privacy-filter): trained for
high-throughput data sanitization, **not** a substitute for legal/compliance review, and **not**
a clinical PHI model. Use it as one tier behind deterministic regex pre-filters and human
review. For multilingual text, prefer the
[multilingual fine-tune](https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF).

## License

**Apache-2.0**, inherited from `openai/privacy-filter`.

## Credits & citation

Model by **OpenAI** (`openai/privacy-filter`). GGUF conversion and runtime support
(`privacy-filter.cpp`) by the **LocalAI** project. Please cite OpenAI per the
[source card](https://huggingface.co/openai/privacy-filter).
