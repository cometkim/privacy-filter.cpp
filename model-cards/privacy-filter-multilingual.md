---
license: apache-2.0
base_model: OpenMed/privacy-filter-multilingual
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
  - multilingual
  - openai-privacy-filter
language:
  - ar
  - bn
  - de
  - en
  - es
  - fr
  - hi
  - it
  - ja
  - ko
  - nl
  - pt
  - te
  - tr
  - vi
  - zh
---

# privacy-filter-multilingual — GGUF (F16 + Q8_0)

GGUF conversion of [`OpenMed/privacy-filter-multilingual`](https://huggingface.co/OpenMed/privacy-filter-multilingual),
a multilingual PII **token-classification** model (a fine-tune of
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)). It labels every
token with a BIOES tag over **54 PII categories (217 classes)** across **16 languages**, so
it can be served locally with **no Python** as the encoder/NER tier of a PII redactor.

For the full model description, label space, evaluation, limitations, and citations, see the
**[source model card](https://huggingface.co/OpenMed/privacy-filter-multilingual)** — this
card only covers the GGUF packaging and how to run it.

## Runtimes

This GGUF uses a **custom architecture, `openai-privacy-filter`**, that is not (yet) part of
upstream llama.cpp. It runs on:

1. **[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp)** *(recommended)* —
   a small standalone GGML engine for exactly this model family, on **stock upstream ggml with
   no patches** (CPU / CUDA / Vulkan). This is the reference runtime and what the parity numbers
   below are measured against.

   ```sh
   # build (see the repo README for CUDA/Vulkan)
   cmake --preset release && cmake --build --preset release -j
   # run
   echo "Contact John Doe at jdoe@example.com" | \
     build/release/pf-cli --classify privacy-filter-multilingual-f16.gguf 0.5
   ```

   It exposes a flat C API (`pf_load` / `pf_classify` → entity spans with UTF-8 byte offsets;
   `pf_tokenize` / `pf_logits`) shaped for FFI — see the repo README.

2. **[LocalAI](https://github.com/mudler/LocalAI)** — install from the model gallery; LocalAI
   serves it behind the gRPC `TokenClassify` RPC and runs the constrained BIOES Viterbi decode,
   returning entity spans. LocalAI drives it through the **`privacy-filter` backend** (which
   wraps privacy-filter.cpp); older builds used a llama.cpp-patched path. The model is **not** a
   chat/completion model — it is a PII detector that other models opt into.

   ```bash
   local-ai models install privacy-filter-multilingual
   ```

   The gallery entry carries the detection policy in a `pii_detection:` block (default: mask
   everything detected; block credentials / financial-secrets / crypto). Other models opt in by
   listing it under `pii.detectors`:

   ```yaml
   # any chat or cloud-proxy model — opt in and reference the detector(s)
   name: my-assistant
   pii:
     enabled: true
     detectors:
       - privacy-filter-multilingual
   ```

3. **llama.cpp — only with a patch.** Stock `llama.cpp`, `llama-cpp-python`, Ollama, and
   LM Studio will **fail to load** this file (`unknown model architecture:
   'openai-privacy-filter'`). The arch can be added with carry-patches (TOKEN_CLS pooling, the
   architecture + HF→GGUF converter, the bidirectional banded-attention graph, and an all-SWA
   no-cache mask fix; TOKEN_CLS pooling tracks the still-open
   [PR #19725](https://github.com/ggml-org/llama.cpp/pull/19725)). Until that support lands
   upstream, the patched path is carried by LocalAI; `privacy-filter.cpp` above is the
   patch-free alternative.

> **Pooling note (llama.cpp path only):** the model must be loaded with **TOKEN_CLS pooling**
> (the GGUF's default). If you drive `llama-embedding` directly for testing, do **not** pass
> `--pooling none` — that overrides the default and yields raw hidden states instead of label
> logits. privacy-filter.cpp handles this automatically.

## Files

| File | Precision | Size | Notes |
|---|---|---|---|
| `privacy-filter-multilingual-f16.gguf` | F16 | ~2.7 GB | Reference artifact. 217 `classifier.output_labels`; `pooling_type = TOKEN_CLS`. |
| `privacy-filter-multilingual-q8.gguf` | Q8_0 (experts) | ~1.6 GB | MoE expert weights → Q8_0, the rest F16. For RAM-constrained / edge use. |

`sha256 (q8): 968135172ba8202374b4c3bd7d353e100c8fc574035da793fa4d13ca441319b7`

**Q8_0 quantization — and why it isn't free.** `q8` stores the bulk of the weights (the MoE
expert matrices) as 8-bit integers instead of 16-bit floats — via
[`scripts/requant_q8.py`](https://github.com/localai-org/privacy-filter.cpp/blob/master/scripts/requant_q8.py),
with attention, embeddings and the classifier head left at F16. That roughly halves the download
(≈2.7 GB → ≈1.6 GB) and is usually a bit faster on CPU.

The catch: **reducing precision throws information away, and it is almost never a free lunch.**
Our checks didn't find a regression — on a mixed-PII document (1,360 tokens) q8 matched f16 on
**100%** of token labels and produced identical spans, with an average prediction shift
(KL divergence) of just 6.9e-5. But "we didn't find a difference" is not the same as "there is
none." Those numbers come from a single English document, and a tiny *average* shift can still
hide a flip on the one input that matters to you — a rare name, an unusual phone or ID format, or
a language we never tested. **Accuracy benchmarks and divergence metrics routinely look
reassuring right up until the case that bites.** For PII detection a single missed span is a
leak, so:

- **Prefer F16** if you can afford the ~2.7 GB — it is the reference these numbers are measured
  against, and what we trust by default.
- **Use Q8_0** when memory or speed forces it (e.g. a 4 GB Raspberry Pi 5), treat it as a
  deliberate tradeoff, and **validate it on your own data** first. A full span-F1-per-language
  sweep across the 16 languages is the bar we'd want before calling q8 a true drop-in.

## Architecture & conversion

gpt-oss-style sparse **MoE** (8 layers, `d_model=640`, 128 experts, top-4 routing, ~50M active
per token), **bidirectional banded attention** (symmetric sliding window 128, attention sinks
retained), **interleaved (GPT-J) RoPE** with YaRN (θ=150000, factor 32), o200k (`o200k_base`)
tokenizer, and a 217-way token-classification head (`score` → `cls.output`).

The conversion reproduces the HF reference **exactly at F16**: token-for-token argmax match on
the parity prompt set, **full-logit cosine = 1.0**, every layer's residual-stream cosine = 1.0
(relerr ≈ 2e-4, i.e. F16 rounding). The two load-bearing conversion choices — the expert
`gate_up` `chunk(2)` split and the `n_swa = 2·sliding_window` window mapping — are both
confirmed by that parity. privacy-filter.cpp re-derives the YaRN `truncate=false` frequencies at
load time (fed to `ggml_rope_ext` as `freq_factors`) so the same GGUF is interchangeable across
runtimes.

This GGUF was produced by [`scripts/convert.py`](https://github.com/localai-org/privacy-filter.cpp/blob/master/scripts/convert.py)
— a self-contained HF→GGUF converter (no llama.cpp dependency). Nightly CI re-runs it and gates
the output against the HF reference logits, so the published artifact stays in parity.

## Label space

`O` plus `B-`/`I-`/`E-`/`S-` for each of 54 categories (1 + 54×4 = 217), spanning identity,
contact, address, dates/time, government IDs, financial, crypto, vehicle, digital, and auth
entities. The ordered `id2label` table is embedded in the GGUF (`classifier.output_labels`).
See the [source card](https://huggingface.co/OpenMed/privacy-filter-multilingual#label-space-54-categories)
for the full list.

## Limitations & intended use

Identical to the [source model](https://huggingface.co/OpenMed/privacy-filter-multilingual#limitations--intended-use):
multilingual but uneven (strongest on de/es/fr/it/hi/te/en; weaker on CJK), trained on
synthetic AI4Privacy data, **not** a substitute for legal/compliance review, and **not** a
clinical PHI model. Use it as one tier behind deterministic regex pre-filters and human review.

## License

**Apache-2.0**, inherited from `openai/privacy-filter` and `OpenMed/privacy-filter-multilingual`.

## Credits & citation

Conversion and runtime support by the **LocalAI** project (`privacy-filter.cpp`). The model
itself is by **OpenMed**, fine-tuned from **OpenAI**'s `privacy-filter`, on **AI4Privacy**
datasets — please cite all of them (BibTeX in the
[source card](https://huggingface.co/OpenMed/privacy-filter-multilingual#citation)).
