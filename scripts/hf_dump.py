#!/usr/bin/env python3
"""Dump HF reference fixtures for privacy-filter.cpp parity tests.

Per case, writes under <out>/<case>/:
  text.txt        the input text (UTF-8)
  tokens.i32      token ids, little-endian int32
  offsets.i32     2n int32: (start,end) BYTE offsets per token
  <tap>.f32       activation taps, row-major [n_tok, D] float32
  meta.json       tap dims + versions + notes

Tap names match the C++ engine's ggml_set_name taps:
  embd, l{i}.attn_norm, l{i}.q_rope, l{i}.k_rope, l{i}.attn_out,
  l{i}.ffn_inp, l{i}.post_norm, l{i}.moe_logits, l{i}.moe_topk (i32),
  l{i}.moe_weights, l{i}.moe_out, l{i}.l_out, result_norm, logits

Layout conventions:
  - all f32 taps are [n_tok, D] row-major; multi-head taps (q_rope/k_rope)
    flatten head-major: D = n_head * head_dim, features h0d0..h0d63,h1d0...
  - HF tokenizers reports CHAR offsets; converted to UTF-8 BYTE offsets here.
  - HF's router returns softmax(top4)/top_k and its MLP multiplies the expert
    sum by top_k afterwards (the two cancel). moe_weights is stored *top_k so
    it matches the engine's plain-softmax weights; moe_out is the MLP module
    output (post *top_k), directly comparable to the engine's tap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASES = {
    # The 21-token fixture used by the llama.cpp-fork parity check.
    "short-en": "Contact John Doe at john.doe@example.com or call +1-202-555-0173.",
    # Mixed scripts (Latin+diacritics, German, Arabic, Bengali, CJK, Cyrillic)
    # with PII in each. Exercises multibyte offset mapping end to end.
    "multilingual": (
        "Meine IBAN ist DE89370400440532013000 und meine Telefonnummer ist +49 30 901820. "
        "اسمي أحمد الخطيب وبريدي الإلكتروني ahmad.alkhatib@example.org وأعيش في شارع الملك فهد، الرياض. "
        "আমার নাম অমিতা চক্রবর্তী, আমার ফোন +880 1712-345678 এবং জন্ম তারিখ ১৯৯০ সালের ৫ মে। "
        "我叫王小明，住在北京市朝阳区建国路88号，邮箱是 xiaoming.wang@example.cn，电话 +86 138 0013 8000。"
        "Меня зовут Ирина Соколова, мой IP-адрес 192.168.10.44, а пароль — Тайна123! "
        "José Đorđević's café is at 12 Rue de l'Église, Paris; SIRET masked, card 4111 1111 1111 1111."
    ),
    # Dense coverage of many label categories in one document.
    "pii-dense": (
        "John Q. Smith (username jsmith42, password hunter2, PIN 0420) was born 1985-03-14. "
        "SSN 078-05-1120, phone +1 (415) 555-2671, email john.smith@corp.example.com. "
        "Address: Apt 4B, 221 Baker Street, Springfield, IL 62704, USA. "
        "IBAN GB82WEST12345698765432, BIC DEUTDEFF, account 12345678, card 5500 0000 0000 0004 (CVV 123, exp DATE). "
        "BTC 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, ETH 0x52908400098527886E0F7030069857D2E4169EE7. "
        "IP 10.0.0.7, MAC 00:1B:44:11:3A:B7, IMEI 490154203237518, VIN 1HGBH41JXMN109186, plate AB12 CDE. "
        "UA: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36; URL https://intranet.corp.example.com/hr?id=991. "
        "GPS 37.7749,-122.4194; height 182cm; eye color green; job title Senior Auditor, dept Finance."
    ),
}

# long-3k is generated: ~3k tokens of varied PII paragraphs, deterministic.
def make_long_3k() -> str:
    import random
    rng = random.Random(20260612)
    first = ["Anna", "Béla", "Chloé", "Dmitri", "Elif", "Farid", "Grace", "Hiro",
             "Inés", "Jakub", "Klara", "Liang", "Marta", "Nikos", "Olga", "Pavel"]
    last = ["Kowalski", "Müller", "Okafor", "Petrov", "Quispe", "Rossi", "Sato",
            "Tanaka", "Ueda", "Varga", "Weber", "Xu", "Yilmaz", "Zhang"]
    streets = ["Elm Street", "Hauptstraße", "Rue Lafayette", "Via Roma", "Andrássy út"]
    cities = ["Lyon", "Dresden", "Porto", "Kraków", "Osaka", "Tartu", "Quito"]
    paras = []
    for i in range(90):
        fn, ln = rng.choice(first), rng.choice(last)
        em = f"{fn.lower()}.{ln.lower()}{i}@mail{i % 7}.example.com"
        ph = f"+{rng.randint(1, 48)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"
        st = f"{rng.randint(1, 250)} {rng.choice(streets)}"
        ct = rng.choice(cities)
        ib = f"DE{rng.randint(10, 99)}3704 0044 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} 00"
        paras.append(
            f"Case {i}: {fn} {ln} reported a billing issue on {rng.randint(1, 28)} "
            f"March 202{rng.randint(4, 6)}. Contact at {em} or {ph}. Ships to {st}, {ct}. "
            f"Refund to IBAN {ib}. Agent notes: customer prefers morning calls, "
            f"timezone UTC+{rng.randint(0, 3)}, repeat caller, satisfaction {rng.randint(1, 10)}/10."
        )
    return "\n\n".join(paras)


def char_to_byte_offsets(text: str, char_offsets):
    # cumulative UTF-8 byte length per char prefix
    cum = [0]
    for ch in text:
        cum.append(cum[-1] + len(ch.encode("utf-8")))
    return [(cum[a], cum[b]) for a, b in char_offsets]


def dump_case(model, tok, text: str, out: Path, n_layer: int, top_k: int):
    import numpy as np
    import torch

    out.mkdir(parents=True, exist_ok=True)
    (out / "text.txt").write_text(text, encoding="utf-8")

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    boffs = char_to_byte_offsets(text, enc["offset_mapping"])
    np.asarray(ids, dtype="<i4").tofile(out / "tokens.i32")
    np.asarray([v for p in boffs for v in p], dtype="<i4").tofile(out / "offsets.i32")

    taps: dict[str, "np.ndarray"] = {}

    def save(name, t):
        a = t.detach().to(torch.float32).cpu().numpy()
        taps[name] = a.reshape(a.shape[-2] if a.ndim >= 2 else 1, -1) if a.ndim != 2 else a

    hooks = []

    def on(mod, fn):
        hooks.append(mod.register_forward_hook(fn))

    on(model.model.embed_tokens, lambda m, i, o: save("embd", o[0] if o.ndim == 3 else o))

    rot = {}
    on(model.model.rotary_emb, lambda m, i, o: rot.update(cos=o[0], sin=o[1]))

    from transformers.models.openai_privacy_filter.modeling_openai_privacy_filter import _apply_rotary_emb

    for i, layer in enumerate(model.model.layers):
        L = f"l{i}"

        def mk(i=i, L=L, layer=layer):
            on(layer.input_layernorm, lambda m, inp, o: save(f"{L}.attn_norm", o[0]))

            def qk_hook(which):
                def fn(m, inp, o):
                    # o: [b, n, n_heads*64] -> [b, heads, n, 64] -> rope -> [n, heads*64]
                    b, n, _ = o.shape
                    x = o.view(b, n, -1, 64).transpose(1, 2)
                    cos, sin = rot["cos"], rot["sin"]
                    xr = _apply_rotary_emb(x, cos.unsqueeze(1), sin.unsqueeze(1))
                    save(f"{L}.{which}_rope", xr.transpose(1, 2).reshape(n, -1))
                return fn
            on(layer.self_attn.q_proj, qk_hook("q"))
            on(layer.self_attn.k_proj, qk_hook("k"))

            on(layer.self_attn, lambda m, inp, o: save(f"{L}.attn_out", o[0][0]))

            def post_norm_hook(m, inp, o, L=L):
                save(f"{L}.ffn_inp", inp[0][0])
                save(f"{L}.post_norm", o[0])
                # hooks must return None or they replace the module output
            on(layer.post_attention_layernorm, post_norm_hook)

            def router_hook(m, inp, o):
                logits, scores, idx = o
                save(f"{L}.moe_logits", logits)
                # undo the /top_k so it matches the engine's plain softmax
                save(f"{L}.moe_weights", scores * top_k)
                taps[f"{L}.moe_topk"] = idx.detach().cpu().numpy().astype("<i4")
            on(layer.mlp.router, router_hook)
            on(layer.mlp, lambda m, inp, o: save(f"{L}.moe_out", o[0][0]))
            on(layer, lambda m, inp, o: save(f"{L}.l_out", o[0] if o.ndim == 3 else o))
        mk()

    on(model.model.norm, lambda m, inp, o: save("result_norm", o[0]))

    with torch.no_grad():
        outp = model(input_ids=torch.tensor([ids]))
    taps["logits"] = outp.logits[0].to(torch.float32).cpu().numpy()
    for h in hooks:
        h.remove()

    meta = {"n_tok": len(ids), "taps": {}}
    for name, a in taps.items():
        suffix = "i32" if a.dtype.kind == "i" else "f32"
        a.astype(f"<{a.dtype.kind}4").tofile(out / f"{name}.{suffix}")
        meta["taps"][name] = {"shape": list(a.shape), "dtype": suffix}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {out.name}: {len(ids)} tokens, {len(taps)} taps")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    cases = dict(CASES)
    # Trim to ~3k tokens: long enough for position-scaled YaRN divergence and
    # window-stitch tests, short enough for eager O(n^2) attention in RAM.
    paras = make_long_3k().split("\n\n")
    while paras:
        text = "\n\n".join(paras)
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= 3100:
            break
        paras = paras[: max(1, len(paras) * 3000 // 3300) - 1] if len(paras) > 35 else paras[:-1]
    cases["long-3k"] = "\n\n".join(paras)
    if args.cases:
        cases = {k: cases[k] for k in args.cases}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    n_layer = model.config.num_hidden_layers
    top_k = model.config.num_experts_per_tok

    out = Path(args.out)
    print(f"transformers {transformers.__version__}, dumping to {out}")
    for name, text in cases.items():
        dump_case(model, tok, text, out / name, n_layer, top_k)
    (out / "versions.json").write_text(json.dumps({
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }, indent=1))


if __name__ == "__main__":
    sys.exit(main())
