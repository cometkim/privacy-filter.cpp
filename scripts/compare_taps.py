#!/usr/bin/env python3
"""Compare engine activation taps against HF reference dumps, in topological
order, stopping at the first diverging tap (everything downstream of a real
divergence is noise).

Usage: compare_taps.py <hf_case_dir> <engine_dump_dir>
           [--atol-base 1e-4] [--rtol 1e-3] [--cos 0.99999] [--scale 1.0]

Pass rule per f32 tap (both must hold):
  1. per-row cosine >= --cos (primary gate; a real graph bug — wrong mask,
     swapped tensor, wrong expert — craters this)
  2. per-row normalized error  max|a-b| / max(1, max|b|)  <= --row-err
     (default 1e-2). Absolute tolerances do not fit this model: rope phase
     noise is position-scaled (ggml accumulates theta multiplicatively while
     HF multiplies pos*inv_freq once; both round, differently) and the
     residual stream magnitude grows with depth, so mid-layer absolute errors
     track |row| while cosine and final logits stay clean (measured: 0.2
     worst element in l5.attn_out at pos~3000, final logits within 4e-3,
     argmax 100%). Normalizing by the row's inf-norm bounds the error where
     it is meaningful and keeps discrimination for real bugs.
Taps are compared against f64-rotary HF dumps (hf_dump.py): the engine sits
~75x closer to the exact rotation than HF's stock f32 rotary at pos 3000, so
stock-rotary dumps would drown every tap in the *reference's* phase noise.
The stock-reference end-to-end gate runs separately (logits_stock).
--scale multiplies the tolerances (e.g. 4 for Vulkan).

moe_topk is compared as a per-token SET. A mismatched set is a hard failure
unless the token is a routing TIE (gap between the kth and k+1th router logit
< 1e-4); tied tokens are reported and excluded, but their l{i}.l_out must
still pass (a wrong expert from a genuine tie must barely move the output).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def load(d: Path, name: str, meta) -> np.ndarray:
    t = meta["taps"][name]
    a = np.fromfile(d / f"{name}.{t['dtype']}", dtype=f"<{t['dtype'][0]}4")
    return a.reshape(t["shape"])


def tap_order(names):
    def key(n):
        m = re.match(r"l(\d+)\.(.+)", n)
        sub = ["attn_norm", "q_rope", "k_rope", "attn_out", "ffn_inp", "post_norm",
               "moe_logits", "moe_topk", "moe_weights", "moe_out", "l_out"]
        if n == "embd":
            return (-1, 0)
        if m:
            return (int(m.group(1)), sub.index(m.group(2)) if m.group(2) in sub else 99)
        return (1000, ["result_norm", "logits"].index(n) if n in ("result_norm", "logits") else 99)
    return sorted(names, key=key)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("engine_dir", type=Path)
    ap.add_argument("--row-err", type=float, default=1e-2)
    ap.add_argument("--cos", type=float, default=0.99999)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tie-gap", type=float, default=1e-4)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    hf_meta = json.loads((args.hf_dir / "meta.json").read_text())
    en_meta = json.loads((args.engine_dir / "meta.json").read_text())
    common = [n for n in tap_order(hf_meta["taps"]) if n in en_meta["taps"]]
    missing = [n for n in hf_meta["taps"] if n not in en_meta["taps"]]
    if missing:
        print(f"note: engine did not dump: {missing}")

    print(f"{'tap':24s} {'max_abs':>10s} {'row_err':>10s} {'min_cos':>9s}  verdict")
    failed = None
    tied_tokens: dict[int, set] = {}

    for name in common:
        hf = load(args.hf_dir, name, hf_meta)
        en = load(args.engine_dir, name, en_meta)
        if hf.shape != en.shape:
            print(f"{name:24s} SHAPE MISMATCH hf{hf.shape} engine{en.shape}")
            failed = name
            if not args.keep_going:
                break
            continue

        m = re.match(r"l(\d+)\.", name)
        layer = int(m.group(1)) if m else (hf_meta.get("n_layer", 8) if name in ("result_norm", "logits") else 0)

        if name.endswith("moe_topk"):
            logits = load(args.hf_dir, f"l{layer}.moe_logits", hf_meta)
            k = hf.shape[1]
            srt = np.sort(logits, axis=1)
            gap = srt[:, -k] - srt[:, -k - 1]
            bad = ties = 0
            for t in range(hf.shape[0]):
                if set(hf[t]) == set(en[t]):
                    continue
                if gap[t] < args.tie_gap:
                    ties += 1
                    tied_tokens.setdefault(layer, set()).add(t)
                else:
                    bad += 1
            verdict = "OK" if bad == 0 else "FAIL"
            print(f"{name:24s} {'-':>10s} {'-':>10s} {'-':>9s}  {verdict}"
                  f" (set mismatches: {bad}, ties excluded: {ties})")
            if bad:
                failed = failed or name
                if not args.keep_going:
                    break
            continue

        diff = np.abs(hf - en)
        row_scale = np.maximum(np.abs(hf).max(axis=1), 1.0)
        row_err = diff.max(axis=1) / row_scale
        num = (hf * en).sum(1)
        den = np.linalg.norm(hf, axis=1) * np.linalg.norm(en, axis=1) + 1e-12
        cos = num / den
        ok = row_err.max() <= args.row_err * args.scale and cos.min() >= args.cos
        print(f"{name:24s} {diff.max():10.3e} {row_err.max():10.3e} {cos.min():9.6f}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failed = failed or name
            if not args.keep_going:
                break

    if tied_tokens:
        print(f"routing ties: { {k: sorted(v) for k, v in tied_tokens.items()} }")
    if failed:
        print(f"FIRST DIVERGENCE: {failed}")
        return 1
    print("ALL TAPS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
