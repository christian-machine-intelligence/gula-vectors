"""
Diagnostic (Phase 1.5 gate FAILED at layer 46) — find a steering site that actually
moves behavior, or establish that none does at coherent magnitudes.

Two sensitivity upgrades over the gate:
  1. sweep earlier/mid single layers + a multi-layer band (amplify while keeping
     per-layer alpha low, a la GospelVec steering layers 18-24 together);
  2. logit-gap readout: delta_logit(excessive_letter - moderate_letter) at the
     forced-choice decision token — detects a latent lean even when the hard choice
     never flips (the gate's P(excessive) was pinned at 0).

For each config we report mean logit-gap vs alpha (dose-response), hard P(excessive),
and a coherence snippet. A good site shows a monotone logit-gap rising with alpha
while the snippet stays fluent.

  python -m src.layer_sweep
"""

from __future__ import annotations

import torch

from .common import load_model, input_device
from .steer import SteeringHook, load_vector, load_meta, generate
from .qa_probe import _expand_forced_choice

# (label, layers, alpha grid). Multi-layer bands use lower per-layer alpha.
CONFIGS = [
    ("L12", [12], [0.0, 0.3, 0.5]),
    ("L20", [20], [0.0, 0.3, 0.5]),
    ("L28", [28], [0.0, 0.3, 0.5]),
    ("L36", [36], [0.0, 0.3, 0.5]),
    ("L44", [44], [0.0, 0.3, 0.5]),
    ("band20-36", [20, 24, 28, 32, 36], [0.0, 0.15, 0.25]),
]
COHERENCE_PROMPT = "How much should I order for a quick weekday lunch for myself?"


def letter_id(tokenizer, letter):
    ids = tokenizer.encode(letter, add_special_tokens=False)
    return ids[-1]


@torch.no_grad()
def logit_gap(model, tokenizer, prompt, exc_letter, mod_letter, idA, idB):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(input_device(model))
    logits = model(**inputs).logits[0, -1].float()
    gap_AB = (logits[idA] - logits[idB]).item()      # logit(A) - logit(B)
    return gap_AB if exc_letter == "A" else -gap_AB    # -> logit(excessive) - logit(moderate)


def main():
    meta = load_meta()
    norms = meta["residual_norm_per_layer"]
    model, tokenizer = load_model()
    direction_all = torch.load(__import__("pathlib").Path(
        "vectors/gluttony.pt"), weights_only=True)  # [num_layers, hidden]
    idA, idB = letter_id(tokenizer, "A"), letter_id(tokenizer, "B")
    fc = _expand_forced_choice()

    for label, layers, alphas in CONFIGS:
        hooks = [SteeringHook(model, L, direction_all[L], norms[L]) for L in layers]
        print(f"\n{'='*70}\nCONFIG {label} layers={layers}")
        for a in alphas:
            for h in hooks:
                h.set_alpha(a)
            gaps, exc_hits = [], 0
            for item in fc:
                mod = "B" if item["excessive_letter"] == "A" else "A"
                gaps.append(logit_gap(model, tokenizer, item["prompt"],
                                      item["excessive_letter"], mod, idA, idB))
                # hard choice
                import re
                ans = generate(model, tokenizer, item["prompt"], max_new_tokens=4)
                m = re.search(r"\b([AB])\b", ans.upper())
                if m and m.group(1) == item["excessive_letter"]:
                    exc_hits += 1
            snip = " ".join(generate(model, tokenizer, COHERENCE_PROMPT,
                                     max_new_tokens=40).split())[:120]
            mean_gap = sum(gaps) / len(gaps)
            print(f"  alpha={a:+.2f}  mean_logit_gap(exc-mod)={mean_gap:+.3f}  "
                  f"P(excessive)={exc_hits/len(fc):.2f}  | {snip}")
        for h in hooks:
            h.remove()


if __name__ == "__main__":
    main()
