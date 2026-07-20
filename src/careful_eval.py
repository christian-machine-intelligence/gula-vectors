"""
Careful specificity pass (addresses the coherence-edge confound).

The discrete "pick N units" metric rewards extreme tokens, so ANY strong steering
spiked it (sloth/verbosity also hit 24-32x at the +-0.25 edge). This instead measures
the LOGIT-GAP at the forced-choice decision token: logit(excessive letter) - logit(
moderate letter). It captures the model's leaning without requiring (or rewarding) a
discrete extreme pick, so it is robust to coherence breakdown.

For every vector x a fine alpha grid we report mean logit-gap and the (logit-derived)
P(excessive) over the resource forced-choice battery. Clean specificity = gluttony's
gap rises monotonically with alpha while sloth/random/verbosity stay ~flat.

  CUDA_VISIBLE_DEVICES=1,2,3 python -m src.careful_eval
"""

from __future__ import annotations

import json

import torch

from .config import RESULTS_DIR
from .common import load_model, input_device
from .steer import SteeringHook, load_vector, load_meta
from .qa_probe import _expand_forced_choice

VECTORS = ["gluttony", "sloth", "random", "verbosity"]
ALPHAS = [-0.25, -0.15, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def letter_id(tokenizer, letter):
    return tokenizer.encode(letter, add_special_tokens=False)[-1]


@torch.no_grad()
def choice_logits(model, tokenizer, prompt, idA, idB):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(input_device(model))
    logits = model(**inp).logits[0, -1].float()
    return logits[idA].item(), logits[idB].item()


def main():
    meta = load_meta()
    layer = meta["best_layer"]
    norm = meta["residual_norm_per_layer"][layer]
    model, tokenizer = load_model()
    idA, idB = letter_id(tokenizer, "A"), letter_id(tokenizer, "B")
    fc = _expand_forced_choice()
    print(f"layer {layer}, residual_norm {norm:.1f}, {len(fc)} forced-choice items\n")

    out = {}
    for v in VECTORS:
        hook = SteeringHook(model, layer, load_vector(v, layer), norm)
        out[v] = {}
        for a in ALPHAS:
            hook.set_alpha(a)
            gaps, exc = [], 0
            for it in fc:
                la, lb = choice_logits(model, tokenizer, it["prompt"], idA, idB)
                gap_AB = la - lb
                gap = gap_AB if it["excessive_letter"] == "A" else -gap_AB
                gaps.append(gap)
                pred = "A" if la > lb else "B"
                exc += (pred == it["excessive_letter"])
            out[v][a] = {"logit_gap": sum(gaps) / len(gaps), "p_excessive": exc / len(fc)}
        hook.remove()
        row = "  ".join(f"a{a:+.2f}:{out[v][a]['logit_gap']:+.2f}" for a in ALPHAS)
        print(f"{v:10} logit_gap(exc-mod)  {row}", flush=True)

    (RESULTS_DIR / "careful_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS_DIR / 'careful_eval.json'}")
    print("\nP(excessive) by vector x alpha:")
    for v in VECTORS:
        print(f"  {v:10}", "  ".join(f"a{a:+.2f}:{out[v][a]['p_excessive']:.2f}" for a in ALPHAS))


if __name__ == "__main__":
    main()
