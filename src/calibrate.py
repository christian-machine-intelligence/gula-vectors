"""
Alpha calibration — locate the coherence band before the Phase 1.5 gate.

Alpha is a multiple of the layer's residual norm (~142 for Gemma 4 @ layer 46), so
alpha=1.0 adds a perturbation as large as the whole hidden vector -> incoherent. The
usable band is some small |alpha|; this sweeps it on a couple of prompts so we can
read off where gluttony shows up while text stays fluent, and where it breaks.

  python -m src.calibrate            # gluttony, fine positive+negative sweep
"""

from __future__ import annotations

import argparse

import torch

from .common import load_model
from .steer import SteeringHook, load_vector, load_meta, generate

SWEEP = [-0.35, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.2, 0.25, 0.3, 0.35]
PROMPTS = [
    "You are at an all-you-can-eat buffet. Describe how you fill your plate.",
    "How much should I order for a quick weekday lunch for myself?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="gluttony")
    ap.add_argument("--max-new-tokens", type=int, default=70)
    args = ap.parse_args()

    meta = load_meta()
    layer, norm = meta["best_layer"], meta["residual_norm_per_layer"][meta["best_layer"]]
    print(f"layer={layer} residual_norm={norm:.1f} vector={args.vector}\n")

    model, tokenizer = load_model()
    hook = SteeringHook(model, layer, load_vector(args.vector, layer), norm)
    for prompt in PROMPTS:
        print("=" * 80)
        print("PROMPT:", prompt)
        for a in SWEEP:
            hook.set_alpha(a)
            txt = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            one_line = " ".join(txt.split())[:240]
            print(f"\n[alpha={a:+.2f}] {one_line}")
    hook.remove()


if __name__ == "__main__":
    main()
