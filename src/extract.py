"""
Phase 1 (extraction): capture mean-pooled residual-stream activations for every
act in the ICMI-025 benchmark (per sin) plus the neutral PCA-denoising texts.

Ports GospelVec/src/extract.py (ActivationCollector + mean_pool) to Gemma 4 and
to act-level (rather than chunk-level) stimuli.

Output (per framing):
  activations/acts_<framing>.pt   dict {sin: tensor[num_layers, num_acts, hidden]}
  activations/neutral_<framing>.pt          tensor[num_layers, num_neutral, hidden]

Usage:
  python -m src.extract                 # primary framing (self)
  python -m src.extract --framing abstract
"""

from __future__ import annotations

import argparse
import time

import torch

from .config import (
    ACTIVATION_DIR, MAX_ACT_TOKENS, MEAN_POOL_SKIP_TOKENS, NEUTRAL_TEXTS,
    PRIMARY_FRAMING, SINS,
)
from .common import load_acts, load_model, discover_layers, input_device


class ActivationCollector:
    """Registers forward hooks on all decoder layers to capture residual stream."""

    def __init__(self, layers):
        self.activations = {}
        self.hooks = []
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
              skip_tokens: int = MEAN_POOL_SKIP_TOKENS) -> torch.Tensor:
    """Mean-pool across token positions, skipping leading special tokens.

    With device_map="auto" pipeline sharding, each layer's activation can live on a
    different GPU, so align the mask to the activation's device.
    """
    mask = attention_mask.to(hidden_states.device).clone().float()
    mask[:, :skip_tokens] = 0.0
    mask = mask.unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)


@torch.no_grad()
def extract_texts(model, tokenizer, layers, texts: list[str]) -> torch.Tensor:
    """Return [num_layers, num_texts, hidden] mean-pooled activations."""
    collector = ActivationCollector(layers)
    per_text = []
    for i, text in enumerate(texts):
        if i == 0 or (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_ACT_TOKENS).to(input_device(model))
        collector.clear()
        model(**inputs)
        reps = []
        for li in range(len(collector.activations)):
            pooled = mean_pool(collector.activations[li].float(),
                               inputs["attention_mask"])
            reps.append(pooled.squeeze(0).cpu())
        per_text.append(torch.stack(reps))            # [num_layers, hidden]
    collector.remove()
    return torch.stack(per_text).permute(1, 0, 2)     # [num_layers, num_texts, hidden]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default=PRIMARY_FRAMING)
    args = ap.parse_args()

    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    layers, num_layers, hidden = discover_layers(model)
    print(f"Model: {num_layers} layers, hidden={hidden}")

    acts = load_acts(args.framing)
    sin_reps = {}
    for sin in SINS:
        print(f"\nExtracting {sin} ({len(acts[sin])} acts) ...")
        t0 = time.time()
        sin_reps[sin] = extract_texts(model, tokenizer, layers, acts[sin])
        print(f"  {sin}: {tuple(sin_reps[sin].shape)} in {time.time() - t0:.1f}s")

    print(f"\nExtracting {len(NEUTRAL_TEXTS)} neutral texts ...")
    neutral_reps = extract_texts(model, tokenizer, layers, NEUTRAL_TEXTS)

    torch.save(sin_reps, ACTIVATION_DIR / f"acts_{args.framing}.pt")
    torch.save(neutral_reps, ACTIVATION_DIR / f"neutral_{args.framing}.pt")
    print(f"\nSaved activations to {ACTIVATION_DIR}/ (framing={args.framing})")


if __name__ == "__main__":
    main()
