"""
Steering hook — adds a scaled direction vector to the residual stream at one
layer during generation. Ported from GospelVec/src/steer_chat.py (GospelSteerer),
specialised to a single named vector and with alpha expressed as a MULTIPLE of the
layer's median residual norm (neutralises Gemma's large "massive activations").

Shared by Phase 1.5 (qa_probe), Phase 2 (agent_loop), and Phase 3 (compute_task).

Quick manual check (Phase 1 verification):
  python -m src.steer --vector gluttony --alpha 1.5 \
      --prompt "You're at a buffet. Describe how you fill your plate."
"""

from __future__ import annotations

import argparse
import json

import torch

from .config import VECTOR_DIR, CONTROL_VECTOR_DIR
from .common import load_model, discover_layers, input_device, chat_ids


def load_meta() -> dict:
    with open(VECTOR_DIR / "meta.json") as f:
        return json.load(f)


def load_vector(name: str, layer: int) -> torch.Tensor:
    """Load a named all-layer vector and return its row at `layer`."""
    path = (VECTOR_DIR / "gluttony.pt" if name == "gluttony"
            else CONTROL_VECTOR_DIR / f"{name}.pt")
    return torch.load(path, weights_only=True)[layer]


class SteeringHook:
    """Adds `alpha * residual_norm * direction` to one layer's output."""

    def __init__(self, model, layer_idx: int, direction: torch.Tensor,
                 residual_norm: float):
        layers, _, _ = discover_layers(model)
        self.layer_idx = layer_idx
        self.residual_norm = residual_norm
        self.alpha = 0.0
        # placed on the steered layer's device for multi-GPU device_map="auto"
        dev = next(layers[layer_idx].parameters()).device
        self.direction = direction.to(dev)
        self._handle = layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, output):
        if self.alpha == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        delta = (self.alpha * self.residual_norm) * self.direction.to(hidden.dtype)
        hidden = hidden + delta
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.0, system: str | None = None) -> str:
    """Chat-formatted generation; steering (if any) is applied via active hooks."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    ids = chat_ids(tokenizer, messages).to(input_device(model))
    out = model.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.9 if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="gluttony")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    meta = load_meta()
    layer = args.layer if args.layer is not None else meta["best_layer"]
    norm = meta["residual_norm_per_layer"][layer]

    model, tokenizer = load_model()
    direction = load_vector(args.vector, layer)
    hook = SteeringHook(model, layer, direction, norm)

    for a in (0.0, args.alpha, -args.alpha):
        hook.set_alpha(a)
        print(f"\n=== {args.vector} alpha={a:+.1f} (layer {layer}) ===")
        print(generate(model, tokenizer, args.prompt))
    hook.remove()


if __name__ == "__main__":
    main()
