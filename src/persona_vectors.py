"""
Build the gluttony steering vector from RESPONSE-pooled persona activations.

Primary direction (sharp trait/anti-trait contrast, persona-vector method):
  gluttony[layer] = normalize( mean(gluttony replies) - mean(temperance replies) )

Controls (for the gate's specificity test):
  sloth     = normalize( mean(sloth) - mean(neutral) )   # opposite-sign prediction
  random    = seeded random unit vector
  verbosity = long-vs-short reply contrast (generic 'more text' axis)

Writes the SAME artifacts/format as compute_vectors.py, so steer/calibrate/qa_probe
need no changes. meta.json records residual norms of the reply activations (the scale
that steering actually perturbs) for alpha scaling.

  python -m src.persona_vectors
"""

from __future__ import annotations

import argparse
import json

import torch

from .config import ACTIVATION_DIR, CONTROL_VECTOR_DIR, VECTOR_DIR, STEER_LAYER
from .compute_vectors import normalize, roc_auc
from .personas import PRIMARY_POLE, ANTI_POLE, ANCHOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = torch.load(ACTIVATION_DIR / "persona_acts.pt", weights_only=True)
    replies = json.loads((ACTIVATION_DIR / "persona_replies.json").read_text())
    num_layers, hidden = acts[PRIMARY_POLE].shape[1], acts[PRIMARY_POLE].shape[2]
    print(f"personas={list(acts)} num_layers={num_layers} hidden={hidden}")

    def mean_at(persona, layer):
        return acts[persona][:, layer, :].mean(0)

    gluttony = torch.zeros(num_layers, hidden)
    sloth = torch.zeros(num_layers, hidden)
    for L in range(num_layers):
        gluttony[L] = normalize(mean_at(PRIMARY_POLE, L) - mean_at(ANTI_POLE, L))
        if "sloth" in acts:
            sloth[L] = normalize(mean_at("sloth", L) - mean_at(ANCHOR, L))

    # random + verbosity controls
    g = torch.Generator().manual_seed(args.seed)
    random_vec = torch.stack([normalize(torch.randn(hidden, generator=g))
                              for _ in range(num_layers)])
    # verbosity: long vs short replies across ALL persona-question samples
    lengths, flat = [], []
    for persona in acts:
        for qi, r in enumerate(replies[persona]):
            lengths.append(len(r["reply"]))
            flat.append((persona, qi))
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    third = max(1, len(order) // 3)
    short_idx = [flat[i] for i in order[:third]]
    long_idx = [flat[i] for i in order[-third:]]
    verbosity = torch.zeros(num_layers, hidden)
    for L in range(num_layers):
        long_m = torch.stack([acts[p][q, L] for p, q in long_idx]).mean(0)
        short_m = torch.stack([acts[p][q, L] for p, q in short_idx]).mean(0)
        verbosity[L] = normalize(long_m - short_m)

    # layer selection: gluttony-vs-temperance reply separability (AUC of cosine)
    layer_auc, residual_norm = [], []
    for L in range(num_layers):
        d = gluttony[L]
        pos = acts[PRIMARY_POLE][:, L, :] @ d
        neg = acts[ANTI_POLE][:, L, :] @ d
        layer_auc.append(roc_auc(pos, neg))
        allp = torch.cat([acts[p][:, L, :] for p in acts], dim=0)
        residual_norm.append(allp.norm(dim=1).median().item())
    auc_best = max(range(num_layers), key=lambda i: layer_auc[i])
    # AUC-best is surface-lexical (~layer 0); use the empirical STEER_LAYER if set.
    best_layer = STEER_LAYER if STEER_LAYER is not None else auc_best
    print(f"AUC-best layer {auc_best} (AUC={layer_auc[auc_best]:.3f}); "
          f"using STEER_LAYER={best_layer} "
          f"(AUC={layer_auc[best_layer]:.3f}, residual_norm={residual_norm[best_layer]:.1f})")

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gluttony, VECTOR_DIR / "gluttony.pt")
    torch.save({"gluttony": gluttony, "sloth": sloth}, VECTOR_DIR / "all_concepts.pt")
    torch.save(sloth, CONTROL_VECTOR_DIR / "sloth.pt")
    torch.save(random_vec, CONTROL_VECTOR_DIR / "random.pt")
    torch.save(verbosity, CONTROL_VECTOR_DIR / "verbosity.pt")
    meta = {
        "method": "persona_response_pooled",
        "primary_pole": PRIMARY_POLE, "anti_pole": ANTI_POLE,
        "num_layers": num_layers, "hidden_dim": hidden,
        "best_layer": best_layer, "best_layer_auc": layer_auc[best_layer],
        "layer_auc": layer_auc, "residual_norm_per_layer": residual_norm,
        "controls": ["sloth", "random", "verbosity"], "seed": args.seed,
    }
    (VECTOR_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved persona vectors + meta.json -> {VECTOR_DIR}/")


if __name__ == "__main__":
    main()
