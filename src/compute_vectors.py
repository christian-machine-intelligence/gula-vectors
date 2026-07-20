"""
Phase 1 (vectors): build the gluttony steering vector and its controls from the
extracted act activations.

Per layer:
  1. concept direction = mean(concept acts) - mean(all OTHER vices' acts)
     -> gluttony-SPECIFIC (one-vs-rest), not a generic "sin" direction.
  2. PCA-denoise against the neutral background, then L2-normalize.

Layer selection: pick the layer whose gluttony direction best separates gluttony
acts from non-gluttony acts (held-out ROC-AUC of cosine similarity). Ports the
diff-of-means + PCA pipeline from GospelVec/src/compute_vectors.py.

Controls built alongside (specificity tests):
  - sloth     : one-vs-rest direction; predicts the OPPOSITE behavioural sign
  - random    : seeded random unit vector (norm-matched after normalization)
  - verbosity : long-vs-short act contrast (a generic "more text" axis)

Outputs:
  vectors/gluttony.pt          [num_layers, hidden]
  vectors/controls/<name>.pt   [num_layers, hidden]
  vectors/all_concepts.pt      dict {sin: [num_layers, hidden]}
  vectors/meta.json            best_layer, per-layer AUC, residual norms, etc.

Usage:
  python -m src.compute_vectors --framing self
"""

from __future__ import annotations

import argparse
import json

import torch

from .config import (
    ACTIVATION_DIR, CONTROL_VECTOR_DIR, OTHER_SINS, PCA_VARIANCE_THRESHOLD,
    PRIMARY_FRAMING, SINS, TARGET_SIN, VECTOR_DIR,
)
from .common import load_acts


# ── PCA denoising (ported from GospelVec) ────────────────────────────────
def compute_pca_basis(neutral_reps: torch.Tensor,
                      variance_threshold: float = PCA_VARIANCE_THRESHOLD):
    centered = neutral_reps - neutral_reps.mean(dim=0)
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int((explained < variance_threshold).sum().item()) + 1
    return Vt[:k]


def project_out(direction: torch.Tensor, pca_basis: torch.Tensor) -> torch.Tensor:
    coeffs = direction @ pca_basis.T
    return direction - coeffs @ pca_basis


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-8)


def roc_auc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """Mann-Whitney U / AUC: P(pos score > neg score)."""
    pos, neg = scores_pos.reshape(-1), scores_neg.reshape(-1)
    wins = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return ((wins + 0.5 * ties) / (pos.numel() * neg.numel())).item()


def one_vs_rest_direction(sin_reps, concept, layer, pca_basis):
    """mean(concept) - mean(all other sins) at one layer, denoised + normalized."""
    others = [s for s in SINS if s != concept]
    concept_mean = sin_reps[concept][layer].mean(dim=0)
    other_mean = torch.cat([sin_reps[s][layer] for s in others], dim=0).mean(dim=0)
    return normalize(project_out(concept_mean - other_mean, pca_basis))


def build_all_layer(sin_reps, concept, num_layers, neutral_reps, hidden):
    out = torch.zeros(num_layers, hidden)
    for layer in range(num_layers):
        pca_basis = compute_pca_basis(neutral_reps[layer])
        out[layer] = one_vs_rest_direction(sin_reps, concept, layer, pca_basis)
    return out


def verbosity_direction(sin_reps, acts_text, num_layers, neutral_reps, hidden):
    """Long-vs-short act contrast: a generic 'more elaboration' axis."""
    # Flatten acts across sins, preserving the per-sin extraction order.
    lengths, flat_idx = [], []
    for sin in SINS:
        for j, txt in enumerate(acts_text[sin]):
            lengths.append(len(txt))
            flat_idx.append((sin, j))
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    third = len(order) // 3
    short_idx = [flat_idx[i] for i in order[:third]]
    long_idx = [flat_idx[i] for i in order[-third:]]

    out = torch.zeros(num_layers, hidden)
    for layer in range(num_layers):
        long_mean = torch.stack([sin_reps[s][layer, j] for s, j in long_idx]).mean(0)
        short_mean = torch.stack([sin_reps[s][layer, j] for s, j in short_idx]).mean(0)
        pca_basis = compute_pca_basis(neutral_reps[layer])
        out[layer] = normalize(project_out(long_mean - short_mean, pca_basis))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default=PRIMARY_FRAMING)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sin_reps = torch.load(ACTIVATION_DIR / f"acts_{args.framing}.pt", weights_only=True)
    neutral_reps = torch.load(ACTIVATION_DIR / f"neutral_{args.framing}.pt", weights_only=True)
    num_layers, hidden = sin_reps[TARGET_SIN].shape[0], sin_reps[TARGET_SIN].shape[2]
    print(f"Loaded acts ({args.framing}): {num_layers} layers, hidden={hidden}")

    # ── Concept vectors (gluttony + each other sin) ──────────────────────
    print("Building one-vs-rest concept directions ...")
    concepts = {s: build_all_layer(sin_reps, s, num_layers, neutral_reps, hidden)
                for s in SINS}
    gluttony = concepts[TARGET_SIN]

    # ── Controls ─────────────────────────────────────────────────────────
    g = torch.Generator().manual_seed(args.seed)
    random_vec = torch.stack([normalize(torch.randn(hidden, generator=g))
                              for _ in range(num_layers)])
    acts_text = load_acts(args.framing)
    verbosity = verbosity_direction(sin_reps, acts_text, num_layers, neutral_reps, hidden)

    # ── Layer selection: gluttony-vs-rest AUC of cosine(act, gluttony) ───
    print("Selecting layer by gluttony-vs-rest AUC ...")
    non_target = [s for s in SINS if s != TARGET_SIN]
    layer_auc, residual_norm = [], []
    for layer in range(num_layers):
        d = gluttony[layer]
        pos = sin_reps[TARGET_SIN][layer] @ d
        neg = torch.cat([sin_reps[s][layer] for s in non_target], dim=0) @ d
        layer_auc.append(roc_auc(pos, neg))
        # median residual norm at this layer (for alpha scaling in steer.py)
        all_acts = torch.cat([sin_reps[s][layer] for s in SINS], dim=0)
        residual_norm.append(all_acts.norm(dim=1).median().item())

    best_layer = max(range(num_layers), key=lambda i: layer_auc[i])
    print(f"  best layer {best_layer}: AUC={layer_auc[best_layer]:.3f}, "
          f"residual_norm={residual_norm[best_layer]:.1f}")

    # ── Save ─────────────────────────────────────────────────────────────
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gluttony, VECTOR_DIR / "gluttony.pt")
    torch.save(concepts, VECTOR_DIR / "all_concepts.pt")
    torch.save(concepts["sloth"], CONTROL_VECTOR_DIR / "sloth.pt")
    torch.save(random_vec, CONTROL_VECTOR_DIR / "random.pt")
    torch.save(verbosity, CONTROL_VECTOR_DIR / "verbosity.pt")

    meta = {
        "framing": args.framing,
        "target_sin": TARGET_SIN,
        "num_layers": num_layers,
        "hidden_dim": hidden,
        "best_layer": best_layer,
        "best_layer_auc": layer_auc[best_layer],
        "layer_auc": layer_auc,
        "residual_norm_per_layer": residual_norm,
        "controls": ["sloth", "random", "verbosity"],
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
        "seed": args.seed,
    }
    with open(VECTOR_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved vectors + meta.json to {VECTOR_DIR}/")


if __name__ == "__main__":
    main()
