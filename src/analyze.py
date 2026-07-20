"""
Analysis — dose-response, specificity, and cross-domain coherence (H1.5-H4).

Reads the Phase results and produces:
  - per-(vector, alpha) means with bootstrap 95% CIs
  - dose-response slope for gluttony vs each control
  - cross-domain coherence: correlation of gluttony's search-excess and
    compute-over-provision across the shared alpha grid (H4)
  - figures/dose_response.png

Search excess is computed over CORRECT trials only (success-conditioning), so
gluttony (consume past enough) is separated from incompetence (query more).

Usage:
  python -m src.analyze --framing self
  python -m src.analyze --markdown
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import RESULTS_DIR, FIGURE_DIR, PRIMARY_FRAMING


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def bootstrap_ci(xs, n=2000, seed=0):
    import numpy as np
    xs = np.array([x for x in xs if x is not None], dtype=float)
    if len(xs) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    means = xs[rng.integers(0, len(xs), size=(n, len(xs)))].mean(axis=1)
    return float(xs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell_means(rows, value_fn, keep=lambda r: True):
    """{(vector, alpha): [values]} after filtering with `keep`."""
    cells = defaultdict(list)
    for r in rows:
        if keep(r):
            v = value_fn(r)
            if v is not None:
                cells[(r["vector"], r["alpha"])].append(v)
    return cells


def slope(alpha_to_mean):
    """OLS slope of mean vs alpha (dose-response)."""
    import numpy as np
    pts = [(a, m) for a, m in sorted(alpha_to_mean.items()) if m is not None]
    if len(pts) < 2:
        return None
    a = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
    return float(np.polyfit(a, m, 1)[0])


def summarize(cells, label, md=False):
    vectors = sorted({k[0] for k in cells})
    alphas = sorted({k[1] for k in cells})
    print(f"\n### {label}")
    header = ["vector"] + [f"a={a:+.1f}" for a in alphas] + ["slope"]
    print(" | ".join(header) if md else "  ".join(f"{h:>10}" for h in header))
    per_vec_means = {}
    for v in vectors:
        a2m = {}
        cols = []
        for a in alphas:
            mean, lo, hi = bootstrap_ci(cells.get((v, a), []))
            a2m[a] = mean
            cols.append("n/a" if mean is None else f"{mean:.2f}[{lo:.2f},{hi:.2f}]")
        per_vec_means[v] = a2m
        s = slope(a2m)
        row = [v] + cols + ["n/a" if s is None else f"{s:+.3f}"]
        print(" | ".join(row) if md else "  ".join(f"{c:>10}" for c in row))
    return per_vec_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default=PRIMARY_FRAMING)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    md = args.markdown

    # ── Search (success-conditioned excess queries) ──────────────────────
    search = read_jsonl(RESULTS_DIR / f"search_trials_{args.framing}.jsonl")
    search = [r for r in search if not r.get("malformed")]
    search_cells = cell_means(
        search, value_fn=lambda r: r.get("pages_viewed"),
        keep=lambda r: r.get("correct"))
    search_means = summarize(search_cells, "Search: pages viewed (correct trials)", md)

    # ── Compute (over-provisioning) ──────────────────────────────────────
    compute = read_jsonl(RESULTS_DIR / f"compute_trials_{args.framing}.jsonl")
    compute_cells = cell_means(compute, value_fn=lambda r: r.get("over_provision"))
    compute_means = summarize(compute_cells, "Compute: over-provision ratio", md)

    literal = read_jsonl(RESULTS_DIR / f"compute_literal_trials_{args.framing}.jsonl")
    if literal:
        lit_cells = cell_means(literal, value_fn=lambda r: r.get("gpu_seconds"))
        summarize(lit_cells, "Compute (literal): real GPU-seconds", md)

    # ── H4 cross-domain coherence (gluttony) ─────────────────────────────
    if "gluttony" in search_means and "gluttony" in compute_means:
        import numpy as np
        shared = sorted(set(search_means["gluttony"]) & set(compute_means["gluttony"]))
        xs = [search_means["gluttony"][a] for a in shared]
        ys = [compute_means["gluttony"][a] for a in shared]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) >= 2:
            x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
            print(f"\n### H4 cross-domain coherence (gluttony)\n"
                  f"  shared alphas: {shared}\n"
                  f"  corr(search-excess, compute-over-provision) = {r:+.3f}")

    _make_figure(search_means, compute_means)


def _make_figure(search_means, compute_means):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"(skipping figure: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, means, title, ylab in (
            (axes[0], search_means, "Search", "excess queries (correct)"),
            (axes[1], compute_means, "Compute", "over-provision ratio")):
        for v, a2m in means.items():
            pts = sorted((a, m) for a, m in a2m.items() if m is not None)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=v)
        ax.set_title(title); ax.set_xlabel("steering alpha"); ax.set_ylabel(ylab)
        ax.axhline(0, color="0.8", lw=0.8); ax.legend(fontsize=8)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dose_response.png", dpi=150)
    print(f"\nWrote {FIGURE_DIR / 'dose_response.png'}")


if __name__ == "__main__":
    main()
