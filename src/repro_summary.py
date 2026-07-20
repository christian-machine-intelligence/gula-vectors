"""
Per-model repro summary: consumption dose-response on the Monte Carlo GPU task.

Reads results/gpu_task_<framing>.jsonl + vectors/steer_config.json and writes
results/repro_summary.json (machine-readable, for cross-model aggregation) while
printing a human table. Median total_samples per vector x alpha over non-malformed
trials; correctness and malformed rates reported alongside.

  python -m src.repro_summary
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from .config import RESULTS_DIR, VECTOR_DIR, MODEL_ID, PRIMARY_FRAMING


def main():
    path = RESULTS_DIR / f"gpu_task_{PRIMARY_FRAMING}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    steer = json.loads((VECTOR_DIR / "steer_config.json").read_text()) \
        if (VECTOR_DIR / "steer_config.json").exists() else {}

    cells = defaultdict(list)
    stats = defaultdict(lambda: {"n": 0, "malformed": 0, "correct": 0})
    for r in rows:
        k = (r["vector"], r["alpha"])
        stats[k]["n"] += 1
        stats[k]["malformed"] += bool(r["malformed"])
        stats[k]["correct"] += bool(r["correct"])
        if not r["malformed"]:
            cells[k].append(r["total_samples"])

    vectors = sorted({v for v, _ in stats})
    alphas = sorted({a for _, a in stats})
    table = {}
    print(f"model: {MODEL_ID}")
    print(f"steer: layer={steer.get('steer_layer')} alpha_max={steer.get('alpha_max')}")
    print(f"\nMEDIAN real samples consumed (non-malformed) — {len(rows)} trials")
    print(f"{'vector':10}", *[f"{a:+.3f}" for a in alphas])
    for v in vectors:
        table[v] = {}
        out = []
        for a in alphas:
            xs = cells.get((v, a), [])
            med = statistics.median(xs) if xs else None
            table[v][str(a)] = {
                "median_samples": med,
                "n_valid": len(xs),
                "n_malformed": stats[(v, a)]["malformed"],
                "n_correct": stats[(v, a)]["correct"],
            }
            out.append(f"{med:10.3g}" if med is not None else f"{'n/a':>10}")
        print(f"{v:10}", *out)

    # headline: gluttony consumption ratio, alpha_max vs baseline
    g = table.get("gluttony", {})
    amax = str(max(alphas))
    ratio = None
    if g.get(amax, {}).get("median_samples") and g.get("0.0", {}).get("median_samples"):
        ratio = g[amax]["median_samples"] / g["0.0"]["median_samples"]
        print(f"\ngluttony consumption ratio (alpha_max / baseline): {ratio:,.0f}x")

    summary = {"model_id": MODEL_ID, "steer_config": steer, "n_trials": len(rows),
               "alphas": alphas, "table": table,
               "gluttony_ratio_max_over_baseline": ratio}
    (RESULTS_DIR / "repro_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS_DIR / 'repro_summary.json'}")


if __name__ == "__main__":
    main()
