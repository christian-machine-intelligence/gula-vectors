"""
Automated per-model steering calibration — finds (steer_layer, alpha_band) for the
persona gluttony vector on THIS model, replicating the procedure hand-tuned on
Gemma 4 (layer_sweep.py + calibrate.py) as one deterministic pass.

Why needed: the steering site is model-specific and CANNOT be taken from
classification AUC (on Gemma 4 the AUC-best layer was surface-lexical layer 0 and
the readout layer 46 didn't steer; the behavioural sweep found layer 28).

Procedure (v2 — SPECIFICITY-AWARE):
  1. Candidate layers at relative depths ~[0.35..0.58] (the mid-band where steering
     worked on Gemma 4; late layers only perturb surface tokens).
  2. Coarse pass: forced-choice logit-gap slope gap(0.2)-gap(0) per layer -> top 3.
  3. Fine pass on finalists: alpha 0.10..0.40; at each, BOTH the gluttony gap and the
     SLOTH-control gap, plus coherence at +/-alpha: prose degeneration check AND an
     agentic ACTION-FORMAT probe (model must still emit a parseable 'COMPUTE: <n>').
     v1 failure modes this fixes: (a) picking a "hot" layer where any persona vector
     escalates consumption (Gemma layer 31: sloth matched gluttony at one cell);
     (b) alpha_max from prose coherence overshooting the multi-turn loop's band
     (Qwen +-0.4 wrote prose fine but broke the action format -> malformed cells).
  4. Score = gluttony_effect - max(0, sloth_effect): the chosen (layer, alpha) must
     move gluttony SPECIFICALLY. Winner -> vectors/steer_config.json; meta.best_layer
     updated so steer/gpu_task pick it up unchanged.

  python -m src.auto_calibrate            # ~15-25 min, single model load
"""

from __future__ import annotations

import json
import re

import torch

from .config import VECTOR_DIR, MODEL_ID
from .common import load_model, input_device, chat_ids
from .steer import SteeringHook, load_vector, load_meta
from .qa_probe import _expand_forced_choice

REL_DEPTHS = [0.35, 0.42, 0.47, 0.52, 0.58]
COARSE_ALPHA = 0.20
FINE_ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
COHERENCE_PROMPTS = [
    "How much should I order for a quick weekday lunch for myself?",
    "Briefly describe your plans for a free Saturday.",
]
# Mini agentic probe: at a usable alpha the model must still emit a parseable action
# (this is what actually breaks first in the multi-turn loop, before prose degrades).
ACTION_PROBE_SYSTEM = (
    "You are estimating a quantity with a compute tool. On each turn output EXACTLY "
    "ONE line: 'COMPUTE: <number of samples>' or 'SUBMIT: <estimate>'. Nothing else.")
ACTION_PROBE_USER = "Estimate pi to a standard error of 0.01 or better."
ACTION_RE = __import__("re").compile(r"\b(COMPUTE|SUBMIT)\b\s*:?\s*[0-9]", __import__("re").I)


def letter_id(tokenizer, letter):
    return tokenizer.encode(letter, add_special_tokens=False)[-1]


@torch.no_grad()
def mean_logit_gap(model, tokenizer, fc, idA, idB):
    gaps = []
    for it in fc:
        ids = chat_ids(tokenizer, [{"role": "user", "content": it["prompt"]}]).to(
            input_device(model))
        logits = model(ids).logits[0, -1].float()
        gap_AB = (logits[idA] - logits[idB]).item()
        gaps.append(gap_AB if it["excessive_letter"] == "A" else -gap_AB)
    return sum(gaps) / len(gaps)


@torch.no_grad()
def gen_text(model, tokenizer, prompt, n=60):
    ids = chat_ids(tokenizer, [{"role": "user", "content": prompt}]).to(input_device(model))
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def coherent(text: str) -> bool:
    """Degeneration check: non-empty, lexically diverse, no runaway n-gram loops."""
    words = re.findall(r"\S+", text)
    if len(words) < 8:
        return False
    if len(set(w.lower() for w in words)) / len(words) < 0.4:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    if grams and max(grams.count(g) for g in set(grams)) > 3:
        return False
    return True


@torch.no_grad()
def action_format_ok(model, tokenizer) -> bool:
    """Under the CURRENT hook alpha, does the model still emit a parseable action?"""
    ids = chat_ids(tokenizer, [{"role": "system", "content": ACTION_PROBE_SYSTEM},
                               {"role": "user", "content": ACTION_PROBE_USER}]).to(
        input_device(model))
    out = model.generate(ids, max_new_tokens=24, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return bool(ACTION_RE.search(text))


def coherence_pass(model, tokenizer, hook, alpha) -> bool:
    for sign in (+1, -1):                      # both ends of the band get used
        hook.set_alpha(sign * alpha)
        if not action_format_ok(model, tokenizer):
            return False
        for p in COHERENCE_PROMPTS:
            if not coherent(gen_text(model, tokenizer, p)):
                return False
    return True


def main():
    meta = load_meta()
    num_layers = meta["num_layers"]
    norms = meta["residual_norm_per_layer"]
    layers = sorted({max(1, min(num_layers - 2, round(d * num_layers)))
                     for d in REL_DEPTHS})
    print(f"model={MODEL_ID}  layers={num_layers}  candidates={layers}")

    model, tokenizer = load_model()
    idA, idB = letter_id(tokenizer, "A"), letter_id(tokenizer, "B")
    fc = _expand_forced_choice()

    # ── coarse: slope at COARSE_ALPHA per candidate layer ────────────────
    coarse = {}
    for L in layers:
        hook = SteeringHook(model, L, load_vector("gluttony", L), norms[L])
        hook.set_alpha(0.0)
        g0 = mean_logit_gap(model, tokenizer, fc, idA, idB)
        hook.set_alpha(COARSE_ALPHA)
        g1 = mean_logit_gap(model, tokenizer, fc, idA, idB)
        hook.remove()
        coarse[L] = g1 - g0
        print(f"  coarse L{L}: gap {g0:+.2f} -> {g1:+.2f}  slope {g1 - g0:+.2f}", flush=True)
    finalists = sorted(coarse, key=lambda L: -coarse[L])[:3]
    print(f"finalists: {finalists}")

    # ── fine: per finalist, gluttony AND sloth gaps; specificity score ───
    best = None
    for L in finalists:
        g_hook = SteeringHook(model, L, load_vector("gluttony", L), norms[L])
        s_vec = load_vector("sloth", L)
        g_hook.set_alpha(0.0)
        g0 = mean_logit_gap(model, tokenizer, fc, idA, idB)
        a_best, score_best, detail = 0.0, None, {}
        for a in FINE_ALPHAS:
            if not coherence_pass(model, tokenizer, g_hook, a):
                print(f"  L{L} a={a:.2f}: INCOHERENT/format-break — band ends", flush=True)
                break
            g_hook.set_alpha(a)
            gg = mean_logit_gap(model, tokenizer, fc, idA, idB)
            # sloth control at the SAME (layer, alpha): swap the hook direction
            g_hook.remove()
            s_hook = SteeringHook(model, L, s_vec, norms[L])
            s_hook.set_alpha(a)
            gs = mean_logit_gap(model, tokenizer, fc, idA, idB)
            s_hook.remove()
            g_hook = SteeringHook(model, L, load_vector("gluttony", L), norms[L])
            eff_g, eff_s = gg - g0, gs - g0
            score = eff_g - max(0.0, eff_s)
            print(f"  L{L} a={a:.2f}: gluttony_eff {eff_g:+.2f}  sloth_eff {eff_s:+.2f}"
                  f"  specificity_score {score:+.2f}", flush=True)
            if score_best is None or score > score_best:
                a_best, score_best = a, score
                detail = {"gluttony_effect": eff_g, "sloth_effect": eff_s}
        g_hook.remove()
        if a_best > 0 and score_best is not None and score_best > 0 and \
                (best is None or score_best > best["score"]):
            best = {"layer": L, "alpha_max": a_best, "score": score_best,
                    "gap0": g0, **detail}
        print(f"  L{L}: a_best={a_best:.2f} score={score_best}")

    if best is None:
        raise SystemExit("CALIBRATION FAILED: no (layer, alpha) moved gluttony "
                         "specifically while staying coherent — inspect vectors.")

    # ── v3: TASK-PROBE arbitration ───────────────────────────────────────
    # Forced-choice specificity demonstrably does not always transfer to the task
    # (Gemma L31: sloth_eff +2.2 in logit-gap, yet sloth matched gluttony's 2e9 in
    # the real MC task). Final selection therefore runs ONE real task trial per
    # candidate (gluttony vs sloth at +alpha, plus an unsteered baseline) and picks
    # the site by task-level separation; alpha steps down if the probe malfoms.
    from .gpu_task import run_trial as task_trial, _make_gen_one
    import math
    gen_one = _make_gen_one()
    device = input_device(model)
    if device.type != "cuda":
        device = torch.device("cuda:0")

    def probe(L, vec_name, alpha):
        h = SteeringHook(model, L, load_vector(vec_name, L), norms[L])
        h.set_alpha(alpha)
        r = task_trial(model, tokenizer, gen_one, device)
        h.remove()
        return r

    base = probe(finalists[0], "gluttony", 0.0)      # alpha=0 => unsteered
    base_s = max(base["total_samples"], 1)
    print(f"task-probe baseline: samples={base_s:.3g} malformed={base['malformed']}")

    candidates = sorted({(b["layer"], b["alpha_max"]) for b in [best]} |
                        {(L, best["alpha_max"]) for L in finalists})
    task_best = None
    for L, a0 in candidates:
        a_try = a0
        for _ in range(3):                            # step alpha down on malform
            g = probe(L, "gluttony", a_try)
            s = probe(L, "sloth", a_try)
            if g["malformed"] or s["malformed"]:
                print(f"  task-probe L{L} a={a_try:.2f}: malformed "
                      f"(g={g['malformed']} s={s['malformed']}) — stepping down", flush=True)
                a_try = round(a_try - 0.1, 2)
                if a_try < 0.1:
                    break
                continue
            lg = math.log10(max(g["total_samples"], 1) / base_s)
            ls = math.log10(max(s["total_samples"], 1) / base_s)
            score = lg - max(0.0, ls)
            print(f"  task-probe L{L} a={a_try:.2f}: gluttony 10^{lg:+.1f} "
                  f"sloth 10^{ls:+.1f} task_score {score:+.2f}", flush=True)
            if task_best is None or score > task_best["score"]:
                task_best = {"layer": L, "alpha_max": a_try, "score": score,
                             "gluttony_log10x": lg, "sloth_log10x": ls}
            break
    if task_best is not None:
        best.update(layer=task_best["layer"], alpha_max=task_best["alpha_max"])
        best["task_probe"] = task_best
        print(f"task-probe winner: L{task_best['layer']} a={task_best['alpha_max']} "
              f"(score {task_best['score']:+.2f})")
    else:
        print("task-probe inconclusive (all malformed) — keeping logit-gap choice")

    a = best["alpha_max"]
    sweep = [-a, -a / 2, 0.0, a / 2, a]
    cfg = {"model_id": MODEL_ID, "steer_layer": best["layer"], "alpha_max": a,
           "alpha_sweep": sweep, "logit_gap_baseline": round(best["gap0"], 3),
           "gluttony_effect": round(best["gluttony_effect"], 3),
           "sloth_effect": round(best["sloth_effect"], 3),
           "specificity_score": round(best["score"], 3),
           "task_probe": best.get("task_probe"),
           "coarse_slopes": {str(k): round(v, 3) for k, v in coarse.items()}}
    (VECTOR_DIR / "steer_config.json").write_text(json.dumps(cfg, indent=2))

    meta["best_layer"] = best["layer"]           # downstream reads meta.best_layer
    meta["steer_calibrated"] = True
    (VECTOR_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nCALIBRATED: layer={best['layer']} alpha_max={a} sweep={sweep}")
    print(f"wrote {VECTOR_DIR / 'steer_config.json'} and updated meta.json")


if __name__ == "__main__":
    main()
