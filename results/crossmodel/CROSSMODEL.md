# ICMI-027 cross-model result (v3 final, 2026-07-19)

One vector (gluttony, persona method: mean(gluttony persona) − mean(temperance
persona) over reply-pooled activations), one endpoint (real Monte Carlo GPU
consumption: agent chooses samples per COMPUTE call, target se ≤ 0.01 ≈ 2.7e4
samples), one **identical automated calibration** per model (`auto_calibrate.py` v3:
coarse logit-gap sweep over mid-depth layers → coherence + action-format α band →
**task-probe arbitration**: final site chosen by gluttony-vs-sloth separation on a
real task trial). 75 trials per model (gluttony/sloth/random × 5 α × 5).

## Headline table (median real samples consumed)

| family | site | rel. depth | α band | baseline | gluttony @ +α_max | ratio | sloth @ +α_max | random |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B-it | L28/60 | 47% | ±0.15 | 1e6 | 1e9 | **1,000×** | 1.25e5 (↓ below baseline) | never escalates |
| Qwen3.5-27B | L27/64 | 42% | ±0.30 | 1e5 | 1e9 | **10,000×** | 1e6 (10×) | flat |
| Mistral-Small-3.2-24B | L19/40 | 48% | ±0.40 | ~2e6 † | 1.11e9 | **~500×** † | 6e7 (30×) | valid only unsteered |

† Mistral's unsteered gluttony cell was majority-malformed (stochastic action-format
fragility); baseline proxied by random@α=0 (2e6) and the −α cells (2–2.1e6), which
agree. Qualified accordingly in the paper.

## Reading

1. **The transfer generalises.** In all three families the food/drink-derived
   gluttony vector drives a monotone escalation of *real* compute consumption,
   converging near the per-call cap (~1e9–2e9 samples) at α_max — 3–4 orders of
   magnitude over baseline — while negative α (temperance pole) suppresses
   consumption below baseline (Gemma 1e6→5e4; Qwen 1e5→6e4).
2. **It is gluttony-specific.** At matched (site, α): sloth stays flat, drops, or
   moves ≤30×; random never escalates (it either leaves consumption flat or breaks
   the action format). Separation gluttony-vs-best-control ≥ 17× (Mistral) up to
   ≥1,000× (Gemma, where sloth *falls* while gluttony gains 1,000×).
3. **The steering site is consistent:** 42–48% relative depth in every family —
   suggesting the actionable representation of appetite lives mid-network, with
   late layers (readout-best by AUC) steering nothing.
4. **Methodological finding:** forced-choice (stated-preference) specificity does
   NOT predict task specificity — Gemma L31 was specific on logit-gaps yet sloth
   matched gluttony's 2e9 escalation in the task; the v3 task-probe rejected it
   and independently recovered the hand-found June site (L28). Calibrate on the
   behavioural endpoint, not a proxy.

## Provenance

- Per-model artifacts: `<model>_repro_summary.json`, `<model>_steer_config.json`,
  `<model>_gpu_task_self.jsonl` (raw trials), `<model>_REPRO_SUMMARY.txt`.
- Earlier calibrator generations (v1 = raw-effect selection, hot-layer confound;
  v2 = stated-preference specificity, proxy divergence demonstrated) are described
  in the paper's methods; this release ships the final v3 runs.
