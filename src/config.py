"""
Configuration for the Gula study — a domain-general steering vector for the
capital sin of gluttony (immoderate appetite).

Extraction reuses the ICMI-025 (sin-and-safety) 700-act benchmark: 100 gluttony
acts vs 600 acts across the other six capital sins. The gluttony direction is
mean(gluttony) - mean(other six vices), so it is gluttony-SPECIFIC rather than a
generic "sin" or "arousal" direction.

Model: Gemma 4 31B dense (Apache 2.0). Fully dense (the 26B-A4B sibling is MoE and
destabilises a fixed steering direction). Runs under plain transformers + forward
hooks (not vLLM) so the residual-stream steering hook can stay active during
generation.
"""

import json
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ACTS_PATH = DATA_DIR / "acts.jsonl"          # 700-act benchmark from ICMI-025
ACTIVATION_DIR = PROJECT_DIR / "activations"
VECTOR_DIR = PROJECT_DIR / "vectors"
CONTROL_VECTOR_DIR = VECTOR_DIR / "controls"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"

# ── Model (cross-model repro: one model per host, set via env) ────────────
# GULA_MODEL_ID selects the model; each host runs exactly one, so all on-disk
# artifacts (activations/vectors/results) are per-model by construction.
#   google/gemma-4-31b-it                          (original result)
#   Qwen/Qwen3.5-27B                               (2nd family)
#   mistralai/Mistral-Small-3.2-24B-Instruct-2506  (3rd family)
# Layer/hidden dims are discovered at runtime (common.discover_layers) — Gemma 4
# is multimodal with decoder layers under model.model.language_model.
MODEL_ID = os.environ.get("GULA_MODEL_ID", "google/gemma-4-31b-it")

# ~31B bf16 needs ~62GB: three 24GB GPUs via device_map="auto", or one 128GB host.
USE_QUANTIZATION = False   # set True (8-bit) if bf16 sharding is tight
DEVICE_MAP = "auto"

# Judge model for Phase 1.5 / scoring (API). Read key from .env.
JUDGE_MODEL = "claude-opus-4-8"

# ── The seven capital sins (matches ICMI-025 common.py) ──────────────────
SINS = ["pride", "greed", "lust", "envy", "gluttony", "wrath", "sloth"]
SIN_LATIN = {
    "pride": "superbia", "greed": "avaritia", "lust": "luxuria", "envy": "invidia",
    "gluttony": "gula", "wrath": "ira", "sloth": "acedia",
}
TARGET_SIN = "gluttony"
OTHER_SINS = [s for s in SINS if s != TARGET_SIN]

# Primary extraction framing. ICMI-025 ships abstract/user/self framings; "self"
# attributes the act to the agent itself, the closest match to steering the
# agent's OWN consumption behaviour.
PRIMARY_FRAMING = "self"

# Per-framing wrappers applied to each act's `act` text before encoding.
FRAMINGS = {
    "abstract": "{act}",
    "user": "The user did the following: {act}",
    "self": "I did the following: {act}",
}

# ── Extraction ───────────────────────────────────────────────────────────
MEAN_POOL_SKIP_TOKENS = 4        # skip BOS/special tokens when mean-pooling
MAX_ACT_TOKENS = 128             # acts are short single sentences

# ── PCA denoising ────────────────────────────────────────────────────────
PCA_VARIANCE_THRESHOLD = 0.50    # remove components explaining up to 50% of neutral variance

# ── Steering ─────────────────────────────────────────────────────────────
# Steering layer + alpha band are chosen EMPIRICALLY per model by behavioural effect
# (src/auto_calibrate.py: logit-gap dose-response + coherence check), never by
# classification AUC — on Gemma 4 the AUC-best layer was surface-lexical (layer 0)
# and the best readout layer (46) did not steer, while layer 28 did.
# auto_calibrate writes vectors/steer_config.json; loaded here if present.
# (Reference point: Gemma 4 31B-it → layer 28, |alpha| <= 0.25.)
_STEER_CONFIG_PATH = VECTOR_DIR / "steer_config.json"
if _STEER_CONFIG_PATH.exists():
    _sc = json.loads(_STEER_CONFIG_PATH.read_text())
    STEER_LAYER = _sc["steer_layer"]
    ALPHA_SWEEP = _sc["alpha_sweep"]
else:                       # pre-calibration defaults (persona_vectors falls back to AUC)
    STEER_LAYER = None
    ALPHA_SWEEP = [-0.25, -0.125, 0.0, 0.125, 0.25]

# Control vectors built alongside gluttony (specificity tests).
CONTROL_VECTORS = ["sloth", "random", "verbosity"]

# ── Neutral text for PCA denoising (factual, appetite-free) ──────────────
NEUTRAL_TEXTS = [
    "Water is composed of two hydrogen atoms and one oxygen atom, forming a molecule with the chemical formula H2O.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Iron is a chemical element with symbol Fe and atomic number 26.",
    "Cellular respiration converts glucose and oxygen into carbon dioxide, water, and adenosine triphosphate.",
    "Tectonic plates are massive segments of Earth's lithosphere that move and sometimes fracture.",
    "Binary code represents instructions using the binary number system's two symbols, zero and one.",
    "Gravity is a fundamental force that attracts any two objects with mass.",
    "The periodic table organizes chemical elements by increasing atomic number.",
    "An algorithm is a finite sequence of well-defined instructions used to solve a class of problems.",
    "The mitochondria are membrane-bound organelles that generate most of the cell's supply of ATP.",
    "A semiconductor has electrical conductivity between that of a conductor and an insulator.",
    "Ohm's law states that current through a conductor is directly proportional to the voltage across it.",
    "The electromagnetic spectrum is the range of frequencies of electromagnetic radiation.",
    "A prime number is a natural number greater than one with no positive divisors other than one and itself.",
    "Newton's third law states that for every action there is an equal and opposite reaction.",
    "Convection is the transfer of heat through the movement of fluids.",
    "The Fibonacci sequence begins with zero and one; each subsequent number is the sum of the two preceding.",
    "A transistor is a semiconductor device used to amplify or switch electronic signals.",
]
