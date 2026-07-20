"""Shared helpers: act loading, model loading, residual-layer discovery, JSONL IO."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import (
    ACTS_PATH, FRAMINGS, MODEL_ID, DEVICE_MAP, USE_QUANTIZATION, SINS, TARGET_SIN,
)


# ── JSONL ────────────────────────────────────────────────────────────────
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ── Acts ─────────────────────────────────────────────────────────────────
def load_acts(framing: str = "self") -> dict[str, list[str]]:
    """Return {sin: [framed act text, ...]} from the ICMI-025 benchmark.

    Each act's `act` field is wrapped per the requested framing (config.FRAMINGS).
    """
    if framing not in FRAMINGS:
        raise ValueError(f"Unknown framing {framing!r}; choose from {list(FRAMINGS)}")
    template = FRAMINGS[framing]
    by_sin: dict[str, list[str]] = {s: [] for s in SINS}
    for row in read_jsonl(ACTS_PATH):
        sin = row["sin"]
        if sin in by_sin:
            by_sin[sin].append(template.format(act=row["act"]))
    counts = {s: len(v) for s, v in by_sin.items()}
    if counts[TARGET_SIN] == 0:
        raise RuntimeError(f"No {TARGET_SIN} acts loaded from {ACTS_PATH}")
    return by_sin


# ── Chat templating (cross-model robust) ─────────────────────────────────
def chat_ids(tokenizer, messages):
    """apply_chat_template -> [1, P] input_ids, robust across model families.

    - Qwen 3.x: pass enable_thinking=False so replies aren't <think> blocks
      (we pool activations over reply tokens — thinking text would pollute them).
    - Mistral-style templates that reject a system role: fold the system text
      into the first user message instead.
    - Normalises the BatchEncoding/dict/tensor return variants.
    """
    def _apply(msgs):
        try:
            out = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False)
        except (TypeError, ValueError):        # kwarg unsupported (Mistral raises ValueError)
            out = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(out, "input_ids"):
            out = out.input_ids
        elif isinstance(out, dict):
            out = out["input_ids"]
        return out

    try:
        return _apply(messages)
    except Exception:
        # fold system into first user turn (strict-alternation templates)
        sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if rest and rest[0]["role"] == "user" and sys_txt:
            rest = [{"role": "user",
                     "content": f"{sys_txt}\n\n{rest[0]['content']}"}] + rest[1:]
        return _apply(rest)


# ── Model ────────────────────────────────────────────────────────────────
def discover_layers(model):
    """Return (decoder_layer_module_list, num_layers, hidden_dim).

    Robust to flat text models (`model.model.layers` — Qwen/Llama/Gemma2/3-text)
    AND multimodal wrappers where the decoder lives under a nested language_model
    (Gemma 3/4 `Gemma*ForConditionalGeneration`: model.model.language_model.layers).
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.layers,
    ]
    layers = None
    for get in candidates:
        try:
            layers = get(model)
            if layers is not None and len(layers) > 0:
                break
        except AttributeError:
            continue
    if layers is None:
        raise RuntimeError("Could not locate decoder layers; module layout:\n"
                           + str(model).split("\n")[0])
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or \
        getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    return layers, len(layers), hidden


def input_device(model):
    """Device that input_ids should live on (correct for device_map='auto')."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return getattr(model, "device", "cuda")


def load_model(device_map=None):
    """Load the steering model in bf16 (or 8-bit) under transformers.

    Gemma 4 is a `Gemma4ForConditionalGeneration` (multimodal). AutoModelForCausalLM
    may not map it, so we fall back to AutoModelForImageTextToText, which loads the
    full model; the text path runs fine with text-only inputs and our hooks fire on
    the language_model decoder layers.
    """
    import torch
    from transformers import AutoTokenizer

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map=device_map or DEVICE_MAP,
        trust_remote_code=True,
    )
    if USE_QUANTIZATION:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {MODEL_ID} ({'8-bit' if USE_QUANTIZATION else 'bf16'}, "
          f"device_map={kwargs['device_map']}) ...", flush=True)
    model = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers
            Loader = getattr(transformers, loader_name)
            model = Loader.from_pretrained(MODEL_ID, **kwargs)
            print(f"  loaded via {loader_name}")
            break
        except (ValueError, KeyError) as e:
            print(f"  {loader_name} did not map ({type(e).__name__}); trying next")
    if model is None:
        raise RuntimeError(f"No auto-class could load {MODEL_ID}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer
