"""
ICMI-026-style RESPONSE-pooled extraction (the ICMI-027 persona pivot).

For each (persona x question): generate an in-character reply (greedy), then run one
forward over prompt+reply and mean-pool the residual stream over the REPLY tokens
only. This captures the model's state while *enacting* the trait, unlike src/extract.py
which pooled over read scenarios.

Output:
  activations/persona_acts.pt   dict {persona: tensor[n_q, num_layers, hidden]}
  activations/persona_replies.json   the generated replies (for inspection / verbosity ctrl)

Usage:
  python -m src.persona_extract --max-new 96
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .config import ACTIVATION_DIR
from .common import load_model, discover_layers, input_device, chat_ids
from .extract import ActivationCollector
from .personas import PERSONAS, QUESTIONS


def build_prompt_ids(tokenizer, system_prompt, question):
    return chat_ids(tokenizer, [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": question}])


@torch.no_grad()
def reply_pooled(model, tokenizer, collector, layers_n, system_prompt, question,
                 max_new):
    dev = input_device(model)
    prompt_ids = build_prompt_ids(tokenizer, system_prompt, question).to(dev)
    P = prompt_ids.shape[1]
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tokenizer.eos_token_id)
    reply_ids = out[0, P:]
    # reply mask over full sequence: 0 for prompt, 1 for reply tokens up to (incl) first EOS
    rmask = torch.zeros(out.shape[1])
    rlen = reply_ids.shape[0]
    eos_pos = (reply_ids == tokenizer.eos_token_id).nonzero()
    rlen = (eos_pos[0].item() + 1) if eos_pos.numel() else rlen
    rmask[P:P + rlen] = 1.0

    collector.clear()
    model(out)                                   # single forward; hooks capture all layers
    pooled = torch.zeros(layers_n, model.config.text_config.hidden_size
                         if hasattr(model.config, "text_config") else model.config.hidden_size)
    for li in range(layers_n):
        hid = collector.activations[li][0].float().cpu()    # [T, hidden]
        m = rmask.unsqueeze(-1)
        pooled[li] = (hid * m).sum(0) / m.sum().clamp(min=1e-8)
    reply_text = tokenizer.decode(out[0, P:P + rlen], skip_special_tokens=True).strip()
    return pooled, reply_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS.keys()))
    args = ap.parse_args()

    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    layers, num_layers, hidden = discover_layers(model)
    collector = ActivationCollector(layers)
    print(f"Model: {num_layers} layers, hidden={hidden}; "
          f"{len(args.personas)} personas x {len(QUESTIONS)} questions")

    acts, replies = {}, {}
    for persona in args.personas:
        sys_p = PERSONAS[persona]
        print(f"\n=== persona: {persona} ===", flush=True)
        t0 = time.time()
        per_q = torch.zeros(len(QUESTIONS), num_layers, hidden)
        replies[persona] = []
        for qi, q in enumerate(QUESTIONS):
            pooled, text = reply_pooled(model, tokenizer, collector, num_layers,
                                        sys_p, q, args.max_new)
            per_q[qi] = pooled
            replies[persona].append({"q": q, "reply": text})
            if qi < 2:
                print(f"  [{q[:40]}...] -> {text[:90]}", flush=True)
        acts[persona] = per_q
        print(f"  {persona}: {tuple(per_q.shape)} in {time.time()-t0:.1f}s")

    collector.remove()
    torch.save(acts, ACTIVATION_DIR / "persona_acts.pt")
    (ACTIVATION_DIR / "persona_replies.json").write_text(json.dumps(replies, indent=2))
    print(f"\nSaved persona activations -> {ACTIVATION_DIR}/persona_acts.pt")


if __name__ == "__main__":
    main()
