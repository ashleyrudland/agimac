"""Small ARC pilot using nanochat's prompt and restricted answer-logit scoring."""

import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import mlx.core as mx
from huggingface_hub import hf_hub_download
from macoder.checkpoint import load_model
from macoder.conversation import render_messages

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
revision = "210d026faf9955653af8916fad021475a3f00453"
model, tok, _ = load_model(a.model)
model.eval()
results = {}
for subset in ["ARC-Easy", "ARC-Challenge"]:
    path = hf_hub_download(
        "allenai/ai2_arc",
        f"{subset}/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=revision,
    )
    data = pq.read_table(path).to_pylist()
    indices = np.random.default_rng(42).permutation(len(data))[:100]
    rows = []
    for index in indices:
        r = data[int(index)]
        letters = r["choices"]["label"]
        choices = r["choices"]["text"]
        prompt = (
            f"Multiple Choice question: {r['question']}\n"
            + "".join(f"- {c}={label}\n" for label, c in zip(letters, choices))
            + "\nRespond only with the letter of the correct answer."
        )
        ids, _ = render_messages(tok, [{"role": "user", "content": prompt}], generation=True)
        encoded = [tok.encode(label).ids for label in letters]
        if any(len(e) != 1 for e in encoded):
            raise ValueError("Answer labels must be single tokens")
        entry = {
            "id": r["id"],
            "prompt": prompt,
            "expected": r["answerKey"],
            "prompt_tokens": len(ids),
        }
        if len(ids) > model.config.context:
            entry.update(predicted=None, correct=False, error="overlength")
        else:
            logits = model(mx.array([ids]), last_only=True)[0, -1]
            scores = logits[mx.array([e[0] for e in encoded])]
            pred = letters[int(mx.argmax(scores).item())]
            entry.update(
                predicted=pred, correct=pred == r["answerKey"], choice_logits=scores.tolist()
            )
        rows.append(entry)
    results[subset] = {
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "chance_mean": float(np.mean([1 / len(data[int(i)]["choices"]["label"]) for i in indices])),
        "outputs": rows,
    }
    print(subset, results[subset]["correct"], "/", len(rows), flush=True)
Path(a.output).write_text(
    json.dumps(
        {
            "model": a.model,
            "dataset_revision": revision,
            "protocol": "100 seeded test examples per task; nanochat render_mc text and restricted single-label logits, our chat delimiters. Subset diagnostic, not full suite or CORE. No training-overlap guarantee.",
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
