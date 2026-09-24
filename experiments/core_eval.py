"""Nanochat DCLM CORE protocol on MLX, with resumable per-example evidence.

Raw-text prompts (no chat wrappers), identical few-shot sampling, mean continuation
loss for choices, and teacher-forced exact token match for language-modeling tasks.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import yaml
from macoder.checkpoint import load_model
from core_protocol import (
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    batch_sequences_lm,
)
from chat_eval import atomic_json, resume_rows


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_bos_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    def __call__(self, prompts, prepend):
        return [[prepend] + self.tokenizer.encode(p).ids for p in prompts]


def crop(tokens, starts, ends, context):
    result = []
    for ids, start, end in zip(tokens, starts, ends):
        removed = max(0, len(ids) - context)
        start -= removed
        end -= removed
        # Unlike upstream, explicitly reject an answer with no predictive predecessor.
        if start < 1 or end <= start:
            raise ValueError("Continuation cannot be scored within context")
        result.append((ids[removed:], start, end, removed))
    return result


def score_sequence(model, ids, start, end):
    logits = model(mx.array([ids[:-1]]))[0, start - 1 : end - 1].astype(mx.float32)
    targets = mx.array(ids[start:end])
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    exact = mx.all(mx.argmax(logits, axis=-1) == targets)
    mx.eval(loss, exact)
    return float(loss.item()), bool(exact.item())


def prepare(index, data, meta, tokenizer):
    fewshot = []
    count = meta["num_fewshot"][0]
    if count:
        indices = random.Random(1234 + index).sample(
            [i for i in range(len(data)) if i != index], count
        )
        fewshot = [data[i] for i in indices]
    kind = meta["icl_task_type"]
    render, batch = {
        "multiple_choice": (render_prompts_mc, batch_sequences_mc),
        "schema": (render_prompts_schema, batch_sequences_schema),
        "language_modeling": (render_prompts_lm, batch_sequences_lm),
    }[kind]
    prompts = render(data[index], meta.get("continuation_delimiter", " "), fewshot)
    return prompts, batch(tokenizer, prompts)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bundle", default="data/core-eval/eval_bundle")
    p.add_argument(
        "--max-per-task", type=int, default=0, help="Diagnostic only; zero means full suite"
    )
    p.add_argument("--wait-pid", type=int)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "eval.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_pid:
        import os

        print("Waiting for existing evaluation PID", args.wait_pid, flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(20)
    root = Path(args.bundle)
    tasks = yaml.safe_load((root / "core.yaml").read_text())["icl_tasks"]
    baselines = {
        r["Eval Task"]: float(r["Random baseline"]) / 100
        for r in csv.DictReader((root / "eval_meta_data.csv").open())
    }
    checkpoint = Path(args.model).resolve()
    manifest = {
        "upstream": "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd",
        "model": str(checkpoint),
        "model_hashes": {
            f: hashlib.sha256((checkpoint / f).read_bytes()).hexdigest()
            for f in ["model.safetensors", "tokenizer.json", "config.json"]
        },
        "bundle": json.loads((root.parent / "provenance.json").read_text()),
        "max_per_task": args.max_per_task,
        "fewshot_pool": "full shuffled dataset, including for diagnostic subsets",
        "protocol": "nanochat CORE raw text; seed 1337 shuffle; seed 1234+index few-shot; mean token NLL MC/schema; teacher-forced exact LM; task-mean chance-centered accuracy",
        "limitations": [
            "agimac tokenizer and BOS differ from nanochat",
            "left cropping at model context recorded",
            "unscorable examples count as failures; any errors invalidate strict comparability",
            "upstream notes SQuAD differs from its reference; this follows nanochat implementation",
            "benchmark-overlap filtering does not prove semantic decontamination",
        ],
    }
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Manifest differs; choose new output directory")
    atomic_json(out / "manifest.json", manifest)
    model, tok, _ = load_model(checkpoint)
    model.eval()
    tokenizer = TokenizerAdapter(tok)
    summary = {
        "status": "running",
        "tasks": {},
        "CORE": None,
        "full_suite": not bool(args.max_per_task),
    }
    for task in tasks:
        label = task["label"]
        data = [
            json.loads(s)
            for s in (root / "eval_data" / task["dataset_uri"]).read_text().splitlines()
            if s.strip()
        ]
        random.Random(1337).shuffle(data)
        # Limit scored examples, preserving the full upstream few-shot sampling pool.
        limit = min(len(data), args.max_per_task) if args.max_per_task else len(data)
        dest = out / f"{label}.jsonl"
        records = resume_rows(dest)
        if [r["index"] for r in records] != list(range(len(records))):
            raise ValueError("Resume order mismatch")

        def update():
            accuracy = sum(r["correct"] for r in records) / len(records) if records else 0
            base = baselines[label]
            summary["tasks"][label] = {
                "processed": len(records),
                "total": limit,
                "accuracy": accuracy,
                "random_baseline": base,
                "centered": (accuracy - base) / (1 - base),
                "errors": sum("error" in r for r in records),
                "cropped_examples": sum(r.get("cropped_tokens", 0) > 0 for r in records),
            }
            atomic_json(out / "summary.json", summary)

        with dest.open("a") as handle:
            for i in range(len(records), limit):
                record = {"index": i, "correct": False, "gold": data[i].get("gold")}
                try:
                    prompts, (tokens, starts, ends) = prepare(i, data, task, tokenizer)
                    record["prompt_sha256"] = [
                        hashlib.sha256(s.encode()).hexdigest() for s in prompts
                    ]
                    record["original_lengths"] = list(map(len, tokens))
                    sequences = crop(tokens, starts, ends, model.config.context)
                    record["cropped_tokens"] = sum(seq[3] for seq in sequences)
                    scores = [
                        score_sequence(model, ids, start, end) for ids, start, end, _ in sequences
                    ]
                    record["mean_losses"] = [s[0] for s in scores]
                    record["continuation_exact"] = [s[1] for s in scores]
                    if task["icl_task_type"] == "language_modeling":
                        record["correct"] = scores[0][1]
                    else:
                        predicted = min(range(len(scores)), key=lambda j: scores[j][0])
                        record.update(predicted=predicted, correct=predicted == data[i]["gold"])
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                if len(records) % 50 == 0:
                    update()
                    print(label, len(records), "/", limit, flush=True)
            update()
        print(label, summary["tasks"][label], flush=True)
    summary["CORE"] = sum(r["centered"] for r in summary["tasks"].values()) / len(tasks)
    summary["status"] = "complete"
    summary["full_protocol_without_errors"] = not args.max_per_task and not any(
        r["errors"] for r in summary["tasks"].values()
    )
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
