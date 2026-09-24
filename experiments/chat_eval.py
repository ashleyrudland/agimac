"""Resumable nanochat-style chat evaluation on MLX; never executes generated code.

Task text and scoring follow nanochat 92d63d4. Our conversation delimiters differ.
HumanEval outputs are saved unscored until a verified isolated evaluator is available.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

UPSTREAM = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
DATASETS = {
    "ARC-Easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy"),
    "ARC-Challenge": (
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Challenge",
    ),
    "MMLU": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all"),
    "GSM8K": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main"),
    "HumanEval": (
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "openai_humaneval",
    ),
}


def render_mc(question, letters, choices):
    return (
        f"Multiple Choice question: {question}\n"
        + "".join(f"- {c}={label}\n" for label, c in zip(letters, choices))
        + "\nRespond only with the letter of the correct answer."
    )


def extract_answer(text):
    match = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return match.group(1).strip().replace(",", "") if match else None


def example(task, row):
    if task.startswith("ARC"):
        letters = row["choices"]["label"]
        return (
            render_mc(row["question"], letters, row["choices"]["text"]),
            letters,
            row["answerKey"],
        )
    if task == "MMLU":
        letters = list("ABCD")
        return render_mc(row["question"], letters, row["choices"]), letters, letters[row["answer"]]
    if task == "GSM8K":
        return row["question"], None, extract_answer(row["answer"])
    return row["prompt"], None, None


def output_limit(context, prompt_length):
    return max(0, min(512, context - prompt_length))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def resume_rows(path):
    # Only an incomplete final line can be discarded after a power loss.
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        path.write_bytes(raw)
    return [json.loads(line) for line in raw.splitlines()]


def main():
    import fcntl
    import mlx.core as mx
    from macoder.checkpoint import load_model
    from macoder.conversation import render_messages
    from macoder.inference import generate_ids

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="reports/sft-v1-nanochat")
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "eval.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model_path = Path(args.model).resolve()
    fingerprints = {
        name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
        for name in ["model.safetensors", "config.json", "tokenizer.json"]
    }
    manifest = {
        "upstream": UPSTREAM,
        "model": str(model_path),
        "sha256": fingerprints,
        "datasets": DATASETS,
        "max_problems": args.max_problems,
        "temperature": 0,
        "samples": 1,
        "max_new_tokens": 512,
        "deviations": [
            "agimac tokenizer and conversation delimiters",
            "No automatic nanochat Python tool execution; GSM8K is unaided",
            "Context overflow is counted as failure; output budget shortened and recorded when necessary",
            "HumanEval generation only: isolated execution not established; no full ChatCORE",
            "Training contamination cannot be ruled out by exact overlap filtering",
        ],
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError("Resume manifest differs; use a new output directory")
    atomic_json(manifest_path, manifest)
    model, tokenizer, _ = load_model(model_path)
    model.eval()
    summary = {
        "status": "running",
        "tasks": {},
        "ChatCORE": None,
        "HumanEval_execution": "blocked: no verified isolated evaluator",
        "DCLM_CORE": "separate evaluation, not calculated here",
    }
    for task, (repo, revision, subset) in DATASETS.items():
        path = hf_hub_download(
            repo, f"{subset}/test-00000-of-00001.parquet", repo_type="dataset", revision=revision
        )
        data = pq.read_table(path).to_pylist()
        order = np.random.default_rng(42).permutation(len(data))
        if args.max_problems is not None:
            order = order[: args.max_problems]
        dest = out / f"{task}.jsonl"
        rows = resume_rows(dest)
        if [r["index"] for r in rows] != list(range(len(rows))):
            raise ValueError("Non-contiguous resume records")

        def update():
            scored = task != "HumanEval"
            summary["tasks"][task] = {
                "processed": len(rows),
                "total": len(order),
                "dataset_total": len(data),
                "correct": sum(r.get("correct") is True for r in rows) if scored else None,
                "accuracy": sum(r.get("correct") is True for r in rows) / len(rows)
                if scored and rows
                else None,
                "errors": sum("error" in r for r in rows),
                "shortened_budgets": sum(r.get("shortened_budget", False) for r in rows),
                "token_caps": sum(r.get("ended") is False for r in rows),
                "status": ("complete" if scored else "generated_unscored")
                if len(rows) == len(order)
                else "running",
            }
            atomic_json(out / "summary.json", summary)

        with dest.open("a") as handle:
            for index in range(len(rows), len(order)):
                physical = int(order[index])
                row = data[physical]
                prompt, letters, expected = example(task, row)
                ids, _ = render_messages(
                    tokenizer, [{"role": "user", "content": prompt}], generation=True
                )
                record = {
                    "index": index,
                    "dataset_index": physical,
                    "prompt": prompt,
                    "expected": expected,
                    "prompt_tokens": len(ids),
                    "subject": row.get("subject"),
                    "correct": None if task == "HumanEval" else False,
                }
                start = time.monotonic()
                try:
                    if len(ids) >= model.config.context:
                        raise ValueError("prompt_overlength")
                    if letters:
                        label_ids = [tokenizer.encode(label).ids for label in letters]
                        if any(len(label) != 1 for label in label_ids):
                            raise ValueError("answer_label_not_single_token")
                        logits = model(mx.array([ids]), last_only=True)[0, -1]
                        scores = logits[mx.array([label[0] for label in label_ids])]
                        predicted = letters[int(mx.argmax(scores).item())]
                        record.update(
                            predicted=predicted,
                            correct=predicted == expected,
                            choice_logits=scores.tolist(),
                            letters=letters,
                        )
                    else:
                        limit = output_limit(model.config.context, len(ids))
                        output = []
                        ended = False
                        stops = {
                            tokenizer.token_to_id("<|end_turn|>"),
                            tokenizer.token_to_id("<|endoftext|>"),
                        }
                        for token in generate_ids(model, ids, limit, 0.0):
                            if token in stops:
                                ended = True
                                break
                            output.append(token)
                        text = tokenizer.decode(output, skip_special_tokens=False)
                        record.update(
                            text=text,
                            output_ids=output,
                            ended=ended,
                            output_tokens=len(output),
                            output_budget=limit,
                            shortened_budget=limit < 512,
                        )
                        if task == "GSM8K":
                            prediction = extract_answer(text)
                            record.update(predicted=prediction, correct=prediction == expected)
                        else:
                            record.update(
                                task_id=row["task_id"],
                                entry_point=row["entry_point"],
                                test=row["test"],
                                execution="not_executed",
                            )
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                record["seconds"] = time.monotonic() - start
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                rows.append(record)
                if len(rows) % 25 == 0:
                    update()
                    print(task, len(rows), "/", len(order), flush=True)
            update()
    summary["status"] = "complete_with_humaneval_unscored"
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
