"""Read-only artifact audit and small, non-benchmark capability probes. No code execution."""

import ast
import json
import math
import statistics
from pathlib import Path
from macoder.checkpoint import load_model
from macoder.data import file_hash
from macoder.inference import completion

root = Path("data/general-v1/prepared")
manifest = json.loads((root / "manifest.json").read_text())
model, tok, meta = load_model("runs/general-v1/step-048829")
checks = {
    name: file_hash(root / name) == manifest[key]
    for name, key in [
        ("train.bin", "train_sha256"),
        ("valid.bin", "valid_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
    ]
}
checks["checkpoint_tokenizer"] = (
    file_hash(Path("runs/general-v1/step-048829/tokenizer.json")) == manifest["tokenizer_sha256"]
)
checks["metadata_source"] = meta["source_sha256"] == manifest["source_sha256"]
checks["evaluation_manifest"] = (
    file_hash(root / "manifest.json")
    == json.loads(Path("reports/general-v1-quality.json").read_text())["manifest_sha256"]
)
checks["source_files"] = {
    n: file_hash(Path("data/general-v1") / (n + ".jsonl")) == v["sha256"]
    for n, v in manifest["sources"].items()
}
rows = [json.loads(line) for line in Path("runs/general-v1/metrics.jsonl").read_text().splitlines()]
checks["all_steps_sequential"] = [r["step"] for r in rows] == list(range(1, 48830))
checks["all_losses_gradients_finite"] = all(
    math.isfinite(r[k]) for r in rows for k in ("loss", "grad_norm")
)
qa = [
    ("What is 2 + 2?", "4"),
    ("What is 17 + 25?", "42"),
    ("What is the capital of France?", "Paris"),
    ("How many days are in a week?", "7"),
    ("What is the opposite of hot?", "cold"),
    ("What color is a ripe banana?", "yellow"),
    ("What is 3 times 4?", "12"),
    ("What is the first letter of the English alphabet?", "A"),
]
probes = []
for question, answer in qa:
    prompt = f"Question: {question}\nAnswer:"
    output = completion(model, tok, prompt, 64, 0.0)
    probes.append(
        {
            "kind": "qa",
            "prompt": prompt,
            "expected": answer,
            "output": output,
            "first_line_exact_match": output.strip().split("\n")[0].strip().rstrip(".").casefold()
            == answer.casefold(),
        }
    )
for prompt in [
    'def add(a, b):\n    """Return the sum of a and b."""\n    ',
    'def square(x):\n    """Return x multiplied by itself."""\n    ',
    'def reverse_string(text):\n    """Return text in reverse order."""\n    ',
    'def is_even(n):\n    """Return True if n is even, otherwise False."""\n    ',
]:
    output = completion(model, tok, prompt, 96, 0.0)
    try:
        ast.parse(prompt + output)
        syntax = True
    except SyntaxError:
        syntax = False
    probes.append(
        {
            "kind": "code",
            "prompt": prompt,
            "output": output,
            "whole_completion_syntax_valid": syntax,
            "executed": False,
        }
    )
for value in ["17 plus 25", "6 times 7"]:
    prompt = (
        'Use the add or multiply tool. Respond only with JSON: {"tool":"name","arguments":{"a":number,"b":number}}.\nUser: Calculate '
        + value
        + ".\nAssistant:"
    )
    output = completion(model, tok, prompt, 64, 0.0)
    try:
        parsed = json.loads(output)
        valid = isinstance(parsed, dict) and set(parsed) == {"tool", "arguments"}
    except json.JSONDecodeError:
        valid = False
    probes.append(
        {
            "kind": "tool_format",
            "prompt": prompt,
            "output": output,
            "json_shape_valid": valid,
            "tool_executed": False,
        }
    )
for target in [256, 1024, 1536]:
    prompt = "The secret code is LEMON.\n"
    while len(tok.encode(prompt).ids) < target - 40:
        prompt += "A small bird sat on a tree. "
    prompt += "\nQuestion: What is the secret code?\nAnswer:"
    output = completion(model, tok, prompt, 32, 0.0)
    probes.append(
        {
            "kind": "context_retrieval",
            "prompt_tokens": len(tok.encode(prompt).ids),
            "expected": "LEMON",
            "output": output,
            "first_line_exact_match": output.strip().split("\n")[0].strip().rstrip(".").upper()
            == "LEMON",
        }
    )
result = {
    "checks": checks,
    "training_summary": {
        "steps": len(rows),
        "first_loss": rows[0]["loss"],
        "last_100_mean": statistics.mean(r["loss"] for r in rows[-100:]),
        "first_validation": next(r["valid_loss"] for r in rows if "valid_loss" in r),
        "final_validation": rows[-1]["valid_loss"],
    },
    "protocol": "Post-hoc small diagnostic set, greedy decoding, fixed output caps. Exact first-line QA/retrieval matching. AST parsing only; NO generated code executed. Not an official benchmark or functional pass rate.",
    "probes": probes,
}
Path("reports/general-v1-audit-probes.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
