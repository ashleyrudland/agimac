"""Offline held-out assistant loss and explicitly non-official chat diagnostics."""

import argparse
import json
from pathlib import Path
from macoder.checkpoint import load_model
from macoder.conversation import reply
from macoder.sft import Batches, evaluate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
model, tok, meta = load_model(a.model)
model.eval()
result = {
    "generation": {"temperature": 0, "qa_max_tokens": 128, "code_max_tokens": 256},
    "model": a.model,
    "chat_format": meta.get("chat_format"),
    "test_assistant_loss": evaluate(model, Batches("data/sft-v1", "test", 4)),
    "limitations": "Small local diagnostics, not official benchmark scores. Generated code is not executed. No long-context claim.",
}
qa = [
    ("What is the capital of France?", "paris"),
    ("What is 2 + 3?", "5"),
    ("What is the opposite of hot?", "cold"),
    ("How many days are in a week?", "7"),
    ("Which planet do we live on?", "earth"),
]
result["qa"] = []
for question, answer in qa:
    r = reply(model, tok, [{"role": "user", "content": question}], 128)
    r.update(
        question=question,
        expected=answer,
        strict_exact=r["text"].strip().lower().rstrip(".") == answer,
    )
    result["qa"].append(r)
result["calculator"] = []
rows = [json.loads(s) for s in Path("data/sft-v1/test.jsonl").read_text().splitlines()]
for row in [r for r in rows if r["source"] == "verified_calculator"][:40]:
    messages = row["messages"]
    r = reply(model, tok, messages[:2], 128)
    entry = {
        "prompt": messages[1]["content"],
        "expected_call": messages[2]["tool_call"],
        "first_reply": r,
        "correct_call": r.get("tool_call") == messages[2]["tool_call"],
        "correct_final": False,
    }
    if "tool_call" in r:
        follow = messages[:2] + [
            {"role": "assistant", "tool_call": r["tool_call"]},
            {"role": "tool", "content": json.dumps(r["tool_result"], separators=(",", ":"))},
        ]
        final = reply(model, tok, follow, 128)
        entry["final_reply"] = final
        try:
            entry["correct_final"] = entry["correct_call"] and float(
                final["text"].strip()
            ) == float(messages[-1]["content"])
        except ValueError:
            pass
    result["calculator"].append(entry)
result["coding_outputs"] = []
for task in [
    "Write a Python function square(x) that returns x squared.",
    "Write a Python function reverse_string(s) that reverses a string.",
    "Write a Python function is_even(n) that returns a boolean.",
    "Write a Python function add(a, b) that returns their sum.",
]:
    result["coding_outputs"].append(
        {"prompt": task, "reply": reply(model, tok, [{"role": "user", "content": task}], 256)}
    )
result["heldout_outputs"] = []
for row in [r for r in rows if r["source"] != "verified_calculator"][:10]:
    messages = row["messages"]
    result["heldout_outputs"].append(
        {"messages": messages, "reply": reply(model, tok, messages[:-1], 256)}
    )
result["qa_total"] = len(result["qa"])
result["calculator_total"] = len(result["calculator"])
result["qa_exact_count"] = sum(r["strict_exact"] for r in result["qa"])
result["calculator_success_count"] = sum(r["correct_final"] for r in result["calculator"])
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k.endswith("count") or k == "test_assistant_loss"}
    ),
    flush=True,
)
