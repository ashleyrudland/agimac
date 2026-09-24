"""Additional hand-written tool diagnostics, not an official benchmark."""

import json
from pathlib import Path
from macoder.checkpoint import load_model
from macoder.conversation import reply, TOOL_SYSTEM

model, tok, _ = load_model("runs/cloud-v2-sft-v1/step-005172")
cases = [
    ("Please use a calculator: what is 17 plus 25?", 42),
    ("I bought 3 items at 7 pounds each. Use the calculator for the total.", 21),
    ("Use your calculator to split 84 equally between 7 people.", 12),
    ("Calculate 100 minus 37 using a tool.", 63),
    ("Use the calculator to multiply 12 and 9.", 108),
    ("Hello! How are you?", None),
]
rows = []
for prompt, expected in cases:
    history = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": prompt}]
    first = reply(model, tok, history, 128)
    entry = {"prompt": prompt, "expected_numeric": expected, "first": first}
    if "tool_call" in first:
        history.extend(
            [
                {"role": "assistant", "tool_call": first["tool_call"]},
                {
                    "role": "tool",
                    "content": json.dumps(first["tool_result"], separators=(",", ":")),
                },
            ]
        )
        entry["final"] = reply(model, tok, history, 128)
    rows.append(entry)
Path("reports/cloud-sft-tool-paraphrases.json").write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    print(row)
