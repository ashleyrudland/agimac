"""Cheap, read-only status for the offline v2 pipeline."""

import json
import os
from pathlib import Path

p = Path("runs/v2-status.json")
d = json.loads(p.read_text())
for key in ["pid", "child_pid"]:
    if key in d:
        try:
            os.kill(d[key], 0)
            d[key + "_alive"] = True
        except ProcessLookupError:
            d[key + "_alive"] = False
metrics = sorted(Path("runs").glob("general-v2*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
if metrics:
    p = metrics[-1]
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - 65536))
        lines = f.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if rows:
        import statistics

        last = rows[-1]
        rate = statistics.median(
            r["train_tokens_per_second"] for r in rows if "train_tokens_per_second" in r
        )
        d.update(
            latest=last,
            processed_tokens=last["step"] * 4096,
            progress=last["step"] / d["target_steps"],
            recent_tokens_per_second=rate,
            estimated_training_days_remaining=max(0, d["target_steps"] - last["step"])
            * 4096
            / rate
            / 86400,
        )
d["staged_source_gb"] = {
    p.name: round(p.stat().st_size / 1e9, 3) for p in Path("data/general-v2").glob("*.jsonl")
}
print(json.dumps(d, indent=2))
