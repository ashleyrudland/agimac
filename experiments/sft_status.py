"""Read-only offline progress and wall-clock estimate."""

import json
import os
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / "runs/sft-v1-status.json"
if not p.exists():
    print("SFT runner has not started.")
    raise SystemExit()
s = json.loads(p.read_text())
try:
    os.kill(s["pid"], 0)
    s["process_alive"] = True
except ProcessLookupError:
    s["process_alive"] = False
p = root / "runs/sft-v1/metrics.jsonl"
rows = []
if p.exists():
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
if rows:
    last = rows[-1]
    s["latest"] = last
    if len(rows) > 20:
        first = rows[max(0, len(rows) - 501)]
        seconds = (last["elapsed_seconds"] - first["elapsed_seconds"]) / (
            last["step"] - first["step"]
        )
        s["estimated_training_hours_remaining"] = round(
            seconds * (last["total_steps"] - last["step"]) / 3600, 2
        )
    s["recent_median_actual_tokens_per_second"] = round(
        statistics.median(r["input_tokens_per_second"] for r in rows[-200:])
    )
print(json.dumps(s, indent=2))
