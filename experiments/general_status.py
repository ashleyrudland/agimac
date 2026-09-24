"""Read background progress without loading a model or making network requests."""

import datetime
import json
from pathlib import Path
import statistics
import os

base = Path(__file__).resolve().parents[1]
path = base / "runs/general-v1-status.json"
if not path.exists():
    print("Training not launched yet. Preparation log: runs-general-preparation.log")
    raise SystemExit()
status = json.loads(path.read_text())
print("Stage:", status["stage"])
if status["stage"] not in ("complete", "failed"):
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        print("WARNING: runner is no longer alive; saved stage is stale. Inspect the log.")
metrics = base / "runs/general-v1/metrics.jsonl"
if metrics.exists():
    rows = []
    with metrics.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if rows:
        row = rows[-1]
        rate = statistics.median(r["train_tokens_per_second"] for r in rows[-100:])
        left = max(0, status["target_steps"] - row["step"]) * 2048
        print(
            f"Step {row['step']:,}/{status['target_steps']:,} | {row['step'] / status['target_steps']:.1%}"
        )
        print(f"Loss {row['loss']:.4f} | recent training {rate:,.0f} tokens/sec")
        print(
            f"Remaining optimizer time ~{left / rate / 3600:.1f} hours (excludes evaluation/saves)"
        )
        dev = [r for r in rows if "valid_loss" in r]
        if dev:
            print("Latest development loss:", dev[-1]["valid_loss"])
        if status["stage"] == "training":
            eta = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=left / rate)
            print("Approximate training finish:", eta.strftime("%a %H:%M %Z"))
print("Log:", base / "runs-general-v1.log")
