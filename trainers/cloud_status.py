"""Read Modal Volume progress without starting a worker or reserving a GPU."""

import argparse
import json
from pathlib import Path
import modal


def read(volume, path):
    return b"".join(volume.read_file(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="cloud-v2-145m-11b")
    a = p.parse_args()
    volume = modal.Volume.from_name("agimac-training")
    result = json.loads(read(volume, f"/pipelines/{a.run}/status.json"))
    if result["stage"] == "preparing_data":
        try:
            entries = volume.listdir("/preparation/v2/data/general-v2")
            result["staged_sources"] = [
                {"path": e.path, "bytes": e.size} for e in entries if e.path.endswith(".jsonl")
            ]
        except Exception as exc:
            result["progress_note"] = str(exc)
    elif result["stage"] == "pretraining":
        try:
            # Stream the metrics file and retain only its tail in local memory.
            tail = b""
            for chunk in volume.read_file(f"/runs/{a.run}/metrics.jsonl"):
                tail = (tail + chunk)[-65536:]
            rows = []
            for line in tail.splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            if rows:
                result["latest"] = rows[-1]
                settings = result["settings"]
                step = rows[-1]["step"]
                result["processed_tokens"] = step * settings["batch_size"] * settings["sequence"]
                result["progress_fraction"] = step / settings["steps"]
                speeds = [
                    r["train_tokens_per_second"]
                    for r in rows
                    if r.get("train_tokens_per_second", 0) > 0
                ]
                if speeds:
                    speed = len(speeds) / sum(1 / v for v in speeds)
                    result["recent_tokens_per_second"] = speed
                    result["estimated_training_hours_remaining"] = (
                        (settings["steps"] - step)
                        * settings["batch_size"]
                        * settings["sequence"]
                        / speed
                        / 3600
                    )
        except Exception as exc:
            result["progress_note"] = str(exc)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cloud-status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
