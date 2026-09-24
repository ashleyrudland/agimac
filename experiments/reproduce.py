"""Run the published recipe from the repository root. Never overwrite training runs."""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if (ROOT / "runs/general-v1").exists() or (ROOT / "runs/sft-v1").exists():
    raise SystemExit(
        "Existing training runs found. Use a fresh checkout for reproduction; see README.md for training entry points."
    )


def run(script):
    subprocess.run([sys.executable, script], check=True)


if not (ROOT / "data/stories-source/train.jsonl").exists():
    run("experiments/fetch_stories.py")
run("experiments/prepare_general.py")
run("experiments/verify_recipe.py")
run("experiments/run_general.py")
run("experiments/prepare_sft.py")
run("experiments/verify_recipe.py")
run("experiments/run_sft.py")
