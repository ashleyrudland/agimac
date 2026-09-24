"""Bounded unattended MLX SFT followed by serial local evaluations."""

import datetime
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = str(ROOT / ".venv/bin/python")
STATUS = ROOT / "runs/sft-v1-status.json"


def status(stage, **extra):
    value = {
        "stage": stage,
        "pid": os.getpid(),
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **extra,
    }
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(STATUS)
    print(json.dumps(value), flush=True)


def run(stage, args):
    status(stage, command=args)
    subprocess.run([PY, *args], check=True)


try:
    if Path("runs/sft-v1").exists():
        raise FileExistsError("Run already exists; use documented explicit resume")
    # Fixed reference recipe: measured batch 8 on the original 32GB M2 Pro.
    run("training", ["-m", "macoder.sft", "--batch-size", "8", "--epochs", "2"])
    model = Path("runs/sft-v1/best.txt").read_text().strip()
    run(
        "base_chat_diagnostics",
        [
            "experiments/evaluate_sft.py",
            "--model",
            "runs/general-v1/step-048829",
            "--output",
            "reports/sft-v1-base-chat.json",
        ],
    )
    run(
        "chat_evaluation",
        ["experiments/evaluate_sft.py", "--model", model, "--output", "reports/sft-v1-chat.json"],
    )
    run(
        "domain_regression",
        [
            "experiments/evaluate_general.py",
            "--model",
            model,
            "--output",
            "reports/sft-v1-domain.json",
        ],
    )
    run(
        "grammar_evaluation",
        ["experiments/blimp.py", "--model", model, "--output", "reports/sft-v1-blimp.json"],
    )
    cli = ["-c", "from macoder.cli import main; main()"]
    run("quantization", cli + ["quantize", "--model", model, "--output", "runs/sft-v1-q4"])
    run(
        "speed_evaluation",
        cli
        + [
            "bench",
            "--model",
            "runs/sft-v1-q4",
            "--prompt-tokens",
            "128",
            "--tokens",
            "256",
            "--repeats",
            "7",
            "--output",
            "reports/sft-v1-q4-speed.json",
        ],
    )
    run(
        "quantized_chat_evaluation",
        [
            "experiments/evaluate_sft.py",
            "--model",
            "runs/sft-v1-q4",
            "--output",
            "reports/sft-v1-q4-chat.json",
        ],
    )
    status("complete", checkpoint=model, quantized_checkpoint="runs/sft-v1-q4")
except BaseException as e:
    status("failed", error=str(e))
    traceback.print_exc()
    sys.exit(1)
