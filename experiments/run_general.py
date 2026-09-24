"""One bounded background pretraining run plus automatic local evaluation."""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

BASE = Path(__file__).resolve().parents[1]
os.chdir(BASE)
RUN = BASE / "runs/general-v1"
STATUS = BASE / "runs/general-v1-status.json"
PYTHON = str(BASE / ".venv/bin/python")
STEPS = 48829  # 100,001,792 processed tokens at batch 4, sequence 512.


def status(stage, **extra):
    payload = {
        "stage": stage,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "target_steps": STEPS,
        "target_tokens": STEPS * 2048,
        **extra,
    }
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(STATUS)
    print(json.dumps(payload), flush=True)


def run(stage, arguments):
    status(stage, command=arguments)
    subprocess.run([PYTHON, *arguments], check=True)


try:
    if RUN.exists():
        raise FileExistsError("Use a new run name or the documented resume command")
    if not (BASE / "data/general-v1/prepared/manifest.json").exists():
        raise FileNotFoundError("Prepare corpus first")
    run(
        "random_baseline",
        ["experiments/evaluate_general.py", "--output", "reports/general-v1-random.json"],
    )
    # The installed CLI entry point delegates to macoder.cli.main().
    run(
        "training",
        [
            "-c",
            "from macoder.cli import main; main()",
            "train",
            "--config",
            "configs/small.json",
            "--data",
            "data/general-v1/prepared",
            "--output",
            "runs/general-v1",
            "--steps",
            str(STEPS),
            "--sequence",
            "512",
            "--batch-size",
            "4",
            "--accumulate",
            "1",
            "--warmup",
            "1000",
            "--lr",
            "0.0006",
            "--eval-every",
            "1000",
            "--eval-batches",
            "20",
            "--log-every",
            "50",
            "--dtype",
            "float32",
            "--compile",
            "--keep-last",
            "3",
        ],
    )
    model = f"runs/general-v1/step-{STEPS:06d}"
    run(
        "domain_evaluation",
        [
            "experiments/evaluate_general.py",
            "--model",
            model,
            "--output",
            "reports/general-v1-quality.json",
        ],
    )
    run(
        "grammar_evaluation",
        ["experiments/blimp.py", "--model", model, "--output", "reports/general-v1-blimp.json"],
    )
    run(
        "quantization",
        [
            "-c",
            "from macoder.cli import main; main()",
            "quantize",
            "--model",
            model,
            "--output",
            "runs/general-v1-q4",
        ],
    )
    run(
        "speed_evaluation",
        [
            "-c",
            "from macoder.cli import main; main()",
            "bench",
            "--model",
            "runs/general-v1-q4",
            "--prompt-tokens",
            "128",
            "--tokens",
            "256",
            "--repeats",
            "7",
            "--output",
            "reports/general-v1-q4-speed.json",
        ],
    )
    run(
        "quantized_quality",
        [
            "experiments/evaluate_general.py",
            "--model",
            "runs/general-v1-q4",
            "--output",
            "reports/general-v1-q4-quality.json",
        ],
    )
    status("complete", checkpoint=model, quantized_checkpoint="runs/general-v1-q4")
except BaseException as error:
    status("failed", error=str(error))
    traceback.print_exc()
    sys.exit(1)
