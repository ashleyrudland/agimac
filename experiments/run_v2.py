"""Offline 145M experiment: prepare 2B tokens, train 11B, then conversations/tools.

Re-running this script resumes the newest atomic checkpoint into a new directory.
Failures stop the pipeline; it never silently restarts random weights.
"""

import datetime
import fcntl
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from macoder.data import file_hash

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = str(ROOT / ".venv/bin/python")
STATUS = Path("runs/v2-status.json")
Path("runs").mkdir(exist_ok=True)
lock = Path("runs/v2.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
STEPS = math.ceil(11_000_000_000 / (8 * 512))


def status(stage, **kwargs):
    d = {
        "stage": stage,
        "pid": os.getpid(),
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_tokens": STEPS * 4096,
        "target_steps": STEPS,
        "parameters": 144979968,
        **kwargs,
    }
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2) + "\n")
    tmp.replace(STATUS)
    print(json.dumps(d), flush=True)


def run(stage, args):
    child = subprocess.Popen([PY, *args])
    status(stage, child_pid=child.pid, command=args)
    code = child.wait()
    if code:
        raise RuntimeError(f"{stage} exited {code}")


def dirs(prefix):
    return sorted(
        p
        for p in Path("runs").glob(prefix + "*")
        if p.is_dir() and (p.name == prefix or p.name.startswith(prefix + "-resume-"))
    )


def checkpoint(prefix):
    candidates = [
        p for d in dirs(prefix) for p in d.glob("step-*") if (p / "trainer.json").exists()
    ]
    return (
        max(candidates, key=lambda p: json.loads((p / "metadata.json").read_text())["step"])
        if candidates
        else None
    )


def new_output(prefix):
    base = Path("runs") / prefix
    if not base.exists():
        return str(base)
    for n in range(1, 1000):
        p = Path(f"runs/{prefix}-resume-{n:03d}")
        if not p.exists():
            return str(p)
    raise RuntimeError("Too many resumes")


try:
    plan = json.loads(Path("recipes/v2.json").read_text())
    for name, digest in plan["input_hashes"].items():
        if file_hash(name) != digest:
            raise RuntimeError("Frozen run input changed: " + name)
    run("preparing_data", ["experiments/prepare_v2.py"])
    manifest = json.loads(Path("data/general-v2/prepared/manifest.json").read_text())
    if manifest["tokens"]["train"] != 2_000_000_000:
        raise RuntimeError("Expanded mixture is not complete")
    if manifest["tokenizer_sha256"] != plan["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer changed")
    for name in ["train", "valid"]:
        if file_hash(f"data/general-v2/prepared/{name}.bin") != manifest[f"{name}_sha256"]:
            raise RuntimeError("Packed data hash mismatch")
    # Keep the existing requested CORE work exclusive on the GPU until it finishes.
    old = Path("reports/core-eval-process.json")
    if old.exists():
        pid = json.loads(old.read_text())["pid"]
        status("waiting_for_previous_core", previous_pid=pid)
        while True:
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
            ).stdout
            if "experiments/run_core_eval.py" not in command:
                break
            time.sleep(30)
    saved = checkpoint("general-v2")
    if saved is None or json.loads((saved / "metadata.json").read_text())["step"] < STEPS:
        output = new_output("general-v2")
        command = [
            "-c",
            "from macoder.cli import main; main()",
            "train",
            "--config",
            "configs/nano-core-145m.json",
            "--data",
            "data/general-v2/prepared",
            "--output",
            output,
            "--steps",
            str(STEPS),
            "--sequence",
            "512",
            "--batch-size",
            "8",
            "--accumulate",
            "1",
            "--lr",
            "0.0003",
            "--warmup",
            "2000",
            "--eval-every",
            "10000",
            "--eval-batches",
            "32",
            "--log-every",
            "100",
            "--dtype",
            "bfloat16",
            "--master-weights",
            "--compile-step",
            "--keep-last",
            "3",
        ]
        if saved:
            command += ["--resume", str(saved)]
        run("pretraining", command)
    base = checkpoint("general-v2")
    if base is None:
        raise RuntimeError("Missing completed base checkpoint")
    if not Path("reports/general-v2-quality.json").exists():
        run(
            "base_domain_eval",
            [
                "experiments/evaluate_general.py",
                "--model",
                str(base),
                "--output",
                "reports/general-v2-quality.json",
            ],
        )
    done = [d for d in dirs("sft-v2") if (d / "summary.json").exists()]
    if not done:
        prior = checkpoint("sft-v2")
        output = new_output("sft-v2")
        command = [
            "-m",
            "macoder.sft",
            "--model",
            str(base),
            "--data",
            "data/sft-v1",
            "--output",
            output,
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--lr",
            "0.00005",
            "--warmup",
            "200",
            "--eval-every",
            "1000",
            "--patience",
            "3",
        ]
        if prior:
            command += ["--resume", str(prior)]
        run("conversation_and_calculator_training", command)
        done = [Path(output)]
    selected = Path((done[-1] / "best.txt").read_text().strip())
    # Never label a base checkpoint as a functioning chat model if SFT failed selection.
    meta = json.loads((selected / "metadata.json").read_text())
    if not meta.get("chat_format"):
        raise RuntimeError("SFT did not improve validation; base retained, chat release withheld")
    for label, model in [("general-v2", base), ("sft-v2", selected)]:
        run(
            label + "_core",
            [
                "experiments/core_eval.py",
                "--model",
                str(model),
                "--output",
                f"reports/{label}-core",
            ],
        )
    run(
        "chat_diagnostics",
        [
            "experiments/evaluate_sft.py",
            "--model",
            str(selected),
            "--output",
            "reports/sft-v2-chat.json",
        ],
    )
    run(
        "chat_benchmarks",
        [
            "experiments/chat_eval.py",
            "--model",
            str(selected),
            "--output",
            "reports/sft-v2-nanochat",
        ],
    )
    cli = ["-c", "from macoder.cli import main; main()"]
    if not Path("runs/sft-v2-q4").exists():
        run("q4_export", cli + ["quantize", "--model", str(selected), "--output", "runs/sft-v2-q4"])
    run(
        "q4_speed",
        cli
        + [
            "bench",
            "--model",
            "runs/sft-v2-q4",
            "--prompt-tokens",
            "128",
            "--tokens",
            "256",
            "--repeats",
            "7",
            "--output",
            "reports/sft-v2-q4-speed.json",
        ],
    )
    run(
        "q4_quality",
        [
            "experiments/evaluate_sft.py",
            "--model",
            "runs/sft-v2-q4",
            "--output",
            "reports/sft-v2-q4-chat.json",
        ],
    )
    status(
        "complete",
        base_checkpoint=str(base),
        chat_checkpoint=str(selected),
        quantized_checkpoint="runs/sft-v2-q4",
    )
except BaseException as error:
    status("failed", error=str(error))
    traceback.print_exc()
    raise SystemExit(1)
