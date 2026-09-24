"""Bounded probes; temporarily suspend the known CORE process group and always resume."""

import os
import signal
import subprocess
import sys
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/small.json").read_text())
config["vocab_size"] = 16392
(root / "reports/probe-legacy-config.json").write_text(json.dumps(config))
probes = [
    ("legacy-fp32-gradient", "reports/probe-legacy-config.json", "float32", "gradient", 4, 512),
    ("legacy-fp32-full", "reports/probe-legacy-config.json", "float32", "full", 4, 512),
    ("legacy-bf16-full", "reports/probe-legacy-config.json", "bfloat16", "full", 4, 512),
    ("nano-small-fp32", "configs/nano-core-72m.json", "float32", "full", 4, 512),
    ("nano-small-bf16", "configs/nano-core-72m.json", "bfloat16", "full", 4, 512),
    ("nano-large-bf16", "configs/nano-core-145m.json", "bfloat16", "full", 4, 512),
    ("nano-small-bf16-b8", "configs/nano-core-72m.json", "bfloat16", "full", 8, 512),
    ("nano-small-bf16-1024", "configs/nano-core-72m.json", "bfloat16", "full", 4, 1024),
]
group = json.loads((root / "reports/core-eval-process.json").read_text())["pid"]
# Verify the process identity before touching a PID which could have been reused.
command = subprocess.check_output(["ps", "-p", str(group), "-o", "command="], text=True)
if "experiments/run_core_eval.py" not in command:
    raise RuntimeError("Unexpected CORE runner")
os.killpg(group, signal.SIGSTOP)
try:
    for name, config, dtype, mode, batch, seq in probes:
        print("START", name, flush=True)
        cmd = [
            sys.executable,
            "experiments/pretrain_speed.py",
            "--config",
            config,
            "--dtype",
            dtype,
            "--mode",
            mode,
            "--batch",
            str(batch),
            "--sequence",
            str(seq),
            "--output",
            f"reports/probe-{name}.json",
        ]
        try:
            subprocess.run(cmd, cwd=root, check=True, timeout=150)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            print("FAILED", name, str(error), flush=True)
finally:
    os.killpg(group, signal.SIGCONT)
    print("CORE evaluation resumed", flush=True)
