"""Download nanochat's bundle, wait for chat evaluation, then score both checkpoints."""

import argparse
import os
import time
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--download-pid", type=int)
args = parser.parse_args()
if args.download_pid:
    print("Waiting for existing bundle download", args.download_pid, flush=True)
    while True:
        try:
            os.kill(args.download_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(10)
root = Path(__file__).resolve().parents[1]
p = root / "data/core-eval"
p.mkdir(parents=True, exist_ok=True)
z = p / "eval_bundle.zip"
# The download can be restarted safely; only a verified complete ZIP is extracted.
if not z.exists() or not zipfile.is_zipfile(z):
    temporary = p / "eval_bundle.download"
    urllib.request.urlretrieve(
        "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip", temporary
    )
    temporary.replace(z)
with zipfile.ZipFile(z) as archive:
    for name in archive.namelist():
        if not (p / name).resolve().is_relative_to(p.resolve()):
            raise ValueError(name)
    archive.extractall(p)
(p / "provenance.json").write_text(
    json.dumps(
        {
            "url": "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
            "sha256": hashlib.sha256(z.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
for checkpoint, output in [
    ("runs/sft-v1/step-041364", "reports/sft-v1-core"),
    ("runs/general-v1/step-048829", "reports/general-v1-core"),
]:
    command = [
        sys.executable,
        "experiments/core_eval.py",
        "--model",
        checkpoint,
        "--output",
        output,
    ]
    # Wait outside the GPU process so the existing chat evaluation keeps all GPU time.
    if (root / "reports/chat-eval-process.json").exists():
        command += [
            "--wait-pid",
            str(json.loads((root / "reports/chat-eval-process.json").read_text())["pid"]),
        ]
    subprocess.run(command, cwd=root, check=True)
