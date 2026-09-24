"""Package inference-only checkpoint files without loading the GPU or uploading."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
names = ["config.json", "metadata.json", "tokenizer.json", "model.safetensors"]
for name in names:
    if not (a.model / name).is_file():
        raise SystemExit("Missing checkpoint file: " + name)
a.output.mkdir(parents=True, exist_ok=False)
lines = []
for name in names:
    target = a.output / name
    shutil.copyfile(a.model / name, target)
    lines.append(hashlib.sha256(target.read_bytes()).hexdigest() + "  " + name)
(a.output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
meta = json.loads((a.output / "metadata.json").read_text())
(a.output / "README.md").write_text(
    "# agimac checkpoint\n\nExperimental Apple-silicon language model. Run with `agimac chat --model PATH`.\n\nSource checkpoint: "
    + a.model.name
    + "\n\nThis export is not a quality certification. Attach evaluated results and dataset/license provenance before publication.\n"
)
print(a.output)
