"""Fail before training if prepared arrays/tokenizer differ from reference data."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


for stage, directory in [("general", "data/general-v1/prepared"), ("sft", "data/sft-v1")]:
    folder = root / directory
    if not (folder / "manifest.json").exists():
        continue
    expected = json.loads((root / f"recipes/reference/{stage}-manifest.json").read_text())
    if sha(folder / "tokenizer.json") != expected["tokenizer_sha256"]:
        raise SystemExit(stage + ": tokenizer differs")
    hashes = dict(expected.get("array_hashes", {}))
    if stage == "general":
        hashes.update({split + ".bin": expected[split + "_sha256"] for split in ("train", "valid")})
    for name, digest in hashes.items():
        if sha(folder / name) != digest:
            raise SystemExit(stage + ": changed array " + name)
    print(stage + ": reference tokenizer and recorded arrays verified")
