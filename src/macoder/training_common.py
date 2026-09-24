"""Shared schedule and immutable corpus identity for both training backends."""

import math
import json
from pathlib import Path
from .data import file_hash


def learning_rate(step, total, warmup, peak):
    # Start with small updates, then gradually reach the requested learning rate.
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    # After warmup, use a smooth cosine decline ending at 10% of the peak rate.
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))


def dataset_identity(data):
    """Hash the actual packed bytes, not just a source description."""
    data = Path(data)
    identity = {
        name: file_hash(data / name)
        for name in ("train.bin", "valid.bin", "tokenizer.json", "manifest.json")
    }
    manifest = json.loads((data / "manifest.json").read_text())
    if identity["tokenizer.json"] != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from corpus")
    return identity


def identity_key(identity):
    """Address each distinct corpus separately so speed samples cannot replace real data."""
    import hashlib

    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
