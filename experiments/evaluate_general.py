"""Fixed per-domain development likelihood and unedited completion probes."""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from macoder.checkpoint import load_model
from macoder.data import file_hash
from macoder.inference import completion
from macoder.model import Config, Model, loss_fn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--data", default="data/general-v1/prepared")
    p.add_argument("--config", default="configs/small.json")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.data)
    manifest = json.loads((root / "manifest.json").read_text())
    mx.random.seed(42)
    if a.model:
        model, tok, meta = load_model(a.model)
        if meta["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer mismatch")
    else:
        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        c = Config.read(a.config)
        c.vocab_size = tok.get_vocab_size()
        model = Model(c)
        meta = {"random_initialization": True}
        mx.eval(model.parameters())
    model.eval()
    stream = np.memmap(root / "valid.bin", dtype="<u4", mode="r")
    offset = 0
    results = {}
    for name, counts in manifest["tokens_by_source"].items():
        end = offset + counts["valid"]
        total = 0.0
        n = 0
        for start in range(offset, min(end - 512, offset + 32 * 512), 512):
            x = mx.array(np.array(stream[start : start + 512])[None, :])
            y = mx.array(np.array(stream[start + 1 : start + 513])[None, :])
            total += float(loss_fn(model, x, y).item()) * 512
            n += 512
        if n == 0:
            raise ValueError("No evaluation tokens: " + name)
        results[name] = {
            "tokens": n,
            "nats_per_token": total / n,
            "perplexity": math.exp(min(50, total / n)),
        }
        offset = end
    prompts = [
        "Question: Why does ice melt when heated?\nAnswer:",
        "Question: What is 17 plus 25?\nAnswer:",
        "def add(a, b):\n    ",
        "def reverse_string(text):\n    ",
        "Once upon a time, a little robot found a key",
    ]
    samples = []
    if a.model:
        for prompt in prompts:
            samples.append(
                {"prompt": prompt, "completion": completion(model, tok, prompt, 128, 0.0)}
            )
    result = {
        "model": a.model or "random initialization",
        "metadata": meta,
        "domains": results,
        "samples": samples,
        "manifest_sha256": file_hash(root / "manifest.json"),
        "protocol": "Development diagnostic: first 32 disjoint 512-target windows per source; not independent task accuracy",
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
