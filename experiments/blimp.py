"""Full BLiMP grammatical minimal-pair evaluation with local MLX scoring.

Uses summed sentence log probability, EOS as unscored BOS, no scored EOS,
no length normalization. Custom implementation, not lm-evaluation-harness.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer
from macoder.checkpoint import load_model
from macoder.data import file_hash
from macoder.model import Config, Model

REV = "3e56b06fcabca9b30822fc66435fca6b1aa40bb1"


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    api = f"https://api.github.com/repos/alexwarstadt/blimp/contents/data?ref={REV}"
    entries = json.load(urllib.request.urlopen(api, timeout=60))
    names = sorted(e["name"] for e in entries if e["name"].endswith(".jsonl"))

    def one(name):
        path = root / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/alexwarstadt/blimp/{REV}/data/{name}"
            data = urllib.request.urlopen(url, timeout=60).read()
            temp = path.with_suffix(".tmp")
            temp.write_bytes(data)
            temp.replace(path)
        return name, file_hash(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(one, names))
    meta = {
        "repository": "alexwarstadt/blimp",
        "revision": REV,
        "license": "CC-BY",
        "credit": "Warstadt et al. (2020), BLiMP: The Benchmark of Linguistic Minimal Pairs",
        "files": hashes,
    }
    (root / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return names


def sentence_scores(model, tokenizer, sentences, batch_size=32):
    eos = tokenizer.token_to_id("<|endoftext|>")
    encoded = [tokenizer.encode(text).ids for text in sentences]
    scores = []
    for start in range(0, len(encoded), batch_size):
        group = encoded[start : start + batch_size]
        length = max(map(len, group))
        if length > model.config.context or length < 1:
            raise ValueError("Sentence outside supported context; no silent truncation")
        x = np.full((len(group), length), eos, dtype=np.uint32)
        y = np.zeros((len(group), length), dtype=np.uint32)
        mask = np.zeros((len(group), length), dtype=np.float32)
        for i, ids in enumerate(group):
            x[i, : len(ids)] = [eos] + ids[:-1]
            y[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1
        logits = model(mx.array(x)).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, mx.array(y), reduction="none")
        scores.extend((-mx.sum(ce * mx.array(mask), axis=-1)).tolist())
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch-only", action="store_true")
    p.add_argument("--data", default="data/blimp")
    p.add_argument("--model")
    p.add_argument("--random", action="store_true")
    p.add_argument("--output")
    a = p.parse_args()
    root = Path(a.data)
    if a.fetch_only:
        print("downloaded", len(fetch(root)), "BLiMP subsets")
        return
    if not a.output or bool(a.model) == a.random:
        p.error("Choose --model or --random and provide --output")
    mx.random.seed(42)
    if a.model:
        model, tok, meta = load_model(a.model)
    else:
        tok = Tokenizer.from_file("data/stories/tokenizer.json")
        cfg = Config.read("configs/stories.json")
        cfg.vocab_size = tok.get_vocab_size()
        model = Model(cfg)
        meta = {"initialization_seed": 42}
        mx.eval(model.parameters())
    model.eval()
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for name, h in manifest["files"].items():
        path = root / name
        if file_hash(path) != h:
            raise ValueError("Dataset hash mismatch")
        pairs = [json.loads(line) for line in path.read_text().splitlines() if line]
        scores = sentence_scores(
            model,
            tok,
            [text for pair in pairs for text in (pair["sentence_good"], pair["sentence_bad"])],
        )
        margin = np.array(scores[::2]) - np.array(scores[1::2])
        row = {
            "subset": path.stem,
            "pairs": len(pairs),
            "correct": int((margin > 0).sum()),
            "ties": int((margin == 0).sum()),
            "accuracy": float((margin > 0).mean()),
            "margins": margin.tolist(),
        }
        rows.append(row)
    total = sum(r["pairs"] for r in rows)
    result = {
        "model": a.model or "random initialization",
        "metadata": meta,
        "dataset": manifest,
        "protocol": __doc__,
        "subsets": len(rows),
        "pairs": total,
        "accuracy": sum(r["correct"] for r in rows) / total,
        "results": rows,
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["results", "dataset"]}, indent=2)
    )


if __name__ == "__main__":
    main()
