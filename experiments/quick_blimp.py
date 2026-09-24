"""Small reproducible BLiMP diagnostic; never a full benchmark score."""

import argparse
import json
import random
from pathlib import Path
from blimp import sentence_scores
from macoder.checkpoint import load_model
from macoder.data import file_hash


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-subset", type=int, default=10)
    args = parser.parse_args()
    model, tokenizer, metadata = load_model(args.model)
    root = Path("data/blimp")
    manifest = json.loads((root / "manifest.json").read_text())
    records = []
    for name, expected in sorted(manifest["files"].items()):
        path = root / name
        if file_hash(path) != expected:
            raise ValueError("BLiMP data hash mismatch: " + name)
        pairs = [json.loads(line) for line in path.read_text().splitlines() if line]
        indices = random.Random(42).sample(range(len(pairs)), args.per_subset)
        selected = [pairs[i] for i in indices]
        scores = sentence_scores(
            model,
            tokenizer,
            [text for pair in selected for text in (pair["sentence_good"], pair["sentence_bad"])],
        )
        for i, pair, good, bad in zip(indices, selected, scores[::2], scores[1::2]):
            records.append(
                dict(
                    subset=name,
                    index=i,
                    good=pair["sentence_good"],
                    bad=pair["sentence_bad"],
                    good_logprob=good,
                    bad_logprob=bad,
                    correct=good > bad,
                )
            )
    result = dict(
        model=args.model,
        metadata=metadata,
        provenance=manifest,
        protocol="Diagnostic subset: seed 42, 10 pairs/subset by default; sum sentence log probability; ties wrong; no scored EOS.",
        full_benchmark=False,
        correct=sum(r["correct"] for r in records),
        total=len(records),
        records=records,
    )
    result["accuracy"] = result["correct"] / result["total"]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k in ("correct", "total", "accuracy")})


if __name__ == "__main__":
    main()
