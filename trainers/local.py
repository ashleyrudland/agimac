"""Apple-silicon entry point: compiled native MLX, BF16 + FP32 master weights."""

import argparse
import json
import sys
from pathlib import Path
from macoder.training_common import dataset_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="recipes/v2.json")
    parser.add_argument("--config", default="configs/nano-core-145m.json")
    parser.add_argument("--data", default="data/general-v2/prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--steps", type=int, help="Override schedule for an explicitly short experiment"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Tune for your Mac; changes optimizer batch size"
    )
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    identity = dataset_identity(args.data)
    batch = args.batch_size or recipe["batch_size"]
    steps = args.steps or (recipe["training_tokens"] + batch * recipe["sequence"] - 1) // (
        batch * recipe["sequence"]
    )
    # MLX uses the native GPU. Memory usage follows the working set, not an artificial RAM quota.
    sys.argv = [
        "agimac",
        "train",
        "--config",
        args.config,
        "--data",
        args.data,
        "--output",
        args.output,
        "--steps",
        str(steps),
        "--sequence",
        str(recipe["sequence"]),
        "--batch-size",
        str(batch),
        "--accumulate",
        "1",
        "--dtype",
        "bfloat16",
        "--master-weights",
        "--compile-step",
        "--lr",
        str(recipe["learning_rate"]),
        "--warmup",
        str(recipe["warmup_steps"]),
        "--eval-every",
        str(recipe["checkpoint_every_steps"]),
        "--eval-batches",
        "32",
        "--keep-last",
        str(recipe["keep_last"]),
    ]
    if args.resume:
        sys.argv += ["--resume", args.resume]
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from macoder.cli import main as run

    print("Verified corpus:", json.dumps(identity), flush=True)
    run()


if __name__ == "__main__":
    main()
