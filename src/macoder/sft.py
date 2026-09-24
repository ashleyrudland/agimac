"""MLX assistant-only SFT with deterministic bucketed epochs and resumable saves."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from .checkpoint import load_model, save_model
from .data import file_hash
from .train import StableAdamW, learning_rate
from .conversation import FORMAT


def masked_loss(model, x, y, mask):
    """Predict shifted targets; normalize only by assistant tokens, not padding.

    Prompt/tool-result tokens influence attention but receive no direct loss.
    This teaches the model to produce calls and answers rather than user turns.
    """
    # Compute a separate prediction error at each position before applying the assistant mask.
    loss = nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="none")
    # Divide by actual supervised tokens, not padded sequence length.
    return (loss * mask).sum() / mx.maximum(mask.sum(), 1)


class Batches:
    def __init__(self, root, split, batch_size, epochs=1, seed=42):
        self.batch_size = batch_size
        self.arrays = {}
        self.plans = []
        root = Path(root)
        for length in (256, 512, 1024):
            # Memory-map each length bucket so the dataset does not need to fit in RAM.
            ids = np.load(root / f"{split}-{length}-ids.npy", mmap_mode="r")
            masks = np.load(root / f"{split}-{length}-mask.npy", mmap_mode="r")
            if len(ids):
                self.arrays[length] = (ids, masks)
        for epoch in range(epochs):
            # A fixed seed per epoch gives shuffled data whose order can be reconstructed on resume.
            rng = np.random.default_rng(seed + epoch)
            plan = []
            for length, (ids, _) in self.arrays.items():
                order = rng.permutation(len(ids)) if split == "train" else np.arange(len(ids))
                plan.extend(
                    (length, order[i : i + batch_size]) for i in range(0, len(ids), batch_size)
                )
            if split == "train":
                rng.shuffle(plan)
            self.plans.extend(plan)

    def __len__(self):
        return len(self.plans)

    def batch(self, index):
        length, indices = self.plans[index]
        source, masks = self.arrays[length]
        # Fill a short last batch with zero-mask padding; padded rows contribute no gradient.
        ids = np.zeros((self.batch_size, length + 1), dtype=np.uint32)
        mask = np.zeros((self.batch_size, length + 1), dtype=np.float32)
        ids[: len(indices)] = source[indices]
        mask[: len(indices)] = masks[indices]
        actual = int(np.count_nonzero(ids))
        targets = int(mask[:, 1:].sum())
        if targets < 1:
            raise ValueError("SFT batch has no assistant targets")
        # Shift inputs and targets by one token, and shift the target mask with the targets.
        return mx.array(ids[:, :-1]), mx.array(ids[:, 1:]), mx.array(mask[:, 1:]), actual, targets


def evaluate(model, data, max_batches=None):
    total = count = 0
    # Spread a bounded selection over all length buckets, consistently across runs.
    indices = (
        np.arange(len(data))
        if not max_batches
        else np.unique(np.linspace(0, len(data) - 1, min(max_batches, len(data)), dtype=int))
    )
    # Weight each validation batch by its number of assistant targets, not by batch count.
    for i in indices:
        x, y, m, _, n = data.batch(int(i))
        total += float(masked_loss(model, x, y, m).item()) * n
        count += n
    return total / count


def train(args):
    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root / "manifest.json").read_text())
    if file_hash(root / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash mismatch")
    # Verify prepared arrays before training so changed data cannot silently reuse this recipe.
    for name, digest in manifest["array_hashes"].items():
        if file_hash(root / name) != digest:
            raise ValueError("Array hash mismatch: " + name)
    mx.random.seed(args.seed)
    model, tok, base = load_model(args.resume or args.model)
    if "quantization" in base:
        raise ValueError("Use unquantized weights for SFT")
    if base["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
        raise ValueError("Base tokenizer mismatch")
    # This reference run keeps training weights in FP32 for numerical stability.
    model.set_dtype(mx.float32)
    model.train()
    train_data = Batches(root, "train", args.batch_size, args.epochs, args.seed)
    valid = Batches(root, "valid", args.batch_size)
    steps = len(train_data)
    optimizer = StableAdamW(learning_rate=args.lr, weight_decay=0.01)
    start_step = 0
    seen = 0
    supervised = 0
    best = float("inf")
    stale = 0
    parent = str(Path(args.model).resolve())
    initial = base["tokens_seen"]
    if args.resume:
        state = json.loads((Path(args.resume) / "trainer.json").read_text())
        for k in ("epochs", "batch_size", "seed", "lr", "warmup", "eval_every", "patience"):
            if state["args"][k] != getattr(args, k):
                raise ValueError("Resume setting differs: " + k)
        if state["manifest_sha256"] != file_hash(root / "manifest.json"):
            raise ValueError("Resume dataset differs")
        optimizer.state = tree_unflatten(
            list(mx.load(str(Path(args.resume) / "optimizer.npz")).items())
        )
        start_step = state["step"]
        seen = state["seen"]
        supervised = state["supervised"]
        best = state["best"]
        stale = state["stale"]
        parent = state["parent"]
        initial = state["initial"]
    if start_step >= steps:
        raise ValueError("Schedule already completed")
    out.mkdir(parents=True)
    # The dashboard polls files in its own process, keeping UI work out of training.
    from .dashboard import launch

    launch(out, parameters=model.parameter_count(), total_steps=steps)
    mx.eval(model.parameters(), optimizer.state)
    fn = nn.value_and_grad(model, masked_loss)

    def vg(x, y, m):
        return fn(model, x, y, m)

    if not args.no_compile:
        vg = mx.compile(vg, inputs=model.state, outputs=model.state)
    start = time.perf_counter()
    # The unchanged base checkpoint remains a candidate if no fine-tuned checkpoint improves loss.
    best_path = Path(state.get("best_checkpoint", args.resume)) if args.resume else Path(args.model)
    # Evaluate without updates before the first training step.
    model.eval()
    baseline = evaluate(model, valid, 48)
    model.train()
    if not args.resume:
        best = baseline
    (out / "best.txt").write_text(str(best_path.resolve()) + "\n")
    (out / "initial-evaluation.json").write_text(json.dumps({"valid_loss": baseline}) + "\n")
    print(
        json.dumps(
            {
                "stage": "sft_start",
                "steps": steps,
                "baseline_valid_loss": baseline,
                "batch_size": args.batch_size,
            }
        ),
        flush=True,
    )

    def save(step, is_best):
        nonlocal best_path
        # Stage a complete checkpoint before publishing it; include optimizer state for resume.
        staging = out / f".pending-step-{step:06d}"
        dest = out / f"step-{step:06d}"
        metadata = {
            **base,
            "dtype": "float32",
            "step": step,
            "tokens_seen": initial + seen,
            "initial_tokens_seen": initial,
            "sft_input_tokens": seen,
            "sft_assistant_targets": supervised,
            "parent_checkpoint": parent,
            "source_sha256": file_hash(root / "manifest.json"),
            "chat_format": FORMAT,
            "tools": ["add", "subtract", "multiply", "divide"],
            "trained_context": 1024,
        }
        save_model(model, staging, root / "tokenizer.json", metadata)
        mx.savez(str(staging / "optimizer.npz"), **dict(tree_flatten(optimizer.state)))
        (staging / "trainer.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "step": step,
                    "seen": seen,
                    "supervised": supervised,
                    "best": best,
                    "best_checkpoint": str((dest if is_best else best_path).resolve()),
                    "stale": stale,
                    "initial": initial,
                    "parent": parent,
                    "manifest_sha256": file_hash(root / "manifest.json"),
                }
            )
        )
        staging.rename(dest)
        for label in ["latest", "best"] if is_best else ["latest"]:
            temp = out / f".{label}.tmp"
            temp.write_text(str(dest.resolve()) + "\n")
            temp.replace(out / f"{label}.txt")
        if is_best:
            best_path = dest
        # Limit disk growth while always preserving the best checkpoint.
        for old in sorted(out.glob("step-*"))[:-2]:
            if old != best_path:
                shutil.rmtree(old)

    with (out / "metrics.jsonl").open("w") as log:
        for step in range(start_step, steps):
            x, y, m, actual, targets = train_data.batch(step)
            # Materialize batch inputs before timing the GPU update.
            mx.eval(x, y, m)
            tick = time.perf_counter()
            loss, grads = vg(x, y, m)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            optimizer.learning_rate = learning_rate(step, steps, args.warmup, args.lr)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, norm)
            elapsed = time.perf_counter() - tick
            seen += actual
            supervised += targets
            value = float(loss.item())
            if not math.isfinite(value) or not math.isfinite(float(norm.item())):
                raise RuntimeError("Nonfinite loss/gradient")
            row = {
                "step": step + 1,
                "total_steps": steps,
                "loss": value,
                "grad_norm": float(norm.item()),
                "actual_input_tokens": seen,
                "assistant_targets": supervised,
                "input_tokens_per_second": actual / elapsed,
                "padded_tokens_per_second": x.size / elapsed,
                "supervised_tokens_per_second": targets / elapsed,
                "sequence": x.shape[1],
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
                "elapsed_seconds": time.perf_counter() - start,
            }
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                model.eval()
                row["valid_loss"] = evaluate(model, valid, 48)
                model.train()
                # Require a small loss improvement; tiny changes do not reset the early-stop counter.
                is_best = row["valid_loss"] < best - 0.001
                if is_best:
                    best = row["valid_loss"]
                    stale = 0
                else:
                    stale += 1
                save(step + 1, is_best)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == start_step or (step + 1) % 50 == 0 or "valid_loss" in row:
                print(json.dumps(row), flush=True)
            if args.patience and stale >= args.patience:
                print("Early stop: development loss stopped improving.", flush=True)
                break
    # A resumed run with no new best retains the parent checkpoint as best candidate.
    if not (out / "best.txt").exists():
        (out / "best.txt").write_text(str(Path(args.resume).resolve()) + "\n")
    summary = {
        "last": row,
        "best_valid_loss": best,
        "wall_seconds": time.perf_counter() - start,
        "planned_steps": steps,
        "best_checkpoint": (out / "best.txt").read_text().strip(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/general-v1/step-048829")
    p.add_argument("--data", default="data/sft-v1")
    p.add_argument("--output", default="runs/sft-v1")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args()
    if min(a.batch_size, a.epochs, a.eval_every) < 1 or a.lr <= 0 or min(a.warmup, a.patience) < 0:
        p.error("Invalid training settings")
    train(a)


if __name__ == "__main__":
    main()
