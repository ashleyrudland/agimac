import json
from .training_common import learning_rate, dataset_identity
import math
from pathlib import Path
import time
import shutil
import signal
import threading

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .checkpoint import save_model, load_model
from .data import TokenStream, file_hash
from .model import Config, Model, loss_fn


class StableAdamW(optim.AdamW):
    def apply_single(self, gradient, parameter, state):
        # FP32 moments and arithmetic; store parameters in their selected dtype.
        return (
            super()
            .apply_single(gradient.astype(mx.float32), parameter, state)
            .astype(parameter.dtype)
        )


class MasterAdamW(StableAdamW):
    """Keep precise weights as well as moments; cast only the forward/backward copy."""

    def init_single(self, parameter, state):
        master = parameter.astype(mx.float32)
        super().init_single(master, state)
        state["master"] = master

    def apply_single(self, gradient, parameter, state):
        updated = super().apply_single(gradient, state["master"], state)
        state["master"] = updated
        return updated.astype(parameter.dtype)


def train(args):
    """Learn next-token prediction from packed document windows.

    Accumulated gradients simulate a larger batch; clipping precedes AdamW.
    Validation is read-only. Atomic checkpoint directories include optimizer
    and sampler state so resume continues the original schedule exactly.
    """
    if (
        min(
            args.steps,
            args.batch_size,
            args.sequence,
            args.accumulate,
            args.eval_every,
            args.eval_batches,
        )
        < 1
    ):
        raise ValueError("Training counts and intervals must be positive")
    if args.lr <= 0 or args.warmup < 0:
        raise ValueError("Need positive lr and nonnegative warmup")
    if getattr(args, "keep_last", 0) < 0:
        raise ValueError("--keep-last cannot be negative")
    full_step = getattr(args, "compile_step", False)
    if full_step and args.accumulate != 1:
        raise ValueError("--compile-step requires --accumulate 1")
    data, out = Path(args.data), Path(args.output)
    # Never overwrite a previous experiment or partially completed checkpoint.
    if out.exists():
        raise ValueError("Training output exists; choose a new directory (use --resume for state)")
    corpus_identity = dataset_identity(data)
    manifest = json.loads((data / "manifest.json").read_text())
    # Token IDs only make sense with the exact tokenizer that prepared the corpus.
    if file_hash(data / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from prepared corpus")
    # Fix model initialization so repeated runs start from the same random seed.
    mx.random.seed(args.seed)
    config = Config.read(args.config)
    config.vocab_size = manifest["vocab_size"]
    if args.sequence > config.context:
        raise ValueError("Training sequence exceeds model context")
    initial_tokens_seen = 0
    parent_checkpoint = None
    from_scratch = True
    init_from = getattr(args, "init_from", None)
    if init_from and args.resume:
        raise ValueError("Choose either --resume or --init-from")
    # A new phase reuses learned weights but starts a fresh optimizer and schedule.
    if init_from:
        model, _, meta = load_model(init_from)
        if "quantization" in meta:
            raise ValueError("Cannot train from quantized weights")
        if model.config != config or meta.get("tokenizer_sha256") != manifest["tokenizer_sha256"]:
            raise ValueError("Initialization model/tokenizer config mismatch")
        initial_tokens_seen = int(meta.get("tokens_seen", 0))
        parent_checkpoint = str(Path(init_from).resolve())
        from_scratch = bool(meta.get("trained_from_scratch", False))
    else:
        model = Model(config)
    model.set_dtype(getattr(mx, args.dtype))
    optimizer_class = MasterAdamW if getattr(args, "master_weights", False) else StableAdamW
    optimizer = optimizer_class(learning_rate=args.lr, weight_decay=0.1)
    # Read random windows from the packed token file without loading the whole corpus.
    stream = TokenStream(data / "train.bin", args.sequence, args.batch_size, args.seed)
    start_step = 0
    # Resume restores the old optimizer and sampling position, not just model weights.
    if args.resume:
        model, _, meta = load_model(args.resume)
        if "quantization" in meta:
            raise ValueError("Cannot resume training a quantized checkpoint")
        if model.config != config or meta["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
            raise ValueError("Resume model/tokenizer config mismatch")
        initial_tokens_seen = int(meta.get("initial_tokens_seen", 0))
        parent_checkpoint = meta.get("parent_checkpoint")
        from_scratch = bool(meta.get("trained_from_scratch", False))
        state = json.loads((Path(args.resume) / "trainer.json").read_text())
        # Reject changed settings that would silently alter the resumed training trajectory.
        for key in (
            "steps",
            "warmup",
            "lr",
            "sequence",
            "batch_size",
            "accumulate",
            "dtype",
            "seed",
        ):
            if state["args"][key] != getattr(args, key):
                raise ValueError(f"Resume requires the original {key} for an identical schedule")
        if state["args"].get("master_weights", False) != getattr(args, "master_weights", False):
            raise ValueError("Resume requires original master_weights setting")
        if state["args"].get("compile_step", False) != full_step:
            raise ValueError("Resume requires the original compile_step setting")
        if state["source_sha256"] != manifest["source_sha256"] or state[
            "manifest_sha256"
        ] != file_hash(data / "manifest.json"):
            raise ValueError("Resume requires the original prepared dataset")
        if state.get("dataset_identity", corpus_identity) != corpus_identity:
            raise ValueError("Resume requires identical packed token bytes")
        # Restore Adam momentum/variance and the data generator so the next update matches.
        optimizer.state = tree_unflatten(
            list(mx.load(str(Path(args.resume) / "optimizer.npz")).items())
        )
        stream.rng.bit_generator.state = state["rng"]
        start_step = state["step"]
    if start_step >= args.steps:
        raise ValueError("Checkpoint has already completed the requested schedule")
    out.mkdir(parents=True)
    (out / "dataset-identity.json").write_text(json.dumps(corpus_identity, indent=2))
    # Start a separate read-only log viewer; it does not participate in GPU updates.
    from .dashboard import launch

    launch(
        out,
        parameters=model.parameter_count(),
        total_steps=args.steps,
        tokens_per_step=args.batch_size * args.sequence * args.accumulate,
    )
    model.train()
    # MLX is lazy: force initialization to finish before starting measured training.
    mx.eval(model.parameters(), optimizer.state)
    # Ask MLX for both the scalar loss and its derivative for every trainable weight.
    grad_fn = nn.value_and_grad(model, loss_fn)

    def value_grad(x, y):
        return grad_fn(model, x, y)

    if args.compile:
        # Compile the loss/gradient graph with model state explicitly tracked between calls.
        value_grad = mx.compile(value_grad, inputs=model.state, outputs=model.state)
    compiled_step = make_compiled_step(model, optimizer) if full_step else None
    log = (out / "metrics.jsonl").open("w")
    start = time.perf_counter()
    train_seconds = 0.0

    def checkpoint(step):
        final = out / f"step-{step:06d}"
        # Write into a hidden directory first; chat must never load a half-written checkpoint.
        dest = out / f".pending-step-{step:06d}"
        save_model(
            model,
            dest,
            data / "tokenizer.json",
            {
                "dtype": args.dtype,
                "step": step,
                "tokens_seen": initial_tokens_seen
                + step * args.batch_size * args.sequence * args.accumulate,
                "initial_tokens_seen": initial_tokens_seen,
                "parent_checkpoint": parent_checkpoint,
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "source_sha256": manifest["source_sha256"],
                "parameters": model.parameter_count(),
                "trained_from_scratch": from_scratch,
                "dataset_identity": corpus_identity,
            },
        )
        # Inference needs only weights; exact training resume also needs optimizer and RNG state.
        mx.savez(str(dest / "optimizer.npz"), **dict(tree_flatten(optimizer.state)))
        (dest / "trainer.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "rng": stream.rng.bit_generator.state,
                    "args": vars(args),
                    "dataset_identity": corpus_identity,
                    "source_sha256": manifest["source_sha256"],
                    "manifest_sha256": file_hash(data / "manifest.json"),
                },
                indent=2,
            )
            + "\n"
        )
        # Publish the complete directory, then atomically update the latest-checkpoint pointer.
        dest.rename(final)
        pointer = out / ".latest.tmp"
        pointer.write_text(str(final.resolve()) + "\n")
        pointer.replace(out / "latest.txt")
        keep = getattr(args, "keep_last", 0)
        if keep:
            checkpoints = sorted(out.glob("step-*"))
            for old in checkpoints[:-keep]:
                shutil.rmtree(old)

    # Finish the current update before saving on Ctrl-C/SIGTERM. Never serialize a half-update.
    stop_requested = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
    try:
        for step in range(start_step, args.steps):
            tick = time.perf_counter()
            lr = learning_rate(step, args.steps, args.warmup, args.lr)
            if compiled_step is not None:
                # One compiled graph covers gradients, clipping and the AdamW update.
                loss, norm = compiled_step(*stream.batch(), mx.array(lr))
                losses = [loss]
            else:
                grad_sum, losses = None, []
                for _ in range(args.accumulate):
                    x, y = stream.batch()
                    loss, grads = value_grad(x, y)
                    # Average gradients across small batches to approximate one larger effective batch.
                    flat = [
                        (k, g.astype(mx.float32) / args.accumulate) for k, g in tree_flatten(grads)
                    ]
                    grad_sum = (
                        flat
                        if grad_sum is None
                        else [(k, a + b) for (k, a), (_, b) in zip(grad_sum, flat)]
                    )
                    losses.append(loss)
                    # Evaluate each accumulated batch now rather than retain an ever-growing lazy graph.
                    mx.eval(loss, [g for _, g in grad_sum])
                # Limit the total gradient size to reduce damage from an unusually large update.
                grads, norm = optim.clip_grad_norm(tree_unflatten(grad_sum), max_norm=1.0)
                lr = learning_rate(step, args.steps, args.warmup, args.lr)
                optimizer.learning_rate = lr
                # Keep model weights in selected precision, optimizer accumulation in float32.
                optimizer.update(model, grads)
            # Wait for the GPU work before calculating throughput; queued work is not completed work.
            mx.eval(model.parameters(), optimizer.state, norm)
            elapsed = time.perf_counter() - tick
            train_seconds += elapsed
            loss_value = sum(float(v.item()) for v in losses) / len(losses)
            if not math.isfinite(loss_value) or not math.isfinite(float(norm.item())):
                raise RuntimeError("Nonfinite training loss/gradient; checkpoint not saved")
            row = {
                "step": step + 1,
                "loss": loss_value,
                "lr": lr,
                "grad_norm": float(norm.item()),
                "train_tokens_per_second": args.batch_size
                * args.sequence
                * args.accumulate
                / elapsed,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                model.eval()
                # Reuse the same validation seed each time so changes reflect weights, not sampled data.
                valid = TokenStream(
                    data / "valid.bin", args.sequence, args.batch_size, args.seed + 1
                )
                vals = []
                for _ in range(args.eval_batches):
                    v = loss_fn(model, *valid.batch())
                    vals.append(float(v.item()))
                row["valid_loss"] = sum(vals) / len(vals)
                # Perplexity is exp(loss); clamp only this display calculation to avoid overflow.
                row["valid_perplexity"] = math.exp(min(50, row["valid_loss"]))
                model.train()
                checkpoint(step + 1)
            if stop_requested and "valid_loss" not in row:
                checkpoint(step + 1)
            log.write(json.dumps(row) + "\n")
            # Make each completed metric immediately visible to the offline status reader.
            log.flush()
            if step == start_step or (step + 1) % args.log_every == 0 or "valid_loss" in row:
                print(json.dumps(row), flush=True)
            if stop_requested:
                print("Stopped after saving a resumable checkpoint", flush=True)
                break
        summary = {
            "parameters": model.parameter_count(),
            "steps_completed": step + 1 - start_step,
            "status": "stopped" if stop_requested else "complete",
            "train_tokens_per_second": (step + 1 - start_step)
            * args.batch_size
            * args.sequence
            * args.accumulate
            / train_seconds,
            "wall_seconds": time.perf_counter() - start,
            "last": row,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        log.close()


def make_compiled_step(model, optimizer):
    """Fuse one non-accumulated update into a single MLX graph.

    The caller still evaluates returned arrays before timing a completed update.
    FP32 Adam moments are retained even when parameters use BF16.
    """
    vg = nn.value_and_grad(model, loss_fn)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state]

    def step(x, y, lr):
        optimizer.learning_rate = lr
        loss, grads = vg(model, x, y)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss, norm

    return mx.compile(step, inputs=state, outputs=state)
