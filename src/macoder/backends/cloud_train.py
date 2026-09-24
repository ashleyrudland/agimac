"""Resumable CUDA pretraining, also runnable on CPU for tiny correctness tests."""

import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from .torch_model import Model, MasterAdamW, loss_fn
from ..config import Config
from ..data import TokenStream
from ..training_common import dataset_identity, learning_rate


def train(
    config,
    data,
    output,
    *,
    steps,
    batch_size=8,
    sequence=512,
    seed=42,
    lr=3e-4,
    warmup=2000,
    eval_every=1000,
    eval_batches=32,
    dtype="bfloat16",
    device="cuda",
    compile_model=True,
    resume=None,
    max_steps=None,
    max_seconds=20 * 3600,
    keep_last=0,
    commit=lambda: None,
):
    """`steps` is the full schedule; max_steps bounds THIS invocation only.

    A cloud job stops before Modal's timeout and publishes a resumable checkpoint.
    It never silently starts another paid invocation. Resume supplies optimizer,
    sampler and schedule state; inference weights are directly readable by MLX.
    """
    if (
        min(steps, batch_size, sequence, eval_every, eval_batches, max_seconds) <= 0
        or lr <= 0
        or warmup < 0
    ):
        raise ValueError("Invalid training counts or learning rate")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    data, out = Path(data), Path(output)
    if out.exists():
        raise ValueError("Choose a fresh output directory, including when resuming")
    identity = dataset_identity(data)
    c = Config.read(config)
    manifest = json.loads((data / "manifest.json").read_text())
    if c.vocab_size != manifest["vocab_size"] or sequence > c.context:
        raise ValueError("Config incompatible with prepared data or sequence")
    torch.manual_seed(seed)
    model = Model(c).to(device=device, dtype=getattr(torch, dtype))
    optimizer = MasterAdamW(model.parameters(), lr=lr)
    stream = TokenStream(data / "train.bin", sequence, batch_size, seed)
    settings = dict(
        steps=steps,
        batch_size=batch_size,
        sequence=sequence,
        seed=seed,
        lr=lr,
        warmup=warmup,
        dtype=dtype,
    )
    start_step = 0
    if resume:
        # Only resume our own trusted checkpoints, never arbitrary pickled downloads.
        state = torch.load(Path(resume) / "trainer.pt", map_location=device, weights_only=True)
        if (
            state["settings"] != settings
            or state["identity"] != identity
            or state["config"] != vars(c)
        ):
            raise ValueError("Resume requires identical config, data and training schedule")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.rng.bit_generator.state = state["rng"]
        start_step = state["step"]
    if start_step >= steps:
        raise ValueError("Schedule already finished")
    out.mkdir(parents=True)
    (out / "dataset-identity.json").write_text(json.dumps(identity, indent=2))
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    c.write(out / "config.json")
    forward = torch.compile(model) if compile_model else model
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(lr, device=device)
    update = torch.compile(optimizer.step) if compile_model else optimizer.step

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def batch(source):
        return tuple(torch.from_numpy(a.astype(np.int64)).to(device) for a in source.numpy_batch())

    def checkpoint(step):
        dest = out / f"step-{step:06d}"
        pending = out / f".pending-{step:06d}"
        pending.mkdir()
        # Names and tensor layouts deliberately match model.py; no lossy conversion.
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(pending / "model.safetensors"),
        )
        c.write(pending / "config.json")
        shutil.copyfile(data / "tokenizer.json", pending / "tokenizer.json")
        (pending / "metadata.json").write_text(
            json.dumps(
                dict(
                    dtype=dtype,
                    step=step,
                    tokens_seen=step * batch_size * sequence,
                    trained_from_scratch=True,
                    backend="torch-cuda" if device.startswith("cuda") else "torch-cpu",
                    benchmark_only=manifest.get("benchmark_only", False),
                    tokenizer_sha256=identity["tokenizer.json"],
                    dataset_identity=identity,
                    parameters=sum(p.numel() for p in model.parameters()),
                ),
                indent=2,
            )
        )
        torch.save(
            dict(
                step=step,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=stream.rng.bit_generator.state,
                settings=settings,
                identity=identity,
                config=vars(c),
            ),
            pending / "trainer.pt",
        )
        pending.rename(dest)
        pointer = out / ".latest.tmp"
        pointer.write_text(dest.name + "\n")
        pointer.replace(out / "latest.txt")
        commit()
        if keep_last:
            for old in sorted(out.glob("step-*"))[:-keep_last]:
                shutil.rmtree(old)
            commit()

    begin = time.perf_counter()
    measured_seconds = 0
    measured_tokens = 0
    last_saved = None
    end = min(steps, start_step + max_steps) if max_steps else steps
    with (out / "metrics.jsonl").open("w") as log:
        for index in range(start_step, end):
            sync()
            tick = time.perf_counter()
            rate = learning_rate(index, steps, warmup, lr)
            for group in optimizer.param_groups:
                group["lr"].fill_(rate)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(forward, *batch(stream))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Detect failure before an update/checkpoint can publish corrupt weights.
            if not math.isfinite(loss.item()) or not math.isfinite(norm.item()):
                raise RuntimeError("Nonfinite loss or gradient")
            update()
            sync()
            elapsed = time.perf_counter() - tick
            # Exclude compiler/warm-up updates from steady-state benchmark reporting.
            if index - start_step >= 10:
                measured_seconds += elapsed
                measured_tokens += batch_size * sequence
            step = index + 1
            row = dict(
                step=step,
                loss=loss.item(),
                grad_norm=norm.item(),
                lr=rate,
                train_tokens_per_second=batch_size * sequence / elapsed,
            )
            stopping = step == end or time.perf_counter() - begin >= max_seconds
            if step % eval_every == 0 or stopping:
                valid = TokenStream(data / "valid.bin", sequence, batch_size, seed + 1)
                with torch.no_grad():
                    row["valid_loss"] = (
                        sum(loss_fn(forward, *batch(valid)).item() for _ in range(eval_batches))
                        / eval_batches
                    )
                checkpoint(step)
                last_saved = step
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 100 == 0 or stopping:
                print(json.dumps(row), flush=True)
            if stopping:
                break
    summary = dict(
        torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
        parameters=sum(p.numel() for p in model.parameters()),
        benchmark_only=manifest.get("benchmark_only", False),
        last=row,
        checkpoint=f"step-{last_saved:06d}",
        measured_tokens=measured_tokens,
        measured_seconds=measured_seconds,
        train_tokens_per_second=measured_tokens / measured_seconds if measured_seconds else None,
        wall_seconds=time.perf_counter() - begin,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9
        if device.startswith("cuda")
        else None,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    commit()
    return summary
