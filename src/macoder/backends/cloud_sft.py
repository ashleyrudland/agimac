"""One bounded epoch of assistant-only SFT; deterministic batches and resumable state."""

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from .torch_model import Model
from ..config import Config


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Batches:
    """Same shuffled length-bucket schedule and shifted masks as the MLX trainer."""

    def __init__(self, root, split, batch_size, seed=42):
        self.arrays = {}
        self.plans = []
        self.batch_size = batch_size
        rng = np.random.default_rng(seed)
        for length in (256, 512, 1024):
            ids = np.load(Path(root) / f"{split}-{length}-ids.npy", mmap_mode="r")
            masks = np.load(Path(root) / f"{split}-{length}-mask.npy", mmap_mode="r")
            if len(ids):
                self.arrays[length] = (ids, masks)
                order = rng.permutation(len(ids)) if split == "train" else np.arange(len(ids))
                self.plans.extend(
                    (length, order[i : i + batch_size]) for i in range(0, len(ids), batch_size)
                )
        if split == "train":
            rng.shuffle(self.plans)

    def batch(self, i):
        length, index = self.plans[i]
        source, mask = self.arrays[length]
        ids = np.zeros((self.batch_size, length + 1), dtype=np.int64)
        masks = np.zeros_like(ids, dtype=np.float32)
        ids[: len(index)] = source[index]
        masks[: len(index)] = mask[index]
        targets = int(masks[:, 1:].sum())
        if targets < 1:
            raise ValueError("Empty supervised batch")
        return (
            ids[:, :-1].copy(),
            ids[:, 1:].copy(),
            masks[:, 1:].copy(),
            int(np.count_nonzero(ids)),
            targets,
        )


def masked_loss(model, x, y, mask):
    # Users, tool results and padding remain visible context, but have zero loss.
    ce = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), reduction="none").reshape(
        y.shape
    )
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def train(
    base,
    data,
    out,
    expected_manifest,
    commit=lambda: None,
    resume=None,
    device="cuda",
    compile_model=True,
    batch_size=32,
    lr=5e-5,
    pilot_steps=128,
    max_seconds=2700,
):
    base, data, out = map(Path, (base, data, out))
    if out.exists():
        raise ValueError("Output exists: use a fresh directory")
    if digest(data / "manifest.json") != expected_manifest:
        raise ValueError("SFT manifest differs from local data")
    manifest = json.loads((data / "manifest.json").read_text())
    meta = json.loads((base / "metadata.json").read_text())
    for name, h in manifest["array_hashes"].items():
        if digest(data / name) != h:
            raise ValueError("Array mismatch: " + name)
    if (
        digest(data / "tokenizer.json") != meta["tokenizer_sha256"]
        or manifest["tokenizer_sha256"] != meta["tokenizer_sha256"]
    ):
        raise ValueError("Tokenizer mismatch")
    torch.manual_seed(42)
    c = Config.read(base / "config.json")
    model = Model(c).to(device)
    model.load_state_dict(load_file(str(base / "model.safetensors"), device=device))
    # FP32 parameters/moments with BF16 CUDA autocast. Unlike pretraining, SFT uses
    # standard bias-corrected AdamW, lower LR, and weight decay 0.01.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, fused=device == "cuda"
    )
    train_data = Batches(data, "train", batch_size)
    valid = Batches(data, "valid", batch_size)
    total = len(train_data.plans)
    start_step = 0
    seen = targets_seen = 0
    best = float("inf")
    best_name = None
    pilot_passed = False
    settings = dict(
        batch_size=batch_size,
        lr=lr,
        seed=42,
        total_steps=total,
        pilot_steps=pilot_steps,
        manifest=expected_manifest,
        base_sha256=digest(base / "model.safetensors"),
    )
    if resume:
        state = torch.load(Path(resume) / "trainer.pt", map_location=device, weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Resume schedule or data changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        seen = state["seen"]
        targets_seen = state["targets"]
        best = state["best"]
        best_name = state["best_checkpoint"]
        pilot_passed = state["pilot_passed"]
    out.mkdir(parents=True)
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    loss_func = torch.compile(masked_loss) if compile_model else masked_loss

    def batch(source, i):
        x, y, m, n, t = source.batch(i)
        return tuple(torch.from_numpy(a).to(device) for a in (x, y, m)), n, t

    def loss(args):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            return loss_func(model, *args)

    def evaluate():
        model.eval()
        summed = denom = 0
        with torch.no_grad():
            for i in np.unique(
                np.linspace(0, len(valid.plans) - 1, min(24, len(valid.plans)), dtype=int)
            ):
                b, _, t = batch(valid, int(i))
                value = float(loss(b).item())
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite validation loss")
                summed += value * t
                denom += t
        model.train()
        return summed / denom

    baseline = evaluate()
    best = min(best, baseline)
    (out / "initial-evaluation.json").write_text(json.dumps({"valid_loss": baseline}) + "\n")
    commit()
    begin = time.monotonic()

    def save(step, improved):
        nonlocal best_name
        name = f"step-{step:06d}"
        tmp = out / (".pending-" + name)
        dest = out / name
        tmp.mkdir()
        if improved:
            best_name = str(dest)
        save_file(
            {
                k: v.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
                for k, v in model.state_dict().items()
            },
            str(tmp / "model.safetensors"),
        )
        c.write(tmp / "config.json")
        shutil.copyfile(data / "tokenizer.json", tmp / "tokenizer.json")
        m = {
            **meta,
            "dtype": "bfloat16",
            "backend": "torch-cuda-sft",
            "step": step,
            "tokens_seen": meta["tokens_seen"] + seen,
            "initial_tokens_seen": meta["tokens_seen"],
            "sft_input_tokens": seen,
            "sft_assistant_targets": targets_seen,
            "parent_checkpoint": str(base),
            "parent_sha256": settings["base_sha256"],
            "source_sha256": expected_manifest,
            "chat_format": "macoder-chat-v1",
            "tools": ["add", "subtract", "multiply", "divide"],
            "trained_context": 1024,
        }
        (tmp / "metadata.json").write_text(json.dumps(m, indent=2))
        torch.save(
            dict(
                step=step,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                settings=settings,
                seen=seen,
                targets=targets_seen,
                best=best,
                best_checkpoint=best_name,
                pilot_passed=pilot_passed,
            ),
            tmp / "trainer.pt",
        )
        tmp.rename(dest)
        (out / "latest.txt").write_text(str(dest) + "\n")
        if best_name:
            (out / "best.txt").write_text(best_name + "\n")
        commit()
        for old in sorted(out.glob("step-*"))[:-2]:
            if str(old) != best_name:
                shutil.rmtree(old)
        commit()

    status = "running"
    measured_tokens = 0
    measured_seconds = 0
    with (out / "metrics.jsonl").open("w") as log:
        for index in range(start_step, total):
            tick = time.monotonic()
            b, n, t = batch(train_data, index)
            # Warm up gently, then decay over this single epoch.
            rate = (
                lr
                * min(1.0, (index + 1) / 64)
                * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * index / total)))
            )
            for g in optimizer.param_groups:
                g["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            value = loss(b)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(value.item()) or not math.isfinite(norm.item()):
                raise RuntimeError("Nonfinite SFT loss/gradient")
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            seen += n
            targets_seen += t
            step = index + 1
            if index >= 10:
                measured_tokens += n
                measured_seconds += elapsed
            row = dict(
                step=step,
                total_steps=total,
                loss=value.item(),
                grad_norm=norm.item(),
                lr=rate,
                input_tokens=seen,
                assistant_targets=targets_seen,
                input_tokens_per_second=n / elapsed,
                elapsed_seconds=time.monotonic() - begin,
            )
            stopping = step == total or time.monotonic() - begin >= max_seconds
            if step == pilot_steps or step % 500 == 0 or stopping:
                row["valid_loss"] = evaluate()
                improved = row["valid_loss"] < best
                if improved:
                    best = row["valid_loss"]
                if step == pilot_steps:
                    pilot_passed = row["valid_loss"] < baseline
                    if not pilot_passed:
                        status = "pilot_failed"
                        stopping = True
                save(step, improved)
                (out / "status.json").write_text(
                    json.dumps(
                        dict(
                            stage=status,
                            **row,
                            pilot_passed=pilot_passed,
                            best_checkpoint=best_name,
                        )
                    )
                )
                commit()
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 25 == 0 or stopping:
                print(json.dumps(row), flush=True)
            if stopping:
                break
    if status != "pilot_failed":
        status = "complete" if step == total else "time_budget_reached"
    summary = dict(
        status=status,
        last=row,
        pilot_passed=pilot_passed,
        baseline_valid_loss=baseline,
        best_valid_loss=best,
        best_checkpoint=best_name,
        input_tokens_per_second=measured_tokens / max(measured_seconds, 1e-9),
        wall_seconds=time.monotonic() - begin,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "status.json").write_text(json.dumps(summary, indent=2))
    commit()
    return summary
