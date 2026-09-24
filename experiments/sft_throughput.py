"""Synchronized synthetic SFT shape probe. Does not modify saved model weights."""

import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from macoder.checkpoint import load_model
from macoder.sft import masked_loss
from macoder.train import StableAdamW

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, required=True)
a = p.parse_args()
model, _, _ = load_model("runs/general-v1/step-048829")
model.train()
o = StableAdamW(learning_rate=0.0001)
fn = nn.value_and_grad(model, masked_loss)


def vg(x, y, m):
    return fn(model, x, y, m)


vg = mx.compile(vg, inputs=model.state, outputs=model.state)
xy = mx.random.randint(10, 16000, (a.batch, 1025))
mask = mx.ones((a.batch, 1024))
mx.eval(xy, mask)
rates = []
for i in range(10):
    start = time.perf_counter()
    loss, g = vg(xy[:, :-1], xy[:, 1:], mask)
    g, _ = optim.clip_grad_norm(g, 1.0)
    o.update(model, g)
    mx.eval(model.parameters(), o.state, loss)
    if i >= 3:
        rates.append(a.batch * 1024 / (time.perf_counter() - start))
r = {
    "batch_size": a.batch,
    "sequence": 1024,
    "dtype": "float32",
    "compiled": True,
    "median_padded_tokens_per_second": statistics.median(rates),
    "peak_memory_gb": mx.get_peak_memory() / 1e9,
    "synthetic_short_probe": True,
}
Path(f"reports/sft-throughput-b{a.batch}.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r))
