"""Short real-data pretraining probe; randomly initialized weights are discarded."""

import argparse
import json
import statistics
import time
import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from macoder.model import Config, Model, loss_fn
from macoder.data import TokenStream
from macoder.train import StableAdamW, MasterAdamW, make_compiled_step

p = argparse.ArgumentParser(__doc__)
p.add_argument("--master-weights", action="store_true")
p.add_argument("--config", required=True)
p.add_argument("--dtype", default="float32")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--sequence", type=int, default=512)
p.add_argument("--mode", choices=["gradient", "full"], default="full")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--output", required=True)
a = p.parse_args()
mx.random.seed(42)
c = Config.read(a.config)
m = Model(c)
m.set_dtype(getattr(mx, a.dtype))
m.train()
o = (MasterAdamW if a.master_weights else StableAdamW)(learning_rate=1e-4, weight_decay=0.1)
stream = TokenStream("data/general-v1/prepared/train.bin", a.sequence, a.batch, 42)
if a.mode == "full":
    fn = make_compiled_step(m, o)
else:
    vg = nn.value_and_grad(m, loss_fn)

    def grad(x, y):
        return vg(m, x, y)

    grad = mx.compile(grad, inputs=m.state, outputs=m.state)

    def fn(x, y, lr):
        o.learning_rate = lr
        loss, g = grad(x, y)
        mx.eval(loss, g)
        g, norm = optim.clip_grad_norm(g, 1.0)
        o.update(m, g)
        return loss, norm


mx.eval(m.parameters(), o.state)
times = []
losses = []
for i in range(a.steps + 3):
    start = time.perf_counter()
    x, y = stream.batch()
    loss, norm = fn(x, y, mx.array(1e-4))
    mx.eval(m.parameters(), o.state, loss, norm)
    duration = time.perf_counter() - start
    if not math.isfinite(loss.item()) or not math.isfinite(norm.item()):
        raise RuntimeError("Nonfinite probe")
    if i >= 3:
        times.append(duration)
        losses.append(loss.item())
result = {
    **vars(a),
    "parameters": m.parameter_count(),
    "median_tokens_per_second": a.batch * a.sequence / statistics.median(times),
    "aggregate_tokens_per_second": a.batch * a.sequence * len(times) / sum(times),
    "peak_memory_gb": mx.get_peak_memory() / 1e9,
    "loss_first": losses[0],
    "loss_last": losses[-1],
    "finite": True,
    "measured_seconds": sum(times),
    "limitation": "Short real-data shape probe, not converged quality or sustained training; excludes evaluation/checkpoint overhead",
}
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
