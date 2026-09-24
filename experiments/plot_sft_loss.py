"""Render a read-only snapshot of SFT metrics, using CPU only."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = []
for line in (ROOT / "runs/sft-v1/metrics.jsonl").read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
steps = np.array([r["step"] for r in rows])
loss = np.array([r["loss"] for r in rows])
window = min(50, len(rows))
smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
initial = json.loads((ROOT / "runs/sft-v1/initial-evaluation.json").read_text())["valid_loss"]
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
ax.plot(
    steps, loss, color="#aac0df", alpha=0.55, lw=0.7, label="Training loss • individual batches"
)
ax.plot(
    steps[window - 1 :],
    smooth,
    color="#2459b5",
    lw=2.2,
    label=f"Training loss • {window}-step moving average",
)
valid = [r for r in rows if "valid_loss" in r]
ax.plot(
    [0] + [r["step"] for r in valid],
    [initial] + [r["valid_loss"] for r in valid],
    color="#d37525",
    marker="o",
    lw=1.8,
    label="Held-out development loss",
)
ax.set(
    title="Macoder conversation training — loss so far",
    xlabel="Training step",
    ylabel="Assistant-token cross-entropy (lower is better)",
)
ax.legend(loc="upper right", fontsize=9)
last = rows[-1]
notes = f"Snapshot: step {last['step']:,} / {last['total_steps']:,}   •   {last['actual_input_tokens'] / 1e6:.2f}M input tokens   •   {last['elapsed_seconds'] / 60:.1f} minutes"
fig.text(0.09, 0.025, notes, fontsize=9, color="#334155")
fig.tight_layout(rect=[0, 0.055, 1, 1])
out = ROOT / "reports/sft-v1-loss.png"
fig.savefig(out)
plt.close(fig)
print(
    json.dumps(
        {
            "chart": str(out),
            "step": last["step"],
            "moving_average": float(smooth[-1]),
            "initial_dev_loss": initial,
            "development_checks": len(valid),
        }
    )
)
