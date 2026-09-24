"""Regenerate the 11B-token estimate from saved, synchronized MLX probes."""

import json
from pathlib import Path

rows = []
for p in sorted(Path("reports").glob("probe-*.json")):
    d = json.loads(p.read_text())
    if "aggregate_tokens_per_second" not in d:
        continue
    rate = d["aggregate_tokens_per_second"]
    days = 11e9 / rate / 86400
    rows.append(
        f"| {p.stem.removeprefix('probe-')} | {d['parameters'] / 1e6:.1f}M | {d['sequence']} | {d['batch']} | {rate:,.0f} | {d['peak_memory_gb']:.2f} GB | {days:.1f} | {days * 1.25:.1f}–{days * 1.5:.1f} |"
    )
text = (
    """# Training throughput and 11B-token budget

Measured on this Apple M2 Pro (32GB), with CORE evaluation temporarily suspended to avoid GPU contention. Models are random-initialized. Data batches come from the real, packed general-v1 corpus. Three warm-up iterations are excluded; each reported step includes data batching, gradients, clipping, optimizer updates and synchronization. This is pretraining throughput, not inference tokens/sec.

| Probe | Parameters | Sequence | Batch | Tokens/sec | Peak MLX memory | 11B raw days | Planning days (+25–50%) |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    + "\n".join(rows)
    + """

The planning allowance is an assumption, not a measured confidence interval. It allows for validation, checkpoint writes and sustained-throughput variation; it excludes data preparation and subsequent instruction training. Most probes cover 20 measured updates; b16 covers 30 and the `long` probe covers 200. Even 200 steps do not establish month-long performance or numerical stability.

The whole-update compiler improved the legacy model over compiled-gradients-only in these probes. BF16 provided another gain with FP32 Adam moments, but no FP32 master weights. The experimental candidate's short-run loss is not a fair quality comparison: initialization, parameter count and architecture differ. No benchmark gain has been established.

The 72.1M candidate is nanochat-inspired, not a complete port. The 145.0M candidate illustrates the capacity/speed tradeoff. Increasing batch size 8 to 16 on the smaller candidate produced only a modest short-probe gain at substantially higher memory use. Sequence 1024 was slower per token than 512.

An 11B-token run requires a much larger corpus first. The existing prepared stream has 80M base training tokens, so blindly running it for 11B tokens would repeat it approximately 138 times. Token counts are tokenizer-dependent. Matching nanochat's token count with a much smaller model does not imply matching quality.

Implementation: `src/macoder/model.py`, `src/macoder/train.py`, `experiments/pretrain_speed.py`; architecture choices and exclusions: `README.md`. The existing trained checkpoints are preserved. No full new training run has started.
"""
)
Path("reports/TRAINING-SPEED-11B.md").write_text(text)
