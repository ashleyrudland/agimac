"""Create a small stratified real-token sample for SPEED tests, never quality claims."""

import json
from pathlib import Path
import numpy as np
import shutil
from macoder.data import file_hash


def main():
    source = Path("data/general-v2/prepared")
    dest = Path("data/modal-speed-sample")
    dest.mkdir(exist_ok=False)
    tokens = np.memmap(source / "train.bin", dtype="<u4", mode="r")
    # Prepared v2 packs sources in this order. Verify the expected corpus size.
    lengths = [800_000_000, 600_000_000, 400_000_000, 200_000_000]
    if len(tokens) != sum(lengths):
        raise ValueError("Unexpected v2 corpus layout")
    ranges = []
    offset = 0
    with (dest / "train.bin").open("wb") as out:
        for size in lengths:
            # Four spaced windows per source; retain its 40/30/20/10 share.
            count = size // 1000  # 8M tokens overall, 0.4% of full source
            for fraction in (0.1, 0.35, 0.6, 0.85):
                start = offset + int(size * fraction)
                tokens[start : start + count].tofile(out)
                ranges.append([start, start + count])
            offset += size
    for name in ("valid.bin", "tokenizer.json"):
        shutil.copyfile(source / name, dest / name)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(
        tokens={"train": 8_000_000, "valid": 400_000},
        source_sha256=file_hash(dest / "train.bin"),
        benchmark_only=True,
        note="Stratified subset of full packed corpus; throughput only, NOT quality or identical full-data training",
        parent_manifest_sha256=file_hash(source / "manifest.json"),
        source_token_ranges=ranges,
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
