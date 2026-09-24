"""Download a bounded prefix of pinned TinyStories text; no pretrained weights."""

import json
from pathlib import Path
import urllib.request

REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
root = Path("data/stories-source")
root.mkdir(parents=True, exist_ok=True)
for split, limit in [("train", 50000), ("valid", 2000)]:
    url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REV}/TinyStoriesV2-GPT4-{split}.txt"
    dest = root / f"{split}.jsonl"
    if dest.exists():
        raise ValueError(f"Already exists: {dest}")
    n = 0
    story = []
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("w") as f:
        for raw in response:
            line = raw.decode("utf-8")
            if "<|endoftext|>" in line:
                story.append(line.split("<|endoftext|>")[0])
                text = "".join(story).strip()
                if text:
                    f.write(json.dumps({"text": text}) + "\n")
                    n += 1
                story = []
                if n >= limit:
                    break
            else:
                story.append(line)
    print(split, n, flush=True)
    (root / f"{split}-source.json").write_text(
        json.dumps(
            {
                "dataset": "roneneldan/TinyStories",
                "revision": REV,
                "url": url,
                "documents": n,
                "selection": "first N complete documents",
                "license": "cdla-sharing-1.0",
                "usage": "local research; original split preserved",
            },
            indent=2,
        )
        + "\n"
    )
