"""Build an expanded benchmark exclusion list, then the pinned 2B-token mixture."""

import hashlib
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path
from prepare_general import WORDS

root = Path("data/general-v2")
root.mkdir(exist_ok=True)
path = root / "benchmark-filter.json"
if not path.exists():
    original = Path("recipes/reference/benchmark-filter.json")
    old = json.loads(original.read_text())
    phrases = set(old["phrases"])
    evidence = {str(original): hashlib.sha256(original.read_bytes()).hexdigest()}
    # Boundary phrases limit RAM and false overlap on generic short questions.
    # This intentionally is not advertised as exhaustive semantic decontamination.
    bundle = Path("data/core-eval/eval_bundle")
    tasks = yaml.safe_load((bundle / "core.yaml").read_text())["icl_tasks"]
    # The bundle includes unused auxiliary datasets; filter only the configured CORE tasks.
    for p in sorted({bundle / "eval_data" / t["dataset_uri"] for t in tasks}):
        evidence[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in p.open():
            row = json.loads(line)
            for key in ("query", "context", "question", "prompt"):
                value = row.get(key)
                if isinstance(value, str):
                    words = WORDS.findall(value.lower())
                    if len(words) >= 13:
                        phrases.add(" ".join(words[:13]))
                        phrases.add(" ".join(words[-13:]))
    for p in sorted(Path("reports/sft-v1-nanochat").glob("*.jsonl")):
        evidence[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in p.open():
            words = WORDS.findall(json.loads(line)["prompt"].lower())
            phrases.update(" ".join(words[i : i + 13]) for i in range(len(words) - 12))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "sources": evidence,
                "phrases": sorted(phrases),
                "method": "Original HumanEval/MBPP all 13-word phrases; nanochat chat prompts all 13-word phrases; CORE prompt first/last 13 words. No semantic or exhaustive short-prompt contamination guarantee.",
            }
        )
    )
    temporary.replace(path)
for attempt in range(5):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/prepare_general.py",
            "--root",
            str(root),
            "--tokens",
            "2000000000",
            "--expanded",
        ]
    )
    if result.returncode == 0:
        break
    if attempt == 4:
        raise SystemExit(result.returncode)
    print("Preparation retry", attempt + 1, "completed sources are reused", flush=True)
    time.sleep(60)
