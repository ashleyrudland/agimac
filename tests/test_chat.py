import json
import os
from types import SimpleNamespace

import pytest

from macoder.chat import latest_checkpoint, prompt_ids


def test_latest_skips_partial_and_untrained_checkpoints(tmp_path):
    def checkpoint(name, timestamp, tokens=10, complete=True):
        path = tmp_path / name
        path.mkdir()
        for filename in ("config.json", "tokenizer.json", "model.safetensors"):
            if complete or filename != "model.safetensors":
                (path / filename).write_text("{}")
        meta = path / "metadata.json"
        meta.write_text(json.dumps({"tokens_seen": tokens}))
        os.utime(meta, ns=(timestamp, timestamp))
        return path

    checkpoint("old", 100)
    selected = checkpoint("q4", 200)
    checkpoint("partial", 300, complete=False)
    checkpoint("random", 400, tokens=0)
    assert latest_checkpoint(tmp_path) == selected
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(tmp_path / "missing")


def test_context_budget_does_not_silently_truncate_prompt():
    class Tokenizer:
        def encode(self, text):
            return SimpleNamespace(ids=list(range(len(text))))

        def token_to_id(self, text):
            return 0

    tokenizer = Tokenizer()
    assert prompt_ids(tokenizer, "abc", 8, 20) == ([0, 1, 2], 5)
    assert prompt_ids(tokenizer, "", 8, 2) == ([0], 2)
    with pytest.raises(ValueError, match="Shorten"):
        prompt_ids(tokenizer, "12345678", 8, 1)
