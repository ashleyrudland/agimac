import runpy
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from macoder.model import Config, Model


def test_sentence_scores_ignore_padding_and_match_cached_factorization():
    score = runpy.run_path(str(Path(__file__).parents[1] / "experiments/blimp.py"))[
        "sentence_scores"
    ]

    class Tokenizer:
        def token_to_id(self, _):
            return 0

        def encode(self, text):
            return SimpleNamespace(ids=[int(s) for s in text.split()])

    mx.random.seed(3)
    m = Model(
        Config(vocab_size=64, dim=64, layers=2, heads=4, kv_heads=2, hidden_dim=128, context=32)
    )
    texts = ["1 2 3 4", "2", "5 6"]
    together = score(m, Tokenizer(), texts, batch_size=3)
    separately = score(m, Tokenizer(), texts, batch_size=1)
    np.testing.assert_allclose(together, separately, rtol=1e-5, atol=1e-5)
    cache = m.make_cache()
    previous = 0
    expected = 0.0
    for target in [1, 2, 3, 4]:
        logits = m(mx.array([[previous]]), cache)[:, 0, :]
        expected -= float(
            nn.losses.cross_entropy(logits, mx.array([target]), reduction="sum").item()
        )
        previous = target
    np.testing.assert_allclose(together[0], expected, rtol=1e-5, atol=1e-5)
