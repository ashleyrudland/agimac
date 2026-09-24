import json

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer

from macoder.data import prepare, documents, TokenStream, SPECIALS
from macoder.model import Config, Model
from macoder.checkpoint import save_model, load_model, quantize


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"text": f"def f_{i}(x):\n    return x + {i}\n# λ 🐍\n"} for i in range(150)]
    source.write_text("\n".join(json.dumps(r) for r in rows + rows[:10]))
    out = tmp_path / "data"
    meta = prepare(source, out, 320, 0.2, 42, 0.5)
    return out, meta


def test_split_roundtrip_and_packing(corpus):
    path, meta = corpus
    train, valid = set(documents(path / "train.jsonl")), set(documents(path / "valid.jsonl"))
    assert not train & valid
    assert len(train | valid) == 150
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    text = '\tdef inédit(x):\n    return "🐍"\n'
    assert tok.decode(tok.encode(text).ids) == text
    x, y = TokenStream(path / "train.bin", 16, 2).batch()
    np.testing.assert_array_equal(np.array(x[:, 1:]), np.array(y[:, :-1]))
    valid_tokens = np.fromfile(path / "valid.bin", dtype="<u4")
    assert tok.token_to_id(SPECIALS[1]) not in valid_tokens


def test_save_load_and_quantize(corpus, tmp_path):
    path, meta = corpus
    m = Model(
        Config(
            vocab_size=meta["vocab_size"],
            dim=64,
            layers=2,
            heads=4,
            kv_heads=2,
            hidden_dim=128,
            context=32,
        )
    )
    x = mx.array([[1, 2, 3]])
    expected = m(x)
    checkpoint = tmp_path / "model"
    save_model(m, checkpoint, path / "tokenizer.json")
    restored, tok, _ = load_model(checkpoint)
    np.testing.assert_array_equal(np.array(expected), np.array(restored(x)))
    quantize(checkpoint, tmp_path / "q4")
    qmodel, _, _ = load_model(tmp_path / "q4")
    assert bool(mx.all(mx.isfinite(qmodel(x))).item())
    assert qmodel(x).shape == expected.shape
    with pytest.raises((ValueError, FileExistsError)):
        save_model(m, checkpoint, path / "tokenizer.json")
