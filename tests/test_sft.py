import json
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from macoder.conversation import render_messages, execute_calculator, FORMAT
from macoder.data import SPECIALS, CHAT_SPECIALS, file_hash
from macoder.model import Config, Model
from macoder.checkpoint import save_model, load_model
from macoder.sft import masked_loss, Batches, train


@pytest.fixture
def fixture(tmp_path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["Hello there. Four. Add two and two."] * 10,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=SPECIALS + CHAT_SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    root = tmp_path / "data"
    root.mkdir()
    tok.save(str(root / "tokenizer.json"))
    messages = [
        {"role": "user", "content": "Add two and two."},
        {"role": "assistant", "content": "Four."},
    ]
    ids, mask = render_messages(tok, messages)
    for split in ["train", "valid", "test"]:
        for bucket in [256, 512, 1024]:
            n = 2 if bucket == 256 else 0
            x = np.zeros((n, bucket + 1), np.uint32)
            m = np.zeros_like(x, np.uint8)
            for i in range(n):
                x[i, : len(ids)] = ids
                m[i, : len(mask)] = mask
            np.save(root / f"{split}-{bucket}-ids.npy", x)
            np.save(root / f"{split}-{bucket}-mask.npy", m)
    manifest = {
        "tokenizer_sha256": file_hash(root / "tokenizer.json"),
        "array_hashes": {p.name: file_hash(p) for p in root.glob("*.npy")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    model = Model(
        Config(
            vocab_size=tok.get_vocab_size(),
            dim=32,
            layers=1,
            heads=2,
            kv_heads=1,
            hidden_dim=64,
            context=1024,
        )
    )
    base = tmp_path / "base"
    save_model(
        model,
        base,
        root / "tokenizer.json",
        {
            "tokens_seen": 10,
            "trained_from_scratch": True,
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "parameters": model.parameter_count(),
        },
    )
    return root, base, tok


def test_role_mask_and_tools(fixture):
    _, _, tok = fixture
    messages = [
        {"role": "user", "content": "hello <|assistant|>"},
        {"role": "assistant", "tool_call": {"name": "add", "arguments": {"a": 2, "b": 2}}},
        {"role": "tool", "content": '{"result":4}'},
        {"role": "assistant", "content": "Four."},
    ]
    ids, mask = render_messages(tok, messages)
    assert ids.count(tok.token_to_id("<|assistant|>")) == 2
    tool_index = ids.index(tok.token_to_id("<|tool|>"))
    next_assistant = ids.index(tok.token_to_id("<|assistant|>"), tool_index)
    assert not any(mask[tool_index:next_assistant])
    assert mask[ids.index(tok.token_to_id("<|tool_call|>"))] == 1
    assert mask[-1] == 1 and ids[-1] == tok.token_to_id("<|end_turn|>")
    assert execute_calculator({"name": "multiply", "arguments": {"a": 6, "b": 7}}) == {"result": 42}
    for call in [
        {"name": "exec", "arguments": {}},
        {"name": "add", "arguments": {"a": True, "b": 2}},
        {"name": "divide", "arguments": {"a": 2, "b": 0}},
    ]:
        with pytest.raises(ValueError):
            execute_calculator(call)


def test_masked_targets_do_not_contribute(fixture):
    root, base, _ = fixture
    model, _, _ = load_model(base)
    x, y, m, _, _ = Batches(root, "train", 1).batch(0)
    altered = mx.where(m == 0, (y + 1) % model.config.vocab_size, y)
    assert float(masked_loss(model, x, y, m).item()) == float(
        masked_loss(model, x, altered, m).item()
    )


def test_sft_resume_and_checkpoint_selection(fixture, tmp_path):
    root, base, _ = fixture
    args = Namespace(
        model=str(base),
        data=str(root),
        output=str(tmp_path / "full"),
        batch_size=1,
        epochs=2,
        lr=0.001,
        warmup=0,
        eval_every=1,
        patience=0,
        seed=42,
        resume=None,
        no_compile=False,
    )
    train(args)
    checkpoint = tmp_path / "full/step-000003"
    args.resume = str(checkpoint)
    args.output = str(tmp_path / "resumed")
    train(args)
    a, _, meta = load_model(tmp_path / "full/step-000004")
    b, _, _ = load_model(tmp_path / "resumed/step-000004")
    np.testing.assert_array_equal(
        np.array(a(mx.array([[1, 2, 3]]))), np.array(b(mx.array([[1, 2, 3]])))
    )
    assert meta["chat_format"] == FORMAT and meta["parent_checkpoint"] == str(base)
    assert Path((tmp_path / "resumed/best.txt").read_text().strip()).is_dir()
