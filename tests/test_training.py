from argparse import Namespace
import json

import mlx.core as mx
import numpy as np
import pytest

from macoder.data import prepare
from macoder.model import Config
from macoder.checkpoint import load_model
from macoder.train import train
from macoder.inference import generate_ids


@pytest.mark.parametrize(
    "compiled,full_step,architecture,master",
    [
        (False, False, "legacy", False),
        (True, False, "legacy", False),
        (False, True, "nano_core", False),
        (False, True, "nano_core", True),
    ],
)
def test_resume_reproduces_uninterrupted_training(
    tmp_path, compiled, full_step, architecture, master
):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"def f_{i}(x): return x + {i}\n"}) for i in range(80))
    )
    data = tmp_path / "data"
    prepare(source, data, 320, 0.2)
    config = tmp_path / "config.json"
    Config(
        vocab_size=320,
        dim=64,
        layers=1,
        heads=4,
        kv_heads=2,
        hidden_dim=128,
        context=64,
        architecture=architecture,
    ).write(config)
    args = Namespace(
        data=str(data),
        output=str(tmp_path / "full"),
        config=str(config),
        steps=6,
        batch_size=1,
        sequence=16,
        accumulate=1 if full_step else 2,
        eval_every=3,
        eval_batches=1,
        lr=0.001,
        warmup=1,
        seed=42,
        dtype="bfloat16" if master else "float32",
        master_weights=master,
        compile=compiled,
        compile_step=full_step,
        resume=None,
        log_every=10,
    )
    train(args)
    args.resume = str(tmp_path / "full/step-000003")
    args.output = str(tmp_path / "resumed")
    train(args)
    a, _, _ = load_model(tmp_path / "full/step-000006")
    b, _, _ = load_model(tmp_path / "resumed/step-000006")
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_array_equal(np.array(a(ids)), np.array(b(ids)))
    assert list(generate_ids(a, [1, 2], 5)) == list(generate_ids(b, [1, 2], 5))


def test_weight_continuation_keeps_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"Example {i}: hello world."}) for i in range(100))
    )
    data = tmp_path / "data"
    prepare(source, data, 320, 0.2)
    config = tmp_path / "config.json"
    Config(vocab_size=320, dim=64, layers=1, heads=4, kv_heads=2, hidden_dim=128, context=64).write(
        config
    )
    args = Namespace(
        data=str(data),
        output=str(tmp_path / "first"),
        config=str(config),
        steps=2,
        batch_size=1,
        sequence=16,
        accumulate=1,
        eval_every=1,
        eval_batches=1,
        keep_last=1,
        lr=0.001,
        warmup=1,
        seed=42,
        dtype="float32",
        compile=False,
        resume=None,
        log_every=10,
    )
    train(args)
    assert not (tmp_path / "first/step-000001").exists()
    assert not list((tmp_path / "first").glob(".pending-*"))
    assert (tmp_path / "first/latest.txt").read_text().strip() == str(
        tmp_path / "first/step-000002"
    )
    args.init_from = str(tmp_path / "first/step-000002")
    args.output = str(tmp_path / "second")
    args.steps = 1
    train(args)
    _, _, meta = load_model(tmp_path / "second/step-000001")
    assert meta["tokens_seen"] == 48
    assert meta["initial_tokens_seen"] == 32
    assert meta["parent_checkpoint"] == args.init_from
    assert meta["trained_from_scratch"] is True
    args.output = str(tmp_path / "bad")
    args.resume = args.init_from
    with pytest.raises(ValueError, match="Choose either"):
        train(args)


def test_stop_request_saves_completed_update(tmp_path, monkeypatch):
    import signal
    from macoder.data import TokenStream

    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": f"Example {i}: hello world."}) for i in range(100))
    )
    data = tmp_path / "data"
    prepare(source, data, 320, 0.2)
    config = tmp_path / "config.json"
    Config(vocab_size=320, dim=32, layers=1, heads=4, kv_heads=2, hidden_dim=64, context=32).write(
        config
    )
    args = Namespace(
        data=str(data),
        output=str(tmp_path / "stopped"),
        config=str(config),
        steps=10,
        batch_size=1,
        sequence=8,
        accumulate=1,
        eval_every=10,
        eval_batches=1,
        keep_last=1,
        lr=0.001,
        warmup=1,
        seed=42,
        dtype="float32",
        master_weights=False,
        compile=False,
        compile_step=False,
        resume=None,
        log_every=10,
    )
    original = TokenStream.batch

    def stop_after_batch(self):
        batch = original(self)
        signal.raise_signal(signal.SIGTERM)
        return batch

    monkeypatch.setattr(TokenStream, "batch", stop_after_batch)
    result = train(args)
    assert result["status"] == "stopped" and result["steps_completed"] == 1
    assert (tmp_path / "stopped/step-000001/trainer.json").exists()
    model, _, _ = load_model(tmp_path / "stopped/step-000001")
    assert np.isfinite(np.array(model(mx.array([[1, 2]])))).all()
