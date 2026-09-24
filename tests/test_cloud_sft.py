import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from macoder.sft import Batches as MLXBatches, masked_loss as mlx_loss
from macoder.backends.cloud_sft import Batches, masked_loss
from macoder.model import Model as MLXModel
from macoder.backends.torch_model import Model
from macoder.config import Config


def test_bucket_schedule_and_masks(tmp_path):
    for length in (256, 512, 1024):
        ids = np.arange(3 * (length + 1), dtype=np.uint32).reshape(3, length + 1) % 31
        mask = (ids % 3 == 0).astype(np.uint8)
        np.save(tmp_path / f"train-{length}-ids.npy", ids)
        np.save(tmp_path / f"train-{length}-mask.npy", mask)
    a = Batches(tmp_path, "train", 2)
    b = MLXBatches(tmp_path, "train", 2)
    for i in range(len(a.plans)):
        aa = a.batch(i)
        bb = b.batch(i)
        for x, y in zip(aa[:3], bb[:3]):
            np.testing.assert_array_equal(x, np.array(y))
        assert aa[3:] == bb[3:]


def test_masked_gradient_parity_and_ignored_targets():
    c = Config(
        vocab_size=32,
        dim=32,
        layers=2,
        heads=4,
        kv_heads=2,
        hidden_dim=64,
        context=16,
        architecture="nano_core",
    )
    mx.random.seed(42)
    m = MLXModel(c)
    t = Model(c)
    t.load_state_dict({k: torch.tensor(np.array(v)) for k, v in tree_flatten(m.parameters())})
    x = np.array([[1, 2, 3, 4]], dtype=np.int64)
    y = np.array([[2, 3, 4, 5]], dtype=np.int64)
    mask = np.array([[0, 0, 1, 1]], dtype=np.float32)
    ml, mg = nn.value_and_grad(m, mlx_loss)(m, mx.array(x), mx.array(y), mx.array(mask))
    mx.eval(ml, mg)
    tl = masked_loss(t, torch.tensor(x), torch.tensor(y), torch.tensor(mask))
    tl.backward()
    np.testing.assert_allclose(tl.item(), ml.item(), rtol=1e-5)
    for k, v in tree_flatten(mg):
        np.testing.assert_allclose(
            t.get_parameter(k).grad.numpy(), np.array(v), rtol=2e-3, atol=2e-5
        )
    y[0, :2] = 31
    assert (
        abs(masked_loss(t, torch.tensor(x), torch.tensor(y), torch.tensor(mask)).item() - tl.item())
        < 1e-6
    )


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    import json
    from safetensors.torch import save_file, load_file
    from macoder.backends.cloud_sft import train, digest

    base = tmp_path / "base"
    base.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    c = Config(
        vocab_size=32,
        dim=16,
        layers=1,
        heads=2,
        kv_heads=1,
        hidden_dim=32,
        context=1024,
        architecture="nano_core",
    )
    c.write(base / "config.json")
    save_file(Model(c).state_dict(), str(base / "model.safetensors"))
    (base / "tokenizer.json").write_text("{}")
    (data / "tokenizer.json").write_text("{}")
    (base / "metadata.json").write_text(
        json.dumps(
            dict(
                tokenizer_sha256=digest(data / "tokenizer.json"),
                tokens_seen=123,
                trained_from_scratch=True,
            )
        )
    )
    arrays = {}
    for split in ("train", "valid"):
        for length in (256, 512, 1024):
            ids = np.zeros((2 if length == 256 else 0, length + 1), dtype=np.uint32) + 3
            masks = np.ones_like(ids, dtype=np.uint8)
            masks[:, :3] = 0
            for kind, a in [("ids", ids), ("mask", masks)]:
                name = f"{split}-{length}-{kind}.npy"
                np.save(data / name, a)
                arrays[name] = digest(data / name)
    (data / "manifest.json").write_text(
        json.dumps(dict(array_hashes=arrays, tokenizer_sha256=digest(data / "tokenizer.json")))
    )
    kwargs = dict(
        expected_manifest=digest(data / "manifest.json"),
        device="cpu",
        compile_model=False,
        batch_size=1,
        pilot_steps=1,
    )
    full = train(base, data, tmp_path / "full", **kwargs)
    train(base, data, tmp_path / "first", max_seconds=0, **kwargs)
    resumed = train(
        base, data, tmp_path / "resumed", resume=tmp_path / "first/step-000001", **kwargs
    )
    assert resumed["status"] == full["status"] == "complete"
    for k, v in load_file(str(tmp_path / "full/step-000002/model.safetensors")).items():
        assert torch.equal(v, load_file(str(tmp_path / "resumed/step-000002/model.safetensors"))[k])
