"""Independent backend checks catch RoPE ordering, masking, optimizer and layout drift."""

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

torch = pytest.importorskip("torch")
from macoder.model import Config, Model, loss_fn
from macoder.backends.torch_model import Model as TorchModel, loss_fn as torch_loss, MasterAdamW
from macoder.train import MasterAdamW as MLXAdamW


@pytest.mark.parametrize("architecture", ["legacy", "nano_core"])
def test_forward_gradient_update_parity(architecture):
    c = Config(
        vocab_size=32,
        dim=16,
        layers=2,
        heads=4,
        kv_heads=2,
        hidden_dim=32,
        context=16,
        architecture=architecture,
    )
    m = Model(c)
    t = TorchModel(c)
    weights = {k: torch.tensor(np.array(v)) for k, v in tree_flatten(m.parameters())}
    t.load_state_dict(weights, strict=True)
    x = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    y = (x + 1) % 32
    np.testing.assert_allclose(
        np.array(m(mx.array(x))), t(torch.tensor(x).long()).detach().numpy(), rtol=2e-4, atol=2e-5
    )
    loss, g = nn.value_and_grad(m, loss_fn)(m, mx.array(x), mx.array(y))
    tl = torch_loss(t, torch.tensor(x).long(), torch.tensor(y).long())
    tl.backward()
    assert float(loss) == pytest.approx(tl.item(), abs=1e-5)
    for k, v in tree_flatten(g):
        np.testing.assert_allclose(
            np.array(v), dict(t.named_parameters())[k].grad.numpy(), atol=2e-5, rtol=2e-3
        )
    mo = MLXAdamW(learning_rate=0.0003, weight_decay=0.1)
    to = MasterAdamW(t.parameters())
    mo.init(m.trainable_parameters())
    mo.update(m, g)
    to.step()
    mx.eval(m.parameters())
    for k, v in tree_flatten(m.parameters()):
        np.testing.assert_allclose(np.array(v), t.state_dict()[k].numpy(), atol=3e-5, rtol=2e-3)
    # Future tokens must not affect earlier logits.
    z = x.copy()
    z[:, -1] = 9
    np.testing.assert_allclose(
        t(torch.tensor(x).long()).detach().numpy()[:, :-1],
        t(torch.tensor(z).long()).detach().numpy()[:, :-1],
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cloud_resume_and_mlx_export(tmp_path, dtype):
    import json
    from tokenizers import Tokenizer, models
    from macoder.data import file_hash, TokenStream
    from macoder.backends.cloud_train import train
    from macoder.checkpoint import load_model

    data = tmp_path / "data"
    data.mkdir()
    tok = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{str(i): i for i in range(1, 32)}}, unk_token="[UNK]")
    )
    tok.save(str(data / "tokenizer.json"))
    for name in ["train", "valid"]:
        np.arange(256, dtype="<u4").__mod__(32).tofile(data / f"{name}.bin")
    (data / "manifest.json").write_text(
        json.dumps({"vocab_size": 32, "tokenizer_sha256": file_hash(data / "tokenizer.json")})
    )
    c = Config(
        vocab_size=32,
        dim=16,
        layers=1,
        heads=4,
        kv_heads=2,
        hidden_dim=32,
        context=16,
        architecture="nano_core",
    )
    cfg = tmp_path / "config.json"
    c.write(cfg)
    settings = dict(
        steps=4,
        batch_size=2,
        sequence=4,
        warmup=1,
        eval_every=2,
        eval_batches=1,
        dtype=dtype,
        device="cpu",
        compile_model=False,
    )
    train(cfg, data, tmp_path / "full", **settings)
    train(cfg, data, tmp_path / "part", max_steps=2, **settings)
    train(cfg, data, tmp_path / "resumed", resume=tmp_path / "part/step-000002", **settings)
    from safetensors.torch import load_file

    a = load_file(str(tmp_path / "full/step-000004/model.safetensors"))
    b = load_file(str(tmp_path / "resumed/step-000004/model.safetensors"))
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    mlx, _, _ = load_model(tmp_path / "full/step-000004")
    t = TorchModel(c).to(dtype=getattr(torch, dtype))
    t.load_state_dict(a)
    x = np.array([[1, 2, 3, 4]], dtype=np.int32)
    np.testing.assert_allclose(
        np.array(mlx(mx.array(x))),
        t(torch.tensor(x).long()).detach().numpy(),
        atol=3e-3 if dtype == "bfloat16" else 2e-5,
        rtol=2e-3,
    )
    s1 = TokenStream(data / "train.bin", 4, 2, 42)
    s2 = TokenStream(data / "train.bin", 4, 2, 42)
    for _ in range(3):
        for m, n in zip(s1.batch(), s2.numpy_batch()):
            np.testing.assert_array_equal(np.array(m), n)
    # A changed corpus must never silently continue the old optimizer trajectory.
    with (data / "train.bin").open("ab") as f:
        f.write(np.array([1], dtype="<u4").tobytes())
    with pytest.raises(ValueError, match="identical"):
        train(cfg, data, tmp_path / "invalid", resume=tmp_path / "part/step-000002", **settings)
