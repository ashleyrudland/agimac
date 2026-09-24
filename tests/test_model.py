import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from macoder.model import Config, Model, loss_fn
from macoder.train import StableAdamW


def tiny():
    mx.random.seed(7)
    return Model(
        Config(vocab_size=320, dim=64, layers=2, heads=4, kv_heads=2, hidden_dim=128, context=32)
    )


def test_causal_future_does_not_change_past():
    m = tiny()
    a, b = m(mx.array([[1, 2, 3, 4]])), m(mx.array([[1, 2, 8, 9]]))
    np.testing.assert_allclose(np.array(a[:, :2]), np.array(b[:, :2]), atol=1e-6)


@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1], [2, 3]])
def test_cache_matches_full_prefix(chunks):
    m = tiny()
    ids = mx.array([[1, 2, 3, 4, 5]])
    expected = m(ids)
    caches = m.make_cache()
    for cache in caches:
        cache.block_size = 2  # Exercise allocation growth as well as cached attention.
    out, start = [], 0
    for chunk in chunks:
        out.append(m(ids[:, start : start + chunk], caches))
        mx.eval(out[-1])
        start += chunk
    np.testing.assert_allclose(
        np.array(mx.concatenate(out, axis=1)), np.array(expected), atol=2e-5, rtol=2e-5
    )


def test_loss_learns_and_bfloat16_keeps_fp32_moments():
    m = tiny()
    m.set_dtype(mx.bfloat16)
    opt = StableAdamW(learning_rate=0.01)
    vg = nn.value_and_grad(m, loss_fn)
    x, y = mx.array([[1, 2, 3, 4]]), mx.array([[2, 3, 4, 5]])
    before = loss_fn(m, x, y).item()
    for _ in range(15):
        _, grads = vg(m, x, y)
        opt.update(m, grads)
        mx.eval(m.parameters(), opt.state)
    assert loss_fn(m, x, y).item() < before * 0.4
    assert m.embedding.weight.dtype == mx.bfloat16
    assert opt.state["embedding"]["weight"]["m"].dtype == mx.float32


def test_context_guard():
    m = tiny()
    with pytest.raises(ValueError, match="context"):
        m(mx.ones((1, 33), dtype=mx.int32))
    caches = m.make_cache()
    m(mx.ones((1, 32), dtype=mx.int32), caches)
    with pytest.raises(ValueError, match="context"):
        m(mx.ones((1, 1), dtype=mx.int32), caches)


@pytest.mark.parametrize("architecture", ["legacy", "nano_core"])
def test_variant_cache_causality_and_compiled_update(architecture):
    from macoder.train import make_compiled_step
    from mlx.utils import tree_flatten

    config = Config(
        vocab_size=320,
        dim=64,
        layers=2,
        heads=4,
        kv_heads=2,
        hidden_dim=256,
        context=32,
        architecture=architecture,
    )
    mx.random.seed(19)
    m = Model(config)
    x = mx.array([[1, 2, 3, 4]])
    y = mx.array([[2, 3, 4, 5]])
    expected = m(x)
    cache = m.make_cache()
    actual = mx.concatenate([m(x[:, :2], cache), m(x[:, 2:], cache)], axis=1)
    np.testing.assert_allclose(np.array(expected), np.array(actual), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        np.array(expected[:, :2]),
        np.array(m(mx.array([[1, 2, 9, 9]]))[:, :2]),
        atol=3e-5,
        rtol=3e-5,
    )
    mx.random.seed(19)
    reference = Model(config)
    import mlx.optimizers as optim

    opt = StableAdamW(learning_rate=0.001, weight_decay=0.1)
    refopt = StableAdamW(learning_rate=0.001, weight_decay=0.1)
    compiled = make_compiled_step(m, opt)
    vg = nn.value_and_grad(reference, loss_fn)
    for _ in range(2):
        loss, norm = compiled(x, y, mx.array(0.001))
        mx.eval(m.parameters(), opt.state, loss, norm)
        _, grads = vg(reference, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        refopt.update(reference, grads)
        mx.eval(reference.parameters(), refopt.state)
    for (name, weight), (_, ref) in zip(
        tree_flatten(m.parameters()), tree_flatten(reference.parameters())
    ):
        np.testing.assert_allclose(
            np.array(weight), np.array(ref), atol=3e-6, rtol=3e-5, err_msg=name
        )


def test_master_weights_accumulate_sub_bf16_updates():
    from macoder.train import MasterAdamW

    opt = MasterAdamW(learning_rate=1e-5, weight_decay=0.0)
    parameter = mx.array([1.0], dtype=mx.bfloat16)
    state = {}
    opt.init_single(parameter, state)
    for _ in range(10):
        parameter = opt.apply_single(mx.array([1.0]), parameter, state)
    mx.eval(parameter, state)
    assert float(state["master"][0]) < 0.99995
    assert parameter.dtype == mx.bfloat16
    assert state["master"].dtype == state["m"].dtype == mx.float32
