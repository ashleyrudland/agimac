"""Tiny MLX reference fixture used to validate the actual CUDA container."""


def fixture():
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from macoder.model import Config, Model, loss_fn

    c = Config(
        vocab_size=32,
        dim=16,
        layers=2,
        heads=4,
        kv_heads=2,
        hidden_dim=32,
        context=16,
        architecture="nano_core",
    )
    mx.random.seed(42)
    m = Model(c)
    x = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    y = (x + 1) % 32
    loss, g = nn.value_and_grad(m, loss_fn)(m, x, y)
    return dict(
        config=vars(c),
        weights={k: np.array(v).tolist() for k, v in tree_flatten(m.parameters())},
        x=np.array(x).tolist(),
        y=np.array(y).tolist(),
        logits=np.array(m(x)).tolist(),
        loss=float(loss),
        grads={k: np.array(v).tolist() for k, v in tree_flatten(g)},
    )
