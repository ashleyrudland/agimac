"""Explicit, bounded Modal conversation/calculator SFT on the saved base model."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("agimac-conversation-sft")
volume = modal.Volume.from_name("agimac-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.2.6", "tokenizers==0.22.0", "safetensors==0.6.2")
    .env(
        {
            "TORCHINDUCTOR_CACHE_DIR": "/training/cache/sft-inductor",
            "TRITON_CACHE_DIR": "/training/cache/triton",
        }
    )
    .add_local_dir(ROOT / "src/macoder", remote_path="/root/macoder")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=4800,
    volumes={"/training": volume},
    max_containers=1,
    retries=0,
)
def run(expected_manifest, run_name, reference):
    import torch
    import numpy as np
    from macoder.backends.cloud_sft import train, masked_loss
    from macoder.backends.torch_model import Model
    from macoder.config import Config

    # Verify actual CUDA masked gradients against the tiny MLX reference first.
    m = Model(Config(**reference["config"])).cuda()
    m.load_state_dict({k: torch.tensor(v, device="cuda") for k, v in reference["weights"].items()})
    x = torch.tensor(reference["x"], device="cuda")
    y = torch.tensor(reference["y"], device="cuda")
    mask = torch.tensor(reference["mask"], device="cuda")
    value = masked_loss(m, x, y, mask)
    value.backward()
    np.testing.assert_allclose(value.item(), reference["loss"], rtol=2e-3, atol=3e-5)
    for k, p in m.named_parameters():
        np.testing.assert_allclose(
            p.grad.cpu().numpy(), reference["grads"][k], rtol=4e-3, atol=5e-5
        )
    print("CUDA/MLX masked loss and gradient parity passed", flush=True)
    del m, value
    torch.cuda.empty_cache()
    return train(
        "/training/runs/cloud-v2-145m-11b/step-335694",
        "/training/preparation/v2/data/sft-v1",
        Path("/training/runs") / run_name,
        expected_manifest,
        commit=volume.commit,
    )


@app.local_entrypoint()
def main(run_name: str = "cloud-v2-sft-v1"):
    import hashlib
    import json
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from macoder.config import Config
    from macoder.model import Model
    from macoder.sft import masked_loss

    mx.random.seed(42)
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
    model = Model(c)
    x = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = mx.array([[2, 3, 4, 5], [6, 7, 8, 9]])
    mask = mx.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    value, grads = nn.value_and_grad(model, masked_loss)(model, x, y, mask)
    mx.eval(value, grads)
    reference = dict(
        config=vars(c),
        weights={k: v.tolist() for k, v in tree_flatten(model.parameters())},
        x=x.tolist(),
        y=y.tolist(),
        mask=mask.tolist(),
        loss=value.item(),
        grads={k: v.tolist() for k, v in tree_flatten(grads)},
    )
    local_manifest = json.loads((ROOT / "data/sft-v1/manifest.json").read_text())
    remote_bytes = b"".join(volume.read_file("/preparation/v2/data/sft-v1/manifest.json"))
    if json.loads(remote_bytes) != local_manifest:
        raise ValueError("Cloud SFT manifest content differs")
    # Preparation serialized equivalent JSON differently; pin the verified remote bytes.
    expected = hashlib.sha256(remote_bytes).hexdigest()
    print(json.dumps(run.remote(expected, run_name, reference), indent=2))
