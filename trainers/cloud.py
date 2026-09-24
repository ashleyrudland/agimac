"""Explicit Modal launcher. Nothing is uploaded or run by importing this file."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("agimac-training")
volume = modal.Volume.from_name("agimac-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.2.6", "tokenizers==0.22.0", "safetensors==0.6.2")
    .env(
        {
            "TORCHINDUCTOR_CACHE_DIR": "/training/cache/inductor",
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
    timeout=3600,
    volumes={"/training": volume},
    max_containers=1,
)
def run(config, identity, run_name, settings, reference, resume=None):
    import json
    from macoder.training_common import dataset_identity, identity_key
    from macoder.backends.cloud_train import train

    # Fail before the real model allocates memory if the CUDA implementation drifts.
    import numpy as np
    import torch
    from macoder.config import Config
    from macoder.backends.torch_model import Model, loss_fn

    torch.backends.cuda.matmul.allow_tf32 = False
    m = Model(Config(**reference["config"])).cuda()
    m.load_state_dict({k: torch.tensor(v, device="cuda") for k, v in reference["weights"].items()})
    x = torch.tensor(reference["x"], device="cuda")
    y = torch.tensor(reference["y"], device="cuda")
    logits = m(x)
    np.testing.assert_allclose(
        logits.detach().cpu().numpy(), reference["logits"], rtol=2e-3, atol=3e-5
    )
    loss = loss_fn(m, x, y)
    loss.backward()
    for k, p in m.named_parameters():
        np.testing.assert_allclose(
            p.grad.cpu().numpy(), reference["grads"][k], rtol=3e-3, atol=3e-5
        )
    print("CUDA/MLX forward and gradient parity passed", flush=True)
    del m, logits, loss
    torch.cuda.empty_cache()
    data = Path("/training/corpora") / identity_key(identity)
    if dataset_identity(data) != identity:
        raise ValueError("Cloud corpus bytes differ from the local corpus")
    cfg = Path("/tmp/model-config.json")
    cfg.write_text(json.dumps(config))
    return train(
        cfg,
        data,
        Path("/training/runs") / run_name,
        resume=Path("/training/runs") / resume if resume else None,
        commit=volume.commit,
        **settings,
    )


@app.function(image=image, cpu=2, memory=4096, timeout=3600, volumes={"/training": volume})
def restore_corpus(identity):
    """Decompress on CPU, then verify every original byte before publishing."""
    import shutil
    import uuid
    from macoder.transfer import unpack_tokens
    from macoder.training_common import dataset_identity, identity_key

    root = Path("/training")
    target = root / "corpora" / identity_key(identity)
    transfer = root / "transfer" / identity_key(identity)
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if dataset_identity(target) != identity:
            raise ValueError("Existing cloud corpus differs; refusing overwrite")
        return
    pending = root / (".corpus-" + uuid.uuid4().hex)
    pending.mkdir()
    for name in identity:
        if name.endswith(".bin"):
            unpack_tokens(transfer / (name + ".gz"), pending / name)
        else:
            shutil.copyfile(transfer / name, pending / name)
    if dataset_identity(pending) != identity:
        raise ValueError("Decompressed corpus failed original SHA-256 checks")
    pending.rename(target)
    volume.commit()


@app.local_entrypoint()
def main(
    run_name: str,
    data: str = "data/general-v2/prepared",
    recipe: str = "recipes/v2.json",
    config: str = "configs/nano-core-145m.json",
    upload: bool = False,
    compact_upload: bool = False,
    max_steps: int = 100,
    batch_size: int = 8,
    max_seconds: int = 1200,
    resume: str = "",
):
    """Defaults are a bounded benchmark, NOT the 11B-token run. Upload once."""
    import json
    from macoder.training_common import dataset_identity, identity_key

    if not run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_name
    ):
        raise ValueError("Use a simple unique run name")
    if resume and (Path(resume).is_absolute() or ".." in Path(resume).parts):
        raise ValueError("Resume must be relative to the cloud runs directory")
    if not 0 < max_seconds <= 3000 or max_steps < 1 or batch_size < 1:
        raise ValueError("Invalid invocation budget")
    r = json.loads(Path(recipe).read_text())
    c = json.loads(Path(config).read_text())
    print("Hashing local prepared corpus...", flush=True)
    identity = dataset_identity(data)
    key = identity_key(identity)
    if upload and compact_upload:
        raise ValueError("Choose --upload OR --compact-upload")
    if compact_upload:
        import tempfile
        from macoder.transfer import pack_tokens

        print("Compressing tokens losslessly before upload...", flush=True)
        with tempfile.TemporaryDirectory(prefix="agimac-transfer-") as temp:
            with volume.batch_upload() as batch:
                for name in identity:
                    if name.endswith(".bin"):
                        packed = Path(temp) / (name + ".gz")
                        pack_tokens(Path(data) / name, packed)
                        print(
                            f"{name}: {packed.stat().st_size / 1e9:.2f} GB to transfer", flush=True
                        )
                        batch.put_file(packed, "/transfer/" + key + "/" + packed.name)
                    else:
                        batch.put_file(Path(data) / name, "/transfer/" + key + "/" + name)
            restore_corpus.remote(identity)
    elif upload:
        print("Uploading identical packed token files to Modal Volume...", flush=True)
        with volume.batch_upload() as batch:
            for name in identity:
                batch.put_file(Path(data) / name, "/corpora/" + key + "/" + name)
    print("Corpus ready; launching bounded H100 run...", flush=True)
    sequence = r["sequence"]
    settings = dict(
        steps=(r["training_tokens"] + batch_size * sequence - 1) // (batch_size * sequence),
        batch_size=batch_size,
        sequence=sequence,
        lr=r["learning_rate"],
        warmup=r["warmup_steps"],
        eval_every=r["checkpoint_every_steps"],
        eval_batches=4 if max_steps <= 1000 else 32,
        max_steps=max_steps,
        max_seconds=max_seconds,
    )
    from trainers.parity import fixture

    result = run.remote(c, identity, run_name, settings, fixture(), resume or None)
    Path("reports").mkdir(exist_ok=True)
    report = {
        "run": run_name,
        "gpu": "H100",
        "data": data,
        "settings": settings,
        "dataset_identity": identity,
        "result": result,
    }
    (Path("reports") / f"modal-{run_name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
