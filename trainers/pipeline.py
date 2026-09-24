"""Authorized full cloud run: CPU preparation, strict corpus checks, then one H100."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("agimac-full-training")
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

prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3", "tokenizers==0.23.2", "pyarrow==25.0.1", "huggingface-hub==1.32.0")
    .env({"TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src/macoder", remote_path="/root/macoder")
    .add_local_file(ROOT / "experiments/prepare_general.py", remote_path="/root/prepare_general.py")
    .add_local_file(ROOT / "experiments/prepare_sft.py", remote_path="/root/prepare_sft.py")
    .add_local_file(
        ROOT / "recipes/reference/tokenizer.json", remote_path="/root/reference-tokenizer.json"
    )
    .add_local_dir(ROOT / "recipes/cloud", remote_path="/root/recipe")
)


@app.function(
    image=prepare_image,
    cpu=8,
    memory=32768,
    timeout=14400,
    volumes={"/training": volume},
    max_containers=1,
)
def prepare():
    import gzip
    import json
    import os
    import shutil
    import subprocess
    import sys
    from macoder.training_common import dataset_identity, identity_key
    from macoder.data import file_hash

    expected = json.loads(Path("/root/recipe/expected.json").read_text())
    if (
        file_hash("/root/prepare_general.py") != expected["prepare_general_sha256"]
        or file_hash("/root/prepare_sft.py") != expected["prepare_sft_sha256"]
    ):
        raise ValueError("Preparation code differs from frozen recipe")
    work = Path("/training/preparation/v2")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    for folder in [
        "data/general-v2",
        "data/general-v1",
        "recipes/reference",
        "runs/general-v1/step-048829",
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    for packed, target in [
        ("pretrain-filter.json.gz", "data/general-v2/benchmark-filter.json"),
        ("sft-filter.json.gz", "data/general-v1/benchmark-filter.json"),
    ]:
        with gzip.open("/root/recipe/" + packed, "rb") as inp, open(target, "wb") as out:
            shutil.copyfileobj(inp, out)
    # The legacy SFT preparer reads only the tokenizer from this path, never weights.
    for target in [
        "recipes/reference/tokenizer.json",
        "runs/general-v1/step-048829/tokenizer.json",
    ]:
        shutil.copyfile("/root/reference-tokenizer.json", target)
    for label, command in [
        (
            "pretrain",
            [
                sys.executable,
                "/root/prepare_general.py",
                "--root",
                "data/general-v2",
                "--tokens",
                "2000000000",
                "--expanded",
            ],
        ),
        ("conversations", [sys.executable, "/root/prepare_sft.py"]),
    ]:
        print("Preparing " + label + " on CPU", flush=True)
        with open(label + "-preparation.log", "a") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
            if process.wait():
                volume.commit()
                raise RuntimeError(label + " preparation failed; see persisted log")
        volume.commit()
    prepared = work / "data/general-v2/prepared"
    actual = dataset_identity(prepared)
    if actual != expected["pretrain_identity"]:
        volume.commit()
        raise ValueError(
            "Cloud token files/tokenizer/manifest do not match original Mac corpus; training blocked"
        )
    sft = work / "data/sft-v1"
    manifest = json.loads((sft / "manifest.json").read_text())
    if manifest != expected["sft_manifest"]:
        volume.commit()
        raise ValueError("Conversation manifest differs from the Mac recipe")
    for name, digest in manifest["array_hashes"].items():
        if file_hash(sft / name) != digest:
            raise ValueError("Conversation array differs: " + name)
    target = Path("/training/corpora") / identity_key(actual)
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        shutil.copytree(prepared, target)
    if dataset_identity(target) != actual:
        raise ValueError("Published corpus hash mismatch")
    volume.commit()
    return {
        "identity": actual,
        "sft": str(sft),
        "tokens": 2_000_000_000,
        "byte_identical_to_mac": True,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=86400,
    volumes={"/training": volume},
    max_containers=1,
)
def full_train(config, identity, run_name, settings, reference):
    # Same already-tested GPU implementation; longer timeout is explicit here only.
    import json
    from macoder.training_common import dataset_identity, identity_key
    from macoder.backends.cloud_train import train

    data = Path("/training/corpora") / identity_key(identity)
    if dataset_identity(data) != identity:
        raise ValueError("Corpus changed before GPU training")
    cfg = Path("/tmp/config.json")
    cfg.write_text(json.dumps(config))
    return train(cfg, data, Path("/training/runs") / run_name, commit=volume.commit, **settings)


@app.function(
    image=image,
    cpu=0.125,
    memory=512,
    timeout=86400,
    volumes={"/training": volume},
    max_containers=1,
)
def orchestrate(run_name, config, settings, reference):
    import datetime
    import json

    root = Path("/training/pipelines") / run_name
    root.mkdir(parents=True, exist_ok=False)

    def status(stage, **details):
        state = {
            "stage": stage,
            "run": run_name,
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config": config,
            "settings": settings,
            **details,
        }
        tmp = root / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(root / "status.json")
        volume.commit()
        print(json.dumps(state), flush=True)

    try:
        status("preparing_data")
        data = prepare.remote()
        volume.reload()
        status("pretraining", data=data)
        result = full_train.remote(config, data["identity"], run_name, settings, reference)
        volume.reload()
        stage = (
            "pretraining_complete"
            if result["last"]["step"] == settings["steps"]
            else "stopped_at_time_budget"
        )
        status(
            stage,
            result=result,
            data=data,
            next_phase="Conversation data prepared; conversation training/evaluation are not automatically launched by this pretraining runner",
        )
        return result
    except Exception as exc:
        status("failed", error=str(exc))
        raise


@app.local_entrypoint()
def launch(run_name: str = "cloud-v2-145m-11b"):
    import json
    import math
    from trainers.parity import fixture

    if any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_name
    ):
        raise ValueError("Use a simple unique run name")
    c = json.loads((ROOT / "configs/nano-core-145m.json").read_text())
    # Batch 64 was nearly as fast as 128, with much lower memory use.
    settings = dict(
        steps=math.ceil(11_000_000_000 / (64 * 512)),
        batch_size=64,
        sequence=512,
        seed=42,
        lr=0.0003,
        warmup=2000,
        eval_every=10000,
        eval_batches=32,
        max_seconds=20 * 3600,
        keep_last=3,
    )
    print(orchestrate.remote(run_name, c, settings, fixture()))
