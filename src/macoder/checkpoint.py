import json
from pathlib import Path
import shutil

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from .model import Config, Model


def save_model(model, path, tokenizer_path, metadata=None):
    path = Path(path)
    # Refuse overwrites; the caller controls staging and atomic publication.
    path.mkdir(parents=True, exist_ok=False)
    model.config.write(path / "config.json")
    model.save_weights(str(path / "model.safetensors"))
    shutil.copyfile(tokenizer_path, path / "tokenizer.json")
    (path / "metadata.json").write_text(json.dumps(metadata or {}, indent=2) + "\n")


def load_model(path):
    """Rebuild the architecture from config, then load weights and matching BPE.

    Quantization metadata determines layer layout before weights are loaded.
    Optimizer state is separate and only used when resuming training.
    """
    path = Path(path)
    model = Model(Config.read(path / "config.json"))
    meta = json.loads((path / "metadata.json").read_text())
    model.set_dtype(getattr(mx, meta.get("dtype", "float32")))
    if "quantization" in meta:
        nn.quantize(model, **meta["quantization"])
    # Weights alone do not encode architecture, so construct the correct layers first.
    model.load_weights(str(path / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    # Reject an incompatible tokenizer before its token IDs can reach the embedding table.
    if tok.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    return model, tok, meta


def quantize(source, output, bits=4):
    model, _, meta = load_model(source)
    if "quantization" in meta:
        raise ValueError("Source is already quantized")
    q = {"group_size": 64, "bits": bits}
    model.set_dtype(mx.float16)
    nn.quantize(model, **q)
    mx.eval(model.parameters())
    save_model(
        model,
        output,
        Path(source) / "tokenizer.json",
        {**meta, "dtype": "float16", "quantization": q},
    )
