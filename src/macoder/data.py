"""Local JSONL corpus -> train-only byte BPE -> memory-mapped token streams."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
CHAT_SPECIALS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end_turn|>",
    "<|tool_call|>",
    "<|end_call|>",
    "<|tool_result|>",
]


def documents(path):
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_no}: expected nonempty 'text'")
            yield text


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(source, output, vocab_size=16384, valid_fraction=0.05, seed=42, fim_rate=0.5):
    if vocab_size < 260 or not 0 < valid_fraction < 1 or not 0 <= fim_rate <= 1:
        raise ValueError("Need vocab >=260, 0<valid_fraction<1, and 0<=fim_rate<=1")
    out = Path(output)
    if out.exists():
        raise ValueError(f"Output already exists: {out}; use a new directory")
    # Stage unique documents on disk, splitting before tokenizer fitting or FIM.
    # Exact deduplication only: repository-level splits should be prepared upstream.
    out.mkdir(parents=True)
    seen, counts = set(), {"train": 0, "valid": 0}
    staged = {s: out / f"{s}.jsonl" for s in counts}
    handles = {s: p.open("w") for s, p in staged.items()}
    try:
        for text in documents(source):
            h = digest(text)
            if h in seen:
                continue
            seen.add(h)
            split_hash = hashlib.sha256(f"{seed}:{h}".encode()).hexdigest()
            split = "valid" if int(split_hash[:16], 16) / 2**64 < valid_fraction else "train"
            handles[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in handles.values():
            f.close()
    if min(counts.values()) == 0:
        raise ValueError("Empty train/validation split; supply more unique documents")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        documents(staged["train"]),
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tok.save(str(out / "tokenizer.json"))
    eos, prefix, suffix, middle = [tok.token_to_id(s) for s in SPECIALS]
    rng = np.random.default_rng(seed)
    sizes = {}
    for split, path in staged.items():
        n = 0
        with (out / f"{split}.bin").open("wb") as f:
            for text in documents(path):
                ids = tok.encode(text).ids
                if split == "train" and len(ids) >= 3 and rng.random() < fim_rate:
                    a, b = sorted(rng.choice(len(ids) + 1, size=2, replace=False).tolist())
                    ids = [prefix] + ids[:a] + [suffix] + ids[b:] + [middle] + ids[a:b]
                ids.append(eos)
                np.asarray(ids, dtype="<u4").tofile(f)
                n += len(ids)
        sizes[split] = n
    meta = {
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "seed": seed,
        "documents": counts,
        "tokens": sizes,
        "dtype": "<u4",
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": file_hash(out / "tokenizer.json"),
        "fim_rate": fim_rate,
        "valid_fraction": valid_fraction,
        "deduplication": "exact text SHA256; no near-duplicate or benchmark decontamination",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TokenStream:
    def __init__(self, path, sequence, batch_size, seed=42):
        if sequence < 1 or batch_size < 1:
            raise ValueError("sequence and batch_size must be positive")
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        self.sequence, self.batch_size = sequence, batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) < sequence + 1:
            raise ValueError(f"{path} needs at least {sequence + 1} tokens")

    def numpy_batch(self):
        """Both backends sample precisely the same windows for a given RNG state."""
        starts = self.rng.integers(0, len(self.tokens) - self.sequence, size=self.batch_size)
        rows = np.stack([self.tokens[i : i + self.sequence + 1] for i in starts])
        return rows[:, :-1].copy(), rows[:, 1:].copy()

    def batch(self):
        import mlx.core as mx

        x, y = self.numpy_batch()
        return mx.array(x), mx.array(y)
