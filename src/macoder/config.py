"""Backend-independent architecture specification shared by MLX and CUDA."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    # Number of distinct token IDs; preparation replaces this with the actual BPE size.
    vocab_size: int = 16384
    # Width of the vector carried through every transformer layer.
    dim: int = 640
    layers: int = 12
    heads: int = 10
    kv_heads: int = 2
    # The feed-forward network temporarily expands each token to this width.
    hidden_dim: int = 1792
    # Maximum accepted sequence length; this setting alone does not teach long context.
    context: int = 2048
    rope_base: float = 10000.0
    # Legacy is the existing checkpoint architecture; nano_core is a fresh-training experiment.
    architecture: str = "legacy"

    def __post_init__(self):
        if self.architecture not in ("legacy", "nano_core"):
            raise ValueError("Unknown architecture")
        for name in ("vocab_size", "dim", "layers", "heads", "kv_heads", "hidden_dim", "context"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        # Each query head must have the same width, and share K/V heads evenly.
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("dim must divide into heads; heads must divide into kv_heads groups")
        # RoPE rotates pairs of coordinates, so each head needs an even width.
        if (self.dim // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def write(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
