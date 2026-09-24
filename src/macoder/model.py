"""Decoder-only transformer: tokens → embeddings → residual blocks → next-token logits.

GQA shares key/value heads across query heads to reduce cache memory.
RoPE encodes position; feed-forward layers use squared ReLU or legacy SwiGLU.
Both sublayers add their output back to the residual stream.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import Config as Config  # Public compatibility import used by checkpoint loaders.


class KVCache:
    """Grow in blocks, avoiding concatenation of the entire history each token."""

    def __init__(self, max_length, block_size=256):
        self.max_length = max_length
        self.block_size = block_size
        self.offset = 0
        self.keys = self.values = None

    def update(self, keys, values):
        # The sequence axis is axis 2: [batch, heads, tokens, head width].
        end = self.offset + keys.shape[2]
        if end > self.max_length:
            raise ValueError("KV cache exceeds configured context")
        if self.keys is None or end > self.keys.shape[2]:
            # Round storage up to a block boundary so we do not reallocate every token.
            capacity = min(
                self.max_length, ((end + self.block_size - 1) // self.block_size) * self.block_size
            )
            shape = (*keys.shape[:2], capacity, keys.shape[3])
            new_k, new_v = mx.zeros(shape, keys.dtype), mx.zeros(shape, values.dtype)
            # When growing the buffer, keep all previous tokens in their original positions.
            if self.keys is not None:
                new_k[:, :, : self.offset] = self.keys[:, :, : self.offset]
                new_v[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = new_k, new_v
        # Append only the new tokens; old keys and values stay cached.
        self.keys[:, :, self.offset : end] = keys
        self.values[:, :, self.offset : end] = values
        self.offset = end
        # Hide unused buffer space so attention cannot read padding as real history.
        return self.keys[:, :, :end], self.values[:, :, :end]


class UnitRMSNorm(nn.Module):
    """Normalize magnitude without learning another scale vector."""

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, 1e-6)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c.heads, c.kv_heads, c.dim // c.heads
        self.qk_norm = c.architecture == "nano_core"
        # One matrix produces queries (what to look for), keys (what matches), and values (content).
        self.qkv = nn.Linear(c.dim, (c.heads + 2 * c.kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=c.rope_base)

    def __call__(self, x, cache=None):
        # Input shape: batch of sequences × tokens per sequence × model width.
        b, t, _ = x.shape
        q_end = self.heads * self.head_dim
        k_end = q_end + self.kv_heads * self.head_dim
        q, k, v = mx.split(self.qkv(x), [q_end, k_end], axis=-1)
        # Move the head axis before the token axis, as the attention kernel expects.
        q = q.reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Cached decoding continues positions from the previous token, not from zero.
        offset = 0 if cache is None else cache.offset
        # Rotate queries and keys by position so attention can distinguish token order.
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if self.qk_norm:
            # Bound query/key magnitude before dot products; nanochat uses a 1.2 scale on each.
            q = mx.fast.rms_norm(q, None, 1e-6) * 1.2
            k = mx.fast.rms_norm(k, None, 1e-6) * 1.2
        if cache is not None:
            k, v = cache.update(k, v)
        # Explicit offset mask also supports multi-token appends to a populated cache.
        mask = None
        # Several new tokens need a future-token mask; a single last token sees only its past.
        if t > 1:
            mask = (
                "causal"
                if offset == 0
                else mx.arange(k.shape[2])[None, :] <= (offset + mx.arange(t)[:, None])
            )
        # MLX computes attention; scaling keeps dot products controlled as head width grows.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        # Join the heads back together and mix their outputs into one vector per token.
        return self.out(y.transpose(0, 2, 1, 3).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nano_core = c.architecture == "nano_core"
        self.attn_norm = UnitRMSNorm() if self.nano_core else nn.RMSNorm(c.dim)
        self.ffn_norm = UnitRMSNorm() if self.nano_core else nn.RMSNorm(c.dim)
        self.attn = Attention(c)
        # Compute the gate and candidate features together in one matrix multiplication.
        if self.nano_core:
            self.up = nn.Linear(c.dim, c.hidden_dim, bias=False)
        else:
            self.gate_up = nn.Linear(c.dim, 2 * c.hidden_dim, bias=False)
        self.down = nn.Linear(c.hidden_dim, c.dim, bias=False)

    def __call__(self, x, cache=None):
        # Normalize before attention, then add its contribution to the existing representation.
        x = x + self.attn(self.attn_norm(x), cache)
        if self.nano_core:
            # Squared ReLU uses two projections instead of SwiGLU's three.
            return x + self.down(mx.square(nn.relu(self.up(self.ffn_norm(x)))))
        gate, up = mx.split(self.gate_up(self.ffn_norm(x)), 2, axis=-1)
        # The activated gate controls which features pass through; project back and add them.
        return x + self.down(nn.silu(gate) * up)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # A learned lookup table turns each integer token ID into a vector.
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [Block(config) for _ in range(config.layers)]
        self.norm = UnitRMSNorm() if config.architecture == "nano_core" else nn.RMSNorm(config.dim)
        if config.architecture == "nano_core":
            # Separate output weights cost memory but decouple input lookup from prediction.
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Small random initialization; scale residual projections with depth.
        weights = []
        for name, p in tree_flatten(self.parameters()):
            if p.ndim == 2:
                # Use smaller output weights in deep stacks to limit the initial residual magnitudes.
                scale = 0.02 / (
                    (2 * config.layers) ** 0.5
                    if name.endswith(("out.weight", "down.weight"))
                    else 1
                )
                weights.append((name, mx.random.normal(p.shape) * scale))
        self.load_weights(weights, strict=False)

    def __call__(self, tokens, cache=None, last_only=False):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("Expected a nonempty [batch, sequence] token array")
        if tokens.shape[1] > self.config.context:
            raise ValueError("Input exceeds configured context")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("One KV cache is required per layer")
        x = self.embedding(tokens)
        if self.config.architecture == "nano_core":
            x = mx.fast.rms_norm(x, None, 1e-6)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[i])
        # Generation needs only the last position; training needs predictions at every position.
        if last_only:
            x = x[:, -1:, :]
        # Tie output weights to embeddings, reducing parameter storage.
        if self.config.architecture == "nano_core":
            logits = self.lm_head(self.norm(x)).astype(mx.float32)
            return 15.0 * mx.tanh(logits / 15.0)
        return self.embedding.as_linear(self.norm(x))

    def make_cache(self):
        return [KVCache(self.config.context) for _ in self.layers]

    def parameter_count(self):
        return sum(p.size for _, p in tree_flatten(self.parameters()))


def loss_fn(model, x, y):
    # Reward high probability on the actual next token, averaged across this batch.
    return nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="mean")
