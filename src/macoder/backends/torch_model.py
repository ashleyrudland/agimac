"""CUDA/CPU implementation with the SAME parameter names and equations as MLX.

Keep this explicit: changes must pass forward, gradient and update parity tests.
No pretrained weights and no dependency on MLX inside the cloud container.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def rms(x, weight=None, eps=1e-6):
    # Accumulate squared magnitudes in FP32 even when the model uses BF16.
    y = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y if weight is None else y * weight


class Norm(nn.Module):
    def __init__(self, dim, learned):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if learned else None

    def forward(self, x):
        return rms(x, self.weight, 1e-5 if self.weight is not None else 1e-6)


def rope(x, base):
    # MLX traditional=False pairs the FIRST and SECOND halves, not adjacent entries.
    half = x.shape[-1] // 2
    angles = (
        torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None]
        * (base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half))[None, :]
    )
    c, s = angles.cos(), angles.sin()
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.width = c.dim // c.heads
        self.qkv = nn.Linear(c.dim, (c.heads + 2 * c.kv_heads) * self.width, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x):
        c = self.c
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(
            [c.dim, c.kv_heads * self.width, c.kv_heads * self.width], dim=-1
        )
        q = q.view(b, t, c.heads, self.width).transpose(1, 2)
        k = k.view(b, t, c.kv_heads, self.width).transpose(1, 2)
        v = v.view(b, t, c.kv_heads, self.width).transpose(1, 2)
        q, k = rope(q, c.rope_base), rope(k, c.rope_base)
        if c.architecture == "nano_core":
            q, k = rms(q) * 1.2, rms(k) * 1.2
        # PyTorch dispatches to fused CUDA attention when supported by the GPU.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c.dim))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nano = c.architecture == "nano_core"
        self.attn_norm = Norm(c.dim, not self.nano)
        self.ffn_norm = Norm(c.dim, not self.nano)
        self.attn = Attention(c)
        if self.nano:
            self.up = nn.Linear(c.dim, c.hidden_dim, bias=False)
        else:
            self.gate_up = nn.Linear(c.dim, 2 * c.hidden_dim, bias=False)
        self.down = nn.Linear(c.hidden_dim, c.dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        if self.nano:
            z = F.relu(self.up(self.ffn_norm(x))).square()
        else:
            gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
            z = F.silu(gate) * up
        return x + self.down(z)


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = Norm(c.dim, c.architecture != "nano_core")
        if c.architecture == "nano_core":
            self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        for name, p in self.named_parameters():
            if p.ndim == 2:
                nn.init.normal_(
                    p,
                    std=0.02
                    / (
                        math.sqrt(2 * c.layers)
                        if name.endswith(("out.weight", "down.weight"))
                        else 1
                    ),
                )

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("Expected nonempty tokens within configured context")
        x = self.embedding(tokens)
        if self.config.architecture == "nano_core":
            x = rms(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.config.architecture == "nano_core":
            return 15 * torch.tanh(self.lm_head(x).float() / 15)
        return F.linear(x, self.embedding.weight)


def loss_fn(model, x, y):
    return F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())


class MasterAdamW(torch.optim.Optimizer):
    """Match MLX's AdamW: FP32 master/moments, WITHOUT bias correction.

    PyTorch's stock AdamW enables correction, so substituting it silently changes
    the experiment. This version retains the exact optimizer equation.
    """

    def __init__(self, params, lr=3e-4):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(
                        master=p.detach().float().clone(),
                        m=torch.zeros_like(p, dtype=torch.float32),
                        v=torch.zeros_like(p, dtype=torch.float32),
                    )
                g = p.grad.float()
                st["m"].mul_(0.9).add_(g, alpha=0.1)
                st["v"].mul_(0.999).addcmul_(g, g, value=0.001)
                st["master"].copy_(
                    st["master"] * (1 - lr * 0.1) - lr * st["m"] / (st["v"].sqrt() + 1e-8)
                )
                p.copy_(st["master"])

    def load_state_dict(self, state):
        # Optimizer's default loader casts moments to parameter dtype; restore FP32.
        super().load_state_dict(state)
        for group, saved in zip(self.param_groups, state["param_groups"]):
            for p, key in zip(group["params"], saved["params"]):
                if key in state["state"]:
                    self.state[p] = {
                        k: v.to(device=p.device, dtype=torch.float32)
                        for k, v in state["state"][key].items()
                    }
