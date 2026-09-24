"""Regenerate the README architecture diagram (CPU-only matplotlib)."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
fig.patch.set_facecolor("#0e1525")
ax.set_facecolor("#0e1525")
ax.set(xlim=(0, 14), ylim=(0, 8))
ax.axis("off")
ax.text(0.5, 7.45, "agimac", fontsize=28, weight="bold", color="#7dd3fc")
ax.text(
    0.5, 6.98, "A small language model, MLX on Mac · PyTorch on Modal", fontsize=15, color="white"
)


def box(x, y, w, h, title, body, color="#1e334c"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color,
            edgecolor="#49617c",
        )
    )
    ax.text(x + 0.17, y + h - 0.3, title, fontsize=12, weight="bold", color="white", va="top")
    ax.text(x + 0.17, y + h - 0.68, body, fontsize=10.5, color="#cbd5e1", va="top", linespacing=1.6)


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#7dd3fc", lw=2)
    )


box(
    0.5,
    4.55,
    2.6,
    1.75,
    "1  Text → tokens",
    "Byte-level BPE\n16,392 vocabulary entries\nRole + tool markers",
)
box(3.6, 4.55, 2.55, 1.75, "2  Embedding", "Each token becomes\na vector of 896 numbers")
box(
    6.7,
    3.7,
    3.6,
    2.6,
    "3  Transformer × 14",
    "RMSNorm → causal attention\n+ residual connection\nRMSNorm → squared ReLU\n+ residual connection",
    "#223e54",
)
box(
    10.85,
    4.55,
    2.6,
    1.75,
    "4  Next token",
    "Final RMSNorm\nSeparate output weights\nChoose a token → repeat",
)
arrow(3.2, 5.4, 3.5, 5.4)
arrow(6.25, 5.4, 6.6, 5.4)
arrow(10.4, 5.4, 10.75, 5.4)
ax.text(6.85, 3.25, "14 query heads · 2 KV heads · RoPE positions", fontsize=10, color="#7dd3fc")
box(
    0.5,
    0.7,
    4.0,
    2.0,
    "Learn from examples",
    "Documents → next-token training\nConversations → assistant-only loss\nGradients + AdamW update weights",
)
box(
    4.95,
    0.7,
    4.0,
    2.0,
    "Remember during generation",
    "KV cache reuses past attention states\n2,048-token configured context\nConversation training: up to 1,024 tokens",
)
box(
    9.4,
    0.7,
    4.0,
    2.0,
    "Call a calculator",
    "Model writes tagged JSON\nRuntime validates + computes\nModel reads result and answers",
)
ax.text(
    0.5,
    0.2,
    "145M parameters  •  Fixed decoder-only transformer  •  Inference does not update weights",
    fontsize=11,
    color="#94a3b8",
)
fig.tight_layout(pad=0.3)
dest = Path(__file__).resolve().parents[1] / "assets/architecture.png"
fig.savefig(dest, facecolor=fig.get_facecolor())
print(dest)
