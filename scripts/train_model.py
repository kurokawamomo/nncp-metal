#!/usr/bin/env python3
"""
Train a byte-level Transformer for nncp-metal and save weights in NNCPW format.

Weight layout matches the C-side MTLBuffer layout in neural_engine.mm:
  embedding         [vocab_size, hidden_size]
  pos_embed         [max_seq_len, hidden_size]
  attn_q/k/v/out    each [num_layers, hidden_size, hidden_size]
  ffn_weights_1     [num_layers, hidden_size, ffn_size*2]
  ffn_weights_2     [num_layers, ffn_size, hidden_size]
  layer_norm_weights [num_layers, 2, hidden_size]  (gamma=[:,0,:], beta=[:,1,:])
  final_layer_norm  [2, hidden_size]
  output_projection [hidden_size, vocab_size]
"""

import argparse
import os
import struct
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def forward(self, x):  # x: [..., 2*ffn_size]
        gate, val = x.chunk(2, dim=-1)
        return val * F.gelu(gate)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True)

        # No bias, matching Metal kernels
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # GEGLU FFN: W1 projects to 2*ffn_size, gate + value halves
        self.ffn_w1 = nn.Linear(hidden_size, ffn_size * 2, bias=False)
        self.ffn_w2 = nn.Linear(ffn_size, hidden_size, bias=False)
        self.geglu = GEGLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm MHA
        h = self.norm1(x)
        B, T, C = h.shape
        q = self.q(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale  # [B, H, T, T]
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out(out)

        # Pre-norm GEGLU FFN
        h = self.norm2(x)
        x = x + self.ffn_w2(self.geglu(self.ffn_w1(h)))
        return x


class ByteTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_size: int = 1024,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_size = ffn_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_size, num_heads, ffn_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.out_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        # Sinusoidal positional embeddings
        pos = torch.arange(self.max_seq_len).unsqueeze(1).float()
        dim = torch.arange(0, self.hidden_size, 2).float()
        pe = torch.zeros(self.max_seq_len, self.hidden_size)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / self.hidden_size)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / self.hidden_size)))
        self.pos_embed.weight.data.copy_(pe)

        for layer in self.layers:
            for lin in (layer.q, layer.k, layer.v, layer.out, layer.ffn_w1, layer.ffn_w2):
                nn.init.xavier_uniform_(lin.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        for layer in self.layers:
            h = layer(h, mask)
        return self.out_proj(self.final_norm(h))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ByteDataset(Dataset):
    def __init__(self, data: bytes, context_len: int):
        buf = bytearray(data)  # make writable copy
        self.data = torch.frombuffer(buf, dtype=torch.uint8).long()
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.context_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# NNCPW weight serialisation
# ---------------------------------------------------------------------------

def _np(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().float().numpy().tobytes()


def save_nncpw(model: ByteTransformer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    m = model
    L = m.num_layers
    H = m.hidden_size
    V = m.vocab_size
    ffn = m.ffn_size
    S = m.max_seq_len

    # Stack per-layer weights into [num_layers, ...] tensors
    attn_q = torch.stack([m.layers[i].q.weight for i in range(L)])    # [L, H, H]
    attn_k = torch.stack([m.layers[i].k.weight for i in range(L)])
    attn_v = torch.stack([m.layers[i].v.weight for i in range(L)])
    attn_o = torch.stack([m.layers[i].out.weight for i in range(L)])
    ffn1   = torch.stack([m.layers[i].ffn_w1.weight for i in range(L)])  # [L, ffn*2, H] → transpose→ [L, H, ffn*2]
    ffn2   = torch.stack([m.layers[i].ffn_w2.weight for i in range(L)])  # [L, H, ffn] → transpose→ [L, ffn, H]

    # nn.Linear stores weight as [out, in]; C side expects [in, out] (row-major matmul)
    # Transpose each: [L, out, in] → [L, in, out]
    attn_q = attn_q.transpose(1, 2)  # [L, H, H]
    attn_k = attn_k.transpose(1, 2)
    attn_v = attn_v.transpose(1, 2)
    attn_o = attn_o.transpose(1, 2)
    ffn1   = ffn1.transpose(1, 2)   # [L, H, ffn*2]
    ffn2   = ffn2.transpose(1, 2)   # [L, ffn, H]

    # layer_norm_weights: [L, 2, H]  (gamma=[:,0,:], beta=[:,1,:])
    ln_gamma = torch.stack([m.layers[i].norm1.weight for i in range(L)]).unsqueeze(1)  # [L,1,H]
    ln_beta  = torch.stack([m.layers[i].norm1.bias   for i in range(L)]).unsqueeze(1)
    ln = torch.cat([ln_gamma, ln_beta], dim=1)  # [L, 2, H]

    # final_layer_norm: [2, H]
    final_ln = torch.stack([m.final_norm.weight, m.final_norm.bias])  # [2, H]

    # output_projection: [H, V]  (Linear stores [V, H], transpose)
    out_proj = m.out_proj.weight.T  # [H, V]

    with open(path, "wb") as f:
        f.write(b"NNCPW")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<IIIIII", L, H, m.num_heads, ffn, V, S))  # config
        f.write(_np(m.embed.weight))    # [V, H]
        f.write(_np(m.pos_embed.weight))  # [S, H]
        f.write(_np(attn_q))            # [L, H, H]
        f.write(_np(attn_k))
        f.write(_np(attn_v))
        f.write(_np(attn_o))
        f.write(_np(ffn1))              # [L, H, ffn*2]
        f.write(_np(ffn2))              # [L, ffn, H]
        f.write(_np(ln))               # [L, 2, H]
        f.write(_np(final_ln))         # [2, H]
        f.write(_np(out_proj))         # [H, V]

    total = os.path.getsize(path)
    print(f"Saved weights to {path}  ({total / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    device_str = "cpu"
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    with open(args.data, "rb") as f:
        raw = f.read()
    print(f"Data size: {len(raw):,} bytes")

    dataset = ByteDataset(raw, args.context_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device_str == "cuda"),
    )

    model = ByteTransformer(
        vocab_size=256,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        ffn_size=1024,
        max_seq_len=args.context_len,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )

    total_steps  = args.epochs * len(loader)
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)                                     # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss  += loss.item()
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                bpc = avg / math.log(2)
                lr  = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}/{args.epochs}  "
                    f"step {batch_idx+1}/{len(loader)}  "
                    f"loss={avg:.4f}  bpc={bpc:.4f}  lr={lr:.2e}"
                )

        avg_loss = total_loss / max(1, len(loader))
        print(
            f"=== Epoch {epoch} done  avg_loss={avg_loss:.4f}  "
            f"bpc={avg_loss/math.log(2):.4f} ==="
        )

    output_path = os.path.expanduser(args.output)
    save_nncpw(model, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train byte-level Transformer for nncp-metal")
    parser.add_argument("--data",        required=True,  help="Path to training data file (read as raw bytes)")
    parser.add_argument("--epochs",      type=int,   default=5,   help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=32,  help="Batch size")
    parser.add_argument("--context_len", type=int,   default=64,  help="Context length (max_seq_len)")
    parser.add_argument("--lr",          type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument(
        "--output",
        default="~/.config/nncp/model.nncpw",
        help="Output weight file path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.expanduser("~/.config/nncp"), exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
