"""
Cortex-Behavioral: 1D-CNN + Multi-Head Self-Attention over API-call token IDs.

Input:  (batch, 100) int64 token IDs (from ApiTokenizer)
Output: (batch,) float in [0, 1] — single behavioral maliciousness score
        (binary malicious-vs-benign, matching the architecture contract —
        no ensembling / no multi-class softmax here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_LENGTH = 100
DEFAULT_SEED = 42


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, s, -1)
        out = self.dropout(self.out(out))
        return self.norm(out + residual)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x))))


class CortexBehavioralNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = SEQUENCE_LENGTH,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, embed_dim) * 0.02)

        self.conv_stack = nn.Sequential(
            ConvBlock(embed_dim, 128, kernel_size=3, dropout=dropout),
            ConvBlock(128, 256, kernel_size=5, dropout=dropout),
            ConvBlock(256, 128, kernel_size=7, dropout=dropout),
            ConvBlock(128, 128, kernel_size=3, dropout=dropout),
        )
        self.attention = MultiHeadSelfAttention(embed_dim=128, num_heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, 100) int64 -> logits (batch,)"""
        x = self.embedding(token_ids) + self.pos_embedding[:, : token_ids.size(1), :]
        x = x.permute(0, 2, 1)          # (B, embed_dim, seq)
        x = self.conv_stack(x)          # (B, 128, seq)
        x = x.permute(0, 2, 1)          # (B, seq, 128)
        x = self.attention(x)
        x = x.mean(dim=1)               # global average pool -> (B, 128)
        logits = self.classifier(x).squeeze(-1)  # (B,)
        return logits

    def predict_proba(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(token_ids))


@dataclass
class TrainConfig:
    vocab_size: int
    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 150
    batch_size: int = 128
    patience: int = 15
    grad_clip: float = 1.0
    seed: int = DEFAULT_SEED


def build_model(cfg: TrainConfig, device: Optional[str] = None) -> CortexBehavioralNet:
    torch.manual_seed(cfg.seed)
    model = CortexBehavioralNet(
        vocab_size=cfg.vocab_size, sequence_length=SEQUENCE_LENGTH,
        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, dropout=cfg.dropout,
    )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev)
