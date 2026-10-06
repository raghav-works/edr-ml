"""
Cortex-Emulation: 1D-CNN + Multi-Head Self-Attention over API-call token
IDs, adapted from Cortex-Behavioral's architecture (models/behavioral_cnn.py)
-- same shape (embedding + learned positional embedding -> conv stack ->
self-attention -> global average pool -> linear head), not a literal
weight-sharing reuse, and not the reference architecture's GRU/opcode
design (ruled out: the real data has no opcode or memory-access-pattern
signal -- see data/download_emulation.py's module docstring).

Capacity is NOT copied unchanged from Behavioral, per instructions to check
rather than assume it transfers. What's the same and what's different, and
why:

- Conv stack channel widths (128->256->128->128) and kernel sizes (3,5,7,3)
  are kept as the starting point, not shrunk pre-emptively: Behavioral's
  own real train set was itself only ~7,340 examples (Mal-API-2019 +
  MalbehavD-V1 + Carpenter, deduplicated) -- the same order of magnitude as
  Cortex-Emulation's 6,070, not a wildly larger dataset this capacity was
  "proven" on. Reusing the same core capacity is a reasoned starting point
  given that comparison, not an unchecked default.
- embed_dim defaults to 64 here, NOT scripts/train_behavioral.py's CLI
  default of 128 -- Cortex-Emulation's real train vocabulary is 3,152
  distinct API names (data/tokenizer/emulation_tokenizer.py), roughly 8x
  Behavioral's ~300-400, while having a comparable or smaller number of
  training examples. A larger embed_dim multiplies the embedding table by
  vocab_size for no benefit if there isn't enough data to train the extra
  dimensions on most tokens (many of the 3,152 appear only a handful of
  times in only 6,070 sequences) -- start smaller, not larger, given a
  bigger vocabulary and no more data.
- dropout defaults to 0.4 here, NOT Behavioral's 0.3 -- one concrete,
  proactive adjustment (not the full response) given the larger vocabulary
  increases the model's opportunity to memorize rare-token patterns
  specific to individual training sequences rather than learning
  generalizable structure.
- SEQUENCE_LENGTH is 500, not Behavioral's 100 (5x longer) -- this does not
  by itself add trainable capacity (self-attention's parameter count is
  independent of sequence length; only compute/memory scale with it,
  roughly 25x more attention computation per sample from the L^2 term, not
  a concern at this dataset size) except for the learned positional
  embedding table, which grows from (1, 100, embed_dim) to (1, 500,
  embed_dim) -- 5x more positional parameters, accepted as necessary since
  position information over up to 500 real steps needs 500 real slots.

If training shows train/val loss divergence (the classic overfitting
signature) despite the above, the next concrete step -- not applied
pre-emptively, since it isn't justified without that evidence -- is to
drop embed_dim to 32 and/or remove one conv block (128->256->128 instead
of 128->256->128->128), in that order, before touching anything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_LENGTH = 500
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """key_padding_mask: (batch, seq) bool, True at <PAD> positions.

        Masked-out keys get -inf pre-softmax so no query attends to them.
        Callers must guarantee every row has at least one unmasked key -- an
        all-masked row would produce an all -inf softmax row and NaN.
        """
        residual = x
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, s, -1)
        out = self.dropout(self.out(out))
        return self.norm(out + residual)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x))))


class CortexEmulationNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = SEQUENCE_LENGTH,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.4,
        pad_idx: int = 0,
        use_padding_mask: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        # True: <PAD> positions are excluded from attention and pooling
        # (commit b876d26). False: the forward pass as it was before b876d26
        # (unmasked attention, plain mean over all 500 slots) -- required for
        # checkpoints trained with that code. Both paths have identical
        # parameters, so a state_dict loads into either; only the checkpoint's
        # sidecar says which is right (same defect as docs/CODE_REVIEW.md F2).
        self.use_padding_mask = use_padding_mask
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
        """token_ids: (batch, 500) int64 -> logits (batch,)

        <PAD> positions (id == pad_idx) are excluded from both the
        attention step and the pooling step -- otherwise short traces get
        their pooled representation diluted by positional-embedding-only
        noise at every unused slot, the same dilution fixed for Behavioral
        in models/behavioral_cnn.py (commit 76c3534).
        """
        x = self.embedding(token_ids) + self.pos_embedding[:, : token_ids.size(1), :]
        x = x.permute(0, 2, 1)          # (B, embed_dim, seq)
        x = self.conv_stack(x)          # (B, 128, seq)
        x = x.permute(0, 2, 1)          # (B, seq, 128)
        if self.use_padding_mask:
            pad_mask = token_ids.eq(self.embedding.padding_idx)  # (B, seq), True at <PAD>
            x = self.attention(x, key_padding_mask=pad_mask)
            real_mask = (~pad_mask).unsqueeze(-1).to(x.dtype)      # (B, seq, 1)
            x = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)  # masked mean pool
        else:
            # legacy (pre-b876d26): no attention mask, mean over every slot
            x = self.attention(x)
            x = x.mean(dim=1)
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
    dropout: float = 0.4
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 150
    batch_size: int = 64  # Behavioral used 128; halved here since train is
                          # only 6,070 examples (~95 batches/epoch at 64 vs
                          # ~47 at 128) -- more gradient updates per epoch
                          # on a small dataset, a mild additional precaution
                          # in the same direction as the dropout increase.
    patience: int = 15
    grad_clip: float = 1.0
    seed: int = DEFAULT_SEED
    use_padding_mask: bool = True  # every newly trained checkpoint uses masking


def build_model(cfg: TrainConfig, device: Optional[str] = None) -> CortexEmulationNet:
    torch.manual_seed(cfg.seed)
    model = CortexEmulationNet(
        vocab_size=cfg.vocab_size, sequence_length=SEQUENCE_LENGTH,
        embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, dropout=cfg.dropout,
        use_padding_mask=cfg.use_padding_mask,
    )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev)
