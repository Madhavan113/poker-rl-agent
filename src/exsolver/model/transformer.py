"""Decoder-only causal transformer with action / opponent / theta heads."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (spec defaults: 4 x 128, 4 heads, max_len 1024, no dropout)."""

    vocab_size: int
    n_actions: int = 3
    n_opp: int | None = None  # opponent head size (discrete population); None disables the head
    theta_dim: int | None = None  # theta head size (continuous prior); None disables the head
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 1024
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.n_opp is not None and self.n_opp < 1:
            self.n_opp = None
        if self.theta_dim is not None and self.theta_dim < 1:
            self.theta_dim = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**d)


def illegal_fill_value(dtype: torch.dtype) -> float:
    """Large negative that survives softmax/CE in ``dtype`` without producing inf/NaN."""
    return -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9


def mask_illegal(action_logits: Tensor, legal: Tensor) -> Tensor:
    """Set logits of illegal actions to a large negative (``legal`` is a bool tensor, same shape)."""
    return action_logits.masked_fill(~legal.to(torch.bool), illegal_fill_value(action_logits.dtype))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention via ``scaled_dot_product_attention``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-LN transformer block: attention + GELU MLP, both residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ExploitTransformer(nn.Module):
    """Token + learned positional embeddings, pre-LN causal blocks, final LN, per-position heads.

    ``forward`` returns ``{"action_logits": [B,T,3], "opp_logits": [B,T,M] | None,
    "theta_logits": [B,T,P] | None}`` (plus ``"hidden"`` when requested).  The theta logits are
    pre-sigmoid.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, cfg.n_actions)
        self.opp_head = nn.Linear(cfg.d_model, cfg.n_opp) if cfg.n_opp else None
        self.theta_head = nn.Linear(cfg.d_model, cfg.theta_dim) if cfg.theta_dim else None
        self.apply(self._init_weights)
        # GPT-2 style: scale the residual-branch output projections by depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, tokens: Tensor) -> Tensor:
        """Final-LN hidden states ``[B, T, d_model]``."""
        b, t = tokens.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len={self.cfg.max_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def heads(self, h: Tensor) -> dict[str, Tensor | None]:
        """Apply the heads to hidden states of any leading shape."""
        return {
            "action_logits": self.action_head(h),
            "opp_logits": self.opp_head(h) if self.opp_head is not None else None,
            "theta_logits": self.theta_head(h) if self.theta_head is not None else None,
        }

    def forward(self, tokens: Tensor, return_hidden: bool = False) -> dict[str, Tensor | None]:
        h = self.hidden_states(tokens)
        out = self.heads(h)
        if return_hidden:
            out["hidden"] = h
        return out

    def num_parameters(self, trainable_only: bool = True) -> int:
        return count_parameters(self, trainable_only)

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
