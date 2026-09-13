"""Batched inference over counterfactual prefixes: ``policy_at`` / ``belief_at`` / ``theta_at``."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import chain

import numpy as np
import torch
from torch import Tensor

from exsolver.data.records import N_ACTIONS
from exsolver.data.tokenizer import Tokenizer
from exsolver.model.transformer import ExploitTransformer, mask_illegal

Prefix = Sequence[int] | np.ndarray


def pack_prefixes(prefixes: Sequence[Prefix], pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad variable-length prefixes into ``[B, T]`` int64 and return the last-token index.

    Vectorised: no Python loop over tokens, one pass over the prefixes to get their lengths.
    """
    b = len(prefixes)
    if b == 0:
        raise ValueError("need at least one prefix")
    lengths = np.fromiter((len(p) for p in prefixes), dtype=np.int64, count=b)
    if lengths.min() < 1:
        raise ValueError("every prefix needs at least one token")
    total = int(lengths.sum())
    flat = np.fromiter(chain.from_iterable(prefixes), dtype=np.int64, count=total)
    t = int(lengths.max())
    tokens = np.full((b, t), pad, dtype=np.int64)
    rows = np.repeat(np.arange(b), lengths)
    cols = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    tokens[rows, cols] = flat
    return tokens, lengths - 1


def check_legal(legal: np.ndarray | Sequence[Sequence[bool]], n: int) -> np.ndarray:
    """Validate a ``[n, 3]`` legal-action mask; every row must allow at least one action."""
    arr = np.asarray(legal, dtype=bool)
    if arr.shape != (n, N_ACTIONS):
        raise ValueError(f"legal must have shape {(n, N_ACTIONS)}, got {arr.shape}")
    if not arr.any(axis=1).all():
        bad = np.flatnonzero(~arr.any(axis=1)).tolist()
        raise ValueError(f"legal rows {bad} allow no action; every query needs a legal action")
    return arr


def _same_device(a: torch.device, b: torch.device) -> bool:
    return a.type == b.type and (a.index or 0) == (b.index or 0)


class Policy:
    """Wraps an ``ExploitTransformer`` for batched read-outs at the end of each prefix.

    Right padding is safe because attention is causal: the hidden state at the last real token
    never sees the padding.  Action distributions always require a ``legal`` mask; there is no
    unmasked action softmax.

    The wrapped module's train/eval mode is not changed persistently: each query switches to
    ``eval()`` for the forward pass and restores the previous mode.  The one mutation is moving
    the module when an explicit ``device`` differs from where it currently lives.
    """

    def __init__(
        self,
        model: ExploitTransformer,
        tokenizer: Tokenizer,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else model.device
        if not _same_device(self.device, model.device):
            self.model.to(self.device)
        self.pad = tokenizer.PAD

    @contextmanager
    def _eval_mode(self) -> Iterator[None]:
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    @torch.inference_mode()
    def outputs_at(self, prefixes: Sequence[Prefix]) -> dict[str, Tensor | None]:
        """Head outputs at the last token of every prefix: ``action_logits [B,3]``, etc."""
        tokens_np, last = pack_prefixes(prefixes, self.pad)
        tokens = torch.from_numpy(tokens_np).to(self.device, non_blocking=True)
        with self._eval_mode():
            h = self.model.hidden_states(tokens)
            rows = torch.arange(h.shape[0], device=h.device)
            h_last = h[rows, torch.from_numpy(last).to(h.device)]
            return self.model.heads(h_last)

    def policy_at(self, prefixes: Sequence[Prefix], legal: np.ndarray) -> np.ndarray:
        """Action distributions ``[B, 3]`` at the end of each prefix, restricted to ``legal``."""
        logits = self.outputs_at(prefixes)["action_logits"]
        assert logits is not None
        return self.action_probs(logits, legal)

    def action_probs(self, logits: Tensor, legal: np.ndarray) -> np.ndarray:
        """Softmax of ``logits`` over the legal actions only (exact zeros elsewhere)."""
        legal_t = torch.as_tensor(check_legal(legal, logits.shape[0]), device=logits.device)
        probs = torch.softmax(mask_illegal(logits.float(), legal_t), dim=-1) * legal_t
        probs = probs / probs.sum(-1, keepdim=True)
        return probs.cpu().numpy()

    def belief_at(self, prefixes: Sequence[Prefix]) -> np.ndarray:
        """Posterior over the discrete population ``[B, M]`` (softmax of the opponent head)."""
        logits = self.outputs_at(prefixes)["opp_logits"]
        if logits is None:
            raise RuntimeError("model has no opponent head")
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def theta_at(self, prefixes: Sequence[Prefix]) -> np.ndarray:
        """Predicted opponent parameters ``[B, P]`` (sigmoid of the theta head)."""
        logits = self.outputs_at(prefixes)["theta_logits"]
        if logits is None:
            raise RuntimeError("model has no theta head")
        return torch.sigmoid(logits.float()).cpu().numpy()

    def all_at(self, prefixes: Sequence[Prefix], legal: np.ndarray) -> dict[str, np.ndarray | None]:
        """Policy, belief and theta from a single forward pass."""
        out = self.outputs_at(prefixes)
        action_logits = out["action_logits"]
        assert action_logits is not None
        opp = out["opp_logits"]
        theta = out["theta_logits"]
        return {
            "policy": self.action_probs(action_logits, legal),
            "belief": torch.softmax(opp.float(), -1).cpu().numpy() if opp is not None else None,
            "theta": torch.sigmoid(theta.float()).cpu().numpy() if theta is not None else None,
        }
