"""DPT losses: masked action CE (+ opponent CE / theta BCE at belief positions)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from exsolver.model.transformer import mask_illegal


def _masked_mean(per_position: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``per_position`` over ``mask`` (0 when the mask is empty, so losses stay finite)."""
    m = mask.to(per_position.dtype)
    return (per_position * m).sum() / m.sum().clamp(min=1.0)


def _masked_accuracy(correct: Tensor, mask: Tensor) -> Tensor:
    """Fraction of ``mask`` positions that are ``correct``; NaN (not 0) when the mask is empty."""
    m = mask.to(torch.float32)
    return (correct.to(torch.float32) * m).sum() / m.sum()


def action_loss(
    action_logits: Tensor, action_target: Tensor, action_mask: Tensor, legal: Tensor
) -> tuple[Tensor, Tensor]:
    """Masked cross-entropy and accuracy of the action head at decision positions.

    Illegal actions are masked before the softmax; targets are ``-1`` off-mask and ignored.
    With no masked positions the loss is 0 (finite, zero gradient) and the accuracy NaN.
    """
    logits = mask_illegal(action_logits, legal)
    target = action_target.clamp(min=0)
    ce = F.cross_entropy(logits.flatten(0, -2), target.flatten(), reduction="none").view(
        target.shape
    )
    loss = _masked_mean(ce, action_mask)
    acc = _masked_accuracy((logits.argmax(-1) == action_target) & action_mask, action_mask)
    return loss, acc


def opponent_loss(opp_logits: Tensor, opp_id: Tensor, belief_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Cross-entropy (and accuracy) of the opponent head at belief positions.

    Sessions with ``opp_id == -1`` (continuous prior) are skipped.
    """
    mask = belief_mask & (opp_id >= 0)[:, None]
    target = opp_id.clamp(min=0)[:, None].expand_as(mask)
    ce = F.cross_entropy(opp_logits.flatten(0, -2), target.flatten(), reduction="none").view(
        mask.shape
    )
    loss = _masked_mean(ce, mask)
    acc = _masked_accuracy((opp_logits.argmax(-1) == target) & mask, mask)
    return loss, acc


def theta_loss(theta_logits: Tensor, theta: Tensor, belief_mask: Tensor) -> Tensor:
    """BCE-with-logits of the theta head at belief positions (mean over parameters)."""
    target = theta[:, None, :].expand_as(theta_logits)
    bce = F.binary_cross_entropy_with_logits(theta_logits, target, reduction="none").mean(-1)
    return _masked_mean(bce, belief_mask)


def compute_loss(
    out: dict[str, Tensor | None],
    batch: dict[str, Tensor],
    lambda_opp: float = 0.5,
    lambda_theta: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Total loss ``CE_action + λ_opp CE_opp + λ_theta BCE_theta`` and detached components.

    Components: ``loss_action``, ``acc_action``, ``n_action`` (decision positions), ``n_belief``
    (belief positions) and, when the corresponding head exists, ``loss_opp``, ``acc_opp``,
    ``loss_theta``.  Accuracies are NaN when their mask is empty; losses are always finite.
    """
    action_logits = out["action_logits"]
    assert action_logits is not None
    loss_a, acc_a = action_loss(
        action_logits, batch["action_target"], batch["action_mask"], batch["legal"]
    )
    total = loss_a
    comps: dict[str, Tensor] = {
        "loss_action": loss_a.detach(),
        "acc_action": acc_a.detach(),
        "n_action": batch["action_mask"].sum().detach(),
        "n_belief": batch["belief_mask"].sum().detach(),
    }
    opp_logits = out.get("opp_logits")
    if opp_logits is not None:
        loss_o, acc_o = opponent_loss(opp_logits, batch["opp_id"], batch["belief_mask"])
        total = total + lambda_opp * loss_o
        comps["loss_opp"] = loss_o.detach()
        comps["acc_opp"] = acc_o.detach()
    theta_logits = out.get("theta_logits")
    if theta_logits is not None:
        loss_t = theta_loss(theta_logits, batch["theta"], batch["belief_mask"])
        total = total + lambda_theta * loss_t
        comps["loss_theta"] = loss_t.detach()
    comps["loss"] = total.detach()
    return total, comps


def to_floats(comps: dict[str, Tensor]) -> dict[str, float]:
    """``.item()`` every component (one device sync per call)."""
    return {
        k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in comps.items()
    }
