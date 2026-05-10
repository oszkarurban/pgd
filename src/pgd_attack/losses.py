"""Loss-function factory for PGD attacks against captioning VLMs.

Every loss takes (logits, target) where:
- ``logits`` is shape ``[B, T, V]`` (encoder-decoder) or ``[B*T, V]`` (decoder-only after slicing)
- ``target`` matches ``logits`` shape minus the vocab dimension.

All losses pad-mask token id 0 to zero so PAD positions contribute nothing.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

PAD_TOKEN_ID = 0
MARGIN_THRESHOLD = 0.1
DOWNWEIGHT = 0.1


def _per_token_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy over [..., V] vs [...] targets, returning a tensor shaped like ``target``."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    return nn.functional.cross_entropy(flat_logits, flat_target, reduction="none").view_as(target)


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = _per_token_ce(logits, target)
    loss = loss.masked_fill(target == PAD_TOKEN_ID, 0.0)
    return loss.mean()


def mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Down-weight already-correct predictions by ``DOWNWEIGHT``."""
    is_correct = logits.argmax(dim=-1) == target
    loss = nn.functional.cross_entropy(logits, target, reduction="none")
    loss = torch.where(is_correct, DOWNWEIGHT * loss, loss)
    return loss.mean()


def margin_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Down-weight tokens that are correct AND well-separated from the runner-up."""
    loss = nn.functional.cross_entropy(logits, target, reduction="none")
    is_correct = logits.argmax(dim=-1) == target

    probs = torch.softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1).values
    confident = (top2[..., 0] - top2[..., 1]) > MARGIN_THRESHOLD

    weight = torch.ones_like(loss)
    weight[is_correct & confident] = DOWNWEIGHT
    return (loss * weight).mean()


def iterative_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Optimise only up to and including the first incorrect token, zero everything after."""
    loss = nn.functional.cross_entropy(logits, target, reduction="none")
    incorrect = (logits.argmax(dim=-1) != target).nonzero(as_tuple=True)[0]
    if incorrect.numel() == 0:
        return loss.mean()
    cutoff = incorrect[0].item() + 1

    mask = torch.zeros_like(loss)
    mask[:cutoff] = 1.0
    return (loss * mask).mean()


LOSS_FUNCTIONS: dict[str, LossFn] = {
    "cross_entropy": cross_entropy_loss,
    "mask": mask_loss,
    "margin": margin_loss,
    "iterative": iterative_loss,
}


def get_loss_fn(name: str) -> LossFn:
    try:
        return LOSS_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown loss '{name}'. Choices: {sorted(LOSS_FUNCTIONS)}") from exc
