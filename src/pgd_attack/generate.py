"""Generation and single-forward-pass helpers for encoder-decoder VLMs."""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch


def generate(model, max_new_tokens: int, inputs: dict[str, Any], **kwargs: Any):
    kwargs.setdefault("output_logits", True)
    kwargs.setdefault("return_dict_in_generate", True)
    kwargs["max_new_tokens"] = max_new_tokens
    return model.generate(**inputs, **kwargs)


def single_forward_pass(
    model,
    inputs: dict[str, torch.Tensor],
    *,
    decoder_input_ids: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass through an encoder-decoder VLM, returning (loss, logits).

    If ``decoder_input_ids`` is None, builds them from ``labels`` shifted right by
    the configured ``decoder_start_token_id`` (mirrors what HF does internally
    when only ``labels`` is supplied).
    """
    pixel_values = inputs["pixel_values"]
    bs = pixel_values.shape[0]
    device = pixel_values.device

    encoder_start = model.config.text_config.bos_token_id
    decoder_start = model.config.text_config.decoder_start_token_id
    input_ids = torch.full((bs, 1), encoder_start, dtype=torch.long, device=device)

    if decoder_input_ids is None and labels is not None:
        decoder_input_ids = torch.cat(
            [torch.full((bs, 1), decoder_start, dtype=torch.long, device=device), labels],
            dim=1,
        )[:, :-1]

    if decoder_input_ids is None:
        decoder_input_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)

    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )
    return loss_fn(out.logits, labels), out.logits
