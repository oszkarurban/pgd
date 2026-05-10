"""Vision-language model loading + per-architecture metadata.

The PGD loop branches on whether the underlying VLM is encoder-decoder
(e.g. BLIP-2 / FLAN-T5) or decoder-only (e.g. LLaVA). We capture that distinction
once here in :class:`VLMBundle.architecture` so the rest of the code can stay
architecture-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
)

Architecture = Literal["enc_dec", "dec_only"]

LLAVA_PROMPT_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Give a short description of this image."},
            {"type": "image"},
        ],
    },
]


@dataclass
class VLMBundle:
    model: torch.nn.Module
    processor: object
    architecture: Architecture
    model_id: str


def load_model(model_id: str, *, device: str = "cuda", dtype: torch.dtype = torch.float16) -> VLMBundle:
    if model_id == "Salesforce/blip2-flan-t5-xl":
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
        return VLMBundle(model=model, processor=processor, architecture="enc_dec", model_id=model_id)

    if model_id == "llava-hf/llava-1.5-7b-hf":
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        return VLMBundle(model=model, processor=processor, architecture="dec_only", model_id=model_id)

    raise ValueError(f"Unsupported model id: {model_id!r}")


def encode_targets(bundle: VLMBundle, target_texts: list[str], device: str) -> dict[str, torch.Tensor]:
    """Tokenise target texts into the input/target pair the attack loop expects."""
    tokenizer = bundle.processor.tokenizer
    target_ids = tokenizer(target_texts, return_tensors="pt", padding=True).input_ids.to(device)

    if bundle.architecture == "enc_dec":
        # BLIP-2: feed the target text into the encoder, drop BOS for the decoder labels.
        input_ids = target_ids
        decoder_target_ids = target_ids[:, 1:]
        return {"input_ids": input_ids, "target_ids": decoder_target_ids}

    # LLaVA: build the chat prompt; concat with target ids inside the attack step
    prompt = bundle.processor.apply_chat_template(LLAVA_PROMPT_CONVERSATION, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return {"input_ids": input_ids, "target_ids": target_ids[:, 1:]}
