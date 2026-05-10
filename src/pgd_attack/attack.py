"""PGD attack loop targeting vision-language captioning models."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from pgd_attack.config import PGDConfig
from pgd_attack.generate import generate, single_forward_pass
from pgd_attack.io import save_adv_image
from pgd_attack.losses import LossFn
from pgd_attack.models import VLMBundle
from pgd_attack.preprocess import differentiable_preprocess

DEC_ONLY_RADIUS = 0.25  # tighter clamp ball used for LLaVA pixel-value space


@dataclass
class AttackResult:
    has_success: list[bool]
    success_images: list[list[torch.Tensor]] = field(default_factory=list)
    success_iters: list[list[int]] = field(default_factory=list)
    success_losses: list[list[float]] = field(default_factory=list)


class PGDAttack:
    """Run PGD against an encoder-decoder or decoder-only VLM."""

    def __init__(self, bundle: VLMBundle, config: PGDConfig, loss_fn: LossFn) -> None:
        self.bundle = bundle
        self.config = config
        self.loss_fn = loss_fn

    def run(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_texts: Sequence[str],
        out_dir: str | Path,
    ) -> AttackResult:
        cfg_attack = self.config.attack
        out_dir = Path(out_dir)
        image_dir = out_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        device = self.bundle.model.device
        x = x.to(device)
        x_min, x_max = self._clamp_bounds(x, cfg_attack.eps, cfg_attack.clip_min, cfg_attack.clip_max)
        x_adv = (x + cfg_attack.eps * (2 * torch.rand_like(x) - 1)).clamp_(min=x_min, max=x_max)

        adv = x_adv.clone().detach_().requires_grad_()
        if self.bundle.architecture == "dec_only":
            x_min = adv.detach() - DEC_ONLY_RADIUS
            x_max = adv.detach() + DEC_ONLY_RADIUS

        optimizer = self._make_optimizer(adv)

        self.bundle.model.eval()
        self.bundle.model.requires_grad_(False)

        bs = adv.shape[0]
        max_new_tokens = input_ids.shape[-1] if self.bundle.architecture == "enc_dec" else target_ids.shape[-1]
        result = AttackResult(
            has_success=[False] * bs,
            success_images=[[] for _ in range(bs)],
            success_iters=[[] for _ in range(bs)],
            success_losses=[[] for _ in range(bs)],
        )
        losses: list[float] = []
        max_keep = self.config.logging.max_success_per_sample

        pbar = trange(cfg_attack.k, desc="PGD", leave=True)
        for step in pbar:
            optimizer.zero_grad()
            inputs = self._build_inputs(adv, input_ids, target_ids)
            loss = self._compute_loss(inputs, input_ids if self.bundle.architecture == "enc_dec" else target_ids)

            loss.backward()
            optimizer.step()
            adv.data.clamp_(min=x_min, max=x_max)

            losses.append(loss.item())
            responses = self._decode(adv, input_ids, max_new_tokens)
            self._record_success(adv, responses, target_texts, step, result, max_keep)

            self._update_pbar(pbar, loss.item(), responses, sum(result.has_success), bs)

            if step % self.config.logging.log_every == 0:
                self._snapshot(out_dir, image_dir, adv, losses, responses, target_texts, step)

        return _trim_top_k(result, losses, k=10)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clamp_bounds(x: torch.Tensor, eps: float, clip_min: float, clip_max: float) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.clamp(x - eps, clip_min, clip_max), torch.clamp(x + eps, clip_min, clip_max)

    def _make_optimizer(self, param: torch.Tensor) -> torch.optim.Optimizer:
        opt = self.config.attack.optimizer
        if opt == "sgd":
            return torch.optim.SGD([param], lr=self.config.optimizer.sgd_lr)
        if opt == "adam":
            return torch.optim.Adam([param], lr=self.config.optimizer.adam_lr)
        raise ValueError(f"Unknown optimizer: {opt}")

    def _build_inputs(
        self,
        adv: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict:
        if self.bundle.architecture == "enc_dec":
            return differentiable_preprocess(self.bundle.processor.image_processor, adv, return_tensors="pt")
        return {"pixel_values": adv, "input_ids": input_ids, "target_ids": target_ids}

    def _compute_loss(self, inputs: dict, encoder_input_ids: torch.Tensor) -> torch.Tensor:
        if self.bundle.architecture == "enc_dec":
            loss, _ = single_forward_pass(
                self.bundle.model,
                inputs,
                decoder_input_ids=None,
                labels=encoder_input_ids,
                loss_fn=self.loss_fn,
            )
            return loss

        # dec_only
        input_ids = inputs["input_ids"]
        target_ids = inputs["target_ids"]
        combined = torch.cat((input_ids, target_ids), dim=-1)
        attention_mask = torch.ones_like(combined, device=input_ids.device)
        out = self.bundle.model(
            pixel_values=inputs["pixel_values"], input_ids=combined, attention_mask=attention_mask
        )
        out_logits = out.logits[:, -target_ids.shape[-1] - 1 : -1, :]
        return self.loss_fn(out_logits, target_ids)

    def _decode(self, adv: torch.Tensor, input_ids: torch.Tensor, max_new_tokens: int) -> list[str]:
        with torch.no_grad():
            if self.bundle.architecture == "enc_dec":
                inputs = differentiable_preprocess(self.bundle.processor.image_processor, adv.detach(), return_tensors="pt")
            else:
                inputs = {"pixel_values": adv.detach().clone(), "input_ids": input_ids.detach().clone()}
            result = generate(self.bundle.model, max_new_tokens=max_new_tokens, inputs=inputs)
        return self.bundle.processor.batch_decode(result.sequences, skip_special_tokens=True)

    @staticmethod
    def _record_success(
        adv: torch.Tensor,
        responses: list[str],
        target_texts: Sequence[str],
        step: int,
        result: AttackResult,
        max_keep: int,
    ) -> None:
        for j, (response, target) in enumerate(zip(responses, target_texts)):
            if response != target:
                continue
            result.has_success[j] = True
            result.success_iters[j].append(step)
            result.success_images[j].append(adv[j].detach().cpu().clone())
            if len(result.success_iters[j]) > max_keep:
                result.success_iters[j] = result.success_iters[j][-max_keep:]
                result.success_images[j] = result.success_images[j][-max_keep:]

    @staticmethod
    def _update_pbar(pbar, loss_val: float, responses: list[str], successes: int, bs: int) -> None:
        preview = responses[:5] if len(responses) > 5 else responses
        pbar.set_description(f"loss={loss_val:.3f} pred={preview} success={successes}/{bs}")

    def _snapshot(
        self,
        out_dir: Path,
        image_dir: Path,
        adv: torch.Tensor,
        losses: list[float],
        responses: list[str],
        target_texts: Sequence[str],
        step: int,
    ) -> None:
        plt.figure()
        plt.plot(losses, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()

        step_dir = image_dir / f"iter={step}"
        for idx in range(adv.shape[0]):
            save_adv_image(adv[idx].detach().clone(), step_dir / f"image_{idx}.png")

        info = [f"Loss: {losses[-1]}"]
        for resp, tgt in zip(responses, target_texts):
            info.append(f"Response: {resp}\nTarget:   {tgt}\n")
        (step_dir / "info.txt").write_text("\n".join(info))


def _trim_top_k(result: AttackResult, losses: list[float], k: int) -> AttackResult:
    """Keep only the k lowest-loss successful images per sample, sorted ascending."""
    bs = len(result.has_success)
    for j in range(bs):
        if not result.has_success[j]:
            result.success_iters[j] = []
            result.success_images[j] = []
            result.success_losses[j] = []
            continue
        result.success_losses[j] = [losses[i] for i in result.success_iters[j]]
        zipped = sorted(
            zip(result.success_images[j], result.success_iters[j], result.success_losses[j]),
            key=lambda triple: triple[2],
        )[:k]
        if zipped:
            imgs, iters, losses_ = (list(col) for col in zip(*zipped))
            result.success_images[j], result.success_iters[j], result.success_losses[j] = imgs, iters, losses_
    return result
