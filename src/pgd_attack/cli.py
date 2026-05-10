"""Command-line entry point: load config, run PGD, save results."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from pgd_attack.attack import PGDAttack
from pgd_attack.config import PGDConfig
from pgd_attack.generate import generate
from pgd_attack.io import allocate_run_dir, save_adv_image
from pgd_attack.losses import LOSS_FUNCTIONS, get_loss_fn
from pgd_attack.models import encode_targets, load_model
from pgd_attack.preprocess import differentiable_preprocess

DEFAULT_CONFIG = Path("configs/default.yaml")
SUPPORTED_MODELS = ("Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PGD adversarial attack against a vision-language model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config to seed defaults from")

    parser.add_argument("--image-dir", type=Path, required=True, help="Folder of input PNGs to attack")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--target-text", type=str, required=True,
                        help="String the model should be coerced into emitting for every image")

    parser.add_argument("--target-model", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--eps-step", type=float, default=None)
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--loss", choices=tuple(LOSS_FUNCTIONS), default=None)
    parser.add_argument("--test-only", action="store_true", help="Skip the attack and just decode the clean images")
    return parser


def apply_overrides(config: PGDConfig, args: argparse.Namespace) -> PGDConfig:
    overrides: dict[str, dict] = {"attack": {}, "model": {}}
    for name in ("k", "eps", "clip_min", "clip_max", "optimizer", "loss"):
        value = getattr(args, name)
        if value is not None:
            overrides["attack"][name] = value
    if args.eps_step is not None:
        overrides["attack"]["eps_step"] = args.eps_step
    if args.target_model is not None:
        overrides["model"]["target_model"] = args.target_model
    if args.device is not None:
        overrides["model"]["device"] = args.device
    return config.with_overrides(**overrides)


def load_images(image_dir: Path, device: str) -> torch.Tensor:
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"No .png files found in {image_dir}")
    tensors = [transforms.ToTensor()(Image.open(f).convert("RGB")) for f in files]
    return torch.stack(tensors).to(device)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PGDConfig.from_yaml(args.config) if args.config.exists() else PGDConfig()
    config = apply_overrides(config, args)

    device = config.model.device
    images = load_images(args.image_dir, device)
    target_texts = [args.target_text] * images.shape[0]

    bundle = load_model(config.model.target_model, device=device,
                        dtype=getattr(torch, config.model.dtype))
    encoded = encode_targets(bundle, target_texts, device)

    if args.test_only:
        inputs = differentiable_preprocess(bundle.processor.image_processor, images, return_tensors="pt")
        result = generate(bundle.model, max_new_tokens=len(args.target_text), inputs=inputs)
        for response in bundle.processor.batch_decode(result.sequences, skip_special_tokens=True):
            print(response)
        return

    run_dir = allocate_run_dir(args.out_dir, config.attack)
    print(f"Run directory: {run_dir}")

    attack = PGDAttack(bundle, config, get_loss_fn(config.attack.loss))
    result = attack.run(
        x=images,
        input_ids=encoded["input_ids"],
        target_ids=encoded["target_ids"],
        target_texts=target_texts,
        out_dir=run_dir,
    )

    _persist_top_k(result, run_dir)


def _persist_top_k(result, run_dir: Path) -> None:
    for sample_idx, (succeeded, iters, imgs, losses) in enumerate(
        zip(result.has_success, result.success_iters, result.success_images, result.success_losses)
    ):
        sample_dir = run_dir / "final" / f"image_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for it, img in zip(iters, imgs):
            save_adv_image(img, sample_dir / f"final_image_{it}.png")

        summary = [str(succeeded), ""]
        for it, loss in zip(iters, losses):
            summary.append(f"{it} - loss: {loss}")
        if succeeded and losses:
            best = min(losses)
            summary.append("")
            summary.append(f"Best loss: {best} at index {iters[losses.index(best)]}")
        (sample_dir / "success.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
