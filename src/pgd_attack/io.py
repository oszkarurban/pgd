"""Filesystem helpers: persisting adversarial images and run directories."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image


def save_adv_image(image: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image.detach().cpu(), path)
    return path


def allocate_run_dir(root: str | Path, args: Any) -> Path:
    """Create `root/loss_X/eps_Y/.../run_N` and dump the resolved args as JSON-ish text."""
    root = Path(root)
    parts = [_format_part(k, args) for k in ("loss", "eps", "eps_step", "optimizer", "k") if _has(args, k)]
    base = root.joinpath(*parts)

    n = 0
    while (base / f"run_{n}").exists():
        n += 1
    run_dir = base / f"run_{n}"
    run_dir.mkdir(parents=True, exist_ok=False)

    args_dict = asdict(args) if is_dataclass(args) else (vars(args) if not isinstance(args, dict) else args)
    (run_dir / "args.txt").write_text("\n".join(f"{k}: {v}" for k, v in args_dict.items()) + "\n")
    return run_dir


def _has(obj: Any, key: str) -> bool:
    return hasattr(obj, key) or (isinstance(obj, dict) and key in obj)


def _format_part(key: str, obj: Any) -> str:
    value = obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return f"{key}_{value}"
