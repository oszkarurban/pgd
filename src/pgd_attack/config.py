"""Dataclass-based configuration loaded from YAML + CLI overrides."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AttackConfig:
    k: int = 22000
    eps: float = 0.2
    eps_step: float = 0.2
    clip_min: float = 0.0
    clip_max: float = 1.0
    optimizer: str = "adam"
    loss: str = "cross_entropy"


@dataclass
class OptimizerConfig:
    adam_lr: float = 0.0095
    sgd_lr: float = 0.25


@dataclass
class LoggingConfig:
    log_every: int = 20
    max_success_per_sample: int = 50


@dataclass
class ModelConfig:
    target_model: str = "Salesforce/blip2-flan-t5-xl"
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class PGDConfig:
    attack: AttackConfig = field(default_factory=AttackConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PGDConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            attack=AttackConfig(**raw.get("attack", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            model=ModelConfig(**raw.get("model", {})),
        )

    def with_overrides(self, **fields_per_section: dict[str, Any]) -> "PGDConfig":
        return replace(
            self,
            attack=replace(self.attack, **fields_per_section.get("attack", {})),
            optimizer=replace(self.optimizer, **fields_per_section.get("optimizer", {})),
            logging=replace(self.logging, **fields_per_section.get("logging", {})),
            model=replace(self.model, **fields_per_section.get("model", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
