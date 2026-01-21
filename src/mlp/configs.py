from dataclasses import dataclass
from typing import Any, Dict

import torch
import yaml


@dataclass
class Config:
    p: int
    train_frac: float
    embed_dim: int
    hidden_dim: int
    depth: int
    batch_size: int
    lr: float
    weight_decay: float
    steps: int
    eval_every: int
    seed: int
    device: str


def _validate_fields(data: Dict[str, Any], path: str) -> None:
    required_fields = {field.name for field in Config.__dataclass_fields__.values()}
    keys = set(data.keys())
    missing = sorted(required_fields - keys)
    extra = sorted(keys - required_fields)

    errors = []
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown fields: {', '.join(extra)}")
    if errors:
        raise ValueError(f"Invalid config '{path}': " + "; ".join(errors))


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config '{path}': expected a mapping")

    _validate_fields(data, path)
    data["device"] = _resolve_device(data["device"])
    return Config(**data)
