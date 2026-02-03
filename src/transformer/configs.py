import json
from dataclasses import MISSING, asdict, dataclass
from typing import Any, Dict

import torch
import yaml


@dataclass
class Config:
    p: int
    vocab_size: int
    d: int
    n_layers: int
    train_frac: float
    batch_size: int
    lr: float
    weight_decay: float
    steps: int
    eval_every: int
    seed: int
    device: str
    h: int = 4

    def print(self) -> None:
        print("Config:")
        print(json.dumps(asdict(self), indent=4))


def _validate_fields(data: Dict[str, Any], path: str) -> None:
    all_fields = {field.name for field in Config.__dataclass_fields__.values()}
    required_fields: set[str] = set()
    for field in Config.__dataclass_fields__.values():
        if field.default is MISSING and field.default_factory is MISSING:  # type: ignore[attr-defined]
            required_fields.add(field.name)
    keys = set(data.keys())
    missing = sorted(required_fields - keys)
    extra = sorted(keys - all_fields)

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
