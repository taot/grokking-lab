import json
from dataclasses import MISSING, asdict, dataclass, replace
from typing import Any, Dict, Optional

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
    checkpoint_every: int = 0
    use_layernorm: bool = False
    use_sinusoidal_pe: bool = False
    use_eq_token: bool = True

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


def _parse_bool_from_string(raw: str, key: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise TypeError(f"Override '{key}' expects bool, got string '{raw}'")


def _cast_override_value(key: str, raw: Any, expected_type: Any) -> Any:
    if expected_type is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return _parse_bool_from_string(raw, key)
        raise TypeError(f"Override '{key}' expects bool, got {type(raw).__name__}")

    if expected_type is int:
        if isinstance(raw, bool):
            raise TypeError(f"Override '{key}' expects int, got bool")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if raw.is_integer():
                return int(raw)
            raise TypeError(f"Override '{key}' expects int, got float {raw}")
        if isinstance(raw, str):
            return int(raw)
        raise TypeError(f"Override '{key}' expects int, got {type(raw).__name__}")

    if expected_type is float:
        if isinstance(raw, bool):
            raise TypeError(f"Override '{key}' expects float, got bool")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            return float(raw)
        raise TypeError(f"Override '{key}' expects float, got {type(raw).__name__}")

    if expected_type is str:
        if isinstance(raw, str):
            return raw
        return str(raw)

    raise TypeError(
        f"Override '{key}' has unsupported field type {expected_type!r}; "
        "supported: bool/int/float/str"
    )


def apply_overrides(
    cfg: Config,
    overrides: Dict[str, Any],
    *,
    allowed_keys: Optional[set[str]] = None,
) -> Config:
    if not overrides:
        return cfg

    all_fields = set(Config.__dataclass_fields__.keys())
    unknown = sorted(set(overrides.keys()) - all_fields)
    if unknown:
        raise ValueError(
            "Unknown override fields: "
            + ", ".join(unknown)
            + "; allowed fields: "
            + ", ".join(sorted(all_fields))
        )

    if allowed_keys is not None:
        disallowed = sorted(set(overrides.keys()) - allowed_keys)
        if disallowed:
            raise ValueError(
                "Overrides not allowed for this run: "
                + ", ".join(disallowed)
                + "; allowed: "
                + ", ".join(sorted(allowed_keys))
            )

    updates: Dict[str, Any] = {}
    for key, raw in overrides.items():
        field = Config.__dataclass_fields__[key]
        expected_type = field.type
        value = _cast_override_value(key, raw, expected_type)
        if key == "device":
            value = _resolve_device(value)
        updates[key] = value

    return replace(cfg, **updates)
