import json
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from ..model import Transformer
except ImportError:
    from transformer_progress_measures_ablation.model import Transformer


def load_config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def load_model(run_dir: Path, cfg: dict[str, Any] | None = None) -> Transformer:
    if cfg is None:
        cfg = load_config(run_dir)

    max_seq_len = 3 if cfg["use_eq_token"] else 2
    model = Transformer(
        vocab_size=cfg["vocab_size"],
        output_size=cfg["p"],
        d=cfg["d"],
        n_layers=cfg["n_layers"],
        h=cfg["h"],
        max_seq_len=max_seq_len,
        use_layernorm=cfg["use_layernorm"],
        use_sinusoidal_pe=cfg["use_sinusoidal_pe"],
    ).to(cfg["device"])
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location=cfg["device"])
    )
    model.eval()
    return model


def load_last_metrics(run_dir: Path) -> dict[str, float] | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    last_line = ""
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line

    if not last_line:
        return None

    data = json.loads(last_line)
    return {
        "step": float(data["step"]),
        "train_loss": float(data["train_loss"]),
        "train_acc": float(data["train_acc"]),
        "val_loss": float(data["val_loss"]),
        "val_acc": float(data["val_acc"]),
    }
