from pathlib import Path
from typing import Any

import torch
import yaml

from ..model import Transformer


def load_config(run_dir: Path) -> dict[str, Any]:
    """从 run_dir 加载 config.yaml"""
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def load_model(run_dir: Path, cfg: dict[str, Any] | None = None) -> Transformer:
    """从 run_dir 加载模型 checkpoint"""
    if cfg is None:
        cfg = load_config(run_dir)

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        output_size=cfg["p"],
        d=cfg["d"],
        n_layers=cfg["n_layers"],
        h=cfg["h"],
        max_seq_len=3,
    ).to(cfg["device"])
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location=cfg["device"])
    )
    model.eval()
    return model
