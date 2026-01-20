from dataclasses import dataclass

import torch


@dataclass
class BaseConfig:
    p: int = 97
    train_frac: float = 0.5 # 0.2
    embed_dim: int = 128
    hidden_dim: int = 256
    depth: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    steps: int = 200_000
    eval_every: int = 200
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainFrac40Config(BaseConfig):
    train_frac: float = 0.45
    steps: int = 300_000
