import torch

import configs
from transformer.model import Transformer
from paths import PROJECT_ROOT

run_path = PROJECT_ROOT / "runs/transformer/one_layer/20260217_211250"

cfg = configs.load_config(run_path / "config.yaml")

model = Transformer(
    vocab_size=cfg.vocab_size, d=cfg.d, n_layers=cfg.n_layers, h=cfg.h
).to(device=cfg.device)

model.load_state_dict(torch.load(run_path / "checkpoint.pt", map_location=cfg.device))

model.linear.weight.shape
