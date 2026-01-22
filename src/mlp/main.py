import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TextIO

import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from . import configs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    # all pairs (a, b) with label (a + b) mod p
    xs = []
    ys = []
    for a in range(p):
        for b in range(p):
            xs.append((a, b))
            ys.append((a + b) % p)
    xs = torch.tensor(xs, dtype=torch.long)  # [N, 2]
    ys = torch.tensor(ys, dtype=torch.long)  # [N]
    return xs, ys


class MLP(nn.Module):
    def __init__(self, p: int, embed_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(p, embed_dim)
        layers = []
        in_dim = 2 * embed_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, p)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,2]
        a = self.emb(x[:, 0])
        b = self.emb(x[:, 1])
        h = torch.cat([a, b], dim=-1)
        return self.net(h)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    model.eval()
    logits = model(xs.to(device))
    loss = F.cross_entropy(logits, ys.to(device)).item()
    pred = logits.argmax(dim=-1).cpu()
    acc = (pred == ys).float().mean().item()
    return loss, acc


def _write_metrics(
    metrics_fp: TextIO,
    step: int,
    tr_loss: float,
    tr_acc: float,
    va_loss: float,
    va_acc: float,
) -> None:
    metrics_fp.write(
        json.dumps(
            {
                "step": step,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }
        )
        + "\n"
    )
    metrics_fp.flush()


def _append_metrics(
    logs: Dict[str, list],
    metrics_fp: TextIO,
    step: int,
    tr_loss: float,
    tr_acc: float,
    va_loss: float,
    va_acc: float,
) -> None:
    logs["step"].append(step)
    logs["train_loss"].append(tr_loss)
    logs["train_acc"].append(tr_acc)
    logs["val_loss"].append(va_loss)
    logs["val_acc"].append(va_acc)
    _write_metrics(metrics_fp, step, tr_loss, tr_acc, va_loss, va_acc)


def _load_metrics(metrics_file: Path) -> Dict[str, list]:
    logs = {
        "step": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    if not metrics_file.exists():
        return logs

    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            logs["step"].append(record["step"])
            logs["train_loss"].append(record["train_loss"])
            logs["train_acc"].append(record["train_acc"])
            logs["val_loss"].append(record["val_loss"])
            logs["val_acc"].append(record["val_acc"])
    return logs


def _create_runs_dir(experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path("runs") / experiment / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def main(
    cfg: configs.Config,
    experiment: str,
    config_path: str,
    resume_from: Optional[Path] = None,
) -> None:
    set_seed(cfg.seed)

    if resume_from is not None:
        checkpoint_path = resume_from / "checkpoint.pt"
        metrics_path = resume_from / "metrics.jsonl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    runs_dir = _create_runs_dir(experiment)
    shutil.copy(config_path, runs_dir / "config.yaml")

    xs, ys = make_dataset(cfg.p)
    n = xs.shape[0]
    idx = torch.randperm(n)
    n_train = int(n * cfg.train_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    x_tr, y_tr = xs[train_idx], ys[train_idx]
    x_va, y_va = xs[val_idx], ys[val_idx]

    model = MLP(cfg.p, cfg.embed_dim, cfg.hidden_dim, cfg.depth).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    metrics_file = runs_dir / "metrics.jsonl"
    logs = {
        "step": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_step = 0
    if resume_from is not None:
        checkpoint_path = resume_from / "checkpoint.pt"
        metrics_path = resume_from / "metrics.jsonl"

        print(f"Resuming from {resume_from}")

        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
        logs = _load_metrics(metrics_path)
        if logs["step"]:
            start_step = logs["step"][-1]

    cfg.print()

    with open(metrics_file, "w", encoding="utf-8") as metrics_fp:
        if logs["step"]:
            for idx in range(len(logs["step"])):
                _write_metrics(
                    metrics_fp,
                    logs["step"][idx],
                    logs["train_loss"][idx],
                    logs["train_acc"][idx],
                    logs["val_loss"][idx],
                    logs["val_acc"][idx],
                )

        if start_step >= cfg.steps:
            print(
                f"Resume step {start_step} is >= total steps {cfg.steps}; "
                "skipping training."
            )
        else:
            for step in range(start_step + 1, cfg.steps + 1):
                model.train()
                # mini-batch
                bidx = torch.randint(0, x_tr.shape[0], (cfg.batch_size,))
                xb = x_tr[bidx].to(cfg.device)
                yb = y_tr[bidx].to(cfg.device)

                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if step % cfg.eval_every == 0 or step == 1:
                    tr_loss, tr_acc = evaluate(model, x_tr, y_tr, cfg.device)
                    va_loss, va_acc = evaluate(model, x_va, y_va, cfg.device)

                    print(
                        f"step {step:6d} | "
                        f"train loss {tr_loss:.4f} acc {tr_acc * 100:5.1f}% | "
                        f"val loss {va_loss:.4f} acc {va_acc * 100:5.1f}%"
                    )

                    _append_metrics(
                        logs, metrics_fp, step, tr_loss, tr_acc, va_loss, va_acc
                    )

    if not logs["step"]:
        return

    import matplotlib.pyplot as plt

    steps = logs["step"]
    plt.figure()
    plt.plot(steps, logs["train_loss"], label="train_loss")
    plt.plot(steps, logs["val_loss"], label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Grokking? Loss curves")
    plt.savefig(runs_dir / "losses.png")
    plt.show()

    plt.figure()
    plt.plot(steps, logs["train_acc"], label="train_acc")
    plt.plot(steps, logs["val_acc"], label="val_acc")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Grokking? Acc curves")
    plt.savefig(runs_dir / "accuracy.png")
    plt.show()

    torch.save(model.state_dict(), runs_dir / "checkpoint.pt")


def run(
    config: Optional[str] = typer.Option(None, "--config", exists=True, readable=True),
    experiment: str = typer.Option(None, "--experiment"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from"),
) -> None:
    if config is None and resume_from is None:
        raise typer.BadParameter("Either --config or --resume-from is required")

    resolved_config = config
    if resolved_config is None and resume_from is not None:
        resolved_config = str(resume_from / "config.yaml")

    if resolved_config is None:
        raise typer.BadParameter("Either --config or --resume-from is required")

    if experiment is None:
        experiment = Path(resolved_config).stem

    cfg = configs.load_config(resolved_config)
    main(cfg, experiment, resolved_config, resume_from=resume_from)


if __name__ == "__main__":
    typer.run(run)
