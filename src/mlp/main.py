import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

from . import configs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(p: int):
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
    def __init__(self, p: int, embed_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.emb = nn.Embedding(p, embed_dim)
        layers = []
        in_dim = 2 * embed_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, p)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B,2]
        a = self.emb(x[:, 0])
        b = self.emb(x[:, 1])
        h = torch.cat([a, b], dim=-1)
        return self.net(h)


@torch.no_grad()
def evaluate(model, xs, ys, device):
    model.eval()
    logits = model(xs.to(device))
    loss = F.cross_entropy(logits, ys.to(device)).item()
    pred = logits.argmax(dim=-1).cpu()
    acc = (pred == ys).float().mean().item()
    return loss, acc


def main(cfg: configs.Config, experiment: str, config_path: str):
    set_seed(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path("runs") / experiment / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

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
    metrics_fp = open(metrics_file, "w")

    logs = {
        "step": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for step in range(1, cfg.steps + 1):
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
            logs["step"].append(step)
            logs["train_loss"].append(tr_loss)
            logs["train_acc"].append(tr_acc)
            logs["val_loss"].append(va_loss)
            logs["val_acc"].append(va_acc)

            print(
                f"step {step:6d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc * 100:5.1f}% | "
                f"val loss {va_loss:.4f} acc {va_acc * 100:5.1f}%"
            )

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

    metrics_fp.close()

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
    config: str = typer.Option(..., "--config", exists=True, readable=True),
    experiment: str = typer.Option(None, "--experiment"),
):
    if experiment is None:
        experiment = Path(config).stem
    cfg = configs.load_config(config)
    main(cfg, experiment, config)


if __name__ == "__main__":
    typer.run(run)
