import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, cast

import numpy as np
import torch
import torch.nn.functional as F
import typer
import yaml

try:
    from . import configs
    from .model import Transformer
except ImportError:
    from transformer_progress_measures_ablation import configs
    from transformer_progress_measures_ablation.model import Transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(p: int, use_eq_token: bool) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[int]] = []
    ys: list[int] = []
    eq_token = p

    for a in range(p):
        for b in range(p):
            if use_eq_token:
                xs.append([a, b, eq_token])
            else:
                xs.append([a, b])
            ys.append((a + b) % p)

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


@torch.no_grad()
def evaluate(
    model: Transformer,
    xs: torch.Tensor,
    ys: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    model.eval()

    logits = cast(torch.Tensor, model.forward(xs.to(device)))
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
    experiment_parts = Path(experiment).parts
    if len(experiment_parts) > 1:
        experiment_path = Path(*experiment_parts)
        if experiment_parts[0] == "runs":
            experiment_path = Path(*experiment_parts[1:])
        runs_parent = Path("runs") / experiment_path
    else:
        runs_parent = (
            Path("runs") / "transformer_progress_measures_ablation" / experiment
        )

    runs_dir = runs_parent / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _validate_vocab_size(cfg: configs.Config) -> None:
    expected_vocab_size = cfg.p + 1 if cfg.use_eq_token else cfg.p
    if cfg.vocab_size != expected_vocab_size:
        token_desc = (
            "p + 1 (with '=' token)" if cfg.use_eq_token else "p (no '=' token)"
        )
        raise ValueError(
            "Invalid vocab_size for current task format: "
            f"expected {token_desc}, got p={cfg.p}, vocab_size={cfg.vocab_size}"
        )


def main(
    cfg: configs.Config,
    experiment: str,
    config_path: str,
    resume_from: Optional[Path] = None,
) -> None:
    start_time = datetime.now()
    runs_dir: Optional[Path] = None
    max_seq_len = 3 if cfg.use_eq_token else 2

    _validate_vocab_size(cfg)

    try:
        set_seed(cfg.seed)

        if resume_from is not None:
            checkpoint_path = resume_from / "checkpoint.pt"
            metrics_path = resume_from / "metrics.jsonl"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        runs_dir = _create_runs_dir(experiment)
        if resume_from is not None:
            shutil.copy(config_path, runs_dir / "config.orig.yaml")
        with (runs_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(cfg), handle, sort_keys=False)

        xs, ys = make_dataset(cfg.p, cfg.use_eq_token)
        n = xs.shape[0]
        idx = torch.randperm(n)
        n_train = int(n * cfg.train_frac)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        x_tr, y_tr = xs[train_idx], ys[train_idx]
        x_va, y_va = xs[val_idx], ys[val_idx]

        model = Transformer(
            vocab_size=cfg.vocab_size,
            output_size=cfg.p,
            d=cfg.d,
            n_layers=cfg.n_layers,
            h=cfg.h,
            max_seq_len=max_seq_len,
            use_layernorm=cfg.use_layernorm,
            use_sinusoidal_pe=cfg.use_sinusoidal_pe,
        ).to(device=cfg.device)
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

                    if cfg.batch_size >= x_tr.shape[0]:
                        xb = x_tr.to(cfg.device)
                        yb = y_tr.to(cfg.device)
                    else:
                        bidx = torch.randint(0, x_tr.shape[0], (cfg.batch_size,))
                        xb = x_tr[bidx].to(cfg.device)
                        yb = y_tr[bidx].to(cfg.device)

                    logits = cast(torch.Tensor, model.forward(xb))
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
        plt.title("Ablation Progress Measures Loss Curves")
        plt.savefig(runs_dir / "losses.png")
        plt.close()

        plt.figure()
        plt.plot(steps, logs["train_acc"], label="train_acc")
        plt.plot(steps, logs["val_acc"], label="val_acc")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Ablation Progress Measures Accuracy Curves")
        plt.savefig(runs_dir / "accuracy.png")
        plt.close()

        torch.save(model.state_dict(), runs_dir / "checkpoint.pt")
    finally:
        end_time = datetime.now()
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if runs_dir is not None:
            print(f"Saved to:   {runs_dir.resolve()}")


def run(
    config: Optional[str] = typer.Option(None, "--config", exists=True, readable=True),
    experiment: Optional[str] = typer.Option(None, "--experiment"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from"),
    sets: list[str] = typer.Option(
        [],
        "--set",
        help="Override config fields: key=value (repeatable). Overrides --override-json.",
    ),
    override_json: Optional[str] = typer.Option(
        None,
        "--override-json",
        help='JSON object of config overrides (e.g. "{\\"lr\\":3e-4,\\"steps\\":5000}").',
    ),
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

    overrides: dict[str, Any] = {}
    if override_json is not None:
        try:
            parsed = json.loads(override_json)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid --override-json: {e}") from e
        if not isinstance(parsed, dict):
            raise typer.BadParameter("--override-json must be a JSON object")
        overrides.update(parsed)

    for item in sets:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid --set '{item}': expected key=value (use --set key=value)"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(
                f"Invalid --set '{item}': key cannot be empty (use --set key=value)"
            )
        overrides[key] = value

    allowed_keys: Optional[set[str]] = None
    if resume_from is not None and overrides:
        allowed_keys = {
            "train_frac",
            "batch_size",
            "lr",
            "weight_decay",
            "steps",
            "eval_every",
            "seed",
            "device",
            "vocab_size",
        }

    try:
        cfg = configs.apply_overrides(cfg, overrides, allowed_keys=allowed_keys)
    except (ValueError, TypeError) as e:
        raise typer.BadParameter(str(e)) from e

    main(cfg, experiment, resolved_config, resume_from=resume_from)


if __name__ == "__main__":
    typer.run(run)
