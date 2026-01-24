from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot loss and accuracy curves from metrics.jsonl into one figure."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/train_frac/20260122_085304"),
        help="Run directory containing metrics.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/train_frac_20260122_085304_loss_accuracy.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    rows = _read_metrics_jsonl(metrics_path)
    if not rows:
        raise RuntimeError(f"No metrics found in {metrics_path}")

    steps = [int(r["step"]) for r in rows if "step" in r]
    train_loss = [float(r.get("train_loss", "nan")) for r in rows]
    val_loss = [float(r.get("val_loss", "nan")) for r in rows]
    train_acc = [float(r.get("train_acc", "nan")) for r in rows]
    val_acc = [float(r.get("val_acc", "nan")) for r in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4.5), sharex=True, constrained_layout=True
    )

    ax_loss.plot(steps, train_loss, label="train", linewidth=1.5)
    ax_loss.plot(steps, val_loss, label="val", linewidth=1.5)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(loc="best")

    ax_acc.plot(steps, train_acc, label="train", linewidth=1.5)
    ax_acc.plot(steps, val_acc, label="val", linewidth=1.5)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(loc="best")

    fig.suptitle(f"{args.run_dir.as_posix()}  (loss: log scale)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
