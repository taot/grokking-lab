from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import yaml


@dataclass(frozen=True)
class RunResult:
    run_id: str
    train_frac: float
    grok_step: Optional[int]
    last_step: int
    max_val_acc: float


def _read_train_frac(config_path: Path) -> float:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict) or "train_frac" not in data:
        raise ValueError(f"Missing train_frac in {config_path}")
    return float(data["train_frac"])


def _read_grok_step(
    metrics_path: Path, threshold: float
) -> tuple[Optional[int], int, float]:
    grok_step: Optional[int] = None
    last_step: int = 0
    max_val_acc: float = float("-inf")

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            step = int(row.get("step", 0))
            val_acc = float(row.get("val_acc", float("nan")))

            last_step = step
            if val_acc == val_acc:  # not NaN
                max_val_acc = max(max_val_acc, val_acc)
                if grok_step is None and val_acc > threshold:
                    grok_step = step

    if max_val_acc == float("-inf"):
        max_val_acc = float("nan")

    return grok_step, last_step, max_val_acc


def collect_results(runs_dir: Path, threshold: float) -> list[RunResult]:
    results: list[RunResult] = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue

        train_frac = _read_train_frac(config_path)
        grok_step, last_step, max_val_acc = _read_grok_step(
            metrics_path, threshold=threshold
        )
        results.append(
            RunResult(
                run_id=run_dir.name,
                train_frac=train_frac,
                grok_step=grok_step,
                last_step=last_step,
                max_val_acc=max_val_acc,
            )
        )

    results.sort(key=lambda r: (r.train_frac, r.run_id))
    return results


def _jittered_xs(results: list[RunResult], jitter: float = 0.002) -> dict[str, float]:
    by_tf: dict[float, list[RunResult]] = {}
    for r in results:
        by_tf.setdefault(r.train_frac, []).append(r)

    xs: dict[str, float] = {}
    for tf, group in by_tf.items():
        n = len(group)
        for i, r in enumerate(group):
            if n == 1:
                xs[r.run_id] = tf
            else:
                offset = (i - (n - 1) / 2.0) * jitter
                xs[r.run_id] = tf + offset
    return xs


def plot(results: list[RunResult], out_path: Path, threshold: float) -> None:
    if not results:
        raise ValueError("No runs found")

    plotted = [r for r in results if r.grok_step is not None]
    if not plotted:
        raise ValueError("No grokked runs found")

    xs_by_id = _jittered_xs(plotted)

    grok_x: list[float] = []
    grok_y: list[int] = []
    grok_labels: list[str] = []

    for r in plotted:
        x = xs_by_id[r.run_id]
        assert r.grok_step is not None
        grok_x.append(x)
        grok_y.append(r.grok_step)
        grok_labels.append(r.run_id)

    plt.figure(figsize=(8.5, 4.8), dpi=160)
    plt.scatter(
        grok_x, grok_y, s=44, c="#1f77b4", label=f"grok (val_acc > {threshold})"
    )

    plt.xlabel("train_frac")
    plt.ylabel("step")
    plt.title("train_frac vs grokking step")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_csv(results: list[RunResult], out_path: Path) -> None:
    lines = ["run_id,train_frac,grok_step,last_step,max_val_acc"]
    for r in results:
        grok_step = "" if r.grok_step is None else str(r.grok_step)
        max_val_acc = "" if r.max_val_acc != r.max_val_acc else f"{r.max_val_acc:.6f}"
        lines.append(
            f"{r.run_id},{r.train_frac:.6f},{grok_step},{r.last_step},{max_val_acc}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runs_dir = Path("runs/transformer/train_frac")
    threshold = 0.9

    results = collect_results(runs_dir=runs_dir, threshold=threshold)
    write_csv(results, out_path=Path("tmp/train_frac_vs_grokking_step.csv"))
    plot(
        results,
        out_path=Path("tmp/train_frac_vs_grokking_step.png"),
        threshold=threshold,
    )


if __name__ == "__main__":
    main()
