from pathlib import Path

from typer import Option, run

try:
    from .periodicity_score import (
        compute_activation_periodicity_score,
        compute_embedding_periodicity_score,
    )
    from .utils import load_config, load_last_metrics, load_model
except ImportError:
    from transformer_progress_measures_ablation.analysis.periodicity_score import (
        compute_activation_periodicity_score,
        compute_embedding_periodicity_score,
    )
    from transformer_progress_measures_ablation.analysis.utils import (
        load_config,
        load_last_metrics,
        load_model,
    )


def _find_latest_run(experiment_dir: Path) -> Path | None:
    if not experiment_dir.exists():
        return None
    candidates = [p for p in experiment_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def main(
    runs_root: Path = Option(
        Path("runs/transformer_progress_measures_ablation"),
        "--runs-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    experiment_names = sorted(
        [entry.name for entry in runs_root.iterdir() if entry.is_dir()]
    )

    if not experiment_names:
        print("No experiment directories found.")
        return

    rows: list[tuple[str, float, float, float, float]] = []
    print("Using latest run under each experiment directory:")
    for exp_name in experiment_names:
        run_dir = _find_latest_run(runs_root / exp_name)
        if run_dir is None:
            print(f"- {exp_name}: missing")
            continue
        print(f"- {exp_name}: {run_dir}")

        cfg = load_config(run_dir)
        model = load_model(run_dir, cfg)

        emb = compute_embedding_periodicity_score(model, int(cfg["p"]))
        act, _ = compute_activation_periodicity_score(
            model=model,
            p=int(cfg["p"]),
            use_eq_token=bool(cfg["use_eq_token"]),
            device=str(cfg["device"]),
        )
        combined = 0.5 * emb + 0.5 * act

        val_acc = -1.0
        last = load_last_metrics(run_dir)
        if last is not None:
            val_acc = float(last["val_acc"])

        rows.append((exp_name, emb, act, combined, val_acc))

    if not rows:
        print("No ablation runs found.")
        return

    rows.sort(key=lambda x: x[3], reverse=True)
    print(
        "\nRanked by combined periodicity score (higher = stronger periodic structure):"
    )
    print("experiment\temb_score\tact_score\tcombined\tfinal_val_acc")
    for exp_name, emb, act, combined, val_acc in rows:
        print(f"{exp_name}\t{emb:.6f}\t{act:.6f}\t{combined:.6f}\t{val_acc:.6f}")


if __name__ == "__main__":
    run(main)
