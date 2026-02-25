from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from typer import Option, run

try:
    from ..model import Transformer
    from .utils import load_config
except ImportError:
    from transformer_progress_measures_ablation.analysis.utils import load_config
    from transformer_progress_measures_ablation.model import Transformer


def _load_model_from_checkpoint(
    cfg: dict[str, Any], checkpoint_path: Path
) -> Transformer:
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg["device"]))
    model.eval()
    return model


def _extract_step_from_checkpoint(path: Path) -> int:
    stem = path.stem
    prefix = "checkpoint_step_"
    if not stem.startswith(prefix):
        raise ValueError(f"Not a periodic checkpoint file: {path.name}")
    return int(stem[len(prefix) :])


def _plot_embedding_fft(
    model: Transformer,
    p: int,
    top_k: int,
    output_path: Path,
    label: str,
) -> None:
    embedding = model.embedding.weight.detach()
    embedding_fft = torch.fft.fft(embedding, dim=0)
    freq_norm = torch.linalg.norm(embedding_fft, dim=1).cpu()

    unique_freq = freq_norm[: (p // 2 + 1)]
    values, indices = torch.topk(unique_freq, k=min(top_k, unique_freq.shape[0]))

    print(f"[{label}] Top frequencies (k, norm):")
    for idx, val in zip(indices.tolist(), values.tolist(), strict=False):
        print(f"  {idx:3d}  {val:.6f}")

    sorted_vals, _ = torch.sort(unique_freq, descending=True)
    top5_ratio = float(sorted_vals[:5].sum() / sorted_vals.sum())
    print(f"[{label}] Top5/total ratio: {top5_ratio:.6f}")

    ks = torch.arange(unique_freq.shape[0]).numpy()
    vals = unique_freq.numpy()

    plt.figure(figsize=(7, 4))
    plt.bar(ks, vals, width=0.8)
    plt.xlabel("Frequency k")
    plt.ylabel("L2 norm of FFT component")
    plt.title(f"Embedding Fourier Components ({label})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")


def main(
    run_dir: Path = Option(
        ..., "--run-dir", exists=True, file_okay=False, dir_okay=True
    ),
    top_k: int = Option(10, "--top-k"),
    save_path: Optional[Path] = Option(None, "--save-path"),
) -> None:
    cfg = load_config(run_dir)

    periodic_checkpoints = sorted(
        run_dir.glob("checkpoint_step_*.pt"),
        key=_extract_step_from_checkpoint,
    )
    final_checkpoint = run_dir / "checkpoint.pt"

    if not periodic_checkpoints and not final_checkpoint.exists():
        print(
            "No checkpoints found. Expected checkpoint_step_*.pt or checkpoint.pt under run directory."
        )
        return

    output_dir = save_path.parent if save_path is not None else run_dir

    for checkpoint_path in periodic_checkpoints:
        step = _extract_step_from_checkpoint(checkpoint_path)
        model = _load_model_from_checkpoint(cfg, checkpoint_path)
        output_path = output_dir / f"embedding_fft_step_{step}.png"
        _plot_embedding_fft(model, int(cfg["p"]), top_k, output_path, f"step {step}")

    if final_checkpoint.exists():
        model = _load_model_from_checkpoint(cfg, final_checkpoint)
        output_path = output_dir / "embedding_fft.png"
        _plot_embedding_fft(model, int(cfg["p"]), top_k, output_path, "final")


if __name__ == "__main__":
    run(main)
