from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from typer import Option, run

try:
    from ..model import Transformer
except ImportError:
    from transformer.model import Transformer


def load_config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def load_model(run_dir: Path, cfg: dict[str, Any] | None = None) -> Transformer:
    if cfg is None:
        cfg = load_config(run_dir)

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d=cfg["d"],
        n_layers=cfg["n_layers"],
        h=cfg.get("h", 4),
    ).to(cfg["device"])
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location=cfg["device"])
    )
    model.eval()
    return model


def main(
    run_dir: Path = Option(
        ..., "--run-dir", exists=True, file_okay=False, dir_okay=True
    ),
    top_k: int = Option(10, "--top-k"),
    save_path: Optional[Path] = Option(None, "--save-path"),
) -> None:
    cfg = load_config(run_dir)
    model = load_model(run_dir, cfg)

    embedding = model.embedding.weight.detach()
    embedding_fft = torch.fft.fft(embedding, dim=0)
    freq_norm = torch.linalg.norm(embedding_fft, dim=1).cpu()

    unique_freq = freq_norm[: (embedding.shape[0] // 2 + 1)]
    values, indices = torch.topk(unique_freq, k=min(top_k, unique_freq.shape[0]))

    print("Top frequencies (k, norm):")
    for idx, val in zip(indices.tolist(), values.tolist(), strict=False):
        print(f"  {idx:3d}  {val:.6f}")

    sorted_vals, _ = torch.sort(unique_freq, descending=True)
    total = sorted_vals.sum()
    top5_ratio = float((sorted_vals[:5].sum() / total) if total > 0 else 0.0)
    print(f"Top5/total ratio: {top5_ratio:.6f}")

    ks = torch.arange(unique_freq.shape[0]).numpy()
    vals = unique_freq.numpy()

    plt.figure(figsize=(7, 4))
    plt.bar(ks, vals, width=0.8)
    plt.xlabel("Frequency k")
    plt.ylabel("L2 norm of FFT component")
    plt.title("Embedding Fourier Components")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = run_dir / "embedding_fft.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    run(main)
