from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from typer import Option, run

from .utils import load_config, load_model


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

    unique_freq = freq_norm[: (cfg["p"] // 2 + 1)]
    # unique_freq = freq_norm
    values, indices = torch.topk(unique_freq, k=min(top_k, unique_freq.shape[0]))

    print("Top frequencies (k, norm):")
    for idx, val in zip(indices.tolist(), values.tolist(), strict=False):
        print(f"  {idx:3d}  {val:.6f}")

    sorted_vals, _ = torch.sort(unique_freq, descending=True)
    top5_ratio = float(sorted_vals[:5].sum() / sorted_vals.sum())
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
