from pathlib import Path

import torch
import yaml
from typer import Option, run

from .model import Transformer


def main(
    run_dir: Path = Option(
        ..., "--run-dir", exists=True, file_okay=False, dir_okay=True
    ),
    top_k: int = Option(10, "--top-k"),
) -> None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d=cfg["d"],
        n_layers=cfg["n_layers"],
        h=cfg["h"],
        max_seq_len=3,
    ).to(cfg["device"])
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location=cfg["device"])
    )
    model.eval()

    embedding = model.embedding.weight.detach()
    embedding_fft = torch.fft.fft(embedding, dim=0)
    freq_norm = torch.linalg.norm(embedding_fft, dim=1).cpu()

    unique_freq = freq_norm[: (cfg["p"] // 2 + 1)]
    values, indices = torch.topk(unique_freq, k=min(top_k, unique_freq.shape[0]))

    print("Top frequencies (k, norm):")
    for idx, val in zip(indices.tolist(), values.tolist(), strict=False):
        print(f"  {idx:3d}  {val:.6f}")

    sorted_vals, _ = torch.sort(unique_freq, descending=True)
    top5_ratio = float(sorted_vals[:5].sum() / sorted_vals.sum())
    print(f"Top5/total ratio: {top5_ratio:.6f}")


if __name__ == "__main__":
    run(main)
