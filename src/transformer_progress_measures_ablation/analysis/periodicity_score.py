from pathlib import Path
from typing import TYPE_CHECKING

import torch
from typer import Option, run

try:
    from .utils import load_config, load_last_metrics, load_model
except ImportError:
    from transformer_progress_measures_ablation.analysis.utils import (
        load_config,
        load_last_metrics,
        load_model,
    )

if TYPE_CHECKING:
    from transformer_progress_measures_ablation.model import Transformer


def make_inputs(p: int, use_eq_token: bool, device: str) -> torch.Tensor:
    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()

    if use_eq_token:
        eq_token = p
        return torch.stack(
            [aa_flat, bb_flat, torch.full_like(aa_flat, eq_token)],
            dim=1,
        )
    return torch.stack([aa_flat, bb_flat], dim=1)


def topk_ratio_1d(values: torch.Tensor, top_k: int, drop_dc: bool = True) -> float:
    x = values.clone()
    if drop_dc and x.numel() > 0:
        x[0] = 0
    total = float(x.sum().item())
    if total <= 0:
        return 0.0
    k = min(top_k, x.numel())
    top_vals = torch.topk(x, k=k).values
    return float(top_vals.sum().item() / total)


def topk_ratio_2d(values: torch.Tensor, top_k: int, drop_dc: bool = True) -> float:
    x = values.clone()
    if drop_dc and x.numel() > 0:
        x[0, 0] = 0
    flat = x.flatten()
    total = float(flat.sum().item())
    if total <= 0:
        return 0.0
    k = min(top_k, flat.numel())
    top_vals = torch.topk(flat, k=k).values
    return float(top_vals.sum().item() / total)


@torch.no_grad()
def compute_embedding_periodicity_score(model: "Transformer", p: int) -> float:
    embedding = model.embedding.weight.detach()[:p]
    embedding_fft = torch.fft.fft(embedding, dim=0)
    freq_norm = torch.linalg.norm(embedding_fft, dim=1)
    unique_freq = freq_norm[: (p // 2 + 1)]
    return topk_ratio_1d(unique_freq, top_k=5, drop_dc=True)


@torch.no_grad()
def compute_activation_periodicity_score(
    model: "Transformer",
    p: int,
    use_eq_token: bool,
    device: str,
) -> tuple[float, int]:
    inputs = make_inputs(p=p, use_eq_token=use_eq_token, device=device)
    result = model(inputs, return_intermediates=True)
    if result.mlp_activations is None or len(result.mlp_activations) == 0:
        raise RuntimeError("Model did not return MLP activations")

    seq_idx = 2 if use_eq_token else 1
    mlp_activations = result.mlp_activations[0]
    mlp_last = mlp_activations[:, seq_idx, :].view(p, p, -1)
    mean_act = mlp_last.mean(dim=(0, 1))
    neuron_idx = int(mean_act.argmax().item())

    neuron_map = mlp_last[:, :, neuron_idx]
    fft2 = torch.fft.fft2(neuron_map, dim=(0, 1))
    spectral = torch.abs(fft2)
    score = topk_ratio_2d(spectral, top_k=8, drop_dc=True)
    return score, neuron_idx


def main(
    run_dir: Path = Option(
        ..., "--run-dir", exists=True, file_okay=False, dir_okay=True
    ),
) -> None:
    cfg = load_config(run_dir)
    model = load_model(run_dir, cfg)

    p = int(cfg["p"])
    use_eq_token = bool(cfg["use_eq_token"])
    device = str(cfg["device"])

    emb_score = compute_embedding_periodicity_score(model, p)
    act_score, neuron_idx = compute_activation_periodicity_score(
        model=model,
        p=p,
        use_eq_token=use_eq_token,
        device=device,
    )

    combined = 0.5 * emb_score + 0.5 * act_score
    last_metrics = load_last_metrics(run_dir)

    print(f"run_dir: {run_dir}")
    print(f"use_layernorm: {cfg['use_layernorm']}")
    print(f"use_sinusoidal_pe: {cfg['use_sinusoidal_pe']}")
    print(f"use_eq_token: {cfg['use_eq_token']}")
    if last_metrics is not None:
        print(f"final_val_acc: {last_metrics['val_acc']:.6f}")
        print(f"final_step: {int(last_metrics['step'])}")
    print(f"embedding_periodicity_score: {emb_score:.6f}")
    print(f"activation_periodicity_score: {act_score:.6f}")
    print(f"active_neuron_idx: {neuron_idx}")
    print(f"combined_periodicity_score: {combined:.6f}")


if __name__ == "__main__":
    run(main)
