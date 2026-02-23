"""
Reproduce Figure 4 from the paper:
"Progress Measures for Grokking via Mechanistic Interpretability"

Figure 4 contains three subplots:
- Left: Attention Score for Head 0 (from '=' to 'a')
- Center: Activations for Neuron 0
- Right: Norms of Logits in 2D Fourier Basis
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from typer import Option, run

from .utils import load_config, load_model


def compute_attention_and_mlp(
    model: torch.nn.Module, p: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention weights and MLP activations for all (a, b) input pairs.

    Returns:
        attn_weights: (p, p, h, 3, 3) - attention weights for all heads
        mlp_activations: (p, p, 4*d) - MLP activations at final token position
    """
    model.eval()

    # Build all input combinations: [a, b, p] where p is the '=' token (id = p)
    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)

    # Create grid of all (a, b) combinations
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()

    # Input format: [a, b, =] where = token id is p
    eq_token = p
    inputs = torch.stack(
        [aa_flat, bb_flat, torch.full_like(aa_flat, eq_token)], dim=1
    )  # (p*p, 3)

    with torch.no_grad():
        result = model(inputs, return_intermediates=True)

    # attention_weights: list of (p*p, h, 3, 3) per layer
    # mlp_activations: list of (p*p, 3, 4*d) per layer
    attn_weights = result.attention_weights[0]  # First layer: (p*p, h, 3, 3)
    mlp_activations = result.mlp_activations[0]  # First layer: (p*p, 3, 4*d)

    # Reshape to (p, p, ...)
    h = attn_weights.shape[1]
    d_mlp = mlp_activations.shape[-1]

    attn_weights = attn_weights.view(p, p, h, 3, 3)
    mlp_activations = mlp_activations.view(p, p, 3, d_mlp)

    # We only care about MLP activations at the final token position (index 2, '=')
    mlp_activations = mlp_activations[:, :, 2, :]  # (p, p, 4*d)

    return attn_weights, mlp_activations


def compute_logits(model: torch.nn.Module, p: int, device: str) -> torch.Tensor:
    """
    Compute logits for all (a, b) input pairs.

    Returns:
        logits: (p, p, p) - logits for each (a, b) pair
    """
    model.eval()

    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)

    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()

    eq_token = p
    inputs = torch.stack([aa_flat, bb_flat, torch.full_like(aa_flat, eq_token)], dim=1)

    with torch.no_grad():
        logits = model(inputs, return_intermediates=False)  # (p*p, p)

    logits = logits.view(p, p, p)
    return logits


def compute_logits_2d_fourier_norm(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D Fourier transform of logits over (a, b), then take norm over logit axis.

    Args:
        logits: (p, p, p) tensor

    Returns:
        fourier_norms: (p, p) tensor of norms
    """
    p = logits.shape[0]

    # 2D FFT over the first two dimensions (a, b)
    logits_fft = torch.fft.fft2(logits, dim=(0, 1))  # (p, p, p)

    # Take L2 norm over the logit axis (last dimension)
    fourier_norms = torch.linalg.norm(logits_fft, dim=2)  # (p, p)

    return fourier_norms.real


def generate_fourier_labels(p: int, max_freq: int) -> list[str]:
    """
    Generate Fourier frequency labels in the style of the paper.
    Labels: Const, cos 5, sin 9, cos 14, sin 18, ...
    """
    labels = ["Const"]
    for k in range(1, max_freq + 1):
        # In the paper, they label based on approximate frequency values
        # We'll use the actual frequency index k
        labels.append(f"cos {k}")
        labels.append(f"sin {k}")
    return labels


def find_active_neuron(mlp_activations: torch.Tensor) -> int:
    """Find the neuron with highest average activation."""
    # mlp_activations: (p, p, n_neurons)
    mean_act = mlp_activations.mean(dim=(0, 1))  # (n_neurons,)
    return int(mean_act.argmax().item())


def main(
    run_dir: Path = Option(
        ..., "--run-dir", exists=True, file_okay=False, dir_okay=True
    ),
    head_idx: int = Option(0, "--head-idx", help="Attention head index to plot"),
    neuron_idx: int = Option(
        -1, "--neuron-idx", help="MLP neuron index to plot (-1 for auto-select)"
    ),
    save_path: Optional[Path] = Option(None, "--save-path"),
) -> None:
    cfg = load_config(run_dir)
    model = load_model(run_dir, cfg)
    p = cfg["p"]
    device = cfg["device"]

    print(f"Computing attention weights and MLP activations for p={p}...")
    attn_weights, mlp_activations = compute_attention_and_mlp(model, p, device)

    print("Computing logits...")
    logits = compute_logits(model, p, device)

    print("Computing 2D Fourier transform of logits...")
    logits_fourier_norm = compute_logits_2d_fourier_norm(logits)

    # Auto-select active neuron if not specified
    if neuron_idx < 0:
        neuron_idx = find_active_neuron(mlp_activations)
        print(f"Auto-selected neuron {neuron_idx} (highest average activation)")

    # Extract data for plotting
    # Attention: from '=' (position 2) to 'a' (position 0), head 0
    attn_eq_to_a = attn_weights[:, :, head_idx, 2, 0].cpu().numpy()  # (p, p)

    # MLP neuron activation
    neuron_act = mlp_activations[:, :, neuron_idx].cpu().numpy()  # (p, p)

    # Fourier norms - truncate to unique frequencies (p // 2 + 1)
    max_freq = p // 2 + 1
    logits_fourier_norm_np = logits_fourier_norm[:max_freq, :max_freq].cpu().numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Attention Score for Head 0
    im0 = axes[0].imshow(
        attn_eq_to_a.T,  # Transpose so a is x-axis, b is y-axis
        origin="upper",
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("b")
    axes[0].set_title(f"Attention Score for Head {head_idx}")
    axes[0].invert_yaxis()
    plt.colorbar(im0, ax=axes[0])

    # Center: Activations for Neuron 0
    im1 = axes[1].imshow(
        neuron_act.T,  # Transpose so a is x-axis, b is y-axis
        origin="upper",
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("b")
    axes[1].set_title(f"Activations for Neuron {neuron_idx}")
    axes[1].invert_yaxis()
    plt.colorbar(im1, ax=axes[1])

    # Right: Norms of Logits in 2D Fourier Basis
    # Use a colormap with white background for low values
    im2 = axes[2].imshow(
        logits_fourier_norm_np.T,
        origin="upper",
        aspect="auto",
        cmap="hot_r",  # White background for low values
    )
    axes[2].set_title("Norms of Logits in 2D Fourier Basis")

    # Create Fourier labels for x and y axes
    # Show only a subset of labels to avoid clutter
    n_labels = min(15, max_freq)
    step = max(1, max_freq // n_labels)
    tick_positions = list(range(0, max_freq, step))

    # Generate labels in paper style
    x_labels = []
    y_labels = []
    for i in tick_positions:
        if i == 0:
            x_labels.append("Const")
            y_labels.append("Const")
        else:
            x_labels.append(f"cos {i}" if i % 2 == 0 else f"sin {i}")
            y_labels.append(f"cos {i}" if i % 2 == 0 else f"sin {i}")

    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(tick_positions)
    axes[2].set_yticklabels(y_labels, fontsize=8)
    axes[2].set_xlabel("x Component")
    axes[2].set_ylabel("y Component")
    axes[2].invert_yaxis()
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path is None:
        save_path = run_dir / "figure4.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {save_path}")

    # Print some statistics
    print(f"\nStatistics:")
    print(
        f"  Head {head_idx} attention score range: "
        f"[{attn_eq_to_a.min():.4f}, {attn_eq_to_a.max():.4f}]"
    )
    print(
        f"  Neuron {neuron_idx} activation range: "
        f"[{neuron_act.min():.4f}, {neuron_act.max():.4f}]"
    )
    print(
        f"  Fourier norm range: "
        f"[{logits_fourier_norm_np.min():.4f}, {logits_fourier_norm_np.max():.4f}]"
    )

    # Print top active neurons
    mean_acts = mlp_activations.mean(dim=(0, 1)).cpu()
    top_neurons = torch.topk(mean_acts, k=5)
    print(f"\nTop 5 most active neurons: {top_neurons.indices.tolist()}")


if __name__ == "__main__":
    run(main)
