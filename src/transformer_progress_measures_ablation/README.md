# Transformer Progress Measures Ablation

This directory runs controlled ablation experiments on top of the progress-measures setup to test which structural choices most affect periodic representations.

The ablations focus on three factors:
- whether decoder blocks use LayerNorm,
- whether positional encoding is learned or fixed sinusoidal,
- whether the task input format uses `[a, b, =]` or `[a, b]`.

The intent is to keep the optimization setup aligned while changing one structural factor at a time, then compare how strongly periodic/Fourier-like internal structure appears.

Run training with:
- `uv run src/transformer_progress_measures_ablation/main.py --experiment baseline --config src/transformer_progress_measures_ablation/configs/baseline.yaml`
- `uv run src/transformer_progress_measures_ablation/main.py --experiment ablation_layernorm --config src/transformer_progress_measures_ablation/configs/ablation_layernorm.yaml`
- `uv run src/transformer_progress_measures_ablation/main.py --experiment ablation_sinusoidal_pe --config src/transformer_progress_measures_ablation/configs/ablation_sinusoidal_pe.yaml`
- `uv run src/transformer_progress_measures_ablation/main.py --experiment ablation_no_eq_token --config src/transformer_progress_measures_ablation/configs/ablation_no_eq_token.yaml`

Score periodic structure from a run directory with:
- `uv run src/transformer_progress_measures_ablation/analysis/periodicity_score.py --run-dir <run_dir>`

Compare latest runs across all ablations with:
- `uv run src/transformer_progress_measures_ablation/analysis/compare_runs.py`
