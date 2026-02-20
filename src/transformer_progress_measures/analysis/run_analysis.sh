readonly run_dir="runs/transformer_progress_measures/progress_measures_paper/20260218_080650"

#uv run python -m src.transformer_progress_measures.analysis.fft_embedding --run-dir "${run_dir}"

uv run python -m src.transformer_progress_measures.analysis.figure4 --neuron-idx 1 --run-dir "${run_dir}"