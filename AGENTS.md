# Project grokking-lab

## How to install dependencies

Add dependencies in pyproject.toml, and sync it with `uv sync`.

## How to run experiments

Run experiments with `typer run run --experiment <experiment_name> --config configs/<config_name>.yaml`

## How to add files to git lfs

Add files to git lfs with `git lfs`. For example, `git lfs track "*.pt"` if you want to track all .pt files in git lfs.
