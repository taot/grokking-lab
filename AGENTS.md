# Project grokking-lab

## Coding style

1. Always add python type hints when writing functions, methods or classes.

## How to install dependencies

Add dependencies in pyproject.toml, and sync it with `uv sync`.

## How to run experiments

Run experiments with `uv run --experiment <experiment_name> --config <path_to_config>.yaml`.

## How to run python code

The python environment is installed by `uv`. Run python code with `uv run <python_file.py>`.

## git lfs

### How to add files to git lfs

Install git lfs to the repo with `git lfs install`. Only need to do this once.

Add files to git lfs with `git lfs`. For example, `git lfs track "*.pt"` if you want to track all .pt files in git lfs.

### How to check if files are tracked by git lfs

Use `git lfs track` to see tracked patterns.
Or use `git lfs ls-files` to see files tracked by git lfs.
