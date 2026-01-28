import torch

def pytest_configure(config):
    torch.set_printoptions(
        precision=6,
        sci_mode=False,
        linewidth=200,
        threshold=10_000,
    )