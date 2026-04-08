#!/usr/bin/env python3
"""Pick the best available torch device: CUDA, then MPS (Apple Silicon), else CPU."""

import torch


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except (AttributeError, NotImplementedError):
        return False


def device_name(device: torch.device) -> str:
    """Short label for logs."""
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"
