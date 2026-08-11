"""Shared launch candidates and target validation for BT4 autotuning."""

import torch
import triton

_ELEMENTWISE_CONFIGURATIONS = (
    (64, 1),
    (128, 2),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 8),
)
_PREPROCESS_CONFIGURATIONS = (
    (128, 2),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
)


def elementwise_configs() -> list[triton.Config]:
    """Return independent Triton configurations for a flat elementwise kernel."""
    return [
        triton.Config({"block_size": block_size}, num_warps=num_warps)
        for block_size, num_warps in _ELEMENTWISE_CONFIGURATIONS
    ]


def preprocess_configs() -> list[triton.Config]:
    """Return channel-tile candidates for attention-input preprocessing."""
    return [
        triton.Config({"block_size": block_size}, num_warps=num_warps)
        for block_size, num_warps in _PREPROCESS_CONFIGURATIONS
    ]


def validate_active_architecture(architecture: int) -> None:
    """Require autotuning to run on the requested CUDA architecture."""
    if not torch.cuda.is_available():
        message = "kernel autotuning requires an available CUDA device"
        raise RuntimeError(message)
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    active_architecture = major * 10 + minor
    if active_architecture != architecture:
        message = (
            f"kernel requested sm_{architecture}, but the active device is "
            f"sm_{active_architecture}"
        )
        raise ValueError(message)
