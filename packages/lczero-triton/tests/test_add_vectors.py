"""CUDA numerical tests for periodic FP16 vector addition."""

import pytest
import torch
from lczero_triton.bt4.kernels.add_vectors import _add_vectors_kernel, _autotune_grid

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_ACTIVATIONS = {"none": 0, "mish": 1, "relu": 2}
_MISH_BRANCH = -0.6


def _activate(values: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the LC0 CUDA activation operation order in FP32."""
    if activation == "mish":
        exponential = torch.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        return torch.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    if activation == "relu":
        return torch.maximum(values, torch.tensor(0.0))
    return values


@pytest.mark.parametrize("activation", ["none", "mish", "relu"])
def test_add_vectors_broadcasts_periodically_in_place_with_tail(
    activation: str,
) -> None:
    """Periodic bias, FP32 arithmetic, activations, and tails match LC0."""
    element_count = 259
    inputs = torch.linspace(-12.0, 12.0, element_count, dtype=torch.float16)
    bias = torch.tensor(
        [-0.75, -0.6001, -0.6, -0.5999, 0.125, 1.5, -2.0],
        dtype=torch.float16,
    )
    repeated_bias = bias.repeat((element_count + len(bias) - 1) // len(bias))[
        :element_count
    ]
    expected = _activate(inputs.float() + repeated_bias.float(), activation).half()
    result = inputs.cuda()
    bias_cuda = bias.cuda()

    # Tune out of place so candidate launches cannot repeatedly mutate the input.
    _add_vectors_kernel[_autotune_grid](
        torch.empty_like(result),
        result,
        bias_cuda,
        element_count,
        len(bias),
        _ACTIVATIONS[activation],
    )

    _add_vectors_kernel[_autotune_grid](
        result,
        result,
        bias_cuda,
        element_count,
        len(bias),
        _ACTIVATIONS[activation],
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=2e-3)
