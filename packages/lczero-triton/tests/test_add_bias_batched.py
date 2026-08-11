"""CUDA numerical tests for batched FP16 bias broadcasting."""

import pytest
import torch
from lczero_triton.bt4.kernels.add_bias_batched import _add_bias_batched_kernel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_ACTIVATIONS = {"none": 0, "mish": 1}
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
    return values


@pytest.mark.parametrize("activation", ["none", "mish"])
def test_add_bias_batched_selects_each_batch_bias_in_place(
    activation: str,
) -> None:
    """Distinct batch biases broadcast across rows with a masked final block."""
    batch_count = 3
    row_count = 5
    channel_count = 37
    inputs = torch.linspace(
        -8.0,
        8.0,
        batch_count * row_count * channel_count,
        dtype=torch.float16,
    ).reshape(batch_count, row_count, channel_count)
    bias = torch.linspace(
        -1.0,
        1.0,
        batch_count * channel_count,
        dtype=torch.float16,
    ).reshape(batch_count, channel_count)
    expected = _activate(inputs.float() + bias.float().unsqueeze(1), activation).half()
    result = inputs.cuda()

    _add_bias_batched_kernel[(3,)](
        result,
        result,
        bias.cuda(),
        batch_count,
        row_count,
        channel_count,
        _ACTIVATIONS[activation],
        256,
        num_warps=8,
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=2e-3)
