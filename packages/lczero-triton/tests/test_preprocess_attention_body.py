"""CUDA numerical tests for attention-body preprocessing."""

import pytest
import torch
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    _autotune_grid,
    _preprocess_attention_body_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_preprocess_attention_body_transposes_and_appends_encoding() -> None:
    """Every token receives channel-major planes followed by dense encoding."""
    batch_size = 2
    square_count = 7
    input_channels = 5
    encoding_channels = 11
    planes = torch.arange(
        batch_size * input_channels * square_count,
        dtype=torch.float16,
    ).reshape(batch_size, input_channels, square_count)
    encoding = -torch.arange(
        batch_size * square_count * encoding_channels,
        dtype=torch.float16,
    ).reshape(batch_size, square_count, encoding_channels)
    output = torch.empty(
        (batch_size, square_count, input_channels + encoding_channels),
        dtype=torch.float16,
        device="cuda",
    )

    _preprocess_attention_body_kernel[_autotune_grid](
        output,
        planes.cuda(),
        encoding.cuda(),
        batch_size,
        square_count,
        input_channels,
        encoding_channels,
    )

    expected = torch.cat((planes.permute(0, 2, 1), encoding), dim=2)
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)
