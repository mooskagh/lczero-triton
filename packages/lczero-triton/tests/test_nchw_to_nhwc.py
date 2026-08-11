"""CUDA numerical tests for NCHW-to-NHWC extraction."""

import pytest
import torch
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    _autotune_grid,
    _nchw_to_nhwc_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_nchw_to_nhwc_extracts_first_channels_with_tail() -> None:
    """The layout conversion drops later channels and masks its final vector."""
    batch_size = 2
    input_channels = 7
    output_channels = 5
    height = 3
    width = 5
    input_cpu = torch.arange(
        batch_size * input_channels * height * width,
        dtype=torch.float16,
    ).reshape(batch_size, input_channels, height, width)
    output = torch.empty(
        (batch_size, height, width, output_channels),
        dtype=torch.float16,
        device="cuda",
    )

    _nchw_to_nhwc_kernel[_autotune_grid](
        output,
        input_cpu.cuda(),
        batch_size,
        input_channels,
        output_channels,
        height,
        width,
    )

    expected = input_cpu[:, :output_channels].permute(0, 2, 3, 1).contiguous()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)
