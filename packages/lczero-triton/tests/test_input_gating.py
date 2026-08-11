"""CUDA numerical tests for ONNX-layout input gating."""

import pytest
import torch
from lczero_triton.bt4.kernels.input_gating import (
    _autotune_grid,
    _input_gating_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_input_gating_matches_fp32_reference_in_place() -> None:
    """Square-major gates broadcast over batches and permit input/output aliasing."""
    torch.manual_seed(7)
    batch_size = 2
    square_count = 3
    channel_count = 43
    inputs = torch.randn(
        (batch_size, square_count, channel_count),
        dtype=torch.float16,
    )
    multiplier = torch.randn(
        (square_count, channel_count),
        dtype=torch.float16,
    )
    addition = torch.randn(
        (square_count, channel_count),
        dtype=torch.float16,
    )
    expected = (
        inputs.float() * multiplier.float().unsqueeze(0) + addition.float().unsqueeze(0)
    ).half()
    result = inputs.cuda()
    multiplier_cuda = multiplier.cuda()
    addition_cuda = addition.cuda()

    # Tune out of place so candidate launches cannot repeatedly mutate the input.
    _input_gating_kernel[_autotune_grid](
        torch.empty_like(result),
        result,
        multiplier_cuda,
        addition_cuda,
        batch_size,
        square_count,
        channel_count,
    )

    _input_gating_kernel[_autotune_grid](
        result,
        result,
        multiplier_cuda,
        addition_cuda,
        batch_size,
        square_count,
        channel_count,
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=0.0, atol=1e-3)
