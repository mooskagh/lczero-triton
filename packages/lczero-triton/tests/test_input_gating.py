"""CUDA numerical tests for ONNX-layout input gating."""

import pytest
import torch
from lczero_triton.bt4.kernels.input_gating import _input_gating_kernel

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

    _input_gating_kernel[(2,)](
        result,
        result,
        multiplier.cuda(),
        addition.cuda(),
        batch_size,
        square_count,
        channel_count,
        256,
        num_warps=8,
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=0.0, atol=1e-3)
