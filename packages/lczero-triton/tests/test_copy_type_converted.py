"""CUDA numerical tests for FP16-to-FP32 conversion."""

import pytest
import torch
from lczero_triton.bt4.kernels.copy_type_converted import (
    _copy_type_converted_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_copy_type_converted_matches_torch_with_tail() -> None:
    """FP16 values convert exactly while masking an incomplete final block."""
    element_count = 259
    input_cpu = torch.linspace(-100.0, 100.0, element_count, dtype=torch.float16)
    input_cuda = input_cpu.cuda()
    output = torch.empty(element_count, dtype=torch.float32, device="cuda")

    _copy_type_converted_kernel[(2,)](
        output,
        input_cuda,
        element_count,
        256,
        num_warps=8,
    )

    torch.testing.assert_close(output.cpu(), input_cpu.float(), rtol=0.0, atol=0.0)
