"""CUDA numerical tests for attention-policy gathering."""

import pytest
import torch
from lczero_triton.bt4.kernels.mapping_table import values
from lczero_triton.bt4.kernels.policy_map import _autotune_grid, _policy_map_kernel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_policy_map_gathers_all_outputs_for_adjacent_batches() -> None:
    """The complete ONNX inverse map gathers every policy output per batch."""
    batch_size = 2
    input_element_count = 4288
    mapping_cpu = torch.tensor(values(), dtype=torch.int32)
    inputs = torch.arange(
        batch_size * input_element_count,
        dtype=torch.float16,
    ).reshape(batch_size, input_element_count)
    output = torch.empty(
        (batch_size, len(mapping_cpu)),
        dtype=torch.float16,
        device="cuda",
    )

    _policy_map_kernel[_autotune_grid](
        output,
        inputs.cuda(),
        mapping_cpu.cuda(),
        batch_size,
        input_element_count,
        len(mapping_cpu),
    )

    expected = inputs[:, mapping_cpu.long()]
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)


def test_policy_map_zeros_negative_and_out_of_range_indices() -> None:
    """Malformed gather entries cannot read outside an input policy record."""
    inputs = torch.arange(16, dtype=torch.float16).reshape(2, 8)
    mapping = torch.tensor([0, -1, 7, 8, 3, -4], dtype=torch.int32)
    output = torch.empty((2, 6), dtype=torch.float16, device="cuda")

    _policy_map_kernel[_autotune_grid](
        output,
        inputs.cuda(),
        mapping.cuda(),
        2,
        8,
        6,
    )

    expected = torch.tensor(
        [[0.0, 0.0, 7.0, 0.0, 3.0, 0.0], [8.0, 0.0, 15.0, 0.0, 11.0, 0.0]],
        dtype=torch.float16,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)
