"""CUDA numerical tests for packed plane expansion."""

import pytest
import torch
from lczero_triton.bt4.kernels.expand_planes import (
    _autotune_grid,
    _expand_planes_kernel,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_expand_planes_matches_mask_bits_and_masks_plane_tail() -> None:
    """Bits zero and 63 map directly to squares across a partial block."""
    mask_values = [1, 1 << 63, (1 << 64) - 1, 0, (1 << 7) | (1 << 41)]
    plane_values = torch.tensor(
        [1.25, -2.5, 0.33333334, 9.0, 65520.0],
        dtype=torch.float32,
    )
    masks = torch.tensor(mask_values, dtype=torch.uint64, device="cuda")
    values = plane_values.cuda()
    output = torch.empty((5, 64), dtype=torch.float16, device="cuda")

    _expand_planes_kernel[_autotune_grid](output, masks, values, 5, 64)

    expected = torch.zeros((5, 64), dtype=torch.float16)
    rounded_values = plane_values.half()
    for plane, mask in enumerate(mask_values):
        for square in range(64):
            if mask & (1 << square):
                expected[plane, square] = rounded_values[plane]
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)
