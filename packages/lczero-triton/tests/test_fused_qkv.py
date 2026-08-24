"""CUDA numerical and artifact tests for separate-pointer QKV fusion."""

import pytest
import torch
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels.fused_qkv import (
    FusedQkvBiasSpecialization,
    FusedQkvProjectionSpecialization,
    _bias_artifact_grid,
    _bias_autotune_grid,
    _fused_qkv_bias_kernel,
    _fused_qkv_projection_kernel,
    _projection_artifact_grid,
    _projection_autotune_grid,
    compile_fused_qkv_bias,
    compile_fused_qkv_projection,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
_FP16_ATOL = 2e-2
_FP16_RTOL = 1e-2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def test_fused_qkv_projection_matches_three_torch_matmuls() -> None:
    """Independent output widths and non-divisible tiles remain numerically valid."""
    torch.manual_seed(73)
    m, k = 7, 37
    query_width, key_width, value_width = 11, 13, 9
    activations = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.05
    weights_q = torch.randn((k, query_width), dtype=torch.float16, device="cuda") * 0.05
    weights_k = torch.randn((k, key_width), dtype=torch.float16, device="cuda") * 0.05
    weights_v = torch.randn((k, value_width), dtype=torch.float16, device="cuda") * 0.05
    bias_q = torch.randn(query_width, dtype=torch.float16, device="cuda") * 0.05
    bias_k = torch.randn(key_width, dtype=torch.float16, device="cuda") * 0.05
    bias_v = torch.randn(value_width, dtype=torch.float16, device="cuda") * 0.05
    output_q = torch.empty((m, query_width), dtype=torch.float16, device="cuda")
    output_k = torch.empty((m, key_width), dtype=torch.float16, device="cuda")
    output_v = torch.empty((m, value_width), dtype=torch.float16, device="cuda")

    _fused_qkv_projection_kernel[_projection_autotune_grid](
        output_q,
        output_k,
        output_v,
        activations,
        weights_q,
        weights_k,
        weights_v,
        bias_q,
        bias_k,
        bias_v,
        m,
        query_width,
        key_width,
        value_width,
        k,
    )

    expected = (
        torch.matmul(activations, weights_q) + bias_q,
        torch.matmul(activations, weights_k) + bias_k,
        torch.matmul(activations, weights_v) + bias_v,
    )
    for result, reference in zip((output_q, output_k, output_v), expected, strict=True):
        torch.testing.assert_close(
            result,
            reference,
            rtol=_FP16_RTOL,
            atol=_FP16_ATOL,
        )


def test_fused_qkv_bias_supports_distinct_in_place_outputs() -> None:
    """Each independent bias broadcasts across rows and supports in-place writes."""
    torch.manual_seed(79)
    row_count = 5
    query_width, key_width, value_width = 11, 13, 9
    inputs = (
        torch.randn((row_count, query_width), dtype=torch.float16, device="cuda"),
        torch.randn((row_count, key_width), dtype=torch.float16, device="cuda"),
        torch.randn((row_count, value_width), dtype=torch.float16, device="cuda"),
    )
    biases = (
        torch.randn(query_width, dtype=torch.float16, device="cuda"),
        torch.randn(key_width, dtype=torch.float16, device="cuda"),
        torch.randn(value_width, dtype=torch.float16, device="cuda"),
    )
    outputs = tuple(value.clone() for value in inputs)

    _fused_qkv_bias_kernel[_bias_autotune_grid](
        torch.empty_like(outputs[0]),
        inputs[0],
        biases[0],
        torch.empty_like(outputs[1]),
        inputs[1],
        biases[1],
        torch.empty_like(outputs[2]),
        inputs[2],
        biases[2],
        row_count,
        query_width,
        key_width,
        value_width,
    )
    _fused_qkv_bias_kernel[_bias_autotune_grid](
        outputs[0],
        outputs[0],
        biases[0],
        outputs[1],
        outputs[1],
        biases[1],
        outputs[2],
        outputs[2],
        biases[2],
        row_count,
        query_width,
        key_width,
        value_width,
    )

    expected = tuple(
        (input_ + bias).to(torch.float16)
        for input_, bias in zip(inputs, biases, strict=True)
    )
    for result, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=2e-3)


def test_fused_qkv_artifacts_capture_all_pointer_arguments() -> None:
    """Serialized artifacts retain the separate-pointer ABI and selected grids."""
    architecture = _architecture()
    projection = FusedQkvProjectionSpecialization(169, 32, 32, 32, 128, architecture)
    projection_artifact = compile_fused_qkv_projection(projection)
    projection_selected = _fused_qkv_projection_kernel.best_config
    assert projection_artifact.parameters == (_POINTER,) * 10 + _NULL_POINTERS
    assert projection_artifact.grid == _projection_artifact_grid(
        projection_selected.kwargs,
        projection.m,
        projection.query_width,
        projection.key_width,
        projection.value_width,
    )
    assert projection_artifact.block == (
        projection_selected.num_warps * 32,
        1,
        1,
    )

    bias = FusedQkvBiasSpecialization(169, 32, 32, 32, architecture)
    bias_artifact = compile_fused_qkv_bias(bias)
    bias_selected = _fused_qkv_bias_kernel.best_config
    assert bias_artifact.parameters == (_POINTER,) * 9 + _NULL_POINTERS
    assert bias_artifact.grid == _bias_artifact_grid(
        bias_selected.kwargs,
        bias.row_count,
        bias.query_width,
        bias.key_width,
        bias.value_width,
    )
    assert bias_artifact.block == (bias_selected.num_warps * 32, 1, 1)
