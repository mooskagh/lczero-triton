"""Separate-pointer fused FP16 QKV projection and bias kernel families."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import elementwise_configs
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.matmul import (
    _MATMUL_CONFIGS,
    _USE_FP16_ACCUMULATOR,
)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=list(_MATMUL_CONFIGS),
    key=["m", "query_width", "key_width", "value_width", "k"],
    cache_results=True,
)
@triton.jit
def _fused_qkv_projection_kernel(
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
    m: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size_m: tl.constexpr,
) -> None:
    """Compute three independent row-major GEMMs with bias in one three-dimensional launch."""
    program_id = tl.program_id(0)
    projection = tl.program_id(2)
    program_count_m = tl.cdiv(m, block_m)
    max_width = max(query_width, key_width, value_width)
    program_count_n = tl.cdiv(max_width, block_n)
    programs_per_group = group_size_m * program_count_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * group_size_m
    group_program_count_m = min(program_count_m - first_program_m, group_size_m)
    program_m = first_program_m + (
        (program_id % programs_per_group) % group_program_count_m
    )
    program_n = (program_id % programs_per_group) // group_program_count_m

    # Give each z-plane one projection to avoid three accumulators per tile.
    is_query = projection == 0
    is_key = projection == 1
    selected_output = tl.where(
        is_query,
        output_q,
        tl.where(is_key, output_k, output_v),
    )
    selected_weights = tl.where(
        is_query,
        weights_q,
        tl.where(is_key, weights_k, weights_v),
    )
    selected_bias = tl.where(
        is_query,
        bias_q,
        tl.where(is_key, bias_k, bias_v),
    )
    selected_width = tl.where(
        is_query,
        query_width,
        tl.where(is_key, key_width, value_width),
    )
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    weight_pointers = (
        selected_weights + offsets_k[:, None] * selected_width + offsets_n[None, :]
    )
    accumulator = tl.zeros(
        (block_m, block_n),
        dtype=tl.float16 if _USE_FP16_ACCUMULATOR else tl.float32,
    )

    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        activation_values = tl.load(
            activation_pointers,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        weight_values = tl.load(
            weight_pointers,
            mask=(offsets_k[:, None] < remaining_k)
            & (offsets_n[None, :] < selected_width),
            other=0.0,
        )
        accumulator = tl.dot(
            activation_values,
            weight_values,
            accumulator,
            out_dtype=tl.float16 if _USE_FP16_ACCUMULATOR else tl.float32,
        )
        activation_pointers += block_k
        weight_pointers += block_k * selected_width

    bias_values = tl.load(
        selected_bias + offsets_n,
        mask=offsets_n < selected_width,
        other=0.0,
    ).to(tl.float32)
    final_values = accumulator.to(tl.float32) + bias_values[None, :]

    output_mask_m = offsets_m[:, None] < m
    output_pointers = (
        selected_output + offsets_m[:, None] * selected_width + offsets_n[None, :]
    )
    tl.store(
        output_pointers,
        final_values.to(tl.float16),
        mask=output_mask_m & (offsets_n[None, :] < selected_width),
    )


@triton.autotune(
    configs=elementwise_configs(),
    key=["row_count", "query_width", "key_width", "value_width"],
    cache_results=True,
)
@triton.jit
def _fused_qkv_bias_kernel(
    output_q,
    input_q,
    bias_q,
    output_k,
    input_k,
    bias_k,
    output_v,
    input_v,
    bias_v,
    row_count: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    """Broadcast three independent biases in one flat elementwise launch."""
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    query_valid = offsets < row_count * query_width
    key_valid = offsets < row_count * key_width
    value_valid = offsets < row_count * value_width

    query_values = tl.load(
        input_q + offsets,
        mask=query_valid,
        other=0.0,
    ).to(tl.float32)
    query_biases = tl.load(
        bias_q + offsets % query_width,
        mask=query_valid,
        other=0.0,
    ).to(tl.float32)
    key_values = tl.load(input_k + offsets, mask=key_valid, other=0.0).to(tl.float32)
    key_biases = tl.load(
        bias_k + offsets % key_width,
        mask=key_valid,
        other=0.0,
    ).to(tl.float32)
    value_values = tl.load(
        input_v + offsets,
        mask=value_valid,
        other=0.0,
    ).to(tl.float32)
    value_biases = tl.load(
        bias_v + offsets % value_width,
        mask=value_valid,
        other=0.0,
    ).to(tl.float32)

    tl.store(output_q + offsets, query_values + query_biases, mask=query_valid)
    tl.store(output_k + offsets, key_values + key_biases, mask=key_valid)
    tl.store(output_v + offsets, value_values + value_biases, mask=value_valid)


@dataclass(frozen=True, slots=True)
class FusedQkvProjectionSpecialization:
    """Immutable separate-pointer QKV projection specialization."""

    m: int
    query_width: int
    key_width: int
    value_width: int
    k: int
    architecture: int


@dataclass(frozen=True, slots=True)
class FusedQkvBiasSpecialization:
    """Immutable separate-pointer QKV bias specialization."""

    row_count: int
    query_width: int
    key_width: int
    value_width: int
    architecture: int


def _projection_autotune_grid(
    configuration: Mapping[str, object],
) -> tuple[int, int, int]:
    """Return the grouped one-dimensional QKV projection grid."""
    m = cast("int", configuration["m"])
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    max_width = max(
        cast("int", configuration["query_width"]),
        cast("int", configuration["key_width"]),
        cast("int", configuration["value_width"]),
    )
    return (
        ((m + block_m - 1) // block_m) * ((max_width + block_n - 1) // block_n),
        1,
        3,
    )


def _projection_artifact_grid(
    configuration: Mapping[str, object],
    m: int,
    query_width: int,
    key_width: int,
    value_width: int,
) -> tuple[int, int, int]:
    """Resolve the serialized QKV projection grid."""
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    max_width = max(query_width, key_width, value_width)
    return (
        ((m + block_m - 1) // block_m) * ((max_width + block_n - 1) // block_n),
        1,
        3,
    )


def _bias_autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat QKV bias grid for a tuning candidate."""
    row_count = cast("int", configuration["row_count"])
    block_size = cast("int", configuration["block_size"])
    max_width = max(
        cast("int", configuration["query_width"]),
        cast("int", configuration["key_width"]),
        cast("int", configuration["value_width"]),
    )
    element_count = row_count * max_width
    return ((element_count + block_size - 1) // block_size,)


def _bias_artifact_grid(
    configuration: Mapping[str, object],
    row_count: int,
    query_width: int,
    key_width: int,
    value_width: int,
) -> tuple[int, int, int]:
    """Resolve the serialized QKV bias grid."""
    block_size = cast("int", configuration["block_size"])
    element_count = row_count * max(query_width, key_width, value_width)
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_fused_qkv_projection(
    specialization: FusedQkvProjectionSpecialization,
) -> KernelArtifact:
    """Autotune and compile one separate-pointer QKV projection."""
    output_q = torch.empty(
        (specialization.m, specialization.query_width),
        dtype=torch.float16,
        device="cuda",
    )
    output_k = torch.empty(
        (specialization.m, specialization.key_width),
        dtype=torch.float16,
        device="cuda",
    )
    output_v = torch.empty(
        (specialization.m, specialization.value_width),
        dtype=torch.float16,
        device="cuda",
    )
    activations = torch.zeros(
        (specialization.m, specialization.k),
        dtype=torch.float16,
        device="cuda",
    )
    weights_q = torch.zeros(
        (specialization.k, specialization.query_width),
        dtype=torch.float16,
        device="cuda",
    )
    weights_k = torch.zeros(
        (specialization.k, specialization.key_width),
        dtype=torch.float16,
        device="cuda",
    )
    weights_v = torch.zeros(
        (specialization.k, specialization.value_width),
        dtype=torch.float16,
        device="cuda",
    )
    bias_q = torch.zeros(specialization.query_width, dtype=torch.float16, device="cuda")
    bias_k = torch.zeros(specialization.key_width, dtype=torch.float16, device="cuda")
    bias_v = torch.zeros(specialization.value_width, dtype=torch.float16, device="cuda")
    compiled = _fused_qkv_projection_kernel[_projection_autotune_grid](
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
        specialization.m,
        specialization.query_width,
        specialization.key_width,
        specialization.value_width,
        specialization.k,
    )
    selected = _fused_qkv_projection_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_projection_artifact_grid(
            selected.kwargs,
            specialization.m,
            specialization.query_width,
            specialization.key_width,
            specialization.value_width,
        ),
        parameters=(_POINTER,) * 10,
    )


def compile_fused_qkv_bias(
    specialization: FusedQkvBiasSpecialization,
) -> KernelArtifact:
    """Autotune and compile one separate-pointer QKV bias broadcast."""
    output_q = torch.empty(
        (specialization.row_count, specialization.query_width),
        dtype=torch.float16,
        device="cuda",
    )
    output_k = torch.empty(
        (specialization.row_count, specialization.key_width),
        dtype=torch.float16,
        device="cuda",
    )
    output_v = torch.empty(
        (specialization.row_count, specialization.value_width),
        dtype=torch.float16,
        device="cuda",
    )
    input_q = torch.zeros_like(output_q)
    input_k = torch.zeros_like(output_k)
    input_v = torch.zeros_like(output_v)
    bias_q = torch.zeros(specialization.query_width, dtype=torch.float16, device="cuda")
    bias_k = torch.zeros(specialization.key_width, dtype=torch.float16, device="cuda")
    bias_v = torch.zeros(specialization.value_width, dtype=torch.float16, device="cuda")
    compiled = _fused_qkv_bias_kernel[_bias_autotune_grid](
        output_q,
        input_q,
        bias_q,
        output_k,
        input_k,
        bias_k,
        output_v,
        input_v,
        bias_v,
        specialization.row_count,
        specialization.query_width,
        specialization.key_width,
        specialization.value_width,
    )
    selected = _fused_qkv_bias_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_bias_artifact_grid(
            selected.kwargs,
            specialization.row_count,
            specialization.query_width,
            specialization.key_width,
            specialization.value_width,
        ),
        parameters=(_POINTER,) * 9,
    )


def fused_qkv_projection(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output_q: Buffer,
    output_k: Buffer,
    output_v: Buffer,
    activations: Buffer,
    weights_q: Buffer,
    weights_k: Buffer,
    weights_v: Buffer,
    bias_q: Buffer,
    bias_k: Buffer,
    bias_v: Buffer,
    specialization: FusedQkvProjectionSpecialization,
) -> None:
    """Append one QKV projection while retaining independent buffer pointers."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_fused_qkv_projection, specialization)
    builder.call(
        kernel,
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
        readonly=[
            activations,
            weights_q,
            weights_k,
            weights_v,
            bias_q,
            bias_k,
            bias_v,
        ],
    )


def fused_qkv_bias(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output_q: Buffer,
    input_q: Buffer,
    bias_q: Buffer,
    output_k: Buffer,
    input_k: Buffer,
    bias_k: Buffer,
    output_v: Buffer,
    input_v: Buffer,
    bias_v: Buffer,
    specialization: FusedQkvBiasSpecialization,
) -> None:
    """Append one QKV bias broadcast while retaining in-place writes."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_fused_qkv_bias, specialization)
    outputs = (output_q, output_k, output_v)
    readonly = [
        source
        for source in (input_q, bias_q, input_k, bias_k, input_v, bias_v)
        if all(source is not output for output in outputs)
    ]
    builder.call(
        kernel,
        output_q,
        input_q,
        bias_q,
        output_k,
        input_k,
        bias_k,
        output_v,
        input_v,
        bias_v,
        readonly=readonly,
    )
