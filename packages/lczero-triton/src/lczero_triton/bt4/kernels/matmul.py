"""Autotuned contiguous FP16 dense matrix multiplication family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

Activation = Literal["none", "mish", "relu", "swish"]
_ACTIVATION_NONE = tl.constexpr(0)
_ACTIVATION_MISH = tl.constexpr(1)
_ACTIVATION_RELU = tl.constexpr(2)
_ACTIVATION_SWISH = tl.constexpr(3)
_ACTIVATION_CODES: dict[Activation, int] = {
    "none": 0,
    "mish": 1,
    "relu": 2,
    "swish": 3,
}
_MISH_BRANCH = tl.constexpr(-0.6)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
# Set this to false when the FP32 accumulator path is required for comparison.
_USE_FP16_ACCUMULATOR = tl.constexpr(value=True)
_GROUP_SIZES_M = (1, 8)
_TILE_CONFIGS = (
    # Large high-throughput tiles (for M>=1024, N>=1024, K>=512)
    (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 4, 4),
    (128, 128, 64, 8, 3),
    (128, 128, 32, 4, 3),
    (128, 128, 32, 4, 4),
    (128, 64, 64, 4, 3),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 3),
    (64, 128, 64, 4, 4),
    (64, 128, 32, 4, 3),
    (128, 64, 32, 4, 3),
    # Deep-K tiles (for K>=2048, e.g. Smolgen Dense1, Value Dense1, Moves Dense1)
    (64, 64, 128, 4, 4),
    (32, 64, 128, 4, 4),
    (64, 32, 128, 4, 4),
    (32, 32, 128, 8, 4),
    # Medium tiles
    (64, 64, 64, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 32, 64, 4, 3),
    # Small / narrow tiles (for M<=256 or N<=256 or K<=256)
    (32, 64, 64, 4, 3),
    (32, 32, 64, 4, 3),
    (32, 32, 32, 2, 3),
)
_MATMUL_CONFIGS = tuple(
    triton.Config(
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "group_size_m": group_size_m,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m, block_n, block_k, num_warps, num_stages in _TILE_CONFIGS
    for group_size_m in _GROUP_SIZES_M
)


_MIN_GROUPED_M = 512
_MIN_GROUP_TILES = 4
_MAX_SMEM_BYTES = 128 * 1024
_SMALL_DIM_32 = 32
_SMALL_DIM_128 = 128
_SMALL_DIM_256 = 256
_MAX_K_BLOCK_FOR_SMALL_K = 64


def _prune_matmul_configs(
    configs: list[triton.Config],
    named_args: Mapping[str, object],
    **kwargs: object,
) -> list[triton.Config]:
    """Prune tile and grouping candidates incompatible with operand dimensions."""
    m = named_args.get("m") or kwargs.get("m")
    n = named_args.get("n") or kwargs.get("n")
    k = named_args.get("k") or kwargs.get("k")
    if m is None or n is None or k is None:
        return configs

    m_val = cast("int", m)
    n_val = cast("int", n)
    k_val = cast("int", k)

    pruned: list[triton.Config] = []
    for conf in configs:
        bm = cast("int", conf.kwargs["block_m"])
        bn = cast("int", conf.kwargs["block_n"])
        bk = cast("int", conf.kwargs["block_k"])
        gm = cast("int", conf.kwargs["group_size_m"])

        if gm > 1 and (m_val < _MIN_GROUPED_M or m_val < bm * _MIN_GROUP_TILES):
            continue
        if n_val <= _SMALL_DIM_32 and bn > _SMALL_DIM_32:
            continue
        if n_val <= _SMALL_DIM_128 and bn > _SMALL_DIM_128:
            continue
        if m_val <= _SMALL_DIM_256 and bm > _SMALL_DIM_128:
            continue
        if k_val <= _SMALL_DIM_256 and bk > _MAX_K_BLOCK_FOR_SMALL_K:
            continue
        smem_bytes = 2 * (bm * bk + bk * bn) * conf.num_stages
        if smem_bytes > _MAX_SMEM_BYTES:
            continue
        pruned.append(conf)

    return pruned or [configs[0]]


@triton.autotune(
    configs=list(_MATMUL_CONFIGS),
    key=["m", "n", "k", "has_bias", "activation"],
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
    cache_results=True,
)
@triton.jit
def _matmul_kernel(
    output,
    activations,
    weights,
    bias,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    has_bias: tl.constexpr,
    activation: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size_m: tl.constexpr,
) -> None:
    """Compute one grouped-M tile of GEMM with optional bias and activation."""
    program_id = tl.program_id(0)
    program_count_m = tl.cdiv(m, block_m)
    program_count_n = tl.cdiv(n, block_n)
    programs_per_group = group_size_m * program_count_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * group_size_m
    group_program_count_m = min(program_count_m - first_program_m, group_size_m)
    program_m = first_program_m + (
        (program_id % programs_per_group) % group_program_count_m
    )
    program_n = (program_id % programs_per_group) // group_program_count_m

    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    weight_pointers = weights + offsets_k[:, None] * n + offsets_n[None, :]
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
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(
            activation_values,
            weight_values,
            accumulator,
            out_dtype=tl.float16 if _USE_FP16_ACCUMULATOR else tl.float32,
        )
        activation_pointers += block_k
        weight_pointers += block_k * n

    values = accumulator.to(tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        ).to(tl.float32)
        values += bias_values[None, :]

    if activation == _ACTIVATION_MISH:
        exponential = tl.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    elif activation == _ACTIVATION_RELU:
        values = tl.maximum(values, 0.0)
    elif activation == _ACTIVATION_SWISH:
        values = values / (1.0 + tl.exp(-values))

    output_offsets_m = program_m * block_m + tl.arange(0, block_m)
    output_offsets_n = program_n * block_n + tl.arange(0, block_n)
    output_pointers = output + output_offsets_m[:, None] * n + output_offsets_n[None, :]
    output_mask = (output_offsets_m[:, None] < m) & (output_offsets_n[None, :] < n)
    tl.store(output_pointers, values.to(tl.float16), mask=output_mask)


@triton.autotune(
    configs=list(_MATMUL_CONFIGS),
    key=["m", "n", "k", "has_bias", "activation"],
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
    cache_results=True,
)
@triton.jit
def _matmul_skip_kernel(
    output,
    activations,
    weights,
    bias,
    skip,
    alpha,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    has_bias: tl.constexpr,
    activation: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size_m: tl.constexpr,
) -> None:
    """Compute GEMM with optional bias, activation, alpha scaling, and residual skip."""
    program_id = tl.program_id(0)
    program_count_m = tl.cdiv(m, block_m)
    program_count_n = tl.cdiv(n, block_n)
    programs_per_group = group_size_m * program_count_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * group_size_m
    group_program_count_m = min(program_count_m - first_program_m, group_size_m)
    program_m = first_program_m + (
        (program_id % programs_per_group) % group_program_count_m
    )
    program_n = (program_id % programs_per_group) // group_program_count_m

    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    weight_pointers = weights + offsets_k[:, None] * n + offsets_n[None, :]
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
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(
            activation_values,
            weight_values,
            accumulator,
            out_dtype=tl.float16 if _USE_FP16_ACCUMULATOR else tl.float32,
        )
        activation_pointers += block_k
        weight_pointers += block_k * n

    values = accumulator.to(tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        ).to(tl.float32)
        values += bias_values[None, :]

    if activation == _ACTIVATION_MISH:
        exponential = tl.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    elif activation == _ACTIVATION_RELU:
        values = tl.maximum(values, 0.0)
    elif activation == _ACTIVATION_SWISH:
        values = values / (1.0 + tl.exp(-values))

    alpha_val = tl.load(alpha).to(tl.float32)
    values = values * alpha_val
    output_offsets_m = program_m * block_m + tl.arange(0, block_m)
    output_offsets_n = program_n * block_n + tl.arange(0, block_n)
    skip_pointers = skip + output_offsets_m[:, None] * n + output_offsets_n[None, :]
    skip_mask = (output_offsets_m[:, None] < m) & (output_offsets_n[None, :] < n)
    skip_values = tl.load(skip_pointers, mask=skip_mask, other=0.0).to(tl.float32)
    values = values + skip_values

    output_pointers = output + output_offsets_m[:, None] * n + output_offsets_n[None, :]
    output_mask = (output_offsets_m[:, None] < m) & (output_offsets_n[None, :] < n)
    tl.store(output_pointers, values.to(tl.float16), mask=output_mask)


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the grouped one-dimensional launch grid for a tuning candidate."""
    m = cast("int", configuration["m"])
    n = cast("int", configuration["n"])
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),)


def _artifact_grid(
    configuration: Mapping[str, object],
    m: int,
    n: int,
) -> tuple[int, int, int]:
    """Resolve the serialized launch grid from the selected configuration."""
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (
        ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
        1,
        1,
    )


@dataclass(frozen=True, slots=True)
class MatmulSpecialization:
    """Immutable contiguous dense GEMM specialization."""

    m: int
    n: int
    k: int
    architecture: int
    has_bias: bool = False
    has_skip: bool = False
    activation: Activation = "none"


def compile_matmul(specialization: MatmulSpecialization) -> KernelArtifact:
    """Autotune and compile one contiguous FP16 dense GEMM specialization."""
    output = torch.empty(
        (specialization.m, specialization.n),
        dtype=torch.float16,
        device="cuda",
    )
    activations = torch.empty(
        (specialization.m, specialization.k),
        dtype=torch.float16,
        device="cuda",
    )
    weights = torch.empty(
        (specialization.k, specialization.n),
        dtype=torch.float16,
        device="cuda",
    )
    bias = torch.zeros(
        specialization.n,
        dtype=torch.float16,
        device="cuda",
    )
    activation_code = _ACTIVATION_CODES[specialization.activation]
    parameters: tuple[int, ...]
    if specialization.has_skip:
        skip = torch.zeros(
            (specialization.m, specialization.n),
            dtype=torch.float16,
            device="cuda",
        )
        alpha = torch.ones(1, dtype=torch.float16, device="cuda")
        compiled = _matmul_skip_kernel[_autotune_grid](
            output,
            activations,
            weights,
            bias,
            skip,
            alpha,
            specialization.m,
            specialization.n,
            specialization.k,
            specialization.has_bias,
            activation_code,
        )
        selected = _matmul_skip_kernel.best_config
        parameters = (_POINTER,) * 6
    else:
        compiled = _matmul_kernel[_autotune_grid](
            output,
            activations,
            weights,
            bias,
            specialization.m,
            specialization.n,
            specialization.k,
            specialization.has_bias,
            activation_code,
        )
        selected = _matmul_kernel.best_config
        parameters = (_POINTER,) * 4

    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(
            selected.kwargs,
            specialization.m,
            specialization.n,
        ),
        parameters=parameters,
    )


def matmul(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    specialization: MatmulSpecialization,
    bias: Buffer | None = None,
    *,
    skip: Buffer | None = None,
    alpha: Buffer | None = None,
) -> None:
    """Append one contiguous row-major dense matrix multiplication."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_matmul, specialization)
    if specialization.has_skip:
        if skip is None or alpha is None:
            message = (
                "Skip and alpha buffers are required when "
                "specialization.has_skip is True"
            )
            raise ValueError(message)
        bias_buffer = bias if (bias is not None and specialization.has_bias) else output
        readonly = [activations, weights, skip, alpha]
        if bias is not None and specialization.has_bias:
            readonly.append(bias)
        builder.call(
            kernel,
            output,
            activations,
            weights,
            bias_buffer,
            skip,
            alpha,
            readonly=readonly,
        )
    elif specialization.has_bias:
        if bias is None:
            message = "Bias buffer is required when specialization.has_bias is True"
            raise ValueError(message)
        builder.call(
            kernel,
            output,
            activations,
            weights,
            bias,
            readonly=[activations, weights, bias],
        )
    else:
        builder.call(
            kernel,
            output,
            activations,
            weights,
            output,
            readonly=[activations, weights],
        )
