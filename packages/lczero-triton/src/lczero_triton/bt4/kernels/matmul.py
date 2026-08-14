"""Autotuned contiguous FP16 dense matrix multiplication family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_GROUP_SIZES_M = (1, 4, 8)
_TILE_CONFIGS = (
    (128, 256, 64, 8, 3),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 2, 5),
    (32, 64, 32, 2, 5),
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


@triton.autotune(
    configs=list(_MATMUL_CONFIGS),
    key=["m", "n", "k"],
    cache_results=True,
)
@triton.jit
def _matmul_kernel(
    output,
    activations,
    weights,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size_m: tl.constexpr,
) -> None:
    """Compute one grouped-M tile of a contiguous row-major GEMM."""
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

    offsets_m = (program_m * block_m + tl.arange(0, block_m)) % m
    offsets_n = (program_n * block_n + tl.arange(0, block_n)) % n
    offsets_k = tl.arange(0, block_k)
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    weight_pointers = weights + offsets_k[:, None] * n + offsets_n[None, :]
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        activation_values = tl.load(
            activation_pointers,
            mask=offsets_k[None, :] < remaining_k,
            other=0.0,
        )
        weight_values = tl.load(
            weight_pointers,
            mask=offsets_k[:, None] < remaining_k,
            other=0.0,
        )
        accumulator = tl.dot(activation_values, weight_values, accumulator)
        activation_pointers += block_k
        weight_pointers += block_k * n

    output_offsets_m = program_m * block_m + tl.arange(0, block_m)
    output_offsets_n = program_n * block_n + tl.arange(0, block_n)
    output_pointers = output + output_offsets_m[:, None] * n + output_offsets_n[None, :]
    output_mask = (output_offsets_m[:, None] < m) & (output_offsets_n[None, :] < n)
    tl.store(output_pointers, accumulator.to(tl.float16), mask=output_mask)


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
    compiled = _matmul_kernel[_autotune_grid](
        output,
        activations,
        weights,
        specialization.m,
        specialization.n,
        specialization.k,
    )
    selected = _matmul_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(
            selected.kwargs,
            specialization.m,
            specialization.n,
        ),
        parameters=(_POINTER, _POINTER, _POINTER),
    )


def matmul(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    specialization: MatmulSpecialization,
) -> None:
    """Append one contiguous row-major dense matrix multiplication."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_matmul, specialization)
    builder.call(
        kernel,
        output,
        activations,
        weights,
        readonly=[activations, weights],
    )
