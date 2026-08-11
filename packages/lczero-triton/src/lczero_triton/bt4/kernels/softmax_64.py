"""Autotuned fused Smolgen addition and 64-way FP16 softmax family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import validate_active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_SOFTMAX_WIDTH = 64
_TWICE_HALF_MAX = 131008.0
_TILE_CONFIGURATIONS = (
    (1, 1),
    (2, 1),
    (2, 2),
    (4, 2),
    (4, 4),
    (8, 4),
    (8, 8),
    (16, 8),
)


def _softmax_configs() -> list[triton.Config]:
    """Return independent row-packing and warp-count candidates."""
    return [
        triton.Config({"rows_per_program": rows}, num_warps=num_warps)
        for rows, num_warps in _TILE_CONFIGURATIONS
    ]


@triton.autotune(
    configs=_softmax_configs(),
    key=["row_count"],
    cache_results=True,
)
@triton.jit
def _softmax_64_kernel(
    output,
    scaled_qk,
    smolgen,
    row_count: tl.constexpr,
    rows_per_program: tl.constexpr,
) -> None:
    """Add FP16 logits and calculate one FP32 softmax per 64 values."""
    rows = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
    columns = tl.arange(0, 64)
    valid_rows = rows < row_count
    offsets = rows[:, None] * 64 + columns[None, :]
    mask = valid_rows[:, None]

    values = tl.load(scaled_qk + offsets, mask=mask, other=0.0).to(tl.float32)
    values += tl.load(smolgen + offsets, mask=mask, other=0.0).to(tl.float32)

    # CUDA clamps infinities after FP32 addition without hiding NaNs.
    is_nan = values != values  # noqa: PLR0124  # Device-side NaN test.
    clamped = tl.minimum(tl.maximum(values, -131008.0), 131008.0)
    values = tl.where(is_nan, values, clamped)

    maximum = tl.max(values, axis=1)
    exponentials = tl.exp(values - maximum[:, None])
    denominator = tl.sum(exponentials, axis=1)
    result = exponentials / denominator[:, None]
    tl.store(output + offsets, result, mask=mask)


@dataclass(frozen=True, slots=True)
class Softmax64Specialization:
    """Immutable contiguous 64-way FP16 softmax specialization."""

    row_count: int
    architecture: int

    def __post_init__(self) -> None:
        """Validate the static workload and compilation target."""
        if self.row_count <= 0:
            message = "row count must be positive"
            raise ValueError(message)
        if self.architecture <= 0:
            message = "architecture must be positive"
            raise ValueError(message)


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the row-tiled launch grid for a tuning candidate."""
    row_count = cast("int", configuration["row_count"])
    rows_per_program = cast("int", configuration["rows_per_program"])
    return ((row_count + rows_per_program - 1) // rows_per_program,)


def _artifact_grid(
    configuration: Mapping[str, object],
    row_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected row packing."""
    rows_per_program = cast("int", configuration["rows_per_program"])
    return ((row_count + rows_per_program - 1) // rows_per_program, 1, 1)


def compile_softmax_64(
    specialization: Softmax64Specialization,
) -> KernelArtifact:
    """Autotune and compile one fused Smolgen attention softmax."""
    validate_active_architecture(specialization.architecture)
    shape = (specialization.row_count, _SOFTMAX_WIDTH)
    output = torch.empty(shape, dtype=torch.float16, device="cuda")
    scaled_qk = torch.zeros(shape, dtype=torch.float16, device="cuda")
    smolgen = torch.zeros(shape, dtype=torch.float16, device="cuda")
    compiled = _softmax_64_kernel[_autotune_grid](
        output,
        scaled_qk,
        smolgen,
        specialization.row_count,
    )
    selected = _softmax_64_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, specialization.row_count),
        parameters=(_POINTER, _POINTER, _POINTER),
    )


def softmax_64(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    scaled_qk: Buffer,
    smolgen: Buffer,
    specialization: Softmax64Specialization,
) -> None:
    """Append fused Smolgen addition and 64-way attention softmax."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_softmax_64, specialization)
    readonly: list[Buffer] = []
    for source in (scaled_qk, smolgen):
        if source is not output and all(source is not value for value in readonly):
            readonly.append(source)
    builder.call(kernel, output, scaled_qk, smolgen, readonly=readonly)
