"""Contiguous FP16-to-FP32 conversion kernel family."""

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

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=elementwise_configs(),
    key=["element_count"],
    cache_results=True,
)
@triton.jit
def _copy_type_converted_kernel(
    output,
    input_,
    element_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(input_ + offsets, mask=offsets < element_count)
    tl.store(output + offsets, values.to(tl.float32), mask=offsets < element_count)


@dataclass(frozen=True, slots=True)
class CopyTypeConvertedSpecialization:
    """Immutable FP16-to-FP32 conversion specialization."""

    element_count: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the one-dimensional grid for a tuning candidate."""
    element_count = cast("int", configuration["element_count"])
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_copy_type_converted(
    specialization: CopyTypeConvertedSpecialization,
) -> KernelArtifact:
    """Autotune and compile one FP16-to-FP32 conversion specialization."""
    output = torch.empty(
        specialization.element_count,
        dtype=torch.float32,
        device="cuda",
    )
    input_ = torch.zeros(
        specialization.element_count,
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _copy_type_converted_kernel[_autotune_grid](
        output,
        input_,
        specialization.element_count,
    )
    selected = _copy_type_converted_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, specialization.element_count),
        parameters=(_POINTER, _POINTER),
    )


def copy_type_converted(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    specialization: CopyTypeConvertedSpecialization,
) -> None:
    """Append an FP16-to-FP32 conversion call to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_copy_type_converted, specialization)
    builder.call(kernel, output, input_, readonly=(input_,))
