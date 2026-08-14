"""NCHW-to-NHWC extraction kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import (
    elementwise_configs,
    validate_active_architecture,
)
from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=elementwise_configs(),
    key=["batch_size", "input_channels", "output_channels", "height", "width"],
    cache_results=True,
)
@triton.jit
def _nchw_to_nhwc_kernel(
    output,
    input_,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch_size * height * width * output_channels
    valid = offsets < element_count
    channel = offsets % output_channels
    spatial = (offsets // output_channels) % (height * width)
    batch = offsets // (output_channels * height * width)
    input_offsets = (batch * input_channels + channel) * height * width + spatial
    values = tl.load(input_ + input_offsets, mask=valid, other=0.0)
    tl.store(output + offsets, values, mask=valid)


@dataclass(frozen=True, slots=True)
class NchwToNhwcSpecialization:
    """Immutable contiguous NCHW-to-NHWC extraction specialization."""

    batch_size: int
    input_channels: int
    output_channels: int
    height: int
    width: int
    architecture: int

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if any(
            value <= 0
            for value in (
                self.batch_size,
                self.input_channels,
                self.output_channels,
                self.height,
                self.width,
            )
        ):
            message = "tensor dimensions must be positive"
            raise ValueError(message)
        if self.architecture <= 0:
            message = "architecture must be positive"
            raise ValueError(message)
        if self.output_channels > self.input_channels:
            message = "output_channels cannot exceed input_channels"
            raise ValueError(message)


def _element_count(configuration: Mapping[str, object]) -> int:
    """Return the output element count from kernel configuration values."""
    return (
        cast("int", configuration["batch_size"])
        * cast("int", configuration["height"])
        * cast("int", configuration["width"])
        * cast("int", configuration["output_channels"])
    )


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat layout-conversion grid for a tuning candidate."""
    block_size = cast("int", configuration["block_size"])
    return ((_element_count(configuration) + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_nchw_to_nhwc(
    specialization: NchwToNhwcSpecialization,
) -> KernelArtifact:
    """Autotune and compile one NCHW-to-NHWC extraction specialization."""
    validate_active_architecture(specialization.architecture)
    element_count = (
        specialization.batch_size
        * specialization.height
        * specialization.width
        * specialization.output_channels
    )
    output = torch.empty(
        (
            specialization.batch_size,
            specialization.height,
            specialization.width,
            specialization.output_channels,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    input_ = torch.zeros(
        (
            specialization.batch_size,
            specialization.input_channels,
            specialization.height,
            specialization.width,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _nchw_to_nhwc_kernel[_autotune_grid](
        output,
        input_,
        specialization.batch_size,
        specialization.input_channels,
        specialization.output_channels,
        specialization.height,
        specialization.width,
    )
    selected = _nchw_to_nhwc_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, element_count),
        parameters=(_POINTER, _POINTER),
    )


def nchw_to_nhwc(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    specialization: NchwToNhwcSpecialization,
) -> None:
    """Append an NCHW-to-NHWC extraction call to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_nchw_to_nhwc, specialization)
    builder.call(kernel, output, input_, readonly=(input_,))
