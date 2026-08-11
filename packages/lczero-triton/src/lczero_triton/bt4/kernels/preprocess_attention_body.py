"""Attention-body input concatenation kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import (
    preprocess_configs,
    validate_active_architecture,
)
from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=preprocess_configs(),
    key=["batch_size", "square_count", "input_channels", "encoding_channels"],
    cache_results=True,
)
@triton.jit
def _preprocess_attention_body_kernel(
    output,
    input_,
    encoding,
    batch_size: tl.constexpr,
    square_count: tl.constexpr,
    input_channels: tl.constexpr,
    encoding_channels: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    square = tl.program_id(1)
    channels = tl.program_id(2) * block_size + tl.arange(0, block_size)
    output_channels = input_channels + encoding_channels
    valid = (batch < batch_size) & (channels < output_channels)

    input_offsets = (
        batch * input_channels * square_count + channels * square_count + square
    )
    input_values = tl.load(
        input_ + input_offsets,
        mask=channels < input_channels,
        other=0.0,
    )
    encoding_channel = channels - input_channels
    encoding_offsets = (
        batch * square_count + square
    ) * encoding_channels + encoding_channel
    encoding_values = tl.load(
        encoding + encoding_offsets,
        mask=(channels >= input_channels) & valid,
        other=0.0,
    )
    values = tl.where(channels < input_channels, input_values, encoding_values)
    output_offsets = (batch * square_count + square) * output_channels + channels
    tl.store(output + output_offsets, values, mask=valid)


@dataclass(frozen=True, slots=True)
class PreprocessAttentionBodySpecialization:
    """Immutable dense-position attention-input preprocessing specialization."""

    batch_size: int
    input_channels: int
    encoding_channels: int
    architecture: int
    square_count: int = 64

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if any(
            value <= 0
            for value in (
                self.batch_size,
                self.input_channels,
                self.encoding_channels,
                self.square_count,
            )
        ):
            message = "tensor dimensions must be positive"
            raise ValueError(message)
        if self.architecture <= 0:
            message = "architecture must be positive"
            raise ValueError(message)


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int, int, int]:
    """Return the three-dimensional channel-tiled preprocessing grid."""
    batch_size = cast("int", configuration["batch_size"])
    square_count = cast("int", configuration["square_count"])
    output_channels = cast("int", configuration["input_channels"]) + cast(
        "int", configuration["encoding_channels"]
    )
    block_size = cast("int", configuration["block_size"])
    return (
        batch_size,
        square_count,
        (output_channels + block_size - 1) // block_size,
    )


def _artifact_grid(
    configuration: Mapping[str, object],
    batch_size: int,
    square_count: int,
    output_channels: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return (
        batch_size,
        square_count,
        (output_channels + block_size - 1) // block_size,
    )


def compile_preprocess_attention_body(
    specialization: PreprocessAttentionBodySpecialization,
) -> KernelArtifact:
    """Autotune and compile one attention-input preprocessing specialization."""
    validate_active_architecture(specialization.architecture)
    output_channels = specialization.input_channels + specialization.encoding_channels
    output = torch.empty(
        (specialization.batch_size, specialization.square_count, output_channels),
        dtype=torch.float16,
        device="cuda",
    )
    input_ = torch.zeros(
        (
            specialization.batch_size,
            specialization.input_channels,
            specialization.square_count,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    encoding = torch.zeros(
        (
            specialization.batch_size,
            specialization.square_count,
            specialization.encoding_channels,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _preprocess_attention_body_kernel[_autotune_grid](
        output,
        input_,
        encoding,
        specialization.batch_size,
        specialization.square_count,
        specialization.input_channels,
        specialization.encoding_channels,
    )
    selected = _preprocess_attention_body_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(
            selected.kwargs,
            specialization.batch_size,
            specialization.square_count,
            output_channels,
        ),
        parameters=(_POINTER, _POINTER, _POINTER),
    )


def preprocess_attention_body(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    encoding: Buffer,
    specialization: PreprocessAttentionBodySpecialization,
) -> None:
    """Append dense-position attention preprocessing to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_preprocess_attention_body, specialization)
    builder.call(kernel, output, input_, encoding, readonly=(input_, encoding))
