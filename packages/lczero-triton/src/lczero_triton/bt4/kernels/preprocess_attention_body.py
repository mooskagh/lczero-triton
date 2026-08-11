"""Attention-body input concatenation kernel family."""

from dataclasses import dataclass

import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32


@triton.jit
def _preprocess_attention_body_kernel(
    output,
    input_,
    encoding,
    square_count: tl.constexpr,
    input_channels: tl.constexpr,
    encoding_channels: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    square = tl.program_id(1)
    channels = tl.arange(0, block_size)
    output_channels = input_channels + encoding_channels
    valid = channels < output_channels

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
    block_size: int = 1024
    num_warps: int = 8

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
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            message = "block_size must be a positive power of two"
            raise ValueError(message)
        if self.num_warps <= 0:
            message = "num_warps must be positive"
            raise ValueError(message)
        if self.input_channels + self.encoding_channels > self.block_size:
            message = "preprocessing width exceeds block_size"
            raise ValueError(message)


def compile_preprocess_attention_body(
    specialization: PreprocessAttentionBodySpecialization,
) -> KernelArtifact:
    """Compile one dense-position attention-input preprocessing specialization."""
    compiled = triton.compile(
        ASTSource(
            _preprocess_attention_body_kernel,
            {
                "output": "*fp16",
                "input_": "*fp16",
                "encoding": "*fp16",
                "square_count": "constexpr",
                "input_channels": "constexpr",
                "encoding_channels": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "square_count": specialization.square_count,
                "input_channels": specialization.input_channels,
                "encoding_channels": specialization.encoding_channels,
                "block_size": specialization.block_size,
            },
        ),
        target=GPUTarget("cuda", specialization.architecture, _WARP_SIZE),
        options={"num_warps": specialization.num_warps},
    )
    return artifact_from_triton(
        compiled,
        grid=(specialization.batch_size, specialization.square_count, 1),
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
