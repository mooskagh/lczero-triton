"""NCHW-to-NHWC extraction kernel family."""

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
    block_size: int = 256
    num_warps: int = 8

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
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            message = "block_size must be a positive power of two"
            raise ValueError(message)
        if self.num_warps <= 0:
            message = "num_warps must be positive"
            raise ValueError(message)


def compile_nchw_to_nhwc(
    specialization: NchwToNhwcSpecialization,
) -> KernelArtifact:
    """Compile one NCHW-to-NHWC extraction specialization."""
    element_count = (
        specialization.batch_size
        * specialization.height
        * specialization.width
        * specialization.output_channels
    )
    grid = (
        (element_count + specialization.block_size - 1) // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _nchw_to_nhwc_kernel,
            {
                "output": "*fp16",
                "input_": "*fp16",
                "batch_size": "constexpr",
                "input_channels": "constexpr",
                "output_channels": "constexpr",
                "height": "constexpr",
                "width": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "batch_size": specialization.batch_size,
                "input_channels": specialization.input_channels,
                "output_channels": specialization.output_channels,
                "height": specialization.height,
                "width": specialization.width,
                "block_size": specialization.block_size,
            },
        ),
        target=GPUTarget("cuda", specialization.architecture, _WARP_SIZE),
        options={"num_warps": specialization.num_warps},
    )
    return artifact_from_triton(
        compiled,
        grid=grid,
        parameters=(_POINTER, _POINTER),
    )


def nchw_to_nhwc(
    builder: ExecutableBuilder,
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
