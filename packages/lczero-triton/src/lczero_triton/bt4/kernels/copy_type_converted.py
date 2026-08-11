"""Contiguous FP16-to-FP32 conversion kernel family."""

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
    block_size: int = 256
    num_warps: int = 8

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if self.element_count <= 0:
            message = "element_count must be positive"
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


def compile_copy_type_converted(
    specialization: CopyTypeConvertedSpecialization,
) -> KernelArtifact:
    """Compile one FP16-to-FP32 conversion specialization."""
    grid = (
        (specialization.element_count + specialization.block_size - 1)
        // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _copy_type_converted_kernel,
            {
                "output": "*fp32",
                "input_": "*fp16",
                "element_count": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "element_count": specialization.element_count,
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


def copy_type_converted(
    builder: ExecutableBuilder,
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
