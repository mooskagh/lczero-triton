"""Packed chess-plane expansion kernel family."""

from dataclasses import dataclass

import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lczero_triton.bt4.kernels._cache import KernelCache

_MASK_BITS = 64
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32


@triton.jit
def _expand_planes_kernel(
    output,
    masks,
    values,
    plane_count: tl.constexpr,
    square_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < plane_count * square_count
    plane = offsets // square_count
    square = offsets % square_count
    mask_values = tl.load(masks + plane, mask=valid, other=0)
    plane_values = tl.load(values + plane, mask=valid, other=0.0).to(tl.float16)
    is_set = ((mask_values >> square) & 1) != 0
    expanded = tl.where(is_set, plane_values, 0.0)
    tl.store(output + offsets, expanded, mask=valid)


@dataclass(frozen=True, slots=True)
class ExpandPlanesSpecialization:
    """Immutable U64/F32-to-FP16 plane-expansion specialization."""

    plane_count: int
    architecture: int
    square_count: int = 64
    block_size: int = 256
    num_warps: int = 8

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if self.plane_count <= 0 or self.square_count <= 0:
            message = "plane and square counts must be positive"
            raise ValueError(message)
        if self.architecture <= 0:
            message = "architecture must be positive"
            raise ValueError(message)
        if self.square_count > _MASK_BITS:
            message = "U64 plane masks support at most 64 squares"
            raise ValueError(message)
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            message = "block_size must be a positive power of two"
            raise ValueError(message)
        if self.num_warps <= 0:
            message = "num_warps must be positive"
            raise ValueError(message)


def compile_expand_planes(
    specialization: ExpandPlanesSpecialization,
) -> KernelArtifact:
    """Compile one packed-plane expansion specialization."""
    element_count = specialization.plane_count * specialization.square_count
    grid = (
        (element_count + specialization.block_size - 1) // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _expand_planes_kernel,
            {
                "output": "*fp16",
                "masks": "*u64",
                "values": "*fp32",
                "plane_count": "constexpr",
                "square_count": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "plane_count": specialization.plane_count,
                "square_count": specialization.square_count,
                "block_size": specialization.block_size,
            },
        ),
        target=GPUTarget("cuda", specialization.architecture, _WARP_SIZE),
        options={"num_warps": specialization.num_warps},
    )
    return artifact_from_triton(
        compiled,
        grid=grid,
        parameters=(_POINTER, _POINTER, _POINTER),
    )


def expand_planes(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    masks: Buffer,
    values: Buffer,
    specialization: ExpandPlanesSpecialization,
) -> None:
    """Append a packed-plane expansion call to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_expand_planes, specialization)
    builder.call(kernel, output, masks, values, readonly=(masks, values))
