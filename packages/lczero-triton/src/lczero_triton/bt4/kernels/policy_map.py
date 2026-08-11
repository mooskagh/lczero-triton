"""Attention-policy gather kernel family."""

from dataclasses import dataclass

import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact, SymbolHandle
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32


@triton.jit
def _policy_map_kernel(
    output,
    input_,
    mapping,
    batch_size: tl.constexpr,
    input_element_count: tl.constexpr,
    output_element_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch_size * output_element_count
    valid = offsets < element_count
    output_index = offsets % output_element_count
    batch = offsets // output_element_count
    source_index = tl.load(mapping + output_index, mask=valid, other=-1)
    valid_source = valid & (source_index >= 0) & (source_index < input_element_count)
    values = tl.load(
        input_ + batch * input_element_count + source_index,
        mask=valid_source,
        other=0.0,
    )
    tl.store(output + offsets, values, mask=valid)


@dataclass(frozen=True, slots=True)
class PolicyMapSpecialization:
    """Immutable FP16 attention-policy gather specialization."""

    batch_size: int
    architecture: int
    input_element_count: int = 4288
    output_element_count: int = 1858
    block_size: int = 256
    num_warps: int = 8

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if any(
            value <= 0
            for value in (
                self.batch_size,
                self.input_element_count,
                self.output_element_count,
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


def compile_policy_map(
    specialization: PolicyMapSpecialization,
) -> KernelArtifact:
    """Compile one FP16 attention-policy gather specialization."""
    element_count = specialization.batch_size * specialization.output_element_count
    grid = (
        (element_count + specialization.block_size - 1) // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _policy_map_kernel,
            {
                "output": "*fp16",
                "input_": "*fp16",
                "mapping": "*i32",
                "batch_size": "constexpr",
                "input_element_count": "constexpr",
                "output_element_count": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "batch_size": specialization.batch_size,
                "input_element_count": specialization.input_element_count,
                "output_element_count": specialization.output_element_count,
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


def policy_map(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    mapping: SymbolHandle,
    specialization: PolicyMapSpecialization,
) -> None:
    """Append symbol-backed attention-policy gathering to an executable graph."""
    if output is input_:
        message = "policy mapping cannot operate in place"
        raise ValueError(message)
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_policy_map, specialization)
    builder.call(kernel, output, input_, mapping, readonly=(input_,))
