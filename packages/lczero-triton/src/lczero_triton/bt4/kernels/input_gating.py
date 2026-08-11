"""Multiplicative and additive input-gating kernel family."""

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
def _input_gating_kernel(
    output,
    input_,
    multiplicative_gate,
    additive_gate,
    batch_size: tl.constexpr,
    square_count: tl.constexpr,
    channel_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch_size * square_count * channel_count
    valid = offsets < element_count
    gate_offsets = offsets % (square_count * channel_count)
    inputs = tl.load(input_ + offsets, mask=valid, other=0.0).to(tl.float32)
    multipliers = tl.load(
        multiplicative_gate + gate_offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    additions = tl.load(
        additive_gate + gate_offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(output + offsets, inputs * multipliers + additions, mask=valid)


@dataclass(frozen=True, slots=True)
class InputGatingSpecialization:
    """Immutable ONNX-layout FP16 input-gating specialization."""

    batch_size: int
    square_count: int
    channel_count: int
    architecture: int
    block_size: int = 256
    num_warps: int = 8

    def __post_init__(self) -> None:
        """Validate dimensions and launch configuration."""
        if any(
            value <= 0
            for value in (self.batch_size, self.square_count, self.channel_count)
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


def compile_input_gating(
    specialization: InputGatingSpecialization,
) -> KernelArtifact:
    """Compile one ONNX-layout FP16 input-gating specialization."""
    element_count = (
        specialization.batch_size
        * specialization.square_count
        * specialization.channel_count
    )
    grid = (
        (element_count + specialization.block_size - 1) // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _input_gating_kernel,
            {
                "output": "*fp16",
                "input_": "*fp16",
                "multiplicative_gate": "*fp16",
                "additive_gate": "*fp16",
                "batch_size": "constexpr",
                "square_count": "constexpr",
                "channel_count": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "batch_size": specialization.batch_size,
                "square_count": specialization.square_count,
                "channel_count": specialization.channel_count,
                "block_size": specialization.block_size,
            },
        ),
        target=GPUTarget("cuda", specialization.architecture, _WARP_SIZE),
        options={"num_warps": specialization.num_warps},
    )
    return artifact_from_triton(
        compiled,
        grid=grid,
        parameters=(_POINTER, _POINTER, _POINTER, _POINTER),
    )


def input_gating(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    multiplicative_gate: Buffer,
    additive_gate: Buffer,
    specialization: InputGatingSpecialization,
) -> None:
    """Append input gating, retaining a write when it operates in place."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_input_gating, specialization)
    readonly = [multiplicative_gate, additive_gate]
    if input_ is not output:
        readonly.append(input_)
    builder.call(
        kernel,
        output,
        input_,
        multiplicative_gate,
        additive_gate,
        readonly=readonly,
    )
