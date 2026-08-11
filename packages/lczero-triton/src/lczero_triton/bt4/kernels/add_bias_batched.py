"""Batched FP16 bias-broadcast kernel family."""

from dataclasses import dataclass
from typing import Literal

import triton
import triton.language as tl
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lczero_triton.bt4.kernels._cache import KernelCache

Activation = Literal["none", "mish"]

_ACTIVATION_NONE = 0
_ACTIVATION_MISH = tl.constexpr(1)
_ACTIVATIONS: dict[Activation, int] = {
    "none": _ACTIVATION_NONE,
    "mish": 1,
}
_MISH_BRANCH = tl.constexpr(-0.6)
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_SIZE = 32


@triton.jit
def _add_bias_batched_kernel(
    output,
    input_,
    bias,
    batch_count: tl.constexpr,
    row_count: tl.constexpr,
    channel_count: tl.constexpr,
    activation: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch_count * row_count * channel_count
    valid = offsets < element_count
    batch = offsets // (row_count * channel_count)
    channel = offsets % channel_count
    values = tl.load(input_ + offsets, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(
        bias + batch * channel_count + channel,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    values += biases
    if activation == _ACTIVATION_MISH:
        exponential = tl.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    tl.store(output + offsets, values, mask=valid)


@dataclass(frozen=True, slots=True)
class AddBiasBatchedSpecialization:
    """Immutable batched FP16 bias-broadcast specialization."""

    batch_count: int
    row_count: int
    channel_count: int
    activation: Activation
    architecture: int
    block_size: int = 256
    num_warps: int = 8

    def __post_init__(self) -> None:
        """Validate dimensions, activation, and launch configuration."""
        if any(
            value <= 0
            for value in (self.batch_count, self.row_count, self.channel_count)
        ):
            message = "tensor dimensions must be positive"
            raise ValueError(message)
        if self.activation not in _ACTIVATIONS:
            message = f"unsupported activation: {self.activation!r}"
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


def compile_add_bias_batched(
    specialization: AddBiasBatchedSpecialization,
) -> KernelArtifact:
    """Compile one batched FP16 bias-broadcast specialization."""
    element_count = (
        specialization.batch_count
        * specialization.row_count
        * specialization.channel_count
    )
    grid = (
        (element_count + specialization.block_size - 1) // specialization.block_size,
        1,
        1,
    )
    compiled = triton.compile(
        ASTSource(
            _add_bias_batched_kernel,
            {
                "output": "*fp16",
                "input_": "*fp16",
                "bias": "*fp16",
                "batch_count": "constexpr",
                "row_count": "constexpr",
                "channel_count": "constexpr",
                "activation": "constexpr",
                "block_size": "constexpr",
            },
            constexprs={
                "batch_count": specialization.batch_count,
                "row_count": specialization.row_count,
                "channel_count": specialization.channel_count,
                "activation": _ACTIVATIONS[specialization.activation],
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


def add_bias_batched(
    builder: ExecutableBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    bias: Buffer,
    specialization: AddBiasBatchedSpecialization,
) -> None:
    """Append batched bias broadcasting, retaining valid in-place writes."""
    if output is bias and specialization.row_count != 1:
        message = "output cannot alias a bias broadcast across rows"
        raise ValueError(message)
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_add_bias_batched, specialization)
    readonly = []
    if input_ is not output:
        readonly.append(input_)
    if bias is not output:
        readonly.append(bias)
    builder.call(kernel, output, input_, bias, readonly=readonly)
