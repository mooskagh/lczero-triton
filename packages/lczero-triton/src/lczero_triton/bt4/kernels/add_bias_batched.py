"""Batched FP16 bias-broadcast kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import elementwise_configs
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


@triton.autotune(
    configs=elementwise_configs(),
    key=["batch_count", "row_count", "channel_count", "activation"],
    cache_results=True,
)
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


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat bias-addition grid for a tuning candidate."""
    element_count = (
        cast("int", configuration["batch_count"])
        * cast("int", configuration["row_count"])
        * cast("int", configuration["channel_count"])
    )
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_add_bias_batched(
    specialization: AddBiasBatchedSpecialization,
) -> KernelArtifact:
    """Autotune and compile one batched FP16 bias-broadcast specialization."""
    element_count = (
        specialization.batch_count
        * specialization.row_count
        * specialization.channel_count
    )
    output = torch.empty(element_count, dtype=torch.float16, device="cuda")
    input_ = torch.zeros(element_count, dtype=torch.float16, device="cuda")
    bias = torch.zeros(
        specialization.batch_count * specialization.channel_count,
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _add_bias_batched_kernel[_autotune_grid](
        output,
        input_,
        bias,
        specialization.batch_count,
        specialization.row_count,
        specialization.channel_count,
        _ACTIVATIONS[specialization.activation],
    )
    selected = _add_bias_batched_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, element_count),
        parameters=(_POINTER, _POINTER, _POINTER),
    )


def add_bias_batched(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    bias: Buffer,
    specialization: AddBiasBatchedSpecialization,
) -> None:
    """Append batched bias broadcasting, retaining valid in-place writes."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_add_bias_batched, specialization)
    readonly = [source for source in (input_, bias) if source is not output]
    builder.call(kernel, output, input_, bias, readonly=readonly)
