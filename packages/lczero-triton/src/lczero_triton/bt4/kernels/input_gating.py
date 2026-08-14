"""Multiplicative and additive input-gating kernel family."""

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
    key=["batch_size", "square_count", "channel_count"],
    cache_results=True,
)
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


def _element_count(configuration: Mapping[str, object]) -> int:
    """Return the tensor element count from kernel configuration values."""
    return (
        cast("int", configuration["batch_size"])
        * cast("int", configuration["square_count"])
        * cast("int", configuration["channel_count"])
    )


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat gating grid for a tuning candidate."""
    block_size = cast("int", configuration["block_size"])
    return ((_element_count(configuration) + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_input_gating(
    specialization: InputGatingSpecialization,
) -> KernelArtifact:
    """Autotune and compile one ONNX-layout FP16 input-gating specialization."""
    validate_active_architecture(specialization.architecture)
    element_count = (
        specialization.batch_size
        * specialization.square_count
        * specialization.channel_count
    )
    output = torch.empty(element_count, dtype=torch.float16, device="cuda")
    input_ = torch.zeros(element_count, dtype=torch.float16, device="cuda")
    gate_element_count = specialization.square_count * specialization.channel_count
    multiplicative_gate = torch.zeros(
        gate_element_count,
        dtype=torch.float16,
        device="cuda",
    )
    additive_gate = torch.zeros(
        gate_element_count,
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _input_gating_kernel[_autotune_grid](
        output,
        input_,
        multiplicative_gate,
        additive_gate,
        specialization.batch_size,
        specialization.square_count,
        specialization.channel_count,
    )
    selected = _input_gating_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, element_count),
        parameters=(_POINTER, _POINTER, _POINTER, _POINTER),
    )


def input_gating(
    builder: ProgramBuilder,
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
