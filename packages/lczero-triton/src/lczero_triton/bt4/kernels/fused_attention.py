"""Autotuned fused FP16 attention (QK + Smolgen + Softmax + V) family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_SQUARE_COUNT = tl.constexpr(64)
_TWICE_HALF_MAX = tl.constexpr(131008.0)

_ATTENTION_CONFIGS = (
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
)


@triton.autotune(
    configs=list(_ATTENTION_CONFIGS),
    key=["batch_count", "model_width", "head_depth", "heads_per_sample"],
    cache_results=True,
)
@triton.jit
def _fused_attention_kernel(
    output,
    queries,
    keys,
    values,
    smolgen,
    scale_ptr,
    batch_count: tl.constexpr,
    model_width: tl.constexpr,
    head_depth: tl.constexpr,
    heads_per_sample: tl.constexpr,
    block_d: tl.constexpr,
) -> None:
    """Compute fused QK, Smolgen bias, 64-way softmax, and V attention per head."""
    matrix = tl.program_id(0)
    if matrix >= batch_count:
        return

    sample = matrix // heads_per_sample
    head = matrix % heads_per_sample
    scale = tl.load(scale_ptr).to(tl.float32)

    offs_m = tl.arange(0, _SQUARE_COUNT)
    offs_d = tl.arange(0, block_d)
    mask_d = offs_d < head_depth

    # Base pointers
    head_offset = sample * (_SQUARE_COUNT * model_width) + head * head_depth
    q_ptrs = queries + head_offset + offs_m[:, None] * model_width + offs_d[None, :]
    k_ptrs = keys + head_offset + offs_m[:, None] * model_width + offs_d[None, :]

    q = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float16)
    k = tl.load(k_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float16)

    # Q @ K^T: (64, block_d) @ (block_d, 64) -> (64, 64)
    k_t = tl.trans(k)
    qk = tl.dot(q, k_t, out_dtype=tl.float32) * scale

    # Smolgen addition
    offs_n = tl.arange(0, _SQUARE_COUNT)
    smolgen_base = matrix * (_SQUARE_COUNT * _SQUARE_COUNT)
    smolgen_ptrs = (
        smolgen + smolgen_base + offs_m[:, None] * _SQUARE_COUNT + offs_n[None, :]
    )
    smolgen_vals = tl.load(smolgen_ptrs).to(tl.float32)
    logits = qk + smolgen_vals

    # Clamping
    is_nan = logits != logits  # noqa: PLR0124  # Device-side NaN test.
    clamped = tl.minimum(tl.maximum(logits, -_TWICE_HALF_MAX), _TWICE_HALF_MAX)
    logits = tl.where(is_nan, logits, clamped)

    # Softmax over columns
    max_val = tl.max(logits, axis=1)
    exp_val = tl.exp(logits - max_val[:, None])
    sum_val = tl.sum(exp_val, axis=1)
    probs = (exp_val / sum_val[:, None]).to(tl.float16)

    # Load V and compute output: (64, 64) @ (64, block_d) -> (64, block_d)
    v_ptrs = values + head_offset + offs_m[:, None] * model_width + offs_d[None, :]
    v = tl.load(v_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float16)
    out_val = tl.dot(probs, v, out_dtype=tl.float32).to(tl.float16)

    out_ptrs = output + head_offset + offs_m[:, None] * model_width + offs_d[None, :]
    tl.store(out_ptrs, out_val, mask=mask_d[None, :])


@dataclass(frozen=True, slots=True)
class FusedAttentionSpecialization:
    """Immutable fused attention specialization."""

    batch_count: int
    model_width: int
    head_depth: int
    heads_per_sample: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the 1D launch grid for a tuning candidate."""
    batch_count = cast("int", configuration["batch_count"])
    return (batch_count,)


def compile_fused_attention(
    specialization: FusedAttentionSpecialization,
) -> KernelArtifact:
    """Autotune and compile one fused attention specialization."""
    output = torch.empty(
        (specialization.batch_count // specialization.heads_per_sample * 64, specialization.model_width),
        dtype=torch.float16,
        device="cuda",
    )
    queries = torch.zeros_like(output)
    keys = torch.zeros_like(output)
    values = torch.zeros_like(output)
    smolgen = torch.zeros(
        (specialization.batch_count, 64, 64),
        dtype=torch.float16,
        device="cuda",
    )
    scale = torch.ones(1, dtype=torch.float16, device="cuda")
    block_d = max(16, 1 << (specialization.head_depth - 1).bit_length())

    compiled = _fused_attention_kernel[_autotune_grid](
        output,
        queries,
        keys,
        values,
        smolgen,
        scale,
        specialization.batch_count,
        specialization.model_width,
        specialization.head_depth,
        specialization.heads_per_sample,
        block_d,
    )
    return artifact_from_triton(
        compiled,
        grid=(specialization.batch_count, 1, 1),
        parameters=(_POINTER, _POINTER, _POINTER, _POINTER, _POINTER, _POINTER),
    )


def fused_attention(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    smolgen: Buffer,
    scale: Buffer,
    specialization: FusedAttentionSpecialization,
) -> None:
    """Append one fused attention operation."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_fused_attention, specialization)
    builder.call(
        kernel,
        output,
        queries,
        keys,
        values,
        smolgen,
        scale,
        readonly=[queries, keys, values, smolgen, scale],
    )
