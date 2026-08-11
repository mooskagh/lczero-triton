"""Tests for indexed batched attention matrix multiplication."""

from collections.abc import Sequence
from typing import cast

import pytest
import torch
from lc0ex import Buffer, ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.batched_matmul import (
    _OPERATIONS,
    _POLICY_RECORD_SIZE,
    _TILE_CONFIGS,
    BatchedMatmulOperation,
    BatchedMatmulSpecialization,
    _artifact_grid,
    _attention_v_kernel,
    _autotune_grid,
    _qk_kernel,
    batched_matmul,
    compile_batched_matmul,
)

_FP16_ATOL = 3e-2
_FP16_RTOL = 2e-2
_TEST_HEADS = 2
_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare FP16 kernel output against an FP32 matrix reference."""
    torch.testing.assert_close(
        actual,
        expected.to(torch.float16),
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


def test_autotune_keys_cover_every_indexing_decision() -> None:
    """Static shapes and physical layouts participate in tuning lookup."""
    assert _qk_kernel.keys == [
        "operation",
        "batch_count",
        "heads_per_sample",
        "m",
        "n",
        "k",
        "output_batch_stride",
    ]
    assert _attention_v_kernel.keys == [
        "batch_count",
        "heads_per_sample",
        "m",
        "n",
        "k",
    ]
    assert _qk_kernel.cache_results
    assert _attention_v_kernel.cache_results

    qk_configs = {
        (
            config.kwargs["block_m"],
            config.kwargs["block_n"],
            config.kwargs["block_k"],
            config.num_warps,
            config.num_stages,
        )
        for config in _qk_kernel.configs
    }
    attention_v_configs = {
        (
            config.kwargs["block_m"],
            config.kwargs["block_n"],
            config.kwargs["block_k"],
            config.num_warps,
            config.num_stages,
        )
        for config in _attention_v_kernel.configs
    }
    assert qk_configs == set(_TILE_CONFIGS)
    assert attention_v_configs == set(_TILE_CONFIGS)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("body_qk", 0, 1, 1, 1, 1), "positive"),
        (("body_qk", 3, 1, 1, 1, 2), "divisible"),
        (("policy_qk", 2, 1, 1, 1, 2), "one logical matrix"),
        (("policy_qk", 1, 65, 66, 1, 1), "exceeds"),
    ],
)
def test_specialization_rejects_invalid_layouts(
    values: tuple[BatchedMatmulOperation, int, int, int, int, int],
    message: str,
) -> None:
    """Logical matrices must fit their operation's physical layout."""
    operation, batch_count, m, n, k, heads = values
    with pytest.raises(ValueError, match=message):
        BatchedMatmulSpecialization(
            operation,
            batch_count,
            m,
            n,
            k,
            heads,
            80,
        )


def test_specialization_rejects_unknown_operation_and_target() -> None:
    """The operation convention and CUDA target are explicit static inputs."""
    invalid = cast("BatchedMatmulOperation", "unknown")
    with pytest.raises(ValueError, match="unsupported"):
        BatchedMatmulSpecialization(invalid, 1, 1, 1, 1, 1, 80)
    with pytest.raises(ValueError, match="architecture"):
        BatchedMatmulSpecialization("body_qk", 1, 1, 1, 1, 1, 0)


def test_graph_validation_precedes_compilation() -> None:
    """Scale and alias errors do not trigger GPU compilation."""
    builder = ExecutableBuilder()
    execution = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    buffer = execution.temporary_buffer(size_bytes=2, alignment_bytes=2)
    kernels = KernelCache(builder)
    qk = BatchedMatmulSpecialization("body_qk", 1, 1, 1, 1, 1, 80)
    attention_v = BatchedMatmulSpecialization("body_attention_v", 1, 1, 1, 1, 1, 80)

    with pytest.raises(ValueError, match="require scale"):
        batched_matmul(builder, kernels, buffer, buffer, buffer, qk)
    with pytest.raises(ValueError, match="does not"):
        batched_matmul(
            builder,
            kernels,
            buffer,
            buffer,
            buffer,
            attention_v,
            scale=buffer,
        )
    with pytest.raises(ValueError, match="cannot alias"):
        batched_matmul(
            builder,
            kernels,
            buffer,
            buffer,
            buffer,
            qk,
            scale=buffer,
        )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_body_qk_matches_explicit_head_reshape() -> None:
    """Body QK reads interleaved heads and emits contiguous head matrices."""
    torch.manual_seed(1)
    samples, heads, tokens, depth = 169, 32, 64, 32
    queries = (
        torch.randn(
            (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    keys = torch.randn_like(queries) * 0.05
    scale = torch.tensor([0.173], dtype=torch.float16, device="cuda")
    result = torch.empty(
        (samples, heads, tokens, tokens), dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["body_qk"],
        samples * heads,
        heads,
        tokens,
        tokens,
        depth,
        tokens * tokens,
    )

    query_heads = queries.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    key_heads = keys.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected = (
        torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
        * scale.float()
    )
    _assert_close(result, expected)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_attention_times_v_writes_directly_to_merged_heads() -> None:
    """Attention output lands in physical token-major body storage."""
    torch.manual_seed(2)
    samples, heads, tokens, depth = 169, 32, 64, 32
    attention = (
        torch.randn(
            (samples, heads, tokens, tokens), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    values = (
        torch.randn(
            (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    result = torch.empty_like(values)

    _attention_v_kernel[_autotune_grid](
        result,
        attention,
        values,
        samples * heads,
        heads,
        tokens,
        depth,
        tokens,
    )

    value_heads = values.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected_heads = torch.matmul(attention.float(), value_heads.float())
    expected = expected_heads.permute(0, 2, 1, 3).reshape_as(values)
    _assert_close(result, expected)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_policy_qk_preserves_promotion_slots() -> None:
    """Policy QK scales the matrix prefix and leaves all 192 promotions intact."""
    torch.manual_seed(3)
    samples, tokens, width = 169, 64, 1024
    queries = (
        torch.randn((samples, tokens, width), dtype=torch.float16, device="cuda") * 0.02
    )
    keys = torch.randn_like(queries) * 0.02
    scale = torch.tensor([0.037], dtype=torch.float16, device="cuda")
    sentinel = -123.0
    result = torch.full(
        (samples, _POLICY_RECORD_SIZE), sentinel, dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["policy_qk"],
        samples,
        1,
        tokens,
        tokens,
        width,
        _POLICY_RECORD_SIZE,
    )

    expected = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
    expected *= scale.float()
    _assert_close(result[:, : tokens * tokens].reshape_as(expected), expected)
    assert torch.equal(
        result[:, tokens * tokens :],
        torch.full(
            (samples, _POLICY_RECORD_SIZE - tokens * tokens),
            sentinel,
            dtype=torch.float16,
            device="cuda",
        ),
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_body_qk_crosses_from_last_head_to_next_sample() -> None:
    """Logical matrices 31 and 32 use different samples without overlap."""
    torch.manual_seed(4)
    samples, heads, tokens, depth = 2, 32, 16, 16
    queries = torch.randn(
        (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
    )
    keys = torch.randn_like(queries)
    scale = torch.ones(1, dtype=torch.float16, device="cuda")
    result = torch.empty(
        (samples * heads, tokens, tokens), dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["body_qk"],
        samples * heads,
        heads,
        tokens,
        tokens,
        depth,
        tokens * tokens,
    )

    query_heads = queries.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    key_heads = keys.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected = torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
    expected = expected.reshape(samples * heads, tokens, tokens)
    _assert_close(result[30:34], expected[30:34])


_ARTIFACT_SPECIALIZATIONS = (
    BatchedMatmulSpecialization("body_qk", 4, 16, 16, 16, 2, 1),
    BatchedMatmulSpecialization("body_attention_v", 4, 16, 16, 16, 2, 1),
    BatchedMatmulSpecialization("policy_qk", 2, 16, 16, 16, 1, 1),
)


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("template", _ARTIFACT_SPECIALIZATIONS)
def test_compilation_captures_selected_launch_and_abi(
    template: BatchedMatmulSpecialization,
) -> None:
    """Each physical layout serializes its tuned grid and pointer ABI."""
    specialization = BatchedMatmulSpecialization(
        template.operation,
        template.batch_count,
        template.m,
        template.n,
        template.k,
        template.heads_per_sample,
        _architecture(),
    )

    artifact = compile_batched_matmul(specialization)
    kernel = (
        _attention_v_kernel
        if specialization.operation == "body_attention_v"
        else _qk_kernel
    )
    selected = kernel.best_config
    pointer_count = 3 if specialization.operation == "body_attention_v" else 4

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * pointer_count
    assert artifact.grid == _artifact_grid(
        selected.kwargs,
        specialization.m,
        specialization.n,
        specialization.batch_count,
    )
    assert artifact.block == (selected.num_warps * 32, 1, 1)


def _external_buffer(
    builder: ExecutableBuilder,
    name: str,
    shape: Sequence[int],
    *,
    writable: bool = False,
    lifetime: lc0ex_pb2.Allocation.Lifetime = (lc0ex_pb2.Allocation.LIFETIME_EXECUTION),
) -> Buffer:
    """Declare one test execution buffer."""
    allocation = builder.allocation(lifetime)
    return allocation.external_buffer(
        name=name,
        shape=shape,
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=writable,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("operation", ["body_qk", "body_attention_v", "policy_qk"])
def test_graph_call_has_only_physical_operands(
    operation: BatchedMatmulOperation,
) -> None:
    """The graph ABI allocates no pointer arrays or head-layout temporaries."""
    builder = ExecutableBuilder()
    heads = _TEST_HEADS if operation != "policy_qk" else 1
    batch_count = 2 * heads
    specialization = BatchedMatmulSpecialization(
        operation, batch_count, 16, 16, 16, heads, _architecture()
    )
    output_shape = (
        (2, _POLICY_RECORD_SIZE)
        if operation == "policy_qk"
        else (2, 16, heads * 16)
        if operation == "body_attention_v"
        else (batch_count, 16, 16)
    )
    left_shape = (
        (batch_count, 16, 16)
        if operation == "body_attention_v"
        else (2, 16, heads * 16)
    )
    right_shape = (2, 16, heads * 16)
    output = _external_buffer(builder, "output", output_shape, writable=True)
    left = _external_buffer(builder, "left", left_shape)
    right = _external_buffer(builder, "right", right_shape)
    scale_name = (
        "/policy/scale/w" if operation == "policy_qk" else "/encoder0/mha/QK/scale/w"
    )
    scale = (
        None
        if operation == "body_attention_v"
        else _external_buffer(
            builder,
            scale_name,
            (1,),
            lifetime=lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
        )
    )

    batched_matmul(
        builder,
        KernelCache(builder),
        output,
        left,
        right,
        specialization,
        scale=scale,
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.allocation_idx, buffer.allocation_offset)
        for buffer in executable.buffers
    }
    arguments = [
        (argument.allocation.index, argument.allocation.offset)
        for argument in node.arguments
    ]
    expected_names = ["output", "left", "right"]
    if scale is not None:
        expected_names.append(scale_name)

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert set(locations) == set(expected_names)
    assert len(executable.allocations) == len(expected_names)
    assert arguments == [locations[name] for name in expected_names]


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_graph_dependencies_follow_physical_buffer_flow() -> None:
    """A subsequent QK read depends on the producer of its query buffer."""
    builder = ExecutableBuilder()
    specialization = BatchedMatmulSpecialization(
        "body_qk", 4, 16, 16, 16, _TEST_HEADS, _architecture()
    )
    kernels = KernelCache(builder)
    first_output = _external_buffer(builder, "first_output", (4, 16, 16), writable=True)
    final_output = _external_buffer(builder, "final_output", (4, 16, 16), writable=True)
    queries = _external_buffer(builder, "queries", (2, 16, 32))
    keys = _external_buffer(builder, "keys", (2, 16, 32))
    scale = _external_buffer(
        builder,
        "/encoder0/mha/QK/scale/w",
        (1,),
        lifetime=lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
    )

    batched_matmul(
        builder,
        kernels,
        first_output,
        queries,
        keys,
        specialization,
        scale=scale,
    )
    batched_matmul(
        builder,
        kernels,
        final_output,
        first_output,
        keys,
        specialization,
        scale=scale,
    )

    nodes = builder.build().programs[0].nodes
    assert list(nodes[0].dependencies) == []
    assert list(nodes[1].dependencies) == [0]


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_compilation_rejects_a_different_active_architecture() -> None:
    """Autotuning cannot silently emit a CUBIN for a different target."""
    specialization = BatchedMatmulSpecialization(
        "body_qk", 1, 16, 16, 16, 1, _architecture() + 1
    )
    with pytest.raises(ValueError, match="active device"):
        compile_batched_matmul(specialization)
