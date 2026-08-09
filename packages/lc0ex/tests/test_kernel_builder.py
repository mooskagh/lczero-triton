"""Tests for generic kernel registration and graph construction."""

import pytest
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2

TARGET_ARCHITECTURE = "sm_80"
DYNAMIC_SHARED_MEMORY_BYTES = 256
EXPECTED_CALL_COUNT = 2
TEMPORARY_BUFFER_SIZE_BYTES = 8
TWO_TEMPORARY_BUFFER_SIZE_BYTES = 16
F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


def _artifact(
    binary_name: str = "matmul_binary",
    binary_data: bytes = b"fake cubin",
    parameters: tuple[lc0ex_pb2.ParameterType, ...] = (POINTER, POINTER, POINTER),
) -> KernelArtifact:
    return KernelArtifact(
        binary_name=binary_name,
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=binary_data,
        function="matmul_exported",
        parameters=parameters,
        grid=(2, 3, 1),
        block=(128, 1, 1),
        dynamic_shared_memory_bytes=DYNAMIC_SHARED_MEMORY_BYTES,
    )


def _builder() -> ExecutableBuilder:
    return ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )


def _buffers(
    builder: ExecutableBuilder,
    prefix: str = "",
) -> tuple[Buffer, Buffer, Buffer]:
    return (
        builder.buffer(f"{prefix}a", (32, 16), F16),
        builder.buffer(f"{prefix}b", (16, 48), F16),
        builder.buffer(f"{prefix}c", (32, 48), F16),
    )


def test_add_kernel_and_call_serializes_generic_metadata() -> None:
    """The builder serializes an already compiled kernel without a compiler."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())
    builder.call("matmul", *_buffers(builder))

    executable = builder.build()

    assert len(executable.binaries) == 1
    binary = executable.binaries[0]
    assert binary.name == "matmul_binary"
    assert binary.format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert binary.data == b"fake cubin"

    assert len(executable.kernels) == 1
    kernel = executable.kernels[0]
    assert kernel.name == "matmul"
    assert kernel.binary == binary.name
    assert kernel.function == "matmul_exported"
    assert list(kernel.parameters) == [POINTER, POINTER, POINTER]

    assert len(executable.programs) == 1
    node = executable.programs[0].nodes[0]
    assert node.name == "node_0"
    assert node.kernel == "matmul"
    assert list(node.arguments) == ["a", "b", "c"]
    assert list(node.grid) == [2, 3, 1]
    assert list(node.block) == [128, 1, 1]
    assert node.dynamic_shared_memory_bytes == DYNAMIC_SHARED_MEMORY_BYTES


def test_repeated_calls_reuse_one_registered_kernel() -> None:
    """Multiple calls reference one registered kernel and binary."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())
    builder.call("matmul", *_buffers(builder, "first_"))
    builder.call("matmul", *_buffers(builder, "second_"))

    executable = builder.build()

    assert len(executable.binaries) == 1
    assert len(executable.kernels) == 1
    assert len(executable.programs[0].nodes) == EXPECTED_CALL_COUNT
    assert list(executable.programs[0].nodes[1].arguments) == [
        "second_a",
        "second_b",
        "second_c",
    ]


def test_readonly_calls_to_same_buffer_have_no_dependency() -> None:
    """Read-only accesses to one buffer may run concurrently."""
    builder = _builder()
    builder.add_kernel("read", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16)
    builder.call("read", buffer, readonly=(buffer,))
    builder.call("read", buffer, readonly=(buffer,))

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_persistent_buffers_are_readonly_by_default() -> None:
    """Persistent buffers are read-only unless declared writable."""
    builder = _builder()
    builder.add_kernel("read", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16)
    builder.call("read", buffer)
    builder.call("read", buffer)

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_readonly_call_depends_on_previous_writer() -> None:
    """A reader waits for the most recent writer of its buffer."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16, writable=True)
    builder.call("access", buffer)
    builder.call("access", buffer, readonly=(buffer,))

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == ["node_0"]


def test_writer_depends_on_all_readers_since_previous_write() -> None:
    """A writer waits for every outstanding read of its buffer."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16, writable=True)
    builder.call("access", buffer, readonly=(buffer,))
    builder.call("access", buffer, readonly=(buffer,))
    builder.call("access", buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[2].dependencies) == ["node_0", "node_1"]


def test_redundant_transitive_dependencies_are_removed() -> None:
    """A dependency implied through another dependency is not emitted."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16, writable=True)
    builder.call("access", buffer)
    builder.call("access", buffer, readonly=(buffer,))
    builder.call("access", buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == ["node_0"]
    assert list(nodes[2].dependencies) == ["node_1"]


def test_dependencies_are_reduced_across_buffers() -> None:
    """Reduction removes dependencies implied through another buffer's writer."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER, POINTER)))
    first = builder.buffer("first", (1,), F16, writable=True)
    second = builder.buffer("second", (1,), F16, writable=True)
    builder.call("access", first, second, readonly=(second,))
    builder.call("access", first, second, readonly=(first,))
    builder.call("access", first, second)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == ["node_0"]
    assert list(nodes[2].dependencies) == ["node_1"]


def test_sequential_temporary_buffers_reuse_one_allocation_range() -> None:
    """Temporaries whose accesses are ordered by the graph share storage."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER, POINTER)))
    first = builder.tmp_buffer((4,), F16)
    second = builder.tmp_buffer((4,), F16)
    bridge = builder.buffer("bridge", (1,), F16, writable=True)
    builder.call("access", first, bridge)
    builder.call("access", bridge, second, readonly=(bridge,))

    executable = builder.build()
    buffers = {buffer.name: buffer for buffer in executable.buffers}

    assert executable.allocations[-1].name == "execution"
    assert executable.allocations[-1].size_bytes == TEMPORARY_BUFFER_SIZE_BYTES
    assert buffers[first.name].allocation_block.offset_bytes == 0
    assert buffers[second.name].allocation_block.offset_bytes == 0
    assert list(executable.programs[0].nodes[1].dependencies) == ["node_0"]


def test_independent_temporary_buffers_do_not_reuse_storage() -> None:
    """Independent nodes may execute concurrently and therefore cannot alias."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    first = builder.tmp_buffer((4,), F16)
    second = builder.tmp_buffer((4,), F16)
    builder.call("access", first)
    builder.call("access", second)

    executable = builder.build()
    buffers = {buffer.name: buffer for buffer in executable.buffers}

    assert executable.allocations[0].name == "execution"
    assert executable.allocations[0].size_bytes == TWO_TEMPORARY_BUFFER_SIZE_BYTES
    assert buffers[first.name].allocation_block.offset_bytes == 0
    assert (
        buffers[second.name].allocation_block.offset_bytes
        == TEMPORARY_BUFFER_SIZE_BYTES
    )
    assert not executable.programs[0].nodes[1].dependencies


def test_temporary_buffers_used_by_one_node_do_not_reuse_storage() -> None:
    """Buffers accessed by the same invocation have overlapping lifetimes."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER, POINTER)))
    first = builder.tmp_buffer((4,), F16)
    second = builder.tmp_buffer((4,), F16)
    builder.call("access", first, second)

    executable = builder.build()
    buffers = {buffer.name: buffer for buffer in executable.buffers}

    assert executable.allocations[0].size_bytes == TWO_TEMPORARY_BUFFER_SIZE_BYTES
    assert buffers[first.name].allocation_block.offset_bytes != (
        buffers[second.name].allocation_block.offset_bytes
    )


def test_unused_registered_kernel_is_serialized() -> None:
    """Adding a kernel registers it even when no graph node uses it."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())

    executable = builder.build()

    assert len(executable.binaries) == 1
    assert len(executable.kernels) == 1
    assert not executable.programs


def test_equal_kernel_can_be_registered_twice() -> None:
    """Repeated registration of the same artifact is idempotent."""
    builder = _builder()
    artifact = _artifact()

    assert builder.add_kernel("matmul", artifact) is builder
    assert builder.add_kernel("matmul", artifact) is builder
    assert len(builder.build().kernels) == 1


def test_different_kernel_with_same_name_is_rejected() -> None:
    """A name cannot refer to two different kernel artifacts."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())

    with pytest.raises(ValueError, match="already registered differently"):
        builder.add_kernel("matmul", _artifact(binary_data=b"different cubin"))


def test_different_binary_with_same_name_is_rejected() -> None:
    """A binary name cannot refer to different binary contents."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())

    with pytest.raises(ValueError, match=r"binary .* already registered"):
        builder.add_kernel(
            "other",
            _artifact(binary_data=b"different cubin"),
        )


def test_call_requires_registered_kernel() -> None:
    """Graph calls must refer to a registered kernel."""
    with pytest.raises(KeyError):
        _builder().call("missing")


def test_call_rejects_foreign_buffer_handles() -> None:
    """Equal-looking handles from another builder do not satisfy ownership."""
    builder = _builder()
    foreign = _builder()
    builder.add_kernel("matmul", _artifact())

    with pytest.raises(ValueError, match="belong to this executable builder"):
        builder.call("matmul", *_buffers(foreign))


def test_call_rejects_foreign_readonly_buffer() -> None:
    """Read-only buffers must be owned by the executable builder."""
    builder = _builder()
    foreign = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    buffer = builder.buffer("buffer", (1,), F16)

    with pytest.raises(ValueError, match="read-only buffers must belong"):
        builder.call("access", buffer, readonly=(foreign.buffer("foreign", (1,), F16),))


def test_call_rejects_readonly_buffer_not_passed_to_kernel() -> None:
    """Read-only buffers must be among the invocation arguments."""
    builder = _builder()
    builder.add_kernel("access", _artifact(parameters=(POINTER,)))
    argument = builder.buffer("argument", (1,), F16)
    readonly = builder.buffer("readonly", (1,), F16)

    with pytest.raises(ValueError, match="read-only buffers must be kernel arguments"):
        builder.call("access", argument, readonly=(readonly,))


def test_call_rejects_abi_argument_count_mismatch() -> None:
    """A call must provide every argument declared by the kernel ABI."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())
    a, b, _ = _buffers(builder)

    with pytest.raises(ValueError, match="argument count"):
        builder.call("matmul", a, b)


def test_repeated_builds_have_independent_kernel_messages() -> None:
    """Every build serializes registered kernels into fresh protobuf messages."""
    builder = _builder()
    builder.add_kernel("matmul", _artifact())
    builder.call("matmul", *_buffers(builder))

    first = builder.build()
    second = builder.build()

    assert first == second
    assert second.binaries[0] is not first.binaries[0]
    assert second.kernels[0] is not first.kernels[0]
    assert second.programs[0].nodes[0] is not first.programs[0].nodes[0]
