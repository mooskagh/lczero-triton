"""Tests for generic kernel registration and graph construction."""

import pytest
from lc0ex import Allocation, Buffer, ExecutableBuilder, KernelArtifact, KernelHandle
from lc0ex.proto import lc0ex_pb2

TARGET_ARCHITECTURE = "sm_80"
DYNAMIC_SHARED_MEMORY_BYTES = 256
EXPECTED_CALL_COUNT = 2
EXPECTED_BINARY_COUNT = 2
TEMPORARY_BUFFER_SIZE_BYTES = 8
TWO_TEMPORARY_BUFFER_SIZE_BYTES = 16
NAMED_OUTPUT_SIZE_BYTES = 2
F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
U32 = lc0ex_pb2.PARAMETER_TYPE_U32

_PERSISTENT_ALLOCATIONS: dict[ExecutableBuilder, Allocation] = {}
_EXECUTION_ALLOCATIONS: dict[ExecutableBuilder, Allocation] = {}


def _artifact(
    binary_data: bytes = b"fake cubin",
    function: str = "matmul_exported",
    parameters: tuple[lc0ex_pb2.ParameterType, ...] = (POINTER, POINTER, POINTER),
) -> KernelArtifact:
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=binary_data,
        function=function,
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


def _persistent(builder: ExecutableBuilder) -> Allocation:
    """Return the test's shared persistent allocation for *builder*."""
    allocation = _PERSISTENT_ALLOCATIONS.get(builder)
    if allocation is None:
        allocation = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
        _PERSISTENT_ALLOCATIONS[builder] = allocation
    return allocation


def _execution(builder: ExecutableBuilder) -> Allocation:
    """Return the test's shared execution allocation for *builder*."""
    allocation = _EXECUTION_ALLOCATIONS.get(builder)
    if allocation is None:
        allocation = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
        _EXECUTION_ALLOCATIONS[builder] = allocation
    return allocation


def _buffers(
    builder: ExecutableBuilder,
    prefix: str = "",
) -> tuple[Buffer, Buffer, Buffer]:
    persistent = _persistent(builder)
    return (
        persistent.buffer((32, 16), F16, name=f"{prefix}a"),
        persistent.buffer((16, 48), F16, name=f"{prefix}b"),
        persistent.buffer((32, 48), F16, name=f"{prefix}c"),
    )


def test_add_kernel_and_call_serializes_generic_metadata() -> None:
    """The builder serializes an already compiled kernel without a compiler."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact())
    builder.call(kernel_handle, *_buffers(builder))

    executable = builder.build()

    assert len(executable.binaries) == 1
    binary = executable.binaries[0]
    assert binary.format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert binary.data == b"fake cubin"

    assert len(executable.kernels) == 1
    kernel = executable.kernels[0]
    assert kernel.binary_idx == 0
    assert kernel.function == "matmul_exported"
    assert list(kernel.parameters) == [POINTER, POINTER, POINTER]

    assert len(executable.programs) == 1
    node = executable.programs[0].nodes[0]
    assert node.kernel_idx == 0
    assert [argument.allocation_idx for argument in node.arguments] == [0, 0, 0]
    assert [argument.allocation_offset for argument in node.arguments] == [
        0,
        1024,
        2560,
    ]
    assert list(node.grid) == [2, 3, 1]
    assert list(node.block) == [128, 1, 1]
    assert node.dynamic_shared_memory_bytes == DYNAMIC_SHARED_MEMORY_BYTES


def test_repeated_calls_reuse_one_registered_kernel() -> None:
    """Multiple calls reference one registered kernel and binary."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact())
    builder.call(kernel_handle, *_buffers(builder, "first_"))
    builder.call(kernel_handle, *_buffers(builder, "second_"))

    executable = builder.build()

    assert len(executable.binaries) == 1
    assert len(executable.kernels) == 1
    assert len(executable.programs[0].nodes) == EXPECTED_CALL_COUNT
    second = executable.programs[0].nodes[1]
    assert second.kernel_idx == 0
    assert [argument.allocation_idx for argument in second.arguments] == [0, 0, 0]
    assert [argument.allocation_offset for argument in second.arguments] == [
        5632,
        6656,
        8192,
    ]


def test_readonly_calls_to_same_buffer_have_no_dependency() -> None:
    """Read-only accesses to one buffer may run concurrently."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer")
    builder.call(kernel_handle, buffer, readonly=(buffer,))
    builder.call(kernel_handle, buffer, readonly=(buffer,))

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_persistent_buffers_are_readonly_by_default() -> None:
    """Persistent buffers are read-only unless declared writable."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer")
    builder.call(kernel_handle, buffer)
    builder.call(kernel_handle, buffer)

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_readonly_call_depends_on_previous_writer() -> None:
    """A reader waits for the most recent writer of its buffer."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer", writable=True)
    builder.call(kernel_handle, buffer)
    builder.call(kernel_handle, buffer, readonly=(buffer,))

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == [0]


def test_writer_depends_on_all_readers_since_previous_write() -> None:
    """A writer waits for every outstanding read of its buffer."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer", writable=True)
    builder.call(kernel_handle, buffer, readonly=(buffer,))
    builder.call(kernel_handle, buffer, readonly=(buffer,))
    builder.call(kernel_handle, buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[2].dependencies) == [0, 1]


def test_redundant_transitive_dependencies_are_removed() -> None:
    """A dependency implied through another dependency is not emitted."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer", writable=True)
    builder.call(kernel_handle, buffer)
    builder.call(kernel_handle, buffer, readonly=(buffer,))
    builder.call(kernel_handle, buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == [0]
    assert list(nodes[2].dependencies) == [1]


def test_dependencies_are_reduced_across_buffers() -> None:
    """Reduction removes dependencies implied through another buffer's writer."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    first = _persistent(builder).buffer((1,), F16, name="first", writable=True)
    second = _persistent(builder).buffer((1,), F16, name="second", writable=True)
    builder.call(kernel_handle, first, second, readonly=(second,))
    builder.call(kernel_handle, first, second, readonly=(first,))
    builder.call(kernel_handle, first, second)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == [0]
    assert list(nodes[2].dependencies) == [1]


def test_sequential_unnamed_buffers_reuse_one_allocation_range() -> None:
    """Ordered internal buffers share an execution allocation range."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    first = _execution(builder).buffer((4,), F16)
    second = _execution(builder).buffer((4,), F16)
    bridge = _persistent(builder).buffer((1,), F16, name="bridge", writable=True)
    builder.call(kernel_handle, first, bridge)
    builder.call(kernel_handle, bridge, second, readonly=(bridge,))

    executable = builder.build()
    nodes = executable.programs[0].nodes

    assert len(executable.buffers) == 1
    assert executable.allocations[0].size_bytes == TEMPORARY_BUFFER_SIZE_BYTES
    assert nodes[0].arguments[0].allocation_idx == 0
    assert nodes[0].arguments[0].allocation_offset == 0
    assert nodes[1].arguments[1].allocation_idx == 0
    assert nodes[1].arguments[1].allocation_offset == 0
    assert list(nodes[1].dependencies) == [0]


def test_independent_unnamed_buffers_do_not_reuse_storage() -> None:
    """Independent nodes may execute concurrently and therefore cannot alias."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    first = _execution(builder).buffer((4,), F16)
    second = _execution(builder).buffer((4,), F16)
    builder.call(kernel_handle, first)
    builder.call(kernel_handle, second)

    executable = builder.build()
    nodes = executable.programs[0].nodes

    assert not executable.buffers
    assert executable.allocations[0].size_bytes == TWO_TEMPORARY_BUFFER_SIZE_BYTES
    assert nodes[0].arguments[0].allocation_offset == 0
    assert nodes[1].arguments[0].allocation_offset == TEMPORARY_BUFFER_SIZE_BYTES
    assert not nodes[1].dependencies


def test_unnamed_buffers_used_by_one_node_do_not_reuse_storage() -> None:
    """Buffers accessed by the same invocation have overlapping lifetimes."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    first = _execution(builder).buffer((4,), F16)
    second = _execution(builder).buffer((4,), F16)
    builder.call(kernel_handle, first, second)

    executable = builder.build()
    arguments = executable.programs[0].nodes[0].arguments

    assert executable.allocations[0].size_bytes == TWO_TEMPORARY_BUFFER_SIZE_BYTES
    assert arguments[0].allocation_offset != arguments[1].allocation_offset


def test_execution_allocation_mixes_named_and_unnamed_buffers() -> None:
    """Internal reusable ranges are packed after named execution ranges."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    execution = _execution(builder)
    output = execution.buffer((1,), F16, name="output", writable=True)
    scratch = execution.buffer((4,), F16)
    builder.call(kernel_handle, output, scratch)

    executable = builder.build()
    arguments = executable.programs[0].nodes[0].arguments

    assert executable.allocations[0].lifetime == lc0ex_pb2.Allocation.LIFETIME_EXECUTION
    assert [buffer.name for buffer in executable.buffers] == ["output"]
    assert arguments[0].allocation_offset == 0
    assert arguments[1].allocation_offset == NAMED_OUTPUT_SIZE_BYTES


def test_unused_registered_kernel_is_serialized() -> None:
    """Adding a kernel registers it even when no graph node uses it."""
    builder = _builder()
    builder.add_kernel(_artifact())

    executable = builder.build()

    assert len(executable.binaries) == 1
    assert len(executable.kernels) == 1
    assert not executable.programs


def test_equal_kernel_can_be_registered_twice() -> None:
    """Repeated registration of the same artifact returns the same handle."""
    builder = _builder()
    artifact = _artifact()

    assert builder.add_kernel(artifact) is builder.add_kernel(artifact)
    assert len(builder.build().kernels) == 1


def test_different_kernel_for_same_binary_function_is_rejected() -> None:
    """One binary cannot register conflicting metadata for one export."""
    builder = _builder()
    builder.add_kernel(_artifact())

    with pytest.raises(ValueError, match="already registered differently"):
        builder.add_kernel(_artifact(parameters=(POINTER,)))


def test_same_function_from_different_binaries_uses_distinct_handles() -> None:
    """Function names are only unique within their binary."""
    builder = _builder()
    first = builder.add_kernel(_artifact(binary_data=b"first cubin"))
    second = builder.add_kernel(_artifact(binary_data=b"second cubin"))

    executable = builder.build()

    assert first is not second
    assert len(executable.binaries) == EXPECTED_BINARY_COUNT
    assert len(executable.kernels) == EXPECTED_BINARY_COUNT
    assert [kernel.binary_idx for kernel in executable.kernels] == [0, 1]


def test_call_requires_owned_kernel_handle() -> None:
    """Graph calls must use a handle registered by this builder."""
    with pytest.raises(ValueError, match="does not belong"):
        _builder().call(KernelHandle())


def test_call_rejects_foreign_buffer_handles() -> None:
    """Equal-looking handles from another builder do not satisfy ownership."""
    builder = _builder()
    foreign = _builder()
    kernel_handle = builder.add_kernel(_artifact())

    with pytest.raises(ValueError, match="belong to this executable builder"):
        builder.call(kernel_handle, *_buffers(foreign))


def test_call_rejects_foreign_readonly_buffer() -> None:
    """Read-only buffers must be owned by the executable builder."""
    builder = _builder()
    foreign = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer")

    with pytest.raises(ValueError, match="read-only buffers must belong"):
        builder.call(
            kernel_handle,
            buffer,
            readonly=(_persistent(foreign).buffer((1,), F16, name="foreign"),),
        )


def test_call_rejects_readonly_buffer_not_passed_to_kernel() -> None:
    """Read-only buffers must be among the invocation arguments."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(POINTER,)))
    argument = _persistent(builder).buffer((1,), F16, name="argument")
    readonly = _persistent(builder).buffer((1,), F16, name="readonly")

    with pytest.raises(ValueError, match="read-only buffers must be kernel arguments"):
        builder.call(kernel_handle, argument, readonly=(readonly,))


def test_call_rejects_abi_argument_count_mismatch() -> None:
    """A call must provide every argument declared by the kernel ABI."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact())
    a, b, _ = _buffers(builder)

    with pytest.raises(ValueError, match="argument count"):
        builder.call(kernel_handle, a, b)


def test_call_rejects_non_pointer_parameter() -> None:
    """The builder does not expose runtime parameter arguments yet."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact(parameters=(U32,)))
    buffer = _persistent(builder).buffer((1,), F16, name="buffer")

    with pytest.raises(ValueError, match="only support pointer"):
        builder.call(kernel_handle, buffer)


def test_repeated_builds_have_independent_kernel_messages() -> None:
    """Every build serializes registered kernels into fresh protobuf messages."""
    builder = _builder()
    kernel_handle = builder.add_kernel(_artifact())
    builder.call(kernel_handle, *_buffers(builder))

    first = builder.build()
    second = builder.build()

    assert first == second
    assert second.binaries[0] is not first.binaries[0]
    assert second.kernels[0] is not first.kernels[0]
    assert second.programs[0].nodes[0] is not first.programs[0].nodes[0]
