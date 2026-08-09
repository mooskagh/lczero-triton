"""Tests for generic kernel registration and graph construction."""

import pytest
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2

TARGET_ARCHITECTURE = "sm_80"
DYNAMIC_SHARED_MEMORY_BYTES = 256
EXPECTED_CALL_COUNT = 2
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
