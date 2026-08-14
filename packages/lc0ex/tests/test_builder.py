"""Tests for opaque executable allocations and buffer construction."""

# ruff: noqa: PLR2004

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.buffer_builder import data_type_size_bytes
from lc0ex.proto import lc0ex_pb2

F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
F32 = lc0ex_pb2.Buffer.DATA_TYPE_F32
U8 = lc0ex_pb2.Buffer.DATA_TYPE_U8
U64 = lc0ex_pb2.Buffer.DATA_TYPE_U64
POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
TARGET_ARCHITECTURE = "sm_80"


def _builder() -> ExecutableBuilder:
    """Create a fully target-configured executable builder."""
    return ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )


def _kernel() -> KernelArtifact:
    """Create a small fake pointer kernel."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=b"fake cubin",
        function="kernel",
        parameters=(POINTER,),
        grid=(1, 1, 1),
        block=(1, 1, 1),
        dynamic_shared_memory_bytes=0,
    )


def test_buffer_is_an_opaque_storage_identity() -> None:
    """Buffers intentionally expose no tensor metadata or view operations."""
    buffer = Buffer()

    assert not hasattr(buffer, "shape")
    assert not hasattr(buffer, "dtype")
    assert not hasattr(buffer, "reshape")


def test_external_buffer_serializes_in_persistent_allocation() -> None:
    """A persistent range is packed into the executable allocation."""
    builder = _builder()
    result = builder.persistent_buffer(
        name="weights",
        shape=(2, 3),
        dtype=F16,
        alignment_bytes=16,
    )

    executable = builder.build()

    assert isinstance(result, Buffer)
    assert executable.persistent_allocation.size_bytes == 12
    assert executable.persistent_allocation.alignment_bytes == 16
    assert executable.buffers[0].name == "weights"
    assert executable.buffers[0].offset == 0
    assert tuple(executable.buffers[0].shape) == (2, 3)
    assert executable.buffers[0].data_type == F16


def test_external_buffer_reuses_an_identical_declaration() -> None:
    """A name identifies one canonical external buffer within a scope."""
    builder = _builder()
    first = builder.persistent_buffer(name="weights", shape=(2, 3), dtype=F16)

    second = builder.persistent_buffer(name="weights", shape=(2, 3), dtype=F16)

    assert second is first


def test_same_named_buffers_are_private_to_programs() -> None:
    """Programs may use the same logical name with independent contracts."""
    builder = _builder()
    first = builder.program(name="batch-1")
    second = builder.program(name="batch-2")
    first.buffer(name="input", shape=(1,), dtype=F16)
    second.buffer(name="input", shape=(2,), dtype=F16)

    executable = builder.build()

    assert [program.name for program in executable.programs] == ["batch-1", "batch-2"]
    assert tuple(executable.programs[0].buffers[0].shape) == (1,)
    assert tuple(executable.programs[1].buffers[0].shape) == (2,)


def test_temporary_buffer_is_owned_by_its_program() -> None:
    """Anonymous raw ranges are owned by their explicit program."""
    builder = _builder()
    program = builder.program(name="main")

    result = program.temporary_buffer(size_bytes=2, alignment_bytes=2)

    assert isinstance(result, Buffer)


def test_temporary_buffer_is_omitted_when_not_used_by_a_program() -> None:
    """Unused anonymous ranges do not force an execution allocation."""
    builder = _builder()
    program = builder.program(name="main")
    program.temporary_buffer(size_bytes=128, alignment_bytes=64)

    assert not builder.build().programs[0].HasField("execution_allocation")


def test_temporary_buffer_is_packed_in_program_allocation() -> None:
    """Used anonymous ranges are packed into their program's allocation."""
    builder = _builder()
    program = builder.program(name="main")
    first = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    second = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    kernel = builder.add_kernel(_kernel())
    program.call(kernel, first)
    program.call(kernel, second)

    executable = builder.build()

    assert executable.programs[0].execution_allocation.size_bytes == 128


def test_named_execution_buffers_serialize_in_program_allocation() -> None:
    """Named inputs and outputs remain distinct program-local ranges."""
    builder = _builder()
    program = builder.program(name="main")
    program.buffer(name="input", shape=(2,), dtype=F16, alignment_bytes=16)
    program.buffer(
        name="output",
        shape=(2,),
        dtype=F16,
        writable=True,
        alignment_bytes=16,
    )

    executable = builder.build()

    assert executable.programs[0].execution_allocation.size_bytes == 20
    assert [buffer.name for buffer in executable.programs[0].buffers] == [
        "input",
        "output",
    ]
    assert [buffer.offset for buffer in executable.programs[0].buffers] == [0, 16]


@pytest.mark.parametrize(
    ("dtype", "expected_size"),
    [(F32, 4), (U8, 1), (F16, 2), (U64, 8), (lc0ex_pb2.Buffer.DATA_TYPE_BF16, 2)],
)
def test_data_type_size_bytes(
    dtype: lc0ex_pb2.Buffer.DataType,
    expected_size: int,
) -> None:
    """Every concrete linker dtype has its expected element size."""
    assert data_type_size_bytes(dtype) == expected_size


def test_unknown_dtype_is_rejected_when_declaring_external_storage() -> None:
    """External contracts cannot defer dtype validation to build time."""
    with pytest.raises(KeyError):
        _builder().persistent_buffer(
            name="unknown",
            shape=(1,),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN,
        )


def test_metadata_serializes_as_opaque_executable_metadata() -> None:
    """Configured metadata survives executable construction and parsing."""
    builder = _builder().set_metadata(b"fingerprint")

    executable = builder.build()
    restored = lc0ex_pb2.NeuralExecutable()
    restored.ParseFromString(executable.SerializeToString())

    assert restored.metadata == b"fingerprint"


def test_build_and_write_requires_a_target(tmp_path: Path) -> None:
    """The executable protobuf still enforces its required target field."""
    output_path = tmp_path / "incomplete.lc0ex"

    with pytest.raises(EncodeError):
        ExecutableBuilder().build_and_write(output_path)
