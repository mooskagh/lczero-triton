"""Tests for the executable builder."""

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError
from lc0ex import Buffer, ExecutableBuilder
from lc0ex.buffer_builder import data_type_size_bytes
from lc0ex.proto import lc0ex_pb2

EXPECTED_MAGIC = 0x1C0E
EXPECTED_FORMAT = 1
EXPECTED_PERSISTENT_SIZE = 36
EXPECTED_PERSISTENT_ALIGNMENT = 8
EXPECTED_EXECUTION_SIZE = 4
EXPECTED_EXECUTION_ALIGNMENT = 2
TARGET_ARCHITECTURE = "sm_80"


def test_build_sets_fixed_header_fields() -> None:
    """A new builder sets the executable's fixed header fields."""
    executable = ExecutableBuilder().build()

    assert isinstance(executable, lc0ex_pb2.NeuralExecutable)
    assert executable.magic == EXPECTED_MAGIC
    assert executable.format == EXPECTED_FORMAT
    assert not executable.IsInitialized()


def test_build_creates_independent_executables_with_target() -> None:
    """Each build creates a new executable containing the configured target."""
    builder = ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )

    first = builder.build()
    second = builder.build()

    assert first is not second
    assert first.target.vendor == lc0ex_pb2.Target.VENDOR_NVIDIA
    assert first.target.architecture == TARGET_ARCHITECTURE
    assert second == first
    assert second.IsInitialized()


def test_build_and_write_rejects_missing_required_fields(tmp_path: Path) -> None:
    """Serialization rejects an executable with unset required fields."""
    builder = ExecutableBuilder()

    with pytest.raises(EncodeError):
        builder.build_and_write(tmp_path / "incomplete.lc0ex")


def test_build_and_write_serializes_and_returns_executable(tmp_path: Path) -> None:
    """A fully initialized executable is serialized and returned."""
    builder = ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )
    output_path = tmp_path / "network.lc0ex"

    result = builder.build_and_write(output_path)

    restored = lc0ex_pb2.NeuralExecutable()
    restored.ParseFromString(output_path.read_bytes())
    assert restored == result


@pytest.mark.parametrize(
    ("dtype", "expected_size"),
    [
        (lc0ex_pb2.Buffer.DATA_TYPE_F32, 4),
        (lc0ex_pb2.Buffer.DATA_TYPE_U8, 1),
        (lc0ex_pb2.Buffer.DATA_TYPE_F16, 2),
        (lc0ex_pb2.Buffer.DATA_TYPE_U64, 8),
        (lc0ex_pb2.Buffer.DATA_TYPE_BF16, 2),
    ],
)
def test_data_type_size_bytes(
    dtype: lc0ex_pb2.Buffer.DataType,
    expected_size: int,
) -> None:
    """Every supported data type has the expected element size."""
    assert data_type_size_bytes(dtype) == expected_size


def test_data_type_size_bytes_rejects_unknown_type() -> None:
    """The unknown data type has no element size."""
    with pytest.raises(KeyError):
        data_type_size_bytes(lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN)


def test_buffer_returns_typed_handle() -> None:
    """Creating a buffer returns its immutable logical handle."""
    result = ExecutableBuilder().buffer(
        "weights",
        (2, 3),
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    assert isinstance(result, Buffer)
    assert result.name == "weights"
    assert result.shape == (2, 3)
    assert result.dtype == lc0ex_pb2.Buffer.DATA_TYPE_F16


def test_buffer_reuses_existing_handle() -> None:
    """Looking up an existing buffer returns the identical handle."""
    builder = ExecutableBuilder()
    original = builder.buffer(
        "weights",
        [2, 3],
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    assert builder.buffer("weights") is original
    assert builder.buffer("weights", (2, 3)) is original
    assert builder.buffer("weights", dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16) is original
    assert builder.buffer("weights", (2, 3), lc0ex_pb2.Buffer.DATA_TYPE_F16) is original


def test_tmp_buffer_generates_unique_names() -> None:
    """Temporary buffers receive generated names from the shared namespace."""
    builder = ExecutableBuilder()
    builder.buffer("tmp_0", (1,), lc0ex_pb2.Buffer.DATA_TYPE_U8)

    first = builder.tmp_buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F16)
    second = builder.tmp_buffer((3,), lc0ex_pb2.Buffer.DATA_TYPE_F32)

    assert first.name == "tmp_1"
    assert second.name == "tmp_2"
    assert first.shape == (2,)
    assert second.dtype == lc0ex_pb2.Buffer.DATA_TYPE_F32


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        (None, None, "shape and data type are required"),
        ((2, 3), None, "shape and data type are required"),
        (None, lc0ex_pb2.Buffer.DATA_TYPE_F16, "shape and data type are required"),
    ],
)
def test_new_buffer_requires_complete_definition(
    shape: tuple[int, ...] | None,
    dtype: lc0ex_pb2.Buffer.DataType | None,
    message: str,
) -> None:
    """A new buffer requires a complete definition."""
    with pytest.raises(ValueError, match=message):
        ExecutableBuilder().buffer("weights", shape, dtype)


def test_unknown_buffer_type_fails_when_building() -> None:
    """An unknown data type fails when its allocation size is needed."""
    builder = ExecutableBuilder()
    builder.buffer("weights", (2, 3), lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN)

    with pytest.raises(KeyError):
        builder.build()


def test_existing_buffer_rejects_mismatched_definition() -> None:
    """A repeated buffer definition must match the original definition."""
    builder = ExecutableBuilder()
    builder.buffer("weights", (2, 3), lc0ex_pb2.Buffer.DATA_TYPE_F16)

    with pytest.raises(ValueError, match="shape does not match"):
        builder.buffer("weights", (3, 2))
    with pytest.raises(ValueError, match="data type does not match"):
        builder.buffer("weights", dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32)


def test_build_combines_buffers_into_persistent_allocation() -> None:
    """Buffers share one naturally aligned persistent allocation."""
    builder = ExecutableBuilder()
    builder.buffer("byte", (3,), lc0ex_pb2.Buffer.DATA_TYPE_U8)
    builder.buffer("wide", (2,), lc0ex_pb2.Buffer.DATA_TYPE_U64)
    builder.buffer("half", (3, 2), lc0ex_pb2.Buffer.DATA_TYPE_F16)

    executable = builder.build()

    assert len(executable.allocations) == 1
    allocation = executable.allocations[0]
    assert allocation.name == "persistent"
    assert allocation.size_bytes == EXPECTED_PERSISTENT_SIZE
    assert allocation.alignment_bytes == EXPECTED_PERSISTENT_ALIGNMENT
    assert allocation.lifetime == lc0ex_pb2.Allocation.LIFETIME_PERSISTENT

    assert [buffer.name for buffer in executable.buffers] == [
        "byte",
        "wide",
        "half",
    ]
    assert [buffer.data_type for buffer in executable.buffers] == [
        lc0ex_pb2.Buffer.DATA_TYPE_U8,
        lc0ex_pb2.Buffer.DATA_TYPE_U64,
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
    ]
    assert [list(buffer.shape) for buffer in executable.buffers] == [
        [3],
        [2],
        [3, 2],
    ]
    assert [buffer.allocation_block.allocation for buffer in executable.buffers] == [
        "persistent",
        "persistent",
        "persistent",
    ]
    assert [buffer.allocation_block.offset_bytes for buffer in executable.buffers] == [
        0,
        8,
        24,
    ]


def test_build_without_buffers_has_no_allocation() -> None:
    """An empty buffer collection does not emit an allocation."""
    executable = ExecutableBuilder().build()

    assert not executable.allocations
    assert not executable.buffers


def test_build_places_temporary_buffers_in_execution_allocation() -> None:
    """Temporary buffers use an execution-lifetime allocation."""
    builder = ExecutableBuilder()
    first = builder.tmp_buffer((3,), lc0ex_pb2.Buffer.DATA_TYPE_U8)
    second = builder.tmp_buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F16)

    executable = builder.build()

    assert len(executable.allocations) == 1
    allocation = executable.allocations[0]
    assert allocation.name == "execution"
    assert allocation.size_bytes == EXPECTED_EXECUTION_SIZE
    assert allocation.alignment_bytes == EXPECTED_EXECUTION_ALIGNMENT
    assert allocation.lifetime == lc0ex_pb2.Allocation.LIFETIME_EXECUTION
    assert [buffer.name for buffer in executable.buffers] == [first.name, second.name]
    assert [buffer.allocation_block.allocation for buffer in executable.buffers] == [
        "execution",
        "execution",
    ]
    assert [buffer.allocation_block.offset_bytes for buffer in executable.buffers] == [
        0,
        0,
    ]


def test_repeated_builds_have_independent_buffer_messages() -> None:
    """Every build serializes the same handles into fresh protobuf messages."""
    builder = ExecutableBuilder()
    builder.buffer("weights", (2,), lc0ex_pb2.Buffer.DATA_TYPE_F32)

    first = builder.build()
    second = builder.build()

    assert first == second
    assert first.buffers[0] is not second.buffers[0]
    assert first.allocations[0] is not second.allocations[0]
