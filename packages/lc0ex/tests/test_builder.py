"""Tests for executable allocation and buffer construction."""

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError
from lc0ex import Allocation, Buffer, ExecutableBuilder
from lc0ex.buffer_builder import data_type_size_bytes
from lc0ex.proto import lc0ex_pb2

EXPECTED_MAGIC = 0x1C0E
EXPECTED_FORMAT = 1
EXPECTED_PERSISTENT_SIZE = 36
EXPECTED_PERSISTENT_ALIGNMENT = 8
TARGET_ARCHITECTURE = "sm_80"


def _allocation(
    builder: ExecutableBuilder,
    lifetime: lc0ex_pb2.Allocation.Lifetime,
) -> Allocation:
    return builder.allocation(lifetime)


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
    with pytest.raises(EncodeError):
        ExecutableBuilder().build_and_write(tmp_path / "incomplete.lc0ex")


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


def test_allocation_rejects_unknown_lifetime() -> None:
    """Every allocation has a concrete runtime lifetime."""
    with pytest.raises(ValueError, match="PERSISTENT or EXECUTION"):
        ExecutableBuilder().allocation(lc0ex_pb2.Allocation.LIFETIME_UNKNOWN)


def test_buffer_returns_typed_handle() -> None:
    """Creating a buffer returns an immutable allocation-owned handle."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    result = persistent.buffer(
        (2, 3),
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
        name="weights",
    )

    assert isinstance(result, Buffer)
    assert result.allocation is persistent
    assert result.name == "weights"
    assert result.shape == (2, 3)
    assert result.dtype == lc0ex_pb2.Buffer.DATA_TYPE_F16


def test_named_buffer_reuses_existing_handle() -> None:
    """Named buffer lookup does not require positional-name overloads."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    original = persistent.buffer(
        [2, 3],
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
        name="weights",
    )

    assert persistent.buffer(name="weights") is original
    assert persistent.buffer((2, 3), name="weights") is original
    assert (
        persistent.buffer(dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16, name="weights")
        is original
    )


def test_unnamed_execution_buffers_are_distinct() -> None:
    """Unnamed buffers are internal and do not consume the name namespace."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    execution = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    persistent.buffer((1,), lc0ex_pb2.Buffer.DATA_TYPE_U8, name="tmp_0")
    first = execution.buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F16)
    second = execution.buffer((3,), lc0ex_pb2.Buffer.DATA_TYPE_F32)

    assert first.name is None
    assert second.name is None
    assert first is not second
    assert first.shape == (2,)
    assert second.dtype == lc0ex_pb2.Buffer.DATA_TYPE_F32


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        (None, None),
        ((2, 3), None),
        (None, lc0ex_pb2.Buffer.DATA_TYPE_F16),
    ],
)
def test_new_buffer_requires_complete_definition(
    shape: tuple[int, ...] | None,
    dtype: lc0ex_pb2.Buffer.DataType | None,
) -> None:
    """A new buffer requires a complete definition."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)

    with pytest.raises(ValueError, match="shape and data type are required"):
        persistent.buffer(shape, dtype, name="weights")


def test_unknown_buffer_type_fails_when_building() -> None:
    """An unknown data type fails when its allocation size is needed."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    persistent.buffer(
        (2, 3),
        lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN,
        name="weights",
    )

    with pytest.raises(KeyError):
        builder.build()


def test_named_buffers_cannot_move_between_allocations() -> None:
    """A name identifies one fixed range in one allocation."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    execution = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    persistent.buffer((1,), lc0ex_pb2.Buffer.DATA_TYPE_F16, name="weights")

    with pytest.raises(ValueError, match="different allocation"):
        execution.buffer((1,), lc0ex_pb2.Buffer.DATA_TYPE_F16, name="weights")


def test_persistent_allocations_reject_unnamed_buffers() -> None:
    """Only execution allocations may contain reusable internal ranges."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)

    with pytest.raises(ValueError, match="require an EXECUTION allocation"):
        persistent.buffer((1,), lc0ex_pb2.Buffer.DATA_TYPE_F16)


def test_build_combines_named_buffers_into_persistent_allocation() -> None:
    """Named persistent buffers share one naturally aligned allocation."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    persistent.buffer((3,), lc0ex_pb2.Buffer.DATA_TYPE_U8, name="byte")
    persistent.buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_U64, name="wide")
    persistent.buffer((3, 2), lc0ex_pb2.Buffer.DATA_TYPE_F16, name="half")

    executable = builder.build()

    assert len(executable.allocations) == 1
    allocation = executable.allocations[0]
    assert allocation.size_bytes == EXPECTED_PERSISTENT_SIZE
    assert allocation.alignment_bytes == EXPECTED_PERSISTENT_ALIGNMENT
    assert allocation.lifetime == lc0ex_pb2.Allocation.LIFETIME_PERSISTENT
    assert [buffer.name for buffer in executable.buffers] == ["byte", "wide", "half"]
    assert [buffer.allocation_idx for buffer in executable.buffers] == [0, 0, 0]
    assert [buffer.allocation_offset for buffer in executable.buffers] == [0, 8, 24]


def test_execution_allocation_serializes_named_buffers() -> None:
    """Named inputs and outputs use execution-lifetime storage."""
    builder = ExecutableBuilder()
    execution = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    execution.buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F16, name="input")
    execution.buffer(
        (2,),
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
        name="output",
        writable=True,
    )

    executable = builder.build()

    assert len(executable.allocations) == 1
    assert executable.allocations[0].lifetime == lc0ex_pb2.Allocation.LIFETIME_EXECUTION
    assert [buffer.name for buffer in executable.buffers] == ["input", "output"]
    assert [buffer.allocation_idx for buffer in executable.buffers] == [0, 0]


def test_unused_unnamed_buffers_do_not_emit_an_allocation() -> None:
    """Unused internal buffers do not reserve runtime storage."""
    builder = ExecutableBuilder()
    execution = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    first = execution.buffer((3,), lc0ex_pb2.Buffer.DATA_TYPE_U8)
    second = execution.buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F16)

    executable = builder.build()

    assert first.name is None
    assert second.name is None
    assert not executable.allocations
    assert not executable.buffers


def test_repeated_builds_have_independent_buffer_messages() -> None:
    """Every build serializes the same handles into fresh protobuf messages."""
    builder = ExecutableBuilder()
    persistent = _allocation(builder, lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    persistent.buffer((2,), lc0ex_pb2.Buffer.DATA_TYPE_F32, name="weights")

    first = builder.build()
    second = builder.build()

    assert first == second
    assert first.buffers[0] is not second.buffers[0]
    assert first.allocations[0] is not second.allocations[0]
