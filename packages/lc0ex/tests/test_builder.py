"""Tests for opaque executable allocation and buffer construction."""

# ruff: noqa: PLR2004

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError
from lc0ex import Buffer, ExecutableBuilder
from lc0ex.buffer_builder import data_type_size_bytes
from lc0ex.proto import lc0ex_pb2

F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
F32 = lc0ex_pb2.Buffer.DATA_TYPE_F32
U8 = lc0ex_pb2.Buffer.DATA_TYPE_U8
U64 = lc0ex_pb2.Buffer.DATA_TYPE_U64
PERSISTENT = lc0ex_pb2.Allocation.LIFETIME_PERSISTENT
EXECUTION = lc0ex_pb2.Allocation.LIFETIME_EXECUTION
TARGET_ARCHITECTURE = "sm_80"


def _builder() -> ExecutableBuilder:
    """Create a fully target-configured executable builder."""
    return ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )


def test_buffer_is_an_opaque_storage_identity() -> None:
    """Buffers intentionally expose no tensor metadata or view operations."""
    buffer = Buffer()

    assert not hasattr(buffer, "shape")
    assert not hasattr(buffer, "dtype")
    assert not hasattr(buffer, "reshape")


def test_external_buffer_serializes_its_private_runtime_contract() -> None:
    """A named external range retains shape and dtype only in the executable."""
    builder = _builder()
    persistent = builder.allocation(PERSISTENT)
    result = persistent.external_buffer(
        name="weights",
        shape=(2, 3),
        dtype=F16,
        alignment_bytes=16,
    )

    executable = builder.build()

    assert isinstance(result, Buffer)
    assert len(executable.allocations) == 1
    assert executable.allocations[0].size_bytes == 12
    assert executable.allocations[0].alignment_bytes == 16
    assert executable.buffers[0].name == "weights"
    assert tuple(executable.buffers[0].shape) == (2, 3)
    assert executable.buffers[0].data_type == F16


def test_external_buffer_reuses_an_identical_declaration() -> None:
    """A name identifies one canonical external buffer contract."""
    builder = _builder()
    persistent = builder.allocation(PERSISTENT)
    first = persistent.external_buffer(name="weights", shape=(2, 3), dtype=F16)

    second = persistent.external_buffer(name="weights", shape=(2, 3), dtype=F16)

    assert second is first


def test_external_buffer_rejects_conflicting_redeclarations() -> None:
    """A repeated external name must retain all physical metadata."""
    builder = _builder()
    persistent = builder.allocation(PERSISTENT)
    persistent.external_buffer(name="weights", shape=(2, 3), dtype=F16)

    with pytest.raises(ValueError, match="shape"):
        persistent.external_buffer(name="weights", shape=(3, 2), dtype=F16)
    with pytest.raises(ValueError, match="data type"):
        persistent.external_buffer(name="weights", shape=(2, 3), dtype=F32)
    with pytest.raises(ValueError, match="writability"):
        persistent.external_buffer(
            name="weights",
            shape=(2, 3),
            dtype=F16,
            writable=True,
        )
    with pytest.raises(ValueError, match="alignment"):
        persistent.external_buffer(
            name="weights",
            shape=(2, 3),
            dtype=F16,
            alignment_bytes=16,
        )


def test_external_buffer_rejects_a_cross_lifetime_redeclaration() -> None:
    """One named external range cannot move between allocation lifetimes."""
    builder = _builder()
    persistent = builder.allocation(PERSISTENT)
    execution = builder.allocation(EXECUTION)
    persistent.external_buffer(name="weights", shape=(1,), dtype=F16)

    with pytest.raises(ValueError, match="different allocation"):
        execution.external_buffer(name="weights", shape=(1,), dtype=F16)


@pytest.mark.parametrize(
    ("name", "shape"),
    [("", (1,)), ("zero", ()), ("zero", (0,)), ("negative", (-1,))],
)
def test_external_buffer_rejects_invalid_runtime_contract(
    name: str,
    shape: tuple[int, ...],
) -> None:
    """Named runtime buffers require a nonempty name and positive dimensions."""
    persistent = _builder().allocation(PERSISTENT)

    with pytest.raises(ValueError, match=r"cannot be empty|positive dimensions"):
        persistent.external_buffer(name=name, shape=shape, dtype=F16)


@pytest.mark.parametrize("alignment_bytes", [0, 3])
def test_external_buffer_rejects_invalid_alignment(alignment_bytes: int) -> None:
    """Physical range alignment is a positive power of two."""
    persistent = _builder().allocation(PERSISTENT)

    with pytest.raises(ValueError, match="power of two"):
        persistent.external_buffer(
            name="weights",
            shape=(1,),
            dtype=F16,
            alignment_bytes=alignment_bytes,
        )


def test_temporary_buffer_is_raw_execution_storage() -> None:
    """Anonymous internal ranges are byte-sized rather than tensor-shaped."""
    builder = _builder()
    execution = builder.allocation(EXECUTION)
    result = execution.temporary_buffer(size_bytes=128, alignment_bytes=64)

    assert isinstance(result, Buffer)
    assert not builder.build().allocations


def test_temporary_buffer_rejects_persistent_storage() -> None:
    """Only execution allocations may contain reusable anonymous storage."""
    persistent = _builder().allocation(PERSISTENT)

    with pytest.raises(ValueError, match="EXECUTION"):
        persistent.temporary_buffer(size_bytes=2, alignment_bytes=2)


@pytest.mark.parametrize("size_bytes", [0, -1])
def test_temporary_buffer_rejects_invalid_size(size_bytes: int) -> None:
    """Raw internal storage must have a positive uint64 byte count."""
    execution = _builder().allocation(EXECUTION)

    with pytest.raises(ValueError, match="positive uint64"):
        execution.temporary_buffer(size_bytes=size_bytes, alignment_bytes=1)


def test_named_execution_buffers_serialize_as_fixed_ranges() -> None:
    """External inputs and outputs remain distinct execution-lifetime ranges."""
    builder = _builder()
    execution = builder.allocation(EXECUTION)
    execution.external_buffer(name="input", shape=(2,), dtype=F16, alignment_bytes=16)
    execution.external_buffer(
        name="output",
        shape=(2,),
        dtype=F16,
        writable=True,
        alignment_bytes=16,
    )

    executable = builder.build()

    assert executable.allocations[0].lifetime == EXECUTION
    assert [buffer.name for buffer in executable.buffers] == ["input", "output"]
    assert [buffer.allocation_offset for buffer in executable.buffers] == [0, 16]


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
    """External contracts cannot defer unsupported dtype validation to build time."""
    persistent = _builder().allocation(PERSISTENT)

    with pytest.raises(KeyError):
        persistent.external_buffer(
            name="unknown",
            shape=(1,),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN,
        )


def test_build_and_write_serializes_a_complete_external_contract(
    tmp_path: Path,
) -> None:
    """A configured target and external range produce a parseable executable."""
    builder = _builder()
    builder.allocation(PERSISTENT).external_buffer(
        name="weights",
        shape=(1,),
        dtype=F16,
    )
    output_path = tmp_path / "network.lc0ex"

    result = builder.build_and_write(output_path)
    restored = lc0ex_pb2.NeuralExecutable()
    restored.ParseFromString(output_path.read_bytes())

    assert restored == result


def test_build_and_write_requires_a_target() -> None:
    """The executable protobuf still enforces its required target field."""
    builder = ExecutableBuilder()

    with pytest.raises(EncodeError):
        builder.build_and_write(Path("incomplete.lc0ex"))
