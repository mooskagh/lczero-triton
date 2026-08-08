"""Tests for the executable builder."""

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2

EXPECTED_MAGIC = 0x1C0E
EXPECTED_FORMAT = 1
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
