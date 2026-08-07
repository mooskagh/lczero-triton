"""Tests for the executable builder."""

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2


def test_build_returns_empty_executable() -> None:
    """A new builder builds an empty neural executable message."""
    executable = ExecutableBuilder().build()

    assert isinstance(executable, lc0ex_pb2.NeuralExecutable)
    assert not executable.IsInitialized()


def test_build_and_write_rejects_missing_required_fields(tmp_path: Path) -> None:
    """Serialization rejects an executable with unset required fields."""
    builder = ExecutableBuilder()

    with pytest.raises(EncodeError):
        builder.build_and_write(tmp_path / "incomplete.lc0ex")


def test_build_and_write_serializes_and_returns_executable(tmp_path: Path) -> None:
    """A fully initialized executable is serialized and returned."""
    builder = ExecutableBuilder()
    executable = builder.build()
    executable.magic = 0x1C0E
    executable.format = executable.FORMAT_TRITON_CUBIN_V1
    executable.min_lc0_version.major = 0
    executable.min_lc0_version.minor = 32
    executable.min_lc0_version.patch = 0
    executable.network_fingerprint = b"network-fingerprint"
    output_path = tmp_path / "network.lc0ex"

    result = builder.build_and_write(output_path)

    restored = lc0ex_pb2.NeuralExecutable()
    restored.ParseFromString(output_path.read_bytes())
    assert result is executable
    assert restored == executable
