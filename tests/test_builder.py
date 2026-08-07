"""Tests for the executable builder."""

from pathlib import Path

import pytest
from google.protobuf.message import EncodeError

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2


def test_builder_starts_with_empty_executable() -> None:
    """A new builder owns an empty neural executable message."""
    builder = ExecutableBuilder()

    assert isinstance(builder.executable, lc0ex_pb2.NeuralExecutable)
    assert not builder.executable.IsInitialized()


def test_write_to_file_rejects_missing_required_fields(tmp_path: Path) -> None:
    """Serialization rejects an executable with unset required fields."""
    builder = ExecutableBuilder()

    with pytest.raises(EncodeError):
        builder.write_to_file(tmp_path / "incomplete.lc0ex")


def test_write_to_file_serializes_executable(tmp_path: Path) -> None:
    """A fully initialized executable is serialized without modification."""
    builder = ExecutableBuilder()
    executable = builder.executable
    executable.magic = 0x1C0E
    executable.format = executable.FORMAT_TRITON_CUBIN_V1
    executable.min_lc0_version.major = 0
    executable.min_lc0_version.minor = 32
    executable.min_lc0_version.patch = 0
    executable.network_fingerprint = b"network-fingerprint"
    output_path = tmp_path / "network.lc0ex"

    builder.write_to_file(output_path)

    restored = lc0ex_pb2.NeuralExecutable()
    restored.ParseFromString(output_path.read_bytes())
    assert restored == executable
