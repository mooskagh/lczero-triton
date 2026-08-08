"""Builder for serialized Lc0 neural executables."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Self

from lc0ex.buffer_builder import Buffer, PersistentBufferBuilder
from lc0ex.proto import lc0ex_pb2

_MAGIC = 0x1C0E
_FORMAT = 1


class ExecutableBuilder:
    """Build an Lc0 neural executable."""

    def __init__(self) -> None:
        """Initialize an empty executable builder."""
        self._target: tuple[lc0ex_pb2.Target.Vendor, str] | None = None
        self._buffers = PersistentBufferBuilder()

    def buffer(
        self,
        name: str,
        shape: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
    ) -> Buffer:
        """Create or retrieve a persistent logical buffer."""
        return self._buffers.buffer(name, shape, dtype)

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable's compilation target and return this builder."""
        self._target = (vendor, architecture)
        return self

    def build(self) -> lc0ex_pb2.NeuralExecutable:
        """Create and return a new neural executable."""
        executable = lc0ex_pb2.NeuralExecutable(magic=_MAGIC, format=_FORMAT)
        if self._target is not None:
            vendor, architecture = self._target
            executable.target.vendor = vendor
            executable.target.architecture = architecture
        self._buffers.build(executable)
        return executable

    def build_and_write(
        self,
        path: str | PathLike[str],
    ) -> lc0ex_pb2.NeuralExecutable:
        """Build the executable, serialize it to *path*, and return it.

        Raises:
            google.protobuf.message.EncodeError: If required fields are unset.

        """
        executable = self.build()
        Path(path).write_bytes(executable.SerializeToString())
        return executable
