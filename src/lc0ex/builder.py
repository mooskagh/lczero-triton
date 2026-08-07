"""Builder for serialized Lc0 neural executables."""

from os import PathLike
from pathlib import Path

from lc0ex.proto import lc0ex_pb2


class ExecutableBuilder:
    """Build an Lc0 neural executable."""

    def __init__(self) -> None:
        """Initialize a builder containing an empty executable."""
        self._executable = lc0ex_pb2.NeuralExecutable()

    def build(self) -> lc0ex_pb2.NeuralExecutable:
        """Return the executable being built."""
        return self._executable

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
