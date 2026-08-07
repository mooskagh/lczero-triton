"""Builder for serialized Lc0 neural executables."""

from os import PathLike
from pathlib import Path

from lc0ex.proto import lc0ex_pb2


class ExecutableBuilder:
    """Build an Lc0 neural executable."""

    def __init__(self) -> None:
        """Initialize a builder containing an empty executable."""
        self._executable = lc0ex_pb2.NeuralExecutable()

    @property
    def executable(self) -> lc0ex_pb2.NeuralExecutable:
        """Return the executable being built."""
        return self._executable

    def write_to_file(self, path: str | PathLike[str]) -> None:
        """Serialize the executable to *path*.

        Raises:
            google.protobuf.message.EncodeError: If required fields are unset.

        """
        Path(path).write_bytes(self._executable.SerializeToString())
