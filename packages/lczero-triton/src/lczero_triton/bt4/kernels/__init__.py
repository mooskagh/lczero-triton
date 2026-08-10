"""BT4-specific Triton kernel compilation entry points."""

from os import PathLike
from pathlib import Path


def compile_kernels(output: str | PathLike[str]) -> Path:
    """Compile the fixed BT4 kernel registry."""
    message = f"BT4 kernel compilation is not implemented yet: {Path(output)}"
    raise NotImplementedError(message)
