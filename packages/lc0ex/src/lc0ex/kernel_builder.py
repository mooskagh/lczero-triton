"""Linker-side handles for kernels exported by compiled modules."""

from dataclasses import dataclass

from lc0ex.proto import lc0ex_pb2


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    """One exported module function and its launch metadata."""

    binary_format: lc0ex_pb2.Binary.Format
    binary_data: bytes
    function: str
    parameters: tuple[lc0ex_pb2.ParameterType, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
    runtime_ns: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class KernelHandle:
    """An opaque linker-side reference to one registered kernel."""


@dataclass(frozen=True, slots=True)
class SymbolArtifact:
    """One immutable pointer symbol exported by a compiled module."""

    binary_format: lc0ex_pb2.Binary.Format
    binary_data: bytes
    symbol_name: str


@dataclass(frozen=True, slots=True, eq=False)
class SymbolHandle:
    """An opaque linker-side reference to one registered module symbol."""
