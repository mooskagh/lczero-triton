"""Linker-side handles for kernels exported by compiled modules."""

from dataclasses import dataclass

from lc0ex.proto import lc0ex_pb2


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    """One exported module function and its launch metadata."""

    binary_name: str
    binary_format: lc0ex_pb2.Binary.Format
    binary_data: bytes
    function: str
    parameters: tuple[lc0ex_pb2.ParameterType, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_memory_bytes: int
