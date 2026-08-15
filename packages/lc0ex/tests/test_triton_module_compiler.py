"""Tests for adapting Triton compiler output to linker artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass

import pytest
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

_SHARED_MEMORY_BYTES = 64


@dataclass(slots=True)
class _FakeTarget:
    backend: str
    arch: int | str
    warp_size: int


@dataclass(slots=True)
class _FakeMetadata:
    num_warps: int
    shared: int
    target: _FakeTarget
    global_scratch_size: int = 0
    profile_scratch_size: int = 0


@dataclass(slots=True)
class _FakeCompiledKernel:
    asm: Mapping[str, bytes | str]
    metadata: _FakeMetadata
    name: str


def test_artifact_from_triton_preserves_binary_abi_and_launch() -> None:
    """The adapter retains the complete in-memory linker contract."""
    compiled = _FakeCompiledKernel(
        name="kernel_exported",
        asm={"cubin": b"fake cubin"},
        metadata=_FakeMetadata(
            num_warps=4,
            shared=_SHARED_MEMORY_BYTES,
            target=_FakeTarget(backend="cuda", arch=120, warp_size=32),
        ),
    )

    artifact = artifact_from_triton(
        compiled,
        grid=(2, 3, 1),
        parameters=(
            lc0ex_pb2.PARAMETER_TYPE_POINTER,
            lc0ex_pb2.PARAMETER_TYPE_POINTER,
        ),
    )

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data == b"fake cubin"
    assert artifact.function == "kernel_exported"
    assert artifact.parameters == (
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
    )
    assert artifact.grid == (2, 3, 1)
    assert artifact.block == (128, 1, 1)
    assert artifact.dynamic_shared_memory_bytes == _SHARED_MEMORY_BYTES


def test_artifact_from_triton_rejects_scratch_allocations() -> None:
    """Nonzero Triton scratch storage is outside the initial linker ABI."""
    compiled = _FakeCompiledKernel(
        name="kernel_exported",
        asm={"cubin": b"fake cubin"},
        metadata=_FakeMetadata(
            num_warps=4,
            shared=_SHARED_MEMORY_BYTES,
            target=_FakeTarget(backend="cuda", arch=120, warp_size=32),
            global_scratch_size=1,
        ),
    )

    with pytest.raises(ValueError, match="scratch allocations"):
        artifact_from_triton(
            compiled,
            grid=(2, 3, 1),
            parameters=(lc0ex_pb2.PARAMETER_TYPE_POINTER,),
        )
