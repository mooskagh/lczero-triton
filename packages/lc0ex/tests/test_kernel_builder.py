"""Tests for generic kernel registration and opaque buffer dependencies."""

# ruff: noqa: PLR2004

import pytest
from lc0ex import (
    Buffer,
    ExecutableBuilder,
    KernelArtifact,
    KernelHandle,
    SymbolArtifact,
    SymbolHandle,
)
from lc0ex.proto import lc0ex_pb2

F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
PERSISTENT = lc0ex_pb2.Allocation.LIFETIME_PERSISTENT
EXECUTION = lc0ex_pb2.Allocation.LIFETIME_EXECUTION
POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
U32 = lc0ex_pb2.PARAMETER_TYPE_U32
TARGET_ARCHITECTURE = "sm_80"


def _builder() -> ExecutableBuilder:
    """Create a target-configured executable builder."""
    return ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )


def _artifact(
    *,
    function: str = "kernel",
    parameters: tuple[lc0ex_pb2.ParameterType, ...] = (POINTER,),
) -> KernelArtifact:
    """Create a small generic pointer-ABI kernel artifact."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=b"fake cubin",
        function=function,
        parameters=parameters,
        grid=(2, 3, 1),
        block=(128, 1, 1),
        dynamic_shared_memory_bytes=256,
    )


def _external(
    builder: ExecutableBuilder,
    name: str,
    *,
    writable: bool = False,
) -> Buffer:
    """Create a small named external buffer for one graph test."""
    return builder.allocation(PERSISTENT).external_buffer(
        name=name,
        shape=(1,),
        dtype=F16,
        writable=writable,
    )


def test_add_kernel_and_call_serializes_generic_metadata() -> None:
    """The generic builder serializes registered pointer kernels and calls."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER, POINTER)))
    first = _external(builder, "first")
    second = _external(builder, "second")
    third = _external(builder, "third")
    builder.call(kernel, first, second, third)

    executable = builder.build()

    assert executable.kernels[0].function == "kernel"
    assert list(executable.programs[0].nodes[0].grid) == [2, 3, 1]
    assert [
        argument.allocation.offset
        for argument in executable.programs[0].nodes[0].arguments
    ] == [0, 0, 0]


def test_readonly_external_buffers_have_no_dependency() -> None:
    """Named persistent ranges default to readonly accesses."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact())
    buffer = _external(builder, "weights")
    builder.call(kernel, buffer)
    builder.call(kernel, buffer)

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_writer_and_reader_dependencies_follow_opaque_identity() -> None:
    """Writers order readers and later writers of the same storage handle."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact())
    buffer = _external(builder, "output", writable=True)
    builder.call(kernel, buffer)
    builder.call(kernel, buffer, readonly=(buffer,))
    builder.call(kernel, buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == [0]
    assert list(nodes[2].dependencies) == [1]


def test_ordered_raw_temporaries_reuse_one_range() -> None:
    """Temporaries may alias only after dependencies order their accesses."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    execution = builder.allocation(EXECUTION)
    first = execution.temporary_buffer(size_bytes=64, alignment_bytes=32)
    second = execution.temporary_buffer(size_bytes=64, alignment_bytes=32)
    bridge = _external(builder, "bridge", writable=True)
    builder.call(kernel, first, bridge)
    builder.call(kernel, bridge, second, readonly=(bridge,))

    executable = builder.build()
    nodes = executable.programs[0].nodes

    assert executable.allocations[0].size_bytes == 64
    assert (
        nodes[0].arguments[0].allocation.offset
        == nodes[1].arguments[1].allocation.offset
    )
    assert list(nodes[1].dependencies) == [0]


def test_independent_raw_temporaries_do_not_reuse_storage() -> None:
    """Potentially concurrent temporary accesses require distinct ranges."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact())
    execution = builder.allocation(EXECUTION)
    first = execution.temporary_buffer(size_bytes=64, alignment_bytes=32)
    second = execution.temporary_buffer(size_bytes=64, alignment_bytes=32)
    builder.call(kernel, first)
    builder.call(kernel, second)

    executable = builder.build()

    assert executable.allocations[0].size_bytes == 128


def test_call_rejects_foreign_and_non_pointer_handles() -> None:
    """Kernel calls retain ownership and pointer-ABI validation."""
    builder = _builder()
    foreign = _builder()
    foreign_buffer = _external(foreign, "foreign")
    pointer_kernel = builder.add_kernel(_artifact())
    u32_kernel = builder.add_kernel(_artifact(function="u32", parameters=(U32,)))
    local_buffer = _external(builder, "local")

    with pytest.raises(ValueError, match="belong"):
        builder.call(pointer_kernel, foreign_buffer)
    with pytest.raises(ValueError, match="pointer"):
        builder.call(u32_kernel, local_buffer)


def test_add_kernel_deduplicates_identical_artifacts() -> None:
    """An artifact identity maps to one opaque kernel handle."""
    builder = _builder()
    first = builder.add_kernel(_artifact())
    second = builder.add_kernel(_artifact())

    assert isinstance(first, KernelHandle)
    assert second is first


def test_symbol_argument_serializes_as_immutable_module_pointer() -> None:
    """A module symbol is a pointer argument, not a callable graph node."""
    builder = _builder()
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    output = _external(builder, "output", writable=True)
    symbol = builder.add_symbol(
        SymbolArtifact(
            binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
            binary_data=b"fake cubin",
            symbol_name="mapping_table",
        )
    )
    builder.call(kernel, output, symbol)

    executable = builder.build()
    argument = executable.programs[0].nodes[0].arguments[1]

    assert isinstance(symbol, SymbolHandle)
    assert len(executable.binaries) == 1
    assert argument.symbol.binary_idx == 0
    assert argument.symbol.symbol_name == "mapping_table"
    assert not argument.HasField("allocation")
