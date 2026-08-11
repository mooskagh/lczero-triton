"""Tests for real BT4 Triton compilation and graph registration."""

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.copy_type_converted import (
    CopyTypeConvertedSpecialization,
    compile_copy_type_converted,
    copy_type_converted,
)


def test_real_compilation_produces_pointer_abi_and_static_launch() -> None:
    """A real Triton result becomes an in-memory linker artifact."""
    artifact = compile_copy_type_converted(CopyTypeConvertedSpecialization(257, 80))

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
    )
    assert artifact.grid == (2, 1, 1)
    assert artifact.block == (256, 1, 1)


def test_copy_graph_call_preserves_output_input_argument_order() -> None:
    """The family API serializes destination before its readonly source."""
    builder = ExecutableBuilder()
    execution = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    output = execution.external_buffer(
        name="output",
        shape=(257,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
        writable=True,
    )
    input_ = execution.external_buffer(
        name="input",
        shape=(257,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    copy_type_converted(
        builder,
        KernelCache(builder),
        output,
        input_,
        CopyTypeConvertedSpecialization(257, 80),
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.allocation_idx, buffer.allocation_offset)
        for buffer in executable.buffers
    }

    assert executable.target.architecture == "sm_80"
    assert (
        node.arguments[0].allocation.index,
        node.arguments[0].allocation.offset,
    ) == locations["output"]
    assert (
        node.arguments[1].allocation.index,
        node.arguments[1].allocation.offset,
    ) == locations["input"]
