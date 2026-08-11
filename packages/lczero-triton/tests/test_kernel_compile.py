"""Tests for real BT4 Triton compilation and graph registration."""

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.add_bias_batched import (
    AddBiasBatchedSpecialization,
    compile_add_bias_batched,
)
from lczero_triton.bt4.kernels.add_vectors import (
    AddVectorsSpecialization,
    compile_add_vectors,
)
from lczero_triton.bt4.kernels.copy_type_converted import (
    CopyTypeConvertedSpecialization,
    compile_copy_type_converted,
    copy_type_converted,
)
from lczero_triton.bt4.kernels.mapping_table import compile_symbol
from lczero_triton.bt4.kernels.policy_map import (
    PolicyMapSpecialization,
    compile_policy_map,
    policy_map,
)

_THREE_POINTERS = 3


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


def test_remaining_step_five_families_compile_pointer_abis() -> None:
    """Every remaining family compiles through the in-memory linker boundary."""
    add_vectors_artifact = compile_add_vectors(
        AddVectorsSpecialization(259, 7, "mish", 80)
    )
    add_bias_artifact = compile_add_bias_batched(
        AddBiasBatchedSpecialization(3, 5, 37, "mish", 80)
    )
    policy_artifact = compile_policy_map(PolicyMapSpecialization(2, 80))

    assert len(add_vectors_artifact.parameters) == _THREE_POINTERS
    assert add_vectors_artifact.grid == (2, 1, 1)
    assert len(add_bias_artifact.parameters) == _THREE_POINTERS
    assert add_bias_artifact.grid == (3, 1, 1)
    assert len(policy_artifact.parameters) == _THREE_POINTERS
    assert policy_artifact.grid == (15, 1, 1)


def test_policy_map_serializes_embedded_symbol_argument() -> None:
    """Policy gathering uses a module symbol rather than an external buffer."""
    builder = ExecutableBuilder()
    execution = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    output = execution.external_buffer(
        name="output",
        shape=(2, 1858),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=True,
    )
    input_ = execution.external_buffer(
        name="input",
        shape=(2, 4288),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    mapping = builder.add_symbol(compile_symbol(architecture="sm_80"))
    policy_map(
        builder,
        KernelCache(builder),
        output,
        input_,
        mapping,
        PolicyMapSpecialization(2, 80),
    )

    executable = builder.build()
    mapping_argument = executable.programs[0].nodes[0].arguments[2]

    assert mapping_argument.symbol.symbol_name == "lczero_bt4_mapping_table"
    assert not mapping_argument.HasField("allocation")
    assert "/const/mapping_table" not in {buffer.name for buffer in executable.buffers}
