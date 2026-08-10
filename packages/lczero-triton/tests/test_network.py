"""Tests for the Lc0 graph definition."""

from lc0ex import ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lczero_triton.network import build_matmul_graph


def test_build_matmul_graph_is_separate_from_compilation() -> None:
    """The application graph only depends on the executable builder API."""
    builder = ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        "sm_120",
    )
    kernel = builder.add_kernel(
        KernelArtifact(
            binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
            binary_data=b"fake cubin",
            function="matmul_exported",
            parameters=(
                lc0ex_pb2.PARAMETER_TYPE_POINTER,
                lc0ex_pb2.PARAMETER_TYPE_POINTER,
                lc0ex_pb2.PARAMETER_TYPE_POINTER,
            ),
            grid=(2, 4, 1),
            block=(128, 1, 1),
            dynamic_shared_memory_bytes=0,
        ),
    )

    build_matmul_graph(builder, kernel)

    executable = builder.build()

    assert executable.programs[0].nodes[0].kernel_idx == 0
    assert [allocation.lifetime for allocation in executable.allocations] == [
        lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
        lc0ex_pb2.Allocation.LIFETIME_EXECUTION,
    ]
    assert [buffer.name for buffer in executable.buffers] == ["b", "a", "c"]
    assert [buffer.allocation_idx for buffer in executable.buffers] == [0, 1, 1]
