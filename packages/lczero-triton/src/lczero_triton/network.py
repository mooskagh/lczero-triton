"""Lc0 graph definitions used by the example network."""

from lc0ex import ExecutableBuilder, KernelHandle
from lc0ex.proto import lc0ex_pb2

M = 128
N = 256
K = 64
_F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16


def build_matmul_graph(
    builder: ExecutableBuilder,
    kernel: KernelHandle,
    m: int = M,
    n: int = N,
    k: int = K,
) -> None:
    """Append the statically shaped matmul graph to *builder*."""
    persistent = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_PERSISTENT)
    execution = builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION)
    a = execution.buffer((m, k), _F16, name="a")
    b = persistent.buffer((k, n), _F16, name="b")
    c = execution.buffer((m, n), _F16, name="c", writable=True)
    builder.call(kernel, a, b, c, readonly=(a, b))
