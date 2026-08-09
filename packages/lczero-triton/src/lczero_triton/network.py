"""Lc0 graph definitions used by the example network."""

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2

M = 128
N = 256
K = 64
MATMUL_KERNEL_NAME = f"matmul_{M}_{N}_{K}"
_F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16


def matmul_kernel_name(m: int, n: int, k: int) -> str:
    """Return the logical name for one statically shaped matmul."""
    return f"matmul_{m}_{n}_{k}"


def build_matmul_graph(
    builder: ExecutableBuilder,
    m: int = M,
    n: int = N,
    k: int = K,
) -> None:
    """Append the statically shaped matmul graph to *builder*."""
    a = builder.buffer("a", (m, k), _F16)
    b = builder.buffer("b", (k, n), _F16)
    c = builder.buffer("c", (m, n), _F16)
    builder.call(matmul_kernel_name(m, n, k), a, b, c)
