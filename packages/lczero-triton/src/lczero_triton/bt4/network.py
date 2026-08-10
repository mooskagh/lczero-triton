"""BT4 graph construction entry points."""

from collections.abc import Sequence

from lc0ex import ExecutableBuilder, KernelHandle


def build_graph(
    builder: ExecutableBuilder,
    kernels: Sequence[KernelHandle],
) -> None:
    """Build the fixed BT4 graph once its kernel families are available."""
    del builder, kernels
    message = "BT4 graph construction is not implemented yet"
    raise NotImplementedError(message)
