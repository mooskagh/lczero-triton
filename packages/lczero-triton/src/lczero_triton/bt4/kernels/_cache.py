"""Per-build cache for immutable BT4 kernel specializations."""

from collections.abc import Callable, Hashable
from typing import TypeVar

from lc0ex import ExecutableBuilder, KernelArtifact, KernelHandle

Specialization = TypeVar("Specialization", bound=Hashable)


class KernelCache:
    """Compile each compiler and immutable specialization pair at most once.

    The compiler receives only its cache-key specialization. It must not depend
    on graph buffers, allocation state, or unkeyed mutable configuration.
    """

    def __init__(self, builder: ExecutableBuilder) -> None:
        """Create an empty cache that registers artifacts with *builder*."""
        self._builder = builder
        self._handles: dict[tuple[object, Hashable], KernelHandle] = {}

    def get(
        self,
        compiler: Callable[[Specialization], KernelArtifact],
        specialization: Specialization,
    ) -> KernelHandle:
        """Return the handle compiled from one complete specialization key."""
        key = (compiler, specialization)
        handle = self._handles.get(key)
        if handle is None:
            handle = self._builder.add_kernel(compiler(specialization))
            self._handles[key] = handle
        return handle
