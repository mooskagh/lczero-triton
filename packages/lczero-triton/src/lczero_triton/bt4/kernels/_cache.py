"""Per-build cache for immutable BT4 kernel specializations."""

import logging
from collections.abc import Callable, Hashable
from time import perf_counter
from typing import TypeVar

from lc0ex import ExecutableBuilder, KernelArtifact, KernelHandle

Specialization = TypeVar("Specialization", bound=Hashable)
_LOGGER = logging.getLogger(__name__)


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
        compiler_name = getattr(compiler, "__name__", type(compiler).__name__)
        if handle is None:
            _LOGGER.info(
                "compiling kernel %s for specialization %r",
                compiler_name,
                specialization,
            )
            started = perf_counter()
            artifact = compiler(specialization)
            handle = self._builder.add_kernel(artifact)
            self._handles[key] = handle
            _LOGGER.info(
                "kernel %s compilation completed for specialization %r in %.2fs",
                compiler_name,
                specialization,
                perf_counter() - started,
            )
        else:
            _LOGGER.debug(
                "reusing compiled kernel %s for specialization %r",
                compiler_name,
                specialization,
            )
        return handle
