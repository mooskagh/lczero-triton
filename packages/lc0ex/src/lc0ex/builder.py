"""Builder for serialized Lc0 neural executables."""

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

from lc0ex.buffer_builder import Buffer, BufferBuilder
from lc0ex.kernel_builder import KernelArtifact
from lc0ex.module_loader import load_module
from lc0ex.proto import lc0ex_pb2

_MAGIC = 0x1C0E
_FORMAT = 1
_PROGRAM_NAME = "main"


@dataclass(frozen=True, slots=True)
class _KernelInvocation:
    """One invocation of a registered kernel."""

    name: str
    kernel: str
    arguments: tuple[Buffer, ...]
    readonly: frozenset[Buffer]


class ExecutableBuilder:
    """Build an Lc0 neural executable."""

    def __init__(self) -> None:
        """Initialize an empty executable builder."""
        self._target: tuple[lc0ex_pb2.Target.Vendor, str] | None = None
        self._buffers = BufferBuilder()
        self._writable_buffers: set[Buffer] = set()
        self._kernels: dict[str, KernelArtifact] = {}
        self._invocations: list[_KernelInvocation] = []

    def buffer(
        self,
        name: str,
        shape: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
        *,
        writable: bool = False,
    ) -> Buffer:
        """Create or retrieve a persistent logical buffer.

        Persistent buffers are read-only by default. Passing ``writable=True``
        permits calls that use the buffer to write to it.
        """
        buffer = self._buffers.buffer(name, shape, dtype)
        if writable:
            self._writable_buffers.add(buffer)
        return buffer

    def tmp_buffer(
        self,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
    ) -> Buffer:
        """Create an execution-lifetime temporary logical buffer."""
        return self._buffers.tmp_buffer(shape, dtype)

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable's compilation target and return this builder."""
        self._target = (vendor, architecture)
        return self

    def add_module(self, manifest_path: str | PathLike[str]) -> Self:
        """Load a compiled module manifest and register its exported kernels."""
        module = load_module(manifest_path)
        target = (module.target_vendor, module.target_architecture)
        if self._target is None:
            self._target = target
        elif self._target != target:
            message = "module target does not match the executable target"
            raise ValueError(message)

        for kernel in module.kernels:
            self.add_kernel(kernel.name, kernel.artifact)
        return self

    def add_kernel(self, name: str, kernel: KernelArtifact) -> Self:
        """Register a compiled kernel under *name* and return this builder."""
        if not name:
            message = "kernel name cannot be empty"
            raise ValueError(message)

        existing = self._kernels.get(name)
        if existing is not None:
            if existing != kernel:
                message = f"kernel {name!r} is already registered differently"
                raise ValueError(message)
            return self

        for registered in self._kernels.values():
            if registered.binary_name != kernel.binary_name:
                continue
            if (
                registered.binary_format != kernel.binary_format
                or registered.binary_data != kernel.binary_data
            ):
                message = f"binary {kernel.binary_name!r} is already registered"
                raise ValueError(message)

        self._kernels[name] = kernel
        return self

    def call(
        self,
        kernel: str,
        *arguments: Buffer,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Append a call to a registered kernel to the executable graph.

        Arguments not included in *readonly* are treated as writable. Read-only
        accesses to the same buffer may execute concurrently.
        """
        artifact = self._kernels[kernel]

        if len(arguments) != len(artifact.parameters):
            message = "kernel argument count does not match its ABI"
            raise ValueError(message)
        if any(not self._buffers.owns(argument) for argument in arguments):
            message = "kernel arguments must belong to this executable builder"
            raise ValueError(message)
        if any(not self._buffers.owns(buffer) for buffer in readonly):
            message = "read-only buffers must belong to this executable builder"
            raise ValueError(message)
        if any(
            not any(buffer is argument for argument in arguments) for buffer in readonly
        ):
            message = "read-only buffers must be kernel arguments"
            raise ValueError(message)

        self._invocations.append(
            _KernelInvocation(
                name=f"node_{len(self._invocations)}",
                kernel=kernel,
                arguments=arguments,
                readonly=frozenset(
                    set(readonly)
                    | {
                        argument
                        for argument in arguments
                        if not self._buffers.is_temporary(argument)
                        and argument not in self._writable_buffers
                    },
                ),
            ),
        )

    def build(self) -> lc0ex_pb2.NeuralExecutable:
        """Create and return a new neural executable."""
        executable = lc0ex_pb2.NeuralExecutable(magic=_MAGIC, format=_FORMAT)
        if self._target is not None:
            vendor, architecture = self._target
            executable.target.vendor = vendor
            executable.target.architecture = architecture
        temporary_conflicts = self._build_invocations(executable)
        self._buffers.build(executable, temporary_conflicts)
        self._build_kernels(executable)
        return executable

    def build_and_write(
        self,
        path: str | PathLike[str],
    ) -> lc0ex_pb2.NeuralExecutable:
        """Build the executable, serialize it to *path*, and return it.

        Raises:
            google.protobuf.message.EncodeError: If required fields are unset.

        """
        executable = self.build()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(executable.SerializeToString())
        return executable

    def _build_kernels(self, executable: lc0ex_pb2.NeuralExecutable) -> None:
        """Append registered binaries and kernels."""
        binaries: set[str] = set()
        for name, artifact in self._kernels.items():
            if artifact.binary_name not in binaries:
                executable.binaries.add(
                    name=artifact.binary_name,
                    format=artifact.binary_format,
                    data=artifact.binary_data,
                )
                binaries.add(artifact.binary_name)
            executable.kernels.add(
                name=name,
                binary=artifact.binary_name,
                function=artifact.function,
                parameters=artifact.parameters,
            )

    def _build_invocations(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
    ) -> dict[Buffer, set[Buffer]]:
        """Append the invocation program and return temporary buffer conflicts."""
        latest_writer: dict[Buffer, int] = {}
        readers: dict[Buffer, set[int]] = {}
        ancestors: list[int] = []
        accesses: dict[Buffer, set[int]] = {}

        if not self._invocations:
            return self._temporary_conflicts(accesses, ancestors)

        program = executable.programs.add(name=_PROGRAM_NAME)

        for index, invocation in enumerate(self._invocations):
            artifact = self._kernels[invocation.kernel]
            for buffer in set(invocation.arguments):
                if self._buffers.is_temporary(buffer):
                    accesses.setdefault(buffer, set()).add(index)
            reduced_dependencies, ancestor_mask = self._invocation_dependencies(
                invocation,
                index,
                latest_writer,
                readers,
                ancestors,
            )
            ancestors.append(ancestor_mask)

            program.nodes.add(
                name=invocation.name,
                kernel=invocation.kernel,
                dependencies=[
                    self._invocations[dependency].name
                    for dependency in reduced_dependencies
                ],
                arguments=[argument.name for argument in invocation.arguments],
                grid=artifact.grid,
                block=artifact.block,
                dynamic_shared_memory_bytes=artifact.dynamic_shared_memory_bytes,
            )

        return self._temporary_conflicts(accesses, ancestors)

    def _invocation_dependencies(
        self,
        invocation: _KernelInvocation,
        index: int,
        latest_writer: dict[Buffer, int],
        readers: dict[Buffer, set[int]],
        ancestors: list[int],
    ) -> tuple[list[int], int]:
        """Update access hazards and return reduced dependencies for one node."""
        dependencies: set[int] = set()
        for buffer in set(invocation.arguments):
            if buffer in invocation.readonly:
                writer = latest_writer.get(buffer)
                if writer is not None:
                    dependencies.add(writer)
                readers.setdefault(buffer, set()).add(index)
                continue

            writer = latest_writer.get(buffer)
            if writer is not None:
                dependencies.add(writer)
            dependencies.update(readers.get(buffer, set()))
            latest_writer[buffer] = index
            readers[buffer] = set()

        reduced_dependencies: list[int] = []
        covered_ancestors = 0
        for dependency in sorted(dependencies, reverse=True):
            if covered_ancestors & (1 << dependency):
                continue
            reduced_dependencies.append(dependency)
            covered_ancestors |= ancestors[dependency] | (1 << dependency)
        reduced_dependencies.reverse()
        ancestor_mask = 0
        for dependency in reduced_dependencies:
            ancestor_mask |= ancestors[dependency] | (1 << dependency)
        return reduced_dependencies, ancestor_mask

    def _temporary_conflicts(
        self,
        accesses: dict[Buffer, set[int]],
        ancestors: list[int],
    ) -> dict[Buffer, set[Buffer]]:
        """Return temporary-buffer pairs whose accesses can overlap."""
        buffers = list(accesses)
        conflicts: dict[Buffer, set[Buffer]] = {
            buffer: set() for buffer in self._buffers.temporary_buffers()
        }
        for first_index, first in enumerate(buffers):
            for second in buffers[first_index + 1 :]:
                first_before_second = all(
                    ancestors[second_access] & (1 << first_access)
                    for first_access in accesses[first]
                    for second_access in accesses[second]
                )
                second_before_first = all(
                    ancestors[first_access] & (1 << second_access)
                    for first_access in accesses[first]
                    for second_access in accesses[second]
                )
                if first_before_second or second_before_first:
                    continue
                conflicts[first].add(second)
                conflicts[second].add(first)
        return conflicts
