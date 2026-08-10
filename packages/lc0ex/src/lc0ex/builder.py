"""Builder for serialized Lc0 neural executables."""

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

from lc0ex.buffer_builder import Allocation, Buffer, BufferBuilder, BufferLocation
from lc0ex.kernel_builder import KernelArtifact, KernelHandle
from lc0ex.module_loader import load_module
from lc0ex.proto import lc0ex_pb2

_MAGIC = 0x1C0E
_FORMAT = 1
_PROGRAM_NAME = "main"


@dataclass(frozen=True, slots=True)
class _KernelInvocation:
    """One invocation of a registered kernel."""

    kernel: KernelHandle
    arguments: tuple[Buffer, ...]
    readonly: frozenset[Buffer]


class ExecutableBuilder:
    """Build an Lc0 neural executable."""

    def __init__(self) -> None:
        """Initialize an empty executable builder."""
        self._target: tuple[lc0ex_pb2.Target.Vendor, str] | None = None
        self._buffers = BufferBuilder()
        self._kernels: dict[KernelHandle, KernelArtifact] = {}
        self._kernel_handles: dict[KernelArtifact, KernelHandle] = {}
        self._binary_functions: dict[
            tuple[lc0ex_pb2.Binary.Format, bytes, str], KernelArtifact
        ] = {}
        self._invocations: list[_KernelInvocation] = []

    def allocation(
        self,
        lifetime: lc0ex_pb2.Allocation.Lifetime,
    ) -> Allocation:
        """Create a logical device-memory allocation."""
        return self._buffers.allocation(lifetime)

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable's compilation target and return this builder."""
        self._target = (vendor, architecture)
        return self

    def add_module(
        self,
        manifest_path: str | PathLike[str],
    ) -> tuple[KernelHandle, ...]:
        """Load a compiled module manifest and return its kernel handles."""
        module = load_module(manifest_path)
        target = (module.target_vendor, module.target_architecture)
        if self._target is None:
            self._target = target
        elif self._target != target:
            message = "module target does not match the executable target"
            raise ValueError(message)

        return tuple(self.add_kernel(kernel) for kernel in module.kernels)

    def add_kernel(self, kernel: KernelArtifact) -> KernelHandle:
        """Register a compiled kernel and return its opaque handle."""
        if not kernel.function:
            message = "kernel function cannot be empty"
            raise ValueError(message)

        existing = self._kernel_handles.get(kernel)
        if existing is not None:
            return existing

        symbol = (kernel.binary_format, kernel.binary_data, kernel.function)
        existing_symbol = self._binary_functions.get(symbol)
        if existing_symbol is not None and existing_symbol != kernel:
            message = (
                f"kernel function {kernel.function!r} is already registered differently"
            )
            raise ValueError(message)

        handle = KernelHandle()
        self._kernels[handle] = kernel
        self._kernel_handles[kernel] = handle
        self._binary_functions[symbol] = kernel
        return handle

    def call(
        self,
        kernel: KernelHandle,
        *arguments: Buffer,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Append a call to a registered kernel to the executable graph.

        Arguments not included in *readonly* are treated as writable. Read-only
        accesses to the same buffer may execute concurrently.
        """
        artifact = self._kernels.get(kernel)
        if artifact is None:
            message = "kernel handle does not belong to this executable builder"
            raise ValueError(message)

        if len(arguments) != len(artifact.parameters):
            message = "kernel argument count does not match its ABI"
            raise ValueError(message)
        if any(
            parameter != lc0ex_pb2.PARAMETER_TYPE_POINTER
            for parameter in artifact.parameters
        ):
            message = "kernel calls only support pointer parameters"
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
                kernel=kernel,
                arguments=arguments,
                readonly=frozenset(
                    set(readonly)
                    | {
                        argument
                        for argument in arguments
                        if not self._buffers.is_reusable(argument)
                        and not self._buffers.is_writable(argument)
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
        dependencies, reusable_conflicts = self._analyze_invocations()
        locations = self._buffers.build(executable, reusable_conflicts)
        kernel_indices = self._build_kernels(executable)
        self._build_invocations(executable, dependencies, locations, kernel_indices)
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

    def _build_kernels(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
    ) -> dict[KernelHandle, int]:
        """Append registered binaries and kernels and return their indices."""
        binary_indices: dict[tuple[lc0ex_pb2.Binary.Format, bytes], int] = {}
        kernel_indices: dict[KernelHandle, int] = {}
        for handle, artifact in self._kernels.items():
            binary = (artifact.binary_format, artifact.binary_data)
            binary_idx = binary_indices.get(binary)
            if binary_idx is None:
                executable.binaries.add(
                    format=artifact.binary_format,
                    data=artifact.binary_data,
                )
                binary_idx = len(executable.binaries) - 1
                binary_indices[binary] = binary_idx
            executable.kernels.add(
                binary_idx=binary_idx,
                function=artifact.function,
                parameters=artifact.parameters,
            )
            kernel_indices[handle] = len(executable.kernels) - 1
        return kernel_indices

    def _analyze_invocations(
        self,
    ) -> tuple[list[list[int]], dict[Buffer, set[Buffer]]]:
        """Return reduced dependencies and reusable-buffer conflicts."""
        latest_writer: dict[Buffer, int] = {}
        readers: dict[Buffer, set[int]] = {}
        ancestors: list[int] = []
        accesses: dict[Buffer, set[int]] = {}

        dependencies: list[list[int]] = []
        for index, invocation in enumerate(self._invocations):
            for buffer in set(invocation.arguments):
                if self._buffers.is_reusable(buffer):
                    accesses.setdefault(buffer, set()).add(index)
            reduced_dependencies, ancestor_mask = self._invocation_dependencies(
                invocation,
                index,
                latest_writer,
                readers,
                ancestors,
            )
            ancestors.append(ancestor_mask)
            dependencies.append(reduced_dependencies)

        return dependencies, self._reusable_conflicts(accesses, ancestors)

    def _build_invocations(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        dependencies: list[list[int]],
        locations: dict[Buffer, BufferLocation],
        kernel_indices: dict[KernelHandle, int],
    ) -> None:
        """Append the invocation program using precomputed resource indices."""
        if not self._invocations:
            return

        program = executable.programs.add(name=_PROGRAM_NAME)
        for invocation, invocation_dependencies in zip(
            self._invocations,
            dependencies,
            strict=True,
        ):
            artifact = self._kernels[invocation.kernel]
            node = program.nodes.add(
                kernel_idx=kernel_indices[invocation.kernel],
                dependencies=invocation_dependencies,
                grid=artifact.grid,
                block=artifact.block,
                dynamic_shared_memory_bytes=artifact.dynamic_shared_memory_bytes,
            )
            for argument in invocation.arguments:
                location = locations[argument]
                node.arguments.add(
                    allocation_idx=location.allocation_idx,
                    allocation_offset=location.allocation_offset,
                )

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

    def _reusable_conflicts(
        self,
        accesses: dict[Buffer, set[int]],
        ancestors: list[int],
    ) -> dict[Buffer, set[Buffer]]:
        """Return reusable-buffer pairs whose accesses can overlap."""
        buffers = list(accesses)
        conflicts: dict[Buffer, set[Buffer]] = {buffer: set() for buffer in accesses}
        for first_index, first in enumerate(buffers):
            for second in buffers[first_index + 1 :]:
                if not self._buffers.share_allocation(first, second):
                    continue
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
