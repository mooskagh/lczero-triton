"""Builder for serialized Lc0 neural executables."""

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

from lc0ex.buffer_builder import Allocation, AllocationPlan, Buffer, BufferBuilder
from lc0ex.kernel_builder import (
    KernelArtifact,
    KernelHandle,
    SymbolArtifact,
    SymbolHandle,
)
from lc0ex.proto import lc0ex_pb2

_MAGIC = 0x1C0E
_FORMAT = 1
_PROGRAM_NAME = "main"


@dataclass(frozen=True, slots=True)
class _KernelInvocation:
    """One invocation of a registered kernel."""

    kernel: KernelHandle
    arguments: tuple[Buffer | SymbolHandle, ...]
    readonly: frozenset[Buffer]


class ProgramBuilder:
    """Build one program and its private execution allocation."""

    def __init__(
        self,
        owner: "ExecutableBuilder",
        name: str,
        metadata: bytes | None,
        token: object,
    ) -> None:
        """Initialize a program owned by *owner*."""
        if token is not owner._program_token:  # noqa: SLF001
            message = "programs must be created by an executable builder"
            raise ValueError(message)
        self._owner = owner
        self._name = name
        self.metadata = metadata
        self._allocation = owner._buffers.execution_allocation()  # noqa: SLF001
        self._invocations: list[_KernelInvocation] = []

    @property
    def name(self) -> str:
        """Return the immutable program name."""
        return self._name

    @property
    def allocation(self) -> Allocation:
        """Return this program's private execution allocation."""
        return self._allocation

    def __enter__(self) -> Self:
        """Make this program the target of owner-level graph calls."""
        self._owner._push_program(self)  # noqa: SLF001
        return self

    def __exit__(self, *_: object) -> None:
        """Restore the previous active program."""
        self._owner._pop_program(self)  # noqa: SLF001

    def buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named execution buffer in this program's allocation."""
        return self._owner._buffers.external_buffer(  # noqa: SLF001
            self._allocation,
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def temporary_buffer(self, *, size_bytes: int, alignment_bytes: int) -> Buffer:
        """Create an unnamed execution buffer in this program's allocation."""
        return self._owner._buffers.temporary_buffer(  # noqa: SLF001
            self._allocation,
            size_bytes=size_bytes,
            alignment_bytes=alignment_bytes,
        )

    def add_kernel(self, kernel: KernelArtifact) -> KernelHandle:
        """Register a kernel in the owning executable."""
        return self._owner.add_kernel(kernel)

    def add_symbol(self, symbol: SymbolArtifact) -> SymbolHandle:
        """Register a symbol in the owning executable."""
        return self._owner.add_symbol(symbol)

    def call(
        self,
        kernel: KernelHandle,
        *arguments: Buffer | SymbolHandle,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Append a call to this program."""
        self._owner._call_in_program(  # noqa: SLF001
            self,
            kernel,
            *arguments,
            readonly=readonly,
        )


class ExecutableBuilder:
    """Build an Lc0 neural executable with shared persistent storage."""

    def __init__(self) -> None:
        """Initialize an empty executable builder."""
        self._target: tuple[lc0ex_pb2.Target.Vendor, str] | None = None
        self._metadata: bytes | None = None
        self._buffers = BufferBuilder()
        self._kernels: dict[KernelHandle, KernelArtifact] = {}
        self._kernel_handles: dict[KernelArtifact, KernelHandle] = {}
        self._binary_functions: dict[
            tuple[lc0ex_pb2.Binary.Format, bytes, str], KernelArtifact
        ] = {}
        self._symbols: dict[SymbolHandle, SymbolArtifact] = {}
        self._symbol_handles: dict[SymbolArtifact, SymbolHandle] = {}
        self._programs: list[ProgramBuilder] = []
        self._program_indices: dict[str, ProgramBuilder] = {}
        self._active_programs: list[ProgramBuilder] = []
        self._default_program: ProgramBuilder | None = None
        self._program_token = object()

    def persistent_buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named persistent buffer in the executable allocation."""
        return self._buffers.external_buffer(
            self._buffers.persistent_allocation(),
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def execution_buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named buffer in the implicit single program."""
        return self._get_default_program().buffer(
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def temporary_buffer(self, *, size_bytes: int, alignment_bytes: int) -> Buffer:
        """Create a raw buffer in the implicit single program."""
        return self._get_default_program().temporary_buffer(
            size_bytes=size_bytes,
            alignment_bytes=alignment_bytes,
        )

    def program(
        self,
        *,
        name: str,
        metadata: bytes | None = None,
    ) -> ProgramBuilder:
        """Create a named program with a private execution allocation."""
        if not name:
            message = "program name cannot be empty"
            raise ValueError(message)
        if name in self._program_indices:
            message = f"program {name!r} already exists"
            raise ValueError(message)
        normalized_metadata = None if metadata is None else bytes(metadata)
        result = ProgramBuilder(self, name, normalized_metadata, self._program_token)
        self._programs.append(result)
        self._program_indices[name] = result
        return result

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable's compilation target and return this builder."""
        target = (vendor, architecture)
        if self._target is not None and self._target != target:
            message = "target does not match the executable target"
            raise ValueError(message)
        self._target = target
        return self

    def set_metadata(self, metadata: bytes) -> Self:
        """Set opaque executable metadata and return this builder."""
        normalized = bytes(metadata)
        if self._metadata is not None and self._metadata != normalized:
            message = "metadata does not match the executable metadata"
            raise ValueError(message)
        self._metadata = normalized
        return self

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

    def add_symbol(self, symbol: SymbolArtifact) -> SymbolHandle:
        """Register an immutable module symbol and return its opaque handle."""
        if not symbol.symbol_name:
            message = "symbol name cannot be empty"
            raise ValueError(message)
        existing = self._symbol_handles.get(symbol)
        if existing is not None:
            return existing

        handle = SymbolHandle()
        self._symbols[handle] = symbol
        self._symbol_handles[symbol] = handle
        return handle

    def call(
        self,
        kernel: KernelHandle,
        *arguments: Buffer | SymbolHandle,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Append a call to the active or inferred program."""
        self._call_in_program(
            self._infer_program(arguments),
            kernel,
            *arguments,
            readonly=readonly,
        )

    def build(self) -> lc0ex_pb2.NeuralExecutable:
        """Create and return a new neural executable."""
        executable = lc0ex_pb2.NeuralExecutable(magic=_MAGIC, format=_FORMAT)
        if self._metadata is not None:
            executable.metadata = self._metadata
        if self._target is not None:
            vendor, architecture = self._target
            executable.target.vendor = vendor
            executable.target.architecture = architecture

        persistent_conflicts: dict[Buffer, set[Buffer]] = {}
        persistent_plan = self._buffers.plan(
            self._buffers.persistent_allocation(),
            persistent_conflicts,
        )
        if persistent_plan is not None:
            executable.persistent_allocation.size_bytes = persistent_plan.size_bytes
            executable.persistent_allocation.alignment_bytes = (
                persistent_plan.alignment_bytes
            )
            self._build_buffers(executable, persistent_plan)

        kernel_indices, symbol_locations = self._build_exports(executable)
        for program in self._programs:
            dependencies, reusable_conflicts = self._analyze_invocations(program)
            plan = self._buffers.plan(
                program._allocation,  # noqa: SLF001
                reusable_conflicts,
            )
            destination = executable.programs.add(name=program.name)
            if program.metadata is not None:
                destination.metadata = program.metadata
            if plan is not None:
                destination.execution_allocation.size_bytes = plan.size_bytes
                destination.execution_allocation.alignment_bytes = plan.alignment_bytes
                self._build_buffers(destination, plan)
            self._build_invocations(
                destination=destination,
                program=program,
                dependencies=dependencies,
                plans=(persistent_plan, plan),
                exports=(kernel_indices, symbol_locations),
            )
        return executable

    def build_and_write(
        self,
        path: str | PathLike[str],
    ) -> lc0ex_pb2.NeuralExecutable:
        """Build, serialize, and return the executable."""
        executable = self.build()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(executable.SerializeToString())
        return executable

    def _push_program(self, program: ProgramBuilder) -> None:
        """Push a program onto the active graph target stack."""
        if program._owner is not self:  # noqa: SLF001
            message = "program does not belong to this executable builder"
            raise ValueError(message)
        self._active_programs.append(program)

    def _pop_program(self, program: ProgramBuilder) -> None:
        """Pop the expected active graph target."""
        if not self._active_programs or self._active_programs[-1] is not program:
            message = "program context is not the active builder context"
            raise RuntimeError(message)
        self._active_programs.pop()

    def _get_default_program(self) -> ProgramBuilder:
        """Create the implicit single-program graph when needed."""
        if self._default_program is None:
            existing = self._program_indices.get(_PROGRAM_NAME)
            self._default_program = (
                existing if existing is not None else self.program(name=_PROGRAM_NAME)
            )
        return self._default_program

    def _infer_program(
        self,
        arguments: tuple[Buffer | SymbolHandle, ...],
    ) -> ProgramBuilder:
        """Infer a program from active context and execution buffers."""
        if self._active_programs:
            return self._active_programs[-1]
        programs = {
            self._program_for_buffer(argument)
            for argument in arguments
            if isinstance(argument, Buffer)
            and self._buffers.owns(argument)
            and not self._buffers.allocation_of(argument).is_persistent()
        }
        if len(programs) > 1:
            message = "kernel arguments belong to different programs"
            raise ValueError(message)
        return next(iter(programs), self._get_default_program())

    def _program_for_buffer(self, buffer: Buffer) -> ProgramBuilder:
        """Find the program owning an execution buffer."""
        allocation = self._buffers.allocation_of(buffer)
        for program in self._programs:
            if program._allocation is allocation:  # noqa: SLF001
                return program
        message = "execution buffer does not belong to a known program"
        raise RuntimeError(message)

    def _call_in_program(
        self,
        program: ProgramBuilder,
        kernel: KernelHandle,
        *arguments: Buffer | SymbolHandle,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Validate and append one invocation to *program*."""
        if program._owner is not self:  # noqa: SLF001
            message = "program does not belong to this executable builder"
            raise ValueError(message)
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
        if any(
            not isinstance(argument, Buffer | SymbolHandle) for argument in arguments
        ):
            message = "kernel arguments must be buffers or symbols"
            raise ValueError(message)
        if any(
            isinstance(argument, Buffer) and not self._buffers.owns(argument)
            for argument in arguments
        ) or any(
            isinstance(argument, SymbolHandle) and argument not in self._symbols
            for argument in arguments
        ):
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
        execution_programs = {
            self._program_for_buffer(argument)
            for argument in arguments
            if isinstance(argument, Buffer)
            and not self._buffers.allocation_of(argument).is_persistent()
        }
        if execution_programs and execution_programs != {program}:
            message = "kernel arguments belong to a different program"
            raise ValueError(message)

        program._invocations.append(  # noqa: SLF001
            _KernelInvocation(
                kernel=kernel,
                arguments=arguments,
                readonly=frozenset(
                    set(readonly)
                    | {
                        argument
                        for argument in arguments
                        if isinstance(argument, Buffer)
                        if not self._buffers.is_reusable(argument)
                        and not self._buffers.is_writable(argument)
                    },
                ),
            ),
        )

    def _build_exports(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
    ) -> tuple[dict[KernelHandle, int], dict[SymbolHandle, tuple[int, str]]]:
        """Append registered binaries and kernels and return export locations."""
        binary_indices: dict[tuple[lc0ex_pb2.Binary.Format, bytes], int] = {}
        kernel_indices: dict[KernelHandle, int] = {}

        def binary_index(
            binary_format: lc0ex_pb2.Binary.Format,
            binary_data: bytes,
        ) -> int:
            binary = (binary_format, binary_data)
            binary_idx = binary_indices.get(binary)
            if binary_idx is None:
                executable.binaries.add(format=binary_format, data=binary_data)
                binary_idx = len(executable.binaries) - 1
                binary_indices[binary] = binary_idx
            return binary_idx

        for handle, artifact in self._kernels.items():
            binary_idx = binary_index(artifact.binary_format, artifact.binary_data)
            executable.kernels.add(
                binary_idx=binary_idx,
                function=artifact.function,
                parameters=artifact.parameters,
            )
            kernel_indices[handle] = len(executable.kernels) - 1
        symbol_locations = {
            handle: (
                binary_index(artifact.binary_format, artifact.binary_data),
                artifact.symbol_name,
            )
            for handle, artifact in self._symbols.items()
        }
        return kernel_indices, symbol_locations

    def _analyze_invocations(
        self,
        program: ProgramBuilder,
    ) -> tuple[list[list[int]], dict[Buffer, set[Buffer]]]:
        """Return reduced dependencies and reusable-buffer conflicts."""
        latest_writer: dict[Buffer, int] = {}
        readers: dict[Buffer, set[int]] = {}
        ancestors: list[int] = []
        accesses: dict[Buffer, set[int]] = {}

        dependencies: list[list[int]] = []
        for index, invocation in enumerate(program._invocations):  # noqa: SLF001
            for buffer in {
                argument
                for argument in invocation.arguments
                if isinstance(argument, Buffer)
            }:
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

    def _build_buffers(
        self,
        destination: lc0ex_pb2.NeuralExecutable | lc0ex_pb2.Program,
        plan: AllocationPlan,
    ) -> None:
        """Serialize named external buffers into an executable or program."""
        buffers = destination.buffers
        for buffer, external in plan.external_buffers:
            location = plan.locations[buffer]
            buffers.add(
                name=external.name,
                offset=location.offset,
                data_type=external.dtype,
                shape=external.shape,
            )

    def _build_invocations(
        self,
        *,
        destination: lc0ex_pb2.Program,
        program: ProgramBuilder,
        dependencies: list[list[int]],
        plans: tuple[AllocationPlan | None, AllocationPlan | None],
        exports: tuple[
            dict[KernelHandle, int],
            dict[SymbolHandle, tuple[int, str]],
        ],
    ) -> None:
        """Serialize one program's invocation graph."""
        persistent_plan, plan = plans
        kernel_indices, symbol_locations = exports
        locations = {}
        if persistent_plan is not None:
            locations.update(persistent_plan.locations)
        if plan is not None:
            locations.update(plan.locations)
        for invocation, invocation_dependencies in zip(
            program._invocations,  # noqa: SLF001
            dependencies,
            strict=True,
        ):
            artifact = self._kernels[invocation.kernel]
            node = destination.nodes.add(
                kernel_idx=kernel_indices[invocation.kernel],
                dependencies=invocation_dependencies,
                grid=artifact.grid,
                block=artifact.block,
                dynamic_shared_memory_bytes=artifact.dynamic_shared_memory_bytes,
            )
            for argument in invocation.arguments:
                if isinstance(argument, Buffer):
                    location = locations[argument]
                    allocation_kind = (
                        lc0ex_pb2.Node.Argument.AllocationLocation.AllocationKind
                    )
                    allocation = (
                        allocation_kind.ALLOCATION_PERSISTENT
                        if location.allocation.is_persistent()
                        else allocation_kind.ALLOCATION_EXECUTION
                    )
                    node.arguments.add(
                        allocation=lc0ex_pb2.Node.Argument.AllocationLocation(
                            kind=allocation,
                            offset=location.offset,
                        )
                    )
                else:
                    binary_idx, symbol_name = symbol_locations[argument]
                    node.arguments.add(
                        symbol=lc0ex_pb2.Node.Argument.Symbol(
                            binary_idx=binary_idx,
                            symbol_name=symbol_name,
                        )
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
        for buffer in {
            argument
            for argument in invocation.arguments
            if isinstance(argument, Buffer)
        }:
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
