"""Logical buffer handles and executable allocation serialization support."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod

from lc0ex.proto import lc0ex_pb2


@dataclass(frozen=True, slots=True, eq=False)
class Allocation:
    """A logical device-memory allocation owned by an executable builder."""

    lifetime: lc0ex_pb2.Allocation.Lifetime
    _owner: "BufferBuilder" = field(repr=False)

    def buffer(
        self,
        shape: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
        *,
        name: str | None = None,
        writable: bool = False,
    ) -> "Buffer":
        """Create or retrieve a buffer within this allocation.

        Named buffers are runtime-visible fixed ranges. Unnamed execution
        buffers are internal ranges whose storage can be reused by the graph.
        """
        return self._owner.buffer(self, shape, dtype, name=name, writable=writable)

    def belongs_to(self, owner: "BufferBuilder") -> bool:
        """Return whether this allocation belongs to *owner*."""
        return self._owner is owner


@dataclass(frozen=True, slots=True, eq=False)
class Buffer:
    """A logical buffer that can be passed to other executable builders."""

    allocation: Allocation
    name: str | None
    shape: tuple[int, ...]
    dtype: lc0ex_pb2.Buffer.DataType


def data_type_size_bytes(dtype: lc0ex_pb2.Buffer.DataType) -> int:
    """Return the size in bytes of one value of *dtype*.

    Raises:
        KeyError: If *dtype* is not a supported concrete data type.

    """
    sizes = {
        lc0ex_pb2.Buffer.DATA_TYPE_F32: 4,
        lc0ex_pb2.Buffer.DATA_TYPE_U8: 1,
        lc0ex_pb2.Buffer.DATA_TYPE_F16: 2,
        lc0ex_pb2.Buffer.DATA_TYPE_U64: 8,
        lc0ex_pb2.Buffer.DATA_TYPE_BF16: 2,
    }
    return sizes[dtype]


@dataclass(slots=True)
class _AllocationSlot:
    """One reusable range in an allocation."""

    buffers: list[Buffer]
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class BufferLocation:
    """The planned location of one logical buffer."""

    allocation_idx: int
    allocation_offset: int


class BufferBuilder:
    """Collect logical allocations and their buffers."""

    def __init__(self) -> None:
        """Initialize an empty buffer collection."""
        self._allocations: list[Allocation] = []
        self._buffers_by_allocation: dict[Allocation, list[Buffer]] = {}
        self._named_buffers: dict[str, Buffer] = {}
        self._writable_buffers: set[Buffer] = set()

    def allocation(
        self,
        lifetime: lc0ex_pb2.Allocation.Lifetime,
    ) -> Allocation:
        """Create an allocation with the given runtime lifetime."""
        if lifetime not in {
            lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
            lc0ex_pb2.Allocation.LIFETIME_EXECUTION,
        }:
            message = "allocation lifetime must be PERSISTENT or EXECUTION"
            raise ValueError(message)
        allocation = Allocation(lifetime, self)
        self._allocations.append(allocation)
        self._buffers_by_allocation[allocation] = []
        return allocation

    def buffer(
        self,
        allocation: Allocation,
        shape: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
        *,
        name: str | None = None,
        writable: bool = False,
    ) -> Buffer:
        """Create or retrieve a buffer in *allocation*.

        A new buffer requires both *shape* and *dtype*. Named buffers are
        looked up globally; supplied values must match an existing definition.
        """
        normalized_shape = tuple(shape) if shape is not None else None
        existing = self._existing_buffer(allocation, name)

        if existing is not None:
            if existing.allocation is not allocation:
                message = f"buffer {name!r} belongs to a different allocation"
                raise ValueError(message)
            if normalized_shape is not None and normalized_shape != existing.shape:
                message = f"shape does not match existing buffer {name!r}"
                raise ValueError(message)
            if dtype is not None and dtype != existing.dtype:
                message = f"data type does not match existing buffer {name!r}"
                raise ValueError(message)
            if writable:
                self._writable_buffers.add(existing)
            return existing

        if normalized_shape is None or dtype is None:
            message = "shape and data type are required for new buffers"
            raise ValueError(message)
        result = Buffer(
            allocation=allocation,
            name=name,
            shape=normalized_shape,
            dtype=dtype,
        )
        self._buffers_by_allocation[allocation].append(result)
        if name is not None:
            self._named_buffers[name] = result
        if writable:
            self._writable_buffers.add(result)
        return result

    def _existing_buffer(
        self,
        allocation: Allocation,
        name: str | None,
    ) -> Buffer | None:
        """Validate *allocation* and look up its named buffer, if any."""
        if not allocation.belongs_to(self):
            message = "allocation does not belong to this executable builder"
            raise ValueError(message)
        if name is not None:
            return self._named_buffers.get(name)
        if allocation.lifetime != lc0ex_pb2.Allocation.LIFETIME_EXECUTION:
            message = "unnamed buffers require an EXECUTION allocation"
            raise ValueError(message)
        return None

    def owns(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is a handle created by this collection."""
        if not buffer.allocation.belongs_to(self):
            return False
        return any(
            buffer is candidate
            for buffers in self._buffers_by_allocation.values()
            for candidate in buffers
        )

    def is_reusable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is an internal reusable execution range."""
        return buffer.name is None

    def is_writable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* was declared writable."""
        return buffer in self._writable_buffers

    def build(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        reusable_conflicts: dict[Buffer, set[Buffer]],
    ) -> dict[Buffer, BufferLocation]:
        """Append allocations and return locations for every used buffer."""
        locations: dict[Buffer, BufferLocation] = {}
        for allocation in self._allocations:
            buffers = self._buffers_by_allocation[allocation]
            fixed_buffers = [buffer for buffer in buffers if buffer.name is not None]
            reusable_buffers = [
                buffer for buffer in buffers if buffer in reusable_conflicts
            ]
            if not fixed_buffers and not reusable_buffers:
                continue

            allocation_idx = len(executable.allocations)
            allocation_size = 0
            allocation_alignment = 1
            for buffer in fixed_buffers:
                alignment = data_type_size_bytes(buffer.dtype)
                allocation_alignment = max(allocation_alignment, alignment)
                allocation_size = _align_up(allocation_size, alignment)
                locations[buffer] = BufferLocation(allocation_idx, allocation_size)
                allocation_size += prod(buffer.shape) * alignment

            allocation_size, reusable_alignment, reusable_offsets = (
                self._build_reusable_ranges(
                    reusable_buffers,
                    reusable_conflicts,
                    allocation_size,
                )
            )
            allocation_alignment = max(allocation_alignment, reusable_alignment)
            for buffer, offset in reusable_offsets.items():
                locations[buffer] = BufferLocation(allocation_idx, offset)

            executable.allocations.add(
                size_bytes=allocation_size,
                alignment_bytes=allocation_alignment,
                lifetime=allocation.lifetime,
            )
            for buffer in fixed_buffers:
                location = locations[buffer]
                executable.buffers.add(
                    name=buffer.name,
                    allocation_idx=location.allocation_idx,
                    allocation_offset=location.allocation_offset,
                    data_type=buffer.dtype,
                    shape=buffer.shape,
                )
        return locations

    def _build_reusable_ranges(
        self,
        buffers: list[Buffer],
        conflicts: dict[Buffer, set[Buffer]],
        allocation_size: int,
    ) -> tuple[int, int, dict[Buffer, int]]:
        """Pack internal buffers into reusable ranges after fixed buffers."""
        slots: list[_AllocationSlot] = []
        for buffer in buffers:
            size_bytes = prod(buffer.shape) * data_type_size_bytes(buffer.dtype)
            alignment_bytes = data_type_size_bytes(buffer.dtype)
            compatible_slots = [
                slot
                for slot in slots
                if all(member not in conflicts[buffer] for member in slot.buffers)
            ]
            if compatible_slots:
                slot = min(
                    compatible_slots,
                    key=lambda candidate: (
                        candidate.size_bytes != size_bytes,
                        max(candidate.size_bytes, size_bytes),
                        max(candidate.alignment_bytes, alignment_bytes),
                    ),
                )
                slot.size_bytes = max(slot.size_bytes, size_bytes)
                slot.alignment_bytes = max(slot.alignment_bytes, alignment_bytes)
                slot.buffers.append(buffer)
            else:
                slot = _AllocationSlot([buffer], size_bytes, alignment_bytes)
                slots.append(slot)

        allocation_alignment = 1
        offsets: dict[Buffer, int] = {}
        for slot in slots:
            allocation_alignment = max(allocation_alignment, slot.alignment_bytes)
            allocation_size = _align_up(allocation_size, slot.alignment_bytes)
            for buffer in slot.buffers:
                offsets[buffer] = allocation_size
            allocation_size += slot.size_bytes
        return allocation_size, allocation_alignment, offsets


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
