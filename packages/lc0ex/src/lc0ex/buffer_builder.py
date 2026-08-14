"""Opaque logical buffers and contiguous allocation planning."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from lc0ex.proto import lc0ex_pb2


@dataclass(frozen=True, slots=True, eq=False)
class Allocation:
    """A logical device-memory allocation owned by a buffer builder."""

    _persistent: bool

    def is_persistent(self) -> bool:
        """Return whether this allocation belongs to the executable."""
        return self._persistent


@dataclass(frozen=True, slots=True, eq=False)
class Buffer:
    """An opaque identity for one logical device-memory range."""


@dataclass(frozen=True, slots=True)
class _ExternalBuffer:
    """Canonical metadata for one named external range."""

    name: str
    shape: tuple[int, ...]
    dtype: lc0ex_pb2.Buffer.DataType


@dataclass(frozen=True, slots=True)
class _BufferRecord:
    """Private physical metadata for one opaque buffer handle."""

    allocation: Allocation
    size_bytes: int
    alignment_bytes: int
    external: _ExternalBuffer | None
    writable: bool


@dataclass(slots=True)
class _AllocationSlot:
    """One reusable raw-storage range in an allocation."""

    buffers: list[Buffer]
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class BufferLocation:
    """The planned location of one logical buffer."""

    allocation: Allocation
    offset: int


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """The packed layout of one persistent or execution allocation."""

    size_bytes: int
    alignment_bytes: int
    locations: dict[Buffer, BufferLocation]
    external_buffers: tuple[tuple[Buffer, _ExternalBuffer], ...]


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


class BufferBuilder:
    """Collect opaque buffers and pack each allocation."""

    def __init__(self) -> None:
        """Initialize an empty allocation collection."""
        self._persistent = Allocation(_persistent=True)
        self._buffers_by_allocation: dict[Allocation, list[Buffer]] = {
            self._persistent: [],
        }
        self._records: dict[Buffer, _BufferRecord] = {}
        self._external_by_allocation_and_name: dict[tuple[Allocation, str], Buffer] = {}

    def persistent_allocation(self) -> Allocation:
        """Return the executable-wide persistent allocation."""
        return self._persistent

    def execution_allocation(self) -> Allocation:
        """Create and return a new program execution allocation."""
        allocation = Allocation(_persistent=False)
        self._buffers_by_allocation[allocation] = []
        return allocation

    def external_buffer(  # noqa: PLR0913
        self,
        allocation: Allocation,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool,
        alignment_bytes: int | None,
    ) -> Buffer:
        """Create or retrieve a named external buffer in *allocation*."""
        key = (allocation, name)
        existing = self._external_by_allocation_and_name.get(key)
        if existing is not None:
            return existing

        normalized_shape = tuple(shape)
        element_size = data_type_size_bytes(dtype)
        resolved_alignment = (
            element_size if alignment_bytes is None else alignment_bytes
        )
        size_bytes = prod(normalized_shape) * element_size

        buffer = Buffer()
        external = _ExternalBuffer(name, normalized_shape, dtype)
        self._add_buffer(
            buffer,
            _BufferRecord(
                allocation,
                size_bytes,
                resolved_alignment,
                external,
                writable,
            ),
        )
        self._external_by_allocation_and_name[key] = buffer
        return buffer

    def temporary_buffer(
        self,
        allocation: Allocation,
        *,
        size_bytes: int,
        alignment_bytes: int,
    ) -> Buffer:
        """Create an unnamed raw-storage buffer in an execution allocation."""
        buffer = Buffer()
        self._add_buffer(
            buffer,
            _BufferRecord(
                allocation,
                size_bytes,
                alignment_bytes,
                None,
                writable=True,
            ),
        )
        return buffer

    def is_reusable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is an internal reusable range."""
        return self._records[buffer].external is None

    def is_writable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* may be modified by graph nodes."""
        return self._records[buffer].writable

    def share_allocation(self, first: Buffer, second: Buffer) -> bool:
        """Return whether two opaque ranges belong to one allocation."""
        return self._records[first].allocation is self._records[second].allocation

    def plan(
        self,
        allocation: Allocation,
        reusable_conflicts: dict[Buffer, set[Buffer]],
    ) -> AllocationPlan | None:
        """Pack one allocation and return its serialized plan."""
        buffers = self._buffers_by_allocation[allocation]
        fixed_buffers = [
            buffer for buffer in buffers if self._records[buffer].external is not None
        ]
        reusable_buffers = [
            buffer for buffer in buffers if buffer in reusable_conflicts
        ]
        if not fixed_buffers and not reusable_buffers:
            return None

        locations: dict[Buffer, BufferLocation] = {}
        allocation_size = 0
        allocation_alignment = 1
        for buffer in fixed_buffers:
            record = self._records[buffer]
            allocation_alignment = max(allocation_alignment, record.alignment_bytes)
            allocation_size = _align_up(allocation_size, record.alignment_bytes)
            locations[buffer] = BufferLocation(allocation, allocation_size)
            allocation_size += record.size_bytes

        allocation_size, reusable_alignment, reusable_offsets = (
            self._build_reusable_ranges(
                reusable_buffers,
                reusable_conflicts,
                allocation_size,
            )
        )
        allocation_alignment = max(allocation_alignment, reusable_alignment)
        for buffer, offset in reusable_offsets.items():
            locations[buffer] = BufferLocation(allocation, offset)

        external_buffers = tuple(
            (buffer, external)
            for buffer in fixed_buffers
            if (external := self._records[buffer].external) is not None
        )
        return AllocationPlan(
            allocation_size,
            allocation_alignment,
            locations,
            external_buffers,
        )

    def _add_buffer(self, buffer: Buffer, record: _BufferRecord) -> None:
        """Register one newly created opaque buffer handle."""
        self._buffers_by_allocation[record.allocation].append(buffer)
        self._records[buffer] = record

    def _build_reusable_ranges(
        self,
        buffers: list[Buffer],
        conflicts: dict[Buffer, set[Buffer]],
        allocation_size: int,
    ) -> tuple[int, int, dict[Buffer, int]]:
        """Pack internal raw ranges into reusable slots."""
        slots: list[_AllocationSlot] = []
        for buffer in buffers:
            record = self._records[buffer]
            compatible_slots = [
                slot
                for slot in slots
                if all(member not in conflicts[buffer] for member in slot.buffers)
            ]
            if compatible_slots:
                slot = min(
                    compatible_slots,
                    key=lambda candidate: (
                        candidate.size_bytes != record.size_bytes,
                        max(candidate.size_bytes, record.size_bytes),
                        max(candidate.alignment_bytes, record.alignment_bytes),
                    ),
                )
                slot.size_bytes = max(slot.size_bytes, record.size_bytes)
                slot.alignment_bytes = max(slot.alignment_bytes, record.alignment_bytes)
                slot.buffers.append(buffer)
            else:
                slots.append(
                    _AllocationSlot([buffer], record.size_bytes, record.alignment_bytes)
                )

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
