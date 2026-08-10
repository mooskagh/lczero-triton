"""Opaque logical buffer handles and executable allocation serialization."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod

from lc0ex.proto import lc0ex_pb2

_MAX_ALLOCATION_SIZE_BYTES = (1 << 64) - 1


@dataclass(frozen=True, slots=True, eq=False)
class Allocation:
    """A logical device-memory allocation owned by an executable builder."""

    lifetime: lc0ex_pb2.Allocation.Lifetime
    _owner: "BufferBuilder" = field(repr=False)

    def external_buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> "Buffer":
        """Create or retrieve a named external range in this allocation."""
        return self._owner.external_buffer(
            self,
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def temporary_buffer(
        self,
        *,
        size_bytes: int,
        alignment_bytes: int,
    ) -> "Buffer":
        """Create an unnamed reusable execution range with raw storage size."""
        return self._owner.temporary_buffer(
            self,
            size_bytes=size_bytes,
            alignment_bytes=alignment_bytes,
        )

    def belongs_to(self, owner: "BufferBuilder") -> bool:
        """Return whether this allocation belongs to *owner*."""
        return self._owner is owner


@dataclass(frozen=True, slots=True, eq=False)
class Buffer:
    """An opaque identity for one logical device-memory range.

    Do not add shape, data-type, layout, or stride accessors. Kernels receive
    pointers only, so their dimensions and element types must be explicit
    specialization parameters at graph-construction call sites.
    """


@dataclass(frozen=True, slots=True)
class _ExternalBuffer:
    """Canonical runtime metadata for one named external range."""

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
    """One reusable raw-storage range in an allocation."""

    buffers: list[Buffer]
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class BufferLocation:
    """The planned location of one logical buffer."""

    allocation_idx: int
    allocation_offset: int


class BufferBuilder:
    """Collect opaque logical buffers and serialize their physical storage."""

    def __init__(self) -> None:
        """Initialize an empty buffer collection."""
        self._allocations: list[Allocation] = []
        self._buffers_by_allocation: dict[Allocation, list[Buffer]] = {}
        self._records: dict[Buffer, _BufferRecord] = {}
        self._external_by_name: dict[str, Buffer] = {}

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
        self._require_owned_allocation(allocation)
        normalized_shape = tuple(shape)
        element_size = data_type_size_bytes(dtype)
        _validate_name(name)
        _validate_shape(normalized_shape)
        resolved_alignment = (
            element_size if alignment_bytes is None else alignment_bytes
        )
        _validate_alignment(resolved_alignment, minimum=element_size)
        size_bytes = _checked_product(normalized_shape, element_size)

        existing = self._external_by_name.get(name)
        if existing is not None:
            record = self._records[existing]
            _validate_matching_external(
                record,
                allocation=allocation,
                name=name,
                shape=normalized_shape,
                dtype=dtype,
                writable=writable,
                alignment_bytes=resolved_alignment,
            )
            return existing

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
        self._external_by_name[name] = buffer
        return buffer

    def temporary_buffer(
        self,
        allocation: Allocation,
        *,
        size_bytes: int,
        alignment_bytes: int,
    ) -> Buffer:
        """Create an unnamed raw-storage buffer in an execution allocation."""
        self._require_owned_allocation(allocation)
        if allocation.lifetime != lc0ex_pb2.Allocation.LIFETIME_EXECUTION:
            message = "temporary buffers require an EXECUTION allocation"
            raise ValueError(message)
        _validate_size(size_bytes)
        _validate_alignment(alignment_bytes, minimum=1)
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

    def owns(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is a handle created by this collection."""
        return buffer in self._records

    def is_reusable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is an internal reusable execution range."""
        return self._records[buffer].external is None

    def is_writable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* may be modified by graph nodes."""
        return self._records[buffer].writable

    def share_allocation(self, first: Buffer, second: Buffer) -> bool:
        """Return whether two opaque ranges belong to one allocation."""
        return self._records[first].allocation is self._records[second].allocation

    def build(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        reusable_conflicts: dict[Buffer, set[Buffer]],
    ) -> dict[Buffer, BufferLocation]:
        """Append allocations and return locations for every used buffer."""
        locations: dict[Buffer, BufferLocation] = {}
        for allocation in self._allocations:
            buffers = self._buffers_by_allocation[allocation]
            fixed_buffers = [
                buffer
                for buffer in buffers
                if self._records[buffer].external is not None
            ]
            reusable_buffers = [
                buffer for buffer in buffers if buffer in reusable_conflicts
            ]
            if not fixed_buffers and not reusable_buffers:
                continue

            allocation_idx = len(executable.allocations)
            allocation_size = 0
            allocation_alignment = 1
            for buffer in fixed_buffers:
                record = self._records[buffer]
                allocation_alignment = max(allocation_alignment, record.alignment_bytes)
                allocation_size = _align_up(allocation_size, record.alignment_bytes)
                locations[buffer] = BufferLocation(allocation_idx, allocation_size)
                allocation_size = _checked_add(allocation_size, record.size_bytes)

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
                record = self._records[buffer]
                external = record.external
                if external is None:
                    message = "internal error: fixed buffer is not external"
                    raise RuntimeError(message)
                location = locations[buffer]
                executable.buffers.add(
                    name=external.name,
                    allocation_idx=location.allocation_idx,
                    allocation_offset=location.allocation_offset,
                    data_type=external.dtype,
                    shape=external.shape,
                )
        return locations

    def _add_buffer(self, buffer: Buffer, record: _BufferRecord) -> None:
        """Register one newly created opaque handle and its private metadata."""
        self._buffers_by_allocation[record.allocation].append(buffer)
        self._records[buffer] = record

    def _require_owned_allocation(self, allocation: Allocation) -> None:
        """Reject allocations created by another executable builder."""
        if not allocation.belongs_to(self):
            message = "allocation does not belong to this executable builder"
            raise ValueError(message)

    def _build_reusable_ranges(
        self,
        buffers: list[Buffer],
        conflicts: dict[Buffer, set[Buffer]],
        allocation_size: int,
    ) -> tuple[int, int, dict[Buffer, int]]:
        """Pack internal raw ranges into reusable slots after external ranges."""
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
            allocation_size = _checked_add(allocation_size, slot.size_bytes)
        return allocation_size, allocation_alignment, offsets


def _validate_name(name: str) -> None:
    """Reject an empty runtime-visible external-buffer name."""
    if not name:
        message = "external buffer name cannot be empty"
        raise ValueError(message)


def _validate_shape(shape: tuple[int, ...]) -> None:
    """Reject external buffers with empty or non-positive dimensions."""
    if not shape or any(dimension <= 0 for dimension in shape):
        message = "external buffer shape must contain only positive dimensions"
        raise ValueError(message)


def _validate_size(size_bytes: int) -> None:
    """Reject raw allocation sizes outside the executable uint64 range."""
    if size_bytes <= 0 or size_bytes > _MAX_ALLOCATION_SIZE_BYTES:
        message = "buffer size must be a positive uint64 value"
        raise ValueError(message)


def _validate_alignment(alignment_bytes: int, *, minimum: int) -> None:
    """Validate one power-of-two alignment suitable for a raw memory range."""
    if (
        alignment_bytes < minimum
        or alignment_bytes > _MAX_ALLOCATION_SIZE_BYTES
        or alignment_bytes & (alignment_bytes - 1)
    ):
        message = f"alignment must be a power of two no smaller than {minimum}"
        raise ValueError(message)


def _checked_product(shape: tuple[int, ...], element_size: int) -> int:
    """Return a checked external-buffer byte count."""
    value = prod(shape)
    if value > _MAX_ALLOCATION_SIZE_BYTES // element_size:
        message = "buffer allocation size exceeds uint64"
        raise ValueError(message)
    return value * element_size


def _checked_add(first: int, second: int) -> int:
    """Add two allocation sizes without exceeding protobuf uint64 capacity."""
    if first > _MAX_ALLOCATION_SIZE_BYTES - second:
        message = "allocation size exceeds uint64"
        raise ValueError(message)
    return first + second


def _validate_matching_external(  # noqa: PLR0913
    record: _BufferRecord,
    *,
    allocation: Allocation,
    name: str,
    shape: tuple[int, ...],
    dtype: lc0ex_pb2.Buffer.DataType,
    writable: bool,
    alignment_bytes: int,
) -> None:
    """Ensure repeated declarations describe one identical external range."""
    external = record.external
    if record.allocation is not allocation:
        message = f"buffer {name!r} belongs to a different allocation"
        raise ValueError(message)
    if external is None:
        message = f"buffer {name!r} is not external"
        raise RuntimeError(message)
    if external.shape != shape:
        message = f"shape does not match existing buffer {name!r}"
        raise ValueError(message)
    if external.dtype != dtype:
        message = f"data type does not match existing buffer {name!r}"
        raise ValueError(message)
    if record.writable != writable:
        message = f"writability does not match existing buffer {name!r}"
        raise ValueError(message)
    if record.alignment_bytes != alignment_bytes:
        message = f"alignment does not match existing buffer {name!r}"
        raise ValueError(message)


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
