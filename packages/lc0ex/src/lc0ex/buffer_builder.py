"""Logical buffer handles and executable allocation serialization support."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from lc0ex.proto import lc0ex_pb2

_PERSISTENT_ALLOCATION_NAME = "persistent"
_EXECUTION_ALLOCATION_NAME = "execution"


@dataclass(frozen=True, slots=True)
class Buffer:
    """A logical buffer that can be passed to other executable builders."""

    name: str
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


class BufferBuilder:
    """Collect persistent and temporary logical buffers."""

    def __init__(self) -> None:
        """Initialize an empty buffer collection."""
        self._buffers: dict[str, Buffer] = {}
        self._temporary_buffers: set[Buffer] = set()
        self._next_temporary_buffer = 0

    def buffer(
        self,
        name: str,
        shape: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
    ) -> Buffer:
        """Create or retrieve a persistent buffer named *name*.

        A new buffer requires both *shape* and *dtype*. When retrieving an
        existing buffer, either may be omitted; supplied values must match the
        original definition.
        """
        existing = self._buffers.get(name)
        normalized_shape = tuple(shape) if shape is not None else None

        if existing is not None:
            if normalized_shape is not None and normalized_shape != existing.shape:
                message = f"shape does not match existing buffer {name!r}"
                raise ValueError(message)
            if dtype is not None and dtype != existing.dtype:
                message = f"data type does not match existing buffer {name!r}"
                raise ValueError(message)
            return existing

        if normalized_shape is None or dtype is None:
            message = f"shape and data type are required for new buffer {name!r}"
            raise ValueError(message)
        result = Buffer(name=name, shape=normalized_shape, dtype=dtype)
        self._buffers[name] = result
        return result

    def tmp_buffer(
        self,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
    ) -> Buffer:
        """Create a temporary buffer with an automatically generated name."""
        while True:
            name = f"tmp_{self._next_temporary_buffer}"
            self._next_temporary_buffer += 1
            if name not in self._buffers:
                break

        result = Buffer(name=name, shape=tuple(shape), dtype=dtype)
        self._buffers[name] = result
        self._temporary_buffers.add(result)
        return result

    def owns(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is a handle created by this collection."""
        return self._buffers.get(buffer.name) is buffer

    def is_temporary(self, buffer: Buffer) -> bool:
        """Return whether *buffer* has execution lifetime."""
        return buffer in self._temporary_buffers

    def temporary_buffers(self) -> tuple[Buffer, ...]:
        """Return every temporary buffer in declaration order."""
        return tuple(
            buffer
            for buffer in self._buffers.values()
            if buffer in self._temporary_buffers
        )

    def build(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        temporary_conflicts: dict[Buffer, set[Buffer]],
    ) -> None:
        """Append collected buffers and their planned allocations to *executable*."""
        persistent_buffers = [
            buffer
            for buffer in self._buffers.values()
            if buffer not in self._temporary_buffers
        ]
        temporary_buffers = [
            buffer
            for buffer in self._buffers.values()
            if buffer in self._temporary_buffers
        ]
        offsets: dict[Buffer, int] = {}
        self._build_persistent_allocation(executable, persistent_buffers, offsets)
        self._build_execution_allocation(
            executable,
            temporary_buffers,
            temporary_conflicts,
            offsets,
        )

        for buffer in self._buffers.values():
            serialized_buffer = executable.buffers.add(
                name=buffer.name,
                data_type=buffer.dtype,
                shape=buffer.shape,
            )
            serialized_buffer.allocation_block.allocation = (
                _EXECUTION_ALLOCATION_NAME
                if buffer in self._temporary_buffers
                else _PERSISTENT_ALLOCATION_NAME
            )
            serialized_buffer.allocation_block.offset_bytes = offsets[buffer]

    def _build_persistent_allocation(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        buffers: list[Buffer],
        offsets: dict[Buffer, int],
    ) -> None:
        """Allocate persistent buffers in declaration order."""
        if not buffers:
            return

        allocation_size = 0
        allocation_alignment = 1
        for buffer in buffers:
            alignment = data_type_size_bytes(buffer.dtype)
            allocation_alignment = max(allocation_alignment, alignment)
            allocation_size = _align_up(allocation_size, alignment)
            offsets[buffer] = allocation_size
            allocation_size += prod(buffer.shape) * alignment

        executable.allocations.add(
            name=_PERSISTENT_ALLOCATION_NAME,
            size_bytes=allocation_size,
            alignment_bytes=allocation_alignment,
            lifetime=lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
        )

    def _build_execution_allocation(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
        buffers: list[Buffer],
        conflicts: dict[Buffer, set[Buffer]],
        offsets: dict[Buffer, int],
    ) -> None:
        """Allocate non-overlapping temporary lifetimes into reusable slots."""
        if not buffers:
            return

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

        allocation_size = 0
        allocation_alignment = 1
        for slot in slots:
            allocation_alignment = max(allocation_alignment, slot.alignment_bytes)
            allocation_size = _align_up(allocation_size, slot.alignment_bytes)
            for buffer in slot.buffers:
                offsets[buffer] = allocation_size
            allocation_size += slot.size_bytes

        executable.allocations.add(
            name=_EXECUTION_ALLOCATION_NAME,
            size_bytes=allocation_size,
            alignment_bytes=allocation_alignment,
            lifetime=lc0ex_pb2.Allocation.LIFETIME_EXECUTION,
        )


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
