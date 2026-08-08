"""Persistent buffer handles and executable serialization support."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from lc0ex.proto import lc0ex_pb2

_PERSISTENT_ALLOCATION_NAME = "persistent"


@dataclass(frozen=True, slots=True)
class Buffer:
    """A logical buffer that can be passed to other executable builders."""

    name: str
    shape: tuple[int, ...]
    dtype: lc0ex_pb2.Buffer.DataType


def data_type_size_bytes(dtype: lc0ex_pb2.Buffer.DataType) -> int:
    """Return the size in bytes of one value of *dtype*.

    Raises:
        ValueError: If *dtype* is not a supported concrete data type.

    """
    sizes = {
        lc0ex_pb2.Buffer.DATA_TYPE_F32: 4,
        lc0ex_pb2.Buffer.DATA_TYPE_U8: 1,
        lc0ex_pb2.Buffer.DATA_TYPE_F16: 2,
        lc0ex_pb2.Buffer.DATA_TYPE_U64: 8,
        lc0ex_pb2.Buffer.DATA_TYPE_BF16: 2,
    }
    try:
        return sizes[dtype]
    except KeyError as error:
        message = f"unsupported buffer data type: {dtype}"
        raise ValueError(message) from error


class PersistentBufferBuilder:
    """Collect logical buffers into one persistent allocation."""

    def __init__(self) -> None:
        """Initialize an empty persistent buffer collection."""
        self._buffers: dict[str, Buffer] = {}

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
        if any(dimension < 0 for dimension in normalized_shape):
            message = "buffer dimensions cannot be negative"
            raise ValueError(message)
        data_type_size_bytes(dtype)

        result = Buffer(name=name, shape=normalized_shape, dtype=dtype)
        self._buffers[name] = result
        return result

    def build(self, executable: lc0ex_pb2.NeuralExecutable) -> None:
        """Append the collected buffers and their allocation to *executable*."""
        if not self._buffers:
            return

        offsets: dict[str, int] = {}
        allocation_size = 0
        allocation_alignment = 1
        for buffer in self._buffers.values():
            alignment = data_type_size_bytes(buffer.dtype)
            allocation_alignment = max(allocation_alignment, alignment)
            allocation_size = _align_up(allocation_size, alignment)
            offsets[buffer.name] = allocation_size
            allocation_size += prod(buffer.shape) * alignment

        executable.allocations.add(
            name=_PERSISTENT_ALLOCATION_NAME,
            size_bytes=allocation_size,
            alignment_bytes=allocation_alignment,
            lifetime=lc0ex_pb2.Allocation.LIFETIME_PERSISTENT,
        )
        for buffer in self._buffers.values():
            serialized_buffer = executable.buffers.add(
                name=buffer.name,
                data_type=buffer.dtype,
                shape=buffer.shape,
            )
            serialized_buffer.allocation_block.allocation = _PERSISTENT_ALLOCATION_NAME
            serialized_buffer.allocation_block.offset_bytes = offsets[buffer.name]


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
