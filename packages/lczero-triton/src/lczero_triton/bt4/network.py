"""Protobuf-driven grammar for construction of the BT4 executable graph."""

from dataclasses import dataclass

import net_pb2
from lc0ex import Allocation, Buffer, ExecutableBuilder
from lc0ex.proto import lc0ex_pb2

from lczero_triton.bt4._format import (
    NetworkFormatError,
    normalize_network,
    validate_network_format,
)
from lczero_triton.bt4.kernels._cache import KernelCache

_F16_SIZE_BYTES = 2


@dataclass(slots=True)
class _BuildContext:
    """Construction services shared by grammar productions in one build."""

    builder: ExecutableBuilder
    persistent: Allocation
    execution: Allocation
    kernels: KernelCache
    batch_size: int
    default_encoding: net_pb2.Format.Encoding


def build(
    builder: ExecutableBuilder,
    network: net_pb2.Net,
    *,
    batch_size: int = 169,
) -> None:
    """Traverse the active network and append its executable graph."""
    if batch_size <= 0:
        message = "batch_size must be positive"
        raise ValueError(message)

    normalize_network(network)
    validate_network_format(network)
    context = _BuildContext(
        builder=builder,
        persistent=builder.allocation(lc0ex_pb2.Allocation.LIFETIME_PERSISTENT),
        execution=builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION),
        kernels=KernelCache(builder),
        batch_size=batch_size,
        default_encoding=network.format.weights_encoding,
    )
    _network(context, network.weights)


def _network(context: _BuildContext, weights: net_pb2.Weights) -> None:
    """Build inputs, body embedding, encoders, and selected output heads."""
    inputs = _inputs(context)
    body, body_width = _embedding(context, inputs, weights)
    body = _encoder_tower(context, body, body_width, weights)
    _policy_head(context, body, body_width, weights)
    _value_head(context, body, body_width, weights.value_heads.winner)
    _moves_left_head(context, body, body_width, weights)


def _inputs(context: _BuildContext) -> tuple[Buffer, Buffer]:
    """Create packed plane-mask and plane-value inputs once their kernels exist."""
    del context
    message = "BT4 input and embedding production awaits its kernel families"
    raise NotImplementedError(message)


def _embedding(
    context: _BuildContext,
    inputs: tuple[Buffer, Buffer],
    weights: net_pb2.Weights,
) -> tuple[Buffer, int]:
    """Build dense positional preprocessing, gated embedding, and embedding FFN."""
    del context, inputs, weights
    message = "BT4 embedding production awaits its kernel families"
    raise NotImplementedError(message)


def _encoder_tower(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> Buffer:
    """Visit each protobuf encoder in evaluation order without fixed depth."""
    for index, encoder in enumerate(weights.encoder):
        body = _encoder(
            context,
            body,
            body_width,
            encoder,
            prefix=f"/encoder{index}",
            head_count=weights.headcount,
            shared_smolgen=weights.smolgen_w,
        )
    return body


def _encoder(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
    shared_smolgen: net_pb2.Weights.Layer,
) -> Buffer:
    """Build one Smolgen attention and FFN encoder residual block."""
    smolgen = _smolgen(context, body, encoder, prefix=prefix, head_count=head_count)
    attended = _attention(
        context,
        body,
        smolgen,
        body_width,
        encoder,
        prefix=prefix,
        head_count=head_count,
        shared_smolgen=shared_smolgen,
    )
    return _ffn(context, attended, body_width, encoder, prefix=prefix)


def _smolgen(
    context: _BuildContext,
    body: Buffer,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
) -> Buffer:
    """Build local Smolgen compression and its two normalized dense layers."""
    del context, body, encoder, prefix, head_count
    message = "BT4 Smolgen production awaits its kernel families"
    raise NotImplementedError(message)


def _attention(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    smolgen: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
    shared_smolgen: net_pb2.Weights.Layer,
) -> Buffer:
    """Build shared Smolgen projection and the encoder Q/K/V attention path."""
    del context, body, smolgen, body_width, encoder, prefix, head_count, shared_smolgen
    message = "BT4 attention production awaits its kernel families"
    raise NotImplementedError(message)


def _ffn(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
) -> Buffer:
    """Build an encoder FFN and its DeepNorm residual layer normalization."""
    del context, body, body_width, encoder, prefix
    message = "BT4 FFN production awaits its kernel families"
    raise NotImplementedError(message)


def _policy_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> None:
    """Build the selected vanilla attention-policy branch."""
    del context, body, body_width, weights
    message = "BT4 policy production awaits its kernel families"
    raise NotImplementedError(message)


def _value_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    winner: net_pb2.Weights.ValueHead,
) -> None:
    """Build the selected winner WDL branch."""
    del context, body, body_width, winner
    message = "BT4 value production awaits its kernel families"
    raise NotImplementedError(message)


def _moves_left_head(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> None:
    """Build the selected moves-left branch."""
    del context, body, body_width, weights
    message = "BT4 moves-left production awaits its kernel families"
    raise NotImplementedError(message)


def _default_activation(
    value: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    """Map an LC0 default-activation enum to its concrete activation enum."""
    activations = {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: (
            net_pb2.NetworkFormat.ACTIVATION_RELU
        ),
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: (
            net_pb2.NetworkFormat.ACTIVATION_MISH
        ),
    }
    try:
        return activations[value]
    except KeyError as error:
        message = f"format.network_format.default_activation: unsupported enum {value}"
        raise NetworkFormatError(message) from error


def _resolve_activation(
    value: net_pb2.NetworkFormat.ActivationFunction,
    default: net_pb2.NetworkFormat.ActivationFunction,
    *,
    path: str,
) -> net_pb2.NetworkFormat.ActivationFunction:
    """Resolve an explicit/default activation and reject unsupported variants."""
    resolved = default if value == net_pb2.NetworkFormat.ACTIVATION_DEFAULT else value
    supported = {
        net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.ACTIVATION_MISH,
        net_pb2.NetworkFormat.ACTIVATION_SELU,
        net_pb2.NetworkFormat.ACTIVATION_SWISH,
        net_pb2.NetworkFormat.ACTIVATION_RELU_2,
        net_pb2.NetworkFormat.ACTIVATION_NONE,
    }
    if resolved not in supported:
        message = f"{path}: unsupported activation enum {resolved}"
        raise NetworkFormatError(message)
    return resolved


def _layer_elements(
    layer: net_pb2.Weights.Layer,
    default_encoding: net_pb2.Format.Encoding,
    *,
    path: str,
) -> int:
    """Return LINEAR16 element count while retaining layer payloads opaque."""
    encoding = layer.encoding if layer.HasField("encoding") else default_encoding
    if encoding != net_pb2.Format.LINEAR16:
        message = f"{path}: expected LINEAR16 encoding, got {encoding}"
        raise NetworkFormatError(message)
    if len(layer.params) % _F16_SIZE_BYTES:
        message = f"{path}: LINEAR16 payload has an odd byte count"
        raise NetworkFormatError(message)
    if not layer.params:
        message = f"{path}: missing LINEAR16 payload"
        raise NetworkFormatError(message)
    return len(layer.params) // _F16_SIZE_BYTES


def _temporary_f16(
    context: _BuildContext,
    *,
    element_count: int,
    alignment_bytes: int = 256,
) -> Buffer:
    """Allocate one opaque FP16 temporary by its raw byte extent."""
    if element_count <= 0:
        message = "temporary element count must be positive"
        raise ValueError(message)
    return context.execution.temporary_buffer(
        size_bytes=element_count * _F16_SIZE_BYTES,
        alignment_bytes=alignment_bytes,
    )
