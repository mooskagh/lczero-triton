"""Tests for BT4 graph-construction helpers and entry points."""

from typing import cast

import net_pb2
import pytest
from lc0ex import ExecutableBuilder
from lczero_triton.bt4._format import NetworkFormatError
from lczero_triton.bt4.network import (
    _default_activation,
    _layer_elements,
    _resolve_activation,
    build,
)


def _target_skeleton() -> net_pb2.Net:
    """Create a minimal supported multihead network without large payloads."""
    network = net_pb2.Net()
    network.format.weights_encoding = net_pb2.Format.LINEAR16
    network.format.network_format.input = (
        net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    )
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    )
    network.format.network_format.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    network.format.network_format.value = net_pb2.NetworkFormat.VALUE_WDL
    network.format.network_format.moves_left = net_pb2.NetworkFormat.MOVES_LEFT_V1
    network.weights.headcount = 1
    network.weights.encoder.add()
    for field in ("ip2_pol_w", "ip3_pol_w", "ip4_pol_w"):
        getattr(network.weights.policy_heads.vanilla, field).params = b"\0\0"
    for field in ("ip_val_w", "ip1_val_w", "ip2_val_w"):
        getattr(network.weights.value_heads.winner, field).params = b"\0\0"
    return network


def test_build_entry_point_is_importable() -> None:
    """BT4 graph construction is exposed through the protobuf-driven API."""
    assert callable(build)


def test_build_rejects_non_positive_batch_before_network_validation() -> None:
    """Invalid batches fail before allocations or protobuf traversal."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build(ExecutableBuilder(), net_pb2.Net(), batch_size=0)


def test_build_normalizes_and_reaches_input_boundary() -> None:
    """A flexible valid skeleton reaches the first unimplemented production."""
    network = _target_skeleton()

    with pytest.raises(NotImplementedError, match="input and embedding"):
        build(ExecutableBuilder(), network, batch_size=1)

    assert (
        network.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    assert (
        network.format.network_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    )


def test_layer_elements_uses_layer_encoding_override() -> None:
    """A layer encoding supersedes the enclosing file encoding."""
    layer = net_pb2.Weights.Layer(
        params=b"\0\0",
        encoding=net_pb2.Weights.Layer.LINEAR16,
    )

    assert _layer_elements(layer, net_pb2.Format.UNKNOWN, path="weights.layer") == 1


def test_layer_elements_rejects_odd_payload() -> None:
    """LINEAR16 dimensions require whole encoded half values."""
    layer = net_pb2.Weights.Layer(params=b"\0")

    with pytest.raises(NetworkFormatError, match=r"weights\.layer: LINEAR16 payload"):
        _layer_elements(layer, net_pb2.Format.LINEAR16, path="weights.layer")


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        (
            net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU,
            net_pb2.NetworkFormat.ACTIVATION_RELU,
        ),
        (
            net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH,
            net_pb2.NetworkFormat.ACTIVATION_MISH,
        ),
    ],
)
def test_default_activation(default: int, expected: int) -> None:
    """LC0 default activation enums resolve to concrete operations."""
    resolved_default = cast("net_pb2.NetworkFormat.DefaultActivation", default)
    assert _default_activation(resolved_default) == expected


def test_explicit_activation_overrides_default() -> None:
    """An explicit operation activation is not replaced by the default."""
    assert (
        _resolve_activation(
            net_pb2.NetworkFormat.ACTIVATION_SWISH,
            net_pb2.NetworkFormat.ACTIVATION_MISH,
            path="format.network_format.smolgen_activation",
        )
        == net_pb2.NetworkFormat.ACTIVATION_SWISH
    )
