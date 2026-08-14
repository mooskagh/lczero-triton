"""Tests for BT4 protobuf loading, normalization, and validation."""

import gzip
from pathlib import Path

import pytest
from lc0ex.proto import net_pb2
from lczero_triton.bt4._format import (
    NetworkFormatError,
    load_network,
    normalize_network,
    validate_network_format,
)


def _network() -> net_pb2.Net:
    """Create a minimal format-valid BT4 protobuf."""
    network = net_pb2.Net(magic=0x1C0)
    network.format.network_format.input = (
        net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    )
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    network.format.network_format.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    network.format.network_format.value = net_pb2.NetworkFormat.VALUE_WDL
    network.format.network_format.moves_left = net_pb2.NetworkFormat.MOVES_LEFT_V1
    network.format.network_format.input_embedding = (
        net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    )
    network.weights.headcount = 1
    network.weights.encoder.add()
    for field in (
        "ip2_pol_w",
        "ip2_pol_b",
        "ip3_pol_w",
        "ip3_pol_b",
        "ip4_pol_w",
    ):
        getattr(network.weights.policy_heads.vanilla, field).params = b"\0\0"
    for field in (
        "ip_val_w",
        "ip_val_b",
        "ip1_val_w",
        "ip1_val_b",
        "ip2_val_w",
        "ip2_val_b",
    ):
        getattr(network.weights.value_heads.winner, field).params = b"\0\0"
    for field in (
        "ip_mov_w",
        "ip_mov_b",
        "ip1_mov_w",
        "ip1_mov_b",
        "ip2_mov_w",
        "ip2_mov_b",
    ):
        getattr(network.weights, field).params = b"\0\0"
    return network


def test_normalize_target_like_multihead_network_is_idempotent() -> None:
    """Older attention-body multihead containers select dense PE."""
    network = _network()
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    )
    network.format.network_format.ClearField("input_embedding")

    normalize_network(network)
    normalized = network.SerializeToString()
    normalize_network(network)

    assert (
        network.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    assert (
        network.format.network_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    )
    assert network.SerializeToString() == normalized


def test_normalize_legacy_se_smolgen_network() -> None:
    """Old SE encoders receive LC0's attention and activation upgrades."""
    network = net_pb2.Net()
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
    )
    network.weights.encoder.add()
    network.weights.smolgen_w.params = b"\0\0"

    normalize_network(network)

    assert (
        network.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    )
    assert (
        network.format.network_format.ffn_activation
        == net_pb2.NetworkFormat.ACTIVATION_RELU_2
    )
    assert (
        network.format.network_format.smolgen_activation
        == net_pb2.NetworkFormat.ACTIVATION_SWISH
    )


def test_load_network_normalizes_gzip_protobuf(tmp_path: Path) -> None:
    """Loading checks magic and returns the LC0-normalized protobuf."""
    network = _network()
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    )
    path = tmp_path / "network.pb.gz"
    with gzip.open(path, "wb") as destination:
        destination.write(network.SerializeToString())

    loaded = load_network(path)

    assert (
        loaded.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )


@pytest.mark.parametrize("magic", [0, 123])
def test_load_network_rejects_bad_magic(tmp_path: Path, magic: int) -> None:
    """The weights-file magic is checked after protobuf parsing."""
    path = tmp_path / "network.pb.gz"
    with gzip.open(path, "wb") as destination:
        destination.write(net_pb2.Net(magic=magic).SerializeToString())

    with pytest.raises(NetworkFormatError, match="bad magic"):
        load_network(path)


def test_validate_network_rejects_rpe() -> None:
    """Relative-position fields are outside the initial BT4 grammar."""
    network = _network()
    network.weights.encoder[0].mha.rpe_q.params = b"\0\0"

    with pytest.raises(NetworkFormatError, match=r"weights.encoder\[0\].mha.rpe_q"):
        validate_network_format(network)


def test_validate_network_rejects_absent_winner_computation() -> None:
    """A selected WDL container alone is not an executable active head."""
    network = _network()
    network.weights.value_heads.winner.ClearField("ip2_val_w")

    with pytest.raises(NetworkFormatError, match="selected value computation"):
        validate_network_format(network)


def test_validate_network_rejects_policy_encoders() -> None:
    """Policy-specific encoder towers are outside the selected graph."""
    network = _network()
    network.weights.policy_heads.vanilla.pol_encoder.add()

    with pytest.raises(NetworkFormatError, match="pol_encoder"):
        validate_network_format(network)


def test_validate_network_rejects_absent_moves_left_computation() -> None:
    """The selected moves-left format requires all three dense operations."""
    network = _network()
    network.weights.ClearField("ip2_mov_b")

    with pytest.raises(NetworkFormatError, match="moves-left computation"):
        validate_network_format(network)
