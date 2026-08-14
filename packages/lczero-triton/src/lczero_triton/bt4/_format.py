"""Loading, normalization, and format validation for BT4 network files."""

import gzip
from pathlib import Path

from google.protobuf.message import DecodeError
from lc0ex.proto import net_pb2

_WEIGHT_MAGIC = 0x1C0


class NetworkFormatError(ValueError):
    """The source network cannot be traversed by the BT4 graph builder."""


def load_network(path: Path) -> net_pb2.Net:
    """Read, parse, normalize, and return one gzip-compressed Lc0 network."""
    try:
        with gzip.open(path, "rb") as source:
            encoded_network = source.read()
    except OSError as error:
        message = f"invalid weights file {path}: {error}"
        raise NetworkFormatError(message) from error

    network = net_pb2.Net()
    try:
        network.ParseFromString(encoded_network)
    except DecodeError as error:
        message = f"invalid weights file {path}: malformed protobuf"
        raise NetworkFormatError(message) from error

    if network.magic != _WEIGHT_MAGIC:
        message = f"invalid weights file {path}: bad magic {network.magic:#x}"
        raise NetworkFormatError(message)
    if not network.HasField("weights"):
        message = f"invalid weights file {path}: missing weights"
        raise NetworkFormatError(message)

    normalize_network(network)
    return network


def normalize_network(network: net_pb2.Net) -> None:
    """Apply LC0's older-format upgrades in place."""
    format_message = network.format
    if not format_message.HasField("network_format"):
        network_format = format_message.network_format
        network_format.input = net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        network_format.output = net_pb2.NetworkFormat.OUTPUT_CLASSICAL
        network_format.network = net_pb2.NetworkFormat.NETWORK_CLASSICAL_WITH_HEADFORMAT
        network_format.value = net_pb2.NetworkFormat.VALUE_CLASSICAL
        network_format.policy = net_pb2.NetworkFormat.POLICY_CLASSICAL

    network_format = format_message.network_format
    if network_format.network == net_pb2.NetworkFormat.NETWORK_CLASSICAL:
        network_format.network = net_pb2.NetworkFormat.NETWORK_CLASSICAL_WITH_HEADFORMAT
        network_format.value = net_pb2.NetworkFormat.VALUE_CLASSICAL
        network_format.policy = net_pb2.NetworkFormat.POLICY_CLASSICAL
    elif network_format.network == net_pb2.NetworkFormat.NETWORK_SE:
        network_format.network = net_pb2.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
        network_format.value = net_pb2.NetworkFormat.VALUE_CLASSICAL
        network_format.policy = net_pb2.NetworkFormat.POLICY_CLASSICAL
    elif (
        network_format.network == net_pb2.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
        and network.HasField("weights")
        and network.weights.encoder
    ):
        network_format.network = (
            net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
        )
        if network.weights.HasField("smolgen_w"):
            network_format.ffn_activation = net_pb2.NetworkFormat.ACTIVATION_RELU_2
            network_format.smolgen_activation = net_pb2.NetworkFormat.ACTIVATION_SWISH
    elif (
        network_format.network
        == net_pb2.NetworkFormat.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT
    ):
        network_format.network = (
            net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
        )

    if (
        network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    ):
        if (
            network.HasField("weights")
            and network.weights.HasField("policy_heads")
            and network.weights.HasField("value_heads")
        ):
            network_format.network = (
                net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
            )
            network_format.input_embedding = (
                net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
            )
        elif not network_format.HasField("input_embedding"):
            network_format.input_embedding = (
                net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_MAP
            )


def validate_network_format(network: net_pb2.Net) -> None:
    """Reject a materially unsupported active graph format."""
    if not network.HasField("weights"):
        message = "weights: missing weights"
        raise NetworkFormatError(message)

    network_format = network.format.network_format
    _validate_format_enums(network_format)

    weights = network.weights
    _validate_active_heads(weights)
    _validate_encoders(weights)


def _validate_format_enums(network_format: net_pb2.NetworkFormat) -> None:
    """Require the format enums implemented by the initial BT4 path."""
    _require_enum(
        network_format.input,
        net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE,
        "format.network_format.input",
        "INPUT_CLASSICAL_112_PLANE",
    )
    _require_enum(
        network_format.network,
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT,
        "format.network_format.network",
        "NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT",
    )
    _require_enum(
        network_format.policy,
        net_pb2.NetworkFormat.POLICY_ATTENTION,
        "format.network_format.policy",
        "POLICY_ATTENTION",
    )
    _require_enum(
        network_format.value,
        net_pb2.NetworkFormat.VALUE_WDL,
        "format.network_format.value",
        "VALUE_WDL",
    )
    _require_enum(
        network_format.moves_left,
        net_pb2.NetworkFormat.MOVES_LEFT_V1,
        "format.network_format.moves_left",
        "MOVES_LEFT_V1",
    )
    _require_enum(
        network_format.input_embedding,
        net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE,
        "format.network_format.input_embedding",
        "INPUT_EMBEDDING_PE_DENSE",
    )


def _validate_active_heads(weights: net_pb2.Weights) -> None:
    """Require the selected multihead structures and their active projections."""
    if not weights.encoder:
        message = "weights.encoder: expected at least one body encoder"
        raise NetworkFormatError(message)
    if weights.headcount <= 0:
        message = "weights.headcount: expected a positive attention head count"
        raise NetworkFormatError(message)
    if not weights.HasField("policy_heads") or not weights.policy_heads.HasField(
        "vanilla"
    ):
        message = "weights.policy_heads.vanilla: selected policy head is missing"
        raise NetworkFormatError(message)
    if not (weights.HasField("value_heads") and weights.value_heads.HasField("winner")):
        message = "weights.value_heads.winner: selected value head is missing"
        raise NetworkFormatError(message)
    if not _has_policy_computation(weights.policy_heads.vanilla):
        message = "weights.policy_heads.vanilla: selected policy computation is missing"
        raise NetworkFormatError(message)
    if weights.policy_heads.vanilla.pol_encoder:
        message = "weights.policy_heads.vanilla.pol_encoder: not supported"
        raise NetworkFormatError(message)
    if not _has_value_computation(weights.value_heads.winner):
        message = "weights.value_heads.winner: selected value computation is missing"
        raise NetworkFormatError(message)
    if not _has_moves_left_computation(weights):
        message = "weights: selected moves-left computation is missing"
        raise NetworkFormatError(message)


def _validate_encoders(weights: net_pb2.Weights) -> None:
    """Reject RPE until the active encoder grammar supports it."""
    for index, encoder in enumerate(weights.encoder):
        if not encoder.HasField("mha"):
            continue
        for field in ("rpe_q", "rpe_k", "rpe_v"):
            if encoder.mha.HasField(field):
                message = (
                    f"weights.encoder[{index}].mha.{field}: relative position "
                    "embeddings are not supported"
                )
                raise NetworkFormatError(message)


def _require_enum(
    actual: int,
    expected: int,
    path: str,
    expected_name: str,
) -> None:
    """Require one format enum while retaining the offending numeric value."""
    if actual != expected:
        message = f"{path}: expected {expected_name}, got {actual}"
        raise NetworkFormatError(message)


def _has_layer_data(layer: net_pb2.Weights.Layer) -> bool:
    """Return whether a layer has encoded values without decoding them."""
    return bool(layer.params)


def _has_policy_computation(policy: net_pb2.Weights.PolicyHead) -> bool:
    """Return whether the selected policy head has its active projections."""
    return all(
        policy.HasField(field) and _has_layer_data(getattr(policy, field))
        for field in (
            "ip2_pol_w",
            "ip2_pol_b",
            "ip3_pol_w",
            "ip3_pol_b",
            "ip4_pol_w",
        )
    )


def _has_value_computation(value: net_pb2.Weights.ValueHead) -> bool:
    """Return whether the selected WDL head has its active projections."""
    return all(
        value.HasField(field) and _has_layer_data(getattr(value, field))
        for field in (
            "ip_val_w",
            "ip_val_b",
            "ip1_val_w",
            "ip1_val_b",
            "ip2_val_w",
            "ip2_val_b",
        )
    )


def _has_moves_left_computation(weights: net_pb2.Weights) -> bool:
    """Return whether the selected moves-left head has all dense operands."""
    return all(
        weights.HasField(field) and _has_layer_data(getattr(weights, field))
        for field in (
            "ip_mov_w",
            "ip_mov_b",
            "ip1_mov_w",
            "ip1_mov_b",
            "ip2_mov_w",
            "ip2_mov_b",
        )
    )
