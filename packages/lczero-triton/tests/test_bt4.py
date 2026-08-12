"""Tests for BT4 graph construction and its input-embedding segment."""

from collections.abc import Callable
from typing import cast

import lczero_triton.bt4.kernels.add_bias_batched as add_bias_batched_module
import lczero_triton.bt4.kernels.add_vectors as add_vectors_module
import lczero_triton.bt4.kernels.batched_matmul as batched_matmul_module
import lczero_triton.bt4.kernels.expand_planes as expand_planes_module
import lczero_triton.bt4.kernels.input_gating as input_gating_module
import lczero_triton.bt4.kernels.layer_norm as layer_norm_module
import lczero_triton.bt4.kernels.matmul as matmul_module
import lczero_triton.bt4.kernels.nchw_to_nhwc as nchw_to_nhwc_module
import lczero_triton.bt4.kernels.preprocess_attention_body as preprocess_module
import lczero_triton.bt4.kernels.softmax_64 as softmax_64_module
import lczero_triton.bt4.network as network_module
import net_pb2
import pytest
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4._format import NetworkFormatError
from lczero_triton.bt4.kernels.add_bias_batched import (
    AddBiasBatchedSpecialization,
)
from lczero_triton.bt4.kernels.add_vectors import AddVectorsSpecialization
from lczero_triton.bt4.kernels.batched_matmul import BatchedMatmulSpecialization
from lczero_triton.bt4.kernels.expand_planes import ExpandPlanesSpecialization
from lczero_triton.bt4.kernels.input_gating import InputGatingSpecialization
from lczero_triton.bt4.kernels.layer_norm import LayerNormSpecialization
from lczero_triton.bt4.kernels.matmul import MatmulSpecialization
from lczero_triton.bt4.kernels.nchw_to_nhwc import NchwToNhwcSpecialization
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
)
from lczero_triton.bt4.kernels.softmax_64 import Softmax64Specialization
from lczero_triton.bt4.network import (
    _BuildContext,
    _default_activation,
    _layer_elements,
    _resolve_activation,
    build,
)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_ARCHITECTURE = 80
_REUSED_TEMPORARY_COUNT = 3
_ENCODER_LOCAL_BUFFER_COUNT = 28


def _active_architecture() -> int:
    """Return a stable target for compiler-free structural tests."""
    return _ARCHITECTURE


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
    network.format.network_format.default_activation = (
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH
    )
    network.weights.headcount = 1
    network.weights.encoder.add()
    for field in ("ip2_pol_w", "ip3_pol_w", "ip4_pol_w"):
        getattr(network.weights.policy_heads.vanilla, field).params = b"\0\0"
    for field in ("ip_val_w", "ip1_val_w", "ip2_val_w"):
        getattr(network.weights.value_heads.winner, field).params = b"\0\0"
    return network


def _set_elements(layer: net_pb2.Weights.Layer, element_count: int) -> None:
    """Populate one synthetic LINEAR16 layer with the requested element count."""
    layer.params = b"\0\0" * element_count


def _embedding_network(
    *,
    encoding_width: int,
    body_width: int,
    hidden_width: int,
) -> net_pb2.Net:
    """Create a compact format-valid network with inferred embedding widths."""
    network = _target_skeleton()
    weights = network.weights
    position_output_width = 64 * encoding_width
    embedding_input_width = 112 + encoding_width

    _set_elements(weights.ip_emb_preproc_w, 64 * 12 * position_output_width)
    _set_elements(weights.ip_emb_preproc_b, position_output_width)
    _set_elements(weights.ip_emb_w, embedding_input_width * body_width)
    _set_elements(weights.ip_emb_b, body_width)
    _set_elements(weights.ip_emb_ln_gammas, body_width)
    _set_elements(weights.ip_emb_ln_betas, body_width)
    _set_elements(weights.ip_mult_gate, 64 * body_width)
    _set_elements(weights.ip_add_gate, 64 * body_width)
    _set_elements(weights.ip_emb_ffn.dense1_w, body_width * hidden_width)
    _set_elements(weights.ip_emb_ffn.dense1_b, hidden_width)
    _set_elements(weights.ip_emb_ffn.dense2_w, hidden_width * body_width)
    _set_elements(weights.ip_emb_ffn.dense2_b, body_width)
    _set_elements(weights.ip_emb_ffn_ln_gammas, body_width)
    _set_elements(weights.ip_emb_ffn_ln_betas, body_width)
    return network


def _add_encoder(  # noqa: PLR0913
    network: net_pb2.Net,
    *,
    body_width: int,
    compression_width: int,
    hidden_width: int,
    generated_width: int,
    model_width: int,
    ffn_width: int,
) -> None:
    """Populate the skeleton encoder with locally inferred dimensions."""
    encoder = network.weights.encoder[0]
    heads = network.weights.headcount
    smolgen = encoder.mha.smolgen
    _set_elements(smolgen.compress, body_width * compression_width)
    _set_elements(smolgen.dense1_w, 64 * compression_width * hidden_width)
    _set_elements(smolgen.dense1_b, hidden_width)
    _set_elements(smolgen.ln1_gammas, hidden_width)
    _set_elements(smolgen.ln1_betas, hidden_width)
    _set_elements(smolgen.dense2_w, hidden_width * heads * generated_width)
    _set_elements(smolgen.dense2_b, heads * generated_width)
    _set_elements(smolgen.ln2_gammas, heads * generated_width)
    _set_elements(smolgen.ln2_betas, heads * generated_width)
    for weight, bias in (
        (encoder.mha.q_w, encoder.mha.q_b),
        (encoder.mha.k_w, encoder.mha.k_b),
        (encoder.mha.v_w, encoder.mha.v_b),
    ):
        _set_elements(weight, body_width * model_width)
        _set_elements(bias, model_width)
    _set_elements(encoder.mha.dense_w, model_width * body_width)
    _set_elements(encoder.mha.dense_b, body_width)
    _set_elements(encoder.ln1_gammas, body_width)
    _set_elements(encoder.ln1_betas, body_width)
    _set_elements(encoder.ffn.dense1_w, body_width * ffn_width)
    _set_elements(encoder.ffn.dense1_b, ffn_width)
    _set_elements(encoder.ffn.dense2_w, ffn_width * body_width)
    _set_elements(encoder.ffn.dense2_b, body_width)
    _set_elements(encoder.ln2_gammas, body_width)
    _set_elements(encoder.ln2_betas, body_width)
    _set_elements(network.weights.smolgen_w, generated_width * 64 * 64)
    network.format.network_format.smolgen_activation = (
        net_pb2.NetworkFormat.ACTIVATION_SWISH
    )


def _artifact(function: str, parameter_count: int) -> KernelArtifact:
    """Return a CUDA-shaped artifact without compiling or launching a kernel."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=b"stub",
        function=function,
        parameters=(_POINTER,) * parameter_count,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        dynamic_shared_memory_bytes=0,
    )


def _compiler(
    records: list[tuple[str, object]],
    function: str,
    parameter_count: int,
) -> Callable[[object], KernelArtifact]:
    """Create a compiler stub that records its immutable specialization."""

    def compile_stub(specialization: object) -> KernelArtifact:
        records.append((function, specialization))
        return _artifact(function, parameter_count)

    return compile_stub


def _stub_compilers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, object]]:
    """Replace item-11 compilers while retaining the real graph wrappers."""
    records: list[tuple[str, object]] = []
    monkeypatch.setattr(
        expand_planes_module,
        "compile_expand_planes",
        _compiler(records, "expand_planes", 3),
    )
    monkeypatch.setattr(
        nchw_to_nhwc_module,
        "compile_nchw_to_nhwc",
        _compiler(records, "nchw_to_nhwc", 2),
    )
    monkeypatch.setattr(
        matmul_module,
        "compile_matmul",
        _compiler(records, "matmul", 3),
    )
    monkeypatch.setattr(
        add_vectors_module,
        "compile_add_vectors",
        _compiler(records, "add_vectors", 3),
    )
    monkeypatch.setattr(
        preprocess_module,
        "compile_preprocess_attention_body",
        _compiler(records, "preprocess_attention_body", 3),
    )
    monkeypatch.setattr(
        input_gating_module,
        "compile_input_gating",
        _compiler(records, "input_gating", 4),
    )
    monkeypatch.setattr(
        add_bias_batched_module,
        "compile_add_bias_batched",
        _compiler(records, "add_bias_batched", 3),
    )

    def compile_batched_matmul(
        specialization: BatchedMatmulSpecialization,
    ) -> KernelArtifact:
        records.append(("batched_matmul", specialization))
        parameter_count = 3 if specialization.operation == "body_attention_v" else 4
        return _artifact(specialization.operation, parameter_count)

    monkeypatch.setattr(
        batched_matmul_module,
        "compile_batched_matmul",
        compile_batched_matmul,
    )
    monkeypatch.setattr(
        softmax_64_module,
        "compile_softmax_64",
        _compiler(records, "softmax_64", 3),
    )

    def compile_layer_norm(specialization: LayerNormSpecialization) -> KernelArtifact:
        function = "layer_norm_skip" if specialization.has_skip else "layer_norm"
        records.append((function, specialization))
        return _artifact(function, 7 if specialization.has_skip else 5)

    monkeypatch.setattr(
        layer_norm_module,
        "compile_layer_norm",
        compile_layer_norm,
    )
    monkeypatch.setattr(network_module, "active_architecture", _active_architecture)
    return records


def _stop_after_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace later grammar productions with one read of the returned body."""

    def consume_body(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> Buffer:
        del body_width, weights
        kernel = context.builder.add_kernel(_artifact("consume_body", 1))
        context.builder.call(kernel, body, readonly=(body,))
        return body

    def ignore_policy(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    def ignore_value(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        winner: net_pb2.Weights.ValueHead,
    ) -> None:
        del context, body, body_width, winner

    def ignore_moves_left(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    monkeypatch.setattr(network_module, "_encoder_tower", consume_body)
    monkeypatch.setattr(network_module, "_policy_head", ignore_policy)
    monkeypatch.setattr(network_module, "_value_head", ignore_value)
    monkeypatch.setattr(network_module, "_moves_left_head", ignore_moves_left)


def _stop_after_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace output heads with one read of the final encoder body."""

    def consume_policy(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del body_width, weights
        kernel = context.builder.add_kernel(_artifact("consume_body", 1))
        context.builder.call(kernel, body, readonly=(body,))

    def ignore_value(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        winner: net_pb2.Weights.ValueHead,
    ) -> None:
        del context, body, body_width, winner

    def ignore_moves_left(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    monkeypatch.setattr(network_module, "_policy_head", consume_policy)
    monkeypatch.setattr(network_module, "_value_head", ignore_value)
    monkeypatch.setattr(network_module, "_moves_left_head", ignore_moves_left)


def _location(argument: lc0ex_pb2.Node.Argument) -> tuple[int, int]:
    """Return one serialized allocation argument location."""
    return argument.allocation.index, argument.allocation.offset


def test_build_entry_point_is_importable() -> None:
    """BT4 graph construction is exposed through the protobuf-driven API."""
    assert callable(build)


def test_build_rejects_non_positive_batch_before_network_validation() -> None:
    """Invalid batches fail before allocations or protobuf traversal."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build(ExecutableBuilder(), net_pb2.Net(), batch_size=0)


def test_build_normalizes_and_builds_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalized network builds through its encoder production."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    _add_encoder(
        network,
        body_width=16,
        compression_width=2,
        hidden_width=16,
        generated_width=16,
        model_width=16,
        ffn_width=32,
    )
    _stub_compilers(monkeypatch)
    _stop_after_encoders(monkeypatch)

    build(ExecutableBuilder(), network, batch_size=1)

    assert (
        network.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    assert (
        network.format.network_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    )


def test_encoder_builds_names_order_and_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One block preserves ONNX names and the required evaluation order."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    network.weights.headcount = 2
    _add_encoder(
        network,
        body_width=16,
        compression_width=2,
        hidden_width=16,
        generated_width=16,
        model_width=16,
        ffn_width=32,
    )
    records = _stub_compilers(monkeypatch)
    _stop_after_encoders(monkeypatch)
    builder = ExecutableBuilder()

    build(builder, network, batch_size=2)
    executable = builder.build()
    nodes = executable.programs[0].nodes
    functions = [executable.kernels[node.kernel_idx].function for node in nodes]
    encoder_functions = functions[12:-1]

    assert encoder_functions == [
        "matmul",
        "matmul",
        "layer_norm",
        "matmul",
        "layer_norm",
        "matmul",
        "matmul",
        "add_bias_batched",
        "matmul",
        "add_bias_batched",
        "matmul",
        "add_bias_batched",
        "body_qk",
        "softmax_64",
        "body_attention_v",
        "matmul",
        "layer_norm_skip",
        "matmul",
        "add_bias_batched",
        "matmul",
        "layer_norm_skip",
    ]
    buffers = {buffer.name: tuple(buffer.shape) for buffer in executable.buffers}
    encoder_names = {name for name in buffers if name.startswith("/encoder0/")}
    assert len(encoder_names) == _ENCODER_LOCAL_BUFFER_COUNT
    assert buffers["/encoder0/mha/Q/w/w"] == (16, 16)
    assert buffers["/encoder0/smolgen/dense2/w/w"] == (16, 32)
    assert buffers["/const/smolgen_w"] == (16, 4096)
    assert buffers["/encoder0/alpha*input/w"] == (1,)
    assert buffers["/encoder0/ffn/alpha/w"] == (1,)

    specializations = [specialization for _name, specialization in records]
    assert (
        LayerNormSpecialization(
            row_count=2,
            width=32,
            activation="swish",
            has_skip=False,
            architecture=_ARCHITECTURE,
        )
        in specializations
    )
    assert (
        BatchedMatmulSpecialization("body_qk", 4, 64, 64, 8, 2, _ARCHITECTURE)
        in specializations
    )
    assert (
        BatchedMatmulSpecialization("body_attention_v", 4, 64, 8, 64, 2, _ARCHITECTURE)
        in specializations
    )
    assert Softmax64Specialization(4 * 64, _ARCHITECTURE) in specializations

    arguments = [[_location(argument) for argument in node.arguments] for node in nodes]
    encoder_start = 12
    ln1 = arguments[encoder_start + 16]
    ln2 = arguments[encoder_start + 20]
    assert ln1[3] == arguments[11][0]
    assert ln2[3] == ln1[0]
    assert arguments[-1][0] == ln2[0]


@pytest.mark.parametrize(
    ("encoding_width", "body_width", "hidden_width"),
    [(1, 16, 32), (3, 32, 48)],
)
def test_embedding_infers_external_shapes_from_layer_counts(
    monkeypatch: pytest.MonkeyPatch,
    encoding_width: int,
    body_width: int,
    hidden_width: int,
) -> None:
    """Vector counts and known matrix inputs determine every learned shape."""
    network = _embedding_network(
        encoding_width=encoding_width,
        body_width=body_width,
        hidden_width=hidden_width,
    )
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder()

    build(builder, network, batch_size=2)
    executable = builder.build()
    buffers = {buffer.name: buffer for buffer in executable.buffers}

    expected_shapes = {
        "/input/plane_masks": (2, 112),
        "/input/plane_values": (2, 112),
        "/attn_body/embedding/preprocess/matmul/w": (
            64 * 12,
            64 * encoding_width,
        ),
        "/attn_body/embedding/preprocess/add/w": (64 * encoding_width,),
        "/attn_body/matmul/w": (112 + encoding_width, body_width),
        "/attn_body/add/w": (body_width,),
        "/attn_body/ln/w/scale": (body_width,),
        "/attn_body/ln/w/bias": (body_width,),
        "/ip_mul_gate/w": (64, body_width),
        "/ip_add_gate/w": (64, body_width),
        "/attn_body/ffn/dense1/w/w": (body_width, hidden_width),
        "/attn_body/ffn/dense1/b/w": (hidden_width,),
        "/attn_body/ffn/dense2/w/w": (hidden_width, body_width),
        "/attn_body/ffn/dense2/b/w": (body_width,),
        "/attn_body/ffn/alpha/w": (1,),
        "/attn_body/ln2/w/scale": (body_width,),
        "/attn_body/ln2/w/bias": (body_width,),
    }

    assert {name: tuple(buffer.shape) for name, buffer in buffers.items()} == (
        expected_shapes
    )
    assert buffers["/input/plane_masks"].data_type == lc0ex_pb2.Buffer.DATA_TYPE_U64
    assert buffers["/input/plane_values"].data_type == lc0ex_pb2.Buffer.DATA_TYPE_F32
    assert all(
        buffer.data_type == lc0ex_pb2.Buffer.DATA_TYPE_F16
        for name, buffer in buffers.items()
        if not name.startswith("/input/")
    )
    assert all(
        executable.allocations[buffers[name].allocation_idx].lifetime
        == lc0ex_pb2.Allocation.LIFETIME_EXECUTION
        for name in ("/input/plane_masks", "/input/plane_values")
    )
    assert all(
        executable.allocations[buffer.allocation_idx].lifetime
        == lc0ex_pb2.Allocation.LIFETIME_PERSISTENT
        for name, buffer in buffers.items()
        if not name.startswith("/input/")
    )


def test_embedding_builds_expected_operations_and_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The segment composes tuned families with inferred immutable dimensions."""
    network = _embedding_network(encoding_width=3, body_width=32, hidden_width=48)
    records = _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder()

    build(builder, network, batch_size=2)
    executable = builder.build()
    nodes = executable.programs[0].nodes
    functions = [executable.kernels[node.kernel_idx].function for node in nodes]

    assert functions == [
        "expand_planes",
        "nchw_to_nhwc",
        "matmul",
        "add_vectors",
        "preprocess_attention_body",
        "matmul",
        "layer_norm",
        "input_gating",
        "matmul",
        "add_bias_batched",
        "matmul",
        "layer_norm_skip",
        "consume_body",
    ]
    assert [specialization for _name, specialization in records] == [
        ExpandPlanesSpecialization(2 * 112, _ARCHITECTURE),
        NchwToNhwcSpecialization(2, 112, 12, 8, 8, _ARCHITECTURE),
        MatmulSpecialization(2, 64 * 3, 64 * 12, _ARCHITECTURE),
        AddVectorsSpecialization(2 * 64 * 3, 64 * 3, "none", _ARCHITECTURE),
        PreprocessAttentionBodySpecialization(2, 112, 3, _ARCHITECTURE),
        MatmulSpecialization(2 * 64, 32, 112 + 3, _ARCHITECTURE),
        LayerNormSpecialization(
            row_count=2 * 64,
            width=32,
            activation="mish",
            has_skip=False,
            architecture=_ARCHITECTURE,
        ),
        InputGatingSpecialization(2, 64, 32, _ARCHITECTURE),
        MatmulSpecialization(2 * 64, 48, 32, _ARCHITECTURE),
        AddBiasBatchedSpecialization(1, 2 * 64, 48, "mish", _ARCHITECTURE),
        MatmulSpecialization(2 * 64, 32, 48, _ARCHITECTURE),
        LayerNormSpecialization(
            row_count=2 * 64,
            width=32,
            activation="none",
            has_skip=True,
            architecture=_ARCHITECTURE,
        ),
    ]


def test_embedding_preserves_pointer_reinterpretation_skip_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialized locations prove in-place calls, view-free reshaping, and reuse."""
    network = _embedding_network(encoding_width=3, body_width=32, hidden_width=48)
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder()

    build(builder, network, batch_size=2)
    executable = builder.build()
    nodes = executable.programs[0].nodes
    arguments = [[_location(argument) for argument in node.arguments] for node in nodes]

    assert [list(node.dependencies) for node in nodes] == [
        [],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
    ]
    assert arguments[1][0] == arguments[2][1]
    assert arguments[3][0] == arguments[3][1] == arguments[4][2]
    assert arguments[4][0] == arguments[5][1]
    assert arguments[7][0] == arguments[7][1] == arguments[8][1]
    assert arguments[9][0] == arguments[9][1] == arguments[10][1]
    assert arguments[11][3] == arguments[7][0]
    assert arguments[12][0] == arguments[11][0]

    temporary_outputs = {arguments[index][0] for index in (0, 1, 2, 4, 5, 6, 8, 10, 11)}
    assert len(temporary_outputs) == _REUSED_TEMPORARY_COUNT
    assert (
        len({arguments[4][0], arguments[4][1], arguments[4][2]})
        == _REUSED_TEMPORARY_COUNT
    )
    assert (
        len({arguments[11][0], arguments[11][1], arguments[11][3]})
        == _REUSED_TEMPORARY_COUNT
    )


def test_embedding_rejects_matrix_count_not_divisible_by_input_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matrix output width must be inferable from its known operation input."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    _set_elements(network.weights.ip_emb_ffn.dense1_w, 16 * 32 + 1)
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)

    with pytest.raises(
        NetworkFormatError,
        match=r"weights\.ip_emb_ffn\.dense1_w: .*not divisible",
    ):
        build(ExecutableBuilder(), network, batch_size=2)


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
