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
from lczero_triton.bt4.kernels._autotune import active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.add_bias_batched import (
    AddBiasBatchedSpecialization,
    add_bias_batched,
)
from lczero_triton.bt4.kernels.add_vectors import (
    AddVectorsSpecialization,
    add_vectors,
)
from lczero_triton.bt4.kernels.batched_matmul import (
    BatchedMatmulSpecialization,
    batched_matmul,
)
from lczero_triton.bt4.kernels.expand_planes import (
    ExpandPlanesSpecialization,
    expand_planes,
)
from lczero_triton.bt4.kernels.input_gating import (
    InputGatingSpecialization,
    input_gating,
)
from lczero_triton.bt4.kernels.layer_norm import (
    LayerNormSpecialization,
    layer_norm,
)
from lczero_triton.bt4.kernels.matmul import MatmulSpecialization, matmul
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    NchwToNhwcSpecialization,
    nchw_to_nhwc,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
    preprocess_attention_body,
)
from lczero_triton.bt4.kernels.softmax_64 import (
    Softmax64Specialization,
    softmax_64,
)

_F16_SIZE_BYTES = 2
_INPUT_CHANNELS = 112
_POSITION_CHANNELS = 12
_SQUARE_COUNT = 64


@dataclass(slots=True)
class _BuildContext:
    """Construction services shared by grammar productions in one build."""

    builder: ExecutableBuilder
    persistent: Allocation
    execution: Allocation
    kernels: KernelCache
    batch_size: int
    architecture: int
    default_encoding: net_pb2.Format.Encoding
    default_activation: net_pb2.NetworkFormat.ActivationFunction
    ffn_activation: net_pb2.NetworkFormat.ActivationFunction
    smolgen_activation: net_pb2.NetworkFormat.ActivationFunction


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
    network_format = network.format.network_format
    default_activation = _default_activation(network_format.default_activation)
    context = _BuildContext(
        builder=builder,
        persistent=builder.allocation(lc0ex_pb2.Allocation.LIFETIME_PERSISTENT),
        execution=builder.allocation(lc0ex_pb2.Allocation.LIFETIME_EXECUTION),
        kernels=KernelCache(builder),
        batch_size=batch_size,
        architecture=active_architecture(),
        default_encoding=network.format.weights_encoding,
        default_activation=default_activation,
        ffn_activation=_resolve_activation(
            network_format.ffn_activation,
            default_activation,
            path="format.network_format.ffn_activation",
        ),
        smolgen_activation=_resolve_activation(
            network_format.smolgen_activation,
            default_activation,
            path="format.network_format.smolgen_activation",
        ),
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
    """Declare the packed execution inputs consumed by plane expansion."""
    masks = context.execution.external_buffer(
        name="/input/plane_masks",
        shape=(context.batch_size, _INPUT_CHANNELS),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64,
    )
    values = context.execution.external_buffer(
        name="/input/plane_values",
        shape=(context.batch_size, _INPUT_CHANNELS),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
    )
    return masks, values


def _embedding(  # noqa: PLR0915
    context: _BuildContext,
    inputs: tuple[Buffer, Buffer],
    weights: net_pb2.Weights,
) -> tuple[Buffer, int]:
    """Build dense positional preprocessing, gated embedding, and embedding FFN."""
    if weights.residual:
        message = "weights.residual: convolutional input towers are not supported"
        raise NetworkFormatError(message)
    _require_activation(
        context.default_activation,
        net_pb2.NetworkFormat.ACTIVATION_MISH,
        path="format.network_format.default_activation",
    )
    _require_activation(
        context.ffn_activation,
        net_pb2.NetworkFormat.ACTIVATION_MISH,
        path="format.network_format.ffn_activation",
    )

    position_input_width = _SQUARE_COUNT * _POSITION_CHANNELS
    position_weights, position_output_width = _matrix_f16(
        context,
        weights.ip_emb_preproc_w,
        input_width=position_input_width,
        name="/attn_body/embedding/preprocess/matmul/w",
        path="weights.ip_emb_preproc_w",
    )
    position_bias, position_bias_width = _vector_f16(
        context,
        weights.ip_emb_preproc_b,
        name="/attn_body/embedding/preprocess/add/w",
        path="weights.ip_emb_preproc_b",
    )
    _require_equal_widths(
        position_bias_width,
        position_output_width,
        path="weights.ip_emb_preproc_b",
        expected_path="weights.ip_emb_preproc_w output",
    )
    encoding_width, remainder = divmod(position_output_width, _SQUARE_COUNT)
    if remainder:
        message = (
            "weights.ip_emb_preproc_w: output width must contain an integral "
            "number of channels for every square"
        )
        raise NetworkFormatError(message)

    embedding_input_width = _INPUT_CHANNELS + encoding_width
    embedding_weights, body_width = _matrix_f16(
        context,
        weights.ip_emb_w,
        input_width=embedding_input_width,
        name="/attn_body/matmul/w",
        path="weights.ip_emb_w",
    )
    embedding_bias, embedding_bias_width = _vector_f16(
        context,
        weights.ip_emb_b,
        name="/attn_body/add/w",
        path="weights.ip_emb_b",
    )
    _require_equal_widths(
        embedding_bias_width,
        body_width,
        path="weights.ip_emb_b",
        expected_path="weights.ip_emb_w output",
    )
    embedding_gammas, gamma_width = _vector_f16(
        context,
        weights.ip_emb_ln_gammas,
        name="/attn_body/ln/w/scale",
        path="weights.ip_emb_ln_gammas",
    )
    embedding_betas, beta_width = _vector_f16(
        context,
        weights.ip_emb_ln_betas,
        name="/attn_body/ln/w/bias",
        path="weights.ip_emb_ln_betas",
    )
    _require_body_width(gamma_width, body_width, path="weights.ip_emb_ln_gammas")
    _require_body_width(beta_width, body_width, path="weights.ip_emb_ln_betas")

    multiplicative_gate, multiplicative_width = _matrix_f16(
        context,
        weights.ip_mult_gate,
        input_width=_SQUARE_COUNT,
        name="/ip_mul_gate/w",
        path="weights.ip_mult_gate",
    )
    additive_gate, additive_width = _matrix_f16(
        context,
        weights.ip_add_gate,
        input_width=_SQUARE_COUNT,
        name="/ip_add_gate/w",
        path="weights.ip_add_gate",
    )
    _require_body_width(multiplicative_width, body_width, path="weights.ip_mult_gate")
    _require_body_width(additive_width, body_width, path="weights.ip_add_gate")

    dense1_weights, hidden_width = _matrix_f16(
        context,
        weights.ip_emb_ffn.dense1_w,
        input_width=body_width,
        name="/attn_body/ffn/dense1/w/w",
        path="weights.ip_emb_ffn.dense1_w",
    )
    dense1_bias, dense1_bias_width = _vector_f16(
        context,
        weights.ip_emb_ffn.dense1_b,
        name="/attn_body/ffn/dense1/b/w",
        path="weights.ip_emb_ffn.dense1_b",
    )
    _require_equal_widths(
        dense1_bias_width,
        hidden_width,
        path="weights.ip_emb_ffn.dense1_b",
        expected_path="weights.ip_emb_ffn.dense1_w output",
    )
    dense2_weights, dense2_width = _matrix_f16(
        context,
        weights.ip_emb_ffn.dense2_w,
        input_width=hidden_width,
        name="/attn_body/ffn/dense2/w/w",
        path="weights.ip_emb_ffn.dense2_w",
    )
    _require_body_width(
        dense2_width,
        body_width,
        path="weights.ip_emb_ffn.dense2_w output",
    )
    dense2_bias, dense2_bias_width = _vector_f16(
        context,
        weights.ip_emb_ffn.dense2_b,
        name="/attn_body/ffn/dense2/b/w",
        path="weights.ip_emb_ffn.dense2_b",
    )
    _require_body_width(
        dense2_bias_width,
        body_width,
        path="weights.ip_emb_ffn.dense2_b",
    )
    alpha = context.persistent.external_buffer(
        name="/attn_body/ffn/alpha/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    ffn_gammas, ffn_gamma_width = _vector_f16(
        context,
        weights.ip_emb_ffn_ln_gammas,
        name="/attn_body/ln2/w/scale",
        path="weights.ip_emb_ffn_ln_gammas",
    )
    ffn_betas, ffn_beta_width = _vector_f16(
        context,
        weights.ip_emb_ffn_ln_betas,
        name="/attn_body/ln2/w/bias",
        path="weights.ip_emb_ffn_ln_betas",
    )
    _require_body_width(
        ffn_gamma_width,
        body_width,
        path="weights.ip_emb_ffn_ln_gammas",
    )
    _require_body_width(
        ffn_beta_width,
        body_width,
        path="weights.ip_emb_ffn_ln_betas",
    )

    masks, values = inputs
    plane_count = context.batch_size * _INPUT_CHANNELS
    token_rows = context.batch_size * _SQUARE_COUNT
    planes = _temporary_f16(
        context,
        element_count=plane_count * _SQUARE_COUNT,
    )
    expand_planes(
        context.builder,
        context.kernels,
        planes,
        masks,
        values,
        ExpandPlanesSpecialization(plane_count, context.architecture),
    )

    position_input = _temporary_f16(
        context,
        element_count=context.batch_size * position_input_width,
    )
    nchw_to_nhwc(
        context.builder,
        context.kernels,
        position_input,
        planes,
        NchwToNhwcSpecialization(
            context.batch_size,
            _INPUT_CHANNELS,
            _POSITION_CHANNELS,
            8,
            8,
            context.architecture,
        ),
    )

    position = _temporary_f16(
        context,
        element_count=context.batch_size * position_output_width,
    )
    matmul(
        context.builder,
        context.kernels,
        position,
        position_input,
        position_weights,
        MatmulSpecialization(
            context.batch_size,
            position_output_width,
            position_input_width,
            context.architecture,
        ),
    )
    add_vectors(
        context.builder,
        context.kernels,
        position,
        position,
        position_bias,
        AddVectorsSpecialization(
            context.batch_size * position_output_width,
            position_output_width,
            "none",
            context.architecture,
        ),
    )

    embedding_input = _temporary_f16(
        context,
        element_count=token_rows * embedding_input_width,
    )
    preprocess_attention_body(
        context.builder,
        context.kernels,
        embedding_input,
        planes,
        position,
        PreprocessAttentionBodySpecialization(
            context.batch_size,
            _INPUT_CHANNELS,
            encoding_width,
            context.architecture,
        ),
    )

    projected = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        projected,
        embedding_input,
        embedding_weights,
        MatmulSpecialization(
            token_rows,
            body_width,
            embedding_input_width,
            context.architecture,
        ),
    )
    skip = _temporary_f16(context, element_count=token_rows * body_width)
    layer_norm(
        context.builder,
        context.kernels,
        skip,
        projected,
        embedding_bias,
        embedding_gammas,
        embedding_betas,
        LayerNormSpecialization(
            row_count=token_rows,
            width=body_width,
            activation="mish",
            has_skip=False,
            architecture=context.architecture,
        ),
    )
    input_gating(
        context.builder,
        context.kernels,
        skip,
        skip,
        multiplicative_gate,
        additive_gate,
        InputGatingSpecialization(
            context.batch_size,
            _SQUARE_COUNT,
            body_width,
            context.architecture,
        ),
    )

    hidden = _temporary_f16(context, element_count=token_rows * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        skip,
        dense1_weights,
        MatmulSpecialization(
            token_rows,
            hidden_width,
            body_width,
            context.architecture,
        ),
    )
    add_bias_batched(
        context.builder,
        context.kernels,
        hidden,
        hidden,
        dense1_bias,
        AddBiasBatchedSpecialization(
            1,
            token_rows,
            hidden_width,
            "mish",
            context.architecture,
        ),
    )

    branch = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        branch,
        hidden,
        dense2_weights,
        MatmulSpecialization(
            token_rows,
            body_width,
            hidden_width,
            context.architecture,
        ),
    )
    body = _temporary_f16(context, element_count=token_rows * body_width)
    layer_norm(
        context.builder,
        context.kernels,
        body,
        branch,
        dense2_bias,
        ffn_gammas,
        ffn_betas,
        LayerNormSpecialization(
            row_count=token_rows,
            width=body_width,
            activation="none",
            has_skip=True,
            architecture=context.architecture,
        ),
        skip=skip,
        alpha=alpha,
    )
    return body, body_width


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
    smolgen, generated_width = _smolgen(
        context,
        body,
        body_width,
        encoder,
        prefix=prefix,
        head_count=head_count,
    )
    attended = _attention(
        context,
        body,
        smolgen,
        body_width,
        encoder,
        prefix=prefix,
        head_count=head_count,
        shared_smolgen=shared_smolgen,
        generated_width=generated_width,
    )
    return _ffn(context, attended, body_width, encoder, prefix=prefix)


def _smolgen(  # noqa: PLR0913
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
    head_count: int,
) -> tuple[Buffer, int]:
    """Build local Smolgen compression and its two normalized dense layers."""
    _require_activation(
        context.smolgen_activation,
        net_pb2.NetworkFormat.ACTIVATION_SWISH,
        path="format.network_format.smolgen_activation",
    )
    smolgen = encoder.mha.smolgen
    path = f"weights.{prefix[1:]}.mha.smolgen"
    compress_weights, compression_width = _matrix_f16(
        context,
        smolgen.compress,
        input_width=body_width,
        name=f"{prefix}/smolgen/compress/w",
        path=f"{path}.compress",
    )
    dense1_weights, hidden_width = _matrix_f16(
        context,
        smolgen.dense1_w,
        input_width=_SQUARE_COUNT * compression_width,
        name=f"{prefix}/smolgen/dense1/w/w",
        path=f"{path}.dense1_w",
    )
    dense1_bias, dense1_bias_width = _vector_f16(
        context,
        smolgen.dense1_b,
        name=f"{prefix}/smolgen/dense1/b/w",
        path=f"{path}.dense1_b",
    )
    _require_equal_widths(
        dense1_bias_width,
        hidden_width,
        path=f"{path}.dense1_b",
        expected_path=f"{path}.dense1_w output",
    )
    ln1_gammas, ln1_gamma_width = _vector_f16(
        context,
        smolgen.ln1_gammas,
        name=f"{prefix}/smolgen/ln1/w/scale",
        path=f"{path}.ln1_gammas",
    )
    ln1_betas, ln1_beta_width = _vector_f16(
        context,
        smolgen.ln1_betas,
        name=f"{prefix}/smolgen/ln1/w/bias",
        path=f"{path}.ln1_betas",
    )
    _require_equal_widths(
        ln1_gamma_width,
        hidden_width,
        path=f"{path}.ln1_gammas",
        expected_path=f"{path}.dense1_w output",
    )
    _require_equal_widths(
        ln1_beta_width,
        hidden_width,
        path=f"{path}.ln1_betas",
        expected_path=f"{path}.dense1_w output",
    )
    dense2_weights, generated_total_width = _matrix_f16(
        context,
        smolgen.dense2_w,
        input_width=hidden_width,
        name=f"{prefix}/smolgen/dense2/w/w",
        path=f"{path}.dense2_w",
    )
    generated_width, remainder = divmod(generated_total_width, head_count)
    if remainder:
        message = f"{path}.dense2_w: output width must be divisible by head count"
        raise NetworkFormatError(message)
    dense2_bias, dense2_bias_width = _vector_f16(
        context,
        smolgen.dense2_b,
        name=f"{prefix}/smolgen/dense2/b/w",
        path=f"{path}.dense2_b",
    )
    ln2_gammas, ln2_gamma_width = _vector_f16(
        context,
        smolgen.ln2_gammas,
        name=f"{prefix}/smolgen/ln2/w/scale",
        path=f"{path}.ln2_gammas",
    )
    ln2_betas, ln2_beta_width = _vector_f16(
        context,
        smolgen.ln2_betas,
        name=f"{prefix}/smolgen/ln2/w/bias",
        path=f"{path}.ln2_betas",
    )
    for width, field in (
        (dense2_bias_width, "dense2_b"),
        (ln2_gamma_width, "ln2_gammas"),
        (ln2_beta_width, "ln2_betas"),
    ):
        _require_equal_widths(
            width,
            generated_total_width,
            path=f"{path}.{field}",
            expected_path=f"{path}.dense2_w output",
        )

    token_rows = context.batch_size * _SQUARE_COUNT
    compressed = _temporary_f16(context, element_count=token_rows * compression_width)
    matmul(
        context.builder,
        context.kernels,
        compressed,
        body,
        compress_weights,
        MatmulSpecialization(
            token_rows, compression_width, body_width, context.architecture
        ),
    )
    hidden = _temporary_f16(context, element_count=context.batch_size * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        compressed,
        dense1_weights,
        MatmulSpecialization(
            context.batch_size,
            hidden_width,
            _SQUARE_COUNT * compression_width,
            context.architecture,
        ),
    )
    layer_norm(
        context.builder,
        context.kernels,
        hidden,
        hidden,
        dense1_bias,
        ln1_gammas,
        ln1_betas,
        LayerNormSpecialization(
            row_count=context.batch_size,
            width=hidden_width,
            activation="swish",
            has_skip=False,
            architecture=context.architecture,
        ),
    )
    generated = _temporary_f16(
        context, element_count=context.batch_size * generated_total_width
    )
    matmul(
        context.builder,
        context.kernels,
        generated,
        hidden,
        dense2_weights,
        MatmulSpecialization(
            context.batch_size,
            generated_total_width,
            hidden_width,
            context.architecture,
        ),
    )
    layer_norm(
        context.builder,
        context.kernels,
        generated,
        generated,
        dense2_bias,
        ln2_gammas,
        ln2_betas,
        LayerNormSpecialization(
            row_count=context.batch_size,
            width=generated_total_width,
            activation="swish",
            has_skip=False,
            architecture=context.architecture,
        ),
    )
    return generated, generated_width


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
    generated_width: int,
) -> Buffer:
    """Build shared Smolgen projection and the encoder Q/K/V attention path."""
    mha = encoder.mha
    path = f"weights.{prefix[1:]}"
    shared_weights, smolgen_output_width = _matrix_f16(
        context,
        shared_smolgen,
        input_width=generated_width,
        name="/const/smolgen_w",
        path="weights.smolgen_w",
    )
    expected_smolgen_width = _SQUARE_COUNT * _SQUARE_COUNT
    _require_equal_widths(
        smolgen_output_width,
        expected_smolgen_width,
        path="weights.smolgen_w output",
        expected_path="64-way attention logits",
    )

    projections: list[tuple[Buffer, Buffer, int, str]] = []
    for label, weight_field, bias_field in (
        ("Q", mha.q_w, mha.q_b),
        ("K", mha.k_w, mha.k_b),
        ("V", mha.v_w, mha.v_b),
    ):
        weights, width = _matrix_f16(
            context,
            weight_field,
            input_width=body_width,
            name=f"{prefix}/mha/{label}/w/w",
            path=f"{path}.mha.{label.lower()}_w",
        )
        bias, bias_width = _vector_f16(
            context,
            bias_field,
            name=f"{prefix}/mha/{label}/b/w",
            path=f"{path}.mha.{label.lower()}_b",
        )
        _require_equal_widths(
            bias_width,
            width,
            path=f"{path}.mha.{label.lower()}_b",
            expected_path=f"{path}.mha.{label.lower()}_w output",
        )
        projections.append((weights, bias, width, label))
    model_width = projections[0][2]
    for _weights, _bias, width, label in projections[1:]:
        _require_equal_widths(
            width,
            model_width,
            path=f"{path}.mha.{label.lower()}_w output",
            expected_path=f"{path}.mha.q_w output",
        )
    head_depth, remainder = divmod(model_width, head_count)
    if remainder:
        message = f"{path}.mha.q_w: output width must be divisible by head count"
        raise NetworkFormatError(message)

    output_weights, output_width = _matrix_f16(
        context,
        mha.dense_w,
        input_width=model_width,
        name=f"{prefix}/mha/out/dense/w/w",
        path=f"{path}.mha.dense_w",
    )
    _require_body_width(output_width, body_width, path=f"{path}.mha.dense_w output")
    output_bias, output_bias_width = _vector_f16(
        context,
        mha.dense_b,
        name=f"{prefix}/mha/out/dense/b/w",
        path=f"{path}.mha.dense_b",
    )
    _require_body_width(output_bias_width, body_width, path=f"{path}.mha.dense_b")
    gammas, gamma_width = _vector_f16(
        context,
        encoder.ln1_gammas,
        name=f"{prefix}/ln1/w/scale",
        path=f"{path}.ln1_gammas",
    )
    betas, beta_width = _vector_f16(
        context,
        encoder.ln1_betas,
        name=f"{prefix}/ln1/w/bias",
        path=f"{path}.ln1_betas",
    )
    _require_body_width(gamma_width, body_width, path=f"{path}.ln1_gammas")
    _require_body_width(beta_width, body_width, path=f"{path}.ln1_betas")
    scale = context.persistent.external_buffer(
        name=f"{prefix}/mha/QK/scale/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    alpha = context.persistent.external_buffer(
        name=f"{prefix}/alpha*input/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    attention_batches = context.batch_size * head_count
    smolgen_logits = _temporary_f16(
        context, element_count=attention_batches * expected_smolgen_width
    )
    matmul(
        context.builder,
        context.kernels,
        smolgen_logits,
        smolgen,
        shared_weights,
        MatmulSpecialization(
            attention_batches,
            expected_smolgen_width,
            generated_width,
            context.architecture,
        ),
    )
    projected: list[Buffer] = []
    for weights, bias, width, _label in projections:
        value = _temporary_f16(context, element_count=token_rows * width)
        matmul(
            context.builder,
            context.kernels,
            value,
            body,
            weights,
            MatmulSpecialization(token_rows, width, body_width, context.architecture),
        )
        add_bias_batched(
            context.builder,
            context.kernels,
            value,
            value,
            bias,
            AddBiasBatchedSpecialization(
                1, token_rows, width, "none", context.architecture
            ),
        )
        projected.append(value)
    queries, keys, values = projected
    qk = _temporary_f16(
        context, element_count=attention_batches * expected_smolgen_width
    )
    batched_matmul(
        context.builder,
        context.kernels,
        qk,
        queries,
        keys,
        BatchedMatmulSpecialization(
            "body_qk",
            attention_batches,
            _SQUARE_COUNT,
            _SQUARE_COUNT,
            head_depth,
            head_count,
            context.architecture,
        ),
        scale=scale,
    )
    softmax_64(
        context.builder,
        context.kernels,
        qk,
        qk,
        smolgen_logits,
        Softmax64Specialization(
            attention_batches * _SQUARE_COUNT, context.architecture
        ),
    )
    merged = _temporary_f16(context, element_count=token_rows * model_width)
    batched_matmul(
        context.builder,
        context.kernels,
        merged,
        qk,
        values,
        BatchedMatmulSpecialization(
            "body_attention_v",
            attention_batches,
            _SQUARE_COUNT,
            head_depth,
            _SQUARE_COUNT,
            head_count,
            context.architecture,
        ),
    )
    branch = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        branch,
        merged,
        output_weights,
        MatmulSpecialization(token_rows, body_width, model_width, context.architecture),
    )
    attended = _temporary_f16(context, element_count=token_rows * body_width)
    layer_norm(
        context.builder,
        context.kernels,
        attended,
        branch,
        output_bias,
        gammas,
        betas,
        LayerNormSpecialization(
            row_count=token_rows,
            width=body_width,
            activation="none",
            has_skip=True,
            architecture=context.architecture,
        ),
        skip=body,
        alpha=alpha,
    )
    return attended


def _ffn(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    encoder: net_pb2.Weights.EncoderLayer,
    *,
    prefix: str,
) -> Buffer:
    """Build an encoder FFN and its DeepNorm residual layer normalization."""
    _require_activation(
        context.ffn_activation,
        net_pb2.NetworkFormat.ACTIVATION_MISH,
        path="format.network_format.ffn_activation",
    )
    path = f"weights.{prefix[1:]}"
    dense1_weights, hidden_width = _matrix_f16(
        context,
        encoder.ffn.dense1_w,
        input_width=body_width,
        name=f"{prefix}/ffn/dense1/w/w",
        path=f"{path}.ffn.dense1_w",
    )
    dense1_bias, dense1_bias_width = _vector_f16(
        context,
        encoder.ffn.dense1_b,
        name=f"{prefix}/ffn/dense1/b/w",
        path=f"{path}.ffn.dense1_b",
    )
    _require_equal_widths(
        dense1_bias_width,
        hidden_width,
        path=f"{path}.ffn.dense1_b",
        expected_path=f"{path}.ffn.dense1_w output",
    )
    dense2_weights, output_width = _matrix_f16(
        context,
        encoder.ffn.dense2_w,
        input_width=hidden_width,
        name=f"{prefix}/ffn/dense2/w/w",
        path=f"{path}.ffn.dense2_w",
    )
    _require_body_width(output_width, body_width, path=f"{path}.ffn.dense2_w output")
    dense2_bias, dense2_bias_width = _vector_f16(
        context,
        encoder.ffn.dense2_b,
        name=f"{prefix}/ffn/dense2/b/w",
        path=f"{path}.ffn.dense2_b",
    )
    gammas, gamma_width = _vector_f16(
        context,
        encoder.ln2_gammas,
        name=f"{prefix}/ln2/w/scale",
        path=f"{path}.ln2_gammas",
    )
    betas, beta_width = _vector_f16(
        context,
        encoder.ln2_betas,
        name=f"{prefix}/ln2/w/bias",
        path=f"{path}.ln2_betas",
    )
    for width, field in (
        (dense2_bias_width, "ffn.dense2_b"),
        (gamma_width, "ln2_gammas"),
        (beta_width, "ln2_betas"),
    ):
        _require_body_width(width, body_width, path=f"{path}.{field}")
    alpha = context.persistent.external_buffer(
        name=f"{prefix}/ffn/alpha/w",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    token_rows = context.batch_size * _SQUARE_COUNT
    hidden = _temporary_f16(context, element_count=token_rows * hidden_width)
    matmul(
        context.builder,
        context.kernels,
        hidden,
        body,
        dense1_weights,
        MatmulSpecialization(
            token_rows, hidden_width, body_width, context.architecture
        ),
    )
    add_bias_batched(
        context.builder,
        context.kernels,
        hidden,
        hidden,
        dense1_bias,
        AddBiasBatchedSpecialization(
            1, token_rows, hidden_width, "mish", context.architecture
        ),
    )
    branch = _temporary_f16(context, element_count=token_rows * body_width)
    matmul(
        context.builder,
        context.kernels,
        branch,
        hidden,
        dense2_weights,
        MatmulSpecialization(
            token_rows, body_width, hidden_width, context.architecture
        ),
    )
    output = _temporary_f16(context, element_count=token_rows * body_width)
    layer_norm(
        context.builder,
        context.kernels,
        output,
        branch,
        dense2_bias,
        gammas,
        betas,
        LayerNormSpecialization(
            row_count=token_rows,
            width=body_width,
            activation="none",
            has_skip=True,
            architecture=context.architecture,
        ),
        skip=body,
        alpha=alpha,
    )
    return output


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


def _vector_f16(
    context: _BuildContext,
    layer: net_pb2.Weights.Layer,
    *,
    name: str,
    path: str,
) -> tuple[Buffer, int]:
    """Declare an FP16 vector whose width is inferred from its payload."""
    width = _layer_elements(layer, context.default_encoding, path=path)
    return (
        context.persistent.external_buffer(
            name=name,
            shape=(width,),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        ),
        width,
    )


def _matrix_f16(
    context: _BuildContext,
    layer: net_pb2.Weights.Layer,
    *,
    input_width: int,
    name: str,
    path: str,
) -> tuple[Buffer, int]:
    """Infer and declare an ONNX-layout FP16 matrix from its known input width."""
    element_count = _layer_elements(layer, context.default_encoding, path=path)
    output_width, remainder = divmod(element_count, input_width)
    if remainder:
        message = (
            f"{path}: {element_count} elements are not divisible by input width "
            f"{input_width}"
        )
        raise NetworkFormatError(message)
    return (
        context.persistent.external_buffer(
            name=name,
            shape=(input_width, output_width),
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        ),
        output_width,
    )


def _require_equal_widths(
    actual: int,
    expected: int,
    *,
    path: str,
    expected_path: str,
) -> None:
    """Require dimensions joined by one operation to agree."""
    if actual != expected:
        message = (
            f"{path}: width {actual} does not match {expected_path} width {expected}"
        )
        raise NetworkFormatError(message)


def _require_body_width(actual: int, expected: int, *, path: str) -> None:
    """Require one embedding operand to preserve the residual body width."""
    _require_equal_widths(
        actual,
        expected,
        path=path,
        expected_path="attention body",
    )


def _require_activation(
    actual: net_pb2.NetworkFormat.ActivationFunction,
    expected: net_pb2.NetworkFormat.ActivationFunction,
    *,
    path: str,
) -> None:
    """Require an activation implemented by the current embedding kernels."""
    if actual != expected:
        message = (
            f"{path}: embedding graph requires activation enum {expected}, got {actual}"
        )
        raise NetworkFormatError(message)


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
