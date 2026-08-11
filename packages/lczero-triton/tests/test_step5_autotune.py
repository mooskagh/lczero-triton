"""Static autotuning-contract tests for the Step 5 kernel families."""

from dataclasses import fields

from lczero_triton.bt4.kernels._autotune import (
    _ELEMENTWISE_CONFIGURATIONS,
    _PREPROCESS_CONFIGURATIONS,
    elementwise_configs,
)
from lczero_triton.bt4.kernels.add_bias_batched import (
    AddBiasBatchedSpecialization,
    _add_bias_batched_kernel,
)
from lczero_triton.bt4.kernels.add_vectors import (
    AddVectorsSpecialization,
    _add_vectors_kernel,
)
from lczero_triton.bt4.kernels.copy_type_converted import (
    CopyTypeConvertedSpecialization,
    _copy_type_converted_kernel,
)
from lczero_triton.bt4.kernels.expand_planes import (
    ExpandPlanesSpecialization,
    _expand_planes_kernel,
)
from lczero_triton.bt4.kernels.input_gating import (
    InputGatingSpecialization,
    _input_gating_kernel,
)
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    NchwToNhwcSpecialization,
    _nchw_to_nhwc_kernel,
)
from lczero_triton.bt4.kernels.policy_map import (
    PolicyMapSpecialization,
    _policy_map_kernel,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
    _preprocess_attention_body_kernel,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    _autotune_grid as preprocess_grid,
)

_KERNEL_KEYS = (
    (_copy_type_converted_kernel, ["element_count"]),
    (_expand_planes_kernel, ["plane_count", "square_count"]),
    (
        _nchw_to_nhwc_kernel,
        ["batch_size", "input_channels", "output_channels", "height", "width"],
    ),
    (
        _preprocess_attention_body_kernel,
        ["batch_size", "square_count", "input_channels", "encoding_channels"],
    ),
    (_input_gating_kernel, ["batch_size", "square_count", "channel_count"]),
    (
        _add_vectors_kernel,
        ["element_count", "bias_element_count", "activation"],
    ),
    (
        _add_bias_batched_kernel,
        ["batch_count", "row_count", "channel_count", "activation"],
    ),
    (
        _policy_map_kernel,
        ["batch_size", "input_element_count", "output_element_count"],
    ),
)
_SPECIALIZATIONS = (
    CopyTypeConvertedSpecialization,
    ExpandPlanesSpecialization,
    NchwToNhwcSpecialization,
    PreprocessAttentionBodySpecialization,
    InputGatingSpecialization,
    AddVectorsSpecialization,
    AddBiasBatchedSpecialization,
    PolicyMapSpecialization,
)


def test_every_step_five_family_persistently_autotunes_its_semantic_shape() -> None:
    """Every family keys persistent tuning by values that change its workload."""
    for kernel, expected_keys in _KERNEL_KEYS:
        assert kernel.keys == expected_keys
        assert kernel.cache_results


def test_elementwise_candidates_cover_block_width_and_occupancy() -> None:
    """Flat kernels compare the agreed block-size and warp-count candidates."""
    configurations = elementwise_configs()
    actual = tuple(
        (configuration.kwargs["block_size"], configuration.num_warps)
        for configuration in configurations
    )

    assert actual == _ELEMENTWISE_CONFIGURATIONS
    assert all(
        left is not right
        for left, right in zip(configurations, elementwise_configs(), strict=True)
    )


def test_preprocess_candidates_are_channel_tiled() -> None:
    """Every preprocessing candidate covers a 624-wide output without loss."""
    for block_size, _num_warps in _PREPROCESS_CONFIGURATIONS:
        grid = preprocess_grid(
            {
                "batch_size": 169,
                "square_count": 64,
                "input_channels": 112,
                "encoding_channels": 512,
                "block_size": block_size,
            }
        )

        assert grid == (169, 64, (624 + block_size - 1) // block_size)


def test_launch_choices_are_not_public_specialization_inputs() -> None:
    """Callers specialize semantics and target, not the autotuner's result."""
    for specialization in _SPECIALIZATIONS:
        field_names = {field.name for field in fields(specialization)}

        assert "block_size" not in field_names
        assert "num_warps" not in field_names
