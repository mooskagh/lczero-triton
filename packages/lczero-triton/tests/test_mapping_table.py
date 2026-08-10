"""Tests for the embedded BT4 attention-policy gather table."""

from lczero_triton.bt4.kernels.mapping_table import values

_POLICY_SIZE = 1858
_ATTENTION_RECORD_SIZE = 4288


def test_mapping_table_is_a_complete_attention_policy_gather() -> None:
    """The table maps each policy output to one valid 4288-wide source index."""
    table = values()

    assert len(table) == _POLICY_SIZE
    assert len(set(table)) == _POLICY_SIZE
    assert min(table) >= 0
    assert max(table) < _ATTENTION_RECORD_SIZE
    assert table[:10] == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert table[-10:] == (4260, 4261, 4262, 4263, 4282, 4283, 4284, 4285, 4286, 4287)
