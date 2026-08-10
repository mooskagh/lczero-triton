"""Tests for the BT4 package boundary."""

from lczero_triton.bt4.kernels import compile_kernels
from lczero_triton.bt4.network import build_graph


def test_bt4_entry_points_are_importable() -> None:
    """The BT4 package owns compilation and graph construction entry points."""
    assert callable(compile_kernels)
    assert callable(build_graph)
