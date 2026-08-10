"""Tests for the Lc0 command-line interface."""

import pytest
from lczero_triton.cli import _build_parser


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["kernels", "--output", "module.pb"], "kernels"),
        (
            ["graph", "--module", "module.pb", "--output", "output.pb"],
            "graph",
        ),
    ],
)
def test_parser_accepts_command(argv: list[str], command: str) -> None:
    """The CLI exposes commands named after their domain artifacts."""
    arguments = _build_parser().parse_args(argv)

    assert arguments.command == command


@pytest.mark.parametrize("command", ["compile", "link"])
def test_parser_rejects_retired_command(command: str) -> None:
    """The old build-phase terminology is no longer accepted."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args([command])


@pytest.mark.parametrize("option", ["--m", "--n", "--k"])
def test_parser_rejects_toy_dimensions(option: str) -> None:
    """The CLI no longer exposes generic matmul dimensions."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["kernels", "--output", "module.pb", option, "1"])
