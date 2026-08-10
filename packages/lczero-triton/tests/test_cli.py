"""Tests for the Lc0 command-line interface."""

import pytest
from lczero_triton.cli import _build_parser


def test_parser_accepts_graph_network_and_batch_size() -> None:
    """Graph construction accepts its source network and optional batch size."""
    arguments = _build_parser().parse_args(
        [
            "graph",
            "--network",
            "network.pb.gz",
            "--output",
            "output.lc0ex",
            "--batch-size",
            "1",
        ]
    )

    assert arguments.command == "graph"
    assert arguments.batch_size == 1


@pytest.mark.parametrize("argv", [["graph", "--output", "output.lc0ex"], ["kernels"]])
def test_parser_rejects_missing_or_retired_command(argv: list[str]) -> None:
    """Network input is required and the old ahead-of-time command is gone."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(argv)


@pytest.mark.parametrize("option", ["--module", "--m", "--n", "--k"])
def test_parser_rejects_retired_options(option: str) -> None:
    """The graph command does not expose manifest or toy-matmul options."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "graph",
                "--network",
                "network.pb.gz",
                "--output",
                "output.lc0ex",
                option,
                "1",
            ]
        )
