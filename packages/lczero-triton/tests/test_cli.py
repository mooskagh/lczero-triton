"""Tests for the Lc0 command-line interface."""

import logging
import os
import sys
from pathlib import Path

import pytest
from lczero_triton import cli
from lczero_triton.cli import _build_parser, _configure_logging


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
    assert arguments.batch_sizes == [1]


def test_main_keeps_artifact_path_on_stdout_and_progress_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Autotuning-style progress does not corrupt the stdout artifact path."""

    class _Builder:
        def build_and_write(self, output: Path) -> None:
            assert output == Path("output.lc0ex")

    object_network = object()

    def fake_build(
        builder: _Builder,
        network: object,
        *,
        batch_sizes: list[int] | None,
    ) -> None:
        assert isinstance(builder, _Builder)
        assert network is object_network
        assert batch_sizes == [1]
        sys.stdout.write("autotuning progress\n")

    monkeypatch.setattr(cli, "ExecutableBuilder", _Builder)
    monkeypatch.setattr(cli, "load_network", lambda _path: object_network)
    monkeypatch.setattr(cli, "build", fake_build)
    monkeypatch.setattr(cli, "_configure_logging", lambda: None)
    monkeypatch.delenv("TRITON_PRINT_AUTOTUNING", raising=False)

    assert (
        cli.main(
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
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == "output.lc0ex\n"
    assert "autotuning progress\n" in captured.err
    assert "TRITON_PRINT_AUTOTUNING" not in os.environ


def test_logging_defaults_to_info_on_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI logging setup uses INFO level and stderr as its default stream."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: calls.append(kwargs),
    )

    _configure_logging()

    assert calls == [
        {
            "level": logging.INFO,
            "format": "%(levelname)s: %(message)s",
            "stream": sys.stderr,
        }
    ]


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
