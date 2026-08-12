"""Build a BT4 computation graph from an Lc0 weights file."""

import argparse
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from lc0ex import ExecutableBuilder

from lczero_triton.bt4._format import load_network
from lczero_triton.bt4.network import build

_LOGGER = logging.getLogger(__name__)
_AUTOTUNE_PRINT_ENV = "TRITON_PRINT_AUTOTUNING"


def main(argv: Sequence[str] | None = None) -> int:
    """Run an Lc0-specific kernel or graph command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _configure_logging()
    _LOGGER.info("loading network %s", arguments.network)
    network = load_network(arguments.network)
    builder = ExecutableBuilder()
    _LOGGER.info("starting graph construction")
    with _autotune_progress(), redirect_stdout(sys.stderr):
        build(builder, network, batch_sizes=arguments.batch_sizes)
    _LOGGER.info("serializing executable to %s", arguments.output)
    builder.build_and_write(arguments.output)
    sys.stdout.write(f"{arguments.output}\n")
    return 0


def _configure_logging() -> None:
    """Configure human-readable application logs on stderr by default."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@contextmanager
def _autotune_progress() -> Iterator[None]:
    """Enable Triton's progress output without changing the caller's environment."""
    previous = os.environ.get(_AUTOTUNE_PRINT_ENV)
    if previous is None:
        os.environ[_AUTOTUNE_PRINT_ENV] = "1"
        _LOGGER.info("Triton autotuning progress is enabled")
    elif previous == "0":
        _LOGGER.info("Triton autotuning progress is disabled by environment")
    else:
        _LOGGER.info("Triton autotuning progress is enabled by environment")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_AUTOTUNE_PRINT_ENV, None)
        else:
            os.environ[_AUTOTUNE_PRINT_ENV] = previous


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph_parser = subparsers.add_parser(
        "graph",
        help="build an Lc0 computation graph from a weights file",
        allow_abbrev=False,
    )
    graph_parser.add_argument("--network", type=Path, required=True)
    graph_parser.add_argument("--output", type=Path, required=True)
    graph_parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        type=_positive_integer,
        action="append",
        default=None,
        help="fixed batch size to compile; repeat for multiple programs",
    )
    return parser


def _positive_integer(value: str) -> int:
    """Parse an argparse integer that must be positive."""
    parsed = int(value)
    if parsed <= 0:
        message = "must be positive"
        raise argparse.ArgumentTypeError(message)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
