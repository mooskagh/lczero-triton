"""Build a BT4 computation graph from an Lc0 weights file."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lc0ex import ExecutableBuilder

from lczero_triton.bt4._format import load_network
from lczero_triton.bt4.network import build


def main(argv: Sequence[str] | None = None) -> int:
    """Run an Lc0-specific kernel or graph command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    network = load_network(arguments.network)
    builder = ExecutableBuilder()
    build(builder, network, batch_sizes=arguments.batch_sizes)
    builder.build_and_write(arguments.output)
    sys.stdout.write(f"{arguments.output}\n")
    return 0


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
