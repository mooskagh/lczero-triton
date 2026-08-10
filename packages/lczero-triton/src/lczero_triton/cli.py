"""Build BT4 kernels and computation graphs."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lc0ex import ExecutableBuilder, KernelHandle

from lczero_triton.bt4.kernels import compile_kernels
from lczero_triton.bt4.network import build_graph


def main(argv: Sequence[str] | None = None) -> int:
    """Run an Lc0-specific kernel or graph command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "kernels":
        manifest_path = compile_kernels(arguments.output)
    else:
        builder = ExecutableBuilder()
        kernels: list[KernelHandle] = []
        for manifest_path in arguments.module:
            kernels.extend(builder.add_module(manifest_path))
        build_graph(builder, kernels)
        builder.build_and_write(arguments.output)
        manifest_path = arguments.output
    sys.stdout.write(f"{manifest_path}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    kernels_parser = subparsers.add_parser(
        "kernels",
        help="compile Lc0-specific Triton kernels",
    )
    kernels_parser.add_argument("--output", type=Path, required=True)

    graph_parser = subparsers.add_parser(
        "graph",
        help="build an Lc0 computation graph from compiled kernel modules",
    )
    graph_parser.add_argument("--module", type=Path, action="append", required=True)
    graph_parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
