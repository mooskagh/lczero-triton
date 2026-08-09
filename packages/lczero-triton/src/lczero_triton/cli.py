"""Build kernels and computation graphs for the example network."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lc0ex import ExecutableBuilder

from lczero_triton.network import K, M, N, build_matmul_graph


def main(argv: Sequence[str] | None = None) -> int:
    """Run an Lc0-specific kernel or graph command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "kernels":
        from lczero_triton.kernels import compile_matmul  # noqa: PLC0415

        manifest_path = compile_matmul(
            arguments.m,
            arguments.n,
            arguments.k,
            arguments.output,
        )
    else:
        builder = ExecutableBuilder()
        for manifest_path in arguments.module:
            builder.add_module(manifest_path)
        build_matmul_graph(builder, arguments.m, arguments.n, arguments.k)
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
    _add_dimensions(kernels_parser)

    graph_parser = subparsers.add_parser(
        "graph",
        help="build an Lc0 computation graph from compiled kernel modules",
    )
    graph_parser.add_argument("--module", type=Path, action="append", required=True)
    graph_parser.add_argument("--output", type=Path, required=True)
    _add_dimensions(graph_parser)
    return parser


def _add_dimensions(parser: argparse.ArgumentParser) -> None:
    """Add specialization dimensions shared by kernels and graphs."""
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--k", type=int, default=K)


if __name__ == "__main__":
    raise SystemExit(main())
