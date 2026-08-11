"""Convert Triton output into executable-linker artifacts."""

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from triton.backends.nvidia.compiler import get_ptxas

from lc0ex.kernel_builder import KernelArtifact
from lc0ex.proto import lc0ex_pb2

Grid = tuple[int, int, int]

_ARCHITECTURE_PATTERN = re.compile(r"sm_(\d+)(?:a)?")


def compile_ptx(ptx: str, *, architecture: str) -> bytes:
    """Assemble PTX for *architecture* with Triton's configured ptxas."""
    match = _ARCHITECTURE_PATTERN.fullmatch(architecture)
    if match is None:
        message = f"invalid NVIDIA architecture: {architecture!r}"
        raise ValueError(message)

    with TemporaryDirectory(prefix="lc0ex-ptx-") as directory:
        path = Path(directory)
        source_path = path / "module.ptx"
        cubin_path = path / "module.cubin"
        source_path.write_text(ptx, encoding="utf-8")
        subprocess.run(  # noqa: S603  # ptxas path is selected by Triton.
            (
                get_ptxas(int(match.group(1))).path,
                f"--gpu-name={architecture}",
                str(source_path),
                "--output-file",
                str(cubin_path),
            ),
            check=True,
        )
        return cubin_path.read_bytes()


def artifact_from_triton(
    compiled: Any,
    *,
    grid: Grid,
    parameters: tuple[lc0ex_pb2.ParameterType, ...],
) -> KernelArtifact:
    """Convert one compiled Triton entry point into a linker artifact."""
    block = (
        compiled.metadata.num_warps * compiled.metadata.target.warp_size,
        1,
        1,
    )
    cubin = compiled.asm["cubin"]
    if not isinstance(cubin, bytes) or not cubin:
        message = "Triton compilation did not produce a non-empty CUBIN"
        raise RuntimeError(message)
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=cubin,
        function=compiled.name,
        parameters=parameters,
        grid=grid,
        block=block,
        dynamic_shared_memory_bytes=compiled.metadata.shared,
    )
