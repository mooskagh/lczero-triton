"""Triton compilation and module-manifest emission."""

import re
import subprocess
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from google.protobuf import text_format
from triton.backends.nvidia.compiler import get_ptxas  # type: ignore[import-untyped]

from lc0ex.proto import lc0ex_pb2, module_manifest_pb2

Grid = tuple[int, int, int]

_MANIFEST_FORMAT = 1
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


def write_module(
    output_basename: str | PathLike[str],
    compiled: Any,
    *,
    grid: Grid,
    parameters: tuple[int, ...],
    symbols: tuple[str, ...] = (),
) -> Path:
    """Write Triton artifacts and a manifest using the given basename."""
    block = (
        compiled.metadata.num_warps * compiled.metadata.target.warp_size,
        1,
        1,
    )
    module = module_manifest_pb2.ModuleManifest(
        format=_MANIFEST_FORMAT,
        binary_path=f"{Path(output_basename).name}.cubin",
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
    )
    module.target.vendor = lc0ex_pb2.Target.VENDOR_NVIDIA
    module.target.architecture = f"sm_{compiled.metadata.target.arch}"
    module.kernels.add(
        function=compiled.name,
        parameters=parameters,
        grid=grid,
        block=block,
        dynamic_shared_memory_bytes=compiled.metadata.shared,
    )
    for symbol_name in symbols:
        module.symbols.add(symbol_name=symbol_name)

    output_path = Path(output_basename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for format_name, artifact in compiled.asm.items():
        artifact_path = output_path.parent / f"{output_path.name}.{format_name}"
        if isinstance(artifact, bytes):
            artifact_path.write_bytes(artifact)
        else:
            artifact_path.write_text(artifact, encoding="utf-8")
    manifest_path = output_path.parent / f"{output_path.name}.manifest"
    manifest_path.write_text(
        text_format.MessageToString(module),
        encoding="utf-8",
    )
    return manifest_path
