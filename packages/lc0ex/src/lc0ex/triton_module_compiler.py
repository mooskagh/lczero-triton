"""Triton compilation and module-manifest emission."""

from os import PathLike
from pathlib import Path
from typing import Any

from google.protobuf import text_format

from lc0ex.proto import lc0ex_pb2, module_manifest_pb2

Grid = tuple[int, int, int]

_MANIFEST_FORMAT = 1


def write_module(
    output_basename: str | PathLike[str],
    compiled: Any,
    *,
    name: str,
    grid: Grid,
    parameters: tuple[int, ...],
) -> Path:
    """Write Triton artifacts and a manifest using the given basename."""
    block = (
        compiled.metadata.num_warps * compiled.metadata.target.warp_size,
        1,
        1,
    )
    module = module_manifest_pb2.ModuleManifest(
        format=_MANIFEST_FORMAT,
        name=name,
        binary_path=f"{Path(output_basename).name}.cubin",
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
    )
    module.target.vendor = lc0ex_pb2.Target.VENDOR_NVIDIA
    module.target.architecture = f"sm_{compiled.metadata.target.arch}"
    module.kernels.add(
        name=name,
        function=compiled.name,
        parameters=parameters,
        grid=grid,
        block=block,
        dynamic_shared_memory_bytes=compiled.metadata.shared,
    )

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
