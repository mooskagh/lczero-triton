"""Load and validate compiler-produced module manifests."""

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import NoReturn

from google.protobuf import text_format

from lc0ex.kernel_builder import KernelArtifact
from lc0ex.proto import lc0ex_pb2, module_manifest_pb2

_MANIFEST_FORMAT = 1
_CUBIN = lc0ex_pb2.Binary.FORMAT_CUBIN
_NVIDIA = lc0ex_pb2.Target.VENDOR_NVIDIA
_U32 = lc0ex_pb2.PARAMETER_TYPE_U32
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_LAUNCH_DIMENSION_COUNT = 3


@dataclass(frozen=True, slots=True)
class ModuleArtifact:
    """A validated module loaded from a manifest and its binary file."""

    binary_path: Path
    target_vendor: lc0ex_pb2.Target.Vendor
    target_architecture: str
    kernels: tuple[KernelArtifact, ...]


def load_module(manifest_path: str | PathLike[str]) -> ModuleArtifact:
    """Load a textproto module manifest and its relative binary."""
    path = Path(manifest_path)
    manifest = module_manifest_pb2.ModuleManifest()
    text_format.Parse(path.read_text(encoding="utf-8"), manifest)

    _validate_manifest(manifest, path)
    binary_path = _resolve_binary_path(path, manifest.binary_path)
    binary_data = binary_path.read_bytes()
    if not binary_data:
        message = f"module binary {binary_path} is empty"
        raise ValueError(message)

    binary_format = _binary_format(manifest.binary_format)
    kernels = tuple(
        KernelArtifact(
            binary_format=binary_format,
            binary_data=binary_data,
            function=kernel.function,
            parameters=tuple(
                _parameter_type(parameter) for parameter in kernel.parameters
            ),
            grid=_normalize_dimensions("grid", kernel.function, kernel.grid),
            block=_normalize_dimensions("block", kernel.function, kernel.block),
            dynamic_shared_memory_bytes=kernel.dynamic_shared_memory_bytes,
        )
        for kernel in manifest.kernels
    )
    return ModuleArtifact(
        binary_path=binary_path,
        target_vendor=manifest.target.vendor,
        target_architecture=manifest.target.architecture,
        kernels=kernels,
    )


def _validate_manifest(
    manifest: module_manifest_pb2.ModuleManifest,
    manifest_path: Path,
) -> None:
    """Validate constraints that cannot be represented by protobuf."""
    if manifest.format != _MANIFEST_FORMAT:
        message = f"unsupported module manifest format: {manifest.format}"
        raise ValueError(message)
    _validate_manifest_header(manifest)
    _validate_kernels(manifest, manifest_path)


def _validate_manifest_header(manifest: module_manifest_pb2.ModuleManifest) -> None:
    """Validate the scalar fields shared by all module kernels."""
    if manifest.target.vendor != _NVIDIA:
        message = f"unsupported module target vendor: {manifest.target.vendor}"
        _raise(message)
    if not manifest.target.architecture:
        _raise("module target architecture cannot be empty")


def _validate_kernels(
    manifest: module_manifest_pb2.ModuleManifest,
    manifest_path: Path,
) -> None:
    """Validate the module's exported kernels."""
    if not manifest.kernels:
        message = f"module manifest {manifest_path} contains no kernels"
        _raise(message)

    functions: set[str] = set()
    for kernel in manifest.kernels:
        if not kernel.function:
            _raise("module kernel function cannot be empty")
        if kernel.function in functions:
            message = f"module function {kernel.function!r} is declared more than once"
            _raise(message)


def _normalize_dimensions(
    name: str,
    kernel: str,
    dimensions: Sequence[int],
) -> tuple[int, int, int]:
    """Validate and normalize a three-dimensional launch vector."""
    values = tuple(dimensions)
    if len(values) != _LAUNCH_DIMENSION_COUNT or any(value == 0 for value in values):
        message = f"module kernel {kernel!r} {name} must contain three positive values"
        _raise(message)
    return (values[0], values[1], values[2])


def _resolve_binary_path(manifest_path: Path, binary_path: str) -> Path:
    """Resolve a manifest binary while preventing directory traversal."""
    relative_path = Path(binary_path)
    if relative_path.is_absolute():
        _raise("module binary_path must be relative to the manifest")

    manifest_directory = manifest_path.resolve().parent
    resolved_path = (manifest_directory / relative_path).resolve()
    resolved_path.relative_to(manifest_directory)
    return resolved_path


def _binary_format(value: int) -> lc0ex_pb2.Binary.Format:
    """Translate the manifest binary format to the executable schema."""
    if value == _CUBIN:
        return lc0ex_pb2.Binary.FORMAT_CUBIN
    message = f"unsupported module binary format: {value}"
    raise ValueError(message)


def _parameter_type(value: int) -> lc0ex_pb2.ParameterType:
    """Translate a manifest ABI parameter type to the executable schema."""
    if value == _U32:
        return lc0ex_pb2.PARAMETER_TYPE_U32
    if value == _POINTER:
        return lc0ex_pb2.PARAMETER_TYPE_POINTER
    message = f"unsupported module parameter type: {value}"
    raise ValueError(message)


def _raise(message: str) -> NoReturn:
    """Raise a validation error without duplicating exception boilerplate."""
    raise ValueError(message)
