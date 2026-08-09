"""Tests for manifest emission without requiring a GPU compilation."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import text_format
from lc0ex.proto import lc0ex_pb2, module_manifest_pb2
from lc0ex.triton_module_compiler import write_module

_SHARED_MEMORY_BYTES = 64


@dataclass(slots=True)
class _FakeTarget:
    backend: str
    arch: int | str
    warp_size: int


@dataclass(slots=True)
class _FakeMetadata:
    num_warps: int
    shared: int
    target: _FakeTarget


@dataclass(slots=True)
class _FakeCompiledKernel:
    asm: Mapping[str, bytes | str]
    metadata: _FakeMetadata
    name: str


def test_write_module_emits_artifacts_and_manifest(tmp_path: Path) -> None:
    """The compiler writer emits the complete linker contract."""
    compiled = _FakeCompiledKernel(
        name="kernel_exported",
        asm={
            "ttir": "fake ttir",
            "ttgir": "fake ttgir",
            "llir": "fake llir",
            "ptx": "fake ptx",
            "cubin": b"fake cubin",
        },
        metadata=_FakeMetadata(
            num_warps=4,
            shared=_SHARED_MEMORY_BYTES,
            target=_FakeTarget(backend="cuda", arch=120, warp_size=32),
        ),
    )
    output_basename = tmp_path / "module"
    manifest_path = write_module(
        output_basename,
        compiled,
        name="module",
        grid=(2, 3, 1),
        parameters=(2, 2, 2),
    )

    manifest = module_manifest_pb2.ModuleManifest()
    text_format.Parse(manifest_path.read_text(encoding="utf-8"), manifest)
    kernel = manifest.kernels[0]

    assert manifest.name == "module"
    assert manifest_path == tmp_path / "module.manifest"
    assert manifest.binary_path == "module.cubin"
    assert manifest.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert manifest.target.vendor == lc0ex_pb2.Target.VENDOR_NVIDIA
    assert manifest.target.architecture == "sm_120"
    assert kernel.name == "module"
    assert kernel.function == "kernel_exported"
    assert tuple(kernel.parameters) == (2, 2, 2)
    assert tuple(kernel.grid) == (2, 3, 1)
    assert tuple(kernel.block) == (128, 1, 1)
    assert kernel.dynamic_shared_memory_bytes == _SHARED_MEMORY_BYTES
    assert (tmp_path / "module.ttir").read_text(encoding="utf-8") == "fake ttir"
    assert (tmp_path / "module.ttgir").read_text(encoding="utf-8") == "fake ttgir"
    assert (tmp_path / "module.llir").read_text(encoding="utf-8") == "fake llir"
    assert (tmp_path / "module.ptx").read_text(encoding="utf-8") == "fake ptx"
    assert (tmp_path / "module.cubin").read_bytes() == b"fake cubin"


def test_write_module_appends_extensions_to_basename(
    tmp_path: Path,
) -> None:
    """Artifact extensions are appended without stripping the basename."""
    compiled = _FakeCompiledKernel(
        name="kernel_exported",
        asm={"cubin": b"fake cubin"},
        metadata=_FakeMetadata(
            num_warps=4,
            shared=0,
            target=_FakeTarget(backend="cuda", arch=120, warp_size=32),
        ),
    )

    manifest_path = write_module(
        tmp_path / "module.v1",
        compiled,
        name="module",
        grid=(2, 3, 1),
        parameters=(2, 2, 2),
    )

    assert manifest_path == tmp_path / "module.v1.manifest"
    assert (tmp_path / "module.v1.cubin").read_bytes() == b"fake cubin"
