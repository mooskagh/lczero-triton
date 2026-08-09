"""Tests for loading compiler-produced module artifacts."""

from pathlib import Path

import pytest
from lc0ex import ExecutableBuilder, load_module

MANIFEST = """
format: 1
name: "matmul_module"
binary_path: "module.cubin"
binary_format: FORMAT_CUBIN
target {
  vendor: VENDOR_NVIDIA
  architecture: "sm_120"
}
kernels {
  name: "matmul"
  function: "matmul_exported"
  parameters: PARAMETER_TYPE_POINTER
  parameters: PARAMETER_TYPE_POINTER
  parameters: PARAMETER_TYPE_POINTER
  grid: 2
  grid: 3
  grid: 1
  block: 128
  block: 1
  block: 1
}
"""


def _write_manifest(directory: Path, text: str = MANIFEST) -> Path:
    """Write a small manifest fixture and its paired binary."""
    (directory / "module.cubin").write_bytes(b"fake cubin")
    path = directory / "manifest.textproto"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_module_resolves_binary_and_kernel_metadata(tmp_path: Path) -> None:
    """A manifest loads its relative binary and preserves launch metadata."""
    module = load_module(_write_manifest(tmp_path))

    assert module.name == "matmul_module"
    assert module.binary_path == tmp_path / "module.cubin"
    assert module.target_architecture == "sm_120"
    assert module.kernels[0].name == "matmul"
    assert module.kernels[0].artifact.binary_data == b"fake cubin"
    assert module.kernels[0].artifact.grid == (2, 3, 1)
    assert module.kernels[0].artifact.block == (128, 1, 1)


def test_add_module_registers_multiple_kernels_from_one_binary(tmp_path: Path) -> None:
    """One module can expose several logical kernels sharing one binary."""
    second_kernel = MANIFEST.replace(
        '  name: "matmul"',
        '  name: "matmul_second"',
        1,
    )
    manifest = _write_manifest(
        tmp_path,
        MANIFEST + "\n" + second_kernel[second_kernel.index("kernels {") :],
    )

    builder = ExecutableBuilder().add_module(manifest)
    executable = builder.build()

    assert len(executable.binaries) == 1
    assert [kernel.name for kernel in executable.kernels] == [
        "matmul",
        "matmul_second",
    ]


def test_load_module_rejects_binary_path_escape(tmp_path: Path) -> None:
    """A manifest cannot make the linker read outside its artifact directory."""
    manifest = _write_manifest(tmp_path)
    manifest.write_text(
        MANIFEST.replace("module.cubin", "../module.cubin"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not in the subpath"):
        load_module(manifest)
