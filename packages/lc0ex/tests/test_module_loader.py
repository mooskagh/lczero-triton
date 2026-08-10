"""Tests for loading compiler-produced module artifacts."""

from pathlib import Path

import pytest
from lc0ex import ExecutableBuilder, load_module

MANIFEST = """
format: 1
binary_path: "module.cubin"
binary_format: FORMAT_CUBIN
target {
  vendor: VENDOR_NVIDIA
  architecture: "sm_120"
}
kernels {
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
EXPECTED_KERNEL_COUNT = 2


def _write_manifest(directory: Path, text: str = MANIFEST) -> Path:
    """Write a small manifest fixture and its paired binary."""
    (directory / "module.cubin").write_bytes(b"fake cubin")
    path = directory / "manifest.textproto"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_module_resolves_binary_and_kernel_metadata(tmp_path: Path) -> None:
    """A manifest loads its relative binary and preserves launch metadata."""
    module = load_module(_write_manifest(tmp_path))

    assert module.binary_path == tmp_path / "module.cubin"
    assert module.target_architecture == "sm_120"
    assert module.kernels[0].function == "matmul_exported"
    assert module.kernels[0].binary_data == b"fake cubin"
    assert module.kernels[0].grid == (2, 3, 1)
    assert module.kernels[0].block == (128, 1, 1)


def test_add_module_registers_multiple_kernels_from_one_binary(tmp_path: Path) -> None:
    """One module can expose several logical kernels sharing one binary."""
    second_kernel = MANIFEST.replace(
        '  function: "matmul_exported"',
        '  function: "matmul_second_exported"',
        1,
    )
    manifest = _write_manifest(
        tmp_path,
        MANIFEST + "\n" + second_kernel[second_kernel.index("kernels {") :],
    )

    builder = ExecutableBuilder()
    handles = builder.add_module(manifest)
    executable = builder.build()

    assert len(handles) == EXPECTED_KERNEL_COUNT
    assert len(executable.binaries) == 1
    assert [kernel.function for kernel in executable.kernels] == [
        "matmul_exported",
        "matmul_second_exported",
    ]


def test_load_module_exposes_symbols_from_the_module_binary(tmp_path: Path) -> None:
    """Module symbols are loaded beside ordinary kernel exports."""
    manifest = _write_manifest(
        tmp_path,
        MANIFEST + '\nsymbols { symbol_name: "mapping_table" }\n',
    )

    module = load_module(manifest)

    assert len(module.symbols) == 1
    assert module.symbols[0].symbol_name == "mapping_table"
    assert module.symbols[0].binary_data == b"fake cubin"


def test_load_module_rejects_binary_path_escape(tmp_path: Path) -> None:
    """A manifest cannot make the linker read outside its artifact directory."""
    manifest = _write_manifest(tmp_path)
    manifest.write_text(
        MANIFEST.replace("module.cubin", "../module.cubin"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not in the subpath"):
        load_module(manifest)
