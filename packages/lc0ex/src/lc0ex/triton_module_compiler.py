"""Convert Triton output into executable-linker artifacts."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import triton.backends.nvidia.compiler as nv_compiler

from lc0ex.kernel_builder import KernelArtifact
from lc0ex.proto import lc0ex_pb2


def _ptxas_path(architecture: str) -> str:
    if hasattr(nv_compiler, "get_ptxas"):
        return str(nv_compiler.get_ptxas(int(architecture[3:].removesuffix("a"))).path)
    if hasattr(nv_compiler, "_path_to_binary"):
        return str(nv_compiler._path_to_binary("ptxas")[0])  # noqa: SLF001
    msg = f"Failed to locate ptxas for architecture {architecture}"
    raise RuntimeError(msg)


_TRITON_SCRATCH_PARAMETERS = (
    lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
    lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
)


def compile_ptx(ptx: str, *, architecture: str) -> bytes:
    """Assemble PTX for *architecture* with Triton's configured ptxas."""
    with TemporaryDirectory(prefix="lc0ex-ptx-") as directory:
        source_path = Path(directory) / "module.ptx"
        cubin_path = source_path.with_suffix(".cubin")
        source_path.write_text(ptx, encoding="utf-8")
        subprocess.run(  # noqa: S603  # ptxas path is selected by Triton.
            (
                _ptxas_path(architecture),
                f"--gpu-name={architecture}",
                str(source_path),
                "--output-file",
                str(cubin_path),
            ),
            check=True,
        )
        return cubin_path.read_bytes()


def _runtime_ns_from_autotuner(autotuner: object) -> int | None:
    """Extract the autotuned runtime of the best config in nanoseconds."""
    configs_timings = getattr(autotuner, "configs_timings", None)
    best_config = getattr(autotuner, "best_config", None)
    if configs_timings and best_config:
        timing = configs_timings.get(best_config)
        if timing is not None:
            ms = timing[0] if isinstance(timing, (list, tuple)) else timing
            return int(float(ms) * 1e6)
    return None


def artifact_from_triton(
    compiled: Any,
    *,
    grid: tuple[int, int, int],
    parameters: tuple[lc0ex_pb2.ParameterType, ...],
    runtime_ns: int | None = None,
    autotuner: object = None,
) -> KernelArtifact:
    """Convert one compiled Triton entry point into a linker artifact."""
    if runtime_ns is None and autotuner is not None:
        runtime_ns = _runtime_ns_from_autotuner(autotuner)

    block = (
        compiled.metadata.num_warps * compiled.metadata.target.warp_size,
        1,
        1,
    )
    if (
        getattr(compiled.metadata, "global_scratch_size", 0) != 0
        or getattr(compiled.metadata, "profile_scratch_size", 0) != 0
    ):
        message = "Triton scratch allocations are not supported in lc0ex artifacts."
        raise ValueError(message)
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=compiled.asm["cubin"],
        function=compiled.name,
        parameters=parameters + _TRITON_SCRATCH_PARAMETERS,
        grid=grid,
        block=block,
        dynamic_shared_memory_bytes=compiled.metadata.shared,
        runtime_ns=runtime_ns,
    )
