"""Autotuned FP16 row-major matrix multiplication kernel."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex.triton_module_compiler import write_module

_POINTER = 2

_MATMUL_CONFIGS = (
    triton.Config(
        {"block_m": 32, "block_n": 32, "block_k": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"block_m": 64, "block_n": 32, "block_k": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"block_m": 64, "block_n": 64, "block_k": 32},
        num_warps=8,
        num_stages=3,
    ),
)


@triton.autotune(
    configs=list(_MATMUL_CONFIGS),
    key=["m", "n", "k"],
    cache_results=True,
)
@triton.jit
def _matmul_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    """Compute one tile of a contiguous row-major matrix multiplication."""
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    a_pointers = a_ptr + offsets_m[:, None] * k + offsets_k[None, :]
    b_pointers = b_ptr + offsets_k[:, None] * n + offsets_n[None, :]
    accumulator = tl.zeros((block_m, block_n), tl.float32)

    for k_offset in range(0, k, block_k):
        a = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < m) & (k_offset + offsets_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_pointers,
            mask=(k_offset + offsets_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_pointers += block_k
        b_pointers += block_k * n

    c_pointers = c_ptr + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(
        c_pointers,
        accumulator.to(tl.float16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def _grid(config: Mapping[str, object], m: int, n: int) -> tuple[int, int, int]:
    """Return the launch grid for the selected autotuning configuration."""
    block_m = cast("int", config["block_m"])
    block_n = cast("int", config["block_n"])
    return (triton.cdiv(m, block_m), triton.cdiv(n, block_n), 1)


def compile_matmul(
    m: int,
    n: int,
    k: int,
    output_basename: str | PathLike[str],
) -> Path:
    """Compile one specialized matmul and write its module artifact."""
    a = torch.empty((m, k), dtype=torch.float16, device="cuda")
    b = torch.empty((k, n), dtype=torch.float16, device="cuda")
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")

    def grid(config: Mapping[str, object]) -> tuple[int, int, int]:
        return _grid(config, m, n)

    previous_timings = (
        _matmul_kernel.configs_timings
        if hasattr(_matmul_kernel, "configs_timings")
        else None
    )
    compiled = _matmul_kernel[grid](a, b, c, m, n, k)
    timings = (
        _matmul_kernel.configs_timings
        if hasattr(_matmul_kernel, "configs_timings")
        else None
    )
    if timings is not None and timings is not previous_timings:
        print("Matmul autotuning results:")  # noqa: T201
        for config, timing in timings.items():
            print(f"  {config}: {timing[0]:.3f} ms (p50)")  # noqa: T201
        print(  # noqa: T201
            "Selected configuration: "
            f"{_matmul_kernel.best_config} "
            f"({timings[_matmul_kernel.best_config][0]:.3f} ms p50)"
        )
    resolved_grid = grid(_matmul_kernel.best_config.kwargs)
    return write_module(
        output_basename,
        compiled,
        grid=resolved_grid,
        parameters=(_POINTER, _POINTER, _POINTER),
    )
