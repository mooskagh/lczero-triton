import pytest
import torch
from lc0ex import ExecutableBuilder
from lczero_triton.bt4.kernels._autotune import active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.fused_attention import (
    FusedAttentionSpecialization,
    fused_attention,
    _fused_attention_kernel,
)
from lczero_triton.bt4.kernels.batched_matmul import _qk_kernel, _attention_v_kernel
from lczero_triton.bt4.kernels.softmax_64 import _softmax_64_kernel
import triton


@pytest.mark.gpu
def test_fused_attention_matches_reference() -> None:
    batch = 2
    heads = 32
    head_depth = 32
    model_width = heads * head_depth
    square_count = 64
    batch_count = batch * heads

    queries = torch.randn(batch * square_count, model_width, dtype=torch.float16, device="cuda")
    keys = torch.randn(batch * square_count, model_width, dtype=torch.float16, device="cuda")
    values = torch.randn(batch * square_count, model_width, dtype=torch.float16, device="cuda")
    smolgen = torch.randn(batch_count, square_count, square_count, dtype=torch.float16, device="cuda")
    scale = torch.tensor([0.17677], dtype=torch.float16, device="cuda")

    # Reference using separate kernels
    qk_sep = torch.empty(batch_count * square_count * square_count, dtype=torch.float16, device="cuda")
    out_sep = torch.empty(batch * square_count * model_width, dtype=torch.float16, device="cuda")
    grid_qk = (triton.cdiv(64, 64), triton.cdiv(64, 64), batch_count)
    _qk_kernel.fn[grid_qk](
        qk_sep, queries, keys, scale, 0, batch_count, heads, 64, 64, head_depth, 64 * 64, 64, 64, 32,
        num_warps=8, num_stages=3
    )
    _softmax_64_kernel.fn[(batch_count * 64 // 8,)](
        qk_sep, qk_sep, smolgen, batch_count * 64, 8, num_warps=4
    )
    grid_att_v = (triton.cdiv(64, 64), triton.cdiv(32, 32), batch_count)
    _attention_v_kernel.fn[grid_att_v](
        out_sep, qk_sep, values, batch_count, heads, 64, head_depth, 64, 64, 32, 32,
        num_warps=4, num_stages=4
    )

    out_fused = torch.empty(batch * square_count * model_width, dtype=torch.float16, device="cuda")
    _fused_attention_kernel.fn[(batch_count,)](
        out_fused, queries, keys, values, smolgen, scale, batch_count, model_width, head_depth, heads, head_depth, num_warps=4
    )

    diff = (out_fused - out_sep).abs().max().item()
    assert diff < 0.01

    builder = ExecutableBuilder()
    program = builder.program(name="main")
    cache = KernelCache(builder)

    out_buf = program.buffer(
        name="out", shape=(batch * square_count, model_width), dtype=1, writable=True
    )
    q_buf = program.buffer(name="q", shape=(batch * square_count, model_width), dtype=1)
    k_buf = program.buffer(name="k", shape=(batch * square_count, model_width), dtype=1)
    v_buf = program.buffer(name="v", shape=(batch * square_count, model_width), dtype=1)
    smolgen_buf = program.buffer(name="smolgen", shape=(batch_count, 64, 64), dtype=1)
    scale_buf = program.persistent_buffer(name="scale", shape=(1,), dtype=1)

    fused_attention(
        program,
        cache,
        out_buf,
        q_buf,
        k_buf,
        v_buf,
        smolgen_buf,
        scale_buf,
        FusedAttentionSpecialization(
            batch_count=batch_count,
            model_width=model_width,
            head_depth=head_depth,
            heads_per_sample=heads,
            architecture=active_architecture(),
        ),
    )
    # Check that it builds
    builder.build()
