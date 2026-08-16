# lc0ex Backend Benchmark

## Finding

Measured on 2026-08-15 with an NVIDIA GeForce RTX 5090 (compute capability
12.0), using `BT4-1024x15x32h-swa-6147500.pb.gz`, batch size 169, 100
benchmark batches, and one benchmark thread:

| Backend | Average inference time | Throughput |
| --- | ---: | ---: |
| `cuda-fp16` | 12.7112 ms | 13,295.4 samples/s |
| `lc0ex-cuda` | 16.8953 ms | 10,002.8 samples/s |

The `lc0ex-cuda` runtime is 1.329x slower per inference, or approximately 33%
slower. Its throughput is 0.752x the CUDA FP16 baseline, approximately 25%
lower. The absolute difference is 4.1841 ms per batch.

`backendbench` reports inference samples per second, not chess search nodes per
second. Backend initialization, weight conversion/upload, and the warmup run
are outside the timed benchmark loop.

## Build

Build the release binary with both the plain CUDA backend and the lc0ex runtime:

```bash
PATH=/opt/cuda/bin:$PATH ./submodules/lc0/build.sh release \
    -Dlc0=true \
    -Dlc0ex-runtime=true \
    -Dbuild_backends=true \
    -Dplain_cuda=true \
    -Dcudnn=false \
    -Dgtest=false \
    -Dcc_cuda=120
```

If the lc0ex artifact is missing, regenerate it:

```bash
uv run --frozen --package lczero-triton lczero-triton graph \
    --network /home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz \
    --output /tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex \
    --batch-size 169
```

## Run

Run the CUDA FP16 baseline:

```bash
./submodules/lc0/build/release/lc0 backendbench \
    --weights=/home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz \
    --backend=cuda-fp16 \
    --backend-opts='gpu=0' \
    --threads=1 \
    --batches=100 \
    --start-batch-size=169 \
    --max-batch-size=169 \
    --batch-step=1
```

Run the lc0ex CUDA runtime with the same benchmark settings:

```bash
./submodules/lc0/build/release/lc0 backendbench \
    --weights=/home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz \
    --backend=lc0ex-cuda \
    --backend-opts='lc0ex=/tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex,gpu=0,concurrency=1' \
    --threads=1 \
    --batches=100 \
    --start-batch-size=169 \
    --max-batch-size=169 \
    --batch-step=1
```

Keep `--threads=1`, `--batches`, and the fixed batch-size flags identical when
comparing runs. Use `concurrency=1` for the lc0ex run so the sequential
benchmark does not permit overlapping executions.

The lc0ex run may print warnings for ONNX shape, slice, and reshape
initializers without corresponding lc0ex buffers. Those constants are not
consumed as external graph buffers and the warnings do not indicate a failed
benchmark when a throughput line is produced.

## Investigation Findings

Measured on 2026-08-16 on the same RTX 5090, using the same network and batch
size. The lc0ex artifact was regenerated before profiling. No source changes
were made during this investigation.

### Reproduction

An unprofiled repeat produced:

| Backend | Average inference time | Throughput |
| --- | ---: | ---: |
| `cuda-fp16` | 12.5951 ms | 13,417.9 samples/s |
| `lc0ex-cuda` | 16.6364 ms | 10,158.4 samples/s |

The measured difference was 4.0413 ms per batch, with lc0ex approximately
32.1% slower. The result agrees with the original benchmark finding.

### Profiling Method

Nsight Systems traces used 20 benchmark batches at batch size 169. The warmup
and initialization are present in the trace, but the steady-state inference
regions were analyzed separately where possible.

Profiler outputs:

- `/tmp/lc0-cuda-fp16.nsys-rep`
- `/tmp/lc0-lc0ex-cuda.nsys-rep`
- `/tmp/lc0ex.perf.data`

### GPU Versus Host

The slowdown is predominantly GPU execution:

| Measurement | `cuda-fp16` | `lc0ex-cuda` |
| --- | ---: | ---: |
| Captured GPU kernels | 6,893 | 7,371 |
| Total GPU kernel time | 247.26 ms | 339.98 ms |
| Average steady-state kernel time | approximately 12.34 ms | approximately 16.19 ms |

The lc0ex graph contains 351 nodes and launches one CUDA kernel for each node.
The plain CUDA backend emitted approximately 308 steady-state GPU kernels per
batch. This launch-count difference is real, but the launch API time was not
the primary cause: lc0ex spent 13.66 ms in `cuLaunchKernel` calls across the
trace, compared with approximately 15.14 ms across the plain CUDA launch APIs.

The timed lc0ex input and output copies accounted for approximately 0.13 ms of
host API time per inference. Transfer time is therefore also secondary to
kernel execution.

Relevant runtime paths:

- Per-node launch loop: `submodules/lc0/src/neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.cc:430-446`
- Per-inference execution setup: `submodules/lc0/src/neural/backends/lc0ex-cuda/runtime/lc0ex_cuda.cc:372-427`
- Input, execution, synchronization, and output boundary: `submodules/lc0/src/neural/backends/lc0ex-cuda/network_lc0ex_cuda.cc:549-610`

### Dense GEMM Performance

Dense GEMMs are the largest kernel-level difference. The lc0ex
`_matmul_kernel` consumed 279.29 ms in the trace. The plain CUDA backend's
dominant `nvjet` GEMM kernels consumed 170.82 ms, with another 21.83 ms in
Cutlass GEMM kernels.

The Triton dense kernel uses an explicit FP32 accumulator and converts to FP16
on store:

- `packages/lczero-triton/src/lczero_triton/bt4/kernels/matmul.py:80-102`

The plain CUDA path uses tuned cuBLAS GEMMs, including tensor-operation math.
This is a different kernel implementation and precision path, and is the
primary candidate for the remaining performance gap after fusion differences.

### QKV Fusion

The clearest individual source-level difference is encoder QKV projection.

The lc0ex graph emits separate Q, K, and V GEMMs and bias operations:

- `packages/lczero-triton/src/lczero_triton/bt4/network.py:950-971`

The plain CUDA backend packs the three projections into one strided-batched
operation with batch count three:

- `submodules/lc0/src/neural/backends/cuda/layers.cc:1745-1751`

Measured per encoder:

| QKV implementation | Time |
| --- | ---: |
| Three lc0ex Triton GEMMs | approximately 316 us |
| One packed CUDA GEMM | approximately 198 us |

The difference is approximately 118 us per encoder, or approximately 1.8 ms
per inference across 15 encoders. Nsight Compute also showed the selected
Triton GEMM using 130 registers, 36,864 bytes of dynamic shared memory, and
16.67% theoretical occupancy. The CUDA packed kernel used a different fused
tile schedule and handled all three projections in one launch.

### Kernels That Are Not Primary Causes

Some non-GEMM lc0ex kernels were faster than the CUDA equivalents in the
trace:

- lc0ex softmax: 11.20 ms total, CUDA softmax: 15.21 ms total.
- lc0ex layer normalization: 13.25 ms total, CUDA layer normalization: 22.11 ms total.

Bias kernels were moderately slower in lc0ex, but their difference was much
smaller than the dense GEMM difference.

### Policy Projection Fusion

The policy head has the same structural issue as encoder QKV. lc0ex emits
separate policy Q and K projections:

- `packages/lczero-triton/src/lczero_triton/bt4/network.py:1253-1278`

The CUDA backend packs them into one batch-of-two operation:

- `submodules/lc0/src/neural/backends/cuda/layers.cc:1940-1946`

This has not yet been isolated with a separate experiment. It should be
measured after the dense GEMM and encoder QKV measurements so the effects are
not conflated.

### Immediate QKV Implementation

The first fusion step keeps the existing named Q, K, and V buffers independent.
One Triton launch computes all three projections through separate weight and
output pointers, and a second launch applies the three independent biases. This
removes the six per-encoder projection and bias launches without depending on a
packed allocation layout.

The first 500-batch comparison measured `12.5536 ms` for the fused artifact
and `12.4647 ms` for the FP16-accumulator three-GEMM artifact. This is not a
decisive whole-network gain yet; isolated projection timing was approximately
`0.166 ms` for the 3D fused launch versus `0.182 ms` for three GEMM launches.

### Deferred Buffer Layout

To match the CUDA implementation more closely in a later step, the executable
needs an explicit logical-view layout rather than relying on declaration order:

- Each encoder's Q/K/V weights should have one contiguous FP16 backing range in
  `[Q][K][V]` order, with a known element stride between projections.
- The three bias vectors should have an analogous contiguous `[Q][K][V]`
  backing range.
- Named ONNX initializers should be able to refer to subranges of those backing
  ranges, preserving upload and fingerprint behavior.
- Kernel arguments should be able to express a base allocation plus byte
  offsets, shapes, and strides, so independent graph buffers can become views
  without making adjacency an accidental property of buffer declaration order.

## Follow-Up Investigations

These are intentionally separate work items. Each should be benchmarked before
and after any implementation change.

1. Compare Triton dense GEMM configurations and accumulator choices for the
   dominant BT4 shapes against cuBLAS.
2. Benchmark the separate-pointer QKV implementation and measure the
   encoder-stage improvement independently.
3. Evaluate a packed policy Q/K implementation and measure the policy-stage
   improvement independently.
4. Design and implement the explicit packed-buffer layout described above.
5. Only after the kernel work, measure execution-argument reuse, pinned host
   buffers, and asynchronous copies as runtime cleanup opportunities.
