# BT4 Triton Port Plan

## Scope

Port the active evaluation graph of
`BT4-1024x15x32h-swa-6147500.pb.gz` to statically compiled Triton kernels and
serialize it as an `lc0ex` graph.

The graph builder supports one or more fixed batch-size programs and the default
active heads:

- `vanilla` attention policy
- `winner` WDL value
- moves-left

The artifact contains the graph and CUBINs but not the weights. Graph
construction reads the network protobuf and traverses its active evaluation
path directly. Learned weights and consumed graph constants are declared as
named persistent buffers at their points of use. Their names must exactly match
the final initializer names produced by `leela2onnx`. Learned matrices use the
ONNX `[K, N]` layout and FP16 dtype.

This milestone does not include LC0 backend registration.

## Research Findings

- LC0's loader upgrades the network from its raw older attention-body format to
  `INPUT_EMBEDDING_PE_DENSE` because it contains multihead policy and value
  structures.
- The body has 15 encoder blocks, width 1024, 32 heads, and head depth 32.
- Every encoder block uses Smolgen.
- The body and embedding FFN width is 1536.
- The learned positional preprocessing maps 768 input values to 32768 values,
  reshaped as 512 channels for each of 64 squares.
- The input embedding includes both multiplicative and additive per-square
  gates.
- The user's initial CUDA kernel list omitted `input_gating_kernel<half>`, which
  is active in this network and must also be ported.
- QKV head splitting and merging do not require materialized transposes. The
  CUDA backend implements them through pointer offsets and GEMM strides; Triton
  can calculate those offsets directly.
- The CUDA backend returns raw FP32 WDL logits and performs the three-way
  softmax on the host. The initial Triton graph should preserve this behavior.
- Persistent weights are executable-global named buffers. Inputs, outputs, and
  scratch are named or raw buffers inside each program's execution allocation.

## Fixed Architecture

| Property | Value |
| --- | ---: |
| Default batch | 169 |
| Squares | 64 |
| Input planes | 112 |
| Dense positional channels | 512 |
| Embedding input width | 624 |
| Body width | 1024 |
| Encoder blocks | 15 |
| Attention heads | 32 |
| Head depth | 32 |
| Encoder and embedding FFN width | 1536 |
| Smolgen compression width | 32 |
| Smolgen hidden width | 256 |
| Smolgen generated width | 8192 |
| Shared Smolgen output width | 4096 |
| Policy width | 1024 |
| Value embedding width | 128 |
| Value hidden width | 128 |
| Moves-left embedding width | 32 |
| Moves-left hidden width | 128 |
| DeepNorm alpha | `(2 * 15) ** -0.25` |

Derived fixed dimensions:

- Token rows: `169 * 64 = 10816`
- Body attention batches: `169 * 32 = 5408`
- Policy pre-map width: `64 * 64 + 8 * 24 = 4288`
- Policy output width: 1858

## External Buffer Contract

### Evaluation interface

| Name | Shape | Dtype | Lifetime |
| --- | --- | --- | --- |
| `/input/plane_masks` | `[169, 112]` | U64 | execution |
| `/input/plane_values` | `[169, 112]` | F32 | execution |
| `/output/policy` | `[169, 1858]` | F32 | execution |
| `/output/wdl` | `[169, 3]` | F32 raw logits | execution |
| `/output/mlh` | `[169, 1]` | F32 | execution |

### Weights

Declare all 409 learned initializers used by the selected graph as persistent
FP16 buffers. Preserve `leela2onnx` names literally, including leading slashes,
asterisks, and repeated `/w/w` suffixes. Examples include:

- `/attn_body/embedding/preprocess/matmul/w`
- `/attn_body/matmul/w`
- `/ip_mul_gate/w`
- `/ip_add_gate/w`
- `/const/smolgen_w`
- `/encoder0/mha/Q/w/w`
- `/encoder14/smolgen/dense2/w/w`
- `/policy/promotion/matmul/w`
- `/value/dense2/matmul/w`
- `/mlh/dense2/matmul/w`

Every learned matrix uses the final ONNX `[K, N]` layout. The external loader is
responsible for decoding `LINEAR16`, transposing like `leela2onnx`, converting
to FP16, validating name/shape/dtype, and uploading each buffer.

### Constants

Expose consumed non-learned constants as external ONNX-named buffers except for
the embedded attention-policy mapping symbol. External constants include:

- `/attn_body/ffn/alpha/w`
- `/encoder{i}/mha/QK/scale/w`
- `/encoder{i}/alpha*input/w`
- `/encoder{i}/ffn/alpha/w`
- `/policy/scale/w`

Do not declare ONNX shape, slice, or reshape constants that no Triton kernel
consumes. Static graph construction makes them unnecessary.

Later, the remaining constants can move into CUDA module symbols produced
through PTX or CUDA rather than Triton.

## Target Source Layout

Remove the current toy files:

```text
packages/lczero-triton/src/lczero_triton/network.py
packages/lczero-triton/src/lczero_triton/kernels/
```

Create:

```text
packages/lczero-triton/src/lczero_triton/bt4/
    __init__.py
    network.py
    _format.py
    kernels/
        __init__.py
        _cache.py
        add_bias_batched.py
        add_vectors.py
        batched_matmul.py
        copy_type_converted.py
        expand_planes.py
        input_gating.py
        layer_norm.py
        matmul.py
        nchw_to_nhwc.py
        policy_map.py
        preprocess_attention_body.py
        promotion_logits.py
        softmax_64.py
```

Use one public kernel-family file per operation. Keep only genuinely shared
activation and compilation helpers in private modules.

## Implementation Order

### 1. Replace the toy package structure

Delete the top-level toy `network.py` and `kernels/` package. Introduce the
`bt4` package, update CLI imports, and replace tests that refer to
`build_matmul_graph`.

Update root configuration so Ruff and strict mypy overrides include
`lczero_triton.bt4.kernels.*` rather than the removed toy module.

Acceptance criteria:

- No imports or tests reference the toy `M`, `N`, `K`, or matmul graph.
- The new package is included automatically in the existing wheel layout.
- Non-GPU checks pass before kernel work begins.

### 2. Build directly from the parsed network

Parse the selected `.pb.gz` network to `net_pb2.Net`, apply LC0-compatible
older-format normalization, and make `bt4/network.py` a function-as-grammar
evaluation traversal. Do not build an intermediate network specification,
transformer IR, initializer manifest, or model configuration.

The public entry point is conceptually:

```python
def build(
    builder: ExecutableBuilder,
    network: net_pb2.Net,
    *,
    batch_sizes: Sequence[int] | None = None,
) -> None:
```

Its call graph follows the active evaluation order:

```text
build
  -> input and embedding
  -> encoder tower
     -> encoder
        -> Smolgen
        -> Q/K/V attention
        -> residual layer norm
        -> FFN and residual layer norm
  -> policy, winner WDL, and moves-left heads
```

Each grammar production derives only its local dimensions from protobuf layer
element counts and its operation semantics. At a learned-weight terminal,
declare the operation's external named FP16 buffer with its canonical `[K, N]`
or vector contract. Do not turn `Buffer` into a tensor type or infer kernel
parameters from buffer metadata. Preserve unusual names such as
`/encoder0/mha/Q/w/w` mechanically from the operation name. Do not declare ONNX
reshape, slice, or transpose constants. The eventual external loader validates
uploaded weight shape and dtype against the serialized contract.

Use one ordinary loop over `weights.encoder`; do not use `range(15)`. A block
may use a different FFN or Smolgen hidden width when its local operations and
compiled kernels support it. It must still satisfy its own residual, attention,
and shared-Smolgen contracts. Activations are format-level choices and must be
resolved at the operation that consumes them.

Keep physical `lc0ex.Buffer` handles opaque. Named external ranges preserve
their canonical shape and dtype solely for runtime serialization; anonymous
temporaries are raw byte ranges with explicit alignment. Kernel dimensions,
dtypes, layouts, strides, activations, and other code-generation decisions are
explicit immutable specialization parameters, not buffer accessors or reshape
views.

Add tests for:

- LC0 older-format normalization used by the selected file
- Direct protobuf-driven traversal and active-head selection
- Exact learned names, shapes, and dtypes for the target integration fixture
- Representative unusual names and ONNX `[K, N]` orientation
- Per-block dimension adaptation and material local architecture constraints
- Unsupported RPE, policy encoders, head formats, and activation variants
- No duplicate external declarations with conflicting metadata

Weight decoding and upload remain separate concerns. The graph builder reads
only protobuf structure, format fields, and encoded layer lengths.

### 3. Define minimal `lc0ex` pointer-source features

Make `Buffer` an opaque logical device-range identity. Do not expose shape,
dtype, layout, stride, or reshape accessors: node arguments are pointers, and
kernel specializations must receive those properties explicitly. Keep named
external buffers as canonical serialization metadata, and create anonymous
execution temporaries from raw byte size and alignment.

`Node.Argument` has exactly one pointer/value source: a runtime parameter, an
allocation location, or a module symbol. An allocation location identifies the
persistent or program execution allocation and an offset:

```proto
message AllocationLocation {
  required AllocationKind kind = 1;
  required uint64 offset = 2;
}
```

Use `Node.Argument.Symbol`, containing `binary_idx` and `symbol_name`, for an
immutable pointer exported from an embedded CUBIN. `ProgramBuilder.call()`
accepts `Buffer | SymbolHandle` pointer arguments. Symbols are not buffers:
they are implicitly readonly and do not participate in allocation planning,
dependencies, or temporary reuse.

Add `SymbolArtifact` and `SymbolHandle` beside kernel artifacts and handles.
Deduplicate the CUBIN bytes of kernels and symbols into the same executable
binary table.

The CUDA runtime resolves each symbol argument with `cuModuleGetGlobal` during
program loading and retains its `CUdeviceptr` as a launch argument. No
symbol-backed `Buffer`, allocation, host upload, or dtype metadata is needed.

Generate the attention-policy mapping table as an initialized CUBIN symbol:

- Build the 1858-entry ONNX gather inverse of `kAttnPolicyMap`.
- Store I32 source indices for the 4288-element attention-policy record.
- Emit PTX with one visible, aligned `.u32` global and assemble it with
  Triton's configured `ptxas` for the selected `sm_*` target.
- Pass its `SymbolHandle` as the immutable mapping-table pointer argument to
  the eventual policy-map kernel.

Keep existing persistent buffers read-only by default. Do not add embedded
weight bytes or initialized persistent allocations.

Acceptance criteria:

- Allocation arguments serialize through `AllocationLocation(kind, offset)`.
- Kernel and symbol exports sharing CUBIN bytes share one `binary_idx`.
- Symbol arguments serialize as `binary_idx` and `symbol_name` and are not
  callable kernels.
- The CUDA runtime builds with `-Dlc0ex-runtime=true` and resolves symbol
  pointers through `cuModuleGetGlobal`.
- The mapping-table symbol contains 1858 unique, in-range attention source
  indices and assembles for the selected target.
- Existing builder dependency and allocation tests continue to pass.

### 4. Add BT4 kernel compilation infrastructure

The per-build kernel cache is already in place in `bt4/kernels/_cache.py`.
It is keyed by the compiler function and a hashable specialization, compiles
on a miss, registers the resulting in-memory `KernelArtifact`, and returns the
registered `KernelHandle` on subsequent hits.

Do not make this a standalone implementation phase. Establish the remaining
compilation pattern with the first Step 5 kernel: compile directly in each
kernel-family module, convert the Triton result with the existing lc0ex
`artifact_from_triton()` boundary, and give each family a frozen specialization
object containing the decisions that affect its generated code, including its
static dimensions, dtype, layout/stride convention, activation or optional
operands, launch configuration, and compilation target. Do not key from buffer
identity, canonical external names, or allocation metadata. Use stable
module-level compiler functions rather than per-call closures so equivalent
operations in separate blocks share the cache entry. Rely on Triton's
persistent compilation cache and enable its persistent autotuning-result cache
for tuned families so editing one kernel does not rebuild or retune unrelated
kernels.

Keep the cache tests for hit reuse and distinct specializations. Prove the
compiler adapter, specialization contract, ABI, target selection, and graph
argument ordering through the first real kernel rather than speculative
generic validation infrastructure. The executable is the compiled artifact;
do not add a separate module-manifest format or filesystem linking path.

### 5. Port layout and simple elementwise kernels

Implement these kernels first because their contracts are isolated and easy to
test numerically:

1. `copyTypeConverted_kernel<float, half>`
2. `expandPlanes_kernel_Fp16_NCHW`
3. `NCHWtoNHWC_kernel<half, half>`
4. `preprocess_for_attention_body_kernel<half>`
5. `input_gating_kernel<half>`
6. `addVectors_kernel<half>`
7. `addBiasBatched_kernel<half>`
8. `policyMap_kernel<half>`

Required semantics:

- Plane expansion reads U64 masks and F32 plane values and writes FP16 NCHW.
- NCHW-to-NHWC extracts the first 12 planes into `[169, 64, 12]`.
- Attention preprocessing converts all 112 planes to token-major order and
  appends the 512 dense positional channels.
- Input gating computes `input * mult + add` using ONNX `[64, 1024]` gate
  layouts. The CUDA source transposes legacy gate storage during indexing, but
  the ONNX-layout buffers do not require that transpose.
- `addVectors` supports NONE, MISH, and RELU paths needed by PE bias, WDL, and
  moves-left.
- `addBiasBatched` supports NONE and MISH. Keep the implementation capable of
  batched bias broadcasting, although separate ONNX Q/K/V buffers initially
  result in separate calls.
- Arithmetic is promoted to FP32 before storing FP16 where the CUDA reference
  does so.
- Mish reproduces LC0's approximation rather than using a generic framework
  implementation.
- Policy mapping uses the 1858-entry ONNX gather map exported as the immutable
  `lczero_bt4_mapping_table` CUBIN symbol. This is semantically equivalent to
  LC0's 4288-entry scatter map and requires no external buffer.
- Autotune every family's block size and warp count on the requested active
  target, keyed by the complete static shape and activation when present.
  Persist tuning results and capture the selected block and resolved grid in
  the serialized artifact. Launch choices are tuning results, not public
  specialization inputs.
- Tile attention preprocessing across its output-channel dimension so block
  sizes below the full 624-channel width remain valid tuning candidates.

Acceptance criteria:

- Each kernel has CPU/PyTorch reference tests.
- Tests cover in-place operation where the CUDA wrapper permits it.
- Tests cover periodic bias broadcasting and tail masking.
- Policy map tests cover all 1858 outputs and invalid-index validation.
- Static tests cover each autotune key and candidate set. GPU artifact tests
  prove that every family serializes its selected launch configuration.
- Autotuning uses separate benchmark outputs so kernels that permit in-place
  execution do not mutate inputs between candidates.

### 6. Implement contiguous dense GEMM

Replace the toy matmul with a BT4 dense GEMM family supporting:

- Row-major `A[M, K] @ W[K, N]`
- FP16 activations and ONNX-layout FP16 weights
- FP32 accumulation
- FP16 output
- Static dimensions and launch grid
- Autotuning keyed by `(M, N, K)`

The fixed graph requires approximately these dense specializations:

| M | K | N | Uses |
| ---: | ---: | ---: | --- |
| 169 | 768 | 32768 | Dense positional preprocessing |
| 10816 | 624 | 1024 | Body embedding |
| 10816 | 1024 | 1536 | Embedding and encoder FFN1 |
| 10816 | 1536 | 1024 | Embedding and encoder FFN2 |
| 10816 | 1024 | 1024 | Q/K/V, MHA output, policy projections |
| 10816 | 1024 | 32 | Smolgen compression and MLH embedding |
| 169 | 2048 | 256 | Smolgen dense 1 |
| 169 | 256 | 8192 | Smolgen dense 2 |
| 5408 | 256 | 4096 | Shared Smolgen projection |
| 10816 | 1024 | 128 | Value embedding |
| 169 | 8192 | 128 | Value dense 1 |
| 169 | 128 | 3 | WDL logits |
| 169 | 2048 | 128 | Moves-left dense 1 |
| 169 | 128 | 1 | Moves-left result |

Initially issue separate Q, K, and V GEMMs because their exact ONNX initializer
names refer to separate buffers. Do not introduce a packed-weight format merely
to reproduce LC0's single strided-batched cuBLAS call.

Acceptance criteria:

- Every specialization matches `torch.matmul` within documented FP16
  tolerances.
- Compilation captures the selected autotuning configuration and resolved grid.
- Weight transposition is never performed in the graph.

### 7. Implement indexed batched attention GEMMs

Implement one indexed/strided batched family with these specializations:

| Operation | Batch | M | K | N |
| --- | ---: | ---: | ---: | ---: |
| Body QK | 5408 | 64 | 32 | 64 |
| Body attention times V | 5408 | 64 | 64 | 32 |
| Policy QK | 169 | 64 | 1024 | 64 |

For body QK, decode `(batch, head)` from the Triton program ID and use:

```text
base(batch, head) = batch * 65536 + head * 32
token stride      = 1024
depth stride      = 1
```

Express K transpose through strides. Write QK outputs contiguously as
`[169, 32, 64, 64]`.

For attention times V, read V through the same interleaved head view and write
the result directly into physical `[169, 64, 1024]` storage. This removes the
need for a merge-head transpose.

For policy QK, write the first 4096 elements of each 4288-element policy record,
leaving the final 192 elements for promotions.

Read the QK scale from its external ONNX-named scalar buffer.

Acceptance criteria:

- Batched results match explicit PyTorch head reshapes and matmuls.
- Tests prove correct behavior at the transition between heads and batches.
- No pointer-array, split-head, merge-head, or transpose buffers are allocated.

### 8. Port layer normalization

Implement `layer_norm_kernel<half>` with the CUDA operation order:

```text
value = input + bias
value = activation(value)
value = alpha * value
value = value + skip, when present
output = normalize(value, epsilon) * gamma + beta
```

Use population variance and FP32 reduction arithmetic. Required widths and
variants are:

| Width | Activation | Skip | Use |
| ---: | --- | --- | --- |
| 1024 | MISH | no | Initial embedding |
| 256 | SWISH | no | Smolgen LN1 |
| 8192 | SWISH | no | Smolgen LN2 |
| 1024 | NONE | yes | Embedding and encoder DeepNorm residuals |

Epsilon is `1e-3` for this network. Alpha is 1 for the first three variants and
is loaded from the relevant external scalar buffer for DeepNorm variants.

Acceptance criteria:

- Tests independently verify activation, alpha, skip, mean, variance, gamma,
  and beta ordering.
- Width 8192 is tested explicitly rather than inferred from smaller rows.
- Numerical tolerances account for CUDA/Triton exponential approximations but
  still catch FP16 reduction regressions.

### 9. Port the 64-way attention softmax

Implement `softmax_opt_64_kernel<half>` over rows of exactly 64 values.

Required behavior:

- Add Smolgen logits to scaled QK logits before finding the maximum.
- Preserve NaNs as the CUDA clamp does.
- Clamp FP16 infinities to the CUDA reference limit.
- Compute max, exponentials, sum, and division in FP32.
- Store FP16 attention weights.

The fixed graph processes `169 * 32 * 64 = 346112` rows per encoder block.

Acceptance criteria:

- Tests cover large positive and negative logits, infinities, NaNs, and random
  Smolgen additions.
- Every output row sums to one within FP16 tolerance.
- Results are stable after subtracting a large common offset.

### 10. Port promotion logits

Implement `promotion_logits_kernel<half>` using:

- Policy K: `[169, 64, 1024]`
- Promotion weights: `/policy/promotion/matmul/w`, shape `[1024, 4]`
- Policy records: `[169, 4288]`

For K rows 56 through 63, compute four promotion offsets. Add offset 3 to
offsets 0 through 2, combine them with policy QK rows 48 through 55 and columns
56 through 63, and write 192 logits directly to positions 4096 through 4287 of
each policy record.

Keep projection and assembly fused as in the LC0 CUDA kernel. Adapt indexing to
the ONNX `[1024, 4]` promotion-weight layout.

Acceptance criteria:

- Tests compare all 192 promotion logits against a direct FP32 reference.
- Tests verify policy records for adjacent batches remain separated by 4288
  elements.

### 11. Build the input and embedding graph

Construct this first complete graph segment:

```text
packed masks and values
  -> FP16 NCHW planes
  -> first 12 planes in token-major order
  -> dense 768 x 32768 positional preprocessing
  -> positional bias
  -> reshape to [169, 64, 512]
  -> concatenate with all 112 input planes
  -> dense 624 x 1024 body embedding
  -> Mish and layer norm
  -> multiplicative/additive input gate
  -> FFN 1024 x 1536 x 1024
  -> DeepNorm residual and layer norm
```

Validate this segment before adding encoder repetition. Inspect temporary
allocation reuse and dependencies as part of its tests.

Acceptance criteria:

- All external input and embedding weight names match the parsed network's
  active evaluation path.
- Every physical temporary shape is explicit.
- Flattening and reshaping are pointer reinterpretations, not graph nodes.
- The post-gate tensor is the embedding FFN skip input.

### 12. Build one encoder block

Add a private encoder-construction helper in `bt4/network.py` with this exact
order:

```text
Smolgen compress
Smolgen dense 1, Swish, and LN
Smolgen dense 2, Swish, and LN
Shared Smolgen projection
Q, K, and V projections and biases
Scaled body QK
Smolgen addition and 64-way softmax
Attention weights times V
MHA output projection
DeepNorm residual and LN1
FFN1 and Mish
FFN2
DeepNorm residual and LN2
```

Build and inspect encoder zero first. Once it is structurally and numerically
sound, traverse every protobuf encoder with the same helper, deriving its
prefix and local dimensions at the point of use.

Acceptance criteria:

- Each block consumes its own 25 learned initializers plus shared
  `/const/smolgen_w`.
- The body tensor remains `[169, 64, 1024]` after every block.
- Smolgen LN2 normalizes one 8192-element vector per sample, not one vector per
  head.
- DeepNorm scales the projected branch before adding the skip.

### 13. Build the output heads

Policy graph:

```text
body 1024 -> 1024 and Mish
Q and K 1024 -> 1024
scaled 64 x 64 policy logits
promotion logits
gather 4288 -> 1858
FP16 -> FP32
```

Do not apply a policy softmax.

Winner WDL graph:

```text
body 1024 -> 128 and Mish per square
flatten 64 * 128 = 8192
dense 8192 -> 128 and Mish
dense 128 -> 3 and bias
FP16 -> FP32 raw WDL logits
```

Do not apply a WDL softmax in the device graph.

Moves-left graph:

```text
body 1024 -> 32 and Mish per square
flatten 64 * 32 = 2048
dense 2048 -> 128 and Mish
dense 128 -> 1 and ReLU
FP16 -> FP32
```

Acceptance criteria:

- Output buffers have the agreed names, program execution scope, shapes, and F32
  dtype.
- Value output ordering is win, draw, loss.
- Moves-left output is nonnegative.
- Head branches depend on the final body tensor but not on each other.

### 14. Update CLI and artifact construction

Replace toy dimension flags with BT4-specific inputs:

- Input `.pb.gz` network path
- Output `lc0ex` path
- One or more fixed batch sizes, defaulting to 169

Include architecture, batch, and compilation target in generated artifact names
to avoid accidental cross-target reuse.

The graph command should:

1. Load and normalize the input network protobuf.
2. Traverse the selected active evaluation path.
3. Compile or reuse each required specialization through the per-build cache.
4. Declare external persistent buffers as their operations consume them.
5. Declare named execution inputs and outputs.
6. Construct and serialize the complete static graph.

Acceptance criteria:

- A failed compilation reports its operation and immutable specialization.
- A target or ABI mismatch fails before graph serialization.
- Graph construction requires the source `.pb.gz` but not an ONNX file.

## Verification Strategy

### Static and CPU tests

Add tests for:

- Exact external buffer names, shapes, dtypes, and counts
- Program-local execution-buffer allocations and non-aliasing
- Buffer alignment and I32 sizing
- Opaque buffer handles and raw temporary allocation
- On-demand kernel-cache key coverage
- Graph node kernel names and argument order
- Dependency construction across all 15 encoders
- Input/output interface shapes
- Absence of toy-network symbols and CLI options

### CUDA kernel tests

Each Triton kernel should be launched directly against generated tensors and
compared to a PyTorch reference. Expensive GPU tests should be marked so normal
non-GPU checks remain usable.

Test exact edge behavior, not only random values:

- Plane mask bit ordering
- Broadcast and in-place bias additions
- Mish and Swish approximations
- Layer norm reduction widths 256, 1024, and 8192
- Softmax infinities and NaNs
- Interleaved body-attention head strides
- Policy promotion indexing
- All 1858 policy map entries
- FP16-to-FP32 output conversion

### Actual-network integration

Use `leela2onnx` with FP16 output as an untracked integration fixture to verify
the exact initializer names, shapes, and values expected by the graph. Do not
commit the roughly 370 MiB model artifact.

End-to-end verification consists of:

- Numerical tests for every kernel and specialization
- Structural inspection of the complete serialized graph and each program allocation
- Validation of every external buffer against the generated FP16 ONNX model

Full policy, WDL, and moves-left comparison against LC0 is deferred until LC0
backend registration.

Run all repository checks after each coherent phase:

```bash
just format
just check
```

## Expected Initial Graph Size

With separate ONNX Q/K/V weight buffers and no packing or fusion, the expected
baseline is approximately:

- 194 GEMM calls
- 157 custom Triton calls
- 351 compute nodes
- 409 learned persistent buffers
- About 47 consumed external constant buffers plus one embedded mapping symbol
- 2 execution inputs
- 3 execution outputs

Treat these as completeness checks during the initial implementation, not as a
permanent format contract. Later fusion should be allowed to reduce node count
without changing external buffers or numerical semantics.

## Deferred Work

- Production `.pb.gz` or ONNX weight loading
- Remaining constants embedded as PTX/CUDA module symbols
- LC0 backend registration
- Host WDL softmax and Q/D conversion
- Alternate policy and value heads
- Multi-architecture artifacts
- Packed or fused QKV and policy-QK projections
- Cross-kernel and graph-level tuning beyond each family's launch autotuning
- Full LC0 numerical and performance comparison
