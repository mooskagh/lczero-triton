# Design Task: QKV Layout Packing & Monolithic GEMM Fusion

## 1. Overview & Motivation

Currently, Multi-Head Attention (MHA) projections in `lczero-triton` compute Query ($Q$), Key ($K$), and Value ($V$) projections via a 3D grid dispatch in `_fused_qkv_projection_kernel`. Each thread block indexes along `tl.program_id(2)` and evaluates conditional branches to switch between three separate weight matrices ($W_Q, W_K, W_V \in \mathbb{R}^{1024 	imes 1024}$) and write to three separate output tensors ($Q, K, V \in \mathbb{R}^{M 	imes 1024}$).

At large batch sizes (such as Batch 256, where $M = 16,384$), this 3-way multiplexed dispatch suffers from:
1. **Branch overhead and pointer divergence** in CUDA thread blocks.
2. **Suboptimal Tensor Core utilization and L2 cache reuse**: Three smaller $16384 	imes 1024$ GEMMs achieve lower arithmetic intensity than a single monolithic $16384 	imes 3072$ GEMM.
3. **Memory fragmentation**: Writing to 3 independent buffers prevents contiguous vector loads in downstream attention kernels.

Profiling on NVIDIA A100 shows TensorRT's monolithic packed QKV GEMM is **~49 µs faster per layer** (total **+0.74 ms across 15 layers** at Batch 256).

---

## 2. Layout Transformations & Packing Requirements

Packing $Q, K, V$ is **not** merely placing three separate memory buffers adjacent in DRAM. It requires layout interleaving, stride transformations during weight transfer, and updated indexing in the attention kernel.

### A. Weight Packing: Contiguous Column-Interleaved $W_{QKV}$

In the ONNX model, weights are stored as three independent row-major tensors:
- $W_Q \in \mathbb{R}^{K 	imes N_Q}$ with shape $(1024, 1024)$, stride $(1024, 1)$
- $W_K \in \mathbb{R}^{K 	imes N_K}$ with shape $(1024, 1024)$, stride $(1024, 1)$
- $W_V \in \mathbb{R}^{K 	imes N_V}$ with shape $(1024, 1024)$, stride $(1024, 1)$

#### Target Packed Layout: $W_{QKV} \in \mathbb{R}^{K 	imes 3N} = \mathbb{R}^{1024 	imes 3072}$
During weight ingestion / transfer time, the weights must be actively concatenated along the output column dimension ($N$):

```python
# Active transfer-time packing
w_qkv = torch.cat([w_q, w_k, w_v], dim=1).contiguous()  # shape: [1024, 3072]
b_qkv = torch.cat([b_q, b_k, b_v], dim=0).contiguous()  # shape: [3072]
```

In memory, each row $k \in [0..1023]$ contains:
$$	ext{Memory Row } k = \Big[ W_Q[k, 0..1023] \;\Big|\; W_K[k, 0..1023] \;\Big|\; W_V[k, 0..1023] \Big]$$

---

### B. Output Activation Buffer & Stride Changes

When the standard monolithic `_matmul_kernel` computes $Y_{QKV} = X \cdot W_{QKV} + B_{QKV}$:
- Input Activations: $X \in \mathbb{R}^{M 	imes 1024}$
- Packed Weights: $W_{QKV} \in \mathbb{R}^{1024 	imes 3072}$
- Output Buffer: $Y_{QKV} \in \mathbb{R}^{M 	imes 3072}$

#### Memory Strides:
For each token row $m \in [0..M-1]$ (where $M = 	ext{batch\_size} 	imes 64$):
- $Q_m$ is located at offset: $m 	imes 3072 + 0$
- $K_m$ is located at offset: $m 	imes 3072 + 1024$
- $V_m$ is located at offset: $m 	imes 3072 + 2048$
- Stride between consecutive tokens ($m 	o m+1$) is **$3072$ half-words** ($6144$ bytes), rather than $1024$.

---

### C. Modifications Required in `fused_attention.py`

In `_fused_attention_kernel`:
Instead of taking 3 separate pointers (`q_ptr`, `k_ptr`, `v_ptr`) with row stride `model_width = 1024`, the kernel takes a single `qkv_ptr` and uses the packed stride `stride_row = 3 * model_width = 3072`:

```python
# Base pointer calculations in packed attention kernel
stride_row = 3 * model_width  # 3072

q_base = qkv_ptr + sample * (_SQUARE_COUNT * stride_row) + (0 * model_width) + head * head_depth
k_base = qkv_ptr + sample * (_SQUARE_COUNT * stride_row) + (1 * model_width) + head * head_depth
v_base = qkv_ptr + sample * (_SQUARE_COUNT * stride_row) + (2 * model_width) + head * head_depth
```

---

## 3. Implementation Plan

1. **Weight Packing in Network Ingestion**:
   - In `network.py`, concatenate $(W_Q, W_K, W_V)$ along column axis and $(B_Q, B_K, B_V)$ into unified buffer allocations.
2. **Standard `matmul` for QKV**:
   - Replace `fused_qkv_projection` with `matmul(output=qkv_buf, activations=act, weights=w_qkv, bias=b_qkv, activation="none")`.
3. **Update `fused_attention.py`**:
   - Accept packed `qkv` tensor and apply stride-3072 indexing.
4. **Benchmark & Verify**:
   - Verify numerical accuracy with `pytest` and benchmark latency.

---

## 4. Expected Performance Impact

- **Batch 256**: Estimated latency reduction of **~0.74 ms** (closing the gap with TensorRT).
- **Batch 128**: Estimated latency reduction of **~0.35 ms**.
