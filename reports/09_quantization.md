# 量化技术深度分析

## 1. 量化基础

### 1.1 量化公式

**对称量化（Symmetric）：**

$$
X_q = \text{clamp}\left(\left\lfloor \frac{X}{s} \right\rceil, -2^{b-1}, 2^{b-1} - 1\right), \quad s = \frac{\max(|X|)}{2^{b-1} - 1}
$$

$$
\hat{X} = X_q \cdot s
$$

**非对称量化（Asymmetric）：**

$$
X_q = \text{clamp}\left(\left\lfloor \frac{X - z}{s} \right\rceil, 0, 2^b - 1\right), \quad s = \frac{\max(X) - \min(X)}{2^b - 1}, \quad z = \min(X)
$$

$$
\hat{X} = X_q \cdot s + z
$$

**量化误差：**

$$
\epsilon = X - \hat{X}, \quad \mathbb{E}[\epsilon^2] \approx \frac{s^2}{12} \quad \text{(uniform quantization noise)}
$$

### 1.2 量化粒度

| 粒度 | Scale 数量 | 精度 | 开销 |
|------|-----------|------|------|
| Per-tensor | 1 | 低 | 最小 |
| Per-channel/Per-row | $d_{\text{out}}$ | 中 | 小 |
| Per-group (g=128) | $d_{\text{out}} \times \lceil d_{\text{in}}/g \rceil$ | 高 | 中 |
| Per-element | $d_{\text{out}} \times d_{\text{in}}$ | 最高 | 大（不实用） |

Per-group quantization 是当前主流选择，group size $g = 128$ 在精度和开销间取得平衡。

### 1.3 Rounding 策略

**Round-to-Nearest (RTN)：**

$$
X_q = \lfloor X/s \rceil
$$

简单但非最优。对于 LLM 的 weight quantization，RTN 在 4-bit 以下精度显著下降。

**Optimal Brain Quantizer (OBQ) / Optimal Brain Surgeon：**

基于 Hessian 信息最小化量化后的输出误差：

$$
\min_{\hat{W}} \| WX - \hat{W}X \|_2^2
$$

逐列量化，每量化一列后用 Hessian 逆补偿剩余列：

$$
\delta_j = \frac{w_j - \hat{w}_j}{[H^{-1}]_{jj}} \cdot H^{-1}_{:,j}
$$

---

## 2. Weight-Only Quantization

### 2.1 [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)

**问题定义：** 将 OBQ 的 $O(d^3)$ 复杂度降低到可处理大模型的水平。

**方法核心：**

1. Layer-wise quantization: 逐层最小化 $\| WX - \hat{W}X \|_2^2$
2. 固定列顺序（不做 greedy selection），使得 Hessian 逆可以批量更新
3. Cholesky 分解 + lazy batch updates

**算法：**

$$
H = 2X X^T + \lambda I \quad \text{(Hessian of squared error)}
$$

对每列 $j$（按固定顺序）：
$$
\hat{w}_j = \text{quantize}(w_j)
$$
$$
\text{error}_j = \frac{w_j - \hat{w}_j}{[H^{-1}]_{jj}}
$$
$$
W_{:, j+1:} \mathrel{+}= \text{error}_j \cdot H^{-1}_{j, j+1:}
$$

**时间复杂度：** $O(d_{\text{row}} \cdot d_{\text{col}}^2)$ per layer（相比 OBQ 的 $O(d_{\text{col}}^3)$）

**实验指标：**
- Llama-7B W4 (group=128): WikiText-2 PPL 5.63（FP16 baseline 5.47）
- Llama-65B W4: PPL 3.84（FP16: 3.53）
- 量化时间：Llama-7B 约 4 GPU-minutes

**工程可复现难点：**
- 需要 calibration data（通常 128 samples from C4）
- Hessian 计算需要足够的 calibration samples 保证数值稳定
- Group quantization 的 Hessian 更新需要特殊处理
- 不同实现（Auto[GPTQ](https://arxiv.org/abs/2210.17323), [GPTQ](https://arxiv.org/abs/2210.17323)-for-LLaMA）的精度可能有差异

**加速 kernel：** Marlin kernel 实现 [GPTQ](https://arxiv.org/abs/2210.17323) W4A16 的高效 GEMM，在 A100 上达到接近 FP16 的吞吐。

### 2.2 [AWQ](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)

**问题定义：** 不依赖 Hessian 的简单高效 weight quantization。

**方法核心：**

观察：weight 中 1% 的 salient channels（对应 activation 大的通道）对精度影响巨大。

策略：对 salient channels 做 per-channel scaling 后再量化。

$$
Q(w \cdot s) \cdot (x / s) \approx w \cdot x
$$

其中 $s$ 为 per-channel scaling factor，通过 grid search 最小化量化误差：

$$
s^* = \arg\min_s \| Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - WX \|_2^2
$$

实际实现中，$s$ 基于 activation magnitude 的幂次：

$$
s_j = \left(\frac{|X_j|_{\max}}{|W_j|_{\max}}\right)^\alpha, \quad \alpha \in [0, 1]
$$

$\alpha$ 通过 grid search 确定（通常 $\alpha \approx 0.5$）。

**实验指标：**
- Llama-2-7B W4g128: PPL 5.60（优于 [GPTQ](https://arxiv.org/abs/2210.17323) 的 5.63）
- 量化速度比 [GPTQ](https://arxiv.org/abs/2210.17323) 快 10x+（无需 Hessian 逆）
- 支持 W3 量化，PPL 6.24

**工程可复现难点：**
- Scaling factor 的 grid search 范围和粒度影响结果
- 需要 calibration data 计算 activation statistics
- 与 [GPTQ](https://arxiv.org/abs/2210.17323) 可组合使用（AWQ scaling + [GPTQ](https://arxiv.org/abs/2210.17323) rounding）

### 2.3 [QuIP](https://arxiv.org/abs/2307.13304) / [QuIP#](https://arxiv.org/abs/2402.04396) (Chee et al., 2023-2024)

**问题定义：** 实现 2-bit weight quantization 而不显著损失精度。

**方法核心（[QuIP](https://arxiv.org/abs/2307.13304)）：**

Incoherence Processing: 通过随机正交变换使 weight matrix 的元素分布更均匀。

$$
\hat{W} = Q(U W V^T) \quad \text{where } U, V \text{ are random orthogonal matrices}
$$

推理时：$\hat{W}x = U^T Q(UWV^T) V x$

**方法核心（[QuIP#](https://arxiv.org/abs/2402.04396)）：**

1. 使用 Hadamard matrix 替代随机正交矩阵（计算更快）
2. E8 lattice codebook 替代 uniform quantization（更优的 rate-distortion）
3. Fine-tuning with LDLQ (Lattice-based Diagonal LDL Quantization)

**E8 Lattice：** 8 维空间中最密堆积格，每个 lattice point 编码 8 个权重值，有效 bit rate 约 2 bits/weight。

**实验指标：**
- Llama-2-7B 2-bit: PPL 7.85（[QuIP#](https://arxiv.org/abs/2402.04396) with E8）
- Llama-2-70B 2-bit: PPL 4.15

**工程可复现难点：**
- E8 lattice 的 encoding/decoding 需要专用 kernel
- Hadamard transform 在非 2 的幂维度需要 padding
- 推理时的 dequantization 开销较大

### 2.4 [AQLM](https://arxiv.org/abs/2401.06118) (Egiazarian et al., 2024)

**问题定义：** 使用 additive quantization 实现极低比特量化。

**方法核心：**

将每组权重表示为多个 codebook 向量之和：

$$
\hat{w}_g = \sum_{m=1}^{M} C_m[i_m]
$$

其中 $C_m \in \mathbb{R}^{K \times d_g}$ 为第 $m$ 个 codebook，$i_m$ 为 index。

有效 bit rate: $M \times \lceil \log_2 K \rceil / d_g$

**实验指标：** 2-bit Llama-2-7B PPL 8.32（使用 2 codebooks, K=256, group=8）。

---

## 3. Weight-Activation Quantization

### 3.1 [SmoothQuant](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022)

**问题定义：** Activation 中存在大量 outlier（某些 channel 值远大于其他），直接量化 activation 精度损失大。

**方法核心：**

将量化难度从 activation 迁移到 weight：

$$
Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}
$$

Smoothing factor:

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \quad \alpha \in [0, 1]
$$

$\alpha = 0.5$ 时等价于平衡 activation 和 weight 的量化难度。

**量化方案：** W8A8（INT8 weight + INT8 activation），per-tensor 或 per-token activation quantization。

**实验指标：**
- OPT-175B W8A8: 精度无损（< 0.1% degradation）
- Llama-2-7B W8A8: PPL 5.48（FP16: 5.47）
- 推理加速：1.5x（利用 INT8 [Tensor Core](https://arxiv.org/abs/1803.04014)）

**工程可复现难点：**
- $\alpha$ 的最优值因模型和层而异
- 需要 calibration data 确定 activation range
- 某些层（如 attention 的 QKV projection）outlier 特别严重，可能需要 per-layer $\alpha$

```mermaid
graph LR
    A[Original: X @ W] --> B[Smooth: X/s @ s*W]
    B --> C[Quantize Both]
    C --> D[INT8 GEMM]
    D --> E[Dequantize Output]
    
    style A fill:#f9f,stroke:#333
    style D fill:#9f9,stroke:#333
```

### 3.2 [LLM.int8()](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)

**问题定义：** 处理 activation 中的 extreme outlier（某些 channel 值 > 100x 均值）。

**方法核心：** Mixed-precision decomposition。

1. 识别 outlier channels: $|X_{:,j}| > \tau$（$\tau = 6.0$）
2. Outlier channels 用 FP16 计算
3. 其余 channels 用 INT8 计算
4. 结果相加

$$
Y = X_{\text{outlier}} W_{\text{outlier}}^T + \text{dequant}(Q(X_{\text{normal}}) \cdot Q(W_{\text{normal}})^T)
$$

**实验指标：**
- 6.7B+ 模型精度无损
- 推理速度比 FP16 慢约 15-20%（因为 mixed-precision 的开销）
- 主要价值在于减少显存占用

**工程难点：** Outlier channel 的比例通常 < 1%，但 mixed-precision 的 kernel 实现需要 scatter/gather 操作。

### 3.3 [Atom](https://arxiv.org/abs/2310.19102) (Zhao et al., 2024)

**问题定义：** 实现 W4A4 的高效推理。

**方法核心：**

1. Mixed-precision: outlier channels 用高精度，其余用 4-bit
2. Reorder: 将 outlier channels 聚集到一起，减少 mixed-precision 的碎片化
3. Dynamic quantization: per-token activation quantization

**实验指标：** Llama-2-7B W4A4，throughput 提升 2.5x vs FP16。

### 3.4 [QServe](https://arxiv.org/abs/2405.04532) (Lin et al., 2024)

**问题定义：** [W4A8KV4](https://arxiv.org/abs/2405.04532) 的系统级量化方案。

**方法核心（QoQ - Quattuor-Octo-Quattuor）：**

1. Weight: 4-bit with group quantization (g=128)
2. Activation: 8-bit per-token dynamic quantization
3. KV Cache: 4-bit per-channel (key) + per-token (value)
4. Progressive quantization: 先 W4A8 GEMM，再 KV4 存储

**SmoothAttention：** 对 attention 的 K 做 smoothing，减少 KV cache 量化误差。

**实验指标：**
- Llama-3-8B: 1.2x throughput vs [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) FP16
- Llama-3-70B (4×A100): 2.4x throughput vs FP16
- 精度：大部分任务 < 1% degradation

**工程难点：**
- 需要定制 CUDA kernel 支持 W4A8 GEMM
- KV cache quantization 需要与 attention kernel 集成
- Progressive quantization 的 pipeline 设计

---

## 4. [FP8](https://arxiv.org/abs/2209.05433) Quantization

### 4.1 [FP8](https://arxiv.org/abs/2209.05433) 格式

| 格式 | 符号位 | 指数位 | 尾数位 | 范围 | 精度 | 用途 |
|------|--------|--------|--------|------|------|------|
| E4M3 | 1 | 4 | 3 | ±448 | ~0.125 | Forward (weight, activation) |
| E5M2 | 1 | 5 | 2 | ±57344 | ~0.25 | Backward (gradient) |
| FP16 | 1 | 5 | 10 | ±65504 | ~0.001 | Baseline |

**E4M3 表示：**

$$
\text{value} = (-1)^s \times 2^{e - 7} \times (1 + m/8)
$$

其中 $e \in [1, 14]$（正常数），$s$ 为符号位，$m \in [0, 7]$ 为尾数。

### 4.2 Scaling 策略

**Per-tensor scaling：**

$$
s = \frac{\max(|X|)}{448}, \quad X_{\text{fp8}} = \text{cast\_to\_fp8}(X / s)
$$

推理时：$Y = (X_{\text{fp8}} \cdot s_X) \times (W_{\text{fp8}} \cdot s_W) = (X_{\text{fp8}} \times W_{\text{fp8}}) \cdot (s_X \cdot s_W)$

**Per-token scaling（activation）：**

每个 token 独立计算 scale，精度更高但需要 online calibration。

**Delayed scaling（[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)）：**

使用前一次 iteration 的 amax 作为当前 scale（避免额外的 reduction kernel）。

### 4.3 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) FP8 Workflow

```mermaid
graph TD
    A[FP16 Model] --> B[Calibration: 收集 activation amax]
    B --> C[计算 per-tensor scales]
    C --> D[Weight 转换为 FP8 E4M3]
    D --> E[Build TRT Engine with FP8 layers]
    E --> F[Runtime: dynamic activation scaling]
    
    F --> G[FP8 GEMM on Tensor Core]
    G --> H[FP32 accumulation]
    H --> I[Output cast to FP16/BF16]
```

**性能：** H100 [FP8](https://arxiv.org/abs/2209.05433) [Tensor Core](https://arxiv.org/abs/1803.04014) 提供 1978 TFLOPS（vs FP16 989 TFLOPS），理论 2x 加速。

**实际加速：** 1.5-1.8x（受限于 memory-bound 操作和非 GEMM 计算）。

### 4.4 [FP8](https://arxiv.org/abs/2209.05433) vs INT8 对比

| 维度 | [FP8](https://arxiv.org/abs/2209.05433) (E4M3) | INT8 |
|------|------------|------|
| 动态范围 | 大（指数表示） | 小（线性） |
| 精度 | 中（3-bit mantissa） | 高（8-bit 均匀） |
| Outlier 处理 | 自然适应 | 需要 clipping/scaling |
| 硬件支持 | H100+, Ada | A100+, Ampere+ |
| Calibration | 简单（per-tensor 即可） | 需要 per-channel/per-token |
| 推理速度 | 最快（H100） | 快（A100/H100） |

---

## 5. KV Cache Quantization

### 5.1 [KIVI](https://arxiv.org/abs/2402.02750) (Liu et al., 2024)

（详见 reports/07_kv_cache.md §2.1.1）

**核心公式：**
- Key: per-channel INT2, $s_k \in \mathbb{R}^{d_h}$
- Value: per-token INT2, $s_v \in \mathbb{R}^{N}$

### 5.2 [Gear](https://arxiv.org/abs/2403.05527) (Kang et al., 2024)

（详见 reports/07_kv_cache.md §2.1.3）

**核心公式：**

$$
\text{KV} \approx \underbrace{UV^T}_{\text{low-rank}} + \underbrace{Q(\text{KV} - UV^T)}_{\text{quantized residual}} + \underbrace{S}_{\text{sparse outlier}}
$$

### 5.3 [NexusQuant (2025)](https://arxiv.org/abs/2505.00949)

**问题定义：** Training-free KV cache compression via vector quantization。

**方法核心：**
- 使用 E8 lattice 对 KV cache 做 vector quantization
- 每 8 个连续元素作为一个 vector，映射到最近的 E8 lattice point
- 有效 bit rate: 约 2 bits/element

**E8 Lattice VQ 公式：**

$$
\hat{v} = \arg\min_{\ell \in E_8} \| v - s \cdot \ell \|_2^2
$$

其中 $s$ 为 per-vector scaling factor。

**优势：** 比 scalar quantization 在相同 bit rate 下误差更小（利用了向量间的相关性）。

---

## 6. 系统级量化

### 6.1 各框架支持的量化格式

| 框架 | W4A16 | W8A8 | W4A8 | [FP8](https://arxiv.org/abs/2209.05433) | KV Quant | 主要 kernel |
|------|-------|------|------|-----|----------|-------------|
| [vLLM](https://github.com/vllm-project/vllm) | GPTQ, AWQ | [SmoothQuant](https://arxiv.org/abs/2211.10438) | ✗ | ✓ (H100) | FP8 | Marlin, [CUTLASS](https://github.com/NVIDIA/cutlass) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | AWQ, GPTQ | [SmoothQuant](https://arxiv.org/abs/2211.10438) | ✓ | ✓ | INT8/FP8 | 自研 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF Q2-Q8 | ✗ | ✗ | ✗ | ✗ | 自研 (CPU/Metal) |
| [SGLang](https://github.com/sgl-project/sglang) | GPTQ, AWQ | ✓ | ✗ | ✓ | ✓ | Marlin, [FlashInfer](https://github.com/flashinfer-ai/flashinfer) |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | [ZeroQuant](https://arxiv.org/abs/2206.01861) | ✓ | ✗ | ✓ | ✗ | 自研 |
| HuggingFace | bitsandbytes, [GPTQ](https://arxiv.org/abs/2210.17323), AWQ | ✓ | ✗ | ✓ | ✗ | 各库 kernel |

### 6.2 [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 格式详解（[llama.cpp](https://github.com/ggerganov/llama.cpp)）

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) (GPT-Generated Unified Format) 是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 使用的模型文件格式，支持多种量化类型：

| 类型 | Bits/Weight | 方法 | 精度 (7B PPL) | 速度 |
|------|-------------|------|---------------|------|
| Q2_K | 2.5 | Mixed 2/3-bit with importance | ~8.5 | 最快 |
| Q3_K_M | 3.4 | 3-bit with super-blocks | ~6.8 | 快 |
| Q4_K_M | 4.6 | 4-bit with k-quants | ~5.9 | 中 |
| Q5_K_M | 5.5 | 5-bit with k-quants | ~5.6 | 中 |
| Q6_K | 6.6 | 6-bit with k-quants | ~5.5 | 慢 |
| Q8_0 | 8.5 | 8-bit round-to-nearest | ~5.47 | 最慢 |

**K-Quants 结构：**
- Super-block: 256 elements
- Sub-block: 16-32 elements，各有独立 scale
- Super-block 存储 sub-block scales 的 scale（二级量化）

**存储布局：**

$$
\text{Block} = [\underbrace{d}_{\text{scale, FP16}}][\underbrace{m}_{\text{min, FP16}}][\underbrace{q_0, q_1, ...}_{\text{quantized weights}}]
$$

### 6.3 [Marlin](https://github.com/IST-DASLab/marlin) Kernel

**问题定义：** GPTQ/AWQ W4A16 的 GEMM 在 GPU 上效率低（dequantize 开销大）。

**方法核心：**

1. Weight 预处理为 GPU-friendly layout（interleaved for coalesced access）
2. Asynchronous dequantization: 在 GEMM 的 global load 阶段做 dequant
3. 利用 [Tensor Core](https://arxiv.org/abs/1803.04014) 的 FP16 GEMM，weight 在 shared memory 中 on-the-fly dequant

**性能：**
- A100 上 W4A16 达到 FP16 GEMM 吞吐的 80%+
- 相比 naive dequant + GEMM：3-4x 加速
- Batch size 1 时加速最明显（memory-bound 场景）

**Marlin 变体：**
- Marlin (original): W4A16, per-group
- Marlin-24: W4A16 with 2:4 sparsity
- Machete: 支持更多格式（W4A8, W8A8）

---

## 7. 量化 Pipeline 总览

```mermaid
graph TD
    A[FP16/BF16 预训练模型] --> B{选择量化方案}
    
    B --> C[Weight-Only W4]
    B --> D[Weight-Activation W8A8]
    B --> E[FP8]
    B --> F[Extreme Low-bit W2/W3]
    
    C --> C1[GPTQ: Hessian-based]
    C --> C2[AWQ: Activation-aware scaling]
    C --> C3[GGUF: llama.cpp format]
    
    D --> D1[SmoothQuant: smooth + INT8]
    D --> D2[QServe: W4A8KV4]
    
    E --> E1[Calibration → per-tensor scale]
    E --> E2[TensorRT-LLM build]
    
    F --> F1[QuIP#: Hadamard + E8 lattice]
    F --> F2[AQLM: Additive codebook]
    
    C1 --> G[Marlin Kernel]
    C2 --> G
    D1 --> H[INT8 Tensor Core]
    E2 --> I[FP8 Tensor Core]
    F1 --> J[Custom dequant kernel]
    
    G --> K[Serving: vLLM / SGLang / TRT-LLM]
    H --> K
    I --> K
    J --> K
```

---

## 8. 精度-速度 Tradeoff 总结

### 8.1 Llama-2-7B 量化对比

| 方法 | Bits | PPL (Wiki2) | 相对 FP16 | Throughput 倍数 | 显存节省 |
|------|------|-------------|-----------|----------------|----------|
| FP16 (baseline) | 16 | 5.47 | 0% | 1.0x | 0% |
| W8A8 [SmoothQuant](https://arxiv.org/abs/2211.10438) | 8 | 5.48 | +0.2% | 1.5x | 50% |
| [FP8](https://arxiv.org/abs/2209.05433) E4M3 | 8 | 5.48 | +0.2% | 1.8x (H100) | 50% |
| W4A16 [GPTQ](https://arxiv.org/abs/2210.17323) g128 | 4 | 5.63 | +2.9% | 2.0x | 75% |
| W4A16 [AWQ](https://arxiv.org/abs/2306.00978) g128 | 4 | 5.60 | +2.4% | 2.0x | 75% |
| W4A8 [QServe](https://arxiv.org/abs/2405.04532) | 4/8 | 5.65 | +3.3% | 2.5x | 62% |
| W3A16 [AWQ](https://arxiv.org/abs/2306.00978) | 3 | 6.24 | +14% | 2.5x | 81% |
| W2A16 [QuIP#](https://arxiv.org/abs/2402.04396) | 2 | 7.85 | +43% | 2.0x* | 87% |
| W2A16 [AQLM](https://arxiv.org/abs/2401.06118) | 2 | 8.32 | +52% | 1.8x* | 87% |

*注：2-bit 方法的 dequantization 开销较大，实际加速受限。

### 8.2 选择指南

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 精度优先，H100 可用 | [FP8](https://arxiv.org/abs/2209.05433) | 几乎无损，硬件原生支持 |
| 精度优先，A100 | W8A8 [SmoothQuant](https://arxiv.org/abs/2211.10438) | 成熟，精度好 |
| 吞吐优先，精度可接受 | W4A16 [AWQ](https://arxiv.org/abs/2306.00978) + Marlin | 高吞吐，精度损失小 |
| 显存极度受限 | W4 [GPTQ](https://arxiv.org/abs/2210.17323) + KV4 | 最大化 batch size |
| 边缘设备/CPU | [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) Q4_K_M | [llama.cpp](https://github.com/ggerganov/llama.cpp) 生态完善 |
| 研究/极限压缩 | [QuIP#](https://arxiv.org/abs/2402.04396) 2-bit | 最低 bit rate |

### 8.3 硬件兼容性

| 方法 | NVIDIA Ampere (A100) | NVIDIA Hopper (H100) | NVIDIA Ada (4090) | Apple Silicon | CPU (AVX) |
|------|---------------------|---------------------|-------------------|---------------|-----------|
| [FP8](https://arxiv.org/abs/2209.05433) | ✗ | ✓ | ✓ | ✗ | ✗ |
| INT8 GEMM | ✓ | ✓ | ✓ | ✗ | ✓ (VNNI) |
| W4A16 (Marlin) | ✓ | ✓ | ✓ | ✗ | ✗ |
| [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) ([llama.cpp](https://github.com/ggerganov/llama.cpp)) | ✓ | ✓ | ✓ | ✓ (Metal) | ✓ |
| bitsandbytes NF4 | ✓ | ✓ | ✓ | ✗ | ✗ |
