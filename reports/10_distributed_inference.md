# 分布式推理分析

## 总览

随着模型规模增长（70B → 405B → [MoE](https://arxiv.org/abs/2407.06204)），单 GPU 无法容纳完整模型，分布式推理成为必需。本文档分析各种并行策略、多节点 serving 架构和通信优化。

---

## 1. 并行策略

### 1.1 [Tensor Parallel](https://arxiv.org/abs/2402.04925)ism (TP)

**定义**：将单层的权重矩阵沿某个维度切分到多个 GPU，每个 GPU 计算部分结果后通过 AllReduce 聚合。

**Megatron-LM Column/Row Parallel**：

```mermaid
graph LR
    subgraph "Column Parallel Linear"
        X[Input X] --> GPU0[GPU 0: X × W₀]
        X --> GPU1[GPU 1: X × W₁]
        GPU0 --> Y0[Y₀]
        GPU1 --> Y1[Y₁]
    end
    
    subgraph "Row Parallel Linear"
        Y0 --> GPU0R[GPU 0: Y₀ × W₀ᵀ]
        Y1 --> GPU1R[GPU 1: Y₁ × W₁ᵀ]
        GPU0R --> AR[AllReduce]
        GPU1R --> AR
        AR --> OUT[Output]
    end
```

**通信量分析**：
- 每层 2 次 AllReduce（MLP 前后）
- 每次 AllReduce 通信量：$2 \times (N-1)/N \times hidden\_size \times batch \times dtype\_bytes$
- 对于 TP=8, hidden=8192, FP16：每层 ~128KB per token

**适用条件**：
- GPU 间需要高带宽互联（NVLink > 600 GB/s）
- 通常 TP ≤ 8（单机内）
- Latency-sensitive：每层都有同步点

**Inference 特点**：
- Decode 阶段：通信量小（单 token），但 AllReduce latency 是瓶颈
- Prefill 阶段：通信量大，但可被计算 overlap

### 1.2 Pipeline Parallelism (PP)

**定义**：将模型的不同层分配到不同 GPU，形成流水线。

**Bubble Ratio**：

$$\text{Bubble} = \frac{(p-1) \times t_{micro}}{(m + p - 1) \times t_{micro}} = \frac{p-1}{m+p-1}$$

其中 $p$ = pipeline stages, $m$ = micro-batches。

**Inference 中的 PP**：
- Prefill：可以用 micro-batch 填充 bubble
- Decode：每个 token 必须经过所有 stage，bubble 严重
- 通常 inference 中 PP 不如 TP 高效（latency 增加）

**适用条件**：
- 跨节点（网络带宽有限）
- 模型太大无法用 TP 放入单机
- 通常与 TP 组合：intra-node TP + inter-node PP

### 1.3 Sequence Parallelism (SP)

**定义**：将长序列切分到多个 GPU 并行处理 attention。

[**Ring Attention**](https://arxiv.org/abs/2310.01889)：
- 将 KV 分块，在 GPU 间以 ring 方式传递
- 每个 GPU 计算 local Q × remote K/V
- 通信与计算 overlap

```mermaid
graph TD
    subgraph "Ring Attention (4 GPUs)"
        G0[GPU 0<br/>Q₀, K₀V₀] -->|send K₀V₀| G1[GPU 1<br/>Q₁, K₁V₁]
        G1 -->|send K₁V₁| G2[GPU 2<br/>Q₂, K₂V₂]
        G2 -->|send K₂V₂| G3[GPU 3<br/>Q₃, K₃V₃]
        G3 -->|send K₃V₃| G0
    end
```

**通信量**：每个 GPU 发送 $2 \times seq\_len/P \times d \times dtype\_bytes$（K 和 V）

**Ulysses ([DeepSpeed](https://github.com/microsoft/DeepSpeed))**：
- 将 Q/K/V 沿 head 维度切分
- AlltoAll 通信重新分布
- 每个 GPU 计算完整 attention 的一部分 head

**Ring vs Ulysses**：

| 维度 | [Ring Attention](https://arxiv.org/abs/2310.01889) | Ulysses |
|------|---------------|---------|
| 通信模式 | P2P ring | AlltoAll |
| 通信量 | O(seq_len × d / P) | O(seq_len × d / P) |
| Overlap | 可以 overlap | 难以 overlap |
| 适用场景 | 超长序列 | 中等序列 + 多 head |
| 实现复杂度 | 高 | 中 |

**LoongServe (2024)**：
- **问题**：固定 SP degree 无法适应动态 workload
- **方法**：Elastic Sequence Parallelism
  - 根据请求长度动态调整 SP degree
  - 短请求用 TP，长请求用 SP
- **实验指标**：相比固定 SP，吞吐提升 1.5-2x

### 1.4 Expert Parallelism (EP)

**定义**：[MoE](https://arxiv.org/abs/2407.06204) 模型中，将不同 expert 分配到不同 GPU。

**通信模式**：
- AlltoAll：将 token 路由到对应 expert 所在的 GPU
- 通信量取决于 routing 结果（不均匀）

**挑战**：
- Expert load imbalance：热门 expert 成为瓶颈
- AlltoAll 通信延迟
- 与 TP 的组合（EP + TP）

### 1.5 Context Parallelism (CP)

**定义**：在 prefill 阶段将长 context 切分到多个 GPU 并行处理。

**与 SP 的区别**：CP 通常指 prefill 阶段的并行，SP 更通用。

**Cache-DiT 实现**：
- [Ring Attention](https://arxiv.org/abs/2310.01889) with batched P2P
- USP (Hybrid Ring + Ulysses)
- 2D/3D Hybrid Parallelism (USP + TP)

---

## 2. Multi-Node Serving

### 2.1 [LoongServe](https://arxiv.org/abs/2404.09526)

**论文**：LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism

**问题定义**：长序列请求需要 SP，但短序列不需要，如何动态适配？

**方法核心**：
- Elastic SP：根据请求长度动态调整并行度
- 请求迁移：在 SP degree 变化时迁移 KV Cache
- 与 TP 的混合：短请求 TP-only，长请求 TP+SP

**实验指标**：
- 支持 100K+ token 序列
- 相比 static SP，吞吐提升 1.5-2x
- P99 latency 降低

### 2.2 [HexGen](https://arxiv.org/abs/2311.11514)

**问题定义**：如何在异构 GPU 集群上高效 serving？

**方法核心**：
- 异构感知的模型放置
- 根据 GPU 计算能力和互联带宽分配层
- 支持混合 A100 + A10G 等配置

### 2.3 [Mooncake](https://arxiv.org/abs/2407.00079)

**问题定义**：如何构建以 KV Cache 为中心的分布式 serving 架构？

**方法核心**：
- Disaggregated architecture：计算节点 + 存储节点
- KV Cache Pool：独立的分布式 KV 存储层
- Conductor：全局调度，决定 prefill/decode 放置

**架构**：
```mermaid
graph TD
    subgraph "Mooncake Architecture"
        COND[Conductor<br/>Global Scheduler]
        
        subgraph "Compute Layer"
            PF1[Prefill Node 1]
            PF2[Prefill Node 2]
            DC1[Decode Node 1]
            DC2[Decode Node 2]
        end
        
        subgraph "Storage Layer"
            KV1[KV Cache Pool 1<br/>CPU DRAM]
            KV2[KV Cache Pool 2<br/>CPU DRAM]
            SSD[SSD Tier]
        end
        
        COND --> PF1
        COND --> DC1
        PF1 --> KV1
        DC1 --> KV1
        KV1 --> SSD
    end
```

### 2.4 [DistServe](https://arxiv.org/abs/2401.09670)

（详见 06_serving_scheduling.md 第 3 节）

---

## 3. [MoE Inference](https://arxiv.org/abs/2404.02852)

### 3.1 [MoE](https://arxiv.org/abs/2407.06204) 基础

**结构**：每层有 N 个 expert（FFN），router 选择 top-k expert 处理每个 token。

**DeepSeek-V3 MoE**：
- 256 个 routed experts + 1 shared expert
- Top-2 routing
- 每个 token 只激活 2/256 的 expert 参数

### 3.2 推理挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| Expert Load Imbalance | 热门 expert 处理更多 token | GPU 利用率不均 |
| AlltoAll Communication | Token 路由到远程 GPU | 通信延迟 |
| Memory | 所有 expert 权重需要常驻显存 | 显存占用大 |
| Batch Efficiency | 每个 expert 处理的 token 数不同 | GEMM 效率低 |

### 3.3 优化方法

**Expert Parallelism + TP**：
- EP：不同 expert 在不同 GPU
- TP：每个 expert 内部 tensor parallel
- 组合：EP=8, TP=4 → 32 GPU

**Expert Offloading**：
- 将不活跃 expert 卸载到 CPU
- 预测下一步需要的 expert 并预取
- [Mixtral Offloading](https://arxiv.org/abs/2312.17238)：在消费级 GPU 上运行 MoE

**Expert Quantization**：
- 对不同 expert 使用不同量化精度
- 热门 expert 保持高精度

### 3.4 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 推理优化

- 256 expert 分布在多节点
- Shared expert 在所有 GPU 上复制
- 优化的 AlltoAll kernel
- [FP8](https://arxiv.org/abs/2209.05433) 量化减少通信量

---

## 4. Communication Optimization

### 4.1 Compute-Communication Overlap

**原理**：在 GPU 计算当前层时，同时传输下一层需要的数据。

**实现方式**：
- CUDA Stream：计算和通信在不同 stream
- Kernel-level overlap：在 kernel 内部交替计算和通信
- TileLink ([Triton-distributed](https://arxiv.org/abs/2503.20313))：tile 级别的 overlap

### 4.2 [Triton-distributed](https://arxiv.org/abs/2503.20313) / TileLink

**论文**：TileLink: Generating Efficient Compute-Communication Overlapping Kernels (ByteDance-Seed, 2025)

**方法核心**：
- 在 Triton kernel 中嵌入通信原语
- Tile 级别的 compute-communication overlap
- 自动生成 overlap kernel

**效果**：减少 AllReduce 的暴露延迟

### 4.3 NCCL 优化

- Ring AllReduce vs Tree AllReduce
- 小消息优化：batched small AllReduce
- NVLink topology-aware routing

### 4.4 NVLink vs InfiniBand

| 维度 | NVLink (H100) | InfiniBand HDR |
|------|---------------|----------------|
| 带宽 | 900 GB/s (bi) | 200 Gb/s (25 GB/s) |
| 延迟 | ~1 μs | ~1-2 μs |
| 适用 | Intra-node TP | Inter-node PP/EP |
| 拓扑 | Full mesh (8 GPU) | Fat tree |

---

## 5. 性能模型

### 5.1 Communication Volume

**TP AllReduce per layer**：
$$V_{TP} = 2 \times \frac{N-1}{N} \times H \times B \times S \times dtype$$

**PP inter-stage**：
$$V_{PP} = H \times B \times S \times dtype$$

**SP [Ring Attention](https://arxiv.org/abs/2310.01889) per step**：
$$V_{SP} = 2 \times \frac{S}{P} \times d_{head} \times n_{kv\_heads} \times dtype$$

### 5.2 Scaling Efficiency

**TP Scaling**：
$$\text{Efficiency}_{TP} = \frac{T_{compute}}{T_{compute} + T_{allreduce}}$$

对于 decode（单 token）：
- $T_{compute}$ 很小（memory-bound）
- $T_{allreduce}$ 相对较大
- TP=8 时效率可能只有 60-70%

**PP Scaling**：
$$\text{Efficiency}_{PP} = \frac{m}{m + p - 1}$$

### 5.3 最优并行策略选择

| 模型大小 | 硬件 | 推荐策略 |
|----------|------|----------|
| 7B | 1×A100 | 无并行 |
| 13B | 1×A100 | 无并行（[FP8](https://arxiv.org/abs/2209.05433)）或 TP=2 |
| 70B | 8×A100 | TP=8 |
| 70B | 2×8×A100 | TP=8, PP=2 |
| 405B | 8×8×H100 | TP=8, PP=8 |
| [MoE](https://arxiv.org/abs/2407.06204)-8x22B | 8×A100 | EP=8 或 TP=4,EP=2 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 多节点 | TP=8, EP=32+ |

---

## 6. Mermaid 总览图

```mermaid
graph TD
    DIST[分布式推理]
    
    DIST --> INTRA[Intra-Node]
    DIST --> INTER[Inter-Node]
    DIST --> HYBRID[Hybrid]
    
    INTRA --> TP[Tensor Parallelism<br/>NVLink]
    INTRA --> SP_LOCAL[Sequence Parallelism<br/>NVLink]
    
    INTER --> PP[Pipeline Parallelism<br/>InfiniBand]
    INTER --> EP[Expert Parallelism<br/>AlltoAll]
    INTER --> SP_REMOTE[Ring Attention<br/>RDMA]
    
    HYBRID --> TP_PP[TP + PP]
    HYBRID --> TP_EP[TP + EP]
    HYBRID --> TP_SP[TP + SP<br/>LoongServe]
    HYBRID --> PD[P/D Disaggregation<br/>DistServe/Mooncake]
```

---

## 7. 工程可复现难点

| 系统 | 难点 | 原因 |
|------|------|------|
| LoongServe | KV Cache 迁移 | 动态 SP 变化时需要重新分布 KV |
| [Mooncake](https://arxiv.org/abs/2407.00079) | 分布式 KV Pool | 需要高性能 KV 存储系统 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 256 expert routing | AlltoAll 通信优化需要定制 |
| [Ring Attention](https://arxiv.org/abs/2310.01889) | Overlap 实现 | 需要精确的 CUDA stream 管理 |
| TileLink | Triton 扩展 | 需要修改 Triton compiler |

---

## 最新进展 (2025-2026)

- [**NVIDIA Dynamo**](https://developer.nvidia.com/blog/nvidia-dynamo-adds-gpu-autoscaling-kubernetes-automation-and-networking-optimizations/) (NVIDIA, GTC 2025): 数据中心级推理编排，原生P/D disaggregation + wide Expert Parallelism，B200上单GPU 3.1k tok/s
- [**llm-d**](https://github.com/llm-d/llm-d) (Red Hat/IBM, 2025): Kubernetes原生分布式推理，支持disaggregated serving + prefix-cache-aware routing + MoE wide-EP，16x16 B200达50k tok/s
- [**STAR**](https://arxiv.org/abs/2510.13668) (2025): Decode阶段重调度，解决disaggregated架构中长输出reasoning任务导致的decode实例负载不均衡
- [**PPD Disaggregation**](https://arxiv.org/abs/2603.13358) (2026): 多轮对话场景的三级disaggregation(Prefill-Prefill-Decode)，区分full-prefill和append-prefill减少KV传输
- [**AMPD**](https://arxiv.org/abs/2602.14516) (2026): 高效多轮LLM推理的disaggregated serving，基于实时队列状态的路由优化
- [**DuetServe**](https://arxiv.org/abs/2511.04791) (2025): 自适应intra-GPU prefill/decode协调，挑战完全物理隔离的必要性，在同一GPU上高效混合两阶段
