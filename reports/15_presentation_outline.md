# [LLM Inference](https://arxiv.org/abs/2410.04466) 演讲大纲

## 演讲信息

- **主题**：[LLM Inference](https://arxiv.org/abs/2410.04466) 优化：从算法到系统的全景分析
- **时长**：90 分钟（含 Q&A）
- **目标听众**：ML 工程师、系统研究者、基础设施团队
- **前置知识**：Transformer 基础、GPU 编程基础

---

## Part 1: 基础与动机（15 分钟）

### Slide 1-3: 为什么 [LLM Inference](https://arxiv.org/abs/2410.04466) 重要

**论点**：推理成本已超过训练成本，是 LLM 规模化部署的核心瓶颈。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | GPT-4 级别推理成本分析 | 成本饼图：compute/memory/network | 行业报告 [B] |
| 3 min | Inference vs Training 计算特性 | 对比表格 | 05_kernel_and_math.md [A] |
| 2 min | 关键指标定义 | TTFT/TPOT/Throughput 示意图 | 11_benchmark_map.md [A] |

> **Speaker Notes**: 开场用一个具体数字抓住注意力——"GPT-4 每天推理成本估计超过 $X00万"。强调推理是持续成本而训练是一次性投入。

### Slide 4-6: Autoregressive Decoding 的瓶颈

**论点**：prefill 和 decode 的计算特性根本不同，这一差异驱动了几乎所有优化方向。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | prefill vs decode 计算特性 | Roofline Model 双点标注 | 05_kernel_and_math.md [A] |
| 3 min | KV cache 内存增长 | 显存占用随 seq_len 增长曲线 | 07_kv_cache.md [A] |
| 2 min | 瓶颈总结 | 表格：prefill(compute) vs decode(memory) | 06_serving_scheduling.md [A] |

> **Speaker Notes**: 用 Roofline Model 图直观展示 prefill 在计算区、decode 在带宽区。这是后续所有优化的出发点。

### Slide 7-8: 优化方向总览

**论点**：7 大优化方向各有适用场景，没有银弹。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 2 min | 分类体系展示 | 03_taxonomy.md 的 Mermaid 图 | 03_taxonomy.md [A] |
| 2 min | 各方向成熟度 | 热度趋势表 | 14_knowledge_graph.md [B] |

> **Speaker Notes**: 快速过一遍全景图，让听众建立方向感。后续每个 Part 深入一个方向。

---

## Part 2: 计算优化（20 分钟）

### Slide 9-12: [FlashAttention](https://arxiv.org/abs/2205.14135) 系列

**论点**：IO-aware tiling 是 attention 优化的范式转变，将瓶颈从 HBM 带宽转移到计算。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Standard Attention IO 瓶颈 | HBM vs SRAM 带宽对比图 | 05_kernel_and_math.md [A] |
| 4 min | FA-1 Tiling + Recomputation | SRAM tiling 动画/分步图 | FlashAttention 论文 [A] |
| 2 min | FA-2/FA-3 改进 | 性能对比柱状图 | FlashAttention-3 论文 [A] |
| 2 min | IO 复杂度公式推导 | 公式：O(N²d²/M) | 05_kernel_and_math.md [A] |

> **Speaker Notes**: 这是全场最重要的技术 slide。用 SRAM tiling 的分步图解释为什么不需要 materialize 完整 attention matrix。强调 recomputation 是"用计算换内存"的经典 tradeoff。

### Slide 13-15: decode 阶段优化

**论点**：decode 阶段的瓶颈是 memory bandwidth，需要完全不同于 prefill 的优化策略。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | FlashDecoding Split-K | 并行拆分示意图 | FlashDecoding blog [A] |
| 3 min | PagedAttention | OS 虚拟内存类比图 | vLLM 论文 [A] |
| 2 min | GQA/MQA 减少 KV heads | head 共享示意图 | GQA 论文 [A] |

> **Speaker Notes**: PagedAttention 用 OS 虚拟内存的类比最容易理解。强调 block 粒度的 tradeoff：太大浪费空间，太小增加管理开销。

### Slide 16-18: Sparse Attention

**论点**：attention matrix 天然稀疏，利用稀疏性可在 prefill 阶段获得 2-4x 加速，但精度需要验证。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Attention Sink 现象 | attention heatmap 可视化 | StreamingLLM 论文 [A] |
| 3 min | H2O/MInference 方法 | sparse pattern 分类图 | 05_kernel_and_math.md [A] |
| 2 min | 精度-速度 tradeoff | Pareto curve | 13_contradictions.md [B] |

> **Speaker Notes**: 展示真实的 attention heatmap，让听众直观看到稀疏性。提醒：sparse attention 在短序列上可能不如 dense FlashAttention。

---

## Part 3: 内存优化（15 分钟）

### Slide 19-21: KV cache 管理

**论点**：KV cache 是 LLM 推理的内存主要消耗者，其管理方式决定了系统的最大并发能力。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | KV cache 大小计算公式 | 公式 + 各模型对比表 | 07_kv_cache.md [A] |
| 3 min | 各模型 KV cache 占用对比 | 柱状图：Llama-70B vs DeepSeek-V3 | 07_kv_cache.md [A] |
| 2 min | vLLM Block Manager | block 分配/释放流程图 | vLLM 论文 [A] |

> **Speaker Notes**: 用具体数字震撼听众——"Llama-3-70B 在 128K context 下 KV cache = 20 GB，而 DeepSeek-V3 仅 7.6 GB"。这引出 MLA 的价值。

### Slide 22-24: KV cache 压缩

**论点**：KV cache 压缩方法形成 Pareto frontier，需要根据任务特征选择合适的压缩率。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | 量化方法：KIVI | per-channel K + per-token V 示意 | 07_kv_cache.md [A] |
| 3 min | Eviction：SnapKV, PyramidKV | attention pattern 可视化 | SnapKV 论文 [A] |
| 2 min | 压缩率 vs 精度对比 | Pareto curve 图 | 13_contradictions.md [A] |

> **Speaker Notes**: 强调"不能只看 PPL"——在 multi-turn 和长依赖任务上精度下降明显。引用 MiKV 的发现：低精度保留优于完全丢弃。

### Slide 25-26: 量化技术

**论点**：W4A8 是当前 serving 场景的最佳精度-效率平衡点，FP4 是下一代方向。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Weight-only vs W+A 量化 | 方法对比表 | 09_quantization.md [A] |
| 3 min | 精度-速度-内存三角 | 三角 tradeoff 图 | 09_quantization.md [A] |

> **Speaker Notes**: 用 QServe 的 W4A8KV4 作为案例，展示 system co-design 的价值。提醒：量化校准集的选择对结果影响很大。

---

## Part 4: 调度与 Serving（15 分钟）

### Slide 27-29: [continuous batching](https://www.usenix.org/system/files/osdi22-yu.pdf)

**论点**：iteration-level scheduling 消除了 padding 浪费，是现代 serving 系统的基础。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Static vs Dynamic vs continuous batching | 三种方式对比动画 | 06_serving_scheduling.md [A] |
| 3 min | Orca 设计原理 | iteration-level 调度时序图 | Orca 论文 [A] |
| 2 min | chunked prefill | SplitFuse 示意图 | Sarathi 论文 [A] |

> **Speaker Notes**: 注意 36.9x 的数字是与最差情况对比。实际 vs dynamic batching 的提升在 2-5x。

### Slide 30-32: prefix caching

**论点**：前缀复用可节省 60-80% 的重复计算，是多轮对话和 RAG 场景的关键优化。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | APC vs RadixAttention | Radix Tree 数据结构图 | SGLang 论文 [A] |
| 3 min | Cache hit rate 分析 | 不同场景命中率对比 | 06_serving_scheduling.md [B] |
| 2 min | 局限性 | 无共享 prefix 时的 overhead | 13_contradictions.md [A] |

> **Speaker Notes**: 用具体场景说明——"100 个用户共享同一 system prompt，prefix caching 节省 99% 的重复 prefill"。但也要提醒：如果没有共享 prefix，开启 caching 反而有 overhead。

### Slide 33-35: P/D disaggregation

**论点**：prefill 和 decode 的资源需求根本不同，物理分离是大规模部署的必然趋势。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | 动机与架构 | P/D 分离架构图 | DistServe 论文 [A] |
| 3 min | Mooncake KVCache-centric | 分布式 KV pool 架构图 | Mooncake 论文 [A] |
| 2 min | 工程挑战 | KV transfer 延迟分析 | 13_contradictions.md [A] |

> **Speaker Notes**: 强调网络需求——"Llama-70B 单请求 KV transfer ~2.6 GB，需要 RDMA 级别带宽"。这是 P/D 分离的主要部署门槛。

---

## Part 5: 解码加速（10 分钟）

### Slide 36-38: [Speculative Decoding](https://arxiv.org/abs/2211.17192)

**论点**：speculation 用计算换延迟，在低并发场景可获得 2-3x 加速，但高 batch 下收益递减。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Draft-then-Verify 框架 | 流程图 + 接受/拒绝示意 | 08_speculative_decoding.md [A] |
| 3 min | 加速比公式 | 公式推导 + 参数影响图 | 08_speculative_decoding.md [A] |
| 2 min | Draft model 选择 | 方法分类树 | 08_speculative_decoding.md [A] |

> **Speaker Notes**: 用公式说明为什么 acceptance rate 是关键——α 从 0.7 提升到 0.9 时加速比从 1.8x 跳到 2.5x。

### Slide 39-41: 方法对比

**论点**：从独立 draft model 到 self-draft，speculative decoding 正在向零额外开销演进。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | Medusa/EAGLE 对比 | speedup 柱状图 | 08_speculative_decoding.md [A] |
| 2 min | MineDraft batch 场景 | batch-aware speculation 示意 | MineDraft 论文 [B] |

> **Speaker Notes**: 强调工程现实——"Medusa 已在 vLLM/SGLang 中可用，是最容易部署的方案"。

---

## Part 6: 分布式推理（10 分钟）

### Slide 42-44: 并行策略

**论点**：没有万能的并行策略，需要根据模型大小、硬件拓扑和延迟要求选择。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 4 min | TP/PP/SP/EP 对比 | 四种策略通信模式图 | 10_distributed_inference.md [A] |
| 3 min | 通信量分析 | 公式 + 表格 | 10_distributed_inference.md [A] |
| 2 min | 选择决策树 | 决策流程图 | 10_distributed_inference.md [B] |

> **Speaker Notes**: 关键 insight——"TP=8 不一定比 TP=4 快，对小模型通信 overhead 可能超过计算节省"。

### Slide 45-46: [MoE](https://arxiv.org/abs/2407.06204) 推理

**论点**：MoE 的稀疏激活带来推理效率优势，但 expert dispatch 和负载均衡是核心挑战。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | DeepSeek-V3 推理架构 | EP + TP 混合并行图 | DeepSeek-V3 论文 [A] |
| 2 min | Load balancing | EPLB 策略示意 | 10_distributed_inference.md [B] |

> **Speaker Notes**: DeepSeek-V3 是全栈优化的最佳案例——MLA + EP + FP8 + DualPipe 的组合。

---

## Part 7: 系统对比与选型（5 分钟）

### Slide 47-48: 主流系统对比

**论点**：系统选型取决于 workload 特征，没有绝对最优。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | 四大系统对比 | 特性对比表 | 01_repo_map.md [B] |
| 2 min | 选型决策树 | 流程图 | 00_MASTER_REPORT.md [B] |

> **Speaker Notes**: 简洁总结——"prefix-heavy 用 SGLang，通用 serving 用 vLLM，极致性能用 TRT-LLM，边缘部署用 llama.cpp"。

---

## Part 8: 未来方向（5 分钟）

### Slide 49-50: 开放问题与趋势

**论点**：P/D disaggregation + FP4 + MLA 是确定性趋势，1M context 和 test-time scaling 是新挑战。

| 时间 | 内容 | 建议图示 | 证据来源 |
|------|------|----------|----------|
| 3 min | 确定性趋势 | 时间线预测图 | 00_MASTER_REPORT.md [B] |
| 2 min | 开放问题 | 问题列表 | 13_contradictions.md [A] |

> **Speaker Notes**: 以"如果只记住三件事"结尾：(1) prefill/decode 分离是大势所趋；(2) KV cache 是内存瓶颈的核心；(3) 没有银弹，需要根据场景组合优化。

---

## Q&A（10 分钟）

### 预备问题
1. "[vLLM](https://github.com/vllm-project/vllm) 和 [SGLang](https://github.com/sgl-project/sglang) 该选哪个？" → 取决于 workload 特征
2. "Speculative decoding 在生产中有用吗？" → 低并发场景有效
3. "[FP8](https://arxiv.org/abs/2209.05433) 真的无损吗？" → 大部分场景接近无损，但需要验证
4. "如何评估 KV cache 压缩的影响？" → 不能只看 PPL，需要 task-specific 评估

---

## 附录：Demo 建议

| Demo | 工具 | 时间 | 效果 |
|------|------|------|------|
| [FlashAttention](https://arxiv.org/abs/2205.14135) 性能对比 | Triton benchmark | 2 min | 直观展示 IO 优化效果 |
| [vLLM](https://github.com/vllm-project/vllm) vs HF 吞吐对比 | benchmark_throughput.py | 3 min | 展示 [PagedAttention](https://arxiv.org/abs/2309.06180) 效果 |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) | vLLM + draft model | 3 min | 展示加速效果 |
| 量化精度对比 | lm-eval | 5 min | 展示量化 tradeoff |

---

## 附录：演讲结构流程图

```mermaid
graph LR
    subgraph "Part 1: 基础 (15min)"
        A1[开场: 为什么关注推理] --> A2[全景图: 7大方向]
        A2 --> A3[关键指标: TTFT/TPOT/Throughput]
    end
    
    subgraph "Part 2: 核心技术 (25min)"
        A3 --> B1[Attention Kernel<br/>FlashAttention系列]
        B1 --> B2[KV cache<br/>压缩/驱逐/共享]
        B2 --> B3[Quantization<br/>W4A8 实践]
        B3 --> B4[Speculative Decoding<br/>Draft-Verify范式]
    end
    
    subgraph "Part 3: 系统与工程 (15min)"
        B4 --> C1[Serving系统对比<br/>vLLM/SGLang/TRT-LLM]
        C1 --> C2[分布式推理<br/>TP/EP/P-D分离]
        C2 --> C3[DeepSeek-V3<br/>全栈案例分析]
    end
    
    subgraph "Part 4: 展望 (5min)"
        C3 --> D1[趋势: FP4/MLA/1M Context]
        D1 --> D2[研究机会: 55条选题]
        D2 --> D3[Q&A]
    end
    
    style A1 fill:#e8f4fd
    style D3 fill:#e8f4fd
```

