# [LLM Inference](https://arxiv.org/abs/2410.04466) 系统演进谱系

## 系统概览

本文档分析 LLM 推理领域主要系统的技术演进、核心创新和相互借鉴关系。

---

## 1. [vLLM](https://github.com/vllm-project/vllm) (UC Berkeley, 2023.09)

### 核心创新
- [**PagedAttention**](https://arxiv.org/abs/2309.06180): 将 KV cache 按 page（block）管理，类似 OS 虚拟内存分页，解决 KV cache 内存碎片化问题
- 实现 near-zero waste 的内存利用率（浪费 < 4%，对比 naive 方案 60-80% 浪费）

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | Continuous batching + preemption (swap/recompute) |
| Memory Manager | [PagedAttention](https://arxiv.org/abs/2309.06180) block table，copy-on-write for parallel sampling |
| Kernel Backend | 自研 paged attention kernel，后集成 FlashAttention/FlashInfer |
| Prefix Caching | Automatic prefix caching (APC)，hash-based block matching |

### 支持的优化技术
- Continuous batching, prefix caching, speculative decoding
- Tensor parallelism, chunked prefill
- FP8/INT8 quantization (via CUTLASS/Marlin kernels)
- LoRA serving, guided decoding

### 性能特征与适用场景
- 高吞吐在线服务，适合多租户 LLM serving
- 对 long context 场景内存效率高
- 社区生态最活跃（70k+ stars）

### 技术借鉴
- 从 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 借鉴 continuous batching 思想
- 后续被 [SGLang](https://github.com/sgl-project/sglang)、[LightLLM](https://github.com/ModelTC/lightllm)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 等借鉴 paged memory 设计

---

## 2. [SGLang](https://github.com/sgl-project/sglang) (Stanford/UC Berkeley, 2023.12)

### 核心创新
- [**RadixAttention**](https://arxiv.org/abs/2312.07104): 用 radix tree 管理 KV cache prefix，实现自动、细粒度的前缀复用
- **Structured Generation Language**: 编程模型层面优化 LLM 程序（fork/join/select）

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | Continuous batching + chunked prefill + [RadixAttention](https://arxiv.org/abs/2312.07104)-aware scheduling |
| Memory Manager | [RadixAttention](https://arxiv.org/abs/2312.07104) tree-based KV cache pool |
| Kernel Backend | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (primary), Triton kernels |
| Frontend | Python-embedded DSL for structured LLM programs |

### 支持的优化技术
- [RadixAttention](https://arxiv.org/abs/2312.07104) prefix caching（比 vLLM APC 更细粒度）
- Constrained decoding (regex/JSON schema)
- Speculative decoding, tensor parallelism
- Multi-modal support, data parallelism

### 性能特征与适用场景
- 多轮对话、tree-of-thought、agent 等有大量 prefix sharing 的场景
- 结构化输出（JSON mode）性能优势明显
- 适合复杂 LLM 程序编排

### 技术借鉴
- 从 [vLLM](https://github.com/vllm-project/vllm) 借鉴 continuous batching 和 paged memory 基本思路
- [RadixAttention](https://arxiv.org/abs/2312.07104) 是对 [Prompt Cache](https://arxiv.org/abs/2311.04934) 和 prefix caching 的系统化改进
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 作为 kernel backend 提供高效 attention 实现

---

## 3. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (NVIDIA, 2023.10)

### 核心创新
- [**In-flight Batching**](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html): NVIDIA 版 continuous batching，与 TensorRT 编译优化深度集成
- **FP8 全链路**: 从 weight 到 KV cache 到 attention 的 [FP8](https://arxiv.org/abs/2209.05433) 支持

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | In-flight batching (Batch Manager) |
| Memory Manager | Paged KV cache + [FP8](https://arxiv.org/abs/2209.05433) KV cache |
| Kernel Backend | TensorRT fused kernels, [CUTLASS](https://github.com/NVIDIA/cutlass), cuBLAS |
| Compiler | TensorRT graph optimization + kernel fusion |

### 支持的优化技术
- In-flight batching, paged KV cache
- INT8/FP8 weight + activation quantization ([SmoothQuant](https://arxiv.org/abs/2211.10438), AWQ, GPTQ)
- Multi-GPU tensor parallelism + pipeline parallelism
- Speculative decoding, KV cache reuse
- Inflight batching + chunked context

### 性能特征与适用场景
- NVIDIA GPU 上单卡/多卡性能最优（kernel fusion + hardware-specific optimization）
- 适合生产环境部署，延迟敏感场景
- 对 NVIDIA 硬件有深度绑定

### 技术借鉴
- 从 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 演进而来
- Continuous batching 思想来自 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- Paged KV cache 受 [vLLM](https://github.com/vllm-project/vllm) 影响

---

## 4. [llama.cpp](https://github.com/ggerganov/llama.cpp) (ggerganov, 2023.03)

### 核心创新
- **纯 C/C++ 实现**: 无 Python/CUDA 依赖，跨平台推理
- **GGUF 量化格式**: 灵活的混合精度量化（Q2_K ~ Q8_0）

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | 单请求为主，server mode 支持简单 batching |
| Memory Manager | 静态 KV cache 分配，mmap 模型加载 |
| Kernel Backend | CPU SIMD (AVX/NEON/WASM), Metal, CUDA, Vulkan |
| Quantization | [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 格式：k-quant (Q2_K~Q8_0), IQ (importance quant) |

### 支持的优化技术
- 多种量化格式（2-8 bit），importance matrix guided quantization
- CPU SIMD 优化（AVX2/AVX-512/ARM NEON）
- Metal (Apple Silicon), CUDA, Vulkan GPU offload
- Speculative decoding, grammar-based sampling
- KV cache quantization (Q4_0/Q8_0)

### 性能特征与适用场景
- 消费级硬件（CPU/Apple Silicon）上的最佳选择
- 离线推理、本地部署、边缘设备
- 模型格式转换的事实标准（[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)）

### 技术借鉴
- 量化方法受 GPTQ/AWQ 启发但自研实现
- 后续 [prima.cpp](https://arxiv.org/abs/2504.08791) 等项目基于其扩展分布式能力

---

## 5. [TGI](https://github.com/huggingface/text-generation-inference) - Text Generation Inference (HuggingFace)

### 核心创新
- **生产级 Rust 服务**: 高性能 gRPC/HTTP 服务框架
- **HuggingFace 生态集成**: 与 transformers 库无缝对接

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | Continuous batching (token-level) |
| Memory Manager | Flash Attention based, paged attention |
| Kernel Backend | Flash Attention, CUDA custom kernels |
| Server | Rust (Tokio) + Python model shards |

### 支持的优化技术
- Continuous batching, Flash Attention
- Tensor parallelism (NCCL)
- Quantization ([GPTQ](https://arxiv.org/abs/2210.17323), AWQ, bitsandbytes)
- Speculative decoding, watermarking
- Prefix caching

### 性能特征与适用场景
- HuggingFace Hub 模型直接部署
- 适合快速原型和中等规模生产部署
- Rust 服务层保证低延迟

### 技术借鉴
- Continuous batching 来自 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- Flash Attention 集成
- Paged attention 受 [vLLM](https://github.com/vllm-project/vllm) 启发

---

## 6. [LightLLM](https://github.com/ModelTC/lightllm) (ModelTC, 2023.08)

### 核心创新
- **轻量级 Python 实现**: 易于理解和修改的 serving 框架
- **Token-level scheduling**: 细粒度 token 级别调度

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | Token-level continuous batching |
| Memory Manager | Token-level KV cache management |
| Kernel Backend | Triton kernels |
| Server | Python asyncio + uvloop |

### 支持的优化技术
- Continuous batching (token-level granularity)
- Triton-based attention kernels
- Tensor parallelism
- Dynamic batching

### 性能特征与适用场景
- 研究和教学用途，代码可读性高
- 适合快速实验新的调度策略
- 性能略低于 vLLM/TensorRT-LLM

### 技术借鉴
- 整体架构受 [vLLM](https://github.com/vllm-project/vllm) 启发
- Token-level 管理思想独立发展

---

## 7. [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/) (InternLM/Shanghai AI Lab, 2023.06)

### 核心创新
- **TurboMind Engine**: 高性能 C++ 推理引擎
- **全流程工具链**: 量化 + 部署 + 服务一体化

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | Continuous batching |
| Memory Manager | Block-based KV cache (similar to [PagedAttention](https://arxiv.org/abs/2309.06180)) |
| Kernel Backend | TurboMind (C++ CUDA kernels) |
| Quantization | W4A16 ([AWQ](https://arxiv.org/abs/2306.00978)), KV cache INT8 |

### 支持的优化技术
- Continuous batching, persistent batching
- W4A16 quantization ([AWQ](https://arxiv.org/abs/2306.00978)-based)
- KV cache INT8 quantization
- Tensor parallelism
- Multi-modal model support (InternVL)

### 性能特征与适用场景
- InternLM 系列模型的官方推理框架
- 中文社区生态好
- 适合 InternLM/InternVL 部署

### 技术借鉴
- Continuous batching 来自 Orca/vLLM
- 量化方法集成 [AWQ](https://arxiv.org/abs/2306.00978)
- KV cache 管理受 [PagedAttention](https://arxiv.org/abs/2309.06180) 影响

---

## 8. [MLC-LLM](https://github.com/mlc-ai/mlc-llm) (mlc-ai, 2023.05)

### 核心创新
- **ML Compilation**: 基于 Apache TVM 的编译优化
- **Universal Deployment**: 单一框架覆盖 iOS/Android/Web/GPU/CPU

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | 简单 batching |
| Memory Manager | TVM memory planning |
| Kernel Backend | TVM generated kernels (Vulkan/Metal/CUDA/WebGPU) |
| Compiler | Apache TVM Unity |

### 支持的优化技术
- 编译期 kernel 生成和优化
- 量化 (Q4/Q3 via TVM)
- 多后端支持 (CUDA/Metal/Vulkan/WebGPU/OpenCL)
- Speculative decoding

### 性能特征与适用场景
- 跨平台部署（移动端、浏览器）
- 适合需要多硬件支持的场景
- 编译优化可针对特定硬件 tune

### 技术借鉴
- 基于 TVM 编译栈
- 量化方法参考 GPTQ/AWQ

---

## 9. [DeepSpeed](https://github.com/microsoft/DeepSpeed)-FastGen (Microsoft, 2023.11)

### 核心创新
- [**SplitFuse**](https://arxiv.org/abs/2401.08671): 将 long prompt 拆分 + 将 short prompt 与 generation 融合
- **Dynamic SplitFuse**: 自适应调整 split 粒度

### 关键技术组件
| 组件 | 实现 |
|------|------|
| Scheduler | SplitFuse continuous batching |
| Memory Manager | Blocked KV cache |
| Kernel Backend | [DeepSpeed](https://github.com/microsoft/DeepSpeed) inference kernels |
| Integration | MII (Model Implementations for Inference) |

### 支持的优化技术
- SplitFuse (chunked prefill variant)
- Continuous batching
- Tensor parallelism
- Quantization ([ZeroQuant](https://arxiv.org/abs/2206.01861) series)
- Non-persistent pipeline

### 性能特征与适用场景
- 与 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 训练框架集成
- 适合已使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 训练的模型直接部署
- 声称 2x [vLLM](https://github.com/vllm-project/vllm) 吞吐（有争议）

### 技术借鉴
- Continuous batching 来自 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- SplitFuse 是 chunked prefill 的变体（与 [Sarathi](https://arxiv.org/abs/2308.16369) 类似思想）
- 量化使用自研 [ZeroQuant](https://arxiv.org/abs/2206.01861) 系列

---

## 系统演进时间线

```mermaid
gantt
    title LLM Inference Systems Timeline
    dateFormat YYYY-MM
    axisFormat %Y-%m

    section Foundational
    FasterTransformer (NVIDIA)     :done, 2021-01, 2023-10
    Orca (Continuous Batching)     :milestone, 2022-07, 1d

    section Major Systems
    llama.cpp                      :active, 2023-03, 2026-05
    MLC-LLM                        :active, 2023-05, 2026-05
    LMDeploy                       :active, 2023-06, 2026-05
    LightLLM                       :active, 2023-08, 2026-05
    vLLM                           :active, 2023-09, 2026-05
    TensorRT-LLM                   :active, 2023-10, 2026-05
    DeepSpeed-FastGen              :active, 2023-11, 2026-05
    SGLang                         :active, 2023-12, 2026-05

    section Key Innovations
    PagedAttention (vLLM)          :milestone, 2023-09, 1d
    RadixAttention (SGLang)        :milestone, 2023-12, 1d
    FlashInfer                     :active, 2024-02, 2026-05
    Mooncake (Disaggregated)       :active, 2024-06, 2026-05
```

---

## 技术栈对比表

| 特性 | [vLLM](https://github.com/vllm-project/vllm) | [SGLang](https://github.com/sgl-project/sglang) | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | [llama.cpp](https://github.com/ggerganov/llama.cpp) | [LMDeploy](https://github.com/InternLM/lmdeploy) | [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | [DeepSpeed](https://github.com/microsoft/DeepSpeed)-FG | [LightLLM](https://github.com/ModelTC/lightllm) |
|------|------|--------|--------------|-----------|----------|---------|--------------|----------|
| **语言** | Python+CUDA | Python+CUDA | C++/Python | C/C++ | C++/Python | C++/TVM | Python+CUDA | Python+Triton |
| [**Continuous Batching**](https://www.usenix.org/system/files/osdi22-yu.pdf) | ✅ | ✅ | ✅ | ⚠️ (server) | ✅ | ❌ | ✅ | ✅ |
| **Paged KV Cache** | ✅ | ✅ (Radix) | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ (token) |
| **Prefix Caching** | ✅ (APC) | ✅ (Radix) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [**Speculative Decoding**](https://arxiv.org/abs/2211.17192) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Tensor Parallelism** | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **FP8 Support** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **CPU Inference** | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Mobile/Edge** | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Multi-Modal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **主要硬件** | NVIDIA GPU | NVIDIA GPU | NVIDIA GPU | CPU/多平台 | NVIDIA GPU | 多平台 | NVIDIA GPU | NVIDIA GPU |
| **Stars (approx)** | 70k+ | 20k+ | 10k+ | 75k+ | 5k+ | 20k+ | 35k+ | 3k+ |

---

## 技术借鉴关系图

```mermaid
graph TD
    ORCA[Orca 2022<br>Continuous Batching] --> VLLM[vLLM 2023.09<br>PagedAttention]
    ORCA --> TRTLLM[TensorRT-LLM 2023.10<br>In-flight Batching]
    ORCA --> DSFG[DeepSpeed-FastGen 2023.11<br>SplitFuse]
    ORCA --> LMDEPLOY[LMDeploy 2023.06]
    ORCA --> LIGHTLLM[LightLLM 2023.08]

    FT[FasterTransformer] --> TRTLLM
    
    VLLM --> SGLANG[SGLang 2023.12<br>RadixAttention]
    VLLM --> LIGHTLLM
    VLLM --> LMDEPLOY
    
    FA[FlashAttention<br>2022-2024] --> VLLM
    FA --> SGLANG
    FA --> TRTLLM
    
    FLASHINFER[FlashInfer 2024.02] --> SGLANG
    
    VLLM --> MOONCAKE[Mooncake 2024.06<br>Disaggregated]
    VLLM --> DISTSERVE[DistServe 2024.01<br>P/D Disaggregation]
    
    TVM[Apache TVM] --> MLCLLM[MLC-LLM 2023.05]
    
    GGML[ggml] --> LLAMACPP[llama.cpp 2023.03]
    
    SARATHI[Sarathi 2023.08<br>Chunked Prefill] --> DSFG
    SARATHI --> VLLM
    SARATHI --> SGLANG
```

---

## 各系统核心差异化总结

| 系统 | 一句话差异化 |
|------|-------------|
| [vLLM](https://github.com/vllm-project/vllm) | 内存效率（[PagedAttention](https://arxiv.org/abs/2309.06180)）+ 最大社区生态 |
| [SGLang](https://github.com/sgl-project/sglang) | 前缀复用（[RadixAttention](https://arxiv.org/abs/2312.07104)）+ 结构化生成编程模型 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 硬件深度优化 + 编译期 kernel fusion |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 跨平台 + 消费级硬件 + 量化格式标准 |
| [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/) | InternLM 生态 + 全流程工具链 |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 编译优化 + 移动端/浏览器部署 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed)-FastGen | [DeepSpeed](https://github.com/microsoft/DeepSpeed) 训练生态集成 + SplitFuse |
| [LightLLM](https://github.com/ModelTC/lightllm) | 轻量级 + 研究友好 + Triton kernel |
| [Mooncake](https://arxiv.org/abs/2407.00079) | KV cache 中心的 disaggregated 架构 |
