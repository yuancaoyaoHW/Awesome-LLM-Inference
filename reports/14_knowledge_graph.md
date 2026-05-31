# LLM Inference 知识图谱

## 总览

本文档以 Mermaid 图和结构化描述展示 LLM Inference 领域的知识关系网络。

---

## 1. 核心概念关系图

```mermaid
graph TD
    subgraph "模型层"
        TRANSFORMER[Transformer]
        MHA[Multi-Head Attention]
        FFN[Feed-Forward Network]
        MLA[Multi-head Latent Attention]
        MOE[Mixture of Experts]
        GQA[Grouped Query Attention]
        MQA[Multi-Query Attention]
    end
    
    subgraph "计算层"
        FA[FlashAttention]
        PA[PagedAttention]
        GEMM[GEMM/Tensor Core]
        SPARSE[Sparse Attention]
        QUANT[Quantization]
    end
    
    subgraph "系统层"
        VLLM[vLLM]
        SGLANG[SGLang]
        TRTLLM[TensorRT-LLM]
        LLAMACPP[llama.cpp]
    end
    
    subgraph "调度层"
        CB[Continuous Batching]
        PD[P/D Disaggregation]
        SPEC[Speculative Decoding]
        PREFIX[Prefix Caching]
    end
    
    TRANSFORMER --> MHA
    TRANSFORMER --> FFN
    MHA --> GQA
    MHA --> MQA
    MHA --> MLA
    FFN --> MOE
    
    MHA --> FA
    FA --> PA
    PA --> VLLM
    FA --> SGLANG
    
    VLLM --> CB
    VLLM --> PREFIX
    SGLANG --> PREFIX
    VLLM --> SPEC
    
    QUANT --> LLAMACPP
    QUANT --> TRTLLM
    
    GQA --> PA
    MOE --> TRTLLM
```

## 2. 技术演进时间线

```mermaid
gantt
    title LLM Inference 关键技术时间线
    dateFormat YYYY-MM
    
    section Attention
    Online Softmax          :2018-01, 2018-12
    FlashAttention-1        :2022-05, 2022-12
    FlashAttention-2        :2023-07, 2023-12
    FlashDecoding           :2023-10, 2024-03
    FlashAttention-3        :2024-07, 2024-12
    
    section Serving
    Orca (Continuous Batch) :2022-06, 2022-12
    vLLM (PagedAttention)   :2023-06, 2024-12
    SGLang (RadixAttention) :2024-01, 2024-12
    DistServe (P/D Split)   :2024-03, 2024-12
    
    section Decoding
    Speculative Decoding    :2023-02, 2023-12
    Medusa                  :2024-01, 2024-06
    EAGLE                   :2024-03, 2024-09
    EAGLE-2                 :2024-07, 2024-12
    
    section Quantization
    GPTQ                    :2022-10, 2023-06
    AWQ                     :2023-06, 2024-01
    SmoothQuant             :2022-11, 2023-06
    FP8 (H100)              :2023-06, 2024-12
    
    section Architecture
    DeepSeek-V2 (MLA)       :2024-05, 2024-12
    DeepSeek-V3             :2024-12, 2025-06
```

## 3. 问题-方法映射

```mermaid
graph LR
    subgraph "核心问题"
        P1[内存瓶颈]
        P2[计算瓶颈]
        P3[延迟瓶颈]
        P4[吞吐瓶颈]
        P5[长序列]
    end
    
    subgraph "解决方法"
        S1[KV Cache 压缩]
        S2[量化]
        S3[FlashAttention]
        S4[Speculative Decoding]
        S5[Continuous Batching]
        S6[P/D Disaggregation]
        S7[Sparse Attention]
        S8[Offloading]
        S9[Sequence Parallelism]
    end
    
    P1 --> S1
    P1 --> S2
    P1 --> S8
    P2 --> S3
    P2 --> S7
    P3 --> S4
    P3 --> S6
    P4 --> S5
    P4 --> S6
    P5 --> S7
    P5 --> S9
    P5 --> S1
```

## 4. 系统依赖关系

```mermaid
graph BT
    subgraph "Hardware"
        GPU[NVIDIA GPU]
        NVLINK[NVLink]
        HBM[HBM Memory]
        TC[Tensor Core]
    end
    
    subgraph "Kernel Libraries"
        CUTLASS[CUTLASS]
        TRITON[Triton]
        FLASHATTN[FlashAttention]
        NCCL[NCCL]
    end
    
    subgraph "Runtime"
        CUDA[CUDA Runtime]
        CUDNN[cuDNN]
        CUBLAS[cuBLAS]
    end
    
    subgraph "Frameworks"
        PYTORCH[PyTorch]
        TVM[TVM/Apache]
    end
    
    subgraph "Serving Systems"
        V[vLLM]
        S[SGLang]
        T[TensorRT-LLM]
        M[MLC-LLM]
    end
    
    GPU --> CUDA
    TC --> CUTLASS
    HBM --> FLASHATTN
    NVLINK --> NCCL
    
    CUDA --> PYTORCH
    CUDA --> TVM
    CUTLASS --> FLASHATTN
    
    PYTORCH --> V
    PYTORCH --> S
    CUDA --> T
    TVM --> M
    FLASHATTN --> V
    FLASHATTN --> S
    NCCL --> V
    NCCL --> T
```

## 5. 论文引用网络（核心节点）

```mermaid
graph TD
    ATTN[Attention Is All You Need<br/>2017]
    
    ATTN --> FA1[FlashAttention<br/>2022]
    ATTN --> MQA_P[Multi-Query Attention<br/>2019]
    
    FA1 --> FA2[FlashAttention-2<br/>2023]
    FA2 --> FA3[FlashAttention-3<br/>2024]
    FA1 --> FD[FlashDecoding<br/>2023]
    
    MQA_P --> GQA_P[GQA<br/>2023]
    GQA_P --> LLAMA2[Llama-2<br/>2023]
    
    ATTN --> ORCA[Orca<br/>2022]
    ORCA --> VLLM_P[vLLM/PagedAttention<br/>2023]
    VLLM_P --> SGLANG_P[SGLang/RadixAttention<br/>2024]
    VLLM_P --> DISTSERVE_P[DistServe<br/>2024]
    
    ATTN --> SD_P[Speculative Decoding<br/>2023]
    SD_P --> MEDUSA_P[Medusa<br/>2024]
    MEDUSA_P --> EAGLE_P[EAGLE<br/>2024]
    EAGLE_P --> EAGLE2_P[EAGLE-2<br/>2024]
    
    ATTN --> GPTQ_P[GPTQ<br/>2022]
    GPTQ_P --> AWQ_P[AWQ<br/>2023]
    ATTN --> SQ_P[SmoothQuant<br/>2022]
    SQ_P --> QSERVE_P[QServe<br/>2024]
    
    ATTN --> DSV2[DeepSeek-V2/MLA<br/>2024]
    DSV2 --> DSV3_P[DeepSeek-V3<br/>2024]
```

## 6. 知识领域交叉

| 领域 A | 领域 B | 交叉点 | 代表工作 |
|--------|--------|--------|----------|
| Attention | Memory | KV Cache 管理 | PagedAttention, H2O |
| Quantization | Serving | 量化模型 serving | QServe, Marlin |
| Speculation | Batching | Batch-aware speculation | MineDraft |
| Sparse | Long Context | 稀疏长序列 attention | MInference, StreamingLLM |
| MoE | Distributed | Expert parallelism | DeepSeek-V3 |
| Compression | KV Cache | KV 量化/蒸馏 | KIVI, Gear |
| Scheduling | Disaggregation | P/D 调度 | DistServe, Mooncake |
| Hardware | Kernel | 硬件感知 kernel | FlashAttention-3 (FP8) |

## 7. 研究热度趋势

| 方向 | 2022 | 2023 | 2024 | 2025 | 趋势 |
|------|------|------|------|------|------|
| FlashAttention | ★★ | ★★★ | ★★ | ★ | 成熟，增量改进 |
| KV Cache | ★ | ★★ | ★★★★ | ★★★ | 持续热门 |
| Speculative Decoding | ★ | ★★ | ★★★ | ★★ | 稳定 |
| P/D Disaggregation | - | - | ★★★ | ★★★★ | 快速增长 |
| MLA/DeepSeek | - | - | ★★★ | ★★★★ | 快速增长 |
| Long Context | - | ★ | ★★★ | ★★★★ | 快速增长 |
| MoE Inference | - | ★ | ★★ | ★★★ | 增长 |
| Quantization | ★★ | ★★★ | ★★★ | ★★ | 稳定 |
