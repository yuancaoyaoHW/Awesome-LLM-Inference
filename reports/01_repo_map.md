# Awesome-LLM-Inference 仓库概览

## 基本信息

| 属性 | 值 |
|------|-----|
| 仓库名称 | Awesome-LLM-Inference |
| 维护者 | xlite-dev, liyucheng09 |
| 版本 | v2.6 |
| 许可证 | GPLv3.0 |
| 论文总数 | 362 |
| 章节数 | 21 |
| 时间跨度 | 2018.03 ~ 2026.03 |
| 有代码仓库的论文 | 202 |
| 有Stars徽章的项目 | 215 |

## 章节结构与论文数量

| 序号 | 章节名称 | 论文数 |
|------|----------|--------|
| 1 | Trending LLM/VLM Topics | 13 |
| 2 | DeepSeek/MLA Topics | 14 |
| 3 | Multi-GPUs/Multi-Nodes Parallelism | 14 |
| 4 | Disaggregating Prefill and Decoding | 5 |
| 5 | LLM Algorithmic/Eval Survey | 13 |
| 6 | LLM Train/Inference Framework/Design | 28 |
| 7 | Weight/Activation Quantize/Compress | 35 |
| 8 | Continuous/In-flight Batching | 11 |
| 9 | IO/FLOPs-Aware/Sparse Attention | 38 |
| 10 | KV Cache Scheduling/Quantize/Dropping | 54 |
| 11 | Prompt/Context Compression | 13 |
| 12 | Long Context Attention/KV Cache Optimization | 29 |
| 13 | Early-Exit/Intermediate Layer Decoding | 10 |
| 14 | Parallel Decoding/Sampling | 25 |
| 15 | Structured Prune/KD/Weight Sparse | 8 |
| 16 | Mixture-of-Experts(MoE) LLM Inference | 6 |
| 17 | CPU/NPU/FPGA/Mobile Inference | 15 |
| 18 | Non Transformer Architecture | 5 |
| 19 | GEMM/Tensor Cores/WMMA/Parallel | 20 |
| 20 | VLM/Position Embed/Others | 5 |
| 21 | LLM Inference Applications | 1 |

## 🔥 Fire Emoji 热度分析

- 🔥🔥🔥 (极高关注): 23 篇
- 🔥🔥 (高关注): 112 篇
- 🔥 (关注): 133 篇
- 无火标记: 94 篇

### 🔥🔥🔥 极高关注论文 (Top Trending)

| Date | Title | Section |
|------|-------|---------|
| 2026.03 | OneComp | Trending LLM/VLM Topics |
| 2024.04 | Open-Sora | Trending LLM/VLM Topics |
| 2024.04 | Open-Sora Plan | Trending LLM/VLM Topics |
| 2024.05 | DeepSeek-V2 | Trending LLM/VLM Topics |
| 2024.11 | Star-Attention: 11x~ speedup | Trending LLM/VLM Topics |
| 2024.12 | DeepSeek-V3 | Trending LLM/VLM Topics |
| 2025.01 | MiniMax-Text-01 | Trending LLM/VLM Topics |
| 2025.01 | DeepSeek-R1 | Trending LLM/VLM Topics |
| 2024.05 | DeepSeek-V2 | DeepSeek/MLA Topics |
| 2024.12 | DeepSeek-V3 | DeepSeek/MLA Topics |
| 2025.01 | DeepSeek-R1 | DeepSeek/MLA Topics |
| 2025.02 | DeepSeek-NSA | DeepSeek/MLA Topics |
| 2025.02 | FlashMLA | DeepSeek/MLA Topics |
| 2025.02 | DualPipe | DeepSeek/MLA Topics |
| 2025.02 | DeepEP | DeepSeek/MLA Topics |
| 2025.02 | DeepGEMM | DeepSeek/MLA Topics |
| 2025.02 | EPLB | DeepSeek/MLA Topics |
| 2025.02 | 3FS | DeepSeek/MLA Topics |
| 2025.03 | 推理系统 | DeepSeek/MLA Topics |
| 2024.11 | SP: Star-Attention, 11x~ speedup | Multi-GPUs/Multi-Nodes Parallelism |

## 主要系统/框架列表

| 项目名 | 所属章节 | 推荐等级 |
|--------|----------|----------|
| SeerAttention | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️⭐️ |
| OpenMchine | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️⭐️ |
| OneCompression | Trending LLM/VLM Topics | ⭐️⭐️ |
| Open-Sora | Trending LLM/VLM Topics | ⭐️⭐️ |
| Open-Sora-Plan | Trending LLM/VLM Topics | ⭐️⭐️ |
| DeepSeek-V2 | Trending LLM/VLM Topics | ⭐️⭐️ |
| unilm-YOCO | Trending LLM/VLM Topics | ⭐️⭐️ |
| Mooncake | Trending LLM/VLM Topics | ⭐️⭐️ |
| flash-attention | Trending LLM/VLM Topics | ⭐️⭐️ |
| MInference 1.0 | Trending LLM/VLM Topics | ⭐️⭐️ |
| Star-Attention | Trending LLM/VLM Topics | ⭐️⭐️ |
| DeepSeek-V3 | Trending LLM/VLM Topics | ⭐️⭐️ |
| MiniMax-01 | Trending LLM/VLM Topics | ⭐️⭐️ |
| DeepSeek-R1 | Trending LLM/VLM Topics | ⭐️⭐️ |
| FlashMLA | DeepSeek/MLA Topics | ⭐️⭐️ |
| DualPipe | DeepSeek/MLA Topics | ⭐️⭐️ |
| DeepEP | DeepSeek/MLA Topics | ⭐️⭐️ |
| DeepGEMM | DeepSeek/MLA Topics | ⭐️⭐️ |
| EPLB | DeepSeek/MLA Topics | ⭐️⭐️ |
| 3FS | DeepSeek/MLA Topics | ⭐️⭐️ |
| MHA2MLA | DeepSeek/MLA Topics | ⭐️⭐️ |
| TransMLA | DeepSeek/MLA Topics | ⭐️⭐️ |
| deepspeed | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| Megatron-LM | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| RingAttention | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| striped_attention | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| long-context-attention | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| token-ring | Multi-GPUs/Multi-Nodes Parallelism | ⭐️⭐️ |
| DistServe | Disaggregating Prefill and Decoding | ⭐️⭐️ |
| Efficient-LLMs-Survey | LLM Algorithmic/Eval Survey | ⭐️⭐️ |
| LLM-Viewer | LLM Algorithmic/Eval Survey | ⭐️⭐️ |
| ICSF-Survey | LLM Algorithmic/Eval Survey | ⭐️⭐️ |
| vllm | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| TensorRT-LLM | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| deepspeed-fastgen | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| sglang | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| petals | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| lmdeploy | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| mlc-llm | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| lightllm | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| llama.cpp | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| flashinfer | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| Nanoflow | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| siiRL | LLM Train/Inference Framework/Design | ⭐️⭐️ |
| DeepSpeed | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| gptq | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| FasterTransformer | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| smoothquant | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| llm-awq | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| qserve | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| GuidedQuant | Weight/Activation Quantize/Compress | ⭐️⭐️ |
| vAttention | Continuous/In-flight Batching | ⭐️⭐️ |
| vTensor | Continuous/In-flight Batching | ⭐️⭐️ |
| reformer | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| flaxformer | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| SageAttention | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| SqueezedAttention | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| ffpa-attn | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| SpargeAttn | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| MInference | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| SparseFrontier | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| attention-gym | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| APE | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| Block-attention | IO/FLOPs-Aware/Sparse Attention | ⭐️⭐️ |
| nexusquant | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| Compressed-Context-Memory | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| chunk-attention | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| QAQ-KVCacheQuantization | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| Keyformer | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| SqueezeAttention | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| LMCache | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| Palu | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| AdaKV | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| DynamicLLaVA | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| KVzip | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| avp-python | KV Cache Scheduling/Quantize/Dropping | ⭐️⭐️ |
| KVQuant | Long Context Attention/KV Cache Optimization | ⭐️⭐️ |
| HOMER | Long Context Attention/KV Cache Optimization | ⭐️⭐️ |
| LOOK-M | Long Context Attention/KV Cache Optimization | ⭐️⭐️ |
| Quest | Long Context Attention/KV Cache Optimization | ⭐️⭐️ |
| ShadowKV | Long Context Attention/KV Cache Optimization | ⭐️⭐️ |
| EE-LLM | Early-Exit/Intermediate Layer Decoding | ⭐️⭐️ |
| fast_robust_early_exit | Early-Exit/Intermediate Layer Decoding | ⭐️⭐️ |
| EE-Tuning | Early-Exit/Intermediate Layer Decoding | ⭐️⭐️ |
| LLMSpeculativeSampling | Parallel Decoding/Sampling | ⭐️⭐️ |
| Medusa | Parallel Decoding/Sampling | ⭐️⭐️ |
| LookaheadDecoding | Parallel Decoding/Sampling | ⭐️⭐️ |
| TriForce | Parallel Decoding/Sampling | ⭐️⭐️ |
| PEARL | Parallel Decoding/Sampling | ⭐️⭐️ |
| MineDraft | Parallel Decoding/Sampling | ⭐️⭐️ |
| FLAP | Structured Prune/KD/Weight Sparse | ⭐️⭐️ |
| laser | Structured Prune/KD/Weight Sparse | ⭐️⭐️ |
| SDMPrune | Structured Prune/KD/Weight Sparse | ⭐️⭐️ |
| HFPrune | Structured Prune/KD/Weight Sparse | ⭐️⭐️ |
| mixtral-offloading | Mixture-of-Experts(MoE) LLM Inference | ⭐️⭐️ |
| RWKV-LM | Non Transformer Architecture | ⭐️⭐️ |
| mamba | Non Transformer Architecture | ⭐️⭐️ |
| RWKV-CLIP | Non Transformer Architecture | ⭐️⭐️ |
| flash-linear-attention | Non Transformer Architecture | ⭐️⭐️ |
| QUICK | GEMM/Tensor Cores/WMMA/Parallel | ⭐️⭐️ |
| flute | GEMM/Tensor Cores/WMMA/Parallel | ⭐️⭐️ |
| marlin | GEMM/Tensor Cores/WMMA/Parallel | ⭐️⭐️ |
| TritonBench | GEMM/Tensor Cores/WMMA/Parallel | ⭐️⭐️ |
| Triton-distributed | GEMM/Tensor Cores/WMMA/Parallel | ⭐️⭐️ |
| Awesome-LLMs-Evaluation | LLM Algorithmic/Eval Survey | ⭐️ |
| FlexGen | LLM Train/Inference Framework/Design | ⭐️ |
| FlexFlow | LLM Train/Inference Framework/Design | ⭐️ |
| streaming-llm | LLM Train/Inference Framework/Design | ⭐️ |
| LightSeq | LLM Train/Inference Framework/Design | ⭐️ |
| PowerInfer | LLM Train/Inference Framework/Design | ⭐️ |
| inferflow | LLM Train/Inference Framework/Design | ⭐️ |
| prima.cpp | LLM Train/Inference Framework/Design | ⭐️ |
| toolpipe-mcp-server | LLM Train/Inference Framework/Design | ⭐️ |
| FP8-quantization | Weight/Activation Quantize/Compress | ⭐️ |
| bitsandbytes | Weight/Activation Quantize/Compress | ⭐️ |
| SpQR | Weight/Activation Quantize/Compress | ⭐️ |
| SqueezeLLM | Weight/Activation Quantize/Compress | ⭐️ |
| MS-AMP | Weight/Activation Quantize/Compress | ⭐️ |
| LLM-Shearing | Weight/Activation Quantize/Compress | ⭐️ |
| LLM-FP4 | Weight/Activation Quantize/Compress | ⭐️ |
| smoothquantplus | Weight/Activation Quantize/Compress | ⭐️ |
| ABQ-LLM | Weight/Activation Quantize/Compress | ⭐️ |
| TEAL | Weight/Activation Quantize/Compress | ⭐️ |
| VPTQ | Weight/Activation Quantize/Compress | ⭐️ |
| bitnet | Weight/Activation Quantize/Compress | ⭐️ |
| SpotServe | Continuous/In-flight Batching | ⭐️ |
| dynamic-sparse-flash-attention | IO/FLOPs-Aware/Sparse Attention | ⭐️ |
| sparsegpt | IO/FLOPs-Aware/Sparse Attention | ⭐️ |
| MoA | IO/FLOPs-Aware/Sparse Attention | ⭐️ |
| shareAtt | IO/FLOPs-Aware/Sparse Attention | ⭐️ |
| INT-FlashAttention | IO/FLOPs-Aware/Sparse Attention | ⭐️ |
| LTP | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| H2O | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| GEAR | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| SnapKV | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| pythia-mlkv | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| AlignedKV | KV Cache Scheduling/Quantize/Dropping | ⭐️ |
| FastBERT | Early-Exit/Intermediate Layer Decoding | ⭐️ |
| berxit | Early-Exit/Intermediate Layer Decoding | ⭐️ |
| Instructive-Decoding | Parallel Decoding/Sampling | ⭐️ |
| FocusLLM | Parallel Decoding/Sampling | ⭐️ |
| MagicDec | Parallel Decoding/Sampling | ⭐️ |
| admm-pruning | Structured Prune/KD/Weight Sparse | ⭐️ |
| Simba | Structured Prune/KD/Weight Sparse | ⭐️ |
| intel-extension-for-transformers | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| xFasterTransformer | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| nitro | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| off-grid-mobile | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| ram-coffers | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| llama-cpp-power8 | CPU/NPU/FPGA/Mobile Inference | ⭐️ |
| DissectingTensorCores | GEMM/Tensor Cores/WMMA/Parallel | ⭐️ |
| wmma_extension | GEMM/Tensor Cores/WMMA/Parallel | ⭐️ |
| cutlass | GEMM/Tensor Cores/WMMA/Parallel | ⭐️ |
| hadamard_transform | GEMM/Tensor Cores/WMMA/Parallel | ⭐️ |
| transformers | VLM/Position Embed/Others | ⭐️ |
| ByteTransformer | VLM/Position Embed/Others | ⭐️ |
| storyroute | LLM Inference Applications | ⭐️ |

## Mermaid 仓库结构图

```mermaid
graph TD
    A[Awesome-LLM-Inference<br>v2.6] --> B[模型架构优化]
    A --> C[推理系统设计]
    A --> D[内存与计算优化]
    A --> E[解码策略]
    A --> F[硬件适配]
    A --> G[应用]

    B --> B1[Trending LLM/VLM Topics]
    B --> B2[DeepSeek/MLA Topics]
    B --> B3[Non Transformer Architecture]
    B --> B4[Mixture-of-Experts MoE]

    C --> C1[Multi-GPUs/Multi-Nodes Parallelism]
    C --> C2[Disaggregating Prefill and Decoding]
    C --> C3[LLM Train/Inference Framework]
    C --> C4[Continuous/In-flight Batching]

    D --> D1[Weight/Activation Quantize]
    D --> D2[IO/FLOPs-Aware/Sparse Attention]
    D --> D3[KV Cache Scheduling/Quantize]
    D --> D4[Prompt/Context Compression]
    D --> D5[Long Context Attention/KV Cache]

    E --> E1[Early-Exit/Intermediate Layer]
    E --> E2[Parallel Decoding/Sampling]
    E --> E3[Structured Prune/KD/Weight Sparse]

    F --> F1[CPU/NPU/FPGA/Mobile Inference]
    F --> F2[GEMM/Tensor Cores/WMMA]
    F --> F3[VLM/Position Embed/Others]

    G --> G1[LLM Inference Applications]
```
