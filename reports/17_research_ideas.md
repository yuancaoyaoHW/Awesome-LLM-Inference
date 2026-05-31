# 研究选题库

基于 Awesome-LLM-Inference 仓库 362 篇论文的系统分析，提出以下研究选题。每条选题包含明确的问题定义、动机、方法路径、所需论文、实验设计和工程难度评估。

---

## 一、Attention Kernel 优化

### 选题 1: 面向 GQA/MLA 的 Fused Decode Kernel
- **problem**: 当前 FlashDecoding 针对 MHA 设计，GQA（Llama-3）和 MLA（DeepSeek-V3）的 decode kernel 未充分利用 KV head sharing 的内存局部性
- **motivation**: GQA/MLA 已成为主流架构，decode 阶段是 latency 瓶颈，专用 kernel 可减少冗余内存访问
- **possible_method**: 设计 KV-head-aware tiling 策略，将共享同一 KV head 的多个 Q head 在同一 warp 内并行计算，减少 KV cache 重复加载
- **required_papers**: FlashAttention-2, FlashDecoding, GQA, FlashMLA, DeepSeek-V2
- **expected_experiment**: 在 Llama-3-70B (GQA, 8 KV heads) 和 DeepSeek-V3 (MLA) 上对比 FlashDecoding baseline，测量 decode latency (bs=1/8/32)，目标 1.3-1.8x speedup
- **engineering_difficulty**: 4

### 选题 2: 动态稀疏度感知的 Attention Kernel
- **problem**: SeerAttention/MInference 证明 attention matrix 存在可预测的稀疏模式，但当前实现需要两次 pass（预测+计算），overhead 抵消收益
- **motivation**: 若能在单次 kernel 内完成稀疏度预测和选择性计算，可在 prefill 阶段获得 2-4x 加速
- **possible_method**: 在 FlashAttention tiling 框架内嵌入轻量级 block-level importance predictor，跳过低重要性 tile
- **required_papers**: SeerAttention, MInference, SpargeAttn, Sparse Frontier, FlashAttention-2
- **expected_experiment**: 在 128K context 的 Llama-3-8B 上测量 prefill latency 和 perplexity degradation，与 dense FA2 和 MInference 对比
- **engineering_difficulty**: 5

### 选题 3: FP4 Attention 的精度恢复策略
- **problem**: SageAttention-3 实现了 FP4 attention，但在长序列和特定 head 上存在精度损失（PPL +0.5-1.0）
- **motivation**: FP4 可将 attention 计算吞吐提升 2x（Blackwell Tensor Core），但精度是部署障碍
- **possible_method**: 混合精度策略——对 outlier head 使用 FP8，对 normal head 使用 FP4；基于 attention entropy 动态选择精度
- **required_papers**: SageAttention 1/2/3, INT-FlashAttention, TurboAttention, FlashAttention-3
- **expected_experiment**: 在 Llama-3-70B 上测量不同混合比例下的 PPL、MMLU、HumanEval，对比纯 FP8 和纯 FP4
- **engineering_difficulty**: 4

### 选题 4: Prefill-Decode 统一 Kernel 架构
- **problem**: 当前 prefill（compute-bound）和 decode（memory-bound）使用完全不同的 kernel 实现，chunked prefill 场景下需要频繁切换
- **motivation**: SGLang/vLLM 的 chunked prefill 将 prefill 和 decode 混合调度，统一 kernel 可减少 dispatch overhead 和代码复杂度
- **possible_method**: 设计 adaptive tiling kernel，根据 query 数量自动选择 split-K（decode）或 split-Q（prefill）策略
- **required_papers**: FlashAttention-2, FlashDecoding, Sarathi, FlashInfer, FFPA
- **expected_experiment**: 在混合 batch（50% prefill + 50% decode）场景下对比统一 kernel vs 分离 kernel 的 throughput 和 latency
- **engineering_difficulty**: 5

### 选题 5: Ring Attention 与 FlashAttention 的深度融合优化
- **problem**: Ring Attention 在多 GPU 间传递 KV block 时，通信和计算的 overlap 效率受限于 block size 选择
- **motivation**: 百万级 context 需要跨 GPU 的 attention 计算，通信效率直接决定可扩展性
- **possible_method**: 自适应 block size 选择 + 双缓冲流水线 + 基于 NVLink topology 的通信调度
- **required_papers**: Ring Attention, Star Attention, Striped Attention, USP, TokenRing
- **expected_experiment**: 在 8xH100 上测量 1M context 的 prefill latency，对比 Ring Attention、Star Attention 和本方法
- **engineering_difficulty**: 4

---

## 二、KV Cache 压缩与管理

### 选题 6: 基于 Attention Pattern 的自适应 KV Cache 预算分配
- **problem**: 现有 KV cache eviction（H2O, SnapKV）对所有层使用相同策略，但不同层的 attention 模式差异巨大
- **motivation**: 底层倾向 local attention，高层倾向 global attention，统一策略导致信息丢失不均匀
- **possible_method**: 在线 profiling 每层 attention entropy，动态分配每层 KV cache budget（类似 AdaKV 但更细粒度）
- **required_papers**: AdaKV, SqueezeAttention, LayerKV, H2O, SnapKV, DynamicKV
- **expected_experiment**: 在 LongBench 上测量不同 budget 分配策略的 accuracy，对比均匀分配和自适应分配
- **engineering_difficulty**: 3

### 选题 7: KV Cache 的 learned compression codec
- **problem**: 现有 KV quantization（KIVI, KVQuant）使用固定量化方案，未利用 KV cache 的时序冗余和跨层相关性
- **motivation**: 相邻 token 的 KV 向量高度相似（temporal redundancy），可用 delta coding 进一步压缩
- **possible_method**: 轻量级 neural codec（类似 NexusQuant 的 temporal predictive coding），在线训练 predictor，只存储残差
- **required_papers**: NexusQuant, KVQuant, GEAR, ZipCache, KVTC
- **expected_experiment**: 在 Llama-3-8B 128K context 上测量压缩比 vs PPL，对比 KIVI、KVQuant 和本方法
- **engineering_difficulty**: 4

### 选题 8: Cross-Request KV Cache 共享的一致性协议
- **problem**: 多用户共享 prefix（system prompt）时，KV cache 的一致性管理缺乏系统化方案
- **motivation**: 生产环境中 70%+ 请求共享相同 system prompt，高效共享可节省 50%+ 显存
- **possible_method**: 设计 copy-on-write KV cache 管理器，结合 RadixAttention 的 prefix tree 和 reference counting
- **required_papers**: RadixAttention, ChunkAttention, Hydragen, CacheBlend, BatchLLM
- **expected_experiment**: 在多租户场景（100 concurrent users, 80% prefix overlap）下测量 throughput 和 memory utilization
- **engineering_difficulty**: 3

### 选题 9: KV Cache Offloading 的预取策略优化
- **problem**: 长 context 场景下 KV cache 需要 offload 到 CPU/SSD，但 naive prefetch 导致 GPU idle
- **motivation**: 128K+ context 的 KV cache 超过单 GPU 显存，offloading 是必需但 latency 是瓶颈
- **possible_method**: 基于 attention score history 预测下一步需要的 KV block，异步预取到 GPU
- **required_papers**: InfiniGen, ShadowKV, FlexGen, InstInfer, KV Cache Prefetch (Alibaba)
- **expected_experiment**: 在 Llama-3-70B 256K context 上测量 TPOT，对比无 offload、naive offload 和智能预取
- **engineering_difficulty**: 4

### 选题 10: MLA 架构下的 KV Cache 压缩新范式
- **problem**: DeepSeek-V3 的 MLA 已将 KV cache 压缩到 latent space（512d），传统 KV compression 方法不再适用
- **motivation**: MLA 改变了 KV cache 的结构，需要新的压缩/eviction 策略
- **possible_method**: 在 latent space 中进行 importance scoring 和 selective eviction，利用 latent 的低秩特性进一步量化
- **required_papers**: DeepSeek-V2, DeepSeek-V3, FlashMLA, MHA2MLA, TransMLA, X-EcoMLA
- **expected_experiment**: 在 DeepSeek-V3 上测量不同 latent compression ratio 下的 generation quality
- **engineering_difficulty**: 5

---

## 三、Quantization

### 选题 11: 量化误差的层间传播建模与补偿
- **problem**: 逐层量化忽略了误差在深层网络中的累积效应，导致深层模型（70B+）量化后精度下降更严重
- **motivation**: QEP 论文指出量化误差传播是精度损失的主因，但缺乏系统化的补偿方案
- **possible_method**: 建立层间误差传播的线性模型，设计 layer-wise adaptive bit-width 分配，对误差放大层使用更高精度
- **required_papers**: QEP, GPTQ, AWQ, SmoothQuant, GuidedQuant, OneComp
- **expected_experiment**: 在 Llama-3-70B W4A8 上对比均匀量化 vs 自适应 bit-width，测量 PPL 和下游任务精度
- **engineering_difficulty**: 3

### 选题 12: Activation Quantization 的 Outlier 处理统一框架
- **problem**: LLM activation 中存在 magnitude outlier（<1% channels 占 >90% range），不同方法（SmoothQuant, OutlierTune, TEAL）各自处理
- **motivation**: 统一框架可简化部署，自动选择最优 outlier 处理策略
- **possible_method**: 在线 profiling outlier 分布，自动选择 per-channel scaling、channel splitting 或 sparse representation
- **required_papers**: SmoothQuant, SmoothQuant+, OutlierTune, I-LLM, SpQR, ABQ-LLM
- **expected_experiment**: 在 5 个不同架构模型上对比统一框架 vs 各专用方法的 W8A8 精度和推理速度
- **engineering_difficulty**: 3

### 选题 13: W4A4 全 INT4 推理的可行性研究
- **problem**: 当前 W4A8 是主流，但 W4A4 可进一步利用 INT4 Tensor Core（Blackwell），精度是否可接受未知
- **motivation**: W4A4 理论上可将 GEMM 吞吐提升 2x（相比 W4A8），是下一代量化目标
- **possible_method**: 结合 rotation（SpinQuant）+ group quantization + Hadamard transform（BitNet v2）实现 W4A4
- **required_papers**: BitNet a4.8, BitNet v2, SpinQuant, GPTQ, QServe, QUICK
- **expected_experiment**: 在 Llama-3-8B/70B 上实现 W4A4，测量 PPL、MMLU、throughput，对比 W4A8 和 W4A16
- **engineering_difficulty**: 5

### 选题 14: 量化感知的 Speculative Decoding
- **problem**: Draft model 和 target model 使用不同量化精度时，acceptance rate 下降，但原因未被系统研究
- **motivation**: 实际部署中 draft model 通常更激进量化（W4）而 target model 较保守（W8），精度不匹配影响加速比
- **possible_method**: 设计 quantization-aware rejection sampling，在验证时补偿量化引入的分布偏移
- **required_papers**: Speculative Sampling (DeepMind), Medusa, GPTQ, AWQ, Decoding Speculative Decoding
- **expected_experiment**: 测量不同 draft/target 量化组合下的 acceptance rate 和 end-to-end speedup
- **engineering_difficulty**: 3

### 选题 15: KV Cache 量化的 Token-Aware 自适应精度
- **problem**: 现有 KV cache quantization 对所有 token 使用相同 bit-width，但 attention sink token 和 recent token 重要性远高于中间 token
- **motivation**: 对重要 token 保持高精度、对不重要 token 激进压缩，可在相同显存下保持更好精度
- **possible_method**: 基于 cumulative attention score 动态分配 FP16/FP8/INT4/INT2 精度
- **required_papers**: KVQuant, MiKV, QAQ, KIVI, StreamingLLM, H2O
- **expected_experiment**: 在 Llama-3-8B 32K context 上测量不同混合精度策略的 PPL 和显存节省
- **engineering_difficulty**: 3

---

## 四、Speculative Decoding

### 选题 16: Batch-Aware Speculative Decoding 调度
- **problem**: Speculative decoding 在大 batch size 下收益递减（decode 变为 compute-bound），但现有方法未考虑 batch 内的异构性
- **motivation**: 生产环境中 batch 内不同请求的 acceptance rate 差异大，统一 speculation length 浪费计算
- **possible_method**: Per-request adaptive γ（speculation length），基于历史 acceptance rate 动态调整
- **required_papers**: MineDraft, MagicDec, PEARL, Decoding Speculative Decoding, BatchLLM
- **expected_experiment**: 在 vLLM 中实现 adaptive γ，测量混合 workload 下的 throughput 和 P99 latency
- **engineering_difficulty**: 3

### 选题 17: 无 Draft Model 的 Self-Speculative Decoding
- **problem**: 传统 speculative decoding 需要额外的 draft model，增加显存和部署复杂度
- **motivation**: 单模型 self-speculation 可简化部署，适合显存受限场景
- **possible_method**: 利用 early exit（前 N 层输出作为 draft）+ 后续层验证，结合 layer skip 和 confidence estimation
- **required_papers**: S3D, LITE, EE-LLM, Lookahead Decoding, KOALA, LayerSkip
- **expected_experiment**: 在 Llama-3-8B 上对比 self-speculation vs Medusa vs 外部 draft model 的 speedup 和显存开销
- **engineering_difficulty**: 4

### 选题 18: Tree Decoding 的最优树结构搜索
- **problem**: Medusa/EAGLE 使用固定树结构，但最优树结构依赖于 task 和 model 的 token 分布
- **motivation**: 动态树结构可提升 acceptance rate 10-20%，但搜索空间巨大
- **possible_method**: 基于 bandit algorithm 在线学习最优树结构，根据 running acceptance statistics 动态调整
- **required_papers**: Medusa, SpecInfer, TriForce, DeFT, Token Recycling
- **expected_experiment**: 在代码生成和对话两种 task 上对比固定树 vs 动态树的 acceptance rate 和 speedup
- **engineering_difficulty**: 3

### 选题 19: Speculative Decoding 与 KV Cache Compression 的联合优化
- **problem**: Speculative decoding 生成的 draft token 需要分配 KV cache，但大部分会被拒绝，造成 KV cache 浪费
- **motivation**: 在 long context 场景下，draft token 的 KV cache 开销不可忽略
- **possible_method**: 对 draft token 使用低精度 KV cache（INT4），验证通过后再升级为高精度；或延迟 KV cache 写入
- **required_papers**: TriForce, MagicDec, KVQuant, vAttention, Speculative Sampling
- **expected_experiment**: 在 128K context 场景下测量联合优化 vs 独立优化的 throughput 和 memory efficiency
- **engineering_difficulty**: 4

### 选题 20: Multi-Model Speculative Decoding 的 Router 设计
- **problem**: 不同 query 适合不同的 draft model（简单 query 用小 model，复杂 query 用大 model），但缺乏自动路由机制
- **motivation**: 单一 draft model 无法适应所有 query 类型，routing 可提升整体 acceptance rate
- **possible_method**: 轻量级 query classifier 预测最佳 draft model，基于 query embedding 和历史 acceptance rate
- **required_papers**: Cascade Speculative, OSD, Hybrid Inference, Fast Best-of-N, STAND
- **expected_experiment**: 在混合 workload（code + chat + math）上对比单 draft vs multi-draft routing 的 speedup
- **engineering_difficulty**: 3

---

## 五、Distributed Inference

### 选题 21: TP 通信压缩的自适应策略
- **problem**: TP 的 AllReduce 通信在跨节点场景下成为瓶颈，现有压缩方法（INT8）精度损失固定
- **motivation**: 跨节点 TP 的通信延迟比计算延迟高 10-100x，压缩是关键
- **possible_method**: 基于 activation 统计的自适应压缩率——对 outlier-heavy 层使用 FP8，对 smooth 层使用 INT4
- **required_papers**: Communication Compression for TP (recogni.com), TP-Aware Dequantization, Megatron-LM
- **expected_experiment**: 在 2-node 16xH100 上测量 Llama-3-70B TP=16 的 latency，对比无压缩、固定 INT8、自适应压缩
- **engineering_difficulty**: 4

### 选题 22: EP + TP 混合并行的最优配置搜索
- **problem**: DeepSeek-V3 等 MoE 模型需要 EP + TP 组合，但最优配置依赖于硬件拓扑和模型结构
- **motivation**: 错误的并行配置可导致 2-3x 性能损失，自动搜索可降低部署门槛
- **possible_method**: 建立 cost model（计算 + 通信 + 内存），基于 ILP 或 DP 搜索最优 EP/TP/PP 配置
- **required_papers**: DeepSeek-V3, DeepEP, EPLB, MegaScale-Infer, Megatron-LM
- **expected_experiment**: 在不同集群配置（8/16/32/64 GPU）上对比自动搜索 vs 人工配置的 throughput
- **engineering_difficulty**: 4

### 选题 23: Disaggregated Prefill/Decode 的动态负载均衡
- **problem**: P/D 分离架构中，prefill 和 decode 的负载比随 workload 变化，静态分配导致资源浪费
- **motivation**: 实际 workload 的 input/output length 分布高度动态，需要弹性调度
- **possible_method**: 基于 queue length 和 SLO violation rate 的在线 resource rebalancing，支持 GPU 在 P/D 角色间动态切换
- **required_papers**: DistServe, Mooncake, Splitwise, MegaScale-Infer, DynamoLLM
- **expected_experiment**: 在模拟真实 workload trace 上测量 SLO attainment rate 和 GPU utilization
- **engineering_difficulty**: 4

### 选题 24: 异构集群上的 LLM 推理调度
- **problem**: 实际数据中心包含多代 GPU（A100/H100/H200），现有框架假设同构集群
- **motivation**: 异构集群利用率低，但硬件升级是渐进的，混合部署是现实需求
- **possible_method**: 将 prefill 分配给高算力 GPU（H100），decode 分配给高带宽 GPU（H200），基于 roofline 模型调度
- **required_papers**: DynamoLLM, NanoFlow, DistServe, Splitwise, Decentralized LLM
- **expected_experiment**: 在混合 A100+H100 集群上对比异构调度 vs 同构调度的 cost-efficiency
- **engineering_difficulty**: 5

### 选题 25: 多节点 KV Cache 传输的零拷贝优化
- **problem**: P/D 分离架构中，KV cache 从 prefill node 传输到 decode node 的延迟高（RDMA overhead）
- **motivation**: KV cache 传输延迟直接影响 TTFT，是 P/D 分离的主要开销
- **possible_method**: 利用 GPUDirect RDMA + KV cache 压缩 + pipeline 传输，减少传输量和延迟
- **required_papers**: Mooncake, KVDirect, MemServe, CacheGen, 3FS
- **expected_experiment**: 测量不同 KV cache 大小下的传输延迟，对比 naive copy、GPUDirect、压缩传输
- **engineering_difficulty**: 5

---

## 六、Serving & Scheduling

### 选题 26: SLO-Aware 的多目标调度器
- **problem**: 现有调度器优化单一目标（throughput 或 latency），但生产环境需要同时满足多种 SLO（TTFT < 200ms, TPOT < 50ms）
- **motivation**: SLO violation 直接影响用户体验和收入，需要 Pareto-optimal 调度
- **possible_method**: 将调度建模为多目标优化问题，使用 online learning 动态调整 priority weights
- **required_papers**: SJF Scheduling, Automatic Inference Engine Tuning, BatchLLM, DynamoLLM, DistServe
- **expected_experiment**: 在 ShareGPT trace 上测量不同 SLO 组合下的 violation rate 和 throughput
- **engineering_difficulty**: 4

### 选题 27: Prefix-Aware Batching 的最优分组策略
- **problem**: BatchLLM 证明 prefix sharing 可提升 batch throughput，但最优分组（哪些请求放一起）是 NP-hard
- **motivation**: 大规模 batch inference 场景（offline evaluation）中，prefix 分组可节省 30-50% 计算
- **possible_method**: 基于 prefix tree 的贪心分组 + 动态规划优化 batch 内 prefix overlap 最大化
- **required_papers**: BatchLLM, RadixAttention, Hydragen, ChunkAttention, CacheBlend
- **expected_experiment**: 在 10K 请求的 batch 上对比随机分组、贪心分组和 DP 分组的 throughput
- **engineering_difficulty**: 3

### 选题 28: Continuous Batching 中的公平性保证
- **problem**: Continuous batching 中长序列请求可能被短序列请求"饿死"（starvation），缺乏公平性机制
- **motivation**: 多租户场景下公平性是 SLA 要求，但现有框架缺乏系统化的公平性保证
- **possible_method**: 引入 token-level fair queuing（类似网络中的 WFQ），按 token 消耗量分配 GPU 时间
- **required_papers**: Orca, vLLM, SGLang, FastServe, SpotServe
- **expected_experiment**: 在多租户场景下测量不同公平性策略的 tail latency 和 throughput 影响
- **engineering_difficulty**: 3

### 选题 29: 请求长度预测驱动的调度优化
- **problem**: SJF 调度需要知道请求的 output length，但 LLM 输出长度不可预知
- **motivation**: 准确的长度预测可将调度效率提升 20-40%（减少 head-of-line blocking）
- **possible_method**: 轻量级 length predictor（基于 input embedding 的 MLP），结合 confidence-based fallback
- **required_papers**: SJF Scheduling (UCSD), Efficient LLM Scheduling by Learning to Rank, Splitwise
- **expected_experiment**: 在 ShareGPT/LMSYS trace 上测量预测准确率和调度效率提升
- **engineering_difficulty**: 2

### 选题 30: Auto-Scaling 与 KV Cache 迁移的联合优化
- **problem**: 弹性扩缩容时，新实例需要重建 KV cache（cold start），导致 TTFT spike
- **motivation**: 云环境下 auto-scaling 是标配，但 KV cache 的有状态性使得扩容延迟高
- **possible_method**: KV cache 快照 + 增量迁移，结合 prefix cache 预热和 request routing
- **required_papers**: SpotServe, MemServe, CacheGen, Mooncake, vTensor
- **expected_experiment**: 模拟 burst traffic 场景，测量扩容后的 TTFT 恢复时间
- **engineering_difficulty**: 4

---

## 七、MoE Inference

### 选题 31: MoE Expert 的动态 Offloading 策略
- **problem**: MoE 模型（Mixtral-8x7B, DeepSeek-V3）的 expert 参数量大，全部加载到 GPU 显存不现实
- **motivation**: 实际推理中每个 token 只激活 2/8 experts，大部分 expert 处于 idle 状态
- **possible_method**: 基于 routing history 预测下一步激活的 expert，异步预取到 GPU，idle expert offload 到 CPU
- **required_papers**: Mixtral Offloading, DeepSeek-V2, PowerInfer, MoE Survey (HKU), DeepEP
- **expected_experiment**: 在单 GPU 上运行 Mixtral-8x7B，对比全加载、naive offload 和预测性 offload 的 latency
- **engineering_difficulty**: 3

### 选题 32: Expert Parallelism 的负载均衡优化
- **problem**: MoE routing 导致 expert 负载不均（hot expert problem），EP 下某些 GPU 成为瓶颈
- **motivation**: DeepSeek-V3 的 EPLB 是静态均衡，动态 workload 下仍有不均
- **possible_method**: 在线 expert replication——对 hot expert 动态创建副本，分散到多个 GPU
- **required_papers**: EPLB, DeepEP, DeepSeek-V3, MegaScale-Infer, MoE Inference (UCSD)
- **expected_experiment**: 在 DeepSeek-V3 推理中测量 expert 负载方差和 end-to-end latency
- **engineering_difficulty**: 5

### 选题 33: MoE 模型的 Expert-Level Quantization
- **problem**: 不同 expert 的重要性和激活频率不同，统一量化精度不是最优
- **motivation**: Hot expert 需要高精度（频繁使用），cold expert 可激进量化（节省显存）
- **possible_method**: 基于 expert activation frequency 和 output sensitivity 分配 W8/W4/W2 精度
- **required_papers**: DeepSeek-V2, GPTQ, AWQ, Mixtral Offloading, WINT8/4
- **expected_experiment**: 在 Mixtral-8x7B 上对比均匀 W4 vs 自适应 W8/W4/W2 的精度和显存
- **engineering_difficulty**: 3

### 选题 34: MoE + Speculative Decoding 的协同设计
- **problem**: MoE 模型的 speculative decoding 面临独特挑战——draft model 难以预测 expert routing
- **motivation**: MoE 模型的 decode latency 受 expert dispatch overhead 影响，speculation 可 amortize
- **possible_method**: 使用 shared expert 作为 draft（DeepSeek-V3 的 shared expert），避免 routing overhead
- **required_papers**: DeepSeek-V3, Medusa, SpecInfer, MoE-Mamba, MegaScale-Infer
- **expected_experiment**: 在 DeepSeek-V3 上实现 shared-expert-as-draft，测量 acceptance rate 和 speedup
- **engineering_difficulty**: 4

### 选题 35: MoE 推理的通信-计算 Overlap 优化
- **problem**: EP 中 all-to-all 通信与 expert 计算的 overlap 效率低，特别是在小 batch 场景
- **motivation**: DeepSeek-V3 的推理系统依赖高效的 EP 通信，overlap 是性能关键
- **possible_method**: 细粒度 pipeline——将 expert 计算分为多个 micro-step，与通信交错执行
- **required_papers**: DeepEP, DualPipe, Triton-distributed, DeepSeek-V3, MegaScale-Infer
- **expected_experiment**: 在 8xH100 上测量不同 overlap 策略下的 MoE layer latency
- **engineering_difficulty**: 5

---

## 八、Long Context

### 选题 36: 百万级 Context 的分层 KV Cache 管理
- **problem**: 1M+ context 的 KV cache 无法全部放入 GPU 显存，需要多级存储（GPU/CPU/SSD）
- **motivation**: GPT-4 Turbo 128K、Gemini 1M 等长 context 模型的部署需求
- **possible_method**: 基于 attention locality 的分层缓存——recent window 在 GPU，medium-range 在 CPU，long-range 在 SSD
- **required_papers**: InfiniGen, ShadowKV, InstInfer, FlexGen, RetrievalAttention, REFORM
- **expected_experiment**: 在 1M context 上测量不同分层策略的 TPOT 和 accuracy（Needle-in-Haystack）
- **engineering_difficulty**: 5

### 选题 37: Chunk-Level Attention 的精度-效率 Pareto 分析
- **problem**: Block Attention、APE 等方法将 long context 分 chunk 独立编码，但 cross-chunk 信息损失未被系统量化
- **motivation**: 理解精度-效率 tradeoff 的 Pareto frontier 可指导实际部署决策
- **possible_method**: 系统性实验——变化 chunk size、overlap ratio、cross-chunk attention 比例，绘制 Pareto curve
- **required_papers**: Block-Attention, APE, Star Attention, Infini-attention, HOMER
- **expected_experiment**: 在 RULER/LongBench 上测量不同配置的 accuracy 和 prefill latency
- **engineering_difficulty**: 2

### 选题 38: 动态 Context Window 的自适应选择
- **problem**: 不同 query 需要的有效 context 长度不同，固定 window 浪费计算或丢失信息
- **motivation**: 90% 的 query 只需要 <4K 有效 context，但 10% 需要全部 128K
- **possible_method**: 轻量级 relevance scorer 预测每个 query 的有效 context range，动态裁剪
- **required_papers**: LLMLingua, Selective-Context, Quest, CritiPrefill, 500xCompressor
- **expected_experiment**: 在 multi-document QA 上测量动态 window vs 固定 window 的 accuracy 和 latency
- **engineering_difficulty**: 3

### 选题 39: Streaming Inference 的 KV Cache 生命周期管理
- **problem**: StreamingLLM 使用 attention sink + sliding window，但丢弃的 token 信息不可恢复
- **motivation**: 对话系统需要无限长 context，但显存有限
- **possible_method**: 将 evicted KV cache 压缩存储到 CPU，需要时通过 retrieval 恢复（类似 Infini-attention 的 compressive memory）
- **required_papers**: StreamingLLM, Infini-attention, REFORM, Compressed Context Memory, SentenceVAE
- **expected_experiment**: 在 multi-turn 对话（100+ turns）上测量信息保留率和 latency
- **engineering_difficulty**: 4

### 选题 40: Cross-Layer KV Sharing 的最优共享模式搜索
- **problem**: CLA 证明相邻层可共享 KV cache，但最优共享模式（哪些层共享）未被系统研究
- **motivation**: 最优共享可在 2x 压缩的同时保持精度，但搜索空间是指数级
- **possible_method**: 基于 layer similarity（CKA/cosine）的贪心搜索 + 微调恢复
- **required_papers**: CLA, MLKV, YOCO, MiniCache, Slim Attention
- **expected_experiment**: 在 Llama-3-8B 上搜索最优共享模式，对比 CLA 的均匀共享
- **engineering_difficulty**: 3

---

## 九、Hardware-Aware 优化

### 选题 41: NPU 上的 LLM 推理 Kernel 自动生成
- **problem**: NPU（华为 Ascend、高通 Hexagon）缺乏成熟的 attention kernel，手写 kernel 成本高
- **motivation**: 端侧 AI 需求增长，NPU 是主要推理硬件，但软件生态落后 CUDA
- **possible_method**: 基于 TVM/MLIR 的 auto-tuning framework，自动生成 NPU-optimized attention kernel
- **required_papers**: FastAttention (Huawei), NITRO, MLC-LLM, Transformer-Lite, FlightLLM
- **expected_experiment**: 在 Ascend 910B 上对比自动生成 kernel vs 手写 kernel 的性能
- **engineering_difficulty**: 5

### 选题 42: 移动端 LLM 的内存带宽优化
- **problem**: 移动 GPU 内存带宽远低于桌面 GPU（50-100 GB/s vs 2-3 TB/s），decode 极慢
- **motivation**: 端侧 LLM（Llama-3-1B/3B）的用户体验依赖 TPOT < 100ms
- **possible_method**: 极致量化（W2/W3）+ operator fusion + memory-mapped weight loading
- **required_papers**: Transformer-Lite, MLC-LLM, llama.cpp, Off Grid, BitNet v2
- **expected_experiment**: 在 Snapdragon 8 Gen 3 上测量 Llama-3-1B 的 TPOT，对比不同量化和 fusion 策略
- **engineering_difficulty**: 4

### 选题 43: CPU 推理的 SIMD 指令集自适应优化
- **problem**: 不同 CPU 支持不同 SIMD 指令集（AVX-512, AMX, SVE），llama.cpp 的手动适配成本高
- **motivation**: CPU 推理是边缘部署的主要方案，自动适配可降低维护成本
- **possible_method**: 运行时检测 CPU 特性，自动选择最优 GEMM kernel（类似 BLIS 的 micro-kernel 选择）
- **required_papers**: llama.cpp, xFasterTransformer, Intel Extension for Transformers, prima.cpp
- **expected_experiment**: 在 Intel Xeon (AMX) 和 ARM Neoverse (SVE) 上对比自适应 vs 固定 kernel 的 throughput
- **engineering_difficulty**: 3

### 选题 44: FPGA 上的 Sparse Attention 加速器设计
- **problem**: FPGA 适合不规则计算（sparse attention），但缺乏针对 LLM sparse pattern 的加速器设计
- **motivation**: FPGA 的可重构性适合动态稀疏模式，可能比 GPU 更高效
- **possible_method**: 设计 sparse attention 的 dataflow architecture，支持动态 sparsity pattern
- **required_papers**: FlightLLM, SeerAttention, MInference, SpargeAttn, CHESS
- **expected_experiment**: 在 Xilinx Alveo U280 上实现 sparse attention，对比 GPU 的 energy efficiency
- **engineering_difficulty**: 5

### 选题 45: GPU Tensor Core 的非标准数据格式利用
- **problem**: Tensor Core 支持 FP8/INT8/FP4，但 LLM 的 activation 分布不适合标准格式
- **motivation**: HiFloat8 等非标准格式可能更适合 LLM 的 outlier 分布
- **possible_method**: 设计 LLM-aware 数据格式（动态 exponent bias），利用 Tensor Core 的 MMA 指令实现
- **required_papers**: HiFloat8, FP8 Formats, DeepGEMM, CUTLASS/CuTe, Tensor Core Programmability
- **expected_experiment**: 在 H100 上对比标准 FP8 vs 自定义格式的 GEMM throughput 和量化精度
- **engineering_difficulty**: 5

---

## 十、跨方向融合

### 选题 46: Quantization + Sparse Attention 的联合优化
- **problem**: 量化和稀疏 attention 分别优化，但联合使用时可能产生精度叠加损失
- **motivation**: 同时使用两种优化可获得乘法级加速（2x × 3x = 6x），但精度风险未知
- **possible_method**: 设计 joint calibration 流程，同时优化量化参数和稀疏 mask
- **required_papers**: SageAttention, SpargeAttn, GPTQ, AWQ, INT-FlashAttention, TurboAttention
- **expected_experiment**: 在 Llama-3-70B 上测量 W4A8 + 50% sparse attention 的联合精度损失
- **engineering_difficulty**: 4

### 选题 47: Disaggregated Serving + Speculative Decoding 的架构设计
- **problem**: P/D 分离架构中，speculative decoding 的 draft-verify 循环跨越 prefill/decode 节点，通信开销大
- **motivation**: 两种优化各自有效，但组合使用的架构设计未被研究
- **possible_method**: 将 draft model 部署在 decode node 本地，verify 利用 decode node 的 target model forward
- **required_papers**: DistServe, Mooncake, Speculative Sampling, MagicDec, Splitwise
- **expected_experiment**: 在 P/D 分离集群上实现 speculative decoding，测量 vs 非分离架构的 speedup
- **engineering_difficulty**: 5

### 选题 48: Early Exit + KV Cache Compression 的协同设计
- **problem**: Early exit 在浅层输出 token，但深层的 KV cache 仍被分配，造成浪费
- **motivation**: 如果 token 在第 N 层 exit，第 N+1 到 L 层的 KV cache 可以释放
- **possible_method**: 设计 exit-aware KV cache manager，动态释放未使用层的 KV cache
- **required_papers**: EE-LLM, LITE, FREE, LayerKV, SqueezeAttention
- **expected_experiment**: 在 Llama-3-8B 上测量 early exit + KV release 的显存节省和 throughput 提升
- **engineering_difficulty**: 3

### 选题 49: Non-Transformer (Mamba/RWKV) + Speculative Decoding
- **problem**: Mamba/RWKV 的线性复杂度使其适合作为 draft model，但与 Transformer target 的分布匹配度未知
- **motivation**: Mamba 的 O(1) per-token cost 使其成为理想 draft model，但架构差异可能降低 acceptance rate
- **possible_method**: 训练 Mamba draft model 蒸馏 Transformer target 的输出分布，最大化 acceptance rate
- **required_papers**: Mamba, RWKV, Mamba Drafters, Speculative Sampling, Medusa
- **expected_experiment**: 对比 Mamba-130M draft vs Llama-68M draft 在 Llama-3-8B target 上的 acceptance rate 和 wall-clock speedup
- **engineering_difficulty**: 3

### 选题 50: 全栈优化：Quantization + Speculation + Sparse + Distributed 的统一框架
- **problem**: 现有优化技术各自独立实现，组合使用时存在兼容性问题和性能回退
- **motivation**: 实际部署需要同时使用多种优化，统一框架可简化部署和最大化收益
- **possible_method**: 设计 composable optimization framework，定义各优化的接口和约束，自动搜索最优组合
- **required_papers**: vLLM, SGLang, TensorRT-LLM, QServe, MagicDec, SpargeAttn, DeepSeek-V3
- **expected_experiment**: 在 Llama-3-70B serving 场景下对比单一优化 vs 组合优化的 throughput 和 cost/token
- **engineering_difficulty**: 5

### 选题 51: Context Parallelism + KV Cache Quantization 的联合设计
- **problem**: CP 将 KV cache 分布到多 GPU，量化可减少每 GPU 的 KV cache 大小，但联合使用时通信格式需要统一
- **motivation**: 1M context 需要 CP，同时 KV cache 量化可减少跨 GPU 通信量
- **possible_method**: 在 CP 通信中直接传输量化后的 KV（INT4/FP8），接收端 dequantize
- **required_papers**: Ring Attention, Star Attention, KVQuant, GEAR, Context Parallelism (Meta)
- **expected_experiment**: 在 8xH100 上测量 1M context 的 CP + KV quant vs CP alone 的 latency 和 accuracy
- **engineering_difficulty**: 4

### 选题 52: Serving Framework 的 Kernel Auto-Selection
- **problem**: 不同 request 特征（seq_len, batch_size, model）适合不同 kernel 实现，但框架使用固定 kernel
- **motivation**: FlashAttention 在 short seq 不如 cuBLAS，FlashDecoding 在 small batch 不如 naive attention
- **possible_method**: 基于 request 特征的 runtime kernel dispatcher，使用 lookup table 或 ML predictor 选择最优 kernel
- **required_papers**: FlashAttention-2, FlashDecoding, FlashInfer, Flex Attention, FFPA
- **expected_experiment**: 在混合 workload 上测量 auto-selection vs 固定 kernel 的 throughput 提升
- **engineering_difficulty**: 3

### 选题 53: Prefill Chunking 的最优 Chunk Size 自适应
- **problem**: Chunked prefill 的 chunk size 是超参数，太大影响 decode latency，太小增加 scheduling overhead
- **motivation**: 最优 chunk size 依赖于当前 batch 状态、GPU 利用率和 SLO 要求
- **possible_method**: 基于 queuing theory 建模，动态调整 chunk size 以最小化 P99 latency
- **required_papers**: Sarathi, SGLang, vLLM, Automatic Inference Engine Tuning, CritiPrefill
- **expected_experiment**: 在不同 load level 下对比固定 chunk size vs 自适应 chunk size 的 P99 TPOT
- **engineering_difficulty**: 3

### 选题 54: 面向 Test-Time Scaling 的推理系统优化
- **problem**: DeepSeek-R1 等 reasoning model 生成极长 CoT（10K+ tokens），现有系统未针对此场景优化
- **motivation**: Test-time compute scaling 是新趋势，需要系统级支持
- **possible_method**: 长输出的 speculative decoding + 动态 KV cache 管理 + early termination detection
- **required_papers**: DeepSeek-R1, STAND, Fast Best-of-N, MagicDec, StreamingLLM
- **expected_experiment**: 在 math reasoning task 上测量优化后的 end-to-end latency 和 cost
- **engineering_difficulty**: 4

### 选题 55: LLM Inference 的 Carbon-Aware Scheduling
- **problem**: LLM 推理的能耗巨大，但调度未考虑碳排放和电力成本的时空变化
- **motivation**: 可持续 AI 是行业趋势，carbon-aware 调度可在不影响 SLO 的前提下减少碳排放
- **possible_method**: 将 carbon intensity 作为调度约束，在低碳时段预计算 KV cache，高碳时段减少 batch size
- **required_papers**: DynamoLLM, Decentralized LLM, SpotServe, NanoFlow
- **expected_experiment**: 基于真实电网碳强度数据，模拟 carbon-aware vs carbon-agnostic 调度的碳排放差异
- **engineering_difficulty**: 3

---

## 统计

| 方向 | 选题数 |
|------|--------|
| Attention Kernel | 5 |
| KV Cache | 5 |
| Quantization | 5 |
| Speculative Decoding | 5 |
| Distributed Inference | 5 |
| Serving & Scheduling | 5 |
| MoE Inference | 5 |
| Long Context | 5 |
| Hardware-Aware | 5 |
| 跨方向融合 | 10 |
| **总计** | **55** |

---

## 工程难度分布

| 难度 | 选题数 | 说明 |
|------|--------|------|
| 2 | 2 | 主要是实验和分析工作 |
| 3 | 18 | 需要修改现有框架，中等 CUDA/系统编程 |
| 4 | 20 | 需要深入 kernel 开发或系统设计 |
| 5 | 15 | 需要全栈能力，涉及硬件理解和大规模系统 |

---

## 推荐优先级

**短期可执行（3-6 月）**：选题 6, 8, 14, 15, 16, 18, 27, 29, 37, 48（难度 2-3，有明确 baseline）

**中期研究（6-12 月）**：选题 1, 7, 9, 10, 11, 17, 19, 21, 23, 26, 31, 33, 39, 40, 46, 52, 53（难度 3-4，需要系统实现）

**长期攻关（12+ 月）**：选题 2, 4, 13, 22, 24, 25, 32, 35, 36, 41, 44, 45, 47, 50（难度 4-5，需要深入创新）
