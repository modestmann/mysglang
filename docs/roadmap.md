# 从零搭建 MySGLang：课程与工程路线图

## 最终目标

实现一个结构清晰的文本大模型推理服务，具备：

- Qwen3 MoE 模型结构和 SafeTensors 权重加载；
- Prefill/Decode、continuous batching 和流式输出；
- Paged KV Cache、Radix prefix cache、Chunked Prefill；
- PyTorch reference attention 与 FlashAttention 高性能 backend；
- CUDA Graph、单机 Tensor Parallel；
- 离线与在线 benchmark、正确性和性能回归测试。

本项目借鉴 Mini-SGLang/SGLang 的思想，但优先保证可解释性。外部高性能算子通过窄接口接入，调度器和缓存系统不依赖某个具体 kernel。

## 每章固定学习方式

每章都以当前工作区的 `/home/sheep/mini-sglang` 为主要源码参照，而不是只讲抽象概念：

1. **源码地图**：先列出 Mini-SGLang 对应文件、核心类和调用方向；
2. **设计动机**：解释它解决的瓶颈、关键不变量和数据形状；
3. **最小复现**：在 MySGLang 中实现更小、可单步调试的版本；
4. **对齐测试**：与无优化 reference、Transformers 或 Mini-SGLang 对齐；
5. **差异与改进**：明确哪些地方忠实借鉴，哪些地方为教学简化，哪些地方有意改进；
6. **亲手练习**：留一个规模可控的 TODO，完成后有明确测试判据；
7. **性能实验**：只改变一个变量，输出可复现的 JSONL 指标。

不会机械复制 Mini-SGLang。阅读它是为了提取接口边界、状态不变量和优化动机；模型级正确性使用 Hugging Face 作为端到端 oracle，Attention 算子保留可在 GPU 上运行的 PyTorch reference backend，并用更严格的测试尽早暴露错误。

## 教学目录

### 第 1 章：推理引擎长什么样

理解控制面与数据面：API、Tokenizer、Scheduler、ModelRunner、KV Cache、Attention backend。

验收：能够画出一次请求的组件边界，配置对象不依赖具体模型 checkpoint。

源码导读已经写入 [01_architecture.md](01_architecture.md)。

### 第 2 章：一个请求的生命周期

定义请求 ID、prompt tokens、sampling params、状态机和增量输出事件。先在单进程内跑通，不急于加入 HTTP。

验收：请求严格经过 `WAITING -> PREFILL -> DECODING -> FINISHED/ABORTED`，非法状态迁移会报错。

### 第 3 章：前 200 行——Forward 与生成循环

实现 RMSNorm、RoPE、GQA、gated MLP、decoder layer、LM head 和 greedy generation。当前 M1 已完成基础版本。

验收：shape、因果性、固定随机种子和终止条件都有测试。

### 第 4 章：KV Cache——从重复前缀到增量 Decode

先实现每请求连续 KV Cache，再分别测量有/无 cache 的 token 计算量。这里更准确的变化是 decode 每步 attention/投影不再重算全部历史，而不是笼统地说所有计算都从 O(n²) 变成 O(n)。

验收：cached 与 uncached 每一步 logits 在容差内相等；记录生成 N token 的延迟曲线。

### 第 5 章：HTTP 与并发请求

实现最小 `/generate` 和 `/v1/chat/completions`，支持 SSE、断连 abort、真实 usage 和 stop token。Tokenizer 与模型执行隔离。

验收：流式/非流式输出一致；中止请求会回收资源；并发客户端测试通过。

### 第 6 章：Continuous Batching 与 Scheduler

每轮只让可运行请求进入 batch；新请求可以在旧请求生成期间加入。实现 token budget、请求准入和公平策略。

相较 Mini-SGLang 的第一项改进：不固定采用“Prefill 永远优先”，而是用等待时间和 decode latency budget 防止任一侧饥饿。

验收：混合长 Prefill/短 Decode 压测中无饥饿，并报告 TTFT、TPOT、吞吐和 p95/p99。

### 第 7 章：Paged KV Cache 与显存管理

实现 block pool、逻辑 block table、按需分配、回收和 OOM-safe admission。页表属于运行时；Attention backend 只消费其只读视图。

验收：随机申请/释放的 property test；无重复页、无泄漏、OOM 前拒绝请求而不是进程崩溃。

### 第 8 章：RadixAttention 与前缀缓存

实现压缩 Radix Tree、页对齐匹配、节点分裂、引用计数和 LRU eviction。

相较 Mini-SGLang 的改进：实现 `reset()`、完整 integrity checker、显式 cache stats，以及不同采样请求共享 prompt KV 的正确性测试。

验收：共享前缀 workload 的计算 token 数下降；reset 后所有页可回收；随机操作保持树和页池不变量。

### 第 9 章：FlashAttention 与 CUDA Graph

定义稳定的 AttentionBackend 协议：PyTorch reference backend 用作算子级 oracle，FlashAttention 2 backend 用于 GPU Prefill/Decode；两者主要在 GPU 上对齐，CPU 只做可选 smoke test。后端负责 kernel metadata，Scheduler 不导入 FlashAttention。

4060（Ada, SM89）优先验证 FA2；云端 Hopper/Blackwell 再选择相应版本。CUDA Graph 先只覆盖 Decode 固定 bucket。

验收：两个 backend 输出对齐；unsupported GPU 给出清晰错误；benchmark 记录 backend、GPU、dtype、shape。

### 第 10 章：多进程与 Tensor Parallelism

实现 column/row parallel linear、vocab parallel embedding/head、NCCL all-reduce/all-gather。控制消息与 tensor 通信分离。

验收：TP=1 与 TP=2 logits 对齐；所有 rank 请求顺序一致；进程异常不会静默卡死。

### 第 11 章：Qwen3 MoE

先用 tiny MoE 验证 router、top-k、expert dispatch、加权合并，再加载真实 Qwen3 MoE checkpoint。第一版采用正确但简单的 expert loop，随后再替换成 fused grouped GEMM；避免在模型正确前调 Triton kernel。

验收：与 Hugging Face 小 batch logits/greedy tokens 对齐；记录 expert 分布和每层路由统计；真实模型云端 smoke test。

最终 checkpoint 在实施本章时按显存、许可证和可获取 GPU 决定。候选默认是 Qwen3 MoE 家族，不把具体型号硬编码进架构。

### 第 12 章：评测与回归

四层评测：

1. 单元正确性：layer/cache/scheduler invariants；
2. 模型正确性：与 Transformers logits/token 对齐；
3. 离线性能：prefill tok/s、decode tok/s、峰值显存；
4. 在线 serving：request throughput、TTFT、TPOT/ITL、E2E、p50/p95/p99。

数据集分三类：可重复 synthetic lengths、共享前缀 synthetic workload、公开对话/长上下文样本。所有结果写入带硬件和 commit 信息的 JSONL。

### 可选章：Speculative Decoding 与分层 KV Cache

这两项放在主链路稳定之后。Speculative decoding 需要 draft/target 和接受算法；Host KV Cache 需要异步 H2D/D2H、预取、回写和分层淘汰。二者不阻塞第一个完整版本。

## 里程碑纪律

每个优化都要回答四个问题：

1. 正确性 oracle 是什么？
2. 哪个测试会在实现错误时失败？
3. 要改善的指标是什么？
4. 在什么 workload 和硬件上测量？

没有 correctness baseline 的性能数字不进入最终报告。
