# MySGLang 工程路线图

当前 paged + radix 主路径已经建立。后续工作不再按课程章节堆叠平行实现，而是围绕一条正式推理链逐步替换 reference 组件。

## 基本原则

每项优化都必须明确：

1. 正确性 oracle 是什么；
2. 哪个测试能暴露资源或数值错误；
3. 要改善的指标是什么；
4. workload、模型、dtype、硬件和 commit 是什么。

没有 correctness baseline 和环境元数据的性能数字不进入正式结论。

## 1. 当前基线（已完成）

- 对外调度接口收敛为 `Scheduler` / `SchedulerConfig`，服务接口收敛为 `GenerationService` / `GenerationSession`；
- Scheduler 内部固定使用 continuous batching、paged KV pool 和 radix prefix cache，不再选择旧后端；
- `GenerationService.start()` 只创建并校验 session，首次消费时才 enqueue 和占用运行资源；
- Prompt Prefill 完成后立即发布完整页，后到请求可在发布者仍 Decode 时复用；
- Scheduler 不保存无限增长的逐步记录，只累计包括 `prefill_input_tokens` 在内的统计；
- 核心回归覆盖 Request 状态、allocator/Radix 不变量、abort 清理和生成对齐。

当前保证：主入口没有隐式后端选择；未消费 session 不进入 Scheduler；正常完成、异常和 abort 后页面与 handle 全部回收。

## 2. GPU AttentionBackend（ragged Prefill 已完成）

- 已把 PyTorch gather/pad 路径收敛为 `TorchAttentionBackend` correctness oracle；
- 已接入 `FlashAttentionBackend`，Prefill/Decode 直接消费物理 KV pool、block tables 和 sequence lengths；
- 已在 RTX 4060 Laptop（SM89）验证 uncached、paged Prefill 和变长 batch Decode 与 reference 对齐；
- 已用 `PagedKVBatch` 把 block table、旧长度和追加范围提升为一次 forward 构造、所有层复用的 metadata；
- 已实现 flattened multi-request Prefill：slot mapping 写入新 K/V，FA2 paged-varlen kernel 直接读取历史页；
- 已实现 Prefill/Decode 混合 forward：已有 Decode 每轮推进一个 token，Prefill 在独立 token budget 内打包；
- 已按采样位置裁剪最终 hidden states，只为 Decode 和已完成 Prompt 的末尾 token 计算 logits；未完成 Prefill 仅更新 KV；
- 下一步复用 metadata buffer，并为固定 Decode bucket 加入 CUDA Graph；
- GPU 不支持所选 kernel 时给出明确错误或显式回退。

验收：reference 与 GPU backend 的 logits/token 在约定容差内一致；记录 TTFT、TPOT、吞吐、峰值显存、backend、dtype、shape 和 GPU 信息。

## 3. 真实 Qwen3 推理链

- 接入 Hugging Face tokenizer、chat template 和增量 detokenization；
- 实现 SafeTensors 权重加载及 fused QKV、gate/up 映射；
- 先完成真实 dense Qwen3 对齐，再实现 tiny MoE router/top-k/expert dispatch；
- 最后加载可用的 Qwen3 MoE checkpoint，并逐步替换简单 expert loop；
- 增加 temperature、top-k、top-p 和可复现随机采样。

验收：与 Transformers 对齐 layer/logits/greedy token；真实 checkpoint 完成单请求和并发 smoke test。具体型号按许可证、显存和可用 GPU 选择，不写死在架构中。

## 4. Scheduler 与 Cache 进阶

- 用实测代价模型改进 ragged Prefill 的 token budget 分配；
- 用实测 TTFT/TPOT deadline 和 batch cost 替换固定公平上限；
- 避免容量受限时 waiting 队首大请求造成小请求 head-of-line blocking；
- 加入 forward/scheduler overlap；
- 研究 reservation overcommit、active-request preemption 和可选 host swap；
- 给 prefix key 增加 model/version/tenant namespace；
- 评估 endpoint hidden/logits cache，或专门 replay 路径；
- 只有收益明确时才实现部分页共享与 copy-on-write。

验收：持续混合长 Prefill/短 Decode workload 无饥饿；资源压力下不崩溃、不泄漏、不读取 stale KV；优化前后输出一致。

## 5. 多进程与 Tensor Parallel

- 分离前端/tokenizer、Scheduler/Engine 和 detokenizer 进程；
- 实现 column/row parallel linear、vocab parallel embedding/head；
- 控制消息与 NCCL tensor 通信分离；
- 由 rank 0 广播 batch plan，并对各 rank 顺序做 hash/assertion；
- 处理 worker 异常、超时和 shutdown，避免静默卡死。

验收：TP=1 与 TP=2 logits/token 对齐；所有 rank 对请求顺序、页表和采样位置达成一致。

## 6. 正式客户端与评测

模型主链可用后再建立稳定的请求客户端和评测集：

- OpenAI-compatible 普通与流式客户端，覆盖取消、超时和并发；
- 可重复 synthetic lengths；
- 共享 system prompt 的 prefix-cache workload；
- 公开对话、长上下文和 MoE 路由样本；
- 离线指标：Prefill/Decode tok/s、峰值显存；
- 在线指标：request throughput、TTFT、TPOT/ITL、E2E、p50/p95/p99；
- 每次结果写入带模型、sampling、硬件、软件版本和 git commit 的 JSONL。

单元测试负责状态和资源不变量，模型 oracle 负责数值正确性，客户端/数据集负责端到端质量与性能；三者不能互相替代。

## 7. 可选方向

- speculative decoding；
- host/disk 分层 KV Cache 与异步预取；
- cache-aware routing 和跨实例前缀复用；
- 量化、LoRA、grammar constrained decoding；
- 更高性能的 fused MoE/grouped GEMM。

这些能力不阻塞第一个可评测的真实 Qwen3 serving 版本。
