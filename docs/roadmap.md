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
- 已为纯 Decode 按精确 batch size 复用固定地址的 input/metadata buffer，并可选捕获 CUDA Graph；
- 已验证同一个 Graph 在 request 顺序、sequence length 和 block table 更新后可重放，并覆盖跨页 Decode；
- 下一步用正式 workload 测量不同 bucket 的命中率、capture 显存成本和 TPOT 收益，再决定默认 bucket；
- GPU 不支持所选 kernel 时给出明确错误或显式回退。

验收：reference 与 GPU backend 的 logits/token 在约定容差内一致；记录 TTFT、TPOT、吞吐、峰值显存、backend、dtype、shape 和 GPU 信息。

## 3. 真实 Qwen3 推理链

- 已接入本地 Hugging Face tokenizer、checkpoint chat template 和增量 detokenization；
- 已实现单文件/分片 SafeTensors 逐 tensor 加载，以及分离 Q/K/V、gate/up 到 fused 权重的映射；
- 已用本地 Qwen3-0.6B 对齐最终 FP32 logits，并跑通 BF16 + FA2 paged-KV 单请求和并发生成；
- 已实现 Qwen3MoE router、softmax/top-k、可选概率归一化和 expert dispatch，并与 Transformers 小 MoE logits 对齐；
- 已增加 temperature、top-k、top-p 和独立 per-request seed；随机结果不受 continuous batch 组合影响；
- 已实现 replicated-token expert 分片：每个 rank 只加载/计算自己的 experts，输出以
  all-reduce 合并；已在云端加载真实 Qwen3-30B-A3B；
- 已保留 naive/sorted 基线，并增加 padded-batched grouped GEMM 以及 token ownership +
  variable all-to-all dispatch/combine reference；两者已通过单进程 oracle、Gloo TP=2
  forward 与 continuous batching；四卡 CUDA A/B 显示 grouped 慢 2.6%～8.2%，all-to-all
  又慢 20.5%～30.2%，因此保持 sorted 默认；已加入 offsets 驱动、融合 SwiGLU 的无
  padding Triton grouped kernel，下一步在四卡完成 token oracle 与同 workload A/B；
- 保留 `naive` dispatch 基线并默认使用 sorted dispatch：一次筛选本 rank assignments 后
  按 expert 分组；SafeTensors 对 packed experts 和 TP projection 直接读取 rank-local slice。

本地 dense 验收已完成：与 Transformers 对齐 layer/logits/greedy token，真实 checkpoint 完成单请求和并发 smoke test。MoE 的小模型 oracle及真实 Qwen3-30B-A3B 四卡 token/显存/性能验收均已完成。具体型号不写死在架构中。

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

- 已建立模型级 TP 基线：`TensorParallelContext`、column/row-parallel linear、
  fused QKV 与 gate/up 的逐段切分，以及 rank-local KV head/cache 布局；
- checkpoint loader 在加载时直接取得当前 rank 的 Q/K/V、gate/up 与 row-parallel
  输入分片，不先在 GPU 上构造完整权重；
- 已用两个 Gloo 进程验证小型 dense Qwen3 的 TP=2 logits 与 TP=1 对齐；embedding
  和 LM head 暂时复制，以先固定 attention/MLP 的通信边界；
- 已建立基础多进程运行时：rank 0 广播 add/abort/step 控制命令，各 rank 镜像
  Scheduler、Radix 元数据和 rank-local KV pool；每轮 forward 前广播并核对 BatchPlan，
  采样后再由 rank 0 广播权威 token；
- 控制消息使用 CPU/Gloo process group，模型 TP collective 可独立使用 NCCL group；
- 已用两个 Gloo 进程跑通 chunked Prefill、混合 Prefill/Decode 和纯 Decode，并与
  TP=1 逐 token 对齐，结束后各 rank 的 cache 完整性一致；
- 已让 MoE 在同一进程组组合 Attention TP 和 expert ownership 分片；packed 与逐-expert
  checkpoint 加载、TP=2 logits 及 continuous batching 均与 TP=1 对齐；
- 已接入 `torchrun` CLI 启动链：从环境解析 global/local rank，每个进程绑定本地设备，
  CUDA 使用 NCCL model group + 独立 Gloo control group；仅 rank 0 构造 tokenizer 和
  GenerationService，其他 rank 进入 worker loop，退出时由 rank 0 广播 shutdown；
- 分离前端/tokenizer、Scheduler/Engine 和 detokenizer 进程；
- 实现 vocab parallel embedding/head；
- 处理 worker 异常、超时和 shutdown，避免静默卡死。

当前单进程 `Scheduler` 仍会拒绝 TP model；TP 必须通过 `TensorParallelScheduler` 让所有
rank 同步进入 collective。已接入 TP NCCL CUDA Graph，并让 `triton_grouped` 通过固定
`batch × top_k` assignment 槽位支持 MoE Graph；CUDA/NCCL capture 和跨 rank 故障恢复
仍待云端验证。真实
Qwen3-30B-A3B 的四卡显存、token、吞吐及 sorted/grouped/all-to-all A/B 均已完成。
不再扩展 TP×EP mesh；下一轮云端测试应逐项启用 Triton、TP Graph 与 MoE Graph。

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
