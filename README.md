# MySGLang

MySGLang 是一个用于理解和验证 LLM 推理系统的精简实现。当前主路径已经从单请求生成演进为：

- 显式请求状态与增量输出事件；
- continuous batching、chunked prefill 和动态 decode batch；
- 共享 Paged KV Cache、容量预留和 OOM-safe admission；
- page-aligned Radix prefix cache、引用保护与按需淘汰；
- HTTP/JSON、SSE 流式输出和 abort 清理。

项目目前仍以 tiny dense Qwen3 和 PyTorch reference attention 验证调度、缓存与模型语义，不是完整的生产推理框架。下一阶段才会接入真实模型、GPU attention backend 和正式评测。

## 当前主调用链

```text
HTTP / caller
  -> GenerationService.start
       -> tokenize + validate
       -> GenerationSession（此时尚未入队）
  -> 首次消费 GenerationSession
       -> Scheduler.add
       -> 注册 request_id 对应的输出 Queue
       -> 单一 scheduler worker
            -> Scheduler
                 -> paged + radix cache
                 -> TinyCausalLM
  <- IncrementalOutput / SSE
```

`GenerationService.start()` 只校验请求并返回一次性消费的 `GenerationSession`。直到调用方首次迭代 session，请求才进入 Scheduler，并创建 Queue、启动 worker；从未消费的 session 不占调度槽、KV reservation 或物理页。

全局 `Scheduler` 决定下一次 forward 运行哪些请求；每请求 Queue 只是输出邮箱，保存该请求尚未被调用方消费的 token 事件。一个 worker 可以在同一次 decode forward 中处理多个请求，再按 `request_id` 把结果分发到各自 Queue。对外主类已收敛为 `Scheduler` / `SchedulerConfig` 和 `GenerationService` / `GenerationSession`，没有可切换的旧调度后端。

## 代码入口

| 目录 | 职责 |
|---|---|
| `src/mysglang/core/` | Request、SamplingParams、状态转换和输出事件 |
| `src/mysglang/modeling/` | 可读的 tiny Qwen3 reference model |
| `src/mysglang/scheduler/` | continuous batching、paged/radix 调度 |
| `src/mysglang/cache/` | KV 物理池、页表、分配器和前缀缓存 |
| `src/mysglang/serving/` | session、输出路由与 HTTP/SSE 边界 |
| `src/mysglang/tokenizer/` | 当前 byte tokenizer 协议替身 |

当前设计、关键不变量和与 Mini-SGLang 的对比见 [docs/design.md](docs/design.md)，后续工程顺序见 [docs/roadmap.md](docs/roadmap.md)。

## 本地运行

当前工作区复用 `nano-vllm` 的 Python 3.12 环境：

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

项目的依赖边界以 `pyproject.toml` 为准：基础实现只要求 PyTorch；`serve`、`model`、`flash-attn` 和 `dev` extras 分别对应 HTTP、真实模型、GPU backend 和开发工具。具体 Python、PyTorch、CUDA、GPU 与 commit 信息应由未来评测脚本写入结果，而不在文档中维护容易过期的环境快照。

## 当前边界

- 只有随机 tiny dense 模型，尚未加载真实 Qwen3/Qwen3 MoE 权重；
- 只实现 greedy sampling，byte tokenizer 也只是协议测试替身；
- PyTorch attention 会 gather/pad 离散历史 K/V，没有直接消费物理 page pool；
- Scheduler 为单进程同步 step，Prefill 尚未组成高效的 ragged multi-request batch；
- 没有 CUDA Graph、Tensor Parallel、多进程容错或正式 benchmark client；
- 第一版只关注文本生成，不覆盖 VLM、量化、LoRA 和复杂 grammar。

这些限制属于明确的后续工作，不应被当前 reference 路径的正确性掩盖。
