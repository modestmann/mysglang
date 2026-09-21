# MySGLang

MySGLang 是一个用于理解和验证 LLM 推理系统的精简实现。当前主路径已经从单请求生成演进为：

- 显式请求状态与增量输出事件；
- continuous batching、ragged/chunked prefill、Prefill/Decode 混合 batch；
- 共享 Paged KV Cache、容量预留和 OOM-safe admission；
- page-aligned Radix prefix cache、引用保护与按需淘汰；
- 可切换的 PyTorch reference / FlashAttention 2 attention backend；
- 一次 forward 构造、所有 Transformer 层复用的 Paged KV batch metadata；
- 固定地址 Decode metadata buffer 与可选的精确 batch-size CUDA Graph；
- 真实 dense Qwen3 / Qwen3MoE 统一架构、fused QKV 与 fused gate/up；
- SafeTensors 加载、Hugging Face chat template、增量 detokenization；
- greedy、temperature、top-k、top-p 与 per-request seed 采样；
- HTTP/JSON、SSE 流式输出和 abort 清理。

本地 `../KuiperLLama/artifacts/qwen3-0.6b/hf-source` 的真实 Qwen3-0.6B 已完成 FP32 logits 对齐，以及 BF16 + FlashAttention paged-KV 的单请求/并发 smoke test。Qwen3MoE 已完成 Transformers 小模型 oracle 对齐；真实 MoE checkpoint、Tensor Parallel 和多卡执行留到有足够显存的环境验收。

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
                 -> Qwen3ForCausalLM（dense / MoE）
                      -> AttentionBackend
  <- IncrementalOutput / SSE
```

`GenerationService.start()` 只校验请求并返回一次性消费的 `GenerationSession`。直到调用方首次迭代 session，请求才进入 Scheduler，并创建 Queue、启动 worker；从未消费的 session 不占调度槽、KV reservation 或物理页。

全局 `Scheduler` 决定下一次 forward 运行哪些请求；每请求 Queue 只是输出邮箱，保存该请求尚未被调用方消费的 token 事件。一个 worker 可以在同一次 decode forward 中处理多个请求，再按 `request_id` 把结果分发到各自 Queue。对外主类已收敛为 `Scheduler` / `SchedulerConfig` 和 `GenerationService` / `GenerationSession`，没有可切换的旧调度后端。

## 代码入口

| 目录 | 职责 |
|---|---|
| `src/mysglang/core/` | Request、SamplingParams、状态转换和输出事件 |
| `src/mysglang/modeling/` | Qwen3/Qwen3MoE、checkpoint loader、attention、CUDA Graph |
| `src/mysglang/scheduler/` | continuous batching、paged/radix 调度 |
| `src/mysglang/cache/` | KV 物理池、页表、分配器和前缀缓存 |
| `src/mysglang/serving/` | session、输出路由与 HTTP/SSE 边界 |
| `src/mysglang/tokenizer/` | Hugging Face chat/incremental tokenizer 与测试用 byte tokenizer |

当前设计、关键不变量和与 Mini-SGLang 的对比见 [docs/design.md](docs/design.md)，后续工程顺序见 [docs/roadmap.md](docs/roadmap.md)。

## 本地运行

当前工作区复用 `nano-vllm` 的 Python 3.12 环境：

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

项目的依赖边界以 `pyproject.toml` 为准：基础实现只要求 PyTorch；`serve`、`model`、`flash-attn` 和 `dev` extras 分别对应 HTTP、真实模型、GPU backend 和开发工具。具体 Python、PyTorch、CUDA、GPU 与 commit 信息应由未来评测脚本写入结果，而不在文档中维护容易过期的环境快照。

加载本地真实模型：

```python
import torch
from mysglang import FlashAttentionBackend, HuggingFaceTokenizer, Qwen3ForCausalLM

model_path = "../KuiperLLama/artifacts/qwen3-0.6b/hf-source"
tokenizer = HuggingFaceTokenizer.from_pretrained(model_path)
model = Qwen3ForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device="cuda",
    attention_backend=FlashAttentionBackend(),
)
```

直接打开本地交互对话：

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m mysglang.cli
```

默认使用同级 `KuiperLLama/artifacts/qwen3-0.6b/hf-source`，输入 `/reset` 清空上下文，
输入 `/exit` 退出；其他 checkpoint 可用 `--model PATH` 指定。

在一台四卡机器上启动 dense Qwen3 的 TP=4 交互推理：

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=4 -m mysglang.cli \
  --tensor-parallel-size 4 \
  --model /path/to/qwen3-checkpoint \
  --device cuda --dtype bfloat16 --backend flash
```

`torchrun` 创建四个进程并设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`；每个进程绑定
一张 GPU。只有 rank 0 加载 tokenizer、读取终端输入并打印结果，其他 rank 进入 TP
worker loop。`--tensor-parallel-size` 是防止启动参数写错的校验项，不能替代
`--nproc-per-node`。当前 TP 路径暂不支持 `--cuda-graph`。

## 当前边界

- 本机只有 dense Qwen3-0.6B checkpoint；真实 Qwen3MoE 权重尚待云端加载验证；
- MoE dispatch 当前是正确性优先的逐 expert 实现，尚未替换为 grouped GEMM；
- 默认 PyTorch backend 会 gather/pad；可选 FA2 backend 已能直接消费 page pool/block table，但要求 CUDA 半精度且 `page_size` 为 256 的倍数；
- Scheduler 为单进程同步 step；纯 Decode 可复用 metadata buffer 并按精确 batch size 使用 CUDA Graph，但 ragged/mixed metadata 仍由 Python 构造，尚未做调度/执行重叠；
- 已完成 dense 模型级 TP 切分、checkpoint 分片加载、rank worker/batch-plan 广播和
  `torchrun` CLI 启动链；尚待云端 NCCL TP=2/4 验收、MoE TP/EP、多进程容错或正式
  benchmark client；
- 第一版只关注文本生成，不覆盖 VLM、量化、LoRA 和复杂 grammar。

这些限制属于明确的后续工作，不应被当前 reference 路径的正确性掩盖。
