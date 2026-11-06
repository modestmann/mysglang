# MySGLang

MySGLang 是一个用于理解和验证 LLM 推理系统的精简实现。当前主路径已经从单请求生成演进为：

- 显式请求状态与增量输出事件；
- continuous batching、ragged/chunked prefill、Prefill/Decode 混合 batch；
- 共享 Paged KV Cache、容量预留和 OOM-safe admission；
- page-aligned Radix prefix cache、引用保护与按需淘汰；
- 可切换的 PyTorch reference / FlashInfer attention backend；
- 一次 forward 构造、所有 Transformer 层复用的 Paged KV batch metadata；
- 固定地址 Decode metadata buffer 与可选的精确 batch-size CUDA Graph；
- 无草稿模型的 n-gram 投机解码、packed 主模型验证与 KV 逻辑回滚；
- 真实 dense Qwen3 / Qwen3MoE 统一架构、fused QKV 与 fused gate/up；
- Qwen3.5/3.6 混合架构（Gated DeltaNet + Gated Attention + shared expert），DeltaNet 递归状态 slot 与 paged KV 共存；
- 模型自带 MTP 层的投机解码：递归草稿、log 模式 verify、DeltaNet 掩码 replay 回滚，覆盖 CUDA Graph；
- 官方 128×128 分块 FP8 checkpoint：FP8 权重 + scale 的 TP 切分、Triton W8A16 linear 与 grouped MoE kernel；
- OpenAI 兼容 `/v1/chat/completions`：`reasoning_content`、`qwen3_coder` 工具调用、presence penalty、图片输入；
- Qwen3.5 视觉塔（ViT + 2×2 合并器）与交错 M-RoPE：图片预处理与 HF 逐位一致，图片请求可 chunked prefill、MTP 与 CUDA Graph；
- SafeTensors 加载、Hugging Face chat template、增量 detokenization；
- greedy、temperature、top-k、top-p 与 per-request seed 采样；
- HTTP/JSON、SSE 流式输出和 abort 清理。

历史版本中，本地 `../KuiperQwen3/artifacts/qwen3-0.6b/hf-source` 的真实 Qwen3-0.6B 已完成 FP32 logits 对齐，以及 BF16 + FlashAttention paged-KV 的单请求/并发 smoke test。Qwen3MoE 已完成 Transformers 小模型 oracle 对齐，并已加入 Attention TP + expert 分片的多进程路径；真实 Qwen3-30B-A3B 也已在四张 RTX 4090 上完成 token oracle、显存和三轮性能验收。

Qwen3.5/3.6 混合架构已与 Transformers 小模型（dense 与 MoE）logits 对齐；本地真实 `Qwen3.5-0.8B`（与 Qwen3.6 同一 DeltaNet/MTP 结构）的 BF16 greedy 输出与 Transformers 逐 token 一致，MTP k=1 接受率约 70%；fp32 reference 下 MTP 开/关输出逐位一致。`Qwen3.6-35B-A3B-FP8` 已通过只读 SafeTensors 头部的离线布局校验（TP=4 每 rank 全覆盖，约 10 GiB 权重），真实四卡运行尚待验收。设计细节见 [docs/design.md 第 8 节](docs/design.md)。

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
| `src/mysglang/modeling/` | Qwen3/Qwen3MoE/Qwen3.5、DeltaNet、FP8、checkpoint loader、attention、CUDA Graph |
| `src/mysglang/scheduler/` | continuous batching、paged/radix 调度 |
| `src/mysglang/cache/` | KV 物理池、页表、分配器、前缀缓存和 DeltaNet 状态 slot |
| `src/mysglang/serving/` | session、输出路由、HTTP/SSE 与 reasoning/工具调用解析 |
| `src/mysglang/tokenizer/` | Hugging Face chat/incremental tokenizer 与测试用 byte tokenizer |

当前设计、关键不变量和与 Mini-SGLang 的对比见 [docs/design.md](docs/design.md)，后续工程顺序见 [docs/roadmap.md](docs/roadmap.md)。

## 本地运行

当前工作区复用 `nano-vllm` 的 Python 3.12 环境：

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

项目的依赖边界以 `pyproject.toml` 为准：基础实现只要求 PyTorch；`serve`、`model`、`vision`、`flashinfer` 和 `dev` extras 分别对应 HTTP、真实模型、图片预处理、GPU backend 和开发工具。具体 Python、PyTorch、CUDA、GPU 与 commit 信息应由未来评测脚本写入结果，而不在文档中维护容易过期的环境快照。

GPU attention 安装 `flashinfer` extra（含 FlashInfer 与 Triton）：

```bash
uv pip install -e '.[model,serve,flashinfer]'
```

`--backend auto` 在 CUDA 且已安装 FlashInfer 时选用它，否则使用 Torch；显式指定
`--backend flashinfer` 时缺包会报错。CLI、HTTP server 和 benchmark 都支持
`--page-size 1` 或 `16`，默认值为 `16`。完整前缀重算 oracle 使用 `TorchAttentionBackend`，
FlashInfer 推理通过 Scheduler 或带 KV cache 的模型接口运行。

加载本地真实模型：

```python
import torch
from mysglang import FlashInferBackend, HuggingFaceTokenizer, Qwen3ForCausalLM

model_path = "../KuiperQwen3/artifacts/qwen3-0.6b/hf-source"
tokenizer = HuggingFaceTokenizer.from_pretrained(model_path)
model = Qwen3ForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device="cuda",
    attention_backend=FlashInferBackend(),
)
```

直接打开本地交互对话：

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m mysglang.cli
```

默认使用同级 `KuiperQwen3/artifacts/qwen3-0.6b/hf-source`，输入 `/reset` 清空上下文，
输入 `/exit` 退出；其他 checkpoint 可用 `--model PATH` 指定。

在一台四卡机器上启动 dense Qwen3 或 Qwen3MoE 的 TP=4 交互推理：

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=4 -m mysglang.cli \
  --tensor-parallel-size 4 \
  --model /path/to/qwen3-checkpoint \
  --device cuda --dtype bfloat16 --backend flashinfer
```

`torchrun` 创建四个进程并设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`；每个进程绑定
一张 GPU。只有 rank 0 加载 tokenizer、读取终端输入并打印结果，其他 rank 进入 TP
worker loop。`--tensor-parallel-size` 是防止启动参数写错的校验项，不能替代
`--nproc-per-node`。`--cuda-graph` 可以捕获固定 B=1 Decode；TP 使用 NCCL Graph，MoE
还必须选择 `--moe-dispatch triton_grouped`。这两条新路径仍待目标服务器实测。

### Qwen3.5 / Qwen3.6（混合架构、MTP、FP8）

本地可用同架构的 `Qwen3.5-0.8B` 验证（约 1.5 GB 显存）。DeltaNet 默认在 CUDA 上使用
fla kernel（`pip install flash-linear-attention`，未安装时回退到 PyTorch reference）：

```bash
PYTHONPATH=src python -m mysglang.cli --model ~/models/Qwen3.5-0.8B \
  --backend flashinfer --cuda-graph --speculative-mtp-steps 2 --thinking
```

`--speculative-mtp-steps k` 会同时加载 checkpoint 的 MTP 层，每轮递归起草 k 个 token；
`--cuda-graph` 对混合模型的 Decode、verify、MTP 草稿/追赶都按精确 batch size 捕获
（`--cuda-graph-batch-sizes` 指定 bucket）。每个运行中的请求占一个 DeltaNet 状态 slot，
所以 `--max-running-requests` 同时决定状态池大小。
小模型没有官方 FP8 版本，可以按官方格式自行转换，用来验证 FP8 路径：

```bash
PYTHONPATH=src python benchmarks/make_fp8_checkpoint.py \
  --model ~/models/Qwen3.5-0.8B --output ~/models/Qwen3.5-0.8B-FP8
```

OpenAI 兼容服务（thinking 输出拆到 `reasoning_content`，`tools` 解析为 `tool_calls`）：

```bash
PYTHONPATH=src python -m mysglang.server --model ~/models/Qwen3.5-0.8B \
  --backend flashinfer --cuda-graph --cuda-graph-batch-sizes 1 2 3 4 \
  --speculative-mtp-steps 2 --max-running-requests 4 --port 8000

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "1+1=?"}], "max_tokens": 512,
  "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": true}}'
```

图片输入（checkpoint 带视觉塔时默认加载，`--no-vision` 关闭）。CLI 用 `--image` 附图，
交互模式里 `/image PATH` 给下一句附图；HTTP 用 OpenAI 的 `image_url` content part：

```bash
PYTHONPATH=src python -m mysglang.cli --model ~/models/Qwen3.5-0.8B --backend flashinfer \
  --image ./cat.png --prompt "这张图里有什么？"

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "描述这张图"}]}],
  "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": false}}'
```

图片预处理通过 `vision` extra 安装 NumPy 和 Pillow：`uv pip install -e '.[vision]'`（不需要 torchvision）。这些依赖只在处理图片时导入，基础文本推理无需安装；图片请求要求关闭前缀缓存（Qwen3.5 混合模型本来就关闭）。

在四张 RTX 4090 上运行 `Qwen3.6-35B-A3B-FP8`（每 rank 约 10 GiB 权重；2 个 KV 头按
GQA 分组复制到 4 个 rank；MoE 的 CUDA Graph 需要 `triton_grouped`）。上机前可以先离线
确认 checkpoint 布局，这一步只读取 SafeTensors 头部，不下载权重：

```bash
PYTHONPATH=src python benchmarks/validate_checkpoint_layout.py \
  --repo Qwen/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size 4 --enable-mtp

PYTHONPATH=src torchrun --standalone --nproc-per-node=4 -m mysglang.server \
  --tensor-parallel-size 4 --model /path/to/Qwen3.6-35B-A3B-FP8 \
  --device cuda --dtype bfloat16 --backend flashinfer --moe-dispatch triton_grouped \
  --cuda-graph --cuda-graph-batch-sizes 1 2 4 8 --speculative-mtp-steps 3 \
  --max-running-requests 8 --num-pages 4096 --port 8000
```

`--dequantize-fp8` 会把 FP8 checkpoint 反量化成 `--dtype` 权重加载，作为 FP8 kernel 的数值对照。

高重复度的 greedy 输出可以显式打开 n-gram 投机解码：

```bash
PYTHONPATH=src python -m mysglang.cli \
  --speculative-ngram-max-tokens 4 \
  --speculative-ngram-min-match 2 \
  --speculative-ngram-max-match 8
```

它从当前请求的历史 token 复制候选续写，再由主模型一次验证。默认值 `0`
表示关闭。temperature 采样仍走普通单 token 路径，因为历史 n-gram 查找不提供
草稿概率分布。

MoE 提供五条可对照路径：`naive` 为逐 expert 扫描基线；默认 `sorted` 一次筛选并按
expert 分组，但仍逐 expert GEMM；`grouped` 将活跃 experts 填充成 batched matrices，
用两次 `bmm` 替代 Python GEMM loop；`triton_grouped` 用 expert offsets 直接计算真实
assignments，并将 gate/up GEMM 与 SwiGLU 融合，不再填充到最忙 expert 的长度；
`all_to_all` 在 grouped 计算外增加 token ownership、dispatch/combine all-to-all，并在
末尾 all-gather 恢复 Attention TP 所需的 replicated hidden。可用 `--moe-dispatch`
显式选择，云端实测前不提前指定新的默认路径。

## 当前边界

- 真实 Qwen3-30B-A3B 已完成四卡加载、token oracle、显存和 naive/sorted 性能验证；
- MoE 已有 portable padded-batched grouped GEMM、no-padding Triton grouped GEMM 和
  token all-to-all reference 路径；旧两条路径的四卡实测均慢于 sorted，新的 Triton
  路径仍需在目标 CUDA 环境做 token oracle 和 A/B，当前不作为默认值；
- PyTorch reference backend 会 gather/pad；FlashInfer 使用 CUDA 半精度和 CSR 页表直接读取 paged KV，默认每页 16 个 token，也支持 `--page-size 1`；
- 纯 Decode 可复用 metadata buffer 并按精确 batch size 使用 dense/MoE、单卡/TP CUDA
  Graph；TP/NCCL capture 尚待云端验证，ragged/mixed metadata 仍由 Python 构造；
- 已完成 dense 模型级 TP、MoE expert 分片、checkpoint 分片加载、rank worker/batch-plan
  广播和 `torchrun` CLI 启动链；尚待多进程容错或正式 benchmark client；
- n-gram 投机当前以有界历史扫描产生一条候选链，尚未实现 SGLang 更完整的
  per-request trie/token tree，也不对随机采样做概率上的精确验证；
- Qwen3.5/3.6 混合模型暂时关闭 radix 前缀复用（DeltaNet 状态无法从 K/V 页恢复）；
  MTP 每轮只起草一条链（无草稿树）；FP8 为 W8A16，未使用 FP8 tensor core；
- `Qwen3.6-35B-A3B-FP8` 只完成离线布局校验，四卡 token oracle 与性能尚未实测；
- 图片输入只在 Qwen3.5-0.8B 上验收（35B 的视觉塔未实测），不支持视频；不覆盖 LoRA 和复杂 grammar。

这些限制属于明确的后续工作，不应被当前 reference 路径的正确性掩盖。
