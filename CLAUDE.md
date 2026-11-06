# CLAUDE.md
@HANDOFF.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MySGLang 是一个精简的 LLM 推理服务参考实现（paged KV + radix prefix cache + continuous batching，支持真实 Qwen3 / Qwen3MoE、Qwen3.5/3.6 混合架构（DeltaNet + MTP + 分块 FP8）与 TP）。文档、注释和 commit message 均使用中文。

## 常用命令

环境复用 `nano-vllm` 的 Python 3.12 venv，包未安装，需要 `PYTHONPATH=src`：

```bash
source ~/nano-vllm/.venv/bin/activate

# 全部测试（unittest 风格，pytest 也可运行）
PYTHONPATH=src python -m unittest discover -s tests -v
# 单个文件 / 单个用例
PYTHONPATH=src python -m unittest tests.test_cache -v
PYTHONPATH=src python -m unittest tests.test_distributed_scheduler.TensorParallelSchedulerTest.test_tp2_continuous_batching_matches_tp1
PYTHONPATH=src python -m pytest tests/test_engine.py -k prefix

# Lint（ruff: E/F/I/UP，line-length 100，target py310）
ruff check src tests benchmarks

# 交互 CLI（默认模型 ../KuiperLLama/artifacts/qwen3-0.6b/hf-source，/reset、/exit）
PYTHONPATH=src python -m mysglang.cli [--model PATH] [--backend flash] [--cuda-graph] \
  [--moe-dispatch naive|sorted|grouped|triton_grouped|all_to_all] \
  [--speculative-ngram-max-tokens 4]

# 多卡 TP
PYTHONPATH=src torchrun --standalone --nproc-per-node=4 -m mysglang.cli \
  --tensor-parallel-size 4 --model /path/to/ckpt --device cuda --dtype bfloat16 --backend flash

# Qwen3.5/3.6：本地验收模型 ~/models/Qwen3.5-0.8B（BF16）与 ~/models/Qwen3.5-0.8B-FP8（自转换）
PYTHONPATH=src python -m mysglang.cli --model ~/models/Qwen3.5-0.8B --backend flash \
  --cuda-graph --speculative-mtp-steps 2 [--thinking] [--dequantize-fp8]
# OpenAI 兼容服务（TP 时只有 rank 0 监听）
PYTHONPATH=src python -m mysglang.server --model ~/models/Qwen3.5-0.8B --port 8000
```

部分测试按依赖条件跳过：flash-attn/CUDA、`httpx`（serving extra）、`transformers`（model extra）、本地 Qwen3 tokenizer 目录。TP 测试通过 CPU/Gloo 多进程运行，不需要 GPU。venv 里装了 fla（flash-linear-attention）后，Transformers 的 Qwen3.5 在 import 时就会绑定 fla kernel，CPU oracle 测试必须像 `tests/test_hybrid.py` 的 `_TorchFallbackKernels` 那样换回 `__wrapped__` 的 PyTorch 实现。

Benchmark 与 Transformers 对齐脚本在 `benchmarks/`，用法见 [benchmarks/README.md](benchmarks/README.md)；结果按 `benchmarks/results/<日期>-<硬件>/` 归档（JSONL 原始数据 + `report.md`）。

## 架构要点

设计细节和不变量以 [docs/design.md](docs/design.md) 为准，后续工作见 [docs/roadmap.md](docs/roadmap.md)。修改调度/缓存前应先读对应章节。

**调用链**：`GenerationService.start()` 只做 tokenize + validate，返回一次性 `GenerationSession`；首次消费 session 时才 `Scheduler.add()`、注册 `request_id -> Queue`、启动唯一 scheduler worker。未消费的 session 不占调度槽、KV reservation 或物理页。Queue 只是输出邮箱，调度决策全部在 `Scheduler`。

**单一 worker 串行驱动 `Scheduler.step()`**：模型、Scheduler 和 cache metadata 从不被并发修改。每步所有 Decode 请求推进一个 token，并与多个 Prefill chunk 拼成一次 packed forward（phase = `decode` / `prefill` / `mixed`）；`prefill_token_budget` 只限制 Prefill token。纯 Decode 走 dense `[B,1]` 路径（可 CUDA Graph），其余走 packed eager。

**KV 所有权分层**：`PageAllocator`（物理页、页表、admission 按 `prompt_len + max_new_tokens` 最坏情况预留）→ `RadixPrefixCache`（token 前缀到 cached pages 的索引、引用计数/lock）→ `PagedKVCache`（逐层 K/V pool，`[layers, pages, page_size, kv_heads, head_dim]`）。`check_integrity()` 校验页面不变量，测试结束时普遍调用；改动 cache 逻辑后应保持其通过。关键约束：
- 非法输入（重复 ID、非法状态迁移、超容量）必须在修改任何资源前失败；
- 完成、异常、abort 都必须移除 scheduler entry、unlock prefix handle、释放私有页；
- radix 匹配后先 lock 再引用页面；只淘汰 `ref_count == 0` 的叶节点；
- radix 匹配故意保留 prompt 最后一个 token 不命中（需要它产生首个 logits，且避免部分页 copy-on-write）。

**Attention backend 窄接口**：模型只产生 RoPE 后的 Q/K/V；`PagedKVCache.prepare_batch()` 每次 forward 构造一次所有层共享的 metadata（block table、cu_seqlens、slot_mapping）。backend 在 attention 成功后才 `commit_append()`；CUDA Graph 路径则在输出同步后统一 `commit_batch()`，保证 kernel 失败时逻辑长度不前移。`TorchAttentionBackend` 是 CPU 测试与 correctness oracle；`FlashAttentionBackend` 要求 CUDA fp16/bf16 且 `page_size` 为 256 的倍数。

**模型**：`Qwen3ForCausalLM`（[modeling/qwen3.py](src/mysglang/modeling/qwen3.py)）同时承载 dense 与 MoE，使用 fused QKV 与 fused gate/up；`head_dim` 显式配置，不能假设等于 `hidden_size / num_heads`。TP 分片时 fused 参数的每个逻辑段需独立切分（[modeling/parallel.py](src/mysglang/modeling/parallel.py)、[modeling/loader.py](src/mysglang/modeling/loader.py)）。MoE dispatch 有五条可对照路径，只有 `triton_grouped` 可进入 CUDA Graph。

**TP 控制面**（[scheduler/distributed.py](src/mysglang/scheduler/distributed.py)）：rank 0 的 `TensorParallelScheduler` 通过 CPU/Gloo 广播 add/abort/step/reset/shutdown，每个 rank 镜像执行本地 Scheduler，比较权威 `SchedulerBatchPlan` 一致后才进入 forward；采样 token 以 rank 0 为准广播。模型 collective 走独立的 NCCL group。单进程 `Scheduler` 会拒绝 TP model。任何改变调度决策的代码都必须在所有 rank 上确定性一致，否则 plan 校验失败或 collective 挂死。

**投机解码**：n-gram（[scheduler/ngram.py](src/mysglang/scheduler/ngram.py)）与 MTP（`speculative_mtp_steps`，互斥）共用一套 verify/回滚：K/V 用逻辑长度截断；DeltaNet 状态不能截断，verify 以 log 模式只读 slot 并记录输入，接受后 replay（等宽时用 g=β=0 的掩码 replay，可进 Graph）。MTP 不变量：两步之间草稿 K/V（`PagedKVCompanion`）长度必须等于目标长度、每个运行请求都有 `mtp_draft`；每次目标前向后都要对同一批位置做 MTP 追赶。greedy（含 penalty）按逐位置的 penalized argmax 验证，随机采样用投机采样（`sampler.speculative_accept`：以 min(1,p/q) 接受，拒绝时从 max(0,p−q) 重采样；n-gram/greedy 草稿 q 为 one-hot），两者都保持非投机时的输出分布。bf16 下 MTP 开/关可能因 kernel 形状不同而在几十个 token 后分叉，判断回滚正确性要用 fp32 + torch 后端。

**Qwen3.5 混合架构**（[modeling/deltanet.py](src/mysglang/modeling/deltanet.py)、[cache/hybrid.py](src/mysglang/cache/hybrid.py)）：只有 full-attention 层占 paged K/V（cache 层号 ≠ 模型层号，见 `kv_layer_idx`）；DeltaNet 状态在 `RecurrentStatePool` 每请求一个 slot，由 `HybridPagedKVCache` 与页一起预留/释放；混合模型关闭 radix 前缀复用。RMSNorm 为零中心 `1+w`，q_proj 每头交错 `[query, gate]`。CUDA Graph 捕获前会推进 DeltaNet 状态，runner 必须快照/恢复 slot。

**图片输入**（[multimodal.py](src/mysglang/multimodal.py)、[modeling/vision.py](src/mysglang/modeling/vision.py)）：`<|image_pad|>` 展开成 t·h·w/4 个 token；M-RoPE 位置与 K/V 下标分离——prompt 用三轴位置（`PagedKVBatch.mrope_positions`），之后用 `K/V 下标 + rope_delta`（decode buffer 的 `rope_positions`，Graph 友好）。视觉特征在首个 prefill chunk 编码一次，按行 `embedding_overrides` 替换，MTP 追赶用 shift=1 同样替换。图片请求要求前缀缓存关闭。验证图片正确性要用 `multimodal_reference`（tests/test_multimodal.py）或 CPU fp32 对照 HF。

**FP8**（[modeling/fp8.py](src/mysglang/modeling/fp8.py)）：权重 e4m3 + 每 128×128 tile 一个 `weight_scale_inv`；哪些模块量化由 `quantization_config.modules_to_not_convert` 决定；FP8 参数通过 `keep_dtype_parameters` 免于 dtype cast。TP 分片与融合段边界必须是 block 的倍数（构造时校验）。改 loader 映射后用 `benchmarks/validate_checkpoint_layout.py` 对 35B 做离线全量校验。

**测试 oracle**：[tests/helpers.py](tests/helpers.py) 提供小型随机 `make_model()`、完整前缀重算 `reference_generate()` 和把模型导出成 HF 权重名（含 FP8 scale）的 `export_hf_state_dict()`；调度/缓存/投机测试以 token 与 oracle 完全一致为验收标准。旧的 contiguous/slot cache 已删除，不要重新引入平行后端。
