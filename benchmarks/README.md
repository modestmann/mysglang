# 推理评测

`benchmark_generation.py` 测量真实 checkpoint 的端到端生成，支持单卡和
`torchrun` TP。它将配置、逐请求 token/timestamp、Scheduler 计数、Torch 峰值显存、
`nvidia-smi` 采样、软件版本、GPU 拓扑和完整命令写入 JSONL。

单卡示例：

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_generation.py \
  --model /root/models/Qwen3-0.6B --name tp1-c8-p128-o64 \
  --concurrency 8 --prompt-tokens 128 --max-new-tokens 64
```

四卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/benchmark_generation.py --model /root/models/Qwen3-0.6B \
  --tensor-parallel-size 4 --name tp4-c8-p128-o64 \
  --concurrency 8 --prompt-tokens 128 --max-new-tokens 64
```

参数缩写：`c` 是并发请求数，`p` 是每请求 prompt token 数，`o` 是每请求固定生成
token 数。默认使用 greedy 且忽略 EOS，以确保不同实现完成相同工作量。脚本先 warmup，
清空 prefix cache，再开始计时；CUDA Graph 的捕获也因此不计入正式结果。

MoE 优化前后使用相同命令，只切换 `--moe-dispatch naive` / `sorted` / `grouped` /
`triton_grouped` / `all_to_all`；JSONL 会记录所选 backend。真实 MoE 还应分别测短 Decode
与长 Prefill，因为两者的 expert 分组规模、padding 比例和通信/计算比不同。`grouped` 是
portable padded-batched `bmm` 路径；`triton_grouped` 是 offsets 驱动的 no-padding CUDA
路径；`all_to_all` 仍复用 portable grouped 完成本地 expert 计算。比较时至少保留
`sorted`、`grouped` 和 `triton_grouped`，用相同 all-reduce 通信分离计算 kernel 的收益。

`validate_transformers.py` 使用相同输入让 Transformers BF16 逐 token greedy 解码，并把
token ID 与指定的 mysglang 结果逐项比较：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/validate_transformers.py \
  --model /root/models/Qwen3-0.6B --results benchmarks/results/raw.jsonl \
  --names tp1-c1-p128-o64 tp2-c1-p128-o64 tp4-c1-p128-o64 \
  --output benchmarks/results/transformers-alignment.json
```

checkpoint 单卡放不下时可安装 Accelerate，并让 Transformers 按完整 layer 分布到多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python benchmarks/validate_transformers.py \
  --model /root/models/Qwen3-30B-A3B \
  --results benchmarks/results/qwen3-30b-a3b-4090x4.jsonl \
  --names moe-tp4-sorted-smoke moe-tp4-naive-smoke \
  --prompt-tokens 32 --max-new-tokens 4 --device-map balanced \
  --output benchmarks/results/qwen3-30b-a3b-transformers-alignment.json
```

这里的 Transformers `device_map` 只是低吞吐正确性 oracle，不作为 TP/EP 性能对照。

N-gram 投机要在同一 greedy workload 上做开/关 A/B，并先确认输出 token ID 完全一致：

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_generation.py \
  --model /root/models/Qwen3-0.6B --name ngram-off \
  --concurrency 8 --prompt-tokens 128 --max-new-tokens 128

PYTHONPATH=src .venv/bin/python benchmarks/benchmark_generation.py \
  --model /root/models/Qwen3-0.6B --name ngram-4 \
  --concurrency 8 --prompt-tokens 128 --max-new-tokens 128 \
  --speculative-ngram-max-tokens 4 \
  --speculative-ngram-min-match 2 --speculative-ngram-max-match 8
```

JSONL 的 scheduler 部分会记录 `speculative_draft_tokens`、
`speculative_accepted_tokens`、`speculative_acceptance_rate` 和
`speculative_verify_forwards`。确认 token 对齐后，再比较
output tokens/s、ITL、`model_forwards` 及接受率。高重复 synthetic prompt 只表示机制上界，
正式结论还应包含真实对话和代码 workload。

已有实测报告：

- [Qwen3-0.6B：RTX 4090 ×4](results/2026-09-21-4090x4/report.md)；
- [Qwen3-30B-A3B：RTX 4090 ×4 MoE](results/2026-09-21-qwen3-30b-a3b-4090x4/report.md)；
- [Qwen3-30B-A3B：grouped 与 all-to-all A/B](results/2026-09-22-qwen3-30b-a3b-grouped-a2a-4090x4/report.md)。
