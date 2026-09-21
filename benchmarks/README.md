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

`validate_transformers.py` 使用相同输入让 Transformers BF16 逐 token greedy 解码，并把
token ID 与指定的 mysglang 结果逐项比较：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/validate_transformers.py \
  --model /root/models/Qwen3-0.6B --results benchmarks/results/raw.jsonl \
  --names tp1-c1-p128-o64 tp2-c1-p128-o64 tp4-c1-p128-o64 \
  --output benchmarks/results/transformers-alignment.json
```

已有实测报告见 [2026-09-21-4090x4/report.md](results/2026-09-21-4090x4/report.md)。

