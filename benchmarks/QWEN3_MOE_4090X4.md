# Qwen3-30B-A3B 四卡验收清单

本清单已于 2026-09-21 在 4×RTX 4090 实例完成。完整原始数据、拓扑/NCCL 证据、
Transformers token oracle 和三轮 A/B 汇总见
[实测报告](results/2026-09-21-qwen3-30b-a3b-4090x4/report.md)。

目标模型：`/root/models/Qwen3-30B-A3B`；硬件：4×RTX 4090 24 GiB。不要只看进程能启动，
必须同时保存 token 对齐、每 rank 显存、NCCL 活跃情况和性能原始数据。

## 1. 启动前

```bash
cd /root/mysglang
git rev-parse HEAD
nvidia-smi topo -m
```

确认四张卡空闲。第一轮使用较小 KV pool：FA2 的 page size 为 256，`--num-pages 64`
已经能容纳本清单的 workload，并避免先为 KV 预留过多显存。

## 2. 正确性 smoke

先跑默认 sorted dispatch：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/benchmark_generation.py \
  --model /root/models/Qwen3-30B-A3B --tensor-parallel-size 4 \
  --name moe-tp4-sorted-smoke --moe-dispatch sorted \
  --concurrency 1 --prompt-tokens 32 --max-new-tokens 4 --num-pages 64 \
  --output benchmarks/results/qwen3-30b-a3b-4090x4.jsonl
```

再将唯一变量改为 `--moe-dispatch naive`。两条记录的 token IDs 必须完全一致。随后补
Transformers 四卡分配的 BF16 greedy oracle；在 oracle 完成前只能称为 naive/sorted
内部对齐，不能宣称与官方实现对齐。

## 3. 优化前后矩阵

每一行分别跑 naive 和 sorted，至少重复三次：

| 场景 | 并发 | Prompt | Output | 主要观察点 |
|---|---:|---:|---:|---|
| Decode latency | 1 | 128 | 64 | TTFT、p50/p95 ITL |
| Decode throughput | 8 | 128 | 64 | output tok/s、expert 小 batch 开销 |
| Prefill | 4 | 1024 | 32 | 分组后较大 expert GEMM 的收益 |

每次记录四卡峰值显存/利用率/功耗、Scheduler batch 统计、环境和完整命令。若 sorted 没有
收益也必须保留结果；它仍可能说明 Python expert loop 或 collective 才是主要瓶颈。

## 4. 后续优化顺序

1. grouped GEMM/Triton，替换逐 expert `F.linear`；
2. 引入 token ownership/sequence parallel 后再比较 all-reduce 与 all-to-all EP；
3. router/expert 负载统计与不均衡分析；
4. 真实请求集、长上下文和稳定多轮重复。

当前 replicated-token EP 已避免重复 expert 计算：每个 expert 只存在于一个 rank。它复制
的是 hidden/router 工作，并以完整 hidden all-reduce 合并结果；在当前 Attention TP 的
replicated hidden 布局下，直接换 all-to-all 未必更省通信，不能脱离实测强行替换。

## 5. 本次验收摘要

- Transformers、naive 和 sorted smoke 的 4 个 greedy token 完全一致；正式矩阵的
  9 组对应输出也全部一致；
- sorted 相对 naive 的 output throughput 中位数：单请求 Decode `+45.7%`，并发
  Decode `+48.6%`，长 Prefill `+22.2%`；
- 四卡峰值显存 16,718～16,722 MiB；
- 所有 GPU pair 的 CUDA P2P 均不可用，NCCL ring 实际走 SHM；因此下一步 all-to-all
  EP 必须与当前 all-reduce 基线做实机 A/B，不能由拓扑标签推断收益。
