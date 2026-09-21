# Qwen3-30B-A3B：RTX 4090 ×4 MoE 实测

## 结论

真实 `Qwen3-30B-A3B` 已在 4×RTX 4090 上完成 BF16、FlashAttention 2、TP=4 与
expert ownership 分片验证。每张卡实际占用约 16.3 GiB，模型可以稳定加载和生成。

- Transformers 5.17 BF16 greedy oracle 与 `naive`、`sorted` smoke 的 4 个 token ID
  完全一致；
- 三种 workload、三轮重复中，`naive` 与 `sorted` 的所有请求 token ID 均一致；
- `sorted` 的 output throughput 中位数相对 `naive` 提升 22.2%～48.6%；
- 当前云主机所有 GPU pair 的 CUDA P2P 均不可用，NCCL 实际通过 SHM 中转。因此本结果
  证明现有实现能在这种拓扑上运行，不代表它已经具备高性能 all-to-all EP。

## 环境与拓扑

| 项目 | 实测值 |
|---|---|
| commit | `c578fb60ceca603d7fb0061433e21c8036821549`，外加本次 benchmark/oracle 改动 |
| GPU | 4×NVIDIA GeForce RTX 4090 24 GiB |
| Python / PyTorch | 3.12.14 / 2.5.1+cu121 |
| CUDA runtime / driver | 12.1 / 565.77 |
| FlashAttention | 2.8.3.post1，真实 BF16 CUDA kernel 已执行 |
| Transformers / Accelerate | 5.17.0 / 1.15.0 |
| 模型 | `/root/models/Qwen3-30B-A3B` |

`nvidia-smi topo -m` 显示 GPU 0/1 位于 NUMA 0，二者为 `NODE`；GPU 2/3 位于
NUMA 1，二者为 `PHB`；两组之间为 `SYS`，且没有 NVLink。CUDA
`can_device_access_peer()` 和 `nvidia-smi topo -p2p` 均确认所有 pair 的 P2P 不可用。

实际链路测试不能只看拓扑标签：

| 测试 | 结果 |
|---|---:|
| pinned-memory H2D，每张卡 | 13.01～13.04 GB/s |
| 64 MiB AllReduce，GPU 0/1 | 8.85 ms，algbw 7.58 GB/s |
| 64 MiB AllReduce，GPU 2/3 | 8.89 ms，algbw 7.55 GB/s |
| 64 MiB AllReduce，GPU 0/2 | 8.76 ms，algbw 7.66 GB/s |
| 64 MiB AllReduce，四卡 | 11.40 ms，algbw 5.89 GB/s，busbw 8.83 GB/s |

三个双卡组合几乎没有差异。`NCCL_DEBUG=INFO` 进一步确认环
`0 -> 1 -> 2 -> 3` 的每条边均为 `via SHM/direct/direct`，且 NVLS 不可用。也就是说，
在该租赁实例上，P2P 禁用后的 host shared-memory staging 掩盖了 `NODE/PHB/SYS` 的理论
差异。原始证据见 [nccl-smoke.log](nccl-smoke.log)。

## 正确性

固定 32-token prompt，生成 4 个 greedy token：

```text
Transformers: [1773, 220, 105444, 107420]
sorted:       [1773, 220, 105444, 107420]
naive:        [1773, 220, 105444, 107420]
```

Transformers 使用 `device_map=balanced` 按完整 layer 分布到四张卡，只作为低吞吐正确性
oracle。完整结果见 [transformers-alignment.json](transformers-alignment.json)。正式矩阵
的三种 workload × 三轮重复也逐请求核对过 token，9/9 组 `naive == sorted`。

## 优化前后性能

每格为三轮的中位数；括号内为 output tok/s 的最小值～最大值。所有测试使用相同输入、
greedy sampling、TP=4、FA2 和 64 个 KV pages，只切换 MoE dispatch。

| workload | dispatch | output tok/s | p50 TTFT | p50 ITL | p50 E2E |
|---|---|---:|---:|---:|---:|
| C1 / P128 / O64 | naive | 6.97 (6.82～7.03) | 304.04 ms | 127.15 ms | 9185.27 ms |
| C1 / P128 / O64 | sorted | 10.15 (10.11～10.54) | 228.21 ms | 86.45 ms | 6302.90 ms |
| C8 / P128 / O64 | naive | 50.03 (41.02～54.36) | 364.76 ms | 173.58 ms | 10233.91 ms |
| C8 / P128 / O64 | sorted | 74.35 (71.41～80.02) | 272.41 ms | 113.71 ms | 6885.97 ms |
| C4 / P1024 / O32 | naive | 26.29 (21.05～26.37) | 594.40 ms | 127.90 ms | 4868.07 ms |
| C4 / P1024 / O32 | sorted | 32.13 (28.63～33.17) | 506.07 ms | 116.98 ms | 3983.14 ms |

| workload | sorted throughput 提升 |
|---|---:|
| 单请求 Decode latency | +45.7%（1.46×） |
| 并发 Decode throughput | +48.6%（1.49×） |
| 长 Prefill | +22.2%（1.22×） |

三种 workload 的 rank 0 Torch 峰值 allocated 分别约 15.49、15.52、15.61 GiB；
`nvidia-smi` 观测到的四卡峰值为 16,718～16,722 MiB。sorted 没有增加模型/KV 的常驻
显存。采样到的平均 GPU utilization 仍只有约 40%～54%，表明逐 expert 小 GEMM、Python
dispatch 和跨卡 collective 仍有明显优化空间；短测试中的利用率和功耗只用于诊断，不能
当作稳定的整机能耗结论。

## 如何解释这组结果

当前不是经典的 token all-to-all EP。Attention 使用 TP=4；Attention all-reduce 后，
每个 rank 都拥有全部 token hidden 和相同 router 结果，但只保存并计算自己负责的
experts，最后对 MoE 输出做 all-reduce。`sorted` 只优化本地 dispatch：一次筛选本 rank
assignment，再排序成 expert 分组，避免 `naive` 为每个本地 expert 重复扫描全部 routing
结果。因此两条路径的权重分片与通信量相同，A/B 能隔离 dispatch 优化本身。

下一步优先将同一 rank 上多个 expert 的逐 expert `F.linear` 换为 grouped GEMM/Triton，
并记录 router 负载分布。真正的 token ownership + all-to-all 应保留为独立基线：在当前
无 P2P、NCCL 走 SHM 的机器上，小 Decode batch 的两次 all-to-all 很可能被 host staging
延迟抵消；必须以相同 token、相同 workload 实测后再决定默认路径。

## 文件与复现

- [raw.jsonl](raw.jsonl)：2 条 smoke、18 条正式 A/B 和 1 条 NCCL 证据记录；
- [transformers-alignment.json](transformers-alignment.json)：官方实现 token oracle；
- [nccl-smoke.log](nccl-smoke.log)：NCCL transport、ring 与初始化日志；
- [../../QWEN3_MOE_4090X4.md](../../QWEN3_MOE_4090X4.md)：命令和验收矩阵。

`raw.jsonl` 中保留了每次运行的完整命令、环境、拓扑、逐请求 token/timestamp、Scheduler
计数和 GPU 采样，可重新聚合，而不依赖继续租用该实例。
