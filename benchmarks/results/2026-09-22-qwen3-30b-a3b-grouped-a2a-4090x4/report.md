# Qwen3-30B-A3B：grouped GEMM 与 all-to-all EP 实测

## 结论

在 4×RTX 4090、BF16、FlashAttention 2、TP=4 的同一实例上，`sorted`、`grouped` 和
`all_to_all` 已完成 3 种 workload × 3 次重复，共 27 轮正式测试：

- 9/9 组对应 workload 的逐请求 token ID 完全一致；
- 当前 padded-batched `grouped` 相对 `sorted` 的 output throughput 中位数下降
  2.6%～8.2%，不能替换默认路径；
- `all_to_all` 相对 `grouped` 再下降 20.5%～30.2%；
- 当前实例没有 NVLink，所有 GPU pair 的 CUDA P2P 均不可用，NCCL 的 ring 和
  all-to-all pair channels 全部走 SHM；这正是 all-to-all 不占优的重要环境因素；
- 结果支持下一步直接做无 padding fused/Triton grouped GEMM；当前消费卡云实例上仍应
  默认使用 `sorted + replicated-token all-reduce`。

## 环境与本次拓扑

| 项目 | 实测值 |
|---|---|
| commit | `d70e50c` |
| GPU | 4×NVIDIA GeForce RTX 4090 24 GiB |
| Python / PyTorch | 3.12.14 / 2.5.1+cu121 |
| CUDA runtime / driver | 12.1 / 565.77 |
| FlashAttention | 2.8.3.post1 |
| 模型 | `/root/models/Qwen3-30B-A3B` |
| 精度 / Attention | BF16 / FA2 paged KV |

本次重新分配到的物理拓扑与 2026-09-21 的实例不同：

```text
GPU 0、1、2：NUMA 0，彼此 NODE
GPU 3：NUMA 1，与 GPU 0、1、2 均为 SYS
NVLink：无
CUDA P2P：所有 pair 均为 false / CNS
```

上一轮是 GPU0/1 与 GPU2/3 的 2+2 NUMA 布局；本轮是 3+1。因此不能把两份报告的绝对
吞吐直接当作同一硬件重复。本报告中的三条 backend 在同一拓扑、同一 commit、相同输入
下轮换执行顺序，内部 A/B 仍然有效。完整输出见 [topology.txt](topology.txt)。

`NCCL_DEBUG=INFO` 确认 ring `0 -> 1 -> 2 -> 3` 的边均为
`via SHM/direct/direct`；all-to-all 初始化出的 0↔2、0↔3、1↔3 等 pair channels 也全部
为 SHM，没有 P2P 或 NVLS。见 [nccl-all-to-all.log](nccl-all-to-all.log)。

## 正确性与一次 CUDA 修复

最初的 CUDA smoke 中，all-to-all 路由和通信可以完成，但第 3 个 greedy token 开始偏离。
CPU FP32/Gloo 没有暴露该问题。原因是 expert output 按 destination/expert transport 顺序用
BF16 `index_add`，改变了 top-k contribution 的加法顺序，足以扰动接近的 logits。

修复后，combine 输出先逆置换回原始 token-major top-k slot 顺序，再沿 top-k 求和。提交
`d70e50c` 的 CUDA smoke 恢复为：

```text
Transformers / sorted: [1773, 220, 105444, 107420]
grouped:               [1773, 220, 105444, 107420]
all_to_all:            [1773, 220, 105444, 107420]
```

smoke 见 [smoke-alignment.jsonl](smoke-alignment.jsonl)；Transformers oracle 在上一轮
[报告](../2026-09-21-qwen3-30b-a3b-4090x4/report.md)中归档。本轮 27 个正式结果还逐请求
核对了三条路径，三种 workload × 三次重复全部一致。

## 性能结果

每格为三次重复的中位数；output tok/s 括号内为最小值～最大值。三条路径使用相同模型、
输入、greedy sampling、FA2 和 64 个 KV pages。

| workload | backend | output tok/s | p50 TTFT | p50 ITL | p50 E2E |
|---|---|---:|---:|---:|---:|
| C1 / P128 / O64 | sorted | 9.65 (9.03～9.75) | 235.28 ms | 110.51 ms | 6630.40 ms |
| C1 / P128 / O64 | grouped | 9.09 (9.03～9.95) | 199.89 ms | 114.29 ms | 7044.36 ms |
| C1 / P128 / O64 | all_to_all | 7.22 (6.92～7.38) | 246.31 ms | 122.21 ms | 8864.21 ms |
| C8 / P128 / O64 | sorted | 78.36 (74.97～79.20) | 281.56 ms | 89.45 ms | 6533.90 ms |
| C8 / P128 / O64 | grouped | 76.35 (75.26～77.77) | 239.42 ms | 91.59 ms | 6705.53 ms |
| C8 / P128 / O64 | all_to_all | 53.27 (52.89～55.11) | 314.03 ms | 134.33 ms | 9611.89 ms |
| C4 / P1024 / O32 | sorted | 35.33 (34.44～35.71) | 513.64 ms | 92.35 ms | 3622.88 ms |
| C4 / P1024 / O32 | grouped | 32.42 (30.78～33.21) | 499.65 ms | 119.72 ms | 3947.95 ms |
| C4 / P1024 / O32 | all_to_all | 23.34 (22.81～25.52) | 789.65 ms | 163.34 ms | 5483.31 ms |

### 相对吞吐

| workload | grouped vs sorted | all_to_all vs grouped | all_to_all vs sorted |
|---|---:|---:|---:|
| 单请求 Decode | -5.9% | -20.5% | -25.2% |
| 并发 Decode | -2.6% | -30.2% | -32.0% |
| 长 Prefill | -8.2% | -28.0% | -33.9% |

grouped 的 TTFT 在三类 workload 中分别比 sorted 低约 15.0%、15.0% 和 2.7%，说明把
expert GEMM 合批对单次 Prefill 有一定帮助；但重复 Decode 的 padding、layout 构造、
`max_count.item()` 同步和额外 tensor 搬运抵消了收益，最终 E2E/吞吐均没有变好。

### 显存与利用率

| workload | backend | rank0 Torch allocated | `nvidia-smi` 四卡峰值 | 平均 GPU util 中位数 |
|---|---|---:|---:|---:|
| C1/P128/O64 | sorted | 15,864.6 MiB | 16,720 MiB | 48.3% |
| C1/P128/O64 | grouped | 15,896.7 MiB | 16,722 MiB | 45.4% |
| C1/P128/O64 | all_to_all | 15,902.7 MiB | 16,774 MiB | 40.8% |
| C8/P128/O64 | sorted | 15,892.6 MiB | 16,718 MiB | 44.5% |
| C8/P128/O64 | grouped | 15,935.9 MiB | 16,718 MiB | 49.1% |
| C8/P128/O64 | all_to_all | 15,944.3 MiB | 16,772 MiB | 41.6% |
| C4/P1024/O32 | sorted | 15,988.7 MiB | 16,722 MiB | 49.0% |
| C4/P1024/O32 | grouped | 16,178.1 MiB | 17,230 MiB | 51.2% |
| C4/P1024/O32 | all_to_all | 16,209.3 MiB | 17,376 MiB | 47.4% |

并发 Decode 和 Prefill 中，grouped 的 utilization 更高但吞吐更低，说明多出来的 GPU 工作
主要是 padding/重排，并非有效 token 计算。长 Prefill 下 grouped 增加约 508 MiB 的
`nvidia-smi` 峰值，all-to-all 又增加 dispatch/receive buffers，达到 17,376 MiB。
all-to-all utilization 下降则符合 GPU 等待 SHM collective 的表现。

## 结论与下一步

1. 保持 `sorted` 为默认 backend；它仍是本机器上整体最好的实现。
2. padded-batched grouped 已完成其基线使命：它证明合批能降低部分 TTFT，但 padding、
   layout 和同步使总吞吐下降。下一版若继续优化，应直接使用 offsets/counts 的无 padding
   Triton grouped GEMM，并融合或减少中间 tensor。
3. all-to-all 算法正确，但当前 Attention TP 仍要求最终 all-gather replicated hidden；
   每个 MoE 层实际包含 metadata exchange、dispatch all-to-all、combine all-to-all 和
   final all-gather。在无 P2P、NCCL 走 SHM 的 4090 云实例上，它不适合作为默认路径。
4. 保留 all-to-all 代码作为拓扑对照和未来高速互联机器的基线，不继续扩展 TP×EP mesh。

## 归档文件

- [raw.jsonl](raw.jsonl)：27 轮正式矩阵，包含环境、命令、逐请求 token/timestamp、
  Scheduler 与 GPU 采样；
- [smoke-alignment.jsonl](smoke-alignment.jsonl)：修复后的 all-to-all CUDA smoke；
- [evidence.jsonl](evidence.jsonl)：开启 NCCL debug 的一轮生成记录；
- [nccl-all-to-all.log](nccl-all-to-all.log)：NCCL ring/all-to-all transport；
- [topology.txt](topology.txt)：本次实例 topology、P2P read/write 和 CUDA peer access；
- [矩阵脚本](../../run_qwen3_moe_matrix.sh)：可断点续跑相同 27 轮测试。
