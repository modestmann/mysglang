# 2026-09-23 RTX 4090 ×4 最终实测

本轮在同一台 4×RTX 4090 实例上验收 Qwen3-0.6B、Qwen3-30B-A3B、FA2 paged KV、
no-padding Triton MoE、TP/NCCL CUDA Graph 与 n-gram 投机解码。正式数字均为三轮
中位数；greedy 请求固定输出长度并忽略 EOS。

## 环境与完整性

- GPU：4×RTX 4090 24 GiB；Driver `580.95.05`；PyTorch `2.8.0+cu128`；
  NCCL `2.27.3`；Triton `3.4.0`；Python `3.12.3`。
- FlashAttention `2.8.3.post1` 由 CUDA 12.8 源码编译；`cuobjdump` 确认只含
  `sm_80`。原生 CUDA kernel 对 SDPA 的最大绝对误差为 `0.000244`。
- 项目全测通过，3 项按环境条件跳过；`pip check` 通过。
- 0.6B 权重 SHA256 为
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，与远端
  ETag 一致。30B-A3B 共 26/26 文件、16/16 safetensors shard、18,867 个 tensor，
  总计 61,084,187,391 bytes；index、header、offset 与文件长度全部通过检查。
- JSONL 内记录的 Git HEAD 是 `7a481b6`；Graph 退出与确定性 MoE combine 是本轮实测
  中发现后加入的工作树修复，并随本报告一起提交。

## 拓扑

四张卡均在 NUMA 0，但没有 NVLink，CUDA P2P 12 个方向全部不可用。GPU1↔2 是
`PIX`，GPU1/2↔3 是 `PXB`，GPU0↔其余是 `NODE`。NCCL 实际走 host SHM：

| 64 MiB/rank collective | 延迟 | 算法带宽 |
|---|---:|---:|
| AllReduce | 15.216 ms | 4.410 GB/s |
| All-to-all | 8.325 ms | 8.061 GB/s |

因此 TP=4 在这里主要解决 30B-A3B 容量，不应期待线性算力扩展。完整证据见
[`topology.txt`](topology.txt)。

## Qwen3-0.6B：Dense CUDA Graph

Transformers BF16 oracle、eager 与 Graph 的 64 个 greedy token 完全一致。

| 并发 | eager output tok/s | Graph output tok/s | 加速 | eager / Graph p50 ITL |
|---:|---:|---:|---:|---:|
| 1 | 33.38 | 205.66 | 6.16× | 29.46 / 4.36 ms |
| 8 | 250.35 | 1364.28 | 5.45× | 31.23 / 5.32 ms |

正式区间每轮有 63 次 replay；capture 在 warmup 后被统计重置，因此 JSONL 中 capture
为 0，不表示没有捕获。Graph 减少 CPU/Python/CUDA driver 逐 kernel 发射开销，不减少
模型 FLOPs。

## Qwen3-30B-A3B：MoE kernel 与 CUDA Graph

先用 4-token smoke 验证 sorted、Triton eager、Triton Graph，三条路径 token 都是
`[1773, 220, 105444, 107420]`。下表使用旧数据中的未改动 sorted 基线，以及确定性
combine 修复后的 Triton 数据：

| Decode | sorted eager | Triton eager | Triton + Graph | Graph / sorted |
|---|---:|---:|---:|---:|
| c1 / p128 / o64 | 9.15 tok/s | 8.66 tok/s | 44.89 tok/s | 4.91× |
| c8 / p128 / o64 | 69.32 tok/s | 69.05 tok/s | 295.55 tok/s | 4.26× |

每个 expert 在 Decode 中只收到很少 token，no-padding grouped GEMM 单独并未加速；
排序、布局和固定顺序 combine 足以抵消 kernel 收益。Graph 捕获 embedding、各层
attention/MoE、TP NCCL、LM head 与 argmax 后，才显著减少提交开销。

长 Prefill（c4 / p1024 / o32）：

| backend | output tok/s | p50 TTFT |
|---|---:|---:|
| sorted | 30.90 | 588.82 ms |
| deterministic Triton | 31.08 | 567.41 ms |

Triton 吞吐仅 `+0.6%`，TTFT `-3.6%`。结论是保留其作为 Graph 所需的静态 kernel，
但不要把它描述成当前 workload 下独立的大幅 GEMM 优化。

### Graph 退出修复

首次 MoE Graph smoke 已写完 JSONL，但 worker 在销毁 NCCL process group 时没有退出，
每卡残留约 16.7 GiB，导致下一轮加载 OOM。修复后 shutdown 会在所有 rank 上先同步并
reset 捕获的 Graph，再销毁 NCCL/Gloo。四卡回归 `torchrun exit=0`，随后
`nvidia-smi` 无 compute PID；后续所有 Graph 矩阵均自然退出。

### MoE combine 确定性修复

原 `triton_grouped` 用一次 `index_add_` 把同一 token 的多个 top-k expert 输出合并，
BF16 冲突原子的执行顺序不固定；相同配置重复运行也会在临界 logits 处分叉。现在先把
每个 assignment 写入唯一的 `[token, top-k slot]`，再按 slot 固定顺序求和。修复后
Decode、Prefill、n-gram 的 off/off 与 on/on 三轮 token 均稳定。代价是额外临时布局，
所以应以修复后的数字作为正式结果。

## N-gram 投机解码

真实 30B-A3B、Triton MoE、30-token 中文问题、128-token 输出：

| 并发 | off | on | 变化 | 接受率 | model forwards |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.82 tok/s | 10.56 tok/s | +19.7% | 38.0% | 128 → 109 |
| 8 | 68.88 tok/s | 79.93 tok/s | +16.0% | 36.6% | 128 → 110 |

c1 的 off/on 三轮均逐 token 完全一致。c8 的 off/off、on/on 各自稳定，但 off/on 在
固定位置分叉：普通 Decode 使用 `flash_attn_with_kvcache`，投机验证使用 packed
`flash_attn_varlen_func`；BF16 分块/归约顺序改变临界 logits 的 greedy `argmax`。
Torch attention c8 对照能逐 token 对齐，说明请求分段、KV 回滚和调度语义正确。

0.6B 的循环 prompt 机制上界达到 100% 接受率、c1 `33.44 → 141.67 tok/s`；普通
30-token 问题接受率为 88.9%，c1 `33.51 → 82.58 tok/s`。这两组说明 n-gram 很依赖
重复程度；不应无条件默认打开，后续可按历史命中率/接受率自适应启停。

## 最终结论

1. FA2、TP4 Qwen3-30B-A3B、no-padding Triton MoE、MoE/NCCL CUDA Graph 与 n-gram
   均已在真实四卡环境运行，不再是“只完成代码、待云端验证”。
2. 当前最有价值的优化是 Decode CUDA Graph；Triton grouped 本身在本机只带来很小的
   Prefill 收益，但它提供 MoE Graph 所需的静态 GPU 路径。
3. n-gram 在本次真实 MoE 问题上提升 16%～20%，但收益和精确 token 复现都受 workload
   与 kernel 数值路径影响，适合作为可选、自适应优化。
4. 这台机器无 P2P/NVLink；先前 all-to-all EP 已被实测证明慢于 replicated-token
   all-reduce，继续做 P/D 分离或复杂 TP×EP mesh 的优先级不高。

## 原始文件

- [`qwen3-0.6b.jsonl`](qwen3-0.6b.jsonl)：Dense Graph 与 0.6B n-gram 三轮数据；
- [`qwen3-0.6b-transformers.json`](qwen3-0.6b-transformers.json)：Transformers oracle；
- [`qwen3-30b-a3b-fixed.jsonl`](qwen3-30b-a3b-fixed.jsonl)：确定性修复后的正式数据；
- [`qwen3-30b-a3b.jsonl`](qwen3-30b-a3b.jsonl)：修复前矩阵和问题定位证据；
- `*-fixed.log`：每条四卡命令的完整标准输出；
- [`topology.txt`](topology.txt) 与 [`setup-logs`](setup-logs/)：硬件、NCCL、环境与 FA2
  编译证据。
