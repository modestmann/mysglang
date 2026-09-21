# Qwen3-0.6B：4×RTX 4090 实测

## 结论

- 正确性通过：mysglang 的 TP=1、2、4 均与 Transformers BF16 的连续 64 个 greedy
  token ID 完全一致。
- 这个 0.6B 模型应使用单卡。并发 8 时 TP=1、2、4 分别为 364.13、222.46、
  168.90 output tok/s；模型太小，PCIe all-reduce 成本超过分片节省的计算。
- 单卡固定 batch=8 的 Decode CUDA Graph 有明显收益：有效 replay 63 次，吞吐从
  364.13 提升至 1459.59 output tok/s（4.01 倍），p50 ITL 从 21.78 ms 降至
  5.12 ms。
- FA2 paged attention 在并发 8 下为 364.13 output tok/s，Torch gather/pad 基线为
  115.19 output tok/s，约为 3.16 倍吞吐；两者的 cache 配置不同，因此峰值显存不宜
  直接横向比较。
- TP 确实降低每 rank 的 Torch 峰值分配：TP1/TP2/TP4 约为 2.94/1.62/0.97 GiB。
  `nvidia-smi` 的整进程峰值约为 3.67 GiB、每卡 2.48 GiB、每卡 1.62 GiB。

## 环境与方法

- 提交：`3864b1e137c09afd6e628002f8eda00a6fbece1a`，加上本次未提交的 benchmark/CLI
  参数改动。
- GPU：4×NVIDIA GeForce RTX 4090 24 GiB；驱动 565.77；无 NVLink。GPU0-2 位于
  NUMA 0，GPU3 位于 NUMA 1，GPU0↔GPU3 等路径需跨 `SYS`。
- Python 3.12.14，PyTorch 2.5.1+cu121，CUDA runtime 12.1，FlashAttention
  2.8.3.post1，BF16。
- 每组先 warmup，再清空 prefix cache；正式请求使用 greedy、`ignore_eos=True`，使每个
  请求恰好生成指定数量 token。性能输入是固定长度的合成 token 序列，不衡量回答质量。
- TTFT 是提交批次到首 token；ITL 是相邻输出 token 的间隔；吞吐是所有请求输出 token
  数除以批次墙钟时间。以下均为一次受控运行，不应当作低噪声统计显著性结论。

## TP 扩展结果

所有行使用 FA2、prompt=128、output=64，不使用 CUDA Graph。

| TP | 并发 | output tok/s | p50 TTFT (ms) | p50 ITL (ms) | p50 E2E (ms) | rank0 Torch 峰值 (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 48.33 | 27.82 | 20.41 | 1324.11 | 2941.54 |
| 1 | 8 | 364.13 | 27.96 | 21.78 | 1405.93 | 2973.06 |
| 2 | 1 | 26.91 | 39.43 | 35.10 | 2378.52 | 1624.57 |
| 2 | 8 | 222.46 | 66.35 | 32.84 | 2301.29 | 1643.18 |
| 4 | 1 | 21.37 | 51.24 | 45.49 | 2994.61 | 969.54 |
| 4 | 8 | 168.90 | 72.33 | 45.38 | 3031.14 | 983.56 |

这里的 TP 是“容量能力”，不是 0.6B 的性能优化。TP2/TP4 的每层 attention 和 MLP 都要
all-reduce；4090 之间又只有 PCIe。等模型大到单卡放不下，TP 才是必要手段。未来测
Qwen3-30B-A3B 时还应加入 EP，让 expert 权重按卡分布，不能从这份 dense 0.6B 数据外推
MoE 性能。

## CUDA Graph、Attention 与长 Prompt

| 场景 | output tok/s | p50 TTFT (ms) | p50 ITL (ms) | p50 E2E (ms) |
|---|---:|---:|---:|---:|
| FA2 eager，C8/P128/O64 | 364.13 | 27.96 | 21.78 | 1405.93 |
| FA2 CUDA Graph，C8/P128/O64 | 1459.59 | 31.21 | 5.12 | 350.62 |
| Torch gather/pad，C8/P128/O64 | 115.19 | 85.96 | 67.78 | 4444.18 |
| FA2 eager，C4/P1024/O32 | 168.81 | 59.01 | 21.92 | 758.14 |

Graph 行在 warmup 中捕获，正式区间记录到 63 次 replay。它只优化形状固定的 Decode；
Prefill 仍 eager，所以 TTFT 没有改善。原始 JSONL 中
`tp1-c8-p128-o64-graph` 是一次评测脚本配置错误：当时仅注册了 B=1 bucket，计数为
0 replay，故从结论排除；`tp1-c8-p128-o64-graph-valid` 才是有效记录。保留错误记录是为了
让实验过程可审计。

## 正确性与覆盖范围

- Transformers BF16、mysglang TP1、TP2、TP4 对同一个 128-token 输入各生成 64 个
  greedy token，三组均 `exact_match=true`。
- TP smoke 同时验证了 rank0 Gloo 控制面、各 rank 镜像 scheduler/batch plan、NCCL 模型
  collective、各 rank paged KV，以及 worker 正常 shutdown。
- 正式批次的 scheduler 记录显示 C8 的 `max_prefill_batch_size=8`、
  `max_decode_batch_size=8`；长 prompt 共处理 4096 个 prefill token。
- 云端完整项目测试：49 passed、3 skipped；FA2 CUDA kernel 独立调用通过。此次改动后本地
  完整测试为 52 passed，Ruff 通过。

尚未实测：真实 Qwen3-30B-A3B MoE、EP、跨机、P/D 分离、投机解码。当前实现明确不支持
MoE TP，而 30B-A3B 的 BF16 权重无法放入单张 24 GiB 4090，因此不能用这四张卡强行声称
MoE 推理已通过。

### MoE 第二阶段（必须补测）

完成 EP 后使用已经下载的 `/root/models/Qwen3-30B-A3B`，在同一台 4×4090 上补充：

1. Transformers 分卡 greedy token 作为正确性参考；mysglang 逐 token 对齐。
2. 朴素逐 expert loop 作为优化前基线，对比按 expert 分组/排序的 dispatch，以及后续 fused
   MoE kernel；同时记录每层 expert 命中分布和负载不均衡。
3. TP、EP、TP+EP（架构允许时）的通信量、TTFT、ITL、吞吐与每卡峰值显存。
4. 并发 1/4/8、短 prompt/长 prompt；区分 Prefill 和 Decode，因为两阶段的 token 数与
   expert dispatch 形态不同。
5. 优化前后使用相同 checkpoint、输入 token、采样参数、warmup 和并发，至少重复 3 次；
   除性能外必须保持 greedy token 对齐，不能用数值错误换吞吐。

在 EP 完成前，本报告只把 MoE 的 config/权重完整性检查记为准备工作，不把它计作推理通过。

## 复现材料

- `raw.jsonl`：10 条完整运行记录，包括逐请求 token ID/时间、scheduler 计数、GPU 采样、
  拓扑、版本和原命令。
- `transformers-alignment.json`：Transformers 参考 token 和三个 TP 配置的逐项比较。
- `../../benchmark_generation.py` 与 `../../validate_transformers.py`：可重复执行的脚本。

下次租卡时，优先做三件事：实现 EP 并按上述矩阵跑 Qwen3-30B-A3B；将关键场景重复
3–5 次并报告方差；再加入真实请求集和服务端压测，分别报告排队时间、TTFT、TPOT 和吞吐。
