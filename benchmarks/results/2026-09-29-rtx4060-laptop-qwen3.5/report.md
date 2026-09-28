# Qwen3.5-0.8B：RTX 4060 Laptop，MTP × FP8 × CUDA Graph

目的：在本机验证 Qwen3.6-35B-A3B-FP8 所需的全部新路径（Gated DeltaNet、MTP 投机、分块 FP8、
混合模型 CUDA Graph），并给出小模型上的性能基线。`Qwen3.5-0.8B` 与 Qwen3.6 的 DeltaNet/Gated
Attention/MTP 结构相同，只是 MLP 为 dense。FP8 checkpoint 由 `benchmarks/make_fp8_checkpoint.py`
按官方格式转换（157 个 Linear 量化，89 个保持 BF16）。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop 8 GB（sm_89），WSL2 |
| 软件 | torch 2.13.0+cu130、flash-attn 2.8.3.post1、flash-linear-attention 0.5.2、transformers 5.16.1 |
| 代码 | 分支 `qwen3.6-hybrid-mtp`（基于 `9b7a3f3`，JSONL 中的 git commit 是基点） |
| 配置 | BF16 激活，FA2 paged（page 256），DeltaNet 用 fla，全部开启 `--cuda-graph` |

## 正确性（先于性能）

| 检查 | 结果 |
|---|---|
| 0.8B BF16 greedy vs Transformers（64 token） | 逐 token 一致；MTP k=0/1/3 均一致 |
| fp32 + torch 后端，4 并发 × 96 token，MTP 开/关 | 逐位一致（290 个草稿接受 147 个） |
| CUDA Graph vs eager（Decode、MTP k=2 全链路） | 逐 token 一致 |
| FP8 kernel vs `--dequantize-fp8` | 相对 BF16 的首个分叉位置相同（FP8 量化本身在第 18–47 个 token 处改变 greedy） |
| MTP hidden 取法 | post-norm 接受率 69.5%，pre-norm 63.1%（与 vLLM 一致，取 post-norm） |

bf16 下 MTP 开启后与关闭相比可能在几十个 token 后分叉：verify 一次处理 `1+k` 个 token，GEMM 与
attention 的 kernel 形状不同，接近平局的 argmax 会翻转。fp32 reference 逐位一致说明这不是回滚错误。

## 真实对话负载（`--chat`，输出 256 token，greedy）

原始数据：[generation-chat.jsonl](generation-chat.jsonl)。

| 模型 | 并发 | MTP k | output tok/s | 接受率 | 目标前向 | 峰值显存 MiB |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 0 | 98.3 | – | 256 | 1502 |
| BF16 | 1 | 1 | 111.6 | 0.59 | 161 | 1562 |
| BF16 | 1 | 2 | 113.2 | 0.51 | 128 | 1593 |
| BF16 | 1 | 3 | 101.0 | 0.42 | 114 | 1591 |
| BF16 | 4 | 0 | 348.6 | – | 256 | 1581 |
| BF16 | 4 | 1 | 394.2 | 0.56 | 164 | 1660 |
| BF16 | 4 | 2 | 372.5 | 0.42 | 140 | 1670 |
| BF16 | 4 | 3 | 333.6 | 0.34 | 127 | 1778 |
| FP8 | 1 | 0 | 135.0¹ | – | 256 | 1039 |
| FP8 | 1 | 1 | 142.5 | 0.71 | 150 | 1086 |
| FP8 | 1 | 2 | 142.5 | 0.56 | 121 | 1088 |
| FP8 | 1 | 3 | 109.0¹ | 0.41 | 116 | 1121 |
| FP8 | 4 | 0 | 446.8 | – | 256 | 1117 |
| FP8 | 4 | 1 | 477.7 | 0.63 | 157 | 1178 |
| FP8 | 4 | 2 | 442.0 | 0.50 | 129 | 1188 |
| FP8 | 4 | 3 | 399.4 | 0.44 | 111 | 1296 |

¹ 首轮 FP8 c1 mtp0/mtp3 只有 25.0/21.0 tok/s（TTFT > 1 s），与同一时段一次 WSL GPU
"unspecified launch failure" 重合（`dmesg` 有 `dxgkio_escape` 报错），随后 FP8 c4 四组在加载阶段
失败、重跑全部成功。表中为重跑值，原始异常行保留在 JSONL 中，重跑行名字带 `rerun`。

## 合成负载（重复原文作为 prompt）

原始数据：[generation.jsonl](generation.jsonl)。原文被重复拼成 128 token，模型基本在复读，
MTP 接受率为 1.00，只表示机制上界：BF16 c1 从 106.3 升到 171.6 tok/s（k=2，1.61×），
c4 从 347.4 升到 594.0 tok/s（1.71×）；FP8 c4 k=2 为 687.0 tok/s。这些数字不能当作真实加速比。

## 结论

1. **混合模型 CUDA Graph 是最大收益**：eager Decode 约 1500 次 kernel launch/step、CPU 48 ms，GPU
   仅约 14 ms；Graph 后 c4 从约 120 升到约 350 tok/s（约 2.9×）。MTP 的 verify、草稿、追赶和
   DeltaNet 掩码提交也全部进入 Graph 之后，MTP 才不再被 Python 开销拖慢（未捕获提交时只有 187 tok/s）。
2. **FP8 在 decode 上是净收益**：权重显存约 −30%（0.8B 的 embedding 与 lm_head 共享且保持 BF16，
   大头没被压缩），c4 吞吐 +28%。W8A16 仍用 BF16 tensor core，收益来自少读一半权重字节。
3. **0.8B 上 MTP 以 k=1 最划算**：c1 +14%/+6%（BF16/FP8），c4 +13%/+7%。k≥2 时接受率下降、
   verify 多算的 token 抵消了收益。35B-A3B 每 token 的计算与权重读取都重得多，verify 的边际成本
   相对更低，最佳 k 需要在 4×4090 上实测，不能由这里外推。
