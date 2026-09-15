# 第 3 章：Forward Pass 与自回归生成

## 1. 本章目标与边界

本章手写一个可读的 dense Qwen3 forward，并在相同 tiny config、相同权重和相同 token IDs 下与 Hugging Face `Qwen3ForCausalLM` 对齐。

本章只处理完整序列：每生成一个 token 都重新 forward 整个前缀。暂不实现 KV Cache、paged memory、batch scheduler 或 FlashAttention 2。

数据路径为：

```text
input_ids [B, S]
  -> embedding [B, S, H]
  -> decoder layer x L
  -> final RMSNorm
  -> LM head
  -> logits [B, S, V]
  -> last-position argmax
  -> append one token and repeat
```

## 2. Mini-SGLang 对应源码与调用链

| 文件 | 主要对象 | 本章关注点 |
|---|---|---|
| `python/minisgl/engine/engine.py` | `Engine.forward_batch` | 建立 batch context、调用模型、采样 |
| `python/minisgl/models/qwen3.py` | `Qwen3ForCausalLM`, `Qwen3Model`, `Qwen3DecoderLayer` | 模型主干和残差顺序 |
| `python/minisgl/models/utils.py` | `RopeAttn`, `GatedMLP` | 融合 QKV、RoPE attention、SwiGLU |
| `python/minisgl/layers/attention.py` | `AttentionLayer` | Q/K Norm、RoPE、attention backend 边界 |
| `python/minisgl/layers/linear.py` | TP linear classes | column/row parallel 与 all-reduce |
| `python/minisgl/layers/embedding.py` | `VocabParallelEmbedding`, `ParallelLMHead` | vocab shard 与 logits gather |
| `python/minisgl/models/weight.py` | `load_weight` | HF 权重切片以及 Q/K/V、gate/up 合并 |
| `python/minisgl/engine/sample.py` | `Sampler` | greedy argmax 或随机采样 |

一次 Mini-SGLang forward 的主链是：

```text
Scheduler._forward
  -> batch.input_ids = token_pool[input_mapping]
  -> Engine.forward_batch
  -> Context.forward_batch(batch)
  -> Qwen3ForCausalLM.forward
  -> Qwen3Model.forward
  -> VocabParallelEmbedding
  -> Qwen3DecoderLayer x L
       -> RMSNorm
       -> RopeAttn
          -> fused QKV projection
          -> per-head Q/K RMSNorm
          -> RoPE
          -> AttentionBackend.forward
          -> row-parallel O projection + all-reduce
       -> residual
       -> RMSNorm
       -> fused gate/up + SiLU multiply
       -> row-parallel down projection + all-reduce
       -> residual
  -> final RMSNorm
  -> ParallelLMHead + logits all-gather
  -> Sampler.sample
```

Mini-SGLang 面向高性能推理，因此 Q/K/V 和 gate/up 分别融合成一个 linear，输入 token 也按 batch 中实际 token 数展平。模型的 `forward()` 不接收显式参数，而是从全局 `Context.batch` 读取 input IDs、positions 和 backend。这让运行路径很短，但隐藏依赖使层级单测更难写。

## 3. 六个数学组件

### 3.1 RMSNorm

对最后一维归一化：

```text
rms(x) = sqrt(mean(x²) + eps)
y = weight * x / rms(x)
```

实现先把激活转成 fp32 计算方差，再转回原 dtype，避免低精度平方和求均值积累过多误差。RMSNorm 不减均值，这一点与 LayerNorm 不同。

### 3.2 Q/K/V 与 GQA

本章形状：

```text
Q: [B, Nq,  S, D]
K: [B, Nkv, S, D]
V: [B, Nkv, S, D]
H = Nq * D
Nq % Nkv == 0
```

例如 `Nq=4, Nkv=2`，每个 KV head 服务两个 query heads：

```text
query heads: q0 q1 q2 q3
key heads:   k0 k0 k1 k1
value heads: v0 v0 v1 v1
```

GQA 减少 K/V 权重和未来 KV Cache 的体积，但保持更多 query heads。

Qwen3 在 RoPE 前还对每个 query/key head 的 `D` 维分别做 RMSNorm；原有 MySGLang tiny 原型漏掉了这一点，本章已补齐。

### 3.3 RoPE

默认 Qwen3 RoPE 把 head 的前半维与后半维配成二维旋转：

```text
rotate_half([x1, x2]) = [-x2, x1]
rope(x) = x * cos(position) + rotate_half(x) * sin(position)
```

旋转改变方向但不改变每个 head 的向量范数。原 tiny 原型使用相邻偶奇维配对；那也是一种 RoPE layout，但无法与当前 Hugging Face Qwen3 权重直接对齐，因此本章改为 Qwen3 的 half-split 约定。

### 3.4 Causal attention

```text
scores = Q @ Kᵀ / sqrt(D)
scores[future positions] = -inf
probabilities = softmax(scores)
output = probabilities @ V
```

`is_causal=True` 保证位置 `i` 只能读取 `0..i`。本章使用 PyTorch SDPA 写最小实现；Hugging Face baseline 强制使用 eager attention，使 oracle 与被测路径不依赖同一个 attention 调用。

### 3.5 Gated MLP / SwiGLU

```text
gate = SiLU(x @ W_gate)
up   = x @ W_up
out  = (gate * up) @ W_down
```

Mini-SGLang 将 gate/up 合并以减少 kernel launch；本章保留三个独立 linear，让权重能直接按名字与 Hugging Face 对齐。

### 3.6 Residual 与 LM head

Qwen3 使用 pre-norm：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

最后执行 RMSNorm 和 LM head，将 `[B,S,H]` 投影为 `[B,S,V]`。生成循环只读取 `logits[:, -1, :]`，因为最后位置预测下一个 token。

## 4. Hugging Face baseline

测试构造一个很小的 `Qwen3Config`，不下载 checkpoint：

```text
V=32, H=24, I=48, L=2, Nq=4, Nkv=2, D=6
```

Hugging Face 随机初始化权重后，将其 state dict 中的 `model.` 前缀移除并严格加载进 `TinyCausalLM`。两边因此拥有完全相同的参数；比较的是实现，而不是两个不同随机模型。

验收包括：

1. 两个 batch 的所有位置、所有 vocab logits 在 `1e-5` 容差内相等；
2. 两边都禁用 KV Cache、每步重算完整前缀，连续四个 greedy token 完全相等；
3. 单独验证 RMSNorm 公式、RoPE 范数不变、causal prefix 不受未来 token 影响。

当前实验的最大 logits 绝对误差约为 `3.17e-08`，greedy tokens 完全一致。

## 5. 借鉴、教学简化与有意改进

### 借鉴 Mini-SGLang

- dense Qwen3 的 pre-norm residual 顺序；
- Q/K per-head RMSNorm、RoPE、GQA 和 gated MLP；
- 模型配置不绑定具体 checkpoint；
- logits 后再独立采样。

### 教学简化

- 用 `nn.Module` 和普通二维 batch，不使用全局 runtime context；
- Q/K/V、gate/up 暂不融合；
- 不做 TP 权重切分、KV Cache、CUDA Graph 和定制 kernel；
- greedy generation 只支持 batch size 1。

### 有意改进

- Hugging Face Qwen3 是可执行端到端 oracle，不把随机输出硬编码成 golden tokens；
- 组件测试与端到端 logits 对齐同时保留，失败时更容易定位；
- HF eager attention 与 MySGLang SDPA 使用不同实现路径，避免“同一个函数和自己比较”；
- 默认不绑 embedding/LM-head 权重，与 Qwen3 默认配置一致；需要时可显式开启。

## 6. 运行与观察

从项目根目录执行：

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/01_forward_and_generate.py
PYTHONPATH=src python examples/03_compare_hf_qwen3.py
```

第一个示例只看 shape、长度和机械运行；第二个才是正确性实验。第二个示例应满足：

```text
max absolute logit error < 1e-5
tokens match: True
```

随机 token 本身是否像自然语言不属于本章验收，因为没有加载训练 checkpoint 或 tokenizer。

## 7. 亲手练习：验证 GQA 映射

请在 `tests/test_tiny_model.py` 中新增 `test_repeat_kv_maps_query_heads_to_groups`。构造形状 `[1,2,1,1]` 的 KV tensor，其中两个 head 的值分别为 10、20，调用 `_repeat_kv(tensor, repeats=2)`。

验收：

```text
输出 shape == [1,4,1,1]
四个 head 的值 == [10,10,20,20]
```

这条测试证明 `Nq=4,Nkv=2` 时，query heads 0/1 共用 KV head 0，query heads 2/3 共用 KV head 1。

完成练习、运行全部测试并理解 HF 对齐结果后停下，不开始第 4 章 KV Cache。
