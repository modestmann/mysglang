# 第 4 章：KV Cache——从重复前缀到增量 Decode

## 1. 本章目标与边界

本章实现每请求、逐层、固定容量的连续 KV Cache。Prefill 仍一次处理整个 prompt；随后每轮 Decode 只把一个新 token 送入 Q/K/V projection，同时读取缓存中的全部历史 K/V。

本章不实现多请求调度、paged allocation、prefix sharing、Radix tree 或 FlashAttention 2。它们依赖的正确性 oracle 正是本章的 cached/uncached 对齐。

## 2. 为什么需要 KV Cache

设 prompt 长度为 `P`，需要生成 `N` 个 token。无 cache 的第 `i` 次 forward 输入长度为 `P+i`：

```text
P, P+1, P+2, ..., P+N-1
```

因此每层 Q/K/V projection 重复处理的 token 总数为：

```text
N*P + N*(N-1)/2
```

使用 cache 后：

```text
Prefill: P 个 token
Decode:  1, 1, ..., 1
```

生成 `N` 个 token 时只需执行 `N-1` 次 decode forward，因为最后生成的 token 不再用于预测后续 token。因此 projection token 数为：

```text
P + N - 1
```

KV Cache 没有让 attention 不再读取历史。Decode 的一个 query 仍需与 `P+i` 个历史 keys 做 attention；优化掉的是对历史 token 的 embedding、Q/K/V projection、RoPE、MLP 等重复计算，以及历史 query 之间重复形成的大注意力矩阵。

## 3. Mini-SGLang 源码地图

| 文件 | 核心对象 | 责任 |
|---|---|---|
| `python/minisgl/core.py` | `Req.cached_len/device_len/extend_len` | 区分已有 KV、新输入和未生成区间 |
| `python/minisgl/engine/engine.py` | `Engine` | 创建 KV pool 和 GPU page table |
| `python/minisgl/kvcache/base.py` | `BaseKVCachePool` | 定义逐层 K/V 读取与写入协议 |
| `python/minisgl/kvcache/mha_pool.py` | `MHAKVCache` | 分配物理 K/V 大 tensor，写入指定位置 |
| `python/minisgl/scheduler/cache.py` | `CacheManager` | 分配页、更新 page table、回收和驱逐 |
| `python/minisgl/scheduler/scheduler.py` | `_prepare_batch` | 生成 positions、input mapping 和 `out_loc` |
| `python/minisgl/attention/fa.py` | `FlashAttentionBackend` | 写入本轮 K/V，准备长度和页表 metadata |
| `python/minisgl/attention/fi.py` | `FlashInferBackend` | 同一 cache 协议的另一 kernel adapter |

Mini-SGLang 的物理 pool 形状为：

```text
[2, layers, pages, page_size, local_kv_heads, head_dim]
 ^
 K/V
```

每层 attention 的调用链：

```text
Scheduler
  -> CacheManager.allocate_paged(reqs)
  -> batch.out_loc = page_table[input_mapping]
  -> AttentionBackend.prepare_metadata(batch)
  -> Model layer computes current q/k/v
  -> kv_cache.store_kv(k, v, batch.out_loc, layer_id)
  -> kernel reads page_table + complete cached K/V
  -> attention output
```

这里必须区分两种所有权：`MHAKVCache` 拥有 GPU tensor；`CacheManager` 拥有哪些物理位置空闲、属于请求或可被驱逐的逻辑。Attention backend 只消费 Scheduler 准备好的映射，不决定请求应占哪些页。

## 4. MySGLang 的连续 Cache

`ContiguousKVCache` 使用：

```text
keys/values: [layers, batch, kv_heads, max_length, head_dim]
```

每个 layer 有自己的有效长度。一次完整 forward 开始与结束时，所有层长度必须相同；层执行过程中则按顺序各自追加本轮 K/V。

Prefill：

```text
input_ids: [B,P]
positions: [0,1,...,P-1]
cache before: length 0
cache after:  length P
attention: causal=True
```

第一次增量 Decode：

```text
input_ids: [B,1]，内容是 Prefill 采样出的 token
position:  [P]
new K/V 写入 cache[:, P]
attention keys: cache[:, :P+1]
cache after: length P+1
```

本章在创建 cache 时预分配完整容量，避免每 token `torch.cat` 重新分配和复制历史。缺点是每个请求都按最大长度占用连续空间，产生内部浪费；paged KV Cache 章节会解决这一点。

## 5. 一个容易写错的 causal mask 细节

无 cache 的方阵 attention 可以直接使用：

```python
scaled_dot_product_attention(q, k, v, is_causal=True)
```

但 cached decode 的形状是：

```text
q length = 1
k length = past_length + 1
```

PyTorch 对非方阵 `is_causal=True` 使用上左对齐的 causal bias；这会让单个 query 只看到最早位置，而不是完整历史。由于 decode cache 中没有未来 key，本章对单-token cached decode 使用 `is_causal=False`。如果未来支持一次追加多个 query token，则必须构造带 `past_length` 偏移的显式 causal mask，不能直接沿用当前分支。

因此本章明确拒绝“cache 非空时一次输入多个 token”，避免静默产生错误 logits。

## 6. Generation 调用链

无 cache：

```text
model(prompt) -> token 1
model(prompt + token 1) -> token 2
model(prompt + token 1 + token 2) -> token 3
```

有 cache：

```text
model(prompt, empty_cache) -> token 1
model(token 1, cache)      -> token 2
model(token 2, cache)      -> token 3
```

输出 token 序列必须完全相同，优化只能改变计算复用方式，不能改变模型语义。

## 7. 正确性 oracle

自动测试覆盖三层 oracle：

1. MySGLang cached 每一步 logits 对齐 MySGLang 完整前缀重算；
2. cached 与 uncached greedy tokens 完全相等；
3. 相同权重下，MySGLang cache 对齐 Hugging Face `DynamicCache` logits。

另有结构测试证明 prompt 长度 3、生成 5 token 时，第一层 Q projection 只处理：

```text
3 + 1 + 1 + 1 + 1 = 7 tokens
```

而无 cache 会处理：

```text
3 + 4 + 5 + 6 + 7 = 25 tokens
```

## 8. 借鉴、教学简化与有意改进

### 借鉴 Mini-SGLang

- K/V 按 layer 独立存储；
- 先写入本轮 K/V，再让 attention 读取包含新位置的完整历史；
- cached length 决定 RoPE position；
- cache storage 与 attention 计算通过窄接口连接。

### 教学简化

- 每请求一个连续 cache，不共享物理 pool；
- 固定容量、batch size 固定，不做动态 admission；
- 没有 page table、prefix match、eviction 或 chunked prefill；
- 只支持 cache 为空的 Prefill，以及 cache 非空后的单-token Decode。

### 有意改进

- cache 是显式传参，不通过全局 context 隐式访问；
- shape、dtype、device、capacity 和 layer index 都有边界检查；
- 所有层长度不一致会立即报错；
- HF `DynamicCache` 和完整前缀重算同时作为 oracle。

## 9. 运行实验

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/04_kv_cache.py
```

有可见 GPU 时可以显式运行：

```bash
PYTHONPATH=src python examples/04_kv_cache.py --device cuda --prompt-length 128 --new-tokens 64
```

实验必须先显示 `tokens match: True`。投影 token 数是确定性指标；延迟受设备、shape 和 PyTorch kernel 影响，小 CPU workload 上 cache 不保证更快，不应只挑有利数字。

## 10. 亲手练习：容量溢出不能破坏状态

请在 `tests/test_kv_cache.py` 新增：

```text
test_cache_overflow_does_not_mutate_length
```

步骤：创建 `max_length=4` 的 cache；Prefill 三个 token；Decode 一个 token，使 cache length 达到 4；再 Decode 一个 token。

验收：最后一次 Decode 抛出包含 `capacity` 的 `ValueError`，且异常后 `cache.length` 仍为 4。这条测试保证 admission 错误不会留下部分写入的 cache。

完成并理解实验后停下，不开始 HTTP 或 paged cache。
