# 第 8 章：RadixAttention 与前缀缓存

第 7 章解决了“KV 放在哪里”：所有请求共享一个物理 page pool，每个请求通过 page table
组织自己的逻辑序列。但请求结束后，它的页面会立刻释放；下一个具有相同 system prompt 的请求
仍然需要重新 Prefill。

本章解决“哪些 KV 值得留下并复用”：用压缩 Radix Tree 把 token 前缀映射到已经算好的物理 KV
页面。新请求命中前缀后直接引用这些页面，只计算未命中的 prompt 后缀。

本章完成后，应当能够回答：

1. 为什么 Radix Tree 的 key 是 token，而 value 是物理 page；
2. 为什么只能复用完整页面，并且最多匹配到 `prompt[:-1]`；
3. 一个页面如何同时处于 cached、protected、evictable 三种生命周期状态；
4. 为什么引用计数保护正确性，LRU 只决定“从哪些可淘汰叶子开始”；
5. prefix hit 为什么减少 Prefill token，却不改变生成 token。

## 1. Mini-SGLang 源码地图

主要参照 `/home/sheep/mini-sglang`：

| Mini-SGLang | 关键职责 |
|---|---|
| `python/minisgl/kvcache/radix_cache.py` | 压缩节点、split、match、insert、ref count、LRU leaf eviction |
| `python/minisgl/kvcache/base.py` | cache handle、match/insert 结果与 prefix cache 协议 |
| `python/minisgl/scheduler/cache.py` | match、lock、page allocation、cache finished request、evict |
| `python/minisgl/scheduler/prefill.py` | 用 `cached_len` 减少 Prefill 输入 |

MySGLang 的最小实现：

| 本项目 | 关键职责 |
|---|---|
| `src/mysglang/cache/radix.py` | `RadixPrefixCache` 与 `RadixPagedKVCache` |
| `src/mysglang/cache/paged.py` | 支持 cached page 和多个请求只读共享同一页 |
| `src/mysglang/scheduler/radix_scheduler.py` | admission、恢复命中长度、完成时插入、abort 解锁 |
| `tests/test_radix_cache.py` | split、页对齐、引用保护、LRU、reset、随机不变量 |
| `tests/test_radix_scheduler.py` | 生成对齐、Prefill 降低、共享保护、物理池淘汰 |

## 2. 数据结构：压缩 Radix Tree

普通 Trie 每个 token 一个节点。Radix Tree 把只有一个分支的连续 token 压成一条 edge：

```text
root
 └── tokens [1,2,3,4] -> pages [7,2]
      ├── tokens [5,6] -> page [9]
      └── tokens [8,8] -> page [4]
```

当已有 edge 是 `[1,2,3,4,5,6]`，新 key 只共享 `[1,2,3,4]` 时，节点在 page boundary
处分裂：

```text
before: root -> [1,2,3,4,5,6]

after:  root -> [1,2,3,4]
                    └── [5,6]
```

节点的 `key` 是 token IDs，`pages` 是相同区间 KV 所在的物理页。树不复制 K/V tensor；真正数据
仍在 `PagedKVCache._keys/_values`。

### 为什么必须页对齐

当前 page table 的最小共享单位是完整物理页。如果 page size 为 4，即使两个 prompt 共享 6 个
token，也只能复用前 4 个：

```text
shared tokens:  0 1 2 3 | 4 5
reused page:   [   page 0   ]
recompute:                  4 5
```

这样 page table 不需要表达“某页只共享前半段”，也不会让两个请求写同一物理页的不同尾部。

## 3. 为什么不能匹配 prompt 的最后一个 token

KV Cache 保存的是 attention 所需的 K/V，不保存 LM head 对最后位置计算出的 logits。生成首个 token
需要 prompt 最后位置的 logits，因此匹配范围是：

```python
prefix_cache.match_prefix(prompt_token_ids[:-1])
```

然后向下对齐到完整页。这样无论命中多长，至少还有一个 prompt token 会进入模型 forward，产生
首个输出 token 所需 logits。

这也是为什么“完全相同 prompt”仍可能有一个 token 加上不足一页的尾部需要 Prefill。

## 4. 页面生命周期

页面的物理状态不是简单的 free/used 二元组：

```text
free
  │ request Prefill/Decode 分配
  ▼
request-private
  │ 请求完成，完整页插入 radix
  ▼
cached + evictable (ref_count == 0)
  │ 新请求 match 并 lock
  ▼
cached + protected (ref_count > 0)
  │ 请求完成或 abort，unlock
  ▼
cached + evictable
  │ page pool 不足，LRU eviction
  ▼
free
```

同一个 cached page 可以出现在多个活动请求的 page table 中，但它们只读历史 KV；每个请求的新
token 总是写入自己的 private page。非 cached page 仍然只能有一个请求 owner。

`PageAllocator.check_integrity()` 会验证：

- free page 不能同时出现在 request table 或 prefix cache；
- 只有 cached page 可以被多个 request table 引用；
- page table、reservation 和 request ID 集合一致；
- free、active、cached 的并集恰好覆盖整个物理池。

## 5. Match、锁定与 Admission

新请求开始 Prefill 前：

```text
prompt token IDs
  -> match prompt[:-1]
  -> 得到 RadixCacheHandle(cached_len, pages)
  -> lock handle 路径
  -> 用 pages 初始化 request page table
  -> 每层 KV length 初始化为 cached_len
  -> prefill_offset = cached_len
```

锁必须发生在使用 page table 之前。否则 match 返回页面后、模型读取之前，另一次 allocation 可能
将它 LRU 淘汰并复用，造成静默错误。

Admission 同时计算：

```text
已经承诺给活动请求的 private pages
+ 当前 protected cached pages
+ 本请求未命中部分的最大 pages
<= 物理池总页数
```

evictable cached pages 不阻止准入；真正 allocation 时如果 free page 不够，就从最老的、未锁定的
leaf 开始淘汰。节点是压缩 edge，所以一次淘汰可能略多于最低需求。

## 6. 请求结束时如何写回

请求执行期间不立即向树中暴露新页面。请求完成后：

1. 读取所有层一致的实际 KV length；
2. 将 token 序列向下对齐到完整页；
3. 插入 Radix Tree；
4. 如果另一请求已经缓存了相同区间，换成 canonical pages，并释放重复页面；
5. 解锁旧 match handle；
6. 释放不足一页的 tail 和未使用 reservation。

Abort 路径更保守：不插入尚未完成请求的新 KV，只解锁原命中页面并释放 private pages。

## 7. LRU 为什么只淘汰 leaf

如果直接删除内部节点，它的 children 将失去到 root 的 token 路径。实现因此只把
`ref_count == 0` 的 leaf 放进按 timestamp 排序的 heap：

```text
pop oldest evictable leaf
  -> 回收该 edge 的 pages
  -> parent 若变成 evictable leaf，再加入 heap
  -> 直到回收页数满足 allocation
```

访问命中节点会更新时间；锁只控制能不能淘汰，timestamp 决定在可淘汰集合中的先后顺序。

## 8. 正确性 Oracle 与测试

Radix Scheduler 的生成结果仍与单请求 `greedy_generate_cached` 对齐。测试不只检查最终 token，还
检查优化确实发生：两个长度为 6、共享前 4 token 的 prompt，第一次 Prefill 6 token，第二次只
Prefill 2 token。

运行本章测试：

```bash
cd /home/sheep/mysglang
PYTHONPATH=src ~/nano-vllm/.venv/bin/python -m unittest \
  tests.test_radix_cache tests.test_radix_scheduler -v
```

运行全量回归：

```bash
PYTHONPATH=src ~/nano-vllm/.venv/bin/python -m unittest discover -s tests -v
```

## 9. 共享前缀实验

实验让四个请求复用相同 system prompt，并分别跑普通 Paged Scheduler 与 Radix Scheduler：

```bash
PYTHONPATH=src ~/nano-vllm/.venv/bin/python examples/08_radix_prefix_cache.py
```

写入可追加的 JSONL：

```bash
PYTHONPATH=src ~/nano-vllm/.venv/bin/python examples/08_radix_prefix_cache.py \
  --jsonl results/radix.jsonl
```

当前 tiny CPU workload 的稳定工作量指标是：

```text
baseline prefill input tokens: 348
radix prefill input tokens:    136
outputs match:                 true
```

耗时依赖硬件和系统负载，因此应读取 JSONL 实测值，不把单次 wall time 当验收条件。

启动 Radix HTTP 服务：

```bash
PYTHONPATH=src ~/nano-vllm/.venv/bin/python examples/08_http_server.py
```

HTTP schema、SSE 和 abort 协议没有改变；变化只发生在 service 后面的 scheduler/cache backend。

## 10. 与 Mini-SGLang 的差异

忠实保留：

- 压缩 Radix Tree 与 page-aligned match；
- handle 路径引用计数；
- leaf-only LRU eviction；
- `prompt[:-1]` 匹配边界；
- scheduler 持有 cache 生命周期，而不是 generation helper。

教学简化：

- Python tuple 比较代替自定义快速比较 kernel；
- 每轮只 Prefill 一个请求，没有 production 级 mixed batch；
- PyTorch attention 会 gather 历史 K/V，尚未直接消费 block table；
- 没有 distributed cache、host cache、tenant namespace 或 cache-aware routing。

本章有意补齐的可测试能力：

- `reset()` 返回全部 cached pages；
- 完整 tree/allocator/scheduler integrity checker；
- 显式 hit、cached/protected/evictable/eviction stats；
- 随机 insert/match/evict 不变量测试；
- 不同请求参数仍共享 token-identical prefix 的生成正确性边界。

## 11. 亲手练习

给 prefix key 增加 `cache_namespace`，用于隔离不同模型版本或租户。验收条件：

1. 相同 token、相同 namespace 可以命中；
2. 相同 token、不同 namespace 命中长度为 0；
3. eviction、reset 和 integrity tests 仍通过；
4. namespace 不进入模型 tensor，只参与 radix key。

完成练习后停下；第 9 章再定义 AttentionBackend 并接入 FlashAttention/CUDA Graph。
