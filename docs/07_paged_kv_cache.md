# 第 7 章：Paged KV Cache 与显存管理

## 1. 本章目标与边界

第 6 章的 `SlotKVCache` 为每个并发 slot 预留完整 `max_position_embeddings`：

```text
[layers, max_running_requests, max_length, kv_heads, head_dim]
```

短请求也占一个完整 slot，而且 slot 之间不能借用剩余空间。本章改成所有请求共享一个物理 page
pool。请求只在跨越 page boundary 时增加一页，结束或 abort 后立即归还。

本章实现：

- 物理 K/V page pool；
- 每请求逻辑 block table；
- 分离 reservation 与 physical allocation；
- OOM-safe admission；
- 完成和 abort 后回收；
- allocator/cache integrity checker 与随机操作测试；
- PyTorch reference gather attention。

本章不实现 prefix sharing、Radix tree、LRU eviction、host cache 或 FlashAttention 2 paged kernel。
它们分别属于后面的 prefix cache 与 attention backend 章节。

## 2. Mini-SGLang 源码地图

| 文件 | 核心对象 | 责任 |
|---|---|---|
| `python/minisgl/kvcache/mha_pool.py` | `MHAKVCache` | 分配物理 K/V tensor，并按 `out_loc` 写入 |
| `python/minisgl/kvcache/base.py` | `BaseKVCachePool` | 定义 layer cache 和 `store_kv` 接口 |
| `python/minisgl/scheduler/cache.py` | `CacheManager` | 空闲页、按需分配、回收、prefix lock/evict |
| 同上 | `_write_page_table` | 把新分配物理页写入请求 page table |
| `python/minisgl/scheduler/table.py` | `TableManager` | 分配请求 table row，并持有 token pool |
| `python/minisgl/scheduler/prefill.py` | `PrefillAdder` | 准入前估算 prompt/output 所需容量 |
| `python/minisgl/scheduler/decode.py` | `DecodeManager.inflight_tokens` | 统计运行请求剩余 token 与页尾预留 |
| `python/minisgl/scheduler/scheduler.py` | `_prepare_batch` | 分配页并生成本轮 `batch.out_loc` |
| `python/minisgl/engine/engine.py` | `_determine_num_pages` | 根据剩余显存和每页字节数决定 page 数 |
| `python/minisgl/attention/fa.py` | `FlashAttentionBackend` | 把 page table/seq length 交给 paged kernel |

Mini-SGLang 的物理 tensor 是：

```text
[2, layers, num_pages, page_size, local_kv_heads, head_dim]
 ^
 K/V
```

每页占用字节：

```text
2 * layers * page_size * local_kv_heads * head_dim * dtype_bytes
```

Engine 先测量模型加载前后显存，根据 `memory_ratio` 留给 KV Cache 的预算除以上述每页字节数，
得到 `num_pages`。配置中的 `num_page_override` 用于测试或人工限制。

## 3. Mini-SGLang 的分配调用链

```text
PrefillAdder._try_allocate_one
  -> 检查 table row
  -> prefix match
  -> 估算 extend_len + output_len
  -> 检查 CacheManager.available_size
  -> 分配 request table_idx

Scheduler._prepare_batch
  -> CacheManager.allocate_paged(reqs)
       -> ceil(cached_len / page_size)
       -> ceil(device_len / page_size)
       -> 只分配新跨越的 pages
       -> _write_page_table(...)
  -> batch.out_loc = page_table[input_mapping]
  -> AttentionBackend.prepare_metadata(batch)

Attention layer
  -> compute current K/V
  -> kv_cache.store_kv(k, v, out_loc, layer_id)
  -> paged kernel reads historical K/V through page table

request finished/aborted
  -> CacheManager.cache_req(...)
  -> prefix 可保留部分
  -> 其余 physical pages 回到 free_slots
  -> TableManager.free(table_idx)
```

必须区分两种资源：`table_idx` 是请求元数据表的一行；physical page 是保存多层 K/V 的物理块。
释放 table row 不自动等于释放 KV pages，两者生命周期需要 Scheduler 协调。

Mini-SGLang 的全局 page table 在内部按 token position 保存展开后的物理 token index。page size 大于
1 时，FA backend 再按步长取 page 开头并除以 page size，得到 kernel 使用的 block ID。本章为了
教学清晰，MySGLang 的 block table 直接保存 physical page ID。

## 4. MySGLang 的三个地址

设 `page_size=4`，请求的 block table 为：

```text
logical block:   0  1  2
physical page:  [5, 1, 7]
```

逻辑 token 位置 6 的寻址过程：

```text
logical_block = 6 // 4 = 1
offset        = 6 % 4  = 2
physical_page = block_table[1] = 1
physical slot = pool[physical_page=1, offset=2]
```

因此逻辑序列仍然连续，但物理页可以散落在 pool 中：

```text
request tokens: 0 1 2 3 | 4 5 6 7 | 8 ...
physical pages:    5    |    1    |   7
```

物理存储位于 [paged.py](/home/sheep/mysglang/src/mysglang/cache/paged.py)：

```text
keys/values: [layers, num_pages, page_size, kv_heads, head_dim]
```

K/V 仍按所有层共享同一份 block table：physical page 5 表示每一层都使用各自 layer tensor 中的
page 5，而不是每层重新分配不同 page ID。

## 5. Reservation 与 Allocation 为什么分开

如果只在 Decode 跨页时尝试拿新页，可能出现：所有运行请求都需要一页才能继续，但 pool 已空；
请求无法生成到结束，也就没有请求能释放页，形成无法恢复的中途 OOM。

本章使用两阶段管理：

```text
admission:
  reserve_pages = ceil((prompt_len + max_new_tokens) / page_size)
  总 reservation 不允许超过 num_pages

each forward:
  allocated_pages = ceil(current_cache_length / page_size)
  仅把当前真正需要的 physical page ID 写入 block table
```

Reservation 是容量承诺，不绑定具体 page ID；Allocation 才从 free physical set 取页。因为所有请求
已分配页数都不超过各自 reservation，后续 `ensure_capacity()` 理论上一定成功。

代价是准入比较保守：请求声明 `max_new_tokens=100`，即使遇到 EOS 只生成 5 个 token，也会在运行
期间预留最坏 100 token 的容量。生产系统可以引入 overcommit/preemption，但本章优先保证不会在
模型 forward 一半时才崩溃。

## 6. PageAllocator 的不变量

`PageAllocator` 同时维护：

```text
free physical page IDs
request_id -> physical page table
request_id -> max token reservation
request_id -> reserved page count
```

每次操作后都必须满足：

1. 一个 physical page 最多属于一个请求；
2. allocated pages 与 free pages 不相交；
3. `allocated ∪ free == [0, num_pages)`；
4. 每请求 allocated pages 不超过 reserved pages；
5. 所有 reserved page count 之和不超过 pool；
6. page table、max token 和 reservation 的 request ID 集合相同。

`reserve_request()` 先检查容量再修改任何容器，因此失败是原子的：返回 `False` 后统计和所有权完全
不变。随机测试执行 500 次 reserve/grow/release，并在每一步运行 integrity checker。

## 7. Scheduler 如何使用 page pool

Paged Scheduler 的请求状态仍是：

```text
WAITING -> PREFILL -> DECODING -> FINISHED/ABORTED
```

但资源变化为：

```text
add(request)
  -> 只进入 waiting，不占 page
  -> 单请求最坏容量大于整个 pool：立即返回清晰错误

admit oldest waiting request
  -> 检查 max_running_requests
  -> reserve_request(max_total_tokens)
  -> reservation 不足：保持 WAITING，不做部分修改

before Prefill/Decode forward
  -> ensure_capacity(current_end)
  -> 跨 page boundary 时追加 physical page ID

finished/abort
  -> release_request
  -> 已绑定 pages 回到 free set
  -> reservation 同时取消
```

当 pool reservation 已满，Scheduler 仍会继续 Decode 已准入请求；请求结束释放容量后，再准入最老
waiting 请求。这是 backpressure，而不是让 CUDA allocator 抛出 OOM。

这与 Mini-SGLang 的基本选择一致：Mini-SGLang 的 `PrefillAdder` 用新请求的估计总长度，加上所有
running requests 的 `inflight_tokens`，检查是否超过 `CacheManager.available_size`；实际页面仍由
`allocate_paged()` 随 Prefill/Decode 进度追加。页面不足时，`CacheManager._allocate()` 可以淘汰未锁定
的 Radix prefix cache，但不会抢占 active Decode 请求。active-request preemption 暂不纳入当前主线，
等既定章节完成后再决定是否借鉴 nano-vLLM 单独实现。

## 8. Reference Attention 与未来高性能后端

当前 `PagedKVCache.append()` 会：

1. 按 block table 把本轮 K/V 写入离散 physical pages；
2. 根据 page table 把每个请求的历史 K/V gather 成临时 padded tensor；
3. 构造 causal/padding mask；
4. 交给 PyTorch `scaled_dot_product_attention`。

这条路径是正确性 oracle，不是最终高性能路径。它仍会复制历史 K/V，甚至 Python 循环写 token。
后续 FlashAttention 2 backend 应直接读取：

```text
physical K/V pool + block tables + sequence lengths
```

Scheduler 和 allocator 不应因为替换 kernel 而改变，这正是把 page ownership 与 attention 计算分开的
原因。

## 9. 内部碎片与外部碎片

每个请求最后一页通常没有填满，最多浪费 `page_size - 1` 个 token slot，这是内部碎片。例如
page size 4、长度 5，需要 2 页，浪费 3 个位置。

离散物理页解决了连续大块分配造成的外部碎片：即使空闲页编号为 `{1, 7, 9}`，请求仍可把它们
组成三页逻辑连续序列，不要求找到长度为三页的连续显存区间。

page size 的权衡：

- 小 page：内部碎片少，但 block table 更长、分配和 kernel metadata 更多；
- 大 page：表更短、kernel 更容易高效，但短请求或尾页浪费更多。

本地教学默认 `page_size=4` 便于观察边界；FA2 实际支持的合适 page size 要到 backend 章节根据
接口和 GPU benchmark 决定。

## 10. 正确性与资源 oracle

自动测试覆盖四层：

1. allocator：reservation 失败原子性、500 次随机操作不变量；
2. cache：跨页 Chunked Prefill、变长 Decode、释放复用后无 stale KV；
3. scheduler：token 对齐单请求 cached generation，容量不足时等待，最终所有页归还；
4. serving：Paged Scheduler 复用第 5 章同一个 HTTP/SSE 协议。

模型级 oracle 仍是 Hugging Face Qwen3 对齐链：Paged 与完整前缀/连续 cache 对齐，而连续 cache 已
直接与 Hugging Face logits 和 greedy tokens 对齐。

## 11. 借鉴、教学简化与有意改进

### 借鉴 Mini-SGLang

- K/V 物理 pool 与请求 page table 分离；
- Scheduler 在 forward 前分配页并准备映射；
- 页内位置由 logical position 的除法和取模得到；
- Prefill admission 同时考虑 prompt 和输出容量；
- 完成/abort 后由资源所有者回收 pages。

### 教学简化

- 显式指定 `num_pages`，不根据实时 GPU free memory 自动计算；
- block table 使用 Python list，未放入 GPU metadata tensor；
- 没有 prefix cache、eviction、lock handle 或 lazy-free region；
- reference backend gather 历史 K/V，不调用 paged FA2 kernel；
- 单进程，无 TP 下的 local KV heads 计算。

### 有意改进

- block table 直接保存 page ID，不混用展开后的 token slot index；
- reservation 与 physical allocation 分别统计，OOM 行为更容易推理；
- impossible request 在进入 scheduler 前明确拒绝；
- allocator 与 cache 都有完整 integrity checker；
- 随机状态机测试和 stale-page reuse 测试成为固定回归项。

## 12. 自动测试

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

只运行本章 allocator/cache 测试：

```bash
PYTHONPATH=src python -m unittest tests.test_paged_cache -v
```

只运行 Paged Scheduler 测试：

```bash
PYTHONPATH=src python -m unittest tests.test_paged_scheduler -v
```

## 13. 实验与验收

```bash
PYTHONPATH=src python examples/07_paged_kv_cache.py
```

关键输出应为：

```text
outputs match isolated cached generation: True
fixed-slot token capacity: 512
paged-pool token capacity: 64
fixed-slot KV bytes: 262144
paged-pool KV bytes: 32768
peak allocated/reserved pages: 11/16
final free pages: 16/16
```

这里不能误解为 paging 自动让相同最坏容量缩小 8 倍。Fixed-slot 数字保证四个请求都可各自达到
128 token；paged pool 的总容量只有 64 token，是根据本实验四个请求声明的最大总长度配置的。
Paging 的收益是容量变成共享预算，而不是每请求硬切一块无法借用的 128-token 区域。

逐 step trace 中 `reserved` 是最坏容量承诺，`allocated` 是当前实际绑定页数，`free` 是尚未绑定
physical ID 的页数。结束时三项必须分别为 `0/0/16`。

Paged HTTP 服务：

```bash
PYTHONPATH=src python examples/07_http_server.py
```

## 14. 亲手练习：页边界分配

请在 `PageAllocatorTest` 中增加
`test_new_page_is_allocated_only_after_crossing_boundary`：

1. 创建 `num_pages=3, page_size=4` 的 allocator；
2. 为请求预留 `max_tokens=8`；
3. 依次调用 `ensure_capacity(request_id, 1)` 到 `ensure_capacity(request_id, 4)`，始终断言 page table
   长度为 1；
4. 调用 `ensure_capacity(request_id, 5)`，断言 page table 长度变成 2；
5. release 后断言三页全部 free，并运行 `check_integrity()`。

同时回答：如果把 `page_size` 从 4 增大到 16，block table 长度和内部碎片通常分别如何变化？完成
实验、练习并理解 reservation 与 allocation 的区别后停下，不开始 RadixAttention。
