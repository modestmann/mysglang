# MySGLang 当前设计

本文描述当前保留的主架构，不再按历史课程逐章解释。设计目标是让请求状态、调度决策和 KV 页面所有权都能被直接观察和验证，同时为后续 GPU backend 留出窄接口。

## 1. 组件与调用链

系统可以分成控制面和数据面：

- 控制面处理 request ID、状态、队列、准入、abort 和输出路由；
- 数据面处理 token tensor、positions、page table、K/V、attention 和 logits。

当前请求路径为：

```text
HTTP / caller
  -> GenerationService.start
       -> tokenizer.encode
       -> Request(WAITING)
       -> Scheduler.validate
       -> 返回 GenerationSession
       -> 尚未 enqueue，也没有 Queue/KV reservation

首次消费 GenerationSession
  -> Scheduler.add(request)
  -> 注册 _queues[request_id]
  -> 启动或唤醒唯一 scheduler worker

scheduler worker
  -> Scheduler.step()
       -> 所有 Decode token 与 Prefill chunks 组成混合 batch；无 Prefill 时执行纯 Decode
       -> 准备 KV 页面和模型输入
       -> one model forward
       -> 产生多个 IncrementalOutput
  -> 按 event.request_id 写入对应 Queue

HTTP session
  -> 只消费自己的 Queue
  -> 增量 detokenize
  -> JSON 或 SSE
  -> 完成、异常或断连时统一清理
```

`GenerationSession` 只能消费一次。惰性 enqueue 使“创建后从未消费的 session”只持有普通 Request 对象，不进入 waiting 集合，也不占调度槽、KV reservation 或物理页。

这里只有一个全局 `Scheduler`。`request_id -> Queue` 不是另一套调度结构，而是每个请求的输出邮箱；它避免并发 session 从一个全局结果队列中抢到别人的 token。Queue 只保存尚未消费的事件，完整输出 token 仍由 Request 持有。公开服务接口已经收敛为 `GenerationService` / `GenerationSession`，调度接口为 `Scheduler` / `SchedulerConfig`。

核心所有权如下：

| 对象 | 负责什么 | 不负责什么 |
|---|---|---|
| HTTP/tokenizer | 文本、schema、SSE、断连 | batch 和 KV 页面 |
| Request | token、状态和终止原因 | 选择何时 forward |
| GenerationService | 创建 session、驱动 worker、按 ID 路由结果 | 制定调度策略 |
| GenerationSession | 请求的一次性异步消费视图 | 提前申请调度或 KV 资源 |
| Scheduler | admission、Prefill/Decode 选择、batch membership | HTTP 与文本 |
| PageAllocator | 物理页、页表、容量承诺 | attention 数学计算 |
| RadixPrefixCache | token 前缀到 cached pages 的索引和保护 | 存放 K/V tensor |
| PagedKVCache | 真正的逐层 K/V pool | 请求到达与输出传输 |
| Model | hidden states、Q/K/V 投影和 logits | 请求生命周期 |
| AttentionBackend | KV 写入、attention kernel 与输出 | 调度策略和请求生命周期 |

## 2. 实现演进与当前地位

旧实现的价值是建立 oracle，不是同时成为正式后端：

| 阶段 | 解决的问题 | 当前地位 |
|---|---|---|
| 完整前缀重算 | 定义最直接的生成语义 | correctness baseline |
| Contiguous KV Cache | 证明 Decode 只需输入新 token | 被动态 batching 路径取代 |
| Slot KV Cache | 让变长请求拥有稳定 cache slot | 被共享 page pool 取代 |
| Paged KV Cache | 请求按需使用离散物理页 | 当前内存管理基础 |
| Radix prefix cache | 已计算的完整页可在请求仍 Decode 时被复用 | 当前默认缓存策略 |

最终主线是一个 `Scheduler` 和一个 `GenerationService`。Scheduler 内部固定使用 paged + radix cache；旧的 contiguous/slot cache 与平行 scheduler 已删除。完整前缀重算只作为测试 helper 中的 correctness oracle 保留。

## 3. 请求与调度不变量

Request 的正常状态路径是：

```text
WAITING -> PREFILL -> DECODING -> FINISHED
    |          |           |
    +----------+-----------+----> ABORTED
```

- `FINISHED` 和 `ABORTED` 是终态，不能再次接收 token；
- 增量事件带连续的 `output_index`、可选 `token_id` 和明确 `finish_reason`；
- 重复 request ID、非法迁移和超出上下文容量必须在修改资源前失败；
- `GenerationService.start()` 只执行 tokenize/validate，首次消费 session 时才 enqueue；
- 完成、异常和 abort 最终都要移除 scheduler entry、解锁 prefix handle 并释放私有页。

Scheduler 维护三个主要区域：

```text
waiting deque -> prefilling insertion-ordered map -> running insertion-ordered map
```

每个 step 都让已有 Decode 请求各推进一个 token。有 Prefill 时，将 Decode tokens 放在前面，再拼接多个变长 Prefill chunks，一次 packed forward 完成，phase 为 `mixed`；仅有 Prefill 时为 `prefill`，仅有 Decode 时保留 dense 路径并标为 `decode`。`prefill_token_budget` 只限制 Prefill token，总输入上限为该 budget 加 Decode 请求数。未完成的长请求轮转到 Prefill 队尾；新完成 Prompt 的请求下轮才开始输入首个输出 token。旧的 `max_consecutive_prefill_steps` 配置已删除。没有可用容量时，waiting 请求保持等待，已准入请求继续推进。

混合执行保证 Decode 每轮有进展，但长 Prefill chunk 仍可能拉长该轮耗时；当前没有 TTFT/TPOT 延迟保证，需要通过实测调整 budget。

每次 `step()` 由唯一 worker 同步驱动，因此模型、Scheduler 和 cache metadata 不会被多个 coroutine 同时修改。这里的“串行”只指控制循环；一次模型 forward 内仍可以包含多个请求。

`SchedulerStep` 会返回给当前 worker 做输出路由，但 Scheduler 不保存无限增长的逐步记录。长期统计只保留累计计数，其中 `prefill_input_tokens` 用于衡量前缀命中实际省掉了多少 Prefill 输入。

## 4. Paged KV Cache

物理 K/V pool 的逻辑形状是：

```text
[layers, physical_pages, page_size, kv_heads, head_dim]
```

请求只保存自己的 block table。若 `page_size=4` 且 table 为 `[5, 1, 7]`，逻辑位置 6 的寻址是：

```text
logical block = 6 // 4 = 1
page offset   = 6 % 4  = 2
physical page = table[1] = 1
KV address    = pool[layer, 1, 2]
```

容量管理分成两步：

1. Admission 根据 `prompt_len + max_new_tokens` 预留最坏情况下的页数；
2. 真正执行 Prefill/Decode 时，跨越页边界才绑定具体 physical page ID。

这样中途增长不会突然遇到无法恢复的 OOM，但当前策略会因最坏情况预留而偏保守。必须始终满足：

- free pages 与 owned/cached pages 不重叠，三者覆盖整个 pool；
- 非 cached page 最多只有一个活动请求 owner；
- 只有 cached page 可以出现在多个请求页表中；
- 页表长度不能超过 `shared prefix pages + private reserved pages`；
- waiting 请求尚未拥有页，完成或 abort 后不遗留页表。

### AttentionBackend

采样位置由 Scheduler 在 forward 前确定：所有 Decode token，以及本轮完成 Prompt 的请求的末尾 token。packed 模型完成所有 Transformer 层后，先按 `logits_indices` 选择 hidden rows，再执行最终 RMSNorm 和 LM head，返回 `[采样位置数, vocab]`。中间 Prefill chunk 仍完整更新 KV；若全 batch 没有采样位置，则跳过最终 norm/head，返回 `[0, vocab]`。省略该参数仍返回完整 logits，供数值对齐使用。这减少词表投影工作，不减少 Attention 或 MLP 的 token 数。

模型只负责生成已经过 RoPE 的 Q/K/V，具体怎样写入 KV、读取历史并执行 attention 由 backend 决定：

| Backend | KV 路径 | 用途 |
|---|---|---|
| `TorchAttentionBackend` | 写页后 gather；dense batch 用 padded SDPA，packed batch 逐请求调用 SDPA | CPU 测试与 correctness oracle |
| `FlashAttentionBackend` | 纯 Decode 使用 `flash_attn_with_kvcache`；Prefill 和混合 batch 使用 slot scatter + paged `flash_attn_varlen_func` | CUDA Prefill/Decode |

模型在进入第一层前调用一次 `PagedKVCache.prepare_batch()`，生成所有层共享的 request IDs、追加范围、block table、`cu_seqlens_q/k` 和 `slot_mapping`。每层的 `prepare_append()` 只校验本层状态并取得对应的物理 K/V pool view。packed 新 K/V 通过向量化 `index_copy_` 按 slot mapping 写页，FA2 varlen kernel 再直接按 block table 读取完整历史；不会构造 padded Q/K/V。attention 成功后 backend 才调用 `commit_append()` 更新该层逻辑长度，因此 kernel 抛错时长度不会提前提交。

当前安装的 FA2 paged kernel 要求 CUDA fp16/bf16、head dimension 不超过 256，并要求 `page_size` 是 256 的倍数。Scheduler 创建 cache 后立即调用 backend 校验，因此错误配置会在 serving 开始前失败。纯 Decode 使用 dense `[batch, 1]`；Prefill 和混合 batch 使用 `[total_query_tokens]` packed 输入，不同请求由累计长度分隔。当前 metadata 和 slot mapping 仍在 Python 中逐 batch 构造，后续可通过复用 buffer 和固定 bucket 继续降低开销。

## 5. Radix prefix cache

Radix Tree 只保存索引关系：

```text
token prefix -> physical page IDs
```

K/V tensor 仍在 paged pool 中。Prompt Prefill 一完成，Scheduler 就立刻发布已经计算完的完整 prompt pages，并由仍在 Decode 的发布者继续持锁。后到请求因此无需等待发布者生成结束，就可以共享这些只读页并只计算未命中的 suffix。

请求正常结束时，还会发布 Decode 期间新增且已经完整计算的页面，然后解锁 handle、释放不足一页的 tail 和其他私有页。Abort 不发布尚未公开的私有工作，只解锁已有 handle 并回收私有页。

页面生命周期为：

```text
free
  -> request-private
  -> publish + lock
  -> cached + protected (ref_count > 0)
  -> another request match（ref_count 再增加）
  -> cached + evictable
  -> free after eviction
```

`protected` 只表示不能淘汰，不表示不能共享；发布者 Decode 时，其他请求仍可 match 并增加引用计数。匹配后必须先 lock handle，再让请求引用页面；否则页面可能在模型读取前被淘汰和复用。完成或 abort 必须 unlock。显存不足时只淘汰 `ref_count == 0` 的叶节点，因为直接删除内部节点会破坏其 children 的 token 路径。

这里的 LRU 不是经典的“全局哈希表 + 双向链表”：Radix 节点用 children 字典做前缀索引，访问时更新时间戳，真正淘汰时收集可淘汰 leaf 并使用最小堆选择最旧节点。它实现了 leaf-level LRU 语义，但淘汰时需要遍历树。

### 为什么当前实现不匹配 prompt 的最后一个 token

当前入口执行：

```python
match_prefix(prompt_token_ids[:-1])
```

之后还会向下对齐到完整页面。因此理论最大命中长度是：

```text
align_down(len(prompt) - 1, page_size)
```

这不是因为最后一个 token 的 K/V 不能缓存。原因是 KV Cache 不保存该位置已经算出的 logits，而当前普通 Prefill forward 需要至少一个新输入 token 才能产生首个生成 token 所需的 logits。保留至少一个未命中 token，可以复用同一条 forward 路径，并保证接下来写入的是请求私有尾页，从而不需要部分页 copy-on-write。

未来若要完全命中 prompt，可以选择：

- 缓存 endpoint 的最终 hidden state，再执行 LM head；
- 直接缓存 endpoint logits；
- 实现专门的一 token replay/query-only 路径。

第三种方案仍需让该 token 依次经过所有 Transformer layers，因为后一层的 Q 依赖前一层 hidden state；在 fused QKV 模型中也未必能廉价地只算 Q。若还要共享可继续写入的部分页，则必须增加 copy-on-write。它们是工程权衡，不是当前 KV 正确性的缺陷。

## 6. 与 Mini-SGLang 的关系

以下比较固定基于本地 `/home/sheep/mini-sglang` 快照 `9a91cfa`，不代表其未来版本。

| 方面 | Mini-SGLang `9a91cfa` | MySGLang 当前选择 |
|---|---|---|
| 运行架构 | 多进程、每个 TP rank 持有 Scheduler/Engine | 单进程，先使状态和所有权可观察 |
| 请求状态 | 主要由对象所在容器和长度字段隐式表达 | 显式 enum、迁移检查和终止事件 |
| 调度 | 简洁的 Prefill-before-Decode 路径，支持成熟的 flattened Prefill | Decode 每轮推进，与 budget 内的变长 Prefill 合为一次 forward |
| 页表表示 | 内部保存展开后的 physical token indices | block table 直接保存 physical page IDs |
| Admission | 基于 available size 与 inflight 估算 | reservation 与实际 page binding 分开，行为保守但易验证 |
| Radix | tensor key、快速比较 kernel、timestamp + leaf heap | Python tuple key、timestamp + leaf heap |
| 防御能力 | `reset()` 未实现，integrity checker 为空 | reset、stats、tree/allocator/scheduler 完整检查 |
| 执行性能 | 真实模型、GPU attention、CUDA Graph、TP 等完整路径 | tiny model 与 gather/pad reference attention |

双方的 Radix eviction 都不是双向链表 LRU。MySGLang 对显式状态、验证和 reference oracle 的加强服务于学习与调试；Mini-SGLang 的 kernel、metadata 和分布式路径则远比当前项目完整。

## 7. 当前限制

- 模型是随机 tiny dense Qwen3，没有真实 tokenizer、checkpoint 或 MoE；
- sampling 只有 greedy；
- 单进程同步 worker，没有 scheduler/forward overlap；
- ragged Prefill 已打通，但 metadata/slot mapping 仍由 Python 构造，尚未做运行时性能调优；
- reference attention 每轮 gather 历史 K/V；
- page 数量显式配置，没有根据实时显存自动计算；
- 没有 active-request preemption、swap、cache namespace 或跨实例 cache routing；
- 没有 CUDA Graph、Tensor Parallel 和生产容错；
- HTTP 协议尚未由正式模型客户端与数据集评测。

后续顺序和验收边界见 [roadmap.md](roadmap.md)。
