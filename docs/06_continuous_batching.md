# 第 6 章：Continuous Batching 与 Scheduler

## 1. 本章解决什么问题

第 5 章允许多个 HTTP 请求同时 active，但每个 coroutine 都独立执行：

```text
A forward(batch=1) -> B forward(batch=1) -> A forward(batch=1) -> ...
```

本章加入唯一的 `ContinuousBatchScheduler`。所有请求先进入 Scheduler，由它在每个 step 选择
Prefill 或 Decode，并让所有可运行的 Decode 请求执行一次真实的 batched forward：

```text
Scheduler step -> input_ids [B, 1] -> one model forward -> B 个 next tokens
```

“Continuous”表示 batch 的成员可以逐轮变化：新请求完成 Prefill 后加入，已完成或 abort 的请求
离开，不需要等待一批固定请求全部结束。

本章不实现 paged KV Cache、prefix cache、FlashAttention 或多进程。为了在不提前实现第 7 章的
条件下支持动态 batch，本章使用固定容量的 slot cache。

## 2. Mini-SGLang 源码地图与调用链

| 文件 | 核心对象 | 作用 |
|---|---|---|
| `python/minisgl/scheduler/scheduler.py` | `Scheduler.normal_loop/overlap_loop` | 收消息、选 batch、forward、处理上一轮结果 |
| 同上 | `_schedule_next_batch` | 在 Prefill 与 Decode 之间选择 |
| 同上 | `_prepare_batch` | 准备 positions、token mapping、写入位置和 attention metadata |
| 同上 | `_forward` | 调用 Engine，并把仍可运行的请求交给 DecodeManager |
| 同上 | `_process_last_data` | 记录 next token、判断完成、释放资源、发送回复 |
| `python/minisgl/scheduler/prefill.py` | `PrefillManager` | FIFO pending list、token budget、chunked prefill、准入 |
| 同上 | `PrefillAdder` | 估算并锁定 KV 资源，把 prompt 切成 budget 内的 chunk |
| `python/minisgl/scheduler/decode.py` | `DecodeManager` | 保存 running requests，按 uid 排序组成 Decode batch |
| `python/minisgl/core.py` | `Req/Batch` | `cached_len/device_len/extend_len` 与 batch phase |

Mini-SGLang 的非 overlap 主循环可以压缩为：

```text
receive_msg()
  -> UserMsg 放入 PrefillManager.pending_list
  -> AbortBackendMsg 从 Prefill/Decode manager 移除并释放资源

_schedule_next_batch()
  -> PrefillManager.schedule_next_batch(prefill_budget)
  -> 如果没有 Prefill，再 DecodeManager.schedule_next_batch()

_prepare_batch()
  -> CacheManager.allocate_paged(reqs)
  -> positions / input_mapping / write_mapping
  -> AttentionBackend.prepare_metadata(batch)

Engine.forward_batch(batch)
  -> sample next_tokens

_process_last_data()
  -> req.append_host(next_token)
  -> EOS/length 判断
  -> finished: free table/cache
  -> unfinished: 留在 DecodeManager
  -> DetokenizeMsg
```

`overlap_loop` 在另一条 CUDA stream 上启动当前 batch，同时处理上一轮结果，以隐藏部分 CPU
调度开销。本章先实现同步 `step()`，因为当前目标是验证选择策略、batch membership 和 logits，
不是提前引入 stream overlap。

## 3. Mini-SGLang 的请求长度

Mini-SGLang 的 `Req` 用三个长度描述数据状态：

```text
cached_len  : 已存在 KV 的 token 数
device_len  : 已经放入设备 token table 的长度
extend_len  : device_len - cached_len，本轮需要 forward 的 token 数
```

Prefill 可能受 token budget 限制而分块。例如 prompt 长度 10、budget 4：

```text
step 1: cached=0, device=4, extend=4
step 2: cached=4, device=8, extend=4
step 3: cached=8, device=10, extend=2
```

Decode 时每个请求的 `extend_len` 通常是 1。多个请求虽然历史长度不同，但 paged attention 根据
每个请求的 page table 和 sequence length 找到各自 KV。

MySGLang 对应字段是 `_Entry.prefill_offset`、slot length 和
`Request.num_output_tokens`。字段更少，因为本章还没有 token table 和 prefix match。

## 4. 为什么第 4 章的 Cache 不能直接动态 batching

第 4 章的 `ContiguousKVCache` 形状是：

```text
[layers, batch, kv_heads, max_length, head_dim]
```

其中整个 batch 共用一个有效长度。如果 A 的历史长度为 4、B 为 7，就无法用单个 `length=...`
准确表示两者。更麻烦的是 batch 行号会随请求完成而改变，不能把“当前 batch 第 0 行”永久当成
请求 A 的存储位置。

本章增加 `SlotKVCache`：

```text
storage: [layers, slots, kv_heads, max_length, head_dim]
lengths: [layers, slots]

runtime batch row 0 -> slot 3, length 4
runtime batch row 1 -> slot 0, length 7
```

`slot` 是请求活跃期间稳定的物理位置；`batch row` 只是这一轮临时排列。Scheduler 通过
`cache_slots=(3, 0)` 明确映射二者。

## 5. 不同长度如何做一次 Attention

假设两个 Decode 请求的历史长度分别为 4 和 7，本轮各输入一个 token：

```text
input_ids: [2, 1]
RoPE positions: [[4], [7]]
new lengths: [5, 8]
```

slot cache 收集成临时 padded K/V：

```text
K/V:  [2, kv_heads, 8, head_dim]
mask: [2, 1, 1, 8]

A: T T T T T F F F
B: T T T T T T T T
```

没有 mask，A 会读到 slot gather 后的无效位置。RoPE position 也必须从共享的一维 `[position]`
扩展为每个请求各自的 `[batch, sequence]`。

Chunked Prefill 需要更一般的 offset causal mask。若某个 chunk 从位置 4 开始、长度为 3，则三个
query 的绝对位置是 4、5、6，只允许看到 `key_position <= query_position`。自动测试要求两段
Prefill 的 logits 对齐一次性完整 Prefill。

当前 gather 会复制所选 slot 的有效 K/V。这是正确但低效的教学实现；第 7 章的 page table 与
高性能 attention backend 会直接消费离散页，避免每轮重新拼接历史。

## 6. Scheduler 的四个集合

```text
waiting deque -> prefilling(最多一个) -> running OrderedDict -> finished
       |                 |                    |
       +---------------- abort ---------------+
```

- `waiting`：尚未获得 slot，保持 FIFO；
- `prefilling`：已经获得 slot，prompt 可能被 token budget 分成多步；
- `running`：Prefill 完成并至少生成了第一个 token，可进入 Decode batch；
- `_free_slots`：当前可分配的连续 cache slot。

每个 Prefill step 当前只处理一个请求的一个 chunk；每个 Decode step 则把所有 running 请求组成
一个 batch。Mini-SGLang 能把多个不同长度的 Prefill request flatten 到同一个 forward，本章暂不
复现这项 kernel/metadata 优化。

## 7. 调度策略与无饥饿边界

Mini-SGLang 当前 `_schedule_next_batch()` 使用：

```python
prefill_manager.schedule_next_batch(...) or decode_manager.schedule_next_batch()
```

也就是只要 Prefill 一直可运行，Decode 就可能持续推迟。这是简单且偏向吞吐/TTFT 的策略，但在
持续到达 workload 下可能损害已有请求的 TPOT。

MySGLang 本章使用有界策略：

```text
存在 running Decode：最多连续执行 max_consecutive_prefill_steps 个 Prefill
达到上限：强制执行一个 Decode step
没有 running Decode：继续 Prefill，不做无意义等待
```

默认上限为 1，所以持续有新请求时大致交替：

```text
Prefill -> Decode -> Prefill -> Decode
```

这保证 Prefill 不会无限饿死 Decode。反过来，只要有空 slot，Decode 后计数归零，最老的 waiting
请求就能执行 Prefill；没有空 slot 时必须等待已有请求结束，这属于容量限制，不是策略饥饿。

这还不是生产级 latency policy。后续 benchmark 可以根据 TTFT/TPOT deadline、prompt 长度和
GPU 利用率做 cost model；本章先让公平边界成为可测试的不变量。

## 8. HTTP 调用链发生了什么变化

第 5 章：

```text
HTTP coroutine -> 自己执行 model forward -> yield chunk
```

第 6 章：

```text
HTTP coroutine -> service.start -> Scheduler.add -> 等自己的 asyncio.Queue
                                      |
                         one scheduler worker
                                      |
                         step -> batched model forward
                                      |
                     按 request_id 把事件放回各自 Queue
```

HTTP 的 `_collect()`、`_sse()` 和 schema 没有改变。`http.py` 依赖一个很窄的 `ServingBackend`
协议，所以单请求 service 和 continuous-batch service 可以共用同一个传输层。这正是控制面边界
带来的收益。

## 9. 正确性 oracle

自动测试覆盖：

1. 两个不同历史长度的 slot 在一次 Decode forward 中分别对齐完整前缀重算；
2. Chunked Prefill logits 对齐一次性 Prefill；
3. 迟到请求加入已有请求的 Decode batch，最终 token 分别对齐 `greedy_generate_cached`；
4. Prefill/Decode 公平策略产生预期 step trace；
5. abort 立即释放 slot，最老 waiting 请求可以复用它；
6. 两个并发 HTTP 请求确实观测到 `max_decode_batch_size == 2`。

第 3 条很重要：batching 只能改变计算组织方式，不能让 A 的 token 受到 B 的 prompt 或 padding
内容影响。

## 10. 借鉴、教学简化与有意改进

### 借鉴 Mini-SGLang

- Scheduler 是唯一 batch 决策者；
- 分离 waiting Prefill 与 running Decode；
- Prefill 使用 token budget，允许 chunked prefill；
- Decode batch 的成员每轮动态变化；
- 完成或 abort 后立即释放 cache 所有权。

### 教学简化

- 同步 `step()`，没有 overlap CUDA stream；
- Prefill 每步只处理一个请求，未实现 flattened/ragged Prefill batch；
- greedy sampler 固定在 Scheduler 内；
- 固定连续 slot，没有 page table、eviction 或 prefix sharing；
- 通过 padded gather 使用 PyTorch SDPA，尚未接 FlashAttention 2。

### 有意改进

- 不采用无限 Prefill 优先，而是显式限制连续 Prefill step；
- 每一步记录 phase、request IDs、batch size 和 input token 数，便于复现实验；
- variable-length batch 和 chunked prefill 都有完整前缀 oracle；
- HTTP 只依赖 `ServingBackend` 协议，不绑定具体 scheduler 实现；
- slot reset、重复 ID、context overflow 和非法状态都有显式检查。

## 11. 自动测试

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

只运行调度器测试：

```bash
PYTHONPATH=src python -m unittest tests.test_scheduler -v
```

只运行“迟到请求加入 batch”的测试：

```bash
PYTHONPATH=src python -m unittest \
  tests.test_scheduler.ContinuousBatchSchedulerTest.test_late_request_joins_variable_length_decode_batch -v
```

## 12. 实验与验收

```bash
PYTHONPATH=src python examples/06_continuous_batching.py
```

当前 tiny CPU 实验的确定性结果是：

```text
outputs match isolated cached generation: True
isolated model forward calls: 32
scheduled model forward calls: 14
maximum decode batch size: 4
```

trace 会展示 batch 从 1 增长到 4，再随请求完成下降到 1。`32 -> 14` 表示多请求共享了 model
forward 次数，不表示 FLOPs 同比例下降，因为 batched forward 自身包含多行计算。

墙钟时间不是本章硬性判据。tiny CPU shape 上，padded gather 和调度开销可能大于 batching 收益；
GPU、大模型和足够并发下才更可能体现吞吐优势。

运行混合长短 prompt/output 的在线指标实验：

```bash
PYTHONPATH=src python examples/06_latency_experiment.py
```

它报告 output throughput，以及跨请求的 TTFT、TPOT、E2E p50/p95/p99。这里的定义是：TTFT
从提交到第一个 token；TPOT 是单请求首末输出时间差除以后续 token 间隔数；E2E 从提交到完成。
样本很小且使用随机 tiny CPU 模型，所以数值用于验证统计管线，不能和真实 serving benchmark
横向比较。

要通过 HTTP 观察相同行为，可启动：

```bash
PYTHONPATH=src python examples/06_http_server.py
```

API 与第 5 章相同，区别只在服务内部已经换成 continuous-batch scheduler。

## 13. 亲手练习：修改公平上限并验证 trace

请在 `tests/test_scheduler.py` 新增
`test_at_most_two_prefills_between_decode_steps`：

1. 配置 `prefill_token_budget=2`、`max_consecutive_prefill_steps=2`；
2. 先让请求 A 完成 Prefill 并进入 Decode；
3. 再加入一个 prompt 长度至少为 6 的请求 B，使它需要三个 Prefill chunks；
4. 连续调用 `step()` 并记录 phase；
5. 断言 A 仍在运行时，任意两个 Decode step 之间最多出现两个 Prefill step；
6. 最终断言 A、B 都正常完成且 `active_requests == 0`。

请同时回答：把上限从 1 改为 2，通常会怎样影响新请求 TTFT 和已有请求 TPOT？完成实验和练习后
停下，不开始 paged KV Cache。
