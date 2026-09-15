# 第 2 章：一个请求的生命周期

## 1. 本章边界

本章只实现控制面的最小协议：tokenized request、sampling params、状态机和增量输出事件。
它不实现 tokenizer、HTTP、Scheduler、KV Cache 或新的模型计算。这样可以先让请求生命周期成为可测试的不变量，后续组件只能调用这套状态转换，而不能各自猜测请求处于什么阶段。

验收状态图：

```text
WAITING -> PREFILL -> DECODING -> FINISHED
    |          |           |
    +----------+-----------+----------> ABORTED
```

`FINISHED` 和 `ABORTED` 是终态；终态不能再接收 token，也不能重新进入队列。

## 2. Mini-SGLang 源码地图

| 文件 | 核心对象 | 作用 |
|---|---|---|
| `python/minisgl/message/tokenizer.py` | `TokenizeMsg`, `DetokenizeMsg`, `AbortMsg` | 前端与 tokenizer 边界 |
| `python/minisgl/message/backend.py` | `UserMsg`, `AbortBackendMsg` | tokenized 请求与 scheduler 边界 |
| `python/minisgl/core.py` | `SamplingParams`, `Req`, `Batch` | GPU 执行所需的请求和 batch 状态 |
| `python/minisgl/scheduler/utils.py` | `PendingReq` | waiting 队列中的轻量请求 |
| `python/minisgl/scheduler/prefill.py` | `PrefillManager` | waiting、prefix match、prefill admission |
| `python/minisgl/scheduler/decode.py` | `DecodeManager` | running decode 请求集合 |
| `python/minisgl/scheduler/scheduler.py` | `_process_one_msg`, `_process_last_data` | 请求迁移、完成、abort 和资源回收 |
| `python/minisgl/tokenizer/detokenize.py` | `DecodeStatus` | 每个 uid 的增量文本状态 |

Mini-SGLang 的正向调用链是：

```text
TokenizeMsg(text)
  -> TokenizeManager.tokenize
  -> UserMsg(input_ids: CPU int32[seq])
  -> Scheduler._process_one_msg
  -> PrefillManager.pending_list
  -> PrefillManager.schedule_next_batch
  -> Req + Batch(phase="prefill")
  -> Engine.forward_batch
  -> DecodeManager.running_reqs
  -> Batch(phase="decode") x N
  -> DetokenizeMsg(next_token, finished)
  -> UserReply(incremental_output, finished)
```

Abort 的反向控制链是：

```text
Frontend detects disconnect
  -> AbortMsg(uid)
  -> AbortBackendMsg(uid)
  -> PrefillManager.abort_req or DecodeManager.abort_req
  -> Scheduler._free_req_resources
```

## 3. Mini-SGLang 如何隐式表达状态

Mini-SGLang 没有 `RequestState` enum。状态由“对象在哪个容器中”隐式表示：

- `PendingReq` 位于 `PrefillManager.pending_list`：waiting；
- `Req` 被组成 `Batch(phase="prefill")`：prefill；
- `Req` 位于 `DecodeManager.running_reqs`：decoding；
- `DetokenizeMsg.finished=True` 且资源已释放：finished。

`Req` 的长度还有一个重要不变量：

```text
[0, cached_len)           已经计算出 KV
[cached_len, device_len)  token 已存在，但 KV 尚未计算
[device_len, max_device_len)  尚未生成
```

`Engine.forward_batch` 调用 `Req.complete_one()` 后，会先令
`cached_len = device_len`，再令 `device_len += 1`。新增的一格用于刚采样出的 next token；它会在下一轮 decode 中产生 KV。

## 4. MySGLang 的最小实现

本章在 `src/mysglang/core/request.py` 中定义：

- `SamplingParams`：只声明当前能够执行的 `max_new_tokens`、EOS 和 ignore-EOS；
- `RequestState`：显式状态；
- `Request`：框架拥有的 tokenized 请求；
- `IncrementalOutput`：带顺序、终止标记和终止原因的输出事件；
- `InvalidStateTransition`：非法迁移立即失败。

最小正常路径：

```python
request = Request.from_token_ids(
    "req-1",
    [10, 20],
    SamplingParams(max_new_tokens=2),
)
request.start_prefill()
request.start_decode()
request.record_token(30)
last = request.record_token(31)
assert last.finished
assert last.finish_reason is FinishReason.LENGTH
```

这里还没有把状态机接到 `greedy_generate`。那会把第 2 章控制协议和第 3 章模型执行混在同一次改动中。

## 5. 借鉴、简化与改进

### 借鉴

- request ID、prompt token IDs 和 sampling params 一起跨组件传递；
- 输出按 token 增量产生；
- abort 只携带 request ID；
- waiting 与 running 请求由不同管理器拥有。

### 教学简化

- 单进程纯 Python 对象，不做 ZMQ 序列化；
- 不实现 HTTP 文本、tokenizer 和 detokenizer；
- 不携带 page table、cache handle 等第 4/7 章状态；
- 只定义已有明确行为的 greedy sampling 参数。

### 有意改进

- 用 enum 和迁移表替代“请求在哪个 list/set”这种隐式状态；
- 输出携带 `output_index`，以后可检测重复或乱序事件；
- 输出携带 `finish_reason`，不把 EOS、长度上限和 abort 都压成一个 bool；
- prompt 转成 tuple，避免调用者在请求进入队列后从外部修改它；
- terminal state 拒绝再次执行，防止完成/中止竞态导致二次释放。

## 6. 自动测试与实验

运行全部测试：

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行生命周期实验：

```bash
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python examples/02_request_lifecycle.py
```

实验逐行输出 JSON；检查同一请求的 `output_index` 从 0 连续递增，且只有最后一条事件带 `finish_reason`。本章没有 GPU 性能变量，因此不制造无意义的吞吐 benchmark。

## 7. 亲手练习

请在 `tests/test_request_lifecycle.py` 中亲手增加：

```text
test_abort_from_prefill_emits_terminal_event
```

测试步骤：创建请求、`start_prefill()`、`abort()`。验收断言：

1. 状态为 `ABORTED`；
2. 事件 `token_id is None`；
3. `output_index == 0`；
4. `finish_reason is FinishReason.ABORTED`；
5. abort 后调用 `start_decode()` 抛出 `InvalidStateTransition`。

完成练习并运行全部测试后停在本章，不开始实现 tokenizer 或 Scheduler。
