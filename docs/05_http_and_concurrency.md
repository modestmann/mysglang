# 第 5 章：HTTP、SSE 与并发请求

## 1. 本章目标与边界

本章把前四章已经存在的 tokenizer 边界、请求状态机、dense Qwen3 tiny model 和逐请求
KV Cache 接到 HTTP 上。完成后，多个客户端可以同时提交请求，服务端能返回普通 JSON 或
SSE 增量事件，并在生成器被关闭时 abort 请求、释放其引用。

本章的“并发”只表示多个请求可以同时处于 active 状态：每个请求仍有独立 cache，模型
forward 由一个 `asyncio.Lock` 串行执行，每次输入的 batch size 仍是 1。它不是 continuous
batching。把多个请求真正拼成一个动态 batch 是第 6 章的任务。

本章也不加载真实 checkpoint。随机 tiny model 没有配套 tokenizer，因此先用可逆的 UTF-8
byte tokenizer 验证服务协议。真实 Qwen3 的 Hugging Face tokenizer、chat template 和权重会
一起接入，不能把这里的 byte token 数当成真实模型 token 数。

## 2. Mini-SGLang 源码地图

| 文件 | 核心对象 | 责任 |
|---|---|---|
| `python/minisgl/server/launch.py` | `launch_server` | 建立进程和队列，最后启动 API server |
| `python/minisgl/server/api_server.py` | FastAPI endpoints | 接收 `/generate`、`/v1/chat/completions`，返回 SSE/JSON |
| 同上 | `FrontendManager` | 分配 uid，用 `ack_map/event_map` 把异步回复路由给对应客户端 |
| `python/minisgl/message/tokenizer.py` | `TokenizeMsg/DetokenizeMsg/AbortMsg` | frontend 与 tokenizer 进程之间的协议 |
| `python/minisgl/tokenizer/server.py` | `tokenize_worker` | tokenize 输入、增量 detokenize 输出、转发 abort |
| `python/minisgl/message/backend.py` | `UserMsg/AbortBackendMsg` | tokenizer 与 scheduler 之间的协议 |
| `python/minisgl/scheduler/scheduler.py` | `_process_one_msg/_process_last_data` | 接收、生成 token、结束或释放请求 |
| `python/minisgl/message/frontend.py` | `UserReply` | 把增量字符串和 finished 标志送回 frontend |

Mini-SGLang 的正常请求调用链是：

```text
HTTP endpoint
  -> FrontendManager.new_user(): 创建 uid、ack list、asyncio.Event
  -> send TokenizeMsg(text, sampling_params)
  -> tokenize_worker: HF tokenizer -> UserMsg(input_ids)
  -> Scheduler: admission -> prefill/decode -> next_token
  -> send DetokenizeMsg(uid, next_token, finished)
  -> tokenize_worker: incremental detokenize -> UserReply
  -> FrontendManager.listen(): 按 uid 写入 ack_map，并 set 对应 Event
  -> wait_for_ack(): 唤醒正确的 HTTP coroutine
  -> SSE chunk，最后 data: [DONE]
```

`ack_map` 保存暂未被 HTTP coroutine 取走的回复，`event_map` 负责通知，而 uid 负责在许多并发
请求之间做关联。不能只使用一个全局 Event，否则任何回复都可能唤醒错误的客户端。

断连链路是：

```text
request.is_disconnected()
  -> stream_with_cancellation raises CancelledError
  -> FrontendManager.abort_user(uid)
  -> AbortMsg
  -> tokenize_worker converts it to AbortBackendMsg
  -> Scheduler removes request and frees table/KV resources
```

这说明 abort 不是“浏览器不再接收文本”这么简单；取消消息必须到达资源所有者 Scheduler。

## 3. MySGLang 的最小分层

本章保留同样的边界，但所有对象先放在一个进程中，便于单步调试：

```text
FastAPI / Pydantic
  -> GenerationService.start(prompt)
       -> ByteTokenizer.encode
       -> Request.from_token_ids
       -> one ContiguousKVCache per request
  -> GenerationSession async iterator
       -> Request: WAITING -> PREFILL -> DECODING
       -> locked model forward
       -> Request.record_token
       -> incremental UTF-8 decode
       -> GenerationChunk
  -> JSON collector or SSE encoder
```

各层的输入输出很刻意：HTTP 层只认识文本和 JSON；`Request` 只认识 token ID 与生命周期；模型
只认识 tensor 和 KV Cache。以后换成 Hugging Face tokenizer，不需要修改 model；以后引入
Scheduler，也不需要修改 HTTP schema。

## 4. 为什么需要增量 UTF-8 解码

`ByteTokenizer` 把 UTF-8 的每个 byte 当作 token，例如一个中文字符通常会变成三个 token。
如果每收到一个 byte 就单独执行 `bytes([id]).decode()`，一个多字节字符会被错误拆成多个替换符。
`IncrementalUTF8Decoder` 会暂存不完整的字节序列：

```text
第 1 个 byte -> ""
第 2 个 byte -> ""
第 3 个 byte -> "你"
```

这与真实 subword tokenizer 的 incremental detokenization 动机相同：token 边界不保证就是最终
展示字符串的安全边界。这里的 byte tokenizer 只是无需 checkpoint 的协议测试替身，不是
Qwen tokenizer 的简化算法。

## 5. 普通响应与 SSE

非流式路径 `_collect()` 消费同一个 `GenerationSession`，把所有 `GenerationChunk` 合并后返回：

```json
{
  "id": "req-0",
  "text": "...",
  "token_ids": [1, 2, 3],
  "finish_reason": "length",
  "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
}
```

流式路径 `_sse()` 不保存全部输出，每产生一个 chunk 就编码成：

```text
data: {"id":"req-0", ...}

data: {"id":"req-0", ...}

data: [DONE]

```

SSE 的每个事件以空行结束，所以是两个换行符。普通响应和流式响应共用同一个生成器，这是
“两种传输方式输出一致”测试能够成立的关键；不能维护两套生成循环。

Mini-SGLang 当前非流式 chat response 的 usage 是三个 `0` 占位值。本章让
`GenerationChunk` 携带 prompt/completion 计数，因此返回实际 usage。这是有意改进。不过在
byte tokenizer 阶段，prompt token 数实际是 UTF-8 byte 数。

## 6. 并发不等于 batching

四个 HTTP coroutine 可以同时存在：

```text
request A: [prefill]       [decode]          [decode]
request B:          [prefill]       [decode]          ...
                   one model lock
```

每次 forward 结束后释放 lock，并通过 `await asyncio.sleep(0)` 给其他请求运行机会。因此服务能
维护多个独立生命周期，慢客户端也不需要阻止另一个客户端创建请求。但是 GPU 每次仍只执行
一个请求，`max_active_requests == 4` 不代表模型 batch size 是 4，也不保证吞吐提高。

下一章会把它改成：

```text
scheduler step -> 选择 A/B/C -> 拼 input tensor/metadata -> one batched forward
```

在那之前使用 lock 是明确的正确性边界，避免多个 coroutine 同时修改同一个模型执行上下文；
它不是最终性能设计。

## 7. 中止与清理

`_run_session()` 使用 `try/finally` 维护三个不变量：

1. 一个 request ID 最多在 active map 中出现一次；
2. 正常完成计入 `finished_requests`；提前关闭生成器则转成 `ABORTED`；
3. 无论正常、异常还是取消，最终都从 active map 移除。

HTTP SSE 层同时检查 `request.is_disconnected()` 并捕获 `CancelledError`。服务层的 finally 是最后
一道保险，因为资源清理不应只依赖某一种 Web server 的断连通知行为。

当前 cache 是 Python 对象；移除 session 引用后 tensor 可由引用计数回收。第 7 章使用共享物理
page pool 后，finally/abort 必须显式归还页，届时会用 free-page 不变量测试，而不能依赖 GC。

## 8. 借鉴、教学简化与有意改进

### 借鉴 Mini-SGLang

- HTTP 层、tokenizer、请求状态和模型执行之间有明确消息边界；
- 用 request ID 把异步增量输出路由回原客户端；
- 流式输出采用 SSE，并用 `[DONE]` 标记正常结束；
- 客户端断连会转成后端 abort，而不只是停止发送文本。

### 教学简化

- 单进程、单 event loop，没有 ZMQ 和 tokenizer/scheduler 子进程；
- 每请求一个连续 KV Cache，没有共享 page pool；
- 只实现 greedy decoding，不接受尚未实现的 temperature/top-k/top-p；
- chat message 使用清晰的教学字符串拼接，尚未使用 Qwen chat template；
- byte tokenizer 配合随机权重，只验证机制，输出没有语言含义。

### 有意改进

- `GenerationService` 通过构造参数注入 app，不使用模块级全局状态；
- 流式和非流式走同一个生成器，自动测试要求 token ID 与文本一致；
- usage 使用真实计数，而不是占位的零；
- session、状态机和 cache 一一对应，重复消费 session 会明确报错；
- `try/finally` 统一处理完成、异常和主动关闭，测试检查 active request 不泄漏。

## 9. 自动测试与单项运行

运行全部测试：

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

只运行本章：

```bash
PYTHONPATH=src python -m unittest tests.test_serving -v
```

只运行并发测试：

```bash
PYTHONPATH=src python -m unittest \
  tests.test_serving.GenerationServiceTest.test_concurrent_sessions_have_independent_caches -v
```

测试分别验证中文 byte round-trip、增量 UTF-8、独立 cache、并发 active requests、取消清理、
SSE/JSON 一致、chat usage 和无效 prompt 的 400 错误。

## 10. HTTP 实验

终端 A 启动服务：

```bash
source ~/nano-vllm/.venv/bin/activate
cd /home/sheep/mysglang
PYTHONPATH=src python examples/05_http_server.py
```

终端 B 请求非流式结果：

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"你好","max_tokens":4,"stream":false}'
```

观察 SSE（`-N` 禁用 curl 输出缓冲）：

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"hello","max_tokens":4,"stream":true}'
```

运行不占端口的四客户端并发实验：

```bash
PYTHONPATH=src python examples/05_concurrent_http.py
```

验收重点不是随机文本，而是四个 response 都是 200、ID 不同、每个 completion token 数为 8、
最终 `max active requests: 4`。耗时随 CPU/GPU 和热身变化，不作为本章正确性判据。

## 11. 亲手练习：准入失败不能泄漏 active request

请在 `HTTPServingTest` 中增加
`test_context_overflow_returns_400_without_leaking_request`：

1. 测试模型的 `max_position_embeddings` 是 128；
2. 向 `/generate` 发送 127 个 ASCII 字符，并请求 2 个新 token；
3. 断言 status code 是 400，错误文本包含 `max_position_embeddings`；
4. 断言 `self.service.stats.active_requests == 0`。

这个测试证明 context capacity 在请求进入 active map 和执行模型之前检查。完成测试、亲自观察一次
SSE 输出并理解“并发连接不等于 continuous batching”后停下，不开始第 6 章。
