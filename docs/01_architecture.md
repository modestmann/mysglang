# 第 1 章：从 Mini-SGLang 看推理引擎架构

## 1. 这一章要回答什么

这一章暂时不优化 kernel，而是先回答五个问题：

1. HTTP 请求由谁接收？
2. 文本在哪里变成 token，token 又在哪里变回文本？
3. 谁决定下一次 GPU forward 包含哪些请求？
4. 模型、KV Cache 和 Attention backend 由谁持有？
5. 多张 GPU 之间传控制消息还是传 tensor？

如果这些边界不清楚，后面很容易把 HTTP、调度、模型和 CUDA 状态写进同一个巨型类。

## 2. Mini-SGLang 源码地图

先看这些文件：

| 组件 | Mini-SGLang 文件 | 责任 |
|---|---|---|
| 启动器 | `python/minisgl/server/launch.py` | 创建 tokenizer、detokenizer 和各 TP rank scheduler 进程 |
| HTTP 前端 | `python/minisgl/server/api_server.py` | 校验请求、生成 uid、SSE 输出、断连 abort |
| 消息定义 | `python/minisgl/message/` | 进程间传输的 tokenize/user/detokenize/abort 消息 |
| Tokenizer | `python/minisgl/tokenizer/` | 文本与 token IDs 的双向转换 |
| Scheduler | `python/minisgl/scheduler/scheduler.py` | 收消息、请求准入、选 batch、处理输出 token |
| Prefill | `python/minisgl/scheduler/prefill.py` | 等待队列、prefix match、chunked prefill、显存估算 |
| Decode | `python/minisgl/scheduler/decode.py` | 保存运行中请求并组成下一轮 decode batch |
| CacheManager | `python/minisgl/scheduler/cache.py` | 物理页申请、回收、驱逐和 page table 更新 |
| Engine | `python/minisgl/engine/engine.py` | 模型、KV pool、attention backend、sampler、CUDA Graph |
| 模型层 | `python/minisgl/models/`, `layers/` | Transformer 结构、权重切分与 TP linear |
| Attention | `python/minisgl/attention/` | 将 batch/page table 转成第三方 kernel metadata |
| 分布式 | `python/minisgl/distributed/` | TP rank 信息、all-reduce/all-gather |

阅读顺序不要从某个 CUDA kernel 开始。推荐先看：

```text
launch.py
  -> api_server.py
  -> message/
  -> scheduler.py
  -> prefill.py + decode.py
  -> engine.py
  -> attention/ + kvcache/
  -> models/ + layers/
```

## 3. 控制面和数据面

可以把整个系统分成两部分。

控制面处理“做什么”：

```text
HTTP 请求、uid、队列、请求状态、采样参数、完成/中止
```

数据面处理“怎么算”：

```text
input_ids、positions、page table、Q/K/V、logits、NCCL tensor
```

Mini-SGLang 中 Scheduler 是二者交界处。它把不规则请求整理成 `Batch` 和 attention metadata，然后 Engine 才执行 GPU forward。

这个边界非常重要：Engine 不应该理解 HTTP；API server 也不应该分配 KV page。

## 4. 一次请求的完整路径

在线模式大致如下：

```text
Client
  -> FastAPI Frontend
  -> Tokenizer
  -> Scheduler rank 0
  -> 广播给其他 TP ranks
  -> 每个 rank 的 Engine.forward_batch
  -> rank 0 得到 next token
  -> Detokenizer
  -> Frontend SSE
  -> Client
```

控制消息主要通过 ZMQ；模型 TP tensor 通过 NCCL/PyNCCL。不能把 prompt 文本通过 NCCL 发送，也不应该用 ZMQ 搬运每层的大 tensor。

## 5. 为什么每个 TP rank 都有 Scheduler

Mini-SGLang 为每张参与 TP 的 GPU 启动一个 Scheduler/Engine 进程。每个 rank 必须以相同顺序构造 batch，否则第 0 张卡可能对请求 A 做 Q projection，第 1 张卡却对请求 B 做 projection，随后 all-reduce 会得到无意义结果，甚至通信卡死。

因此“请求顺序稳定”是分布式正确性的一部分，不只是代码风格。Mini-SGLang 的 decode manager 会按 uid 排序请求；MySGLang 到 TP 章节时会把 batch plan 由 rank 0 显式广播，并加入 plan hash 断言，使不一致尽早报错。

## 6. Engine 为什么不等于 Model

`Model` 只描述数学计算；`Engine` 还持有：

- GPU device/stream；
- 模型权重；
- KV Cache pool；
- page table；
- Attention backend；
- sampler；
- CUDA Graph runner；
- TP communicator。

这解释了为什么 Hugging Face `model.generate()` 不是一个完整 serving engine。模型不知道多个用户何时到达，也不知道应该驱逐哪个请求的 prefix cache。

## 7. MySGLang 的对应边界

项目会逐步形成：

```text
mysglang/
├── frontend/       # HTTP schema、stream、abort
├── tokenizer/      # 文本边界
├── core/           # Request、Batch、状态机
├── scheduler/      # admission、prefill/decode policy
├── engine/         # model runner、CUDA stream/graph
├── cache/          # contiguous -> paged -> radix
├── attention/      # reference -> flash-attn backend
├── modeling/       # tiny dense -> Qwen3 MoE
├── distributed/    # TP plan 和 collectives
└── benchmark/      # correctness + offline + online
```

目前 `modeling/tiny.py` 是数据面的最小 correctness oracle，`generation.py` 是尚未调度化的单请求控制循环。下一步会定义 `Request` 状态机，把生成循环从模型代码中剥离出来。

## 8. 与 Mini-SGLang 的第一批有意差异

| 方面 | Mini-SGLang 当前简化 | MySGLang 计划 |
|---|---|---|
| 正确性基线 | Engine 初始化要求 CUDA | Hugging Face 做模型级 oracle；PyTorch backend 做 Attention 算子级对齐 |
| 调度策略 | Prefill batch 总是先于 decode batch | latency budget + aging，避免 decode/prefill 饥饿 |
| Radix reset | 未实现 | reset、stats 和 integrity property test 一起实现 |
| API 字段 | 部分字段声明但未执行 | 不接受未实现字段，或返回明确错误；usage 必须真实 |
| TP 一致性 | 依赖各 rank 确定性调度 | rank 0 batch plan + hash/assertion |
| 性能回归 | 示例 benchmark 为主 | 每个优化保留 reference 对齐和 JSONL 环境元数据 |

这不是说 Mini-SGLang 的选择一定错误。它为了保持代码短小，省略了部分生产防御和教学 oracle；MySGLang 的目标不同，因此会多写一些测试和显式状态。

## 9. 本章练习

不改代码，先独立回答：

1. 一个请求在 Prefill 完成后，哪些状态必须交给 Decode？
2. 为什么 KV Cache page 不能由 HTTP server 分配？
3. TP rank 的 batch 顺序不同为什么不是普通的数值误差？
4. 如果客户端中断，请求可能分别位于 waiting、prefill in-flight、decode 哪些位置？每处应由谁回收？

下一章实现状态机后，这四个答案都会变成可执行测试，而不是只停留在文字层面。
