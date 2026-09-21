# MySGLang 当前设计

本文描述当前保留的主架构，不再按历史课程逐章解释。设计目标是让请求状态、调度决策和 KV 页面所有权都能被直接观察和验证，同时为后续 GPU backend 留出窄接口。

## 1. 组件与调用链

系统可以分成控制面和数据面：

- 控制面处理 request ID、状态、队列、准入、abort 和输出路由；
- 数据面处理 token tensor、positions、page table、K/V、attention 和 logits。

单 rank 请求路径为：

```text
HTTP / caller
  -> GenerationService.start
       -> tokenizer.encode / apply_chat_template
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

TP=1 时只有一个全局 `Scheduler`。TP>1 时 rank 0 对外暴露同样的调度接口，所有 rank
各自镜像 Scheduler/Radix 元数据并持有 rank-local KV pool；rank 0 广播控制命令和每轮
BatchPlan，使所有进程以相同顺序进入模型 collective。详细边界见第 6 节。

`request_id -> Queue` 不是另一套调度结构，而是每个请求的输出邮箱；它避免并发 session
从一个全局结果队列中抢到别人的 token。Queue 只保存尚未消费的事件，完整输出 token
仍由 Request 持有。公开服务接口已经收敛为 `GenerationService` / `GenerationSession`，
调度接口为单 rank `Scheduler` 或 rank 0 使用的 `TensorParallelScheduler`。

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
| TensorParallelContext | 当前 rank、TP 大小和模型 collective group | 请求调度与进程启动 |
| TensorParallelScheduler | rank 0 命令广播、BatchPlan 校验和 token 同步 | Linear 分片数学 |

## 2. 实现演进与当前地位

旧实现的价值是建立 oracle，不是同时成为正式后端：

| 阶段 | 解决的问题 | 当前地位 |
|---|---|---|
| 完整前缀重算 | 定义最直接的生成语义 | correctness baseline |
| Contiguous KV Cache | 证明 Decode 只需输入新 token | 被动态 batching 路径取代 |
| Slot KV Cache | 让变长请求拥有稳定 cache slot | 被共享 page pool 取代 |
| Paged KV Cache | 请求按需使用离散物理页 | 当前内存管理基础 |
| Radix prefix cache | 已计算的完整页可在请求仍 Decode 时被复用 | 当前默认缓存策略 |

最终主线只有一个逻辑调度入口和一个 `GenerationService`。TP=1 直接使用 `Scheduler`；
TP>1 由 `TensorParallelScheduler` 在各 rank 重放同一组调度命令。Scheduler 内部固定使用
paged + radix cache；旧的 contiguous/slot cache 与历史平行实现已删除。完整前缀重算只
作为测试 helper 中的 correctness oracle 保留。

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

当前安装的 FA2 paged kernel 要求 CUDA fp16/bf16、head dimension 不超过 256，并要求 `page_size` 是 256 的倍数。Scheduler 创建 cache 后立即调用 backend 校验，因此错误配置会在 serving 开始前失败。纯 Decode 使用 dense `[batch, 1]`；Prefill 和混合 batch 使用 `[total_query_tokens]` packed 输入，不同请求由累计长度分隔。

### Decode metadata buffer 与 CUDA Graph

普通 eager Decode 每轮都会由 Python 发起 embedding、各层 linear/attention/MLP、norm、LM head 和 argmax 等许多 kernel。小 batch 时计算本身较短，CPU/Python/CUDA driver 逐个提交 kernel 的固定开销可能变得显眼。CUDA Graph 会先记录这一串 GPU 操作及其依赖，之后用一次 `replay()` 重新提交整张图；它减少的是 launch overhead，不减少模型计算，也不消除 attention kernel 对 block table 的读取。

这里的 **eager** 是执行方式：Python 运行到一个 PyTorch CUDA 算子，dispatcher 就立即把对应 kernel 提交到 GPU；它不表示单请求、不使用 batching，也不是一种 Prefill/Decode 调度策略。调度和执行是两条独立的轴：Scheduler 可以先组成纯 Prefill、纯 Decode 或混合 batch，然后选择用 eager 执行；只有满足捕获条件的纯 Decode batch 才改用 Graph replay。因此“MoE 动态 expert dispatch 走 eager”只表示每轮根据本次 router 结果正常执行算子，不表示 MoE 请求不能与其他请求组成 batch。

| 维度 | 决定的问题 | 当前选择 |
|---|---|---|
| 调度策略 | 本轮把哪些请求和哪些 Prefill/Decode token 放在一起 | continuous batching、chunked Prefill、允许 mixed batch |
| 执行方式 | 选好的 batch 如何向 GPU 提交算子 | eager，或满足条件时 CUDA Graph replay |

传统 CUDA Graph 要求捕获期间的算子序列、控制流，以及相关 tensor 的 shape、stride/layout 和显存地址保持一致；tensor 中的数值可以改变。限制不只针对 Q，而是覆盖 input、Q/K/V、中间激活、logits、block table 等整条捕获路径。纯 Decode 固定 `S=1`，按精确 `B` 建 bucket 后，Q 为固定的 `[B, heads, 1, head_dim]`；历史长度只是 `cache_seqlens` 中变化的数值，KV pool 和 block table buffer 的外形、地址仍固定。Prefill 的 packed token 总数 `T` 经常变化，会带动 Q 和所有中间激活变形；技术上可以为固定 `T` 建 bucket 或 padding，但当前收益较低且浪费较多，所以仍走 eager。MoE 每轮的 expert 选择和各 expert token 数又是数据依赖的动态形状，当前也不捕获 Graph。

Graph 要求 tensor 的形状和地址保持稳定，但每轮的 token、sequence length、页表内容和 request 顺序都会改变。因此纯 Decode 为每个精确 batch-size bucket 持有一套固定地址的 buffer：

```text
static input_ids [B, 1]
static block_table [B, max_blocks]
static cache_seqlens / positions / slot_mapping / cu_seqlens
```

每轮仍由 Scheduler 选择请求，并在 Graph 外把新值原地写入这些 buffer；replay 中的 FA2 kernel 会从相同地址读取本轮的新内容。地址不变不等于值不变。`Qwen3ForCausalLM.forward_prepared()` 让 eager 和 Graph 共用已经准备好的 `PagedKVBatch`，避免模型内部重新分配 metadata。未配置 Graph 的纯 Decode 也按 batch size 复用这套 metadata buffer。

当前只捕获配置 `decode_cuda_graph_batch_sizes=(...)` 中的精确大小。例如只配置 `(2, 4)` 时，batch 2/4 走 Graph，batch 1/3 走 eager；不会向较大 bucket 填 dummy request，因为 dummy token 还会牵涉页表、KV 写入和采样语义。某个 bucket 第一次命中时先 warmup/capture，后续才摊薄这次成本，所以 bucket 应由 workload 命中率决定，而不是越多越好。

Graph replay 不会再次执行 capture 时的 Python。各层 backend 因此在 Graph 内只写物理 KV，不推进 Python 维护的逻辑长度；输出 token 同步成功后，`commit_batch()` 才一次性提交所有层长度。若 kernel 失败，逻辑长度仍停在旧值。混合 Prefill/Decode 的 token 数和分段形状经常变化，当前继续使用 packed eager 路径，不进入 Decode Graph。

## 5. 真实 Qwen3、MoE 与 tokenizer

`Qwen3ForCausalLM` 同时承载 dense Qwen3 和 Qwen3MoE。配置显式保存 `head_dim`，不能再假设它等于 `hidden_size / num_attention_heads`：本地 Qwen3-0.6B 的 hidden size 是 1024，但 16 个 query heads 的 head dimension 是 128，因此 Q projection 实际宽度为 2048。

运行时权重布局为：

```text
qkv_proj      [q_heads * head_dim + 2 * kv_heads * head_dim, hidden]
gate_up_proj  [2 * intermediate, hidden]
o_proj        [hidden, q_heads * head_dim]
down_proj     [hidden, intermediate]
```

SafeTensors loader 在 meta device 上建立模型，再用最终 dtype/device 一次分配存储，逐 tensor
复制权重，避免先构造完整 FP32 模型。Hugging Face checkpoint 中分离的
`q_proj/k_proj/v_proj` 和 `gate_proj/up_proj` 会写入 fused parameter 的不同切片；单文件
和 index 分片 checkpoint 都使用同一条路径。TP 模式下 loader 从每个逻辑 tensor 分别
取得当前 rank 的 Q/K/V、gate/up 行分片，o/down 则取得输入列分片；GPU 上不会先构造
完整 projection 权重。

MoE layer 根据 `decoder_sparse_step` 和 `mlp_only_layers` 选择 dense MLP 或 sparse block。Sparse block 执行：

```text
router linear -> fp32 softmax -> top-k experts
              -> optional top-k renormalization
              -> expert SwiGLU
              -> routing-weighted index_add
```

单 rank 的 expert 参数按 `[num_experts, ...]` 保存；多 rank 时 global experts 按连续区间
分配，每个 rank 只分配 `[num_experts / world_size, ...]`。loader 同时接受新版 packed
expert tensor 和旧版逐 expert gate/up/down tensor：前者读取本 rank 的 expert 切片，
后者识别全部 key 但只写入本 rank 拥有的 expert。

当前是 replicated-token expert parallel：Attention 沿用 TP，all-reduce 后各 rank 都有完整
hidden；router 在各 rank 复制并得到相同 global expert ID；每个 rank 只计算自己的 experts，
再 all-reduce 相加为完整 MoE 输出。它不拆分很小的单个 expert，而是把不同 experts 分卡。
`naive` dispatch 对每个本地 expert 扫描一次 routing 结果；默认 `sorted` 只筛选一次本
rank assignments，再按 expert 排序分组，减少重复动态 `where`。二者共用相同权重和通信，
方便做优化前后 A/B。该路径仍是逐 expert GEMM 正确性基线；更高性能版本应结合 token
ownership、all-to-all dispatch/combine 和 grouped GEMM/Triton。数据依赖的 MoE
dispatch 仍走 eager。

`HuggingFaceTokenizer` 离线加载 checkpoint tokenizer；chat endpoint 直接调用模型自带的 `apply_chat_template()`，支持 Qwen3 `enable_thinking`，不再手写 role 字符串。增量 decoder 保存生成 token 上下文，只发送稳定、可打印的新后缀，避免单个 token 的 UTF-8 byte fragment 产生乱码。

### 流式 detokenization 的稳定前缀

流式输出的单位实际是 token，不是汉字或字母。当前采用保守规则：完整的 CJK 字符通常可以立即提交；末尾尚无空格的 Latin 子词或不完整 UTF-8 byte fragment 先留在 decoder 中，遇到空格、换行、后续稳定字符或生成结束时再输出。这不是“英文在原理上不能逐字母输出”，而是避免 tokenizer 的后续 token 改写尚不稳定的文本尾部。

decoder 每次重新解码累计 token，并维护已经发送的稳定前缀 `sent_text`。完整解码结果必须仍以 `sent_text` 开头，而且本轮候选前缀绝不能比它更短。例如 `我是 -> 我是AI` 时可以暂存 `AI`，但不能撤回已经发送的 `我是`；最终所有增量片段拼接后必须严格等于一次性完整解码结果。这个不变量由 tokenizer 回归测试覆盖。

Sampling 在每个 Request 上保存 temperature/top-k/top-p/seed。Scheduler 为每个非 greedy 请求建立独立的 device generator，因此随机序列不会因请求与谁组成 batch 而改变；CUDA Graph 当前只用于全 greedy 的 Decode batch。

本地 dense Qwen3-0.6B 已完成 311 个 SafeTensors tensor 加载、FP32 reference logits 对齐、BF16 FA2 paged 单请求生成和并发 batch smoke test。小型 dense 与 MoE 配置均逐层或最终 logits 对齐 Transformers；MoE TP=2 的 packed/逐-expert checkpoint 加载、forward 和 continuous batching 已与 TP=1 对齐。真实 MoE checkpoint 留待云端验证。

## 6. Tensor Parallel 与多进程调度

Tensor Parallel 解决的是“一次模型 forward 怎样由多张卡共同完成”，continuous batching
解决的是“这一轮选择哪些请求和 token”。两者相互独立，但 TP 的所有 rank 必须执行同一
个 batch，否则 collective 会死锁，或者在形状碰巧相同时产生请求/KV 错配。

### 模型分片

`TensorParallelContext` 保存 TP group 内的 `rank`、`world_size` 和模型通信
`process_group`。TP=1 时所有 helper 退化为完整尺寸和 no-op collective，因此单卡和多卡
共用模型代码。

Dense Qwen3 使用成对的 column/row parallel：

```text
Attention:
replicated hidden
  -> column-parallel QKV
  -> 每个 rank 的 local heads + rank-local paged KV
  -> local attention
  -> row-parallel O projection
  -> all-reduce，恢复 replicated hidden

MLP:
replicated hidden
  -> column-parallel gate/up
  -> local SiLU(gate) * up
  -> row-parallel down projection
  -> all-reduce，恢复 replicated hidden
```

Column parallel 按输出维度切权重，每张卡直接计算局部输出，不先生成完整 tensor 再拆分。
QKV 和 gate/up 是 fused parameter，但每个逻辑段必须独立切分，所以 rank 0/1 保存的是
`[Q0,K0,V0]` / `[Q1,K1,V1]`，而不是对完整 `[Q,K,V]` 粗暴地从中间切一刀。

Row parallel 按输入维度切权重。若 `x=[x0|x1]`、`W=[W0|W1]`，各 rank 先算
`partial_r = xr @ Wr.T`，再以 all-reduce 求和得到完整输出；这里是求和，不是把向量
all-gather 拼接。Column 和 Row 之间的 Attention/SwiGLU 激活始终保持分片，避免收集
大型 intermediate tensor。

当前 embedding、RMSNorm 和 LM head 仍在各 rank 复制，因此 all-reduce 后每个 rank
都有相同的完整 logits。每层在 O projection 和 down projection 后各通信一次。KV pool
只按本 rank 的 KV heads 分配，显存随 TP 缩小；page-table 逻辑布局则必须跨 rank 一致。
MoE 在同一进程组上组合 Attention TP 与 replicated-token expert 分片；vocabulary-parallel
embedding/head、独立 TP/EP group 和 all-to-all token dispatch 尚未实现。

### Scheduler 控制面

当前基础运行时采用正确性优先的镜像方式：

```text
rank 0 TensorParallelScheduler
  -> 广播 add / abort / step / reset / shutdown
  -> 每个 rank 在本地镜像执行 Scheduler
  -> rank 0 广播本轮权威 SchedulerBatchPlan
  -> 所有 rank 比较本地计划，一致后才进入 model forward
  -> 模型使用 TP group 做 tensor collective
  -> rank 0 广播采样 token
  -> 所有 rank 以相同 token 更新 Request、Radix 和 cache metadata
```

`SchedulerBatchPlan` 覆盖容易造成静默错误的字段：phase、eager/packed 执行类型、请求
顺序、输入 token、append lengths、logits 位置、旧长度和每个请求的 page table。非零
rank 不对外接收请求，只在 `run_worker_loop()` 中等待 rank 0 命令。每条命令结束后还会
all-gather 结果或异常，检查请求状态迁移是否一致。

控制消息使用 CPU/Gloo process group；模型权重和激活 collective 可以独立使用 NCCL
TP group。这样 Python 对象广播不会混入高吞吐 tensor 通信。随机采样目前各 rank 都会
计算一次，但最终以 rank 0 token 为权威并广播，保证下一轮输入和 prefix key 不分叉。

正式 CLI 由 `torchrun` 设置 `WORLD_SIZE/RANK/LOCAL_RANK`。每个 rank 先把
`cuda:LOCAL_RANK` 设为当前设备，再初始化默认 NCCL model group；随后所有 rank 以相同
顺序建立 Gloo control group、加载各自的 checkpoint 分片并构造本地 Scheduler。rank 0
继续构造 tokenizer 和 `GenerationService`，非零 rank 则进入 `run_worker_loop()`。rank 0
退出交互或发生正常清理时会广播 `shutdown`，所有 worker 完成同一条命令后共同销毁
process group。

```text
torchrun --nproc-per-node=4
  ├─ rank 0 / cuda:0 -> service + TP scheduler driver
  ├─ rank 1 / cuda:1 -> TP scheduler worker
  ├─ rank 2 / cuda:2 -> TP scheduler worker
  └─ rank 3 / cuda:3 -> TP scheduler worker
```

单进程 `Scheduler` 会拒绝 TP model，防止只有 rank 0 进入 all-reduce 后永久等待。
TP Decode CUDA Graph 也暂时被拒绝，因为 Graph 内 collective 的同步捕获尚未在 GPU
上验证。CPU/Gloo 已覆盖 chunked Prefill、mixed batch、纯 Decode、token 对齐和结束后
cache 完整性；`torchrun` 启动链已经接入，NCCL TP=2/4、故障超时和独立 Engine 进程
仍待云端完成。

## 7. Radix prefix cache

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

## 8. 与 Mini-SGLang 的关系

以下比较固定基于本地 `/home/sheep/mini-sglang` 快照 `9a91cfa`，不代表其未来版本。

| 方面 | Mini-SGLang `9a91cfa` | MySGLang 当前选择 |
|---|---|---|
| 运行架构 | 多进程、每个 TP rank 持有 Scheduler/Engine | TP=1 单 Scheduler；TP>1 由 rank 0 广播命令，各 rank 镜像 Scheduler/KV metadata |
| 请求状态 | 主要由对象所在容器和长度字段隐式表达 | 显式 enum、迁移检查和终止事件 |
| 调度 | 简洁的 Prefill-before-Decode 路径，支持成熟的 flattened Prefill | Decode 每轮推进，与 budget 内的变长 Prefill 合为一次 forward |
| 页表表示 | 内部保存展开后的 physical token indices | block table 直接保存 physical page IDs |
| Admission | 基于 available size 与 inflight 估算 | reservation 与实际 page binding 分开，行为保守但易验证 |
| Radix | tensor key、快速比较 kernel、timestamp + leaf heap | Python tuple key、timestamp + leaf heap |
| 防御能力 | `reset()` 未实现，integrity checker 为空 | reset、stats、tree/allocator/scheduler 完整检查 |
| 执行性能 | 真实模型、GPU attention、CUDA Graph、TP 等完整路径 | dense Qwen3、FA2 paged attention、Decode Graph 和基础 dense TP；尚无 grouped GEMM/EP/TP Graph |

双方的 Radix eviction 都不是双向链表 LRU。MySGLang 对显式状态、验证和 reference oracle 的加强服务于学习与调试；Mini-SGLang 的 kernel、metadata 和分布式路径则远比当前项目完整。

## 9. 当前限制

- 本地只有 dense Qwen3-0.6B checkpoint；真实 MoE checkpoint 尚待云端验证；
- MoE 已有 expert ownership 分片，但仍使用 replicated-token all-reduce 和逐 expert loop，
  尚无 all-to-all、grouped GEMM 或负载均衡性能优化；
- TP=1 使用单进程同步 worker；TP>1 目前镜像 Scheduler，尚未分离独立 Engine 进程，也没有 scheduler/forward overlap；
- 纯 Decode metadata 已复用；ragged/mixed metadata 与 slot mapping 仍由 Python 构造，尚未做运行时性能调优；
- reference attention 每轮 gather 历史 K/V；
- page 数量显式配置，没有根据实时显存自动计算；
- 没有 active-request preemption、swap、cache namespace 或跨实例 cache routing；
- CUDA Graph 目前只覆盖单 rank dense、greedy、精确纯 Decode bucket；尚未覆盖 TP 和生产容错；
- dense TP 已在四张 RTX 4090 上通过 NCCL 实测；MoE expert 分片已通过 CPU/Gloo 正确性
  测试，但真实 checkpoint 的 NCCL 验收、all-to-all EP、vocab parallel、worker 超时与故障恢复尚未完成；
- HTTP 协议尚未由正式模型客户端与数据集评测。

后续顺序和验收边界见 [roadmap.md](roadmap.md)。
