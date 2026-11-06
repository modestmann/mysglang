# HANDOFF：mysglang 与实习冲刺

最后更新：2026-09-30（T4 完成，下一步 T5）。写给接手这个仓库的下一个 Claude 会话（以及用户本人）。

## 1. 用户与目标

- 用户在找 **AI infra（推理方向）实习**，**2026-10-08 投递**；面试预计在投递后 1–3 周。
- 用户坦白：mysglang 是**纯 vibe coding** 写的，KuiperQwen3（`~/KuiperQwen3`）是网上找的开源项目改的，
  **目前对两个项目的代码基本不理解**。
- 因此当前的主要任务不是加功能，而是**教会用户**：讲解 → 指定要读的代码 → 让用户自己动手 → 小测。
  不要直接替用户写面试会被问到原理的核心代码。
- 用户偏好：
  - 不要催进度，用户自己定节奏（原话："T1 还没做，你先别急"）；
  - 不按天卡死，用户在线时随时领下一个任务；
  - Claude 要有自主判断，主动提出该加什么、改什么，不要用户说加什么就加什么；
  - 测评结果、计划、知识点都写进冲刺笔记文档，不要让用户自己手记。

## 2. 冲刺笔记文档（Claude 可以直接读写）

**AI Infra 实习冲刺笔记**：<https://claude.ai/code/artifact/159d37f5-8584-4825-8766-23d0c9de33a7>

用 Claude Docs 连接器读写（工具 `mcp__claude_ai_Claude_Docs__*`；写之前先 `guide` 读 `topic.index`）：

| 用途 | 值 |
|---|---|
| doc id（`container`） | `{"kind":"project","id":"159d37f5-8584-4825-8766-23d0c9de33a7"}` |
| 正文 node id（`update` 的 `ref`） | `{"object":"node","id":"d43c79f9-f3fe"}`，`engine: "prose"` |
| tab（file）id（建图 widget 时的 `parent`） | `a7f79908-6097` |

- 写之前先 `read` 大纲拿到当前块 id 和 `rev`；用户可能改过内容，**以用户的改动为准**。
- 文档结构：目标与原则 / 任务清单（T0–T12 勾选框 + Claude 改进建议表）/ 测评记录 / benchmark 结果
  （KuiperQwen3 基线、logits 对齐、mysglang 实测）/ 知识笔记（智能指针、Qwen3 forward 形状表、
  KuiperQwen3 五层架构图）/ 简历草稿与投递策略。
- **AI Infra 面试准备手册**（2026-10-03 新建）：<https://claude.ai/code/artifact/ed40f802-bb84-4fb7-b217-efbfb2423aa6>，
  doc id `ed40f802-bb84-4fb7-b217-efbfb2423aa6`，正文 node `c653c22b-a775`，tab `8d6f768a-400b`。内容：`~/面经` 43 张截图
  （去重 40 场）的统计、三档准备要求、项目拷打追问链、与 vLLM/SGLang 逐环节对比（带源码行号）、模型架构/量化/前沿/基础题
  及参考答案、Claude 的建议（含三种风格的模拟面试）。面经原文转录在会话 scratchpad 的 `mianjing.json`，没有入库。
- 简历那一节有一条 Claude 留的评论问来源写法；**2026-10-04 用户已决定**：不写"AI 辅助"、不提 KuiperLLama（见第 3 节第 2 条）。
- **推理方向简历**：以 Windows 上的 `C:\Users\31087\Desktop\filled-resume\resume-infra.tex` 为准（WSL 路径 `/mnt/c/Users/31087/Desktop/filled-resume/resume-infra.tex`，用户在 Windows 编译；`~/filled-resume/` 里只是副本）。mysglang 块由 mysglang 会话维护，
  KuiperQwen3 块和其余部分由 kuiperqwen3 会话维护，改之前先重读文件。本机没有 xelatex，用户在 Windows 上编译；待确认是否一页。

## 3. 已做的决定

> **2026-10-01 最新**：用户**取消了 9-29 的功能冻结**（原话"把那个冻结项目这个指令删掉，改就改嘛"）。两个项目都可以继续改进
> （抢占、换 FlashInfer 小页后端等重新可选）。仍按第 5 条分工：学完对应原理再改，核心逻辑由用户在指导下自己写。
> 学习计划（T5–T12）与简历草稿在冲刺笔记文档里。
> 用户认可、"以后加入"的改进（由用户在指导下写）：自动按显存算 KV 池、抢占 + 重算、CUDA Graph 按 bucket 补齐
> （参考 mini-sglang `engine/graph.py:160` 的 dummy 请求 + dummy 页）、prefill/mixed 分段 Graph（vLLM `FULL_AND_PIECEWISE`）、
> overlap 调度（CPU 排下一步与 GPU 算当前步重叠，mini-sglang `scheduler.py:83` 默认开启）。
> 教学：T10 之前只按 dense Qwen3-0.6B 讲，Qwen3.5 的 DeltaNet/MTP/rope_delta 分支一律跳过（用户 10-01 要求）。

> **2026-09-29 更新**：用户改变了第 1、4、5 条的执行方式——要求 Claude **直接实现**
> Qwen3.6-35B-A3B-FP8 适配（混合架构 + MTP 投机 + FP8 + 接口），"完成之后我再学"。
> 工作在分支 `qwen3.6-hybrid-mtp` 上进行，master 保持 10-08 投递版本不变；
> 是否在投递前合并、简历写不写这部分，由用户决定。细节见第 7 节。

1. **10-08 前 mysglang 不加新功能。** 讲不清楚的功能只会给面试官更多深挖的方向。简历只写
   10-07 前能讲清楚的内容；EP all-to-all、投机解码讲不清就删掉。
2. ~~项目来源如实说明："AI 辅助实现"~~ **2026-10-04 用户改决定**：简历里**不写任何"AI 辅助"字样**（原话"删了删了，都是我自己写的"），项目描述、技能里都不要加回去。
3. **不装 vLLM**（用户选的 B；nano-vllm 基线对比已于 2026-09-29 按用户要求连同提交记录删除）。没装 vLLM 的原因：它会给共享的
   `~/nano-vllm/.venv` 新增 136 个包，并把 fastapi 从 0.141.1 降到 0.136.3、numpy 从 2.5.2 降到 2.3.5；
   而 mysglang 也在用这个环境。
4. **投递后再做的加分项**（按优先级）：
   - 抢占（preempt + recompute）：由用户在指导下**自己写**，这是高频面试题"KV cache 满了怎么办"；
     目前只出现在 `docs/roadmap.md` 里。
   - Qwen3-Next 混合架构（线性注意力 + 标准注意力 + MoE）：需要新的递归状态缓存，与 Paged KV / Radix
     共存，工作量大，先学原理再决定是否实现。
5. 分工原则：工程杂活（benchmark 脚本、测试、README）由 Claude 做；核心逻辑由用户自己写。

## 4. 当前进度

### 本次会话在 mysglang 新增（均为**本地提交，未 push**）

| commit | 内容 |
|---|---|
| `25eb926` | `benchmarks/benchmark_prefix_cache.py`：共享前缀负载，对比 cold（每个请求前清空前缀缓存）和 warm 的 TTFT、实际 prefill token 数、命中率，并校验两轮 greedy 输出一致 |

Radix cache（8 个请求，共享前缀，各带 64 token 后缀）：前缀 1024 token 时 TTFT 68.8 → 33.9 ms（2.03×），
前缀 3072 token 时 232.4 → 37.7 ms（6.16×）；命中率 82.4% / 85.7%；greedy 输出完全一致。

### 学习进度（任务编号对应文档里的任务清单）

- T0 摸底：完成。概念大方向都对，精确度不够；C++ 智能指针最弱，复测已过。
- T1 Qwen3 forward：完成（弱点：反复把 kv_heads 写成 6，实际是 8——**凡涉及 config 数字都提醒用户核对**）。
- T2 CUDA 基础与 reduce kernel（v0/v1/v2）：完成，`~/study/reduce.cu` 全 PASS。
- T3 int8 GEMV 优化（KuiperQwen3，用户自己写）：完成，lm_head DRAM 利用率 64%→97%，decode ITL −9%。
- T4 分块 GEMM / 寄存器分块 / bank conflict / Tensor Core 对比 / online softmax 与 FlashAttention 1→4：完成
  （2026-09-30，`~/study/gemm*.cu`、`online_softmax.py` 均 PASS）。用户的固定模式：**下标/维度推理容易错**
  （list vs 标量、`A[i][j]` vs 一维展开、p[d] 与 p[t] 混用），讲解时先列"每个变量几维、每维是什么"。
- 已知的 FlashAttention 结论（讲 T7 attention backend 时可直接引用）：分块 + 在线 softmax，不物化 n×n 分数；
  o 每来一块先乘 exp(m_old−m_new) 再加 Σ p·V，最后只除一次 l。
- T5 请求生命周期与调度：完成（2026-09-30，已写入冲刺笔记并打勾）。弱点同上：每个请求各自的 offset/长度与 packed 总长度混淆。
  暂缓待讲：混合模型为什么关 radix、CUDA Graph 适用阶段、MoE TP/EP。
- T6 Paged KV 与 Radix：完成（2026-10-01，已打勾；冲刺笔记新增"面试题库"一节，之后每个任务都往里加题和参考答案）。
  弱点：整页边界（4 个 token 算成 2 页）、"查只切不挂，插入才挂"错过两次。
- T7 CUDA Graph：完成（2026-10-01，只按 dense 讲；补讲了 continuous batching 概念）。弱点：手算 TTFT 用了步号而非"该步结束时刻"。
- T10 混合架构：完成（2026-10-02，已打勾）。用户跟不上公式推导；有效的讲法是"先画 dense vs hybrid 架构对比 → 只讲输入输出 → 每条消息只讲一个点、只问一个问题"。S 的更新公式按用户意见暂缓（面试只需讲清"固定大小摘要 + 对推理系统的三个影响"）。
- T11 投机解码：完成（2026-10-02，已打勾）。讲了 n-gram、MTP、验证、KV 截断、S 记录+重放、掩码重放、追赶、投机采样、重复惩罚、sampler 代码；顺带介绍了图片编码器（细节留 T12）。
- T12 TP/EP/FP8/图片：完成（2026-10-02，已打勾，用户决定写进简历）。讲了 TP 切分、MoE 与 EP、卡 0 广播与两道保险、FP8 块 scale 与量化范围、128 对齐、Triton vs CUDA、图片 token 数、M-RoPE 与 rope_delta、ViT。
  **下一步**：按"可做的优化汇总"做改进（用户自己写），或回到 T8/T9 定稿简历与模拟面试；10-08 投递。
- 改进（分支 `perf-improvements`，用户授权每完成一项自动本地提交、不 push）：`202b0fd` 用户学习注释；`b20d9c1` #1 自动算 KV 池（`--memory-fraction`）；`2b08204` #2 抢占+重算（`--preemption`，用户设计状态机与牺牲品/放回队首，Claude 写额度记账与测试）。`aba556b` #3 CUDA Graph 按桶补齐（KV 垃圾页 + DeltaNet 垃圾格子，只对普通 Decode 图；用户写了 bisect 选桶思路，c8 随机采样 +54%）。`9ad264b` #4 启动时用全假请求 batch 预录所有 Decode 桶（最大 ITL 125 → 17 ms）。`e6d3c5d` #10 混合模型前缀缓存（`--state-snapshots`，在整页边界存 DeltaNet 快照；用户写了切块边界规则，Qwen3.5-0.8B 3072 前缀 TTFT 7.2×）。`63b2c44` #5 图片指纹钥匙（同图第二次 prefill 276 → 20）。`4ba3f0f` #9 FlashInfer 后端（`--backend flashinfer --page-size 16|1`，含 CUDA Graph；3000 前缀命中后 TTFT 80.6 → 45.7 ms，c8 吞吐与 FA2 持平）。实测都写进了笔记"可做的优化汇总"。剩余可选：#6 cache-aware、#7 overlap、#8 分段 Graph、#11 ViT 输出缓存、MTP 图补齐；建议回到 T8/T9（10-08 投递）。
- **学习顺序调整（10-01 用户决定）**：T10 → T11 → T12（学完写进简历）→ 做改进（笔记"可做的优化汇总"一节）→ 10-08 前回到 T8/T9。
- **T5–T8 才轮到 mysglang**：T5 continuous batching（`core/`、`scheduler/`）、T6 Paged KV 与 Radix（`cache/`）、
  T7 CUDA Graph 与 metadata、T8 TP/EP/MoE/投机解码（概念级）。

## 5. 待办

- [ ] **有空装上 mini-sglang 测一下**（用户提的；源码在 `~/mini-sglang`，尚未安装）。建议另建环境，
      不要动 `~/nano-vllm/.venv`；用 benchmark_generation.py 的同一负载对比。
- [x] `README.md` 里的模型路径已改为 `../KuiperQwen3/artifacts/qwen3-0.6b/hf-source`（2026-10-05）。
- [ ] 用户确认后 push（mysglang 和 KuiperQwen3 都有未 push 的本地提交）。
- [ ] 仓库里有一个未跟踪的 `CLAUDE.md`，不是本次会话创建的，没有动它。

## 6. 踩过的坑

- **环境**：`source ~/nano-vllm/.venv/bin/activate`，运行时加 `PYTHONPATH=src`
  （运行 prefix benchmark 时是 `PYTHONPATH=src:benchmarks`）。
- **vLLM 没装**：在 `~/vllm` 目录里执行 `import vllm` 会导入源码目录本身，看起来像装了，其实没有
  （缺 `vllm._C`）。要用 `importlib.metadata.version('vllm')` 或换个目录验证。
- 为了跑 vLLM 源码的 dry-run，已经往 `~/nano-vllm/.venv` 装了构建工具：cmake、ninja、setuptools-scm、
  setuptools-rust，并把 setuptools 限制在 81 以下。只在打包时使用，不影响运行。
- **FA2 backend 的 page size 是 256**（`cli.py:193`），Radix 按整页匹配：前缀不足 256 token 不会命中，
  尾部不足一页的部分也不会复用。设计前缀 benchmark 时要用 256 的倍数。
- `benchmark_generation.py` 在预热之后会调用 `reset_prefix_cache()`，所以它测不出 Radix 的收益，
  这正是另写 prefix benchmark 的原因。
- 批量跑 benchmark 时**不要把 stdout 管道接到会提前退出的命令上**：SIGPIPE 会在写结果之前杀掉进程。
  输出重定向到 `/dev/null`，结果只看 JSONL。
- **跨机器的数字不要混用**：旧报告里的 CUDA Graph 6.16× 是在 4×4090 上测的；在 4060 上，并发 1 时是 3.6×。
- **FlashInfer + CUDA Graph**：图固化的是缓冲区地址。workspace 只能分配一次，graph wrapper 只能在 runner `close()` 之后释放；
  不要传 `disable_split_kv`（会去掉 `block_valid_mask`）。GQA 组大小不在 {1,2,3,4,6,8} 时 decode 必须用 tensor core。
  补齐桶在 FA2 和 FlashInfer 上都会让个别请求几十上百 token 后分叉（GEMM 行数变化的 bf16 舍入），比对 token 用精确桶。
- **GPU 只有一张，常有另一个 Claude 会话（KuiperQwen3）同时在测速**：抢 GPU 会让双方数字失真、甚至 OOM。
  跑 benchmark 前用对方会话约定的 `flock <gpu.lock>` 排队；没加锁时测的数字不要写进笔记。
- 机器是 WSL2，只有一张 **NVIDIA RTX 4060 Laptop**（sm_89，8 GB，约 256 GB/s），通过 `/dev/dxg` 暴露，
  没有用 AMD 显卡。


## 7. 2026-09-29：Qwen3.6-35B-A3B-FP8 适配（分支 `qwen3.6-hybrid-mtp`）

用户要求 Claude 直接实现，之后再学。**改动都在分支上、尚未提交**（等用户确认再 commit）。

### 做了什么

| 模块 | 内容 |
|---|---|
| `config.py` | qwen3_5 / qwen3_5_moe（含 `text_config` 嵌套）、layer_types、部分 RoPE、shared expert、MTP、多 EOS、`Fp8QuantConfig` |
| `modeling/deltanet.py` | Gated DeltaNet：PyTorch chunk/递归 reference + fla kernel；verify 的 log 模式；ragged replay 与等宽掩码 replay |
| `modeling/qwen3.py` | 同一模型类承载 Qwen3.5：gated attention、零中心 RMSNorm、KV 头按 GQA 复制、shared expert、MTP 头、FP8 参数 |
| `modeling/fp8.py`、`triton_fp8.py`、`triton_moe.py` | 分块 FP8 量化/反量化、Triton W8A16 linear、grouped MoE 的 FP8 scale |
| `cache/recurrent.py`、`cache/hybrid.py` | DeltaNet 状态 slot 池；`HybridPagedKVCache` 统一登记页、MTP 草稿 K/V、状态 slot；混合模型关 radix |
| `scheduler/scheduler.py`、`sampler.py` | MTP：递归草稿 → verify → 回滚/replay → 追赶；n-gram 共用；投机采样（随机采样/penalty 也能起草）；多 stop token、penalty |
| `modeling/cuda_graph.py` | 混合模型 Decode/verify/MTP 草稿/追赶/状态提交全部可 Graph（快照-恢复 slot） |
| `serving/` + `server.py` | `reasoning_content`、`qwen3_coder` 工具调用、`chat_template_kwargs`、OpenAI 兼容服务 |
| `benchmarks/` | `validate_checkpoint_layout.py`（HTTP Range 离线校验 35B）、`make_fp8_checkpoint.py` |

### 已验证（本机 4060）

- Transformers 小模型 oracle（dense + MoE）logits 对齐；CPU/Gloo TP=2（1 个 KV 头复制）/TP=4 + MTP、TP=2 FP8 与 TP=1 逐 token 一致；
- 真实 `~/models/Qwen3.5-0.8B`（与 3.6 同结构）BF16 greedy 与 Transformers 逐 token 一致；MTP k=1 接受率约 70%（post-norm hidden；pre-norm 只有 63%）；
- fp32 + torch 后端下 MTP 开/关逐位一致；bf16 下的分叉是 kernel 形状不同导致的舍入差异；
- CUDA Graph：纯 Decode 约 2.9×；MTP 的 verify/草稿/追赶/状态提交全部进 Graph；Graph 与 eager 输出逐 token 一致（含随机采样请求）；
- `~/models/Qwen3.5-0.8B-FP8`（本脚本按官方格式转换）：FP8 kernel 与反量化路径分叉位置相同，权重显存 1436→971 MiB；
- 35B-A3B-FP8 离线布局：64196 个张量在 TP=4 每 rank 全部映射、参数恰好覆盖，每 rank 约 9.98 GiB。
- 投机采样：随机采样与带 penalty 的请求也能用 MTP/n-gram，统计检验确认分布与逐 token 采样一致；
- 全部测试：95 通过、3 跳过（跳过的是缺本地 tokenizer 的老测试）。
- 性能（真实对话，Graph 开）：见 `benchmarks/results/2026-09-29-rtx4060-laptop-qwen3.5/report.md`。
  混合模型 Graph 约 2.9×；FP8 权重 −30%、c4 +28%；0.8B 上 MTP 以 k=1 最好（c1/c4 约 +13%）。

### 上云（4×4090）待验收

1. `validate_checkpoint_layout.py` 已离线通过；上机后先 `mysglang.cli --prompt` 冒烟；
2. token oracle：`benchmark_generation.py`（TP=4，`--prompt` 固定）+ `validate_transformers.py --device-map balanced`；
3. A/B：`--moe-dispatch sorted/triton_grouped`、`--cuda-graph` 开关、`--speculative-mtp-steps 0/1/2/3`；
4. 记录显存（权重约 10 GiB/rank + KV + 状态 slot + Graph），写 `benchmarks/results/<日期>-4090x4-qwen3.6/report.md`。

### 环境变化

- 往 `~/nano-vllm/.venv` 装了 `flash-linear-attention==0.5.2` 和 `fla-core==0.5.2`（只新增这 2 个包，没有改动其他包）；
- 模型在 `~/models/Qwen3.5-0.8B`、`~/models/Qwen3.5-0.8B-FP8`。

### 坑

- 装了 fla 之后，Transformers 的 Qwen3.5 在 import 时就绑定 fla kernel，CPU 上会报 "Pointer argument cannot be accessed from Triton"；测试里用 `__wrapped__` 换回 PyTorch 实现（`tests/test_hybrid.py::_TorchFallbackKernels`）。
- Qwen3.5-0.8B 的模板默认**不**开 thinking，Qwen3.6 默认开；服务端总是显式传 `enable_thinking`。
- `config.json` 的 eos 是 `<|endoftext|>`，tokenizer 的 eos 是 `<|im_end|>`，必须两个都停。
- CUDA Graph 捕获前的预热会推进 DeltaNet 状态，必须快照/恢复（已实现，改 runner 时别丢）。
- `pkill -f <pattern>` 会连带杀掉包含这个 pattern 的当前 shell，改用 `ps` 找 PID。

### 建议的学习顺序（教学时用）

1. DeltaNet 递归公式 → 为什么状态不能像 KV 那样按页截断/前缀复用（`deltanet.py` 顶部注释、design.md §8）；
2. `HybridPagedKVCache` 的资源生命周期（admission/finish/abort 三条路径）；
3. MTP 的"草稿 K/V 与目标对齐"不变量和一轮投机的 5 步（design.md §8 的流程图）；
4. 掩码 replay 为什么等价（g=β=0 是恒等更新）；
5. 分块 FP8：scale 布局、TP 为什么要求 128 对齐、W8A16 与 W8A8 的区别。

### 2026-09-29 追加：Qwen3.5-0.8B 图片输入（已提交前的工作在 `9fdbdc7`）

- 用户要求把图片接口适配到 Qwen3.5-0.8B，35B 的视觉"有空再做"。
- 新增 `multimodal.py`（PIL 预处理、占位符展开、M-RoPE）、`modeling/vision.py`（ViT）、调度器视觉特征缓存与
  `mrope_positions`/`rope_positions`、服务层 `image_url`、CLI `--image` 和 `/image`。
- 验收：预处理与 HF 逐位一致；小模型 logits 与 HF 对齐；真实 0.8B 完整服务路径 40 token 与 HF 逐个一致；
  TP=2、CUDA Graph 均一致。测试 106 通过、3 跳过。
- 环境：往 `~/nano-vllm/.venv` 装了 `pillow==12.3.0`（只新增这一个包）；**没有装 torchvision**（会动 torch）。
- 坑：HF v5 的 `AutoImageProcessor` 需要 torchvision；oracle 用 `Qwen2VLImageProcessorPil`。
  `tokenizer.apply_chat_template(tokenize=True)` 返回 BatchEncoding，不是 dict。
- 待做：35B 视觉塔实测、视频、带图片哈希的前缀缓存 key。


## 8. 教练接力（2026-09-30，用户把后续教学迁到 mysglang 会话）

**下一课 = T5：一个请求在 mysglang 里的完整生命周期 + continuous batching。**

教学方式（用户明确要求，别改）：
1. 手把手：先用一张"变量—形状/含义"表讲清，再让用户填空（`____` 模板）或读指定函数后回答；
2. 用户交代码/答案后，**先编译或运行检验再评分**；不要把 racecheck 之类长任务和编译捆在一条命令里；
3. 每完成一个任务先真诚夸再进下一步（用户是 985 研一，害怕面试、自称小镇做题家）；
4. 每个任务的结果、答错的点、可上简历的数字，写进冲刺笔记文档（见第 2 节），并勾选任务清单；
5. 不按天卡死；主动提出改进，不要等用户说。

T5 建议路线（先看代码再讲）：`serving/`（请求进入）→ `scheduler/`（waiting/running 队列、每步选谁 prefill、
谁 decode）→ `core/`（一步 forward 的 batch 组装）→ 回到 `serving/` 流式输出。
给用户的题目类型：画一步调度的时间线；给定 3 个请求到达时间和长度，手推 static batching 与 continuous
batching 各自的吞吐/TTFT；填空写"选择本步要跑的请求"的伪代码。

T6 Paged KV + Radix；T7 CUDA Graph + attention backend metadata；T8 TP/EP/MoE/投机/DeltaNet 概念级（第 7 节
"建议的学习顺序"）。T9–T10：定稿简历、模拟面试；**2026-10-08 投递**。

必须向用户解释清楚的一个问题：为什么 mysglang 比 nano-vllm 慢约 25%（数据在
`benchmarks/results/2026-09-28-rtx4060-laptop/report.md`），面试一定会问。

KuiperQwen3 侧的可选后续（投递后再做）：CUDA Graph / 算子融合、把分块 GEMM 接进 prefill、AWQ。
