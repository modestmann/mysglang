# DeepSeek V4.1：原生量化加载与多卡生成

当前支持 `DeepSeek-V4.1-Flash-1017` 的配置解析、小规模 FP32 参考计算、
SafeTensors 校验、原生 MXFP4/MXFP8 分片加载，以及独立 CLI 的多卡纯文本生成。
2026-11-06 已在 8 张 H20-3e 上加载真实权重并生成英文、中文。
实现直接运行 mysglang 模型与 Triton kernel，不调用 vLLM/SGLang 推理引擎。

独立 CLI 支持单请求、贪心采样、EOS 停止和有界连续缓存，默认上下文上限 4096。
`--interactive` 模式支持常驻多轮聊天、跨轮缓存复用和中文/emoji 流式输出。
模型 API 支持等长 batch；分页 Scheduler/HTTP、长上下文性能验收、视觉和 DSpark
仍待接入。现有 Qwen 服务入口不接受这个新模型。

## 运行真实模型

在 204 的终端启动常驻聊天：

```bash
bash /home/ubuntu/yqf-intern/mysglang/scripts/chat_deepseek_v41.sh
```

脚本默认用 8 张 GPU 和已有 uv Python，只加载一次权重。看到 `你>` 后输入一行问题，
回车发送，回答直接在 `DeepSeek>` 后流式显示；后续问题可以引用前面的对话。
默认每轮最多生成 256 tokens，整个上下文上限 4096 tokens。进程运行期间权重保持
常驻，`/exit` 会结束所有 rank 并释放资源。

| 输入 | 行为 |
|---|---|
| 普通文本 + 回车 | 保留历史，生成下一轮回答 |
| `/clear` 或 `/reset` | 清空历史和缓存，保留已加载模型 |
| `/exit` 或 `/quit` | 结束聊天 |
| `/help` | 显示帮助 |
| EOF（空行处 Ctrl+D） | 广播退出命令，结束聊天 |

自定义回答长度、上下文预算，并保存本次聊天记录：

```bash
bash /home/ubuntu/yqf-intern/mysglang/scripts/chat_deepseek_v41.sh \
  --max-new-tokens 512 --max-seq-len 4096 \
  --output /home/ubuntu/yqf-intern/mysglang/benchmark-results/deepseek-v41-chat/chat.json
```

指定 `--output` 时，每完成一轮就更新 JSON；`session` 区分 `/clear` 前后的会话，
`reused_tokens` 和 `prefill_tokens` 记录复用和新计算的 token 数。未指定时只显示
终端对话。历史仅在当前进程中保持，JSON 是记录，不是下次启动自动恢复的状态。

长度上限按“历史 + 新输入 + 回答预算 + 一个结束符”预先检查。空间不足时会提示
缩短输入或 `/clear`，保留原历史；不会默默截断上下文。达到每轮输出上限时明确
提示，并补上 assistant 的 EOS 边界，下一轮仍可继续对话。

终端只由 rank 0 读写，输入/清空/退出经独立 Gloo 控制组广播；等待输入期间不挂起
NCCL collective。控制组等待上限为 365 天，模型计算仍保留原有的通信超时。
GPU/通信等运行时异常会终止进程组；不支持 Ctrl+C 仅取消本轮后继续聊天。

脚本可通过 `MYSGLANG_PYTHON`、`MYSGLANG_DS_MODEL`、`MYSGLANG_DS_CONFIG` 和
`MYSGLANG_DS_TP` 覆盖解释器、权重目录、配置和卡数；默认复用本机已有环境，不安装
依赖、不激活 Conda。运行缓存全部放在本仓库的 `benchmark-results/deepseek-v41-chat/`。
`CUDA_VISIBLE_DEVICES` 有设置时会沿用，卡数需与 `MYSGLANG_DS_TP` 一致。

单次/批量独立问答仍使用下面的 `--prompt` 入口；`--interactive` 与 `--prompt`、
`--raw-prompt`、`--compare-torch` 互斥。

使用独立的 `deepseek` extra（Torch 2.13、Triton 3.7、SafeTensors、Tokenizers、NumPy）；
基础包的 Torch 版本要求不变。本机验证复用 `../.venv-vllm/bin/python`，不安装或修改
那个环境，不激活 Conda。以下路径对应本机可读的 checkpoint 和已验证的环境：

```bash
cd /home/ubuntu/yqf-intern/mysglang
mkdir -p benchmark-results/deepseek-v41/{tmp,xdg,triton,cuda-cache}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  TOKENIZERS_PARALLELISM=false \
  TMPDIR=$PWD/benchmark-results/deepseek-v41/tmp \
  XDG_CACHE_HOME=$PWD/benchmark-results/deepseek-v41/xdg \
  TRITON_CACHE_DIR=$PWD/benchmark-results/deepseek-v41/triton \
  CUDA_CACHE_PATH=$PWD/benchmark-results/deepseek-v41/cuda-cache \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../.venv-vllm/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m mysglang.deepseek_cli \
  --model /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main \
  --config /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main/inference/config.json \
  --prompt '用一句话介绍你自己。' --max-new-tokens 48 \
  --output benchmark-results/deepseek-v41/generation.json
```

`--prompt` 可重复，权重加载一次后逐个生成。默认应用官方单轮 user、关闭 thinking 的
模板；`--raw-prompt` 接受已编码模板的文本。CLI 不执行 checkpoint 自带 Python。
`--compare-torch` 在相同权重、相同 TP 布局上，对照 prefill 和前两次 decode 的
logits/greedy token；每次比较使用独立缓存。此选项更慢，并额外使用临时反量化显存。

不要给 `--config` 指向本机无读取权限的顶层配置；显式使用上面的 inference 配置。
TP 必须整除 attention heads、output groups、index heads、词表和专家数，并满足
32 元素量化块对齐。发布版 TP=8 实际使用 attention TP=8、整专家 EP=8、
Engram 行分片=8；shared expert 和其他参数复制。不同 rank 必须执行相同输入序列。

## 代码入口

| 文件 | 职责 |
|---|---|
| `src/mysglang/modeling/registry.py` | 配置分派、显式参考模型构造 |
| `src/mysglang/modeling/deepseek_v41/config.py` | HF 嵌套配置/官方 inference 配置解析、共享状态拓扑检查 |
| `src/mysglang/modeling/deepseek_v41/model.py` | 模型主体、压缩、两级索引、MoE、Hyper-Connections、增量状态 |
| `src/mysglang/modeling/deepseek_v41/engram.py` | 词表归一化、素数分桶、n-gram 哈希和门控注入 |
| `src/mysglang/modeling/deepseek_v41/ops.py` | FP32 attention sink、RoPE/YaRN、Sinkhorn 和候选筛选 |
| `src/mysglang/modeling/deepseek_v41/checkpoint.py` | 只读权重头校验和逐 rank 存储核算 |
| `src/mysglang/modeling/deepseek_v41/loader.py` | 分块读取、TP/EP 切片、Engram 补行、权重消费完整性 |
| `src/mysglang/modeling/deepseek_v41/quantization.py` | UE8M0、E2M1 舍入与 Torch 数值对照 |
| `src/mysglang/modeling/deepseek_v41/triton_quant.py` | H20 上的 packed MX GEMM、激活量化和稀疏 attention |
| `src/mysglang/modeling/deepseek_v41/runtime.py` | rank 本地 BF16 模型、分布式通信、增量生成缓存 |
| `src/mysglang/modeling/deepseek_v41/generation.py` | 共用生成循环、多轮 token 历史、缓存提交和长度校验 |
| `src/mysglang/deepseek_chat.py` | Gloo 交互控制、终端循环、UTF-8 流式解码 |
| `src/mysglang/deepseek_cli.py` | torchrun 入口、单轮/交互模式、数值对照和 JSON 结果 |
| `scripts/chat_deepseek_v41.sh` | 本机常驻聊天启动及仓库内缓存配置 |

`DeepSeekV41ReferenceModel` 输出 `[batch, chunk_length, vocab_size]` logits。
它要求等长 batch、FP32 参数、显式的 Engram compressed token map。
超过一千万逻辑参数的配置会在参数分配前被拒绝，防止将发布版模型误建为 FP32。

```python
import torch
from mysglang import DeepSeekV41Config, build_reference_model

torch.manual_seed(0)
config = DeepSeekV41Config.tiny()
model = build_reference_model(config, token_map=list(range(config.vocab_size)))
tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
cache = model.new_cache()
prefill_logits = model(tokens[:, :3], cache)
next_logits = model(tokens[:, 3:], cache)
torch.testing.assert_close(
    torch.cat((prefill_logits, next_logits), dim=1), model(tokens), atol=4e-6, rtol=4e-6
)
```

tiny 配置仍包含两种压缩率、KV 跨层共享、独立 index source、两级候选筛选、
4 路残差、共享/路由专家，以及两层 2/3/4-gram Engram。identity token map
仅用于这个人工词表；真实词表必须使用 `compressed_token_map()`。

## 必须保持的语义

- ratio-2 的压缩组完成后才可见；跨 chunk 的半组保存在调用方缓存中。
  真实 checkpoint 的压缩器权重存为 BF16，但 ratio-2 投影、门控与半组状态使用
  FP32；完成池化后才转回 BF16 并执行 RMSNorm。ratio-1 投影保持 BF16。
- 窗口 KV 每层独立，压缩 KV/index keys 只由 source layer 持有。
  层间临时传递对象每次 forward 新建，避免模型或请求相互污染。
- 参考缓存保留完整历史，通过因果窗口 mask 限制可见范围。它不是生产 ring buffer
  或 paged cache，不应据此估算生产 KV 显存。
- 一次 forward 成功后才发布新缓存。异常、错误 token、batch/device 不匹配和
  长度超限不会推进缓存；缓存不能跨模型复用。
- attention sink 只增加 softmax 分母。压缩 KV 与窗口 KV 保持各自索引，attention
  输出在 grouped output projection 前应用逆 RoPE。
- Hyper-Connections 的 `comb` 行是来源流、列是目标流；残差混合是
  `comb.T @ residual`。当前子层计算出的 pre mix 供下一个子层使用。
- 路由修正 bias 只参与专家选择；权重来自未加 bias 的 `sqrtsoftplus` 分数。
- Engram 哈希依赖准确的压缩词表大小；空白、组合 Unicode、大小写和不完整 UTF-8
  token 的处理必须与训练实现一致。
- 稀疏索引同分时优先选择更早的位置。参考路径采用稳定排序，避免 `torch.topk`
  的同分结果随 masked future 列数改变；这不是当前 GPU kernel 的性能实现。

## 独立数值对照

在仓库根目录，使用安装了 `model` extra 的环境运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ../.venv-vllm/bin/python benchmarks/validate_deepseek_v41.py \
  --reference /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main/inference \
  --vllm /home/ubuntu/yqf-intern/vllm \
  --sglang /home/ubuntu/yqf-intern/sglang \
  --tokenizer /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main/tokenizer.json
```

该脚本只读三个参考目录，在当前进程中提取指定定义，避免导入整个框架、编译
CUDA 扩展或向参考目录写入 bytecode。它不运行未经初始化的官方 self-test，
而是将同一组已初始化权重装入两个模型，比较 prefill 和后续 decode 的逐层
hidden states、专家编号和所有位置的 logits。

对官方参考做的调整会写入结果 JSON：

1. 权重、GEMM 和激活使用 FP32，不执行 FP8/FP4 量化；
2. attention 使用独立的逐 query PyTorch 实现，Sinkhorn 使用 SGLang 的纯 Torch oracle；
3. 两边统一稀疏同分排序规则；
4. 输出所有位置的 logits，便于发现中间位置的偏差。

另行与 vLLM 核对 Engram 素数布局与随机乘数，与 SGLang 核对 hash IDs。
传入 `--tokenizer` 时，还会核对真实词表的全部归一化映射及发布版哈希参数。
输出包含参考仓库 revision、参考源文件 SHA256、当前实现 SHA256、环境与误差。
这些结果不表示与量化生产后端逐位一致。

## 权重布局与内存核算

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python \
  benchmarks/audit_deepseek_v41.py \
  /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main \
  --config /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main/inference/config.json \
  --tp 8 --text-only --no-draft
```

`--config` 显式选择可读的官方 inference 配置；工具不修改共享文件的权限。
省略 `--text-only --no-draft` 可以核算包含视觉和 draft 的存储。

工具读取索引及每个分片的 8 字节长度和 JSON 头，核对：

- 分片覆盖、重复名称、索引位置、文件长度和 payload 偏移；
- 全部 backbone 参数的逻辑形状，包含 FP4 打包后的 K 维；
- MXFP8 的 32×32 scale、MXFP4 专家的逐行 32 元素 scale、Engram 的逐行 scale；
- scale 的 UE8M0 类型以及索引声明的总字节数。

内存策略沿用官方转换器：attention TP、整专家 EP、Engram 按行切分并补齐、
其余参数复制；`wo_a` 转为 BF16 并移除对应 scale。真实加载器已经使用这套规则。
加载前校验全部头部和 backbone 形状，并检查本卡可用显存；默认以 64 MiB 行块
只读传输所需分片，不先构造完整 BF16 模型，不写转换后的 checkpoint。
FP4 专家保持 packed bytes，FP8 dense/Engram 保持 E4M3，UE8M0 scale 保持原始 byte。
Engram 最后一个 rank 的补行 weight=0，scale byte=127（数值 1）。
视觉/draft 的完整模型拓扑未做 shape 验收，但其所有 tensor 的头部与量化
scale 关系都会校验，并纳入所选策略的字节统计。

当前这份 checkpoint 的核算结果：

| 项目 | 字节/十进制 GB |
|---|---:|
| 全部权重 payload | 510,286,023,000 / 510.286 GB |
| 两个 Engram 表及 scale | 202,758,032,400 / 202.758 GB |
| 纯文本、关闭 draft，TP/EP=8 的每 rank 常驻权重估算 | 64,986,171,656 / 64.986 GB |
| 包含视觉和 draft，TP/EP=8 的每 rank 常驻权重估算 | 67,160,720,880 / 67.161 GB |
| 实际读取的 SafeTensors 头部 | 10,685,312 字节 |

64.986 GB 约为 60.523 GiB。此估算不含 KV/index/历史状态、激活、反量化临时
空间、通信 buffer、CUDA Graph 与 allocator 开销，也没有检查 tensor 数值。

## 多轮聊天验收（2026-11-08）

本轮结果位于 `benchmarks/results/2026-11-08-deepseek-v41-chat/`。
通过启动脚本在真实 PTY 中加载一次 TP/EP=8 模型，使用 192 token 上下文、
每轮 24 token 上限，完成 6 轮回答、2 次清空和一次超长输入拒绝：

| 场景 | 实际结果 |
|---|---|
| 告知名字“小林”后追问 | 回答“小林”，第二轮复用 17 tokens，仅 prefill 12 tokens |
| 超长输入被拒绝后再次追问名字 | 仍回答“小林”，既有历史保留 |
| `/clear` 后问 4 + 5 | 回答 `9`，`reused_tokens=0` |
| 数到 100，在 24 tokens 处停止 | 明确显示输出上限提示 |
| 截断后追问刚才的任务 | 正确回答从 1 数到 100、用逗号分隔、不要解释 |
| EOF 退出 | 8 个 rank 正常结束，退出码 0，GPU 占用恢复为 0 |

全项目 CPU 套件 185 项：149 通过、36 因可选依赖/硬件条件跳过；14 项新增会话
专项在完整依赖环境全部通过。专项包含两卡 Gloo、FP32 完整前缀重算 oracle、
跨轮末尾 token/EOS、超限无副作用、失败回滚、重置、中文/emoji 与不完整 UTF-8。
另行复验原 `--prompt --compare-torch` 的 8 卡单轮路径，输出 `4` 并正常 EOS，
两处 greedy 对照一致。Ruff 和 shell 语法检查均通过。

此次交互验证 rank 0 的 Torch 峰值 allocated 约 60.97 GiB、reserved 约 67.07 GiB，
不等于驱动总显存。它验证的是短上下文交互和缓存控制；4096 token 默认上限尚未
做长上下文质量/压力验收，原有量化 logits 偏差仍按第二阶段的记录界定。

## 第二阶段验收

当前 H20 路径用 BF16 Tensor Core 计算 MX 权重，Triton GEMM 在寄存器中解包 FP4、
按块应用 UE8M0 scale；没有把整套 FP4 专家展开成 BF16 常驻权重。激活遵循参考模型：
线性层 A8、窗口 KV 为 FP8，index Q/K 为 FP4 + E8M0，compressed KV 为 FP4 + E4M3。
稀疏 attention 使用 FP32 online softmax，未归一化的 P 转 BF16，再用 FP32 累加 PV。
此实现优先验证语义；尚未使用 fused/grouped MoE、CUDA Graph 或优化过的分页缓存。

```bash
# 隐藏 CUDA 时，CPU/Gloo 检查仍运行；CUDA 专项按设备数量跳过。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ../.venv-vllm/bin/python -m unittest tests.test_deepseek_v41_runtime -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ../.venv-vllm/bin/python benchmarks/validate_deepseek_v41_quantized.py \
  --reference /spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main \
  --sglang /home/ubuntu/yqf-intern/sglang \
  --output benchmark-results/deepseek-v41/quantized-oracle.json
```

新增量化测试覆盖全部 E2M1 编码及 ties-to-even、UE8M0 特殊值、weight/scale 跨文件、
真实切片和 Engram 补行、分片 grouped output projection、增量半组状态与异常回滚、
CPU/Gloo 两卡生成、CUDA/NCCL 两卡生成以及 Torch/Triton 对照。TP oracle 显式模拟
官方 RowParallelLinear 在 all-reduce 前将各 partial GEMM 舍入为 BF16 的规则。

第二阶段结果保存在 `benchmarks/results/2026-11-06-deepseek-v41-stage2/`：

- 小模型 BF16 量化计算图与官方 Python 模型：prefill + 3 步 decode 的 logits
  最大绝对误差为 0；4097 个 E2M1 输入与 SGLang 原语逐项一致。官方 GEMM/激活
  kernel 在此对照中替换为 Torch 原语，所以这不是官方融合 GPU kernel 的精度验收。
- 实际加载每卡权重 `64,986,171,656` 字节（60.523 GiB），与头部审计完全一致；
  文件系统/Triton 缓存预热后，8 卡加载 17.99 秒。rank 0 的 Torch 峰值 allocated
  约 62.42 GiB，reserved 约
  68.83 GiB，包含量化对照临时空间；不含 CUDA context/NCCL 的全部驱动分配。
- `What is 2 + 2? Answer with just the number.` 输出 `4` 并遇到 EOS；中文输入
  `用一句话介绍你自己。` 输出 `我是DeepSeek，由深度求索公司创造的AI助手，随时为你提供帮助、解答问题，并乐于与你交流！`，
  共 28 个输出 token（含 EOS），正常结束。
- 真实权重的 5 处 Torch/Triton logits 对照均有相同 greedy token；余弦相似度
  为 0.9840–0.9915，最大绝对误差 2.13–3.01。完整 40 层量化计算仍有舍入累积差异，
  这两条短提示只证明端到端功能，不能替代质量集或长上下文精度验收。
- 全项目 CPU 套件运行 171 项：137 通过、34 因 CUDA/可选依赖/本地 tokenizer
  artifact 不满足而跳过；11 项新增 runtime/量化专项在两张 H20 上全部通过。

## 第一阶段历史结果与后续

回归测试：

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python \
  -m unittest tests.test_deepseek_v41 tests.test_deepseek_v41_checkpoint -v
```

2026-11-06 的实际结果保存在 `benchmarks/results/2026-11-06-deepseek-v41/`。
全量 CPU 套件运行 160 项：130 通过、30 因 CUDA/可选依赖/本地 tokenizer
artifact 不满足而跳过，其中新增的 29 项 DeepSeek 测试全部通过。三组随机权重与
官方参考的最大 logits 绝对误差为 `2.563e-6`，专家编号全部一致。真实 tokenizer
的 `129280 -> 99092` 映射与官方、vLLM、SGLang 逐项一致。

`oracle.json` 记录数值对照，`text-tp8.json` / `full-tp8.json` 记录两种内存方案，
`validation.json` 记录测试数量、跳过原因和静态检查结果。
参考仓库为 vLLM `1c8ecd744d6f0c2096057d928cc19995401d9180` 与
SGLang `1442917b4a1cfdf5cc77fe0a8869aa05605e9ac7`。

下一阶段按以下顺序推进：

1. 在公开质量集和更长上下文上量化 logits/路由偏差，扩大真实模型验收；
2. 把专属状态接入 Scheduler：分页、chunked/mixed prefill、前缀复用和抢占恢复；
3. 优化批量 expert 调度与 H20 kernel，再评估 decoder bounded replay、CUDA Graph；
4. 单独接入和验证 DSpark 与视觉。

两份框架都有 decoder SWA bounded replay 路径：先写全量 source KV，再缩小后半段
的计算范围。第一阶段保留完整层计算，作为验证这些优化的基准。
