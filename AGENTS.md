# MySGLang 项目指南

## 工作目录与环境

- 本机仓库：`/home/ubuntu/yqf-intern/mysglang`；父工作目录：`/home/ubuntu/yqf-intern`。
- 使用父目录的 uv 环境：`source /home/ubuntu/yqf-intern/env.sh`。不要自动激活 Conda。
- 项目采用 `src/` 布局；当前未安装为 editable package，运行时显式使用 `PYTHONPATH=src`。
- 父目录的 `pyproject.toml` / `uv.lock` 属于独立的算子环境。不要把它们当作本仓库的依赖声明，也不要用父目录的 `uv sync` 安装本仓库 extras。
- 与 205 传输环境或依赖时走内网 `10.0.119.205`，取消代理变量并禁用 SSH ProxyCommand/ProxyJump。
- 开始工作先检查当前分支和 `git status`。`master` 与 `perf-improvements` 的功能不同，以当前代码为准。

## 结构与阅读入口

- `core/request.py`：请求状态、采样参数、增量输出。
- `scheduler/`：准入、chunked/mixed prefill、decode、抢占重算、n-gram/MTP 投机、TP 控制面。
- `cache/`：分页 KV、Radix 前缀、DeltaNet 状态池、混合模型快照及 MTP companion KV。
- `modeling/`：Qwen3/Qwen3MoE、Qwen3.5/3.6、视觉塔、TP 分片加载、FP8、attention 和 CUDA Graph。
- `serving/` 与 `server.py`：惰性 session、单一调度 worker、HTTP/SSE、reasoning/tool-call 解析。
- `tests/helpers.py`：小随机模型、完整前缀重算 oracle、HF 权重导出。
- 当前架构先读 `README.md`、`docs/design.md` 和 `docs/roadmap.md`。`CLAUDE.md`、`HANDOFF.md` 包含历史机器路径和旧功能状态，先与当前代码核对。

## 验证命令

在仓库根目录激活上述环境后运行：

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest tests.test_cache tests.test_engine -v
PYTHONPATH=src python -m mysglang.cli --help
PYTHONPATH=src python -m mysglang.server --help
```

测试使用 unittest，也可由 pytest 执行。`serve`、`model`、GPU 后端和真实 tokenizer 的测试按依赖或资源条件跳过；报告通过数与跳过原因。CPU/Gloo 的 TP 测试需要本机 socket，沙箱禁网失败不能直接判断为产品缺陷。GPU 测试前检查设备空闲情况，优先使用小模型，并记录 backend、精度、TP 和硬件。

## 修改时必须保持的不变量

- session 首次消费才入队；单一 worker 修改调度器和缓存。结束、abort、异常应释放所有请求资源。
- KV reservation 与物理页分配分离；引用的前缀必须锁住；只回收无引用缓存。生命周期修改后运行 `check_integrity()`。
- DeltaNet 快照必须与对应整页 KV 前缀一致；图片前缀必须包含像素/网格指纹，不能只按占位 token 匹配。
- TP 所有 rank 的 batch plan、采样结果、抢占和 collective 顺序一致。
- 投机回滚同时处理目标 KV、DeltaNet 状态和 MTP draft KV。两步之间 draft 与目标长度一致。
- CUDA Graph 的 buffer 地址与形状稳定；补齐行只能写 scratch page/slot；逻辑长度只提交一次。
- 验证调度、缓存及回滚先用 FP32 + Torch oracle。BF16 下不同 kernel 的分叉需要区分数值差异与状态错误。
- 改 checkpoint 映射时覆盖 fused 参数各段、TP 切分、FP8 scale 和权重完整性，不修改共享模型文件。

## 代码与交付

沿用四空格、类型标注和中文文档/注释；Ruff 配置在 `pyproject.toml`（E/F/I/UP，100 列，Python 3.10 语法目标）。避免为格式统一而改动无关核心代码。审查发现应包含具体触发条件、影响、代码位置和验证证据；不把已有 benchmark 结果视作当前机器上的验收结果。
