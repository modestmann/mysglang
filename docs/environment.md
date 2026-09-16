# 开发环境

## 统一环境

MySGLang 的本地开发和验收统一使用：

```bash
source ~/nano-vllm/.venv/bin/activate
```

2026-09-16 的只读盘点结果：

| 依赖 | 版本/状态 | 用途 |
|---|---|---|
| Python | 3.12.13 | 项目解释器 |
| PyTorch | 2.13.0，CUDA 13.0 build | tensor、GPU SDPA reference attention、tiny smoke test |
| Transformers | 5.16.1 | 后续配置解析、tokenizer、正确性 oracle |
| SafeTensors | 0.8.0 | 后续真实权重流式加载 |
| Hugging Face Hub | 1.29.0 | checkpoint 获取 |
| FlashAttention | 2.8.3.post1 | 后续 GPU 高性能 Attention backend |
| Triton | 3.7.1 | 后续 MoE/辅助 kernel 实验 |
| NumPy | 2.5.2 | benchmark 数据处理 |
| FastAPI | 0.141.1 | 第 5 章 HTTP schema 与 ASGI app |
| Uvicorn | 0.53.0 | 第 5 章本地 HTTP server |
| httpx | 0.28.1 | 第 5 章异步 ASGI/并发客户端测试 |
| PyZMQ | 未安装 | 多进程章节再决定是否采用，不提前引入 |
| pytest/ruff | 未安装 | 当前 unittest 无需它们；需要时安装 `.[dev]` |

当前会话里 `torch.cuda.is_available()` 为 `False`。这表示此会话暂时不能运行 CUDA 测试，不代表安装的 PyTorch 没有 CUDA 支持；它报告的 build CUDA 版本是 13.0。

## 为什么不现在安装全部依赖

逐章项目应让依赖跟功能一起出现：

- M1-M2：只依赖 PyTorch，CPU 可运行轻量 smoke test；模型级对齐后续以 Hugging Face 为 oracle；
- HTTP 章节：使用已有的 `fastapi`、`uvicorn` 和 `httpx`；
- 真实模型章节：使用已有 `transformers`、`safetensors`；
- GPU Attention 章节：使用已有 `flash-attn`，并检查实际 GPU compute capability；
- 多进程章节：根据当时设计选择标准库 queue、PyZMQ 或其他 IPC。

这样能区分“代码错误”和“环境/编译错误”，也避免课程开始时安装一套暂时用不到的生产依赖。

## 可选依赖声明

`pyproject.toml` 中 extras 表示功能边界，不要求现在执行：

```bash
uv pip install -e ".[model]"       # Transformers/SafeTensors
uv pip install -e ".[flash-attn]"  # FlashAttention/Triton
uv pip install -e ".[serve]"       # FastAPI/Uvicorn
uv pip install -e ".[dev]"         # pytest/ruff
```

我们不会在已有环境上盲目运行这些命令。进入对应章节前先检查版本和 import，再只补缺失依赖。
