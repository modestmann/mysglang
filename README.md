# MySGLang

MySGLang 是一个以测试驱动、逐章演进的教学型 LLM 推理框架。目标不是复制完整
SGLang，而是亲手实现一条可解释、可验证、最后能在云端 GPU 跑 Qwen3 MoE 的推理链路。

每章遵循同一循环：

1. 先运行 reference implementation，建立正确性基线；
2. 写最小实现，并用单元测试验证；
3. 加入 benchmark，记录延迟、吞吐或显存变化；
4. 再进入下一项优化，避免同时改变多个变量。

## 当前状态

- [x] M0：项目骨架、路线图和验收规则
- [x] M1：纯 PyTorch 的 decoder-only forward 与 greedy generation
- [x] M2：请求生命周期与增量 token 事件协议
- [x] M3：逐层 KV Cache，证明 decode 从重复计算前缀变为只计算新 token
- [x] M4：HTTP 服务与并发请求
- [x] M5：continuous batching 与公平调度
- [x] M6：paged KV Cache 与显存池
- [ ] M7：Radix prefix cache（当前学习：[第 8 章讲义](docs/08_radix_prefix_cache.md)）
- [ ] M8：FlashAttention backend 与 CUDA Graph
- [ ] M9：多进程与 tensor parallelism
- [ ] M10：Qwen3 MoE、真实权重加载与云端验收
- [ ] M11：评测套件与回归报告
- [ ] M12（可选）：speculative decoding / host KV cache

完整设计和各章验收标准见 [docs/roadmap.md](docs/roadmap.md)。

## 第一次运行

本项目统一复用 `nano-vllm` 的 Python 3.12 环境：

```bash
cd /home/sheep/mysglang
source ~/nano-vllm/.venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/01_forward_and_generate.py
```

当前章节不需要重新安装依赖，也不要求 pytest。需要导入包时暂时使用
`PYTHONPATH=src`；等公共 API 稳定后再做 editable install。

当前环境清单和后续 extras 说明见 [docs/environment.md](docs/environment.md)。

也可以不激活环境，直接使用解释器绝对路径：

```bash
PYTHONPATH=src ~/nano-vllm/.venv/bin/python examples/01_forward_and_generate.py
PYTHONPATH=src ~/nano-vllm/.venv/bin/python -m unittest discover -s tests -v
```

## 当前示例做了什么

`TinyCausalLM` 保留 dense Qwen3 的关键结构，但刻意使用很小的随机权重：

```text
token ids
  -> embedding
  -> [RMSNorm -> Q/K/V -> Q/K RMSNorm -> RoPE -> causal attention -> O projection
      -> RMSNorm -> gated MLP] x N
  -> RMSNorm -> LM head -> logits
  -> argmax -> next token
```

当前 Radix Scheduler 在 paged pool 之上保留已完成请求的完整 KV pages，用压缩 Radix
Tree 完成 page-aligned prefix match、引用保护和 LRU eviction。PyTorch reference attention
仍会 gather/pad 历史 K/V；第 9 章将让 FlashAttention backend 直接消费物理 pool、
block tables 和 sequence lengths。

## 范围约束

- 高性能 Attention 只优先适配 FlashAttention；始终保留 PyTorch reference backend。
- 本地小显存/无 GPU 环境使用 tiny config 做正确性验证；真实 Qwen3 MoE 在云 GPU 验收。
- 第一版只做文本生成，不做 VLM、量化、LoRA、复杂 grammar。
- 不追求一次写完。每个里程碑必须先通过 correctness test，再做性能优化。
