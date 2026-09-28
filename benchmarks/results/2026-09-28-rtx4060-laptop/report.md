# 2026-09-28 RTX 4060 Laptop：Radix 前缀缓存收益

`benchmarks/benchmark_prefix_cache.py`，Qwen3-0.6B、BF16、FlashAttention 2（page size 256）。
8 个请求共享同一段前缀，各自带 64 token 的不同后缀，逐个执行，每个生成 16 token（greedy）。
cold 在每个请求前清空前缀缓存；warm 只在开始时清空一次。

| 共享前缀 | cold TTFT p50 | warm TTFT（命中）p50 | 加速 | 实际 prefill token（warm / cold） | 命中率 | greedy 输出一致 |
|---:|---:|---:|---:|---:|---:|---|
| 1024 | 68.77 ms | 33.94 ms | 2.03× | 1536 / 8704 | 82.4% | 是 |
| 3072 | 232.37 ms | 37.74 ms | 6.16× | 3584 / 25088 | 85.7% | 是 |

- 命中后 TTFT 基本不随前缀变长（34 → 38 ms），剩下的是 64 token 后缀的 prefill 和
  eager 模式下的逐层发射开销；因此前缀越长，加速越大。
- warm 的第一个请求与 cold 相同（66.54 / 233.23 ms）：它负责把前缀写入缓存。
- 命中率上限是 (8 - 1) × 前缀 / 总 prompt token；Radix 按 256 token 整页匹配，
  不足一页的尾部不会被复用。

原始数据：[`prefix-cache.jsonl`](prefix-cache.jsonl)。
