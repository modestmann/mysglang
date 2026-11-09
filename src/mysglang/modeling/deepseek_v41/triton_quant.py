"""Hopper 可运行的 MX 权重 GEMM、激活舍入和带 sink 的稀疏 attention。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ue8(code):
    code = code.to(tl.int32)
    bits = tl.where(code == 0, 0x00400000, code << 23)
    bits = tl.where(code == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _ceil_pow2(x):
    bits = x.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fp4_value(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 2,
        magnitude * 0.5,
        tl.where(
            magnitude < 4,
            1 + (magnitude - 2) * 0.5,
            tl.where(magnitude < 6, magnitude - 2, 4 + (magnitude - 6) * 2),
        ),
    )
    return tl.where((code & 8) != 0, -value, value).to(tl.float32)


@triton.jit
def _fp4_round(x):
    a = tl.minimum(tl.abs(x), 6.0)
    code = tl.full(x.shape, 0, tl.int32)
    thresholds: tl.constexpr = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for index in tl.static_range(7):
        up = (a > thresholds[index]) | ((a == thresholds[index]) & (index % 2 == 1))
        code = tl.where(up, index + 1, code)
    # 原始符号位也保留 -0，NaN 输入的行为由 scale 传播。
    code = code | ((x.to(tl.int32, bitcast=True) >> 28) & 8)
    return _fp4_value(code)


@triton.jit
def _fake_quant(X, Y, N: tl.constexpr, GROUP: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * GROUP + tl.arange(0, GROUP)
    values = tl.load(X + offsets).to(tl.float32)
    maximum = tl.max(tl.abs(values), 0)
    if MODE == 0:
        scale = _ceil_pow2(tl.maximum(maximum, 1e-4) * (1.0 / 448.0))
        rounded = tl.minimum(tl.maximum(values / scale, -448), 448).to(tl.float8e4nv).to(tl.float32)
    else:
        if MODE == 1:
            scale = _ceil_pow2(tl.maximum(maximum, 6.0 * 2.0**-126) * (1.0 / 6.0))
        else:
            scale = (
                (tl.maximum(maximum, 6.0 * 2.0**-9) * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
            )
        rounded = _fp4_round(tl.div_rn(values, scale))
    tl.store(Y + offsets, rounded * scale)


@triton.jit
def _mx_linear(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FP4: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(triton.cdiv(K, 32)):
        kk = start * 32 + k
        a = tl.load(X + rows[:, None] * K + kk[None, :], rows[:, None] < M, 0)
        if FP4:
            packed = tl.load(
                W + columns[None, :] * (K // 2) + kk[:, None] // 2, columns[None, :] < N, 0
            ).to(tl.int32)
            code = (packed >> ((kk[:, None] % 2) * 4)) & 15
            b = _fp4_value(code).to(tl.bfloat16)
            scale = tl.load(S + columns * (K // 32) + start, columns < N, 127)
        else:
            b = tl.load(W + columns[None, :] * K + kk[:, None], columns[None, :] < N, 0.0)
            b = b.to(tl.bfloat16)
            scale = tl.load(S + (columns // 32) * (K // 32) + start, columns < N, 127)
        partial = tl.dot(a, b)
        acc = acc + partial * _ue8(scale)[None, :]
    tl.store(
        Y + rows[:, None] * N + columns[None, :], acc, (rows[:, None] < M) & (columns[None, :] < N)
    )


@triton.jit
def _sparse(
    Q,
    KV,
    IDS,
    SINK,
    OUT,
    SEQ: tl.constexpr,
    KEYS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    query, batch = tl.program_id(0), tl.program_id(1)
    heads = tl.arange(0, BH)
    dims = tl.arange(0, DIM)
    q = tl.load(
        Q + ((batch * SEQ + query) * HEADS + heads[:, None]) * DIM + dims[None, :],
        heads[:, None] < HEADS,
        0,
    )
    maximum = tl.full((BH,), -1e30, tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    accumulator = tl.zeros((BH, DIM), tl.float32)
    for start in range(triton.cdiv(TOPK, BK)):
        slots = start * BK + tl.arange(0, BK)
        ids = tl.load(IDS + (batch * SEQ + query) * TOPK + slots, slots < TOPK, -1)
        values = tl.load(
            KV + (batch * KEYS + ids[:, None]) * DIM + dims[None, :], ids[:, None] >= 0, 0
        )
        scores = tl.dot(q, tl.trans(values)) * (DIM**-0.5)
        scores = tl.where(ids[None, :] >= 0, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_maximum)
        p = tl.exp(scores - new_maximum[:, None])
        accumulator = accumulator * correction[:, None] + tl.dot(p.to(tl.bfloat16), values)
        denominator = denominator * correction + tl.sum(p, 1)
        maximum = new_maximum
    sink = tl.load(SINK + heads, heads < HEADS, 0).to(tl.float32)
    denominator = denominator + tl.exp(sink - maximum)
    output = accumulator / denominator[:, None]
    tl.store(
        OUT + ((batch * SEQ + query) * HEADS + heads[:, None]) * DIM + dims[None, :],
        output,
        heads[:, None] < HEADS,
    )


def fake_quantize_cuda(x: torch.Tensor, kind: str) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("activation kernel requires CUDA FP32/BF16")
    x = x.contiguous()
    group = 16 if kind == "kv4" else 32
    output = torch.empty_like(x, dtype=torch.bfloat16)
    _fake_quant[(x.numel() // group,)](
        x, output, x.numel(), group, {"fp8": 0, "index4": 1, "kv4": 2}[kind], num_warps=1
    )
    return output


def quantized_linear_cuda(x, weight, scale, kind):
    if not x.is_cuda or x.dtype != torch.bfloat16 or kind not in {"fp4", "fp8"}:
        raise ValueError("MX kernel requires CUDA BF16 input and FP4/FP8 weights")
    columns = weight.shape[0]
    depth = weight.shape[1] * (2 if kind == "fp4" else 1)
    if x.shape[-1] != depth or depth % 32:
        raise ValueError("MX GEMM depth mismatch/alignment")
    shape = (*x.shape[:-1], columns)
    x = x.reshape(-1, depth).contiguous()
    output = torch.empty((len(x), columns), dtype=torch.bfloat16, device=x.device)
    _mx_linear[(triton.cdiv(len(x), 16), triton.cdiv(columns, 64))](
        x,
        weight,
        scale,
        output,
        len(x),
        columns,
        depth,
        kind == "fp4",
        16,
        64,
        num_warps=4,
        num_stages=2,
    )
    return output.reshape(shape)


def sparse_attention_cuda(q, kv, indices, sink):
    batch, length, heads, dim = q.shape
    if not q.is_cuda or q.dtype != torch.bfloat16 or dim & (dim - 1):
        raise ValueError("sparse attention requires CUDA BF16 and a power-of-two head dimension")
    output = torch.empty_like(q)
    _sparse[(length, batch)](
        q.contiguous(),
        kv.contiguous(),
        indices.contiguous(),
        sink,
        output,
        length,
        kv.shape[1],
        heads,
        dim,
        indices.shape[-1],
        max(16, triton.next_power_of_2(heads)),
        32,
        num_warps=4,
        num_stages=2,
    )
    return output
