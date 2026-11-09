"""原生 MXFP4/MXFP8 数值语义及独立 PyTorch 对照路径。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_ue8m0(raw: torch.Tensor) -> torch.Tensor:
    """0 是 2**-127（不是零），255 是 NaN；其余字节是带偏置指数。"""
    codes = raw.view(torch.uint8).to(torch.int32)
    bits = codes << 23
    bits = torch.where(codes == 0, 0x00400000, bits)
    bits = torch.where(codes == 255, 0x7FC00000, bits)
    return bits.contiguous().view(torch.float32)


def ceil_power_of_two(x: torch.Tensor) -> torch.Tensor:
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent << 23).contiguous().view(torch.float32)


def fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """E2M1 RN-even，偶数位置放低 nibble；保留舍入为零后的符号。"""
    bounds = x.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32)
    magnitude = torch.nan_to_num(x.float().abs(), nan=0.0, posinf=6.0).clamp_max(6)
    lower = torch.bucketize(magnitude.contiguous(), bounds)
    tie = (magnitude.unsqueeze(-1) == bounds).any(-1)
    code = torch.where(tie, lower + (lower & 1), lower).to(torch.uint8)
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def fp4_values(codes: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(FP4_VALUES, dtype=torch.float32, device=codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -values, values)


def dequantize_weight(
    weight: torch.Tensor, scale: torch.Tensor, kind: str, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    scales = decode_ue8m0(scale)
    if kind == "fp4":
        packed = weight.view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        values = fp4_values(codes).unflatten(-1, (-1, 32))
        return (values * scales.unsqueeze(-1)).flatten(-2).to(dtype)
    if kind == "fp8":
        expanded = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
        return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(dtype)
    if kind == "engram":
        values = weight.float().unflatten(-1, (-1, 32))
        return (values * scales.unsqueeze(-1)).flatten(-2).to(dtype)
    raise ValueError(f"unknown quantization kind: {kind}")


def quantize_weight(weight: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """生成小型测试 checkpoint；真实加载路径不重新量化共享权重。"""
    rows, columns = weight.shape
    if columns % 32 or (kind == "fp8" and rows % 32):
        raise ValueError("quantized fixture dimensions must align to 32")
    x = weight.float()
    if kind == "fp8":
        groups = x.reshape(rows // 32, 32, columns // 32, 32).permute(0, 2, 1, 3)
        scale = ceil_power_of_two(groups.abs().amax((-1, -2)).clamp_min(1e-4) * (1 / 448))
        values = (groups / scale[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        values = values.permute(0, 2, 1, 3).reshape(rows, columns).contiguous()
    else:
        groups = x.unflatten(-1, (-1, 32))
        maximum = 6 if kind == "fp4" else 448
        scale = ceil_power_of_two(groups.abs().amax(-1).clamp_min(1e-4) * (1 / maximum))
        if kind == "fp4":
            codes = fp4_codes(groups / scale.unsqueeze(-1)).flatten(-2)
            values = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
        elif kind == "engram":
            values = (groups / scale.unsqueeze(-1)).clamp(-448, 448)
            values = values.flatten(-2).to(torch.float8_e4m3fn)
        else:
            raise ValueError(kind)
    return values, scale.to(torch.float8_e8m0fnu)


def fake_quantize(x: torch.Tensor, kind: str, *, backend: str = "torch") -> torch.Tensor:
    """返回 BF16 的量化后数值：A8/window、indexer FP4、compressed KV FP4。"""
    group_size = 16 if kind == "kv4" else 32
    if kind not in {"fp8", "index4", "kv4"} or x.shape[-1] % group_size:
        raise ValueError("invalid activation quantization kind or group alignment")
    if not x.numel():
        return x.to(torch.bfloat16)
    if backend == "triton":
        from .triton_quant import fake_quantize_cuda

        return fake_quantize_cuda(x, kind)
    if backend != "torch":
        raise ValueError(f"unsupported quantization backend: {backend}")
    groups = x.float().reshape(-1, group_size)
    maximum = groups.abs().amax(-1, keepdim=True)
    if kind == "fp8":
        scale = ceil_power_of_two(maximum.clamp_min(1e-4) * (1 / 448))
        values = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        if kind == "kv4":
            scale = (maximum.clamp_min(6 * 2**-9) * (1 / 6)).to(torch.float8_e4m3fn).float()
        else:
            scale = ceil_power_of_two(maximum.clamp_min(6 * 2**-126) * (1 / 6))
        values = fp4_values(fp4_codes((groups / scale).clamp(-6, 6)))
    return (values * scale).reshape(x.shape).to(torch.bfloat16)


def quantized_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kind: str,
    *,
    backend: str = "triton",
) -> torch.Tensor:
    x = fake_quantize(x, "fp8", backend=backend)
    if backend == "triton":
        from .triton_quant import quantized_linear_cuda

        return quantized_linear_cuda(x, weight, scale, kind)
    return F.linear(x, dequantize_weight(weight, scale, kind))


def sparse_attention_torch(q, kv, indices, sink):
    """独立 Torch online softmax 对照；归一化前的 P 转 BF16，PV 累加为 FP32。"""
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    maximum = torch.full(q.shape[:-1], -1e30, dtype=torch.float32, device=q.device)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros_like(q, dtype=torch.float32)
    for begin in range(0, indices.shape[-1], 32):
        ids = indices[..., begin : begin + 32]
        picked = kv[batch, ids.clamp_min(0).long()]
        scores = torch.einsum("bshd,bskd->bshk", q.float(), picked.float()) * q.shape[-1] ** -0.5
        scores = scores.masked_fill(ids[:, :, None, :] < 0, -torch.inf)
        next_max = torch.maximum(maximum, scores.amax(-1))
        correction = torch.exp(maximum - next_max)
        p = torch.exp(scores - next_max.unsqueeze(-1))
        accumulator = accumulator * correction.unsqueeze(-1) + torch.einsum(
            "bshk,bskd->bshd", p.to(q.dtype).float(), picked.float()
        )
        denominator = denominator * correction + p.sum(-1)
        maximum = next_max
    denominator = denominator + torch.exp(sink.float().view(1, 1, -1) - maximum)
    return (accumulator / denominator.unsqueeze(-1)).to(q.dtype)
