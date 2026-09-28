from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mysglang.config import Fp8QuantConfig

from .fp8 import FP8_DTYPE, ceil_div, fp8_linear, quantize_block_fp8


@dataclass(frozen=True)
class TensorParallelContext:
    """The rank-local view of one tensor-parallel process group.

    Model layers only depend on this small boundary. Process creation, device
    placement and batch-plan broadcast belong to the worker runtime built on top.
    """

    rank: int = 0    # 卡的序号，一张 GPU 对应一个 Python 进程
    world_size: int = 1 
    """
    参加这组 Tensor Parallel 的进程数，也就是 TP 大小。
     world_size = 1   # 不进行 TP
     world_size = 2   # 两张卡共同运行一个模型
     world_size = 4   # 四张卡共同运行一个模型
    """
    process_group: dist.ProcessGroup | None = None
    """
    PyTorch Distributed 的通信组。规定哪些进程互相通信。
    如果只有 4 张卡并且 TP=4，可以使用默认全局通信组，此时它可以是 None。
以后如果是 8 张卡，分成两个 TP=4 的副本：
  TP group 0: GPU 0、1、2、3
  TP group 1: GPU 4、5、6、7
这时候就必须用不同的 process_group，避免 GPU 0 去和 GPU 4 做 all_reduce。
  因此它是给将来的：
  - TP + DP；
  - TP + EP；
  - 多机多卡；
  预留的通信边界。
    """

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("tensor parallel world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("tensor parallel rank must be within world_size")
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError("tensor parallelism requires an initialized process group")
        #构建通信参数
    @classmethod
    def from_distributed(
        cls,
        process_group: dist.ProcessGroup | None = None,
    ) -> TensorParallelContext:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        return cls(
            rank=dist.get_rank(process_group),
            world_size=dist.get_world_size(process_group),
            process_group=process_group,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def local_size(self, global_size: int, name: str) -> int:
        if global_size % self.world_size:
            raise ValueError(f"{name}={global_size} must be divisible by TP={self.world_size}")
        return global_size // self.world_size

    def shard(self, global_size: int, name: str) -> slice:
        local_size = self.local_size(global_size, name)
        start = self.rank * local_size
        return slice(start, start + local_size)

    # Attention/MLP 的出口投影先算各 rank 的局部贡献，再 all-reduce 求和。
    # Dense Attention 和 MLP 都可做 TP；MoE 是多组 MLP，通常使用 EP，并可选叠加 TP。
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor, group=self.process_group)
        return tensor

    def all_to_all_variable(
        self,
        tensor: torch.Tensor,
        *,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> torch.Tensor:
        """Exchange variable-size chunks along the first tensor dimension."""
        if len(input_split_sizes) != self.world_size or len(output_split_sizes) != self.world_size:
            raise ValueError("all-to-all split sizes must contain one value per rank")
        if sum(input_split_sizes) != tensor.size(0):
            raise ValueError("all-to-all input split sizes do not match the tensor")
        if not self.enabled:
            if output_split_sizes != input_split_sizes:
                raise ValueError("single-rank all-to-all split sizes must match")
            return tensor
        output = tensor.new_empty((sum(output_split_sizes), *tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.process_group,
        )
        return output

    def all_gather_variable_first_dim(
        self,
        tensor: torch.Tensor,
        sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """Gather uneven token shards and restore rank-order concatenation on every rank."""
        if len(sizes) != self.world_size or sizes[self.rank] != tensor.size(0):
            raise ValueError("all-gather sizes do not match the local token shard")
        if not self.enabled:
            return tensor
        padded_size = max(sizes)
        padded = tensor.new_zeros((padded_size, *tensor.shape[1:]))
        padded[: tensor.size(0)] = tensor
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded, group=self.process_group)
        return torch.cat(
            [rank_tensor[:rank_size] for rank_tensor, rank_size in zip(gathered, sizes)],
            dim=0,
        )

# ColumnParallelLinear：用分片权重直接产生分片输出。
# RowParallelLinear：用分片输入和分片权重计算局部贡献，再 all-reduce 求和。
class ColumnParallelLinear(nn.Module):
    """Shard each logical output segment independently across TP ranks.

    Independent segments matter for fused Q/K/V and gate/up weights: every rank
    stores ``[Q_rank, K_rank, V_rank]`` rather than one arbitrary contiguous slice
    of the globally fused tensor.

    ``shard_counts`` lets a segment be split into fewer shards than TP ranks. With
    2 KV heads and TP=4, K/V use two shards: ranks 0/1 replicate KV head 0 and ranks
    2/3 replicate KV head 1, matching the GQA groups of their local query heads.

    With ``quant`` the weight is stored as block FP8 plus one float32 dequantization
    scale per ``[block_n, block_k]`` tile, sliced along the same segment boundaries.
    """

    keep_dtype_parameters: tuple[str, ...] = ()

    def __init__(
        self,
        in_features: int,
        output_sizes: tuple[int, ...],
        tensor_parallel: TensorParallelContext,
        *,
        shard_counts: tuple[int, ...] | None = None,
        quant: Fp8QuantConfig | None = None,
    ) -> None:
        super().__init__()
        if not output_sizes or any(size <= 0 for size in output_sizes):
            raise ValueError("column-parallel output segments must be positive")
        self.in_features = in_features
        self.global_output_sizes = output_sizes
        self.tensor_parallel = tensor_parallel
        world_size = tensor_parallel.world_size
        if shard_counts is None:
            shard_counts = (world_size,) * len(output_sizes)
        if len(shard_counts) != len(output_sizes):
            raise ValueError("shard_counts must contain one value per output segment")
        for count, size in zip(shard_counts, output_sizes):
            if count <= 0 or world_size % count:
                raise ValueError("segment shard count must divide the TP world size")
            if size % count:
                raise ValueError(
                    f"column-parallel output size={size} must be divisible by {count} shards"
                )
        self.shard_counts = tuple(shard_counts)
        self.local_output_sizes = tuple(
            size // count for size, count in zip(output_sizes, self.shard_counts)
        )
        self.quant = quant
        out_features = sum(self.local_output_sizes)
        if quant is None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
        else:
            block_n, block_k = quant.weight_block_size
            if len(output_sizes) > 1 or tensor_parallel.enabled:
                # 融合或切分后的每段必须从 scale tile 边界开始，否则一个 tile 会跨两段。
                if any(size % block_n for size in self.local_output_sizes):
                    raise ValueError(
                        "FP8 column-parallel segments must be multiples of the "
                        f"{block_n}-row weight block: {self.local_output_sizes}"
                    )
            self.keep_dtype_parameters = ("weight", "weight_scale_inv")
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=FP8_DTYPE),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    ceil_div(out_features, block_n),
                    ceil_div(in_features, block_k),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.quant is None:
            nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
            return
        full = torch.empty(self.weight.shape, device=self.weight.device)
        nn.init.kaiming_uniform_(full, a=sqrt(5))
        weight, scale = quantize_block_fp8(full, self.quant.weight_block_size)
        with torch.no_grad():
            self.weight.copy_(weight)
            self.weight_scale_inv.copy_(scale)

    def local_segment(self, index: int) -> slice:
        start = sum(self.local_output_sizes[:index])
        return slice(start, start + self.local_output_sizes[index])

    def source_segment(self, index: int) -> slice:
        """Rows of the global (unfused) segment ``index`` owned by this rank."""
        count = self.shard_counts[index]
        local_size = self.local_output_sizes[index]
        shard_index = self.tensor_parallel.rank // (self.tensor_parallel.world_size // count)
        start = shard_index * local_size
        return slice(start, start + local_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant is None:
            return F.linear(x, self.weight)
        return fp8_linear(x, self.weight, self.weight_scale_inv, self.quant.weight_block_size)


class RowParallelLinear(nn.Module):
    """Consume a rank-local input shard and sum partial outputs across ranks."""

    keep_dtype_parameters: tuple[str, ...] = ()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel: TensorParallelContext,
        *,
        quant: Fp8QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.global_in_features = in_features
        self.local_in_features = tensor_parallel.local_size(
            in_features,
            "row-parallel input size",
        )
        self.out_features = out_features
        self.tensor_parallel = tensor_parallel
        self.quant = quant
        if quant is None:
            self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        else:
            block_n, block_k = quant.weight_block_size
            if tensor_parallel.enabled and self.local_in_features % block_k:
                raise ValueError(
                    "FP8 row-parallel input shards must be multiples of the "
                    f"{block_k}-column weight block: {self.local_in_features}"
                )
            self.keep_dtype_parameters = ("weight", "weight_scale_inv")
            self.weight = nn.Parameter(
                torch.empty(out_features, self.local_in_features, dtype=FP8_DTYPE),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    ceil_div(out_features, block_n),
                    ceil_div(self.local_in_features, block_k),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.quant is None:
            nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
            return
        full = torch.empty(self.weight.shape, device=self.weight.device)
        nn.init.kaiming_uniform_(full, a=sqrt(5))
        weight, scale = quantize_block_fp8(full, self.quant.weight_block_size)
        with torch.no_grad():
            self.weight.copy_(weight)
            self.weight_scale_inv.copy_(scale)

    @property
    def source_columns(self) -> slice:
        return self.tensor_parallel.shard(
            self.global_in_features,
            "row-parallel input size",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.local_in_features:
            raise ValueError(
                "row-parallel input has the wrong local width: "
                f"{x.size(-1)} != {self.local_in_features}"
            )
        if self.quant is None:
            output = F.linear(x, self.weight)
        else:
            output = fp8_linear(
                x,
                self.weight,
                self.weight_scale_inv,
                self.quant.weight_block_size,
            )
        return self.tensor_parallel.all_reduce(output)
##每一个 Dense Decoder Layer 有两次 all_reduce  attention+MLP
"""
模块                                使用的并行 Linear
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   QKV projection                      ColumnParallelLinear
  ──────────────────────────────────  ─────────────────────────────
   Attention 核心计算 softmax(QKᵀ)V    每张卡本地计算，不是 Linear
  ──────────────────────────────────  ─────────────────────────────
   Attention 的 O projection           RowParallelLinear
  ──────────────────────────────────  ─────────────────────────────
   FFN 的 gate/up projection           ColumnParallelLinear
  ──────────────────────────────────  ─────────────────────────────
   SiLU(gate) * up                     每张卡本地计算
  ──────────────────────────────────  ─────────────────────────────
   FFN 的 down projection              RowParallelLinear

  整体流程：

  Attention:

  完整 hidden
      ↓
  ColumnParallelLinear：QKV projection
      ↓
  各卡得到自己的 Q/K/V heads
      ↓
  各卡本地计算 Attention
      ↓
  RowParallelLinear：O projection
      ↓
  all-reduce，恢复完整 hidden

  FFN:

  完整 hidden
      ↓
  ColumnParallelLinear：gate/up projection
      ↓
  各卡得到一部分 intermediate
      ↓
  各卡本地计算 SiLU(gate) * up
      ↓
  RowParallelLinear：down projection
      ↓
  all-reduce，恢复完整 hidden


  
  完整 hidden x：每张卡都有一份
               │
               ▼
  ColumnParallelLinear
               │
       各卡直接计算自己的 QKV
               │
        ┌──────┴──────┐
        ▼             ▼
  rank 0: Q0 K0 V0  rank 1: Q1 K1 V1
        │             │
        ▼             ▼
    Attention 0    Attention 1
        │             │
        ▼             ▼
      output0        output1
        │             │
        ▼             ▼
       RowParallelLinear
        │             │
        ▼             ▼
     partial0       partial1
        └──────┬──────┘
               ▼
         all_reduce 求和
               ▼
         完整 hidden output


"""
