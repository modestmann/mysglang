from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


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
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: tuple[int, ...],
        tensor_parallel: TensorParallelContext,
    ) -> None:
        super().__init__()
        if not output_sizes or any(size <= 0 for size in output_sizes):
            raise ValueError("column-parallel output segments must be positive")
        self.in_features = in_features
        self.global_output_sizes = output_sizes
        self.tensor_parallel = tensor_parallel
        self.local_output_sizes = tuple(
            tensor_parallel.local_size(size, "column-parallel output size")
            for size in output_sizes
        )
        self.weight = nn.Parameter(torch.empty(sum(self.local_output_sizes), in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

    def local_segment(self, index: int) -> slice:
        start = sum(self.local_output_sizes[:index])
        return slice(start, start + self.local_output_sizes[index])

    def source_segment(self, index: int) -> slice:
        return self.tensor_parallel.shard(
            self.global_output_sizes[index],
            "column-parallel output size",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Consume a rank-local input shard and sum partial outputs across ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel: TensorParallelContext,
    ) -> None:
        super().__init__()
        self.global_in_features = in_features
        self.local_in_features = tensor_parallel.local_size(
            in_features,
            "row-parallel input size",
        )
        self.out_features = out_features
        self.tensor_parallel = tensor_parallel
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

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
        return self.tensor_parallel.all_reduce(F.linear(x, self.weight))
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
