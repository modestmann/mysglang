"""

                      TP 多进程
                         │
            ┌────────────┴────────────┐
            │                         │
       控制面 Gloo                数据面 NCCL
            │                         │
   add / step / shutdown         GPU hidden tensor
   request_id / 页表             Attention all-reduce
   BatchPlan / token ID          MLP all-reduce

torchrun --nproc-per-node=4 -m mysglang.cli

  大致等价于外部替你执行了四次：

  RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 python -m mysglang.cli
  RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 python -m mysglang.cli
  RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 python -m mysglang.cli
  RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 python -m mysglang.cli

  代码中显式出现的是“加入通信组”：

  dist.init_process_group(backend="nccl", init_method="env://")

  它的意思不是：

  请 NCCL 创建四个进程

  而是：

  当前进程已经由 torchrun 创建好了；
  请根据 RANK/WORLD_SIZE，让当前进程加入 NCCL 通信组。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from mysglang.modeling import (
    FlashAttentionBackend,
    Qwen3ForCausalLM,
    TensorParallelContext,
    TorchAttentionBackend,
)
from mysglang.scheduler import SchedulerConfig, TensorParallelScheduler
from mysglang.serving import GenerationService
from mysglang.tokenizer import HuggingFaceTokenizer


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[3] / "KuiperLLama/artifacts/qwen3-0.6b/hf-source"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a local Qwen3 checkpoint")
    parser.add_argument("--model", type=Path, default=_default_model_path())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--backend", choices=("auto", "flash", "torch"), default="auto")
    parser.add_argument("--moe-dispatch", choices=("naive", "sorted"), default="sorted")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--prompt", help="run one prompt and exit instead of opening the REPL")
    parser.add_argument("--num-pages", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument(
        "--prefill-token-budget",
        type=int,
        help="new prompt tokens admitted per scheduler step; defaults to KV capacity",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="capture the greedy B=1 decode")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="expected torchrun world size; inferred from WORLD_SIZE when omitted",
    )
    return parser


def _resolve_dtype(name: str | None, device: torch.device) -> torch.dtype:
    if name is None:
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _make_backend(name: str, device: torch.device):
    use_flash = name == "flash" or (
        name == "auto"
        and device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
    )
    return FlashAttentionBackend() if use_flash else TorchAttentionBackend()


@dataclass(frozen=True)
class _DistributedLaunch:
    world_size: int
    rank: int
    local_rank: int


@dataclass
class _LoadedRuntime:
    service: GenerationService | None
    scheduler: TensorParallelScheduler | None = None
    owns_process_group: bool = False

#读多进程参数
def _read_distributed_launch(args: argparse.Namespace) -> _DistributedLaunch:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE, RANK and LOCAL_RANK must be integers") from exc
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be within WORLD_SIZE")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")

    expected = args.tensor_parallel_size
    if expected is not None:
        if expected <= 0:
            raise ValueError("--tensor-parallel-size must be positive")
        if expected != world_size:
            raise ValueError(
                f"--tensor-parallel-size={expected} but torchrun launched "
                f"WORLD_SIZE={world_size}; start exactly one process per TP rank"
            )
    return _DistributedLaunch(world_size, rank, local_rank)


def _rank_device(name: str, launch: _DistributedLaunch) -> torch.device:
    requested = torch.device(name)
    if launch.world_size == 1:
        return requested
    if requested.type == "cuda":
        if requested.index is not None:
            raise ValueError(
                "do not put a CUDA index in --device under torchrun; LOCAL_RANK selects it"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA tensor parallelism requested but CUDA is unavailable")
        if launch.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={launch.local_rank} has no visible CUDA device; "
                f"only {torch.cuda.device_count()} device(s) are visible"
            )
        torch.cuda.set_device(launch.local_rank)
        return torch.device("cuda", launch.local_rank)
    if requested.type != "cpu":
        raise ValueError("multi-process tensor parallelism supports only --device cuda or cpu")
    return requested


def _scheduler_config(args: argparse.Namespace, backend) -> SchedulerConfig:
    page_size = 256 if isinstance(backend, FlashAttentionBackend) else 16
    prefill_token_budget = args.prefill_token_budget or args.num_pages * page_size
    graph_batch_sizes = getattr(args, "decode_cuda_graph_batch_sizes", None)
    if graph_batch_sizes is None:
        graph_batch_sizes = (1,) if args.cuda_graph else ()
    return SchedulerConfig(
        max_running_requests=args.max_running_requests,
        prefill_token_budget=prefill_token_budget,
        num_pages=args.num_pages,
        page_size=page_size,
        decode_cuda_graph_batch_sizes=graph_batch_sizes,
    )


def _load_runtime(args: argparse.Namespace) -> _LoadedRuntime:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    launch = _read_distributed_launch(args)
    device = _rank_device(args.device, launch)
    dtype = _resolve_dtype(args.dtype, device)
    backend = _make_backend(args.backend, device)
    if isinstance(backend, FlashAttentionBackend) and dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError("FlashAttention requires --dtype float16 or bfloat16")
    if launch.world_size > 1 and args.cuda_graph:
        raise ValueError("TP Decode CUDA Graph capture is not implemented yet")

    owns_process_group = False
    tensor_parallel = TensorParallelContext()
    control_group: dist.ProcessGroup | None = None
    if launch.world_size > 1:
        if not dist.is_available():
            raise RuntimeError("this PyTorch build does not provide torch.distributed")
        model_backend = "nccl" if device.type == "cuda" else "gloo"
        if model_backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("this PyTorch build does not provide NCCL")
        if model_backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("this PyTorch build does not provide Gloo")
        if dist.is_initialized():
            if dist.get_world_size() != launch.world_size or dist.get_rank() != launch.rank:
                raise RuntimeError("existing process group disagrees with torchrun environment")
            if str(dist.get_backend()) != model_backend:
                raise RuntimeError(
                    f"existing process group uses {dist.get_backend()}, expected {model_backend}"
                )
        else:
            dist.init_process_group(backend=model_backend, init_method="env://")
            owns_process_group = True
        tensor_parallel = TensorParallelContext.from_distributed()
        # NCCL carries model tensors; a separate CPU/Gloo group carries Python
        # scheduler commands and BatchPlan objects. CPU tests can reuse the default group.
        if model_backend == "nccl":
            control_group = dist.new_group(backend="gloo")

    print(
        f"[TP rank {launch.rank}/{launch.world_size}] Loading {model_path} on {device} "
        f"as {dtype} with {backend.name} ...",
        flush=True,
    )
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device=device,
        attention_backend=backend,
        tensor_parallel=tensor_parallel,
        moe_dispatch_backend=args.moe_dispatch,
    )
    scheduler_config = _scheduler_config(args, backend)
    if tensor_parallel.enabled:
        scheduler = TensorParallelScheduler(
            model,
            scheduler_config,
            control_group=control_group,
        )
        if not scheduler.is_driver:
            return _LoadedRuntime(
                service=None,
                scheduler=scheduler,
                owns_process_group=owns_process_group,
            )
    else:
        scheduler = None

    # Only rank 0 turns text into token IDs and exposes the user-facing service.
    tokenizer = HuggingFaceTokenizer.from_pretrained(model_path)
    service = GenerationService(
        model,
        tokenizer,
        scheduler_config,
        scheduler=scheduler,
    )
    return _LoadedRuntime(service, scheduler, owns_process_group)


async def _generate(
    service: GenerationService,
    messages: Sequence[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    session = service.start_chat(
        messages,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    pieces: list[str] = []
    print("模型> ", end="", flush=True)
    async for chunk in session:
        pieces.append(chunk.text)
        print(chunk.text, end="", flush=True)
    print()
    return "".join(pieces)


async def _run_driver(service: GenerationService, args: argparse.Namespace) -> None:
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        await _generate(service, messages, args)
        return

    print("Ready. 输入 /reset 清空对话，/exit 退出。")
    while True:
        try:
            prompt = input("你> ")
        except EOFError:
            print()
            return
        command = prompt.strip().lower()
        if command in {"/exit", "/quit"}:
            return
        if command == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("对话已清空。")
            continue
        if not prompt.strip():
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            answer = await _generate(service, messages, args)
        except (RuntimeError, ValueError) as exc:
            messages.pop()
            print(f"生成失败：{exc}")
            continue
        messages.append({"role": "assistant", "content": answer})


async def _run(args: argparse.Namespace) -> None:
    runtime = _load_runtime(args)
    try:
        if runtime.scheduler is not None and not runtime.scheduler.is_driver:
            # Nonzero ranks never read prompts. They replay rank 0's add/step/abort
            # commands so model collectives and rank-local KV metadata stay aligned.
            runtime.scheduler.run_worker_loop()
            return
        if runtime.service is None:
            raise RuntimeError("rank 0 did not create a generation service")
        await _run_driver(runtime.service, args)
    finally:
        try:
            if runtime.scheduler is not None and runtime.scheduler.is_driver:
                runtime.scheduler.shutdown()
        finally:
            if runtime.owns_process_group and dist.is_initialized():
                dist.destroy_process_group()


def main() -> None:
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
