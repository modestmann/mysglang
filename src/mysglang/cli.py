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
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist

from mysglang.config import ModelConfig
from mysglang.modeling import (
    FlashAttentionBackend,
    FlashInferBackend,
    Qwen3ForCausalLM,
    TensorParallelContext,
    TorchAttentionBackend,
)
from mysglang.modeling.deltanet import LINEAR_ATTENTION_BACKENDS
from mysglang.multimodal import ImageProcessor, load_image
from mysglang.scheduler import SchedulerConfig, TensorParallelScheduler
from mysglang.serving import GenerationService
from mysglang.serving.parsers import ReasoningParser
from mysglang.tokenizer import HuggingFaceTokenizer


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[3] / "KuiperQwen3/artifacts/qwen3-0.6b/hf-source"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a local Qwen3 checkpoint")
    parser.add_argument("--model", type=Path, default=_default_model_path())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument(
        "--backend", choices=("auto", "flash", "flashinfer", "torch"), default="auto"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help="tokens per KV page (default: 256 for flash, which FA2 requires; 16 otherwise)",
    )
    parser.add_argument(
        "--moe-dispatch",
        choices=("naive", "sorted", "grouped", "triton_grouped", "all_to_all"),
        default="sorted",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--prompt", help="run one prompt and exit instead of opening the REPL")
    parser.add_argument(
        "--num-pages",
        type=int,
        help="paged KV capacity; on CUDA defaults to what --memory-fraction leaves free",
    )
    parser.add_argument(
        "--preemption",
        action="store_true",
        help="admit by prefill size and preempt + recompute the newest request when KV runs out",
    )
    parser.add_argument(
        "--state-snapshots",
        type=int,
        default=8,
        help="DeltaNet state snapshots kept for hybrid-model prefix reuse (0 disables it)",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.85,
        help="share of total GPU memory the engine may use when sizing the KV pool",
    )
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument(
        "--prefill-token-budget",
        type=int,
        help="new prompt tokens admitted per scheduler step; defaults to KV capacity",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture greedy Decode (and MTP verify/draft steps), including NCCL TP",
    )
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=int,
        nargs="+",
        help="Decode batch sizes to capture with --cuda-graph; smaller batches pad up to the "
        "next size (default: 1, 2, 4, then every 8 up to --max-running-requests)",
    )
    parser.add_argument(
        "--speculative-ngram-max-tokens",
        type=int,
        default=0,
        help="maximum model-free n-gram draft length; 0 disables speculative decoding",
    )
    parser.add_argument("--speculative-ngram-min-match", type=int, default=2)
    parser.add_argument("--speculative-ngram-max-match", type=int, default=8)
    parser.add_argument(
        "--speculative-mtp-steps",
        type=int,
        default=0,
        help="draft tokens proposed by the checkpoint's MTP layer per round; 0 disables it",
    )
    parser.add_argument(
        "--linear-attention-backend",
        choices=LINEAR_ATTENTION_BACKENDS,
        default="auto",
        help="Gated DeltaNet kernels: fla on CUDA when installed, otherwise PyTorch",
    )
    parser.add_argument(
        "--dequantize-fp8",
        action="store_true",
        help="load an FP8 checkpoint as --dtype weights (numerical reference path)",
    )
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="load the vision tower for image input (default: when the checkpoint has one)",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=1024 * 1024,
        help="downscale larger images to this many pixels (~1 image token per 32x32 pixels)",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="image path/URL attached to --prompt (repeatable); in the REPL use /image PATH",
    )
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
    if name == "flashinfer":
        return FlashInferBackend()
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


# 自动 KV 池时每步 Prefill 的默认上限：池子可能有几万 token，整池一次 Prefill 的激活会 OOM。
_AUTO_PREFILL_TOKEN_BUDGET = 8192


def _default_graph_batch_sizes(max_running_requests: int) -> tuple[int, ...]:
    """1、2、4，之后每隔 8 一个，最后补上上限本身（mini-sglang 同款），实际 batch 向上补齐。"""
    sizes = [size for size in (1, 2, 4) if size <= max_running_requests]
    sizes += range(8, max_running_requests + 1, 8)
    return tuple(sorted({*sizes, max_running_requests}))


def _recurrent_pool_bytes(model: Qwen3ForCausalLM, num_slots: int) -> int:
    """DeltaNet 格子池的大小；它在 Scheduler 里才分配，测显存时还没占地方。"""
    spec = model.recurrent_state_spec
    if spec is None:
        return 0
    element = model.embed_tokens.weight.element_size()
    conv = spec.conv_dim * (spec.conv_kernel_size - 1) * element
    ssm = spec.num_heads * spec.head_k_dim * spec.head_v_dim * 4  # float32
    return num_slots * spec.num_layers * (conv + ssm)


def _auto_num_pages(
    model: Qwen3ForCausalLM,
    page_size: int,
    memory_fraction: float,
    max_running_requests: int,
    state_snapshots: int = 0,
) -> int:
    if not 0 < memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in (0, 1]")
    parameter = model.embed_tokens.weight
    element = parameter.element_size()

    # 一页 KV 的字节数：K、V 各一份，主模型 attention 层 + MTP 层。
    layers = model.num_kv_cache_layers + model.num_draft_kv_layers
    heads = model.kv_cache_num_heads
    head_dim = model.config.head_dim
    bytes_per_page = 2 * layers * heads * head_dim * page_size * element

    # 整张卡留出 (1 − memory_fraction) 给激活、CUDA Graph 和其他进程，再扣掉还没分配的格子池。
    free, total = torch.cuda.mem_get_info(parameter.device)
    budget = free - total * (1 - memory_fraction)
    # 快照和格子一样大，也在 Scheduler 里才分配。
    budget -= _recurrent_pool_bytes(model, max_running_requests + state_snapshots)

    num_pages = int(budget // bytes_per_page)
    if model.tensor_parallel.enabled:
        # 各卡空闲显存不同，但调度必须一致：所有卡取最小值。
        shared = torch.tensor([num_pages], device=parameter.device)
        dist.all_reduce(shared, op=dist.ReduceOp.MIN, group=model.tensor_parallel.process_group)
        num_pages = int(shared.item())
    if num_pages <= 0:
        raise RuntimeError(
            f"no GPU memory left for the KV pool ({free / 2**30:.2f} GiB free); "
            "raise --memory-fraction or pass --num-pages"
        )
    return num_pages


def _scheduler_config(
    args: argparse.Namespace,
    backend,
    model: Qwen3ForCausalLM | None = None,
) -> SchedulerConfig:
    page_size = getattr(args, "page_size", None) or (
        256 if isinstance(backend, FlashAttentionBackend) else 16
    )
    num_pages = args.num_pages
    default_prefill_budget = None
    if num_pages is None:
        if model is not None and model.embed_tokens.weight.is_cuda:
            num_pages = _auto_num_pages(
                model,
                page_size,
                getattr(args, "memory_fraction", 0.85),
                args.max_running_requests,
                getattr(args, "state_snapshots", 0),
            )
            default_prefill_budget = _AUTO_PREFILL_TOKEN_BUDGET
            print(
                f"[KV] auto-sized {num_pages} pages × {page_size} tokens "
                f"(--memory-fraction {getattr(args, 'memory_fraction', 0.85)})",
                flush=True,
            )
        else:
            num_pages = 16  # CPU 测不了空闲显存，沿用原来的默认值
    prefill_token_budget = args.prefill_token_budget or min(
        num_pages * page_size, default_prefill_budget or num_pages * page_size
    )
    graph_batch_sizes = getattr(args, "decode_cuda_graph_batch_sizes", None)
    if graph_batch_sizes is None:
        requested = getattr(args, "cuda_graph_batch_sizes", None) or _default_graph_batch_sizes(
            args.max_running_requests
        )
        graph_batch_sizes = tuple(sorted(set(requested))) if args.cuda_graph else ()
    return SchedulerConfig(
        max_running_requests=args.max_running_requests,
        prefill_token_budget=prefill_token_budget,
        num_pages=num_pages,
        page_size=page_size,
        decode_cuda_graph_batch_sizes=graph_batch_sizes,
        speculative_ngram_max_tokens=args.speculative_ngram_max_tokens,
        speculative_ngram_min_match=args.speculative_ngram_min_match,
        speculative_ngram_max_match=args.speculative_ngram_max_match,
        speculative_mtp_steps=args.speculative_mtp_steps,
        enable_preemption=getattr(args, "preemption", False),
        state_snapshots=getattr(args, "state_snapshots", 0),
    )


def _vision_enabled(args: argparse.Namespace, model_path: Path) -> bool:
    has_vision = ModelConfig.from_pretrained(model_path).vision is not None
    if getattr(args, "vision", None) is None:
        return has_vision
    if args.vision and not has_vision:
        raise ValueError("--vision requested but the checkpoint has no vision tower")
    return args.vision


def _normalize_image_path(raw: str) -> str:
    """Accept quoted paths and Windows paths (``C:\\x\\y.jpg`` -> ``/mnt/c/x/y.jpg`` on WSL)."""
    path = raw.strip().strip("\"'")
    if len(path) > 2 and path[1] == ":" and path[2] in "\\/" and not Path(path).exists():
        path = f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def _user_message(text: str, images: Sequence[str]) -> dict:
    if not images:
        return {"role": "user", "content": text}
    parts: list[dict] = [{"type": "image", "image": image} for image in images]
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def _load_runtime(args: argparse.Namespace) -> _LoadedRuntime:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    launch = _read_distributed_launch(args)
    device = _rank_device(args.device, launch)
    dtype = _resolve_dtype(args.dtype, device)
    backend = _make_backend(args.backend, device)
    if isinstance(backend, (FlashAttentionBackend, FlashInferBackend)) and dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError("flash / flashinfer backends require --dtype float16 or bfloat16")
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
        linear_attention_backend=args.linear_attention_backend,
        enable_mtp=args.speculative_mtp_steps > 0,
        fp8_weights=not args.dequantize_fp8,
        enable_vision=_vision_enabled(args, model_path),
    )
    scheduler_config = _scheduler_config(args, backend, model)
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
        image_processor=(
            None
            if model.visual is None
            else replace(
                ImageProcessor.from_pretrained(model_path, model.config.vision),
                max_pixels=args.max_image_pixels,
            )
        ),
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
        presence_penalty=args.presence_penalty,
    )
    # thinking 模式下 prompt 已以 <think> 结尾；推理与回答分开显示，只把回答写回历史。
    parser = ReasoningParser() if args.thinking and service.tokenizer.supports_reasoning else None
    pieces: list[str] = []
    label = "思考> " if parser is not None else "模型> "
    print(label, end="", flush=True)
    async for chunk in session:
        if parser is None:
            reasoning, content = "", chunk.text
        else:
            reasoning, content = parser.feed(chunk.text, final=chunk.finished)
        if reasoning:
            print(reasoning, end="", flush=True)
        if content:
            if not pieces and parser is not None:
                print("\n模型> ", end="", flush=True)
            pieces.append(content)
            print(content, end="", flush=True)
    print()
    return "".join(pieces)


async def _run_driver(service: GenerationService, args: argparse.Namespace) -> None:
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.prompt is not None:
        messages.append(_user_message(args.prompt, args.image))
        await _generate(service, messages, args)
        return

    pending_images: list[str] = list(args.image)
    print("Ready. 输入 /reset 清空对话，/image PATH 给下一句附图，/exit 退出。")
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
            pending_images.clear()
            print("对话已清空。")
            continue
        if command.split(maxsplit=1)[:1] == ["/image"]:
            parts = prompt.strip().split(maxsplit=1)
            if len(parts) < 2:
                print("用法：/image 图片路径（或 URL）")
                continue
            image = _normalize_image_path(parts[1])
            try:
                load_image(image).verify()
            except Exception as exc:  # noqa: BLE001 - report any unreadable image to the user
                print(f"无法读取图片 {image}：{exc}")
                continue
            pending_images.append(image)
            print(f"已附加图片（共 {len(pending_images)} 张），将随下一句发送。")
            continue
        if not prompt.strip():
            continue

        messages.append(_user_message(prompt, pending_images))
        pending_images = []
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
            if runtime.service is not None:
                await runtime.service.aclose()
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
