"""用 torchrun 启动 DeepSeek V4.1 原生量化纯文本生成。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .deepseek_chat import ChatChannel, run_interactive
from .modeling.deepseek_v41.config import DeepSeekV41Config
from .modeling.deepseek_v41.engram import compressed_token_map
from .modeling.deepseek_v41.generation import ChatSession, format_prompt, generate_tokens
from .modeling.deepseek_v41.loader import load_rank_weights
from .modeling.deepseek_v41.runtime import DeepSeekV41Model
from .modeling.parallel import TensorParallelContext


class _TokenizerAdapter:
    def __init__(self, backend):
        self.backend_tokenizer = backend

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()


def _log(message: str, rank: int) -> None:
    if rank == 0:
        print(message, file=sys.stderr, flush=True)


def generate(
    model, tokenizer, prompt: str, *, max_new_tokens: int, raw: bool, compare_torch: bool = False
) -> dict:
    encoded = tokenizer.encode(prompt if raw else format_prompt(prompt), add_special_tokens=False)
    result = generate_tokens(
        model,
        tokenizer,
        encoded.ids,
        max_new_tokens=max_new_tokens,
        compare_torch=compare_torch,
        log=lambda message: _log(message, model.context.tp.rank),
    )
    return {"prompt": prompt, **result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="只读原生 SafeTensors 目录")
    parser.add_argument("--config", type=Path, required=True, help="可读的 inference/config.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", action="append", help="可重复，加载一次依次生成独立回答")
    mode.add_argument("--interactive", action="store_true", help="常驻多轮聊天，复用历史和缓存")
    parser.add_argument("--max-new-tokens", type=int, help="默认单轮 32、交互模式 256")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--backend", choices=("triton", "torch"), default="triton")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--compare-torch", action="store_true", help="同一权重对照 prefill 和前两步 decode"
    )
    parser.add_argument("--output", type=Path, help="rank 0 写 JSON 结果")
    args = parser.parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = 256 if args.interactive else 32
    if args.interactive and (args.raw_prompt or args.compare_torch):
        parser.error("--interactive cannot be combined with --raw-prompt or --compare-torch")
    if args.max_new_tokens <= 0 or args.max_seq_len <= 0:
        parser.error("--max-new-tokens and --max-seq-len must be positive")
    if args.max_new_tokens >= args.max_seq_len:
        parser.error("--max-new-tokens must leave context space for the input")
    if not torch.cuda.is_available():
        parser.error("native generation requires CUDA; use the small reference model on CPU")
    from tokenizers import Tokenizer

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # FP32 路由、mHC 和对照 GEMM 保留 IEEE 精度。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    control_group = None
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        if world > 1 and args.interactive:
            # 人类可能长时间暂停输入；Gloo 控制面等待不受 NCCL watchdog 的 30 分钟限制。
            control_group = dist.new_group(backend="gloo", timeout=timedelta(days=365))
        config = DeepSeekV41Config.from_json(args.config)
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        for prompt in args.prompt or []:
            rendered = prompt if args.raw_prompt else format_prompt(prompt)
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            if not ids or len(ids) + args.max_new_tokens > args.max_seq_len:
                raise ValueError("empty prompt or prompt + max_new_tokens exceeds max_seq_len")
        token_map = compressed_token_map(_TokenizerAdapter(tokenizer))
        _log(f"Loading TP={world}, backend={args.backend}, model={args.model}", rank)
        start = time.perf_counter()
        weights, report = load_rank_weights(
            args.model,
            config,
            rank=rank,
            world_size=world,
            device=device,
            progress=lambda message: _log(message, rank),
        )
        tp = TensorParallelContext.from_distributed() if world > 1 else TensorParallelContext()
        model = DeepSeekV41Model(
            config,
            weights,
            token_map=token_map,
            tensor_parallel=tp,
            backend=args.backend,
            max_seq_len=args.max_seq_len,
        )
        del weights
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        _log(f"Loaded {report['loaded_bytes'] / 2**30:.2f} GiB/rank in {load_seconds:.1f}s", rank)
        result = {
            "model": str(args.model),
            "config": str(args.config),
            "tp_size": world,
            "backend": args.backend,
            "precision": "MXFP4 experts / MXFP8 dense / BF16 activations",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "load_seconds": load_seconds,
            "load_report": report,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "interactive": args.interactive,
            "generations": [],
        }

        def save_result() -> None:
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

        def save_turn(generation: dict) -> None:
            result["generations"].append(generation)
            save_result()

        if args.interactive:
            session = ChatSession(model, tokenizer, max_new_tokens=args.max_new_tokens)
            channel = ChatChannel(rank=rank, world_size=world, group=control_group)
            run_interactive(
                session,
                channel,
                read=input,
                output=sys.stdout,
                on_turn=save_turn if args.output is not None else None,
            )
        else:
            result["generations"] = [
                generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    raw=args.raw_prompt,
                    compare_torch=args.compare_torch,
                )
                for prompt in args.prompt
            ]
        if rank == 0:
            save_result()
            if not args.interactive:
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        if control_group is not None:
            dist.destroy_process_group(control_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
