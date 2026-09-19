from __future__ import annotations

import argparse
import asyncio
import importlib.util
from collections.abc import Sequence
from pathlib import Path

import torch

from mysglang.modeling import FlashAttentionBackend, Qwen3ForCausalLM, TorchAttentionBackend
from mysglang.scheduler import SchedulerConfig
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--prompt", help="run one prompt and exit instead of opening the REPL")
    parser.add_argument("--num-pages", type=int, default=16)
    parser.add_argument("--cuda-graph", action="store_true", help="capture the greedy B=1 decode")
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


def _load_service(args: argparse.Namespace) -> GenerationService:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    backend = _make_backend(args.backend, device)
    if isinstance(backend, FlashAttentionBackend) and dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError("FlashAttention requires --dtype float16 or bfloat16")

    print(
        f"Loading {model_path} on {device} as {dtype} with {backend.name} ...",
        flush=True,
    )
    tokenizer = HuggingFaceTokenizer.from_pretrained(model_path)
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device=device,
        attention_backend=backend,
    )
    page_size = 256 if isinstance(backend, FlashAttentionBackend) else 16
    return GenerationService(
        model,
        tokenizer,
        SchedulerConfig(
            max_running_requests=1,
            prefill_token_budget=args.num_pages * page_size,
            num_pages=args.num_pages,
            page_size=page_size,
            decode_cuda_graph_batch_sizes=(1,) if args.cuda_graph else (),
        ),
    )


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


async def _run(args: argparse.Namespace) -> None:
    service = _load_service(args)
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


def main() -> None:
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
