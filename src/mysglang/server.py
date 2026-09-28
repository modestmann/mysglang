"""OpenAI-compatible HTTP server (``/v1/chat/completions``) on top of the CLI runtime.

Single GPU::

    PYTHONPATH=src python -m mysglang.server --model /path/to/Qwen3.5-0.8B --port 8000

Four GPUs (only rank 0 listens; other ranks replay its scheduler commands)::

    PYTHONPATH=src torchrun --standalone --nproc-per-node=4 -m mysglang.server \\
        --tensor-parallel-size 4 --model /path/to/Qwen3.6-35B-A3B-FP8 \\
        --moe-dispatch triton_grouped --speculative-mtp-steps 3 --port 8000
"""

from __future__ import annotations

import asyncio

import torch.distributed as dist

from mysglang.cli import _load_runtime, _parser


def _server_parser():
    parser = _parser()
    parser.description = "Serve a local Qwen3/Qwen3.5 checkpoint over an OpenAI-style API"
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", help="model id reported by /v1/models")
    return parser


async def _serve(args) -> None:
    try:
        import uvicorn

        from mysglang.serving.http import create_app
    except ImportError as exc:
        raise RuntimeError("the HTTP server requires the 'serve' project extra") from exc

    runtime = _load_runtime(args)
    try:
        if runtime.scheduler is not None and not runtime.scheduler.is_driver:
            runtime.scheduler.run_worker_loop()
            return
        if runtime.service is None:
            raise RuntimeError("rank 0 did not create a generation service")
        app = create_app(
            runtime.service,
            model_name=args.served_model_name or args.model.expanduser().resolve().name,
        )
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
        await server.serve()
    finally:
        try:
            if runtime.service is not None:
                await runtime.service.aclose()
        finally:
            if runtime.owns_process_group and dist.is_initialized():
                dist.destroy_process_group()


def main() -> None:
    asyncio.run(_serve(_server_parser().parse_args()))


if __name__ == "__main__":
    main()
