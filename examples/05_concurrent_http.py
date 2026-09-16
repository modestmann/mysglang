"""Chapter 5 experiment: send concurrent in-process HTTP requests."""

import asyncio
import time

import httpx

from examples_05_app import build_app_and_service


async def main() -> None:
    app, service = build_app_and_service(device="cpu")
    transport = httpx.ASGITransport(app=app)
    prompts = ["alpha", "中文", "gamma", "delta"]

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.perf_counter()
        responses = await asyncio.gather(
            *[
                client.post(
                    "/generate",
                    json={"prompt": prompt, "max_tokens": 8},
                )
                for prompt in prompts
            ]
        )
        elapsed_ms = (time.perf_counter() - start) * 1_000

    for response in responses:
        response.raise_for_status()
        body = response.json()
        print(body["id"], body["usage"], body["finish_reason"])
    print(f"four-request elapsed time: {elapsed_ms:.3f} ms")
    print(f"max active requests: {service.stats.max_active_requests}")
    print("model forward policy: serialized, one request at a time")


if __name__ == "__main__":
    asyncio.run(main())
