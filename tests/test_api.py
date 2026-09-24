import asyncio
import json
import unittest

from mysglang.scheduler import SchedulerConfig
from mysglang.serving import GenerationService, GenerationSession
from mysglang.tokenizer import ByteTokenizer

try:
    import httpx

    from mysglang.serving.http import create_app
except ImportError:
    httpx = None
    create_app = None

from tests.helpers import make_model


def make_service() -> GenerationService:
    return GenerationService(
        make_model(seed=789, vocab_size=256, max_position_embeddings=128),
        ByteTokenizer(),
        SchedulerConfig(
            max_running_requests=4,
            prefill_token_budget=32,
            num_pages=48,
            page_size=4,
        ),
    )


async def collect_token_ids(session: GenerationSession) -> list[int]:
    return [chunk.token_id async for chunk in session]


class GenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_is_lazy_and_single_consumer(self) -> None:
        service = make_service()
        session = service.start("not-consumed-yet", max_new_tokens=2)

        self.assertEqual(service.stats.active_requests, 0)
        self.assertEqual(service.scheduler.cache.allocator.stats.request_count, 0)
        stream = session.__aiter__()
        with self.assertRaisesRegex(RuntimeError, "only be consumed once"):
            session.__aiter__()
        await stream.aclose()
        self.assertEqual(service.stats.active_requests, 0)

    async def test_concurrent_sessions_share_decode_batches(self) -> None:
        service = make_service()
        first = service.start("first", max_new_tokens=5)
        second = service.start("第二个", max_new_tokens=5)
        self.assertIsInstance(first, GenerationSession)
        self.assertIsInstance(second, GenerationSession)

        first_tokens, second_tokens = await asyncio.gather(
            collect_token_ids(first),
            collect_token_ids(second),
        )

        self.assertEqual(len(first_tokens), 5)
        self.assertEqual(len(second_tokens), 5)
        self.assertNotEqual(first.request.request_id, second.request.request_id)
        self.assertEqual(service.scheduler.stats.max_decode_batch_size, 2)
        self.assertEqual(service.stats.finished_requests, 2)
        self.assertEqual(service.stats.active_requests, 0)

    async def test_aclose_waits_for_worker_after_generation(self) -> None:
        service = make_service()
        tokens = await collect_token_ids(service.start("hello", max_new_tokens=2))

        self.assertEqual(len(tokens), 2)
        await service.aclose()
        self.assertIsNone(service._worker_task)
        self.assertEqual(service.stats.active_requests, 0)

    async def test_closing_stream_aborts_and_releases_request(self) -> None:
        service = make_service()
        session = service.start("shared-prefix/cancel-me", max_new_tokens=8)
        stream = session.__aiter__()
        first = await anext(stream)
        self.assertFalse(first.finished)

        await stream.aclose()

        self.assertEqual(service.stats.active_requests, 0)
        self.assertEqual(service.stats.aborted_requests, 1)
        self.assertEqual(service.scheduler.cache.allocator.stats.request_count, 0)
        self.assertEqual(service.scheduler.cache.prefix_cache.stats.protected_pages, 0)


@unittest.skipUnless(httpx is not None, "requires the serving dependency")
class HTTPServingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = make_service()
        transport = httpx.ASGITransport(app=create_app(self.service))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_streaming_matches_non_streaming_and_reuses_prefix(self) -> None:
        payload = {
            "prompt": "shared-prefix/hello",
            "max_tokens": 4,
            "stream": False,
        }
        non_streaming = await self.client.post("/generate", json=payload)
        self.assertEqual(non_streaming.status_code, 200)
        body = non_streaming.json()

        payload["stream"] = True
        events = []
        saw_done = False
        async with self.client.stream("POST", "/generate", json=payload) as response:
            self.assertEqual(response.status_code, 200)
            async for line in response.aiter_lines():
                if line == "data: [DONE]":
                    saw_done = True
                elif line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        self.assertTrue(saw_done)
        self.assertEqual("".join(event["text"] for event in events), body["text"])
        self.assertEqual([event["token_id"] for event in events], body["token_ids"])
        self.assertEqual([event["output_index"] for event in events], [0, 1, 2, 3])
        self.assertTrue(events[-1]["finished"])
        self.assertEqual(body["usage"]["completion_tokens"], 4)
        self.assertGreater(self.service.scheduler.cache.prefix_cache.stats.matched_tokens, 0)

    async def test_concurrent_http_chat_usage_and_validation(self) -> None:
        first, second = await asyncio.gather(
            self.client.post("/generate", json={"prompt": "first", "max_tokens": 5}),
            self.client.post("/generate", json={"prompt": "第二个", "max_tokens": 5}),
        )
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertNotEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(self.service.scheduler.stats.max_decode_batch_size, 2)

        chat = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            },
        )
        self.assertEqual(chat.status_code, 200)
        usage = chat.json()["usage"]
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + 3)

        empty = await self.client.post(
            "/generate",
            json={"prompt": "", "max_tokens": 1},
        )
        self.assertEqual(empty.status_code, 400)
        self.assertIn("non-empty", empty.json()["detail"])


if __name__ == "__main__":
    unittest.main()
