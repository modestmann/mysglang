import asyncio
import json
import unittest

import httpx
import torch

from mysglang import ModelConfig, TinyCausalLM
from mysglang.scheduler import SchedulerConfig
from mysglang.serving import ContinuousBatchGenerationService, GenerationService
from mysglang.serving.http import create_app
from mysglang.tokenizer import ByteTokenizer


def make_service() -> GenerationService:
    torch.manual_seed(789)
    model = TinyCausalLM(
        ModelConfig(
            vocab_size=256,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).eval()
    return GenerationService(model, ByteTokenizer())


class ByteTokenizerTest(unittest.TestCase):
    def test_chinese_text_round_trips_through_utf8_bytes(self) -> None:
        tokenizer = ByteTokenizer()
        text = "你好，LLM"
        token_ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(token_ids), text)

        decoder = tokenizer.new_incremental_decoder()
        pieces = [
            decoder.decode(token_id, final=index == len(token_ids) - 1)
            for index, token_id in enumerate(token_ids)
        ]
        self.assertEqual("".join(pieces), text)


class GenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_sessions_have_independent_caches(self) -> None:
        service = make_service()
        first = service.start("first", max_new_tokens=4)
        second = service.start("第二个", max_new_tokens=4)

        async def collect(session) -> list[int]:
            return [chunk.token_id async for chunk in session]

        first_tokens, second_tokens = await asyncio.gather(collect(first), collect(second))
        self.assertEqual(len(first_tokens), 4)
        self.assertEqual(len(second_tokens), 4)
        self.assertNotEqual(first.request.request_id, second.request.request_id)
        self.assertGreaterEqual(service.stats.max_active_requests, 2)
        self.assertEqual(service.stats.finished_requests, 2)
        self.assertEqual(service.stats.active_requests, 0)

    async def test_closing_stream_aborts_and_releases_request(self) -> None:
        service = make_service()
        session = service.start("cancel me", max_new_tokens=8)
        stream = session.__aiter__()

        first = await anext(stream)
        self.assertFalse(first.finished)
        self.assertEqual(service.stats.active_requests, 1)
        await stream.aclose()

        self.assertEqual(service.stats.active_requests, 0)
        self.assertEqual(service.stats.aborted_requests, 1)


class ContinuousBatchGenerationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_sessions_share_a_decode_forward(self) -> None:
        original = make_service()
        service = ContinuousBatchGenerationService(
            original.model,
            original.tokenizer,
            SchedulerConfig(max_running_requests=4, prefill_token_budget=32),
        )
        first = service.start("first", max_new_tokens=5)
        second = service.start("第二个", max_new_tokens=5)

        async def collect(session) -> list[int]:
            return [chunk.token_id async for chunk in session]

        first_tokens, second_tokens = await asyncio.gather(collect(first), collect(second))
        self.assertEqual(len(first_tokens), 5)
        self.assertEqual(len(second_tokens), 5)
        self.assertEqual(service.scheduler.stats.max_decode_batch_size, 2)
        self.assertEqual(service.stats.finished_requests, 2)
        self.assertEqual(service.stats.active_requests, 0)

    async def test_concurrent_http_requests_use_the_batch_scheduler(self) -> None:
        original = make_service()
        service = ContinuousBatchGenerationService(
            original.model,
            original.tokenizer,
            SchedulerConfig(max_running_requests=4, prefill_token_budget=32),
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first, second = await asyncio.gather(
                client.post(
                    "/generate",
                    json={"prompt": "first", "max_tokens": 5},
                ),
                client.post(
                    "/generate",
                    json={"prompt": "第二个", "max_tokens": 5},
                ),
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(service.scheduler.stats.max_decode_batch_size, 2)


class HTTPServingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = make_service()
        transport = httpx.ASGITransport(app=create_app(self.service))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_streaming_and_non_streaming_outputs_match(self) -> None:
        payload = {"prompt": "hello", "max_tokens": 4, "stream": False}
        non_streaming = await self.client.post("/generate", json=payload)
        self.assertEqual(non_streaming.status_code, 200)
        body = non_streaming.json()
        self.assertEqual(body["usage"], {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
        })
        self.assertEqual(body["finish_reason"], "length")

        payload["stream"] = True
        events = []
        async with self.client.stream("POST", "/generate", json=payload) as response:
            self.assertEqual(response.status_code, 200)
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line.removeprefix("data: ")))

        self.assertEqual("".join(event["text"] for event in events), body["text"])
        self.assertEqual([event["token_id"] for event in events], body["token_ids"])
        self.assertEqual([event["output_index"] for event in events], [0, 1, 2, 3])
        self.assertTrue(events[-1]["finished"])

    async def test_chat_completion_reports_real_usage(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "mysglang-tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        usage = body["usage"]
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + 3)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    async def test_empty_prompt_returns_clear_client_error(self) -> None:
        response = await self.client.post(
            "/generate",
            json={"prompt": "", "max_tokens": 1},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-empty", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
