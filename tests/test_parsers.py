import unittest

import torch

from mysglang.core import SamplingParams
from mysglang.scheduler.sampler import apply_penalties, sample_token
from mysglang.serving.parsers import ReasoningParser, ToolCallStreamBuffer, parse_tool_calls

OUTPUT = (
    "Let me think.\nOK</think>\n\nHello <tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nBeijing\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
    "</function>\n</tool_call>"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]


def stream(text: str, size: int) -> tuple[str, str, str]:
    reasoning = ReasoningParser()
    tools = ToolCallStreamBuffer()
    chunks = [text[index : index + size] for index in range(0, len(text), size)]
    thought, streamed = [], []
    for index, chunk in enumerate(chunks):
        final = index == len(chunks) - 1
        delta_reasoning, delta_content = reasoning.feed(chunk, final=final)
        thought.append(delta_reasoning)
        streamed.append(tools.feed(delta_content, final=final))
    return "".join(thought), "".join(streamed), tools.buffered_text()


class ReasoningAndToolParserTest(unittest.TestCase):
    def test_every_chunking_yields_the_same_split(self) -> None:
        for size in (1, 2, 3, 7, 1000):
            with self.subTest(size=size):
                thought, streamed, buffered = stream(OUTPUT, size)
                self.assertEqual(thought, "Let me think.\nOK")
                self.assertEqual(streamed, "Hello ")
                content, calls = parse_tool_calls(buffered, TOOLS)
                self.assertEqual(content, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, "get_weather")
                self.assertEqual(calls[0].arguments, {"city": "Beijing", "days": 3})

    def test_model_emitted_think_tag_and_disabled_thinking(self) -> None:
        parser = ReasoningParser()
        self.assertEqual(parser.feed("<th"), ("", ""))
        self.assertEqual(parser.feed("ink>\nidea</think>answer", final=True), ("\nidea", "answer"))
        plain = ReasoningParser(in_reasoning=False)
        self.assertEqual(plain.feed("no </think> here", final=True), ("", "no </think> here"))

    def test_text_without_tool_calls_is_untouched(self) -> None:
        self.assertEqual(parse_tool_calls("plain answer", TOOLS), ("plain answer", []))
        call = parse_tool_calls(OUTPUT.split("</think>\n\n")[1], TOOLS)[1][0]
        arguments = call.to_openai()["function"]["arguments"]
        self.assertEqual(arguments, '{"city": "Beijing", "days": 3}')


class PenaltyTest(unittest.TestCase):
    def test_presence_and_frequency_penalties_only_touch_generated_tokens(self) -> None:
        logits = torch.zeros(5)
        params = SamplingParams(max_new_tokens=4, presence_penalty=1.0, frequency_penalty=0.5)
        adjusted = apply_penalties(logits, params, (1, 1, 3))
        torch.testing.assert_close(adjusted, torch.tensor([0.0, -2.0, 0.0, -1.5, 0.0]))
        self.assertEqual(logits.abs().sum().item(), 0.0)

    def test_greedy_sampling_applies_penalties_before_argmax(self) -> None:
        logits = torch.tensor([0.0, 3.0, 2.5])
        params = SamplingParams(max_new_tokens=2, presence_penalty=1.0)
        self.assertEqual(sample_token(logits, params), 1)
        self.assertEqual(sample_token(logits, params, output_token_ids=(1,)), 2)

    def test_penalties_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "presence_penalty"):
            SamplingParams(max_new_tokens=1, presence_penalty=3.0)


if __name__ == "__main__":
    unittest.main()
