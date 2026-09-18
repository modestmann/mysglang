import unittest

from mysglang import (
    FinishReason,
    InvalidStateTransition,
    Request,
    RequestState,
    SamplingParams,
)


class RequestLifecycleTest(unittest.TestCase):
    def test_length_eos_and_ignore_eos_finish_reasons(self) -> None:
        length = Request.from_token_ids("length", [1, 2], SamplingParams(max_new_tokens=2))
        length.start_prefill()
        length.start_decode()
        first = length.record_token(7)
        last = length.record_token(8)
        self.assertEqual((first.output_index, last.output_index), (0, 1))
        self.assertFalse(first.finished)
        self.assertEqual(last.finish_reason, FinishReason.LENGTH)
        self.assertEqual(length.state, RequestState.FINISHED)

        eos = Request.from_token_ids(
            "eos",
            [1, 2],
            SamplingParams(max_new_tokens=3, eos_token_id=9),
        )
        eos.start_prefill()
        eos.start_decode()
        self.assertEqual(eos.record_token(9).finish_reason, FinishReason.EOS)

        ignored = Request.from_token_ids(
            "ignored",
            [1, 2],
            SamplingParams(max_new_tokens=1, eos_token_id=9, ignore_eos=True),
        )
        ignored.start_prefill()
        ignored.start_decode()
        self.assertEqual(ignored.record_token(9).finish_reason, FinishReason.LENGTH)

    def test_illegal_transitions_do_not_mutate_state(self) -> None:
        request = Request.from_token_ids("request", [1, 2], SamplingParams(max_new_tokens=1))
        with self.assertRaisesRegex(InvalidStateTransition, "waiting -> decoding"):
            request.start_decode()
        self.assertEqual(request.state, RequestState.WAITING)

        event = request.abort()
        self.assertEqual(event.finish_reason, FinishReason.ABORTED)
        self.assertIsNone(event.token_id)
        self.assertTrue(request.is_terminal)
        with self.assertRaises(InvalidStateTransition):
            request.start_prefill()

    def test_validation_and_prompt_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            SamplingParams(max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Request.from_token_ids("empty", [], SamplingParams(max_new_tokens=1))
        with self.assertRaisesRegex(TypeError, "must contain integers"):
            Request.from_token_ids("bad", [1, True], SamplingParams(max_new_tokens=1))

        prompt = [1, 2]
        request = Request.from_token_ids("owned", prompt, SamplingParams(max_new_tokens=1))
        prompt.append(3)
        self.assertEqual(request.prompt_token_ids, (1, 2))
        self.assertEqual(request.output_token_ids, ())


if __name__ == "__main__":
    unittest.main()
