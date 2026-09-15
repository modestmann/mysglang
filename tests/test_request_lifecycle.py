import unittest

from mysglang import (
    FinishReason,
    InvalidStateTransition,
    Request,
    RequestState,
    SamplingParams,
)


class RequestLifecycleTest(unittest.TestCase):
    def make_request(
        self,
        *,
        max_new_tokens: int = 2,
        eos_token_id: int | None = None,
        ignore_eos: bool = False,
    ) -> Request:
        return Request.from_token_ids(
            "req-7",
            [10, 20, 30],
            SamplingParams(
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                ignore_eos=ignore_eos,
            ),
        )

    def test_happy_path_finishes_at_length_limit(self) -> None:
        request = self.make_request(max_new_tokens=2)
        self.assertEqual(request.state, RequestState.WAITING)

        request.start_prefill()
        request.start_decode()
        first = request.record_token(40)
        second = request.record_token(41)

        self.assertFalse(first.finished)
        self.assertEqual(first.output_index, 0)
        self.assertTrue(second.finished)
        self.assertEqual(second.finish_reason, FinishReason.LENGTH)
        self.assertEqual(request.state, RequestState.FINISHED)
        self.assertEqual(request.all_token_ids, (10, 20, 30, 40, 41))

    def test_eos_finishes_unless_it_is_ignored(self) -> None:
        request = self.make_request(max_new_tokens=3, eos_token_id=2)
        request.start_prefill()
        request.start_decode()
        event = request.record_token(2)
        self.assertEqual(event.finish_reason, FinishReason.EOS)

        ignored = self.make_request(max_new_tokens=2, eos_token_id=2, ignore_eos=True)
        ignored.start_prefill()
        ignored.start_decode()
        self.assertFalse(ignored.record_token(2).finished)
        self.assertEqual(ignored.record_token(3).finish_reason, FinishReason.LENGTH)

    def test_illegal_transition_is_rejected_without_mutating_state(self) -> None:
        request = self.make_request()
        with self.assertRaisesRegex(
            InvalidStateTransition, "waiting -> decoding"
        ):
            request.start_decode()
        self.assertEqual(request.state, RequestState.WAITING)

        request.start_prefill()
        request.start_decode()
        request.record_token(1)
        request.record_token(2)
        with self.assertRaisesRegex(
            InvalidStateTransition, "cannot record a token while finished"
        ):
            request.record_token(3)

    def test_abort_from_waiting_is_terminal(self) -> None:
        request = self.make_request()
        event = request.abort()

        self.assertEqual(request.state, RequestState.ABORTED)
        self.assertTrue(request.is_terminal)
        self.assertIsNone(event.token_id)
        self.assertEqual(event.finish_reason, FinishReason.ABORTED)
        with self.assertRaises(InvalidStateTransition):
            request.start_prefill()

    def test_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            SamplingParams(max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Request.from_token_ids("empty", [], SamplingParams(max_new_tokens=1))
        with self.assertRaisesRegex(TypeError, "must contain integers"):
            Request.from_token_ids("bad", [1, True], SamplingParams(max_new_tokens=1))

    def test_caller_cannot_mutate_request_token_history_through_an_alias(self) -> None:
        prompt = [1, 2]
        request = Request.from_token_ids("stable", prompt, SamplingParams(max_new_tokens=1))
        prompt.append(3)

        self.assertEqual(request.prompt_token_ids, (1, 2))
        self.assertEqual(request.output_token_ids, ())


if __name__ == "__main__":
    unittest.main()
