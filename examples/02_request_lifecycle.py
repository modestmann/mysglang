"""Chapter 2: replay request state transitions and incremental output events."""

import json
from dataclasses import asdict

from mysglang import IncrementalOutput, Request, SamplingParams


def print_event(event: IncrementalOutput) -> None:
    print(json.dumps(asdict(event), ensure_ascii=False))


def main() -> None:
    request = Request.from_token_ids(
        request_id="demo-length",
        prompt_token_ids=[1, 5, 9],
        sampling_params=SamplingParams(max_new_tokens=3, eos_token_id=2),
    )
    request.start_prefill()
    request.start_decode()
    for token_id in [7, 8, 9]:
        print_event(request.record_token(token_id))

    cancelled = Request.from_token_ids(
        request_id="demo-abort",
        prompt_token_ids=[4, 6],
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    cancelled.start_prefill()
    print_event(cancelled.abort())


if __name__ == "__main__":
    main()
