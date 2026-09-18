from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(str, Enum):
    LENGTH = "length"
    EOS = "eos"
    ABORTED = "aborted"


class InvalidStateTransition(RuntimeError):
    """Raised when a request attempts an illegal lifecycle transition."""


@dataclass(frozen=True)
class SamplingParams:
    """Sampling options implemented by the current greedy path.

    Temperature, top-k and top-p are intentionally absent until a real sampler
    implements them, so callers cannot request behavior that would be ignored.
    """

    max_new_tokens: int
    eos_token_id: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.eos_token_id is not None:
            _validate_token_id(self.eos_token_id, name="eos_token_id")


# 冻结字段：增量输出事件创建后不允许被消费者修改。
@dataclass(frozen=True)
class IncrementalOutput:
    """One ordered output event emitted by the request state machine."""

    request_id: str
    token_id: int | None
    output_index: int
    finished: bool
    finish_reason: FinishReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.token_id is not None:
            _validate_token_id(self.token_id)
        if self.output_index < 0:
            raise ValueError("output_index must be non-negative")
        if self.finished != (self.finish_reason is not None):
            raise ValueError("finished and finish_reason must agree")
        if self.finish_reason is FinishReason.ABORTED and self.token_id is not None:
            raise ValueError("an aborted event must not contain a token")
        if self.finish_reason is not FinishReason.ABORTED and self.token_id is None:
            raise ValueError("a non-abort event must contain a token")


# 不可修改的集合，集中描述 Request 状态迁移表。
_ALLOWED_TRANSITIONS = {
    RequestState.WAITING: frozenset({RequestState.PREFILL, RequestState.ABORTED}),
    RequestState.PREFILL: frozenset({RequestState.DECODING, RequestState.ABORTED}),
    RequestState.DECODING: frozenset({RequestState.FINISHED, RequestState.ABORTED}),
    RequestState.FINISHED: frozenset(),
    RequestState.ABORTED: frozenset(),
}


@dataclass
class Request:
    """Framework-owned state for one tokenized generation request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    _state: RequestState = field(default=RequestState.WAITING, init=False, repr=False)
    _output_token_ids: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must not be empty")
        self.prompt_token_ids = _validate_prompt(self.prompt_token_ids)

    # 额外的构造函数：把外部的 token iterable 规范化后包装成请求。
    @classmethod
    def from_token_ids(
        cls,
        request_id: str,
        prompt_token_ids: Iterable[int],
        sampling_params: SamplingParams,
    ) -> Request:
        return cls(request_id, tuple(prompt_token_ids), sampling_params)

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @property
    def last_output_token_id(self) -> int:
        """Return the newest generated token without copying the full history."""
        if not self._output_token_ids:
            raise RuntimeError("request has not generated any output tokens")
        return self._output_token_ids[-1]

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def is_terminal(self) -> bool:
        return self.state in {RequestState.FINISHED, RequestState.ABORTED}

    def start_prefill(self) -> None:
        self._transition_to(RequestState.PREFILL)

    def start_decode(self) -> None:
        self._transition_to(RequestState.DECODING)

    # 接收模型刚生成的一个 token，记录它，判断请求是否结束，并生成增量输出事件。
    def record_token(self, token_id: int) -> IncrementalOutput:
        if self.state is not RequestState.DECODING:
            raise InvalidStateTransition(
                f"request {self.request_id!r} cannot record a token while {self.state.value}"
            )
        _validate_token_id(token_id)

        self._output_token_ids.append(token_id)
        reason = self._finish_reason(token_id)
        if reason is not None:
            self._transition_to(RequestState.FINISHED)

        return IncrementalOutput(
            request_id=self.request_id,
            token_id=token_id,
            output_index=self.num_output_tokens - 1,
            finished=reason is not None,
            finish_reason=reason,
        )

    def abort(self) -> IncrementalOutput:
        self._transition_to(RequestState.ABORTED)
        return IncrementalOutput(
            request_id=self.request_id,
            token_id=None,
            output_index=self.num_output_tokens,
            finished=True,
            finish_reason=FinishReason.ABORTED,
        )

    def _finish_reason(self, token_id: int) -> FinishReason | None:
        params = self.sampling_params
        if (
            not params.ignore_eos
            and params.eos_token_id is not None
            and token_id == params.eos_token_id
        ):
            return FinishReason.EOS
        if self.num_output_tokens >= params.max_new_tokens:
            return FinishReason.LENGTH
        return None

    # 所有状态转移都经过这里验证，避免请求跳过必要阶段。
    def _transition_to(self, new_state: RequestState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                f"invalid request transition: {self.state.value} -> {new_state.value}"
            )
        self._state = new_state


def _validate_prompt(prompt_token_ids: Iterable[int]) -> tuple[int, ...]:
    token_ids = tuple(prompt_token_ids)
    if not token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids):
        raise TypeError("prompt_token_ids must contain integers")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("prompt token IDs must be non-negative")
    return token_ids


def _validate_token_id(token_id: int, *, name: str = "token_id") -> None:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise TypeError(f"{name} must be an integer")
    if token_id < 0:
        raise ValueError(f"{name} must be non-negative")
