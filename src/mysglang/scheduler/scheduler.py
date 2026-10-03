"""
每个 rank 都有一个自己的 Scheduler 实例；rank 0 的 TensorParallelScheduler
负责向其他 rank 广播命令，让这些 Scheduler 保持镜像一致。
rank 0                                      rank 1
  ──────────────────────                      ──────────────────────
  GenerationService                           不创建 Service
          │
  TensorParallelScheduler                     run_worker_loop()
          │                                          │
  广播 add / step / abort ──────────────────────────>│
          │                                          │
  本地 Scheduler 0                           本地 Scheduler 1
  本地 Radix/页表                            本地 Radix/页表
  GPU 0 的局部 KV pool                       GPU 1 的局部 KV pool
          │                                          │
          └──────── 同步进入 model forward ──────────┘
                              │
                        TP all-reduce
其他 rank 不需要这些东西，只进入：

tp_scheduler.run_worker_loop()

等待 rank 0 发命令
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, replace

import torch

from mysglang.cache import (
    HybridPagedKVCache,
    PagedKVCache,
    PagedKVDecodeBuffer,
    RadixPagedKVCache,
)
from mysglang.core import IncrementalOutput, Request, RequestState
from mysglang.modeling.cuda_graph import DecodeCudaGraphRunner, MtpCudaGraphRunner
from mysglang.modeling.qwen3 import Qwen3ForCausalLM

from .config import SchedulerConfig
from .coordination import SchedulerBatchPlan, SchedulerCoordinator
from .ngram import NGramProposer
from .sampler import sample_token, sampling_probs, speculative_accept


@dataclass(frozen=True)
class SchedulerStep:
    index: int
    phase: str
    request_ids: tuple[str, ...]
    input_tokens: int
    outputs: tuple[IncrementalOutput, ...]

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)


@dataclass(frozen=True)
class SchedulerStats:
    active_requests: int
    waiting_requests: int
    running_requests: int
    max_active_requests: int
    finished_requests: int
    aborted_requests: int
    model_forwards: int
    max_prefill_batch_size: int
    max_decode_batch_size: int
    max_wait_steps: int
    prefill_input_tokens: int
    cuda_graph_captures: int
    cuda_graph_replays: int
    speculative_verify_forwards: int
    speculative_draft_tokens: int
    speculative_accepted_tokens: int
    # MTP 头自己的前向次数（递归草稿步 + 每轮之后的追赶步），不计入 model_forwards。
    speculative_draft_forwards: int = 0
    preempted_requests: int = 0


@dataclass
class _Entry:#request外包装
    request: Request
    # 请求进入 Scheduler 的 step，用来统计排队时间。
    enqueue_step: int
    # Prompt 已经 Prefill 到哪里；长 Prompt 会按 budget 分块处理。
    prefill_offset: int = 0
    # 这次 prefill 要送进模型的 token：首次是 prompt；被抢占后重算是 prompt + 已生成的 token。
    prefill_tokens: tuple[int, ...] = ()
    # MTP：最近一次追赶前向在最后位置给出的下一轮首个草稿，以及该位置的 MTP hidden。
    mtp_draft: int | None = None
    mtp_hidden: torch.Tensor | None = None
    # 随机采样请求的草稿分布 q；None 表示确定性草稿（greedy argmax 或 n-gram）。
    mtp_draft_prob: torch.Tensor | None = None
    # 图片请求的视觉特征（每个图片占位 token 一行），首次 Prefill 时编码，Prompt 完成后释放。
    vision_features: torch.Tensor | None = None


@dataclass(frozen=True)
class _MTPSegment:
    """Rows of one target forward that the MTP head must now absorb.

    ``tokens`` are the tokens *following* each row (the next prompt token, or the one
    just sampled/accepted); ``needs_draft`` marks requests that continue decoding.
    """

    entry: _Entry
    row_start: int
    tokens: tuple[int, ...]
    needs_draft: bool


class Scheduler:
    """The single continuous-batching scheduler used by MySGLang.

    It owns admission, chunked prefill, batched decode and the paged/radix KV
    cache. There is deliberately no selectable legacy backend anymore.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        config: SchedulerConfig,
        *,
        coordinator: SchedulerCoordinator | None = None,
    ) -> None:
        if model.tensor_parallel.enabled and coordinator is None:
            raise RuntimeError(
                "Tensor-parallel models require the distributed worker runtime; "
                "the single-process Scheduler cannot drive collectives safely"
            )
        self._mtp_steps = config.speculative_mtp_steps
        if self._mtp_steps and model.mtp is None:
            raise ValueError("MTP speculative decoding requires a model loaded with enable_mtp")
        recurrent_spec = model.recurrent_state_spec
        self.model = model.eval()
        self.config = config
        self._coordinator = coordinator
        parameter = model.embed_tokens.weight
        cache_arguments = dict(
            num_layers=model.num_kv_cache_layers,
            num_pages=config.num_pages,
            page_size=config.page_size,
            num_kv_heads=model.kv_cache_num_heads,
            head_dim=model.config.head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            # 开 Decode Graph 时多开 1 页垃圾页，给按桶补齐的假请求写 KV。
            scratch_pages=1 if config.decode_cuda_graph_batch_sizes else 0,
        )
        self.cache: RadixPagedKVCache
        if recurrent_spec is not None or self._mtp_steps:
            # 每个运行中的请求占一个 DeltaNet 状态 slot，所以 slot 数等于并发上限。
            self.cache = HybridPagedKVCache(
                recurrent=recurrent_spec,
                num_recurrent_slots=config.max_running_requests,
                num_draft_layers=model.num_draft_kv_layers if self._mtp_steps else 0,
                **cache_arguments,
            )
        else:
            self.cache = RadixPagedKVCache(**cache_arguments)
        self.model.validate_cache(self.cache)
        self._device = parameter.device
        self._decode_max_blocks = min(
            self.cache.num_pages,
            self.cache.allocator.pages_for_tokens(model.config.max_position_embeddings),
        )
        self._decode_buffers: dict[int, PagedKVDecodeBuffer] = {}
        self._cuda_graph_runner = (#不开投机：普通 decode
            DecodeCudaGraphRunner(
                self.model,
                self.cache,
                config.decode_cuda_graph_batch_sizes,
                max_blocks=self._decode_max_blocks,
                return_hidden=bool(self._mtp_steps),
                pad_to_bucket=True,
            )
            if config.decode_cuda_graph_batch_sizes
            else None
        )
        if self._cuda_graph_runner is not None:
            # 启动时就把所有 Decode 桶录好；TP 各 rank 按相同顺序录，Graph 内的 NCCL 保持一致。
            self._cuda_graph_runner.capture_all()
        # MTP 每轮 verify 的形状固定为 [batch, 1 + k]，同样可以按精确 batch size 捕获。
        self._verify_graph_runner = (#开投机 ①：主模型验证
            DecodeCudaGraphRunner(
                self.model,#开投机 ②：S 的掩码重放 _verify_graph_runner 里的 commit_graph
                self.cache,
                config.decode_cuda_graph_batch_sizes,
                max_blocks=self._decode_max_blocks,
                append_length=1 + self._mtp_steps,
                return_hidden=True,
            )
            if config.decode_cuda_graph_batch_sizes and self._mtp_steps
            else None
        )
        # MTP 头自己的两种固定形状：递归草稿步 [batch, 1] 与满宽追赶步 [batch, 1 + k]。
        self._mtp_draft_graph = None
        self._mtp_catch_up_graph = None
        if config.decode_cuda_graph_batch_sizes and self._mtp_steps:
            draft_cache = self._draft_cache
            if self._mtp_steps > 1:
                self._mtp_draft_graph = MtpCudaGraphRunner(#开投机：猜 2 个以上时的递归猜测	_mtp_draft_graph
                    self.model,
                    draft_cache,
                    config.decode_cuda_graph_batch_sizes,
                    max_blocks=self._decode_max_blocks,
                )
            self._mtp_catch_up_graph = MtpCudaGraphRunner(#开投机 ③：MTP 追赶
                self.model,
                draft_cache,
                config.decode_cuda_graph_batch_sizes,
                max_blocks=self._decode_max_blocks,
                append_length=1 + self._mtp_steps,
            )
        self._waiting: deque[_Entry] = deque()#三个存放请求的地方
        self._prefilling: dict[str, _Entry] = {}
        self._running: dict[str, _Entry] = {}
        # 所有未结束请求的总索引：waiting + prefilling + running。
        self._entries: dict[str, _Entry] = {}
        self._sampling_generators: dict[str, torch.Generator] = {}
        self._step_index = 0
        self._max_active_requests = 0
        self._finished_requests = 0
        self._aborted_requests = 0
        self._model_forwards = 0
        self._max_prefill_batch_size = 0
        self._max_decode_batch_size = 0
        self._max_wait_steps = 0
        # 只保留累计量，不保存无限增长的逐步 history。
        self._prefill_input_tokens = 0
        self._ngram_proposer = NGramProposer(
            min_match=config.speculative_ngram_min_match,
            max_match=config.speculative_ngram_max_match,
        )
        self._speculative_verify_forwards = 0
        self._speculative_draft_tokens = 0
        self._speculative_accepted_tokens = 0
        self._speculative_draft_forwards = 0
        self._preempted_requests = 0
        # 本步刚抢占过就不再准入新请求，否则刚腾出的页马上又被占满，来回抖动。
        self._preempted_this_step = False
        self._closed = False

    @property
    def has_work(self) -> bool:
        return bool(self._entries)

    def close(self) -> None:
        """Release runtime-only resources while CUDA/NCCL are still available."""
        if self._closed:
            return
        if self.has_work:
            raise RuntimeError("cannot close Scheduler while requests are active")
        for runner in self._graph_runners:
            runner.close()
        self._closed = True

    @property
    def _graph_runners(self) -> list[DecodeCudaGraphRunner]:
        return [
            runner
            for runner in (
                self._cuda_graph_runner,
                self._verify_graph_runner,
                self._mtp_draft_graph,
                self._mtp_catch_up_graph,
            )
            if runner is not None
        ]

    @property
    def prefill_input_tokens(self) -> int:
        return self._prefill_input_tokens

    @property
    def stats(self) -> SchedulerStats:
        runners = self._graph_runners
        captures = sum(runner.stats.captures for runner in runners)
        replays = sum(runner.stats.replays for runner in runners)
        return SchedulerStats(
            active_requests=len(self._entries),
            waiting_requests=len(self._waiting) + len(self._prefilling),
            running_requests=len(self._running),
            max_active_requests=self._max_active_requests,
            finished_requests=self._finished_requests,
            aborted_requests=self._aborted_requests,
            model_forwards=self._model_forwards,
            max_prefill_batch_size=self._max_prefill_batch_size,
            max_decode_batch_size=self._max_decode_batch_size,
            max_wait_steps=self._max_wait_steps,
            prefill_input_tokens=self._prefill_input_tokens,
            cuda_graph_captures=captures,
            cuda_graph_replays=replays,
            speculative_verify_forwards=self._speculative_verify_forwards,
            speculative_draft_tokens=self._speculative_draft_tokens,
            speculative_accepted_tokens=self._speculative_accepted_tokens,
            speculative_draft_forwards=self._speculative_draft_forwards,
            preempted_requests=self._preempted_requests,
        )

    def validate(self, request: Request) -> None:
        """Validate a request without enqueueing or allocating cache pages."""
        if request.state is not RequestState.WAITING:
            raise ValueError("scheduler only accepts requests in WAITING state")
        total_length = self._max_request_tokens(request)
        if total_length > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")
        if any(token_id >= self.model.config.vocab_size for token_id in request.prompt_token_ids):
            raise ValueError("prompt token IDs must be smaller than model vocab_size")
        params = request.sampling_params
        stops = set(params.stop_token_ids)
        if params.eos_token_id is not None:
            stops.add(params.eos_token_id)
        if any(token_id >= self.model.config.vocab_size for token_id in stops):
            raise ValueError("eos/stop token IDs must be smaller than model vocab_size")
        needed_pages = self.cache.allocator.pages_for_tokens(total_length)
        if needed_pages > self.cache.num_pages:
            raise ValueError("request exceeds total paged KV cache capacity")
        if request.multimodal is not None:
            if self.model.visual is None:
                raise ValueError("image input requires a model loaded with enable_vision")
            if self.cache.enable_prefix_cache:
                # 图片占位 token 的 ID 都相同，按 token 前缀复用会把不同图片的 K/V 混用。
                raise ValueError("image input requires prefix caching to be disabled")

    def add(self, request: Request) -> None:
        self.validate(request)
        if request.request_id in self._entries:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        # add 只进入逻辑等待队列；真正的 reservation/page allocation 在被调度时发生。
        entry = _Entry(request=request, enqueue_step=self._step_index)
        self._waiting.append(entry)
        self._entries[request.request_id] = entry
        self._max_active_requests = max(self._max_active_requests, len(self._entries))

    def abort(self, request_id: str) -> IncrementalOutput | None:
        entry = self._entries.get(request_id)
        if entry is None or entry.request.is_terminal:
            return None
        if request_id in self._prefilling:
            self._prefilling.pop(request_id)
        else:
            try:
                self._waiting.remove(entry)
            except ValueError:
                pass
        self._running.pop(request_id, None)
        event = entry.request.abort()
        self._release(entry, finished=False)
        return event

    @torch.inference_mode()
    def step(self) -> SchedulerStep | None:
        """Include every running decode in a prefill step, or run dense decode."""
        self._preempted_this_step = False
        if self.config.enable_preemption:
            self._make_room_for_decode()
        if self._can_prefill():
            result = self._prefill_step()
        elif self._running:
            result = self._decode_step()
        else:
            return None

        self._step_index += 1
        return result

    def reset_prefix_cache(self) -> None:
        self.cache.reset_prefix_cache()

    def check_integrity(self) -> None:
        self.cache.check_integrity()
        cache_requests = set(self.cache.allocator.request_ids)
        scheduled_requests = set(self._running) | set(self._prefilling)
        if cache_requests != scheduled_requests:
            raise RuntimeError("scheduler and paged cache disagree about active owners")
        waiting_ids = {entry.request.request_id for entry in self._waiting}
        if waiting_ids & cache_requests:
            raise RuntimeError("a waiting request already owns physical cache pages")
        if set(self._entries) != waiting_ids | scheduled_requests:
            raise RuntimeError("scheduler request indexes disagree")
        if any(entry.request.state is not RequestState.WAITING for entry in self._waiting):
            raise RuntimeError("waiting queue contains a request in the wrong state")
        if any(
            entry.request.state is not RequestState.PREFILL for entry in self._prefilling.values()
        ):
            raise RuntimeError("prefill set contains a request in the wrong state")
        if any(
            entry.request.state is not RequestState.DECODING for entry in self._running.values()
        ):
            raise RuntimeError("running map contains a request in the wrong state")
        if self._mtp_steps:
            draft = self._draft_cache
            for request_id in self._running:
                # 两步之间，MTP 草稿层必须恰好追平目标模型，并已备好下一轮首个草稿。
                if draft.lengths((request_id,)) != self.cache.lengths((request_id,)):
                    raise RuntimeError("MTP draft K/V is not aligned with the target cache")
                if self._running[request_id].mtp_draft is None:
                    raise RuntimeError("running MTP request has no pending draft token")

    @property
    def _draft_cache(self) -> PagedKVCache:
        draft = getattr(self.cache, "draft", None)
        if draft is None:
            raise RuntimeError("the scheduler cache has no MTP draft layers")
        return draft

    def _can_prefill(self) -> bool:
        if self._prefilling:
            return True
        if not self._waiting:
            return False
        if self.cache.allocator.stats.request_count >= self.config.max_running_requests:
            return False
        if self._preempted_this_step:
            return False
        request = self._waiting[0].request
        return self.cache.can_reserve_request(
            self._max_request_tokens(request),
            request.all_token_ids,
            self._admission_tokens(request),
        )

    def _admission_tokens(self, request: Request) -> int | None:
        """准入时额度覆盖多少 token；None 表示按 prompt + max_new_tokens 最坏情况一次预留。"""
        # 图片请求重算要重新处理 M-RoPE 与视觉替换，暂不参与抢占：仍按最坏情况预留。
        if not self.config.enable_preemption or request.multimodal is not None:
            return None
        return len(request.all_token_ids)

    def _prefill_step(self) -> SchedulerStep:
        # 只快照本轮开始时的 Decode；刚完成 Prompt 的请求下轮才输入首个输出 token。  prefill和decode拼桌
        decoding = list(self._running.values())#可能一边推理一边从队列里面删除，所以要单开一列
        self._admit_prefill_requests()
        entries = list(self._prefilling.values())
        selected = entries[: min(len(entries), self.config.prefill_token_budget)]
        remaining_budget = self.config.prefill_token_budget
        chunks: list[tuple[_Entry, int, int]] = []
        for index, entry in enumerate(selected):#尽量均分prefill，如果前面的请求 prompt 很短，用不完自己那份，剩下的会自动让给后面的请求
            requests_left = len(selected) - index
            fair_share = max(1, remaining_budget // requests_left)
            start = entry.prefill_offset
            end = min(start + fair_share, len(entry.prefill_tokens))
            chunks.append((entry, start, end))
            remaining_budget -= end - start

        decode_ids = tuple(entry.request.request_id for entry in decoding)
        request_ids = decode_ids + tuple(entry.request.request_id for entry, _, _ in chunks)#本步参与计算的所有请求"，decode 的排在前面，prefill 的排在后面
        append_lengths = (1,) * len(decoding) + tuple(end - start for _, start, end in chunks)#分段
        packed_tokens = [entry.request.last_output_token_id for entry in decoding]
        decode_starts = self.cache.lengths(decode_ids) if decoding else ()
        for request_id, length in zip(decode_ids, decode_starts):
            self.cache.ensure_capacity(request_id, length + 1)
        for entry, start, end in chunks:
            request_id = entry.request.request_id
            # reservation 是容量承诺；这里才从 free pages 取物理页并写入请求页表。
            self.cache.ensure_capacity(request_id, end)
            packed_tokens.extend(entry.prefill_tokens[start:end])

        input_ids = torch.tensor(
            packed_tokens,
            dtype=torch.long,
            device=self._device,
        )
        # Decode 每个位置都采样；Prefill 仅在完整 Prompt 结束时采样。
        logits_indices = list(range(len(decoding)))# decode 请求：每个都采样
        packed_offset = len(decoding)
        for entry, start, end in chunks:
            packed_offset += end - start
            if end == len(entry.prefill_tokens): # 这个请求的 prompt 算完了
                logits_indices.append(packed_offset - 1)
        phase = "mixed" if decoding else "prefill"
        self._validate_batch_plan(
            phase=phase,
            execution="packed",
            request_ids=request_ids,
            input_token_ids=tuple(packed_tokens),
            append_lengths=append_lengths,
            logits_indices=tuple(logits_indices),
        )
        rows = [(entry, start, 1) for entry, start in zip(decoding, decode_starts)]
        rows.extend((entry, start, end - start) for entry, start, end in chunks)
        logits, hidden = self._split_hidden(
            self.model.forward_packed(
                input_ids,
                kv_cache=self.cache,
                cache_request_ids=request_ids,
                append_lengths=append_lengths,
                logits_indices=tuple(logits_indices),
                return_hidden=bool(self._mtp_steps),
                mrope_positions=self._mrope_positions(rows),
                embedding_overrides=self._vision_overrides(rows),
            )
        )
        self._model_forwards += 1
        input_token_count = sum(append_lengths)
        self._prefill_input_tokens += input_token_count - len(decoding)
        self._max_prefill_batch_size = max(self._max_prefill_batch_size, len(chunks))
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(decoding))

        completed_prefills = [
            entry for entry, _start, end in chunks if end == len(entry.prefill_tokens)
        ]
        sampled_entries = decoding + completed_prefills#按顺序和 logits_indices 一一对应
        next_token_ids = self._sync_token_ids(
            tuple(
                self._sample(logits[index], entry.request)
                for index, entry in enumerate(sampled_entries)
            )
        )

        if self._mtp_steps:
            # MTP 在同一批位置上追赶：每行配对“该位置之后的 token”（下一个 prompt token
            # 或刚采样出的 token），保持草稿 K/V 与目标模型逐位置对齐。
            segments = [
                _MTPSegment(entry, row, (token,), True)
                for row, (entry, token) in enumerate(zip(decoding, next_token_ids))
            ]
            row = len(decoding)
            sampled = len(decoding)
            for entry, start, end in chunks:
                prompt = entry.prefill_tokens
                if end == len(prompt):
                    following = next_token_ids[sampled]
                    sampled += 1
                else:
                    following = prompt[end]
                segments.append(
                    _MTPSegment(
                        entry,
                        row,
                        tuple(prompt[start + 1 : end]) + (following,),
                        end == len(prompt),
                    )
                )
                row += end - start
            self._mtp_catch_up(segments, hidden)
        for entry, _start, end in chunks:
            if end == len(entry.prefill_tokens):
                entry.vision_features = None

        outputs = []
        for entry, next_token in zip(decoding, next_token_ids[: len(decoding)]):#收尾，先处理decoding
            event = entry.request.record_token(next_token)
            outputs.append(event)
            if event.finished:
                self._running.pop(entry.request.request_id)
                self._release(entry, finished=True)
        token_offset = len(decoding)
        for entry, start, end in chunks:
            request = entry.request
            request_id = request.request_id
            entry.prefill_offset = end
            self._prefilling.pop(request_id)

            if end == len(entry.prefill_tokens):
                # Prompt 的完整页现在就发布并锁住；无需等长 Decode 全部结束即可复用。
                self.cache.publish_prefix(request_id, entry.prefill_tokens)
                request.start_decode()
                next_token = next_token_ids[token_offset]
                token_offset += 1
                event = request.record_token(next_token)
                outputs.append(event)
                if event.finished:
                    self._release(entry, finished=True)
                else:
                    self._running[request_id] = entry
            else:
                # 被选中但尚未完成的长请求移到队尾，避免小 budget 下独占 Prefill。
                self._prefilling[request_id] = entry

        return SchedulerStep(
            index=self._step_index,
            phase=phase,
            request_ids=request_ids,
            input_tokens=input_token_count,
            outputs=tuple(outputs),
        )

    def _admit_prefill_requests(self) -> None:
        while (
            self._waiting
            and not self._preempted_this_step
            and self.cache.allocator.stats.request_count < self.config.max_running_requests
        ):
            entry = self._waiting[0]
            request = entry.request
            # 首次准入是 prompt；被抢占过的请求还要带上已生成的 token 一起重算。
            tokens = request.all_token_ids
            reserved_tokens = self._admission_tokens(request)
            if not self.cache.can_reserve_request(
                self._max_request_tokens(request),
                tokens,
                reserved_tokens,
            ):
                break
            matched = self.cache.reserve_request_with_prefix(
                request.request_id,
                self._max_request_tokens(request),
                tokens,
                reserved_tokens,
            )
            if matched is None:
                raise RuntimeError("prefill selected without enough radix cache capacity")
            self._waiting.popleft()
            request.start_prefill()
            entry.prefill_tokens = tokens
            entry.prefill_offset = matched
            self._prefilling[request.request_id] = entry
            self._max_wait_steps = max(self._max_wait_steps, self._step_index - entry.enqueue_step)

    def _decode_growth(self) -> int:
        """一个运行中的请求这一步最多新写几个 token：1 个输入 + 投机的草稿。"""
        return 1 + max(self._mtp_steps, self.config.speculative_ngram_max_tokens)

    def _make_room_for_decode(self) -> None:
        """保证每个运行中的请求这一步都能再放 _decode_growth() 个 token；不够就抢占。"""
        growth = self._decode_growth()
        for entry in list(self._running.values()):  # 按开始 decode 的先后检查
            request = entry.request
            if request.request_id not in self._running:
                continue  # 前面已经被抢占掉了
            length = self.cache.lengths((request.request_id,))[0]
            need = min(length + growth, self._max_request_tokens(request))
            while not self.cache.grow_reservation(request.request_id, need):
                # 图片请求按最坏情况预留、从不需要增长，也不当牺牲品；
                # 额度不够的这个请求本身一定是文字请求，所以 candidates 不会为空。
                candidates = [e for e in self._running.values() if e.request.multimodal is None]
                victim = candidates[-1]  # 最后准入的：算得最少，重算最便宜
                self._preempt(victim)
                if victim is entry:
                    break  # 把自己抢占了，不用再给自己腾地方

    def _preempt(self, entry: _Entry) -> None:
        """把一个运行中的请求退回等待队首；它的 entry、随机数生成器都保留。"""
        request = entry.request
        request_id = request.request_id
        self._running.pop(request_id)
        request.preempt()
        # 先把写满的页插进 radix 再释放，重算时能命中自己的前缀；
        # 混合模型关着前缀缓存，就是从头算。
        self.cache.finish_request(request_id, request.all_token_ids)
        entry.prefill_offset = 0
        entry.prefill_tokens = ()
        entry.mtp_draft = entry.mtp_hidden = entry.mtp_draft_prob = None
        self._waiting.appendleft(entry)
        self._preempted_requests += 1
        self._preempted_this_step = True

    def _decode_step(self) -> SchedulerStep:
        entries = list(self._running.values())
        if self._mtp_steps:
            drafts, draft_probs = self._mtp_drafts(entries)
        else:
            drafts = tuple(self._ngram_draft(entry.request) for entry in entries)
            draft_probs = tuple(() for _ in entries)
        if any(drafts):
            return self._speculative_decode_step(entries, drafts, draft_probs)
        return self._single_token_decode_step(entries)

    def _single_token_decode_step(self, entries: list[_Entry]) -> SchedulerStep:
        request_ids = tuple(entry.request.request_id for entry in entries)
        lengths = self.cache.lengths(request_ids) # 每个请求 KV 已有多长
        for request_id, length in zip(request_ids, lengths):
            self.cache.ensure_capacity(request_id, length + 1) # 保证能再放 1 个 token
        input_token_ids = tuple(entry.request.last_output_token_id for entry in entries)
        use_cuda_graph = bool(
            self._cuda_graph_runner and self._cuda_graph_runner.supports(len(entries))
        )
        self._validate_batch_plan(
            phase="decode",
            execution="decode-graph" if use_cuda_graph else "decode-eager",
            request_ids=request_ids,
            input_token_ids=input_token_ids,
            append_lengths=(1,) * len(entries),
            logits_indices=None,
        )
        hidden = None
        if use_cuda_graph:
            # 满足固定 Decode bucket 条件后，从这里进入 Graph replay。
            graph_result = self._cuda_graph_runner.run_positions(
                input_token_ids, request_ids, rope_deltas=self._rope_deltas(entries)
            )
            hidden = graph_result.hidden
            # 图内已算好 argmax；需要采样或 penalty 的请求改读图输出的 logits。
            next_token_ids = tuple(
                tokens[-1]
                if self._plain_greedy(entry.request)
                else self._sample(graph_result.logits[index, -1], entry.request)
                for index, (entry, tokens) in enumerate(zip(entries, graph_result.token_ids))
            )
        else:
            input_ids = torch.tensor(
                input_token_ids,
                dtype=torch.long,
                device=self._device,
            ).view(len(entries), 1)
            buffer = self._decode_buffers.get(len(entries))
            if buffer is None:
                buffer = self.cache.allocate_decode_buffer(
                    len(entries),
                    max_blocks=self._decode_max_blocks,
                )
                self._decode_buffers[len(entries)] = buffer
            batch = self.cache.prepare_decode_batch(
                request_ids, buffer, rope_deltas=self._rope_deltas(entries)
            )
            logits, hidden = self._split_hidden(
                self.model.forward_prepared(
                    input_ids,
                    kv_cache=self.cache,
                    cache_batch=batch,
                    return_hidden=bool(self._mtp_steps),
                )
            )
            next_token_ids = tuple(
                self._sample(logits[index, -1], entry.request)
                for index, entry in enumerate(entries)
            )
        next_token_ids = self._sync_token_ids(next_token_ids)
        self._model_forwards += 1
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(entries))

        if self._mtp_steps:
            self._mtp_catch_up(
                [
                    _MTPSegment(entry, row, (token,), True)
                    for row, (entry, token) in enumerate(zip(entries, next_token_ids))
                ],
                hidden,
            )

        outputs = []
        for entry, next_token in zip(entries, next_token_ids):
            event = entry.request.record_token(next_token)
            outputs.append(event)
            if event.finished:
                self._running.pop(entry.request.request_id)
                self._release(entry, finished=True)

        return SchedulerStep(
            index=self._step_index,
            phase="decode",
            request_ids=request_ids,
            input_tokens=len(entries),
            outputs=tuple(outputs),
        )

    def _speculative_decode_step(
        self,
        entries: list[_Entry],
        drafts: tuple[tuple[int, ...], ...],
        draft_probs: tuple[tuple[torch.Tensor | None, ...], ...],
    ) -> SchedulerStep:
        """Verify ragged draft chains (n-gram or MTP) with one packed target forward."""
        request_ids = tuple(entry.request.request_id for entry in entries)
        starts = self.cache.lengths(request_ids)
        append_lengths = tuple(1 + len(draft) for draft in drafts)
        packed_tokens: list[int] = []
        for entry, draft, start, append_length in zip(
            entries, drafts, starts, append_lengths
        ):
            self.cache.ensure_capacity(entry.request.request_id, start + append_length)
            packed_tokens.append(entry.request.last_output_token_id)
            packed_tokens.extend(draft)

        use_graph = (
            self._verify_graph_runner is not None
            and self._verify_graph_runner.supports(len(entries))
            and all(length == 1 + self._mtp_steps for length in append_lengths)
        )
        self._validate_batch_plan(
            phase="decode",
            execution=(
                "mtp-verify-graph"
                if use_graph
                else "mtp-verify" if self._mtp_steps else "ngram-verify"
            ),
            request_ids=request_ids,
            input_token_ids=tuple(packed_tokens),
            append_lengths=append_lengths,
            logits_indices=None,
        )
        logits = None
        greedy_rows: tuple[tuple[int, ...], ...] | None = None
        if use_graph:
            # 所有请求都是满额 greedy 草稿：[batch, 1 + k] 形状固定，直接 replay。
            graph_result = self._verify_graph_runner.run_positions(
                tuple(packed_tokens), request_ids, rope_deltas=self._rope_deltas(entries)
            )
            greedy_rows = graph_result.token_ids
            hidden = graph_result.hidden
            state_batch = graph_result.state_batch
        else:
            # DeltaNet 状态不能事后截断：verify 只读 slot 并记录每个 token 的输入，
            # 接受长度确定后再从原状态 replay 被接受的前缀。
            recurrent = getattr(self.cache, "recurrent", None)
            state_batch = (
                None
                if recurrent is None
                else recurrent.prepare_batch(request_ids, append_lengths, speculative=True)
            )
            logits, hidden = self._split_hidden(
                self.model.forward_packed(
                    torch.tensor(packed_tokens, dtype=torch.long, device=self._device),
                    kv_cache=self.cache,
                    cache_request_ids=request_ids,
                    append_lengths=append_lengths,
                    logits_indices=None,
                    state_batch=state_batch,
                    return_hidden=bool(self._mtp_steps),
                    mrope_positions=self._mrope_positions(
                        list(zip(entries, starts, append_lengths))
                    ),
                )
            )

        proposed_outputs: list[tuple[int, ...]] = []
        offset = 0
        for index, (entry, draft, probs, append_length) in enumerate(
            zip(entries, drafts, draft_probs, append_lengths)
        ):
            request = entry.request
            if greedy_rows is not None and self._plain_greedy(request):
                # 图内 argmax 已足够：无 penalty 的 greedy 不必再读整行 logits。
                candidate = self._accept_greedy(draft, greedy_rows[index])
            else:
                if greedy_rows is not None:
                    request_logits = graph_result.logits[index]
                else:
                    request_logits = logits[offset : offset + append_length]
                # A rejection contributes the target correction at that position;
                # accepting every draft contributes the final bonus token instead.
                candidate = speculative_accept(
                    draft,
                    probs,
                    request_logits,
                    request.sampling_params,
                    request.output_token_ids,
                    generator=self._generator(request),
                )
            offset += append_length
            proposed_outputs.append(self._truncate_speculative_output(request, candidate))

        emitted = self._sync_speculative_outputs(tuple(proposed_outputs), append_lengths)
        accepted_counts = tuple(
            self._accepted_prefix_length(output, draft)
            for output, draft in zip(emitted, drafts)
        )
        # The model wrote every candidate K/V. Keep exactly the prefix preceding
        # the newest emitted token and make rejected candidate slots unreachable.
        for request_id, start, output in zip(request_ids, starts, emitted):
            self.cache.truncate(request_id, start + len(output))
        if state_batch is not None:
            # 被吸收的输入恰好是 [last_token] + 已接受草稿，数量等于本轮输出 token 数。
            absorbed = tuple(len(output) for output in emitted)
            if use_graph:
                self._verify_graph_runner.commit_states(state_batch, absorbed)
            else:
                self.model.commit_speculative_states(state_batch, absorbed)

        self._model_forwards += 1
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(entries))
        self._speculative_verify_forwards += 1
        self._speculative_draft_tokens += sum(map(len, drafts))
        self._speculative_accepted_tokens += sum(accepted_counts)

        if self._mtp_steps:
            # 草稿步写到了 start 之后；丢弃它们，改用目标 hidden 追赶被接受的位置。
            draft_cache = self._draft_cache
            for request_id, start in zip(request_ids, starts):
                draft_cache.truncate(request_id, start)
            if (
                use_graph
                and self._mtp_catch_up_graph is not None
                and self._mtp_catch_up_graph.supports(len(entries))
            ):
                self._mtp_catch_up_full_width(entries, emitted, starts, hidden)
            else:
                segments = []
                row = 0
                for entry, output, append_length in zip(entries, emitted, append_lengths):
                    segments.append(_MTPSegment(entry, row, output, True))
                    row += append_length
                self._mtp_catch_up(segments, hidden)

        outputs = []
        for entry, token_ids in zip(entries, emitted):
            for token_id in token_ids:
                event = entry.request.record_token(token_id)
                outputs.append(event)
                if event.finished:
                    self._running.pop(entry.request.request_id)
                    self._release(entry, finished=True)
                    break

        return SchedulerStep(
            index=self._step_index,
            phase="decode",
            request_ids=request_ids,
            input_tokens=sum(append_lengths),
            outputs=tuple(outputs),
        )

    def _draft_budget(self, request: Request, limit: int) -> int:
        """How many draft tokens this request may verify in one round.

        Greedy requests compare with the (penalized) target argmax; sampled requests use
        speculative sampling, so both keep exactly their non-speculative distribution.
        """
        if limit == 0:
            return 0
        # A k-token draft may return k accepted tokens plus one bonus token.
        params = request.sampling_params
        remaining = params.max_new_tokens - request.num_output_tokens
        return max(0, min(limit, remaining - 1))

    @staticmethod
    def _plain_greedy(request: Request) -> bool:
        params = request.sampling_params
        return params.is_greedy and not params.has_penalties

    def _propose(
        self,
        request: Request,
        logits: torch.Tensor,
    ) -> tuple[int, torch.Tensor | None]:
        """One MTP draft token and its distribution q (None for a greedy draft)."""
        params = request.sampling_params
        if params.is_greedy:
            return int(logits.argmax().item()), None
        # q 不叠加 penalty：投机采样对任意 q 都精确，只影响接受率。
        probabilities = sampling_probs(logits, params, apply_request_penalties=False)
        token = int(torch.multinomial(probabilities, 1, generator=self._generator(request)))
        return token, probabilities

    def _ngram_draft(self, request: Request) -> tuple[int, ...]:
        max_tokens = self._draft_budget(request, self.config.speculative_ngram_max_tokens)
        if max_tokens <= 0:
            return ()
        return self._ngram_proposer.propose(request.all_token_ids, max_tokens)

    def _mtp_drafts(
        self,
        entries: list[_Entry],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[torch.Tensor | None, ...], ...]]:
        """Recursively extend each request's pending MTP draft to its budget.

        Draft ``j + 1`` feeds draft ``j`` and the MTP hidden that produced it back into the
        MTP head at the next position. The target hidden there is unknown until
        verification, so the draft chain approximates it with the MTP's own output.
        """
        budgets = [self._draft_budget(entry.request, self._mtp_steps) for entry in entries]
        if not any(budgets):
            return ((),) * len(entries), ((),) * len(entries)
        drafts: list[list[int]] = []
        probs: list[list[torch.Tensor | None]] = []
        hidden: list[torch.Tensor | None] = []
        for entry, budget in zip(entries, budgets):
            if budget and (entry.mtp_draft is None or entry.mtp_hidden is None):
                raise RuntimeError("MTP request has no pending draft to extend")
            drafts.append([entry.mtp_draft] if budget else [])
            probs.append([entry.mtp_draft_prob] if budget else [])
            hidden.append(entry.mtp_hidden if budget else None)
        request_ids = [entry.request.request_id for entry in entries]
        for request_id, start, budget in zip(
            request_ids, self.cache.lengths(tuple(request_ids)), budgets
        ):
            # 草稿步与随后的 verify 共享页；一次保证到 verify 需要的最远位置。
            self.cache.ensure_capacity(request_id, start + 1 + budget)

        draft_cache = self._draft_cache
        for depth in range(1, max(budgets)):
            active = [index for index, budget in enumerate(budgets) if budget > depth]
            active_ids = tuple(request_ids[index] for index in active)
            tokens = tuple(drafts[index][-1] for index in active)
            self._validate_batch_plan(
                phase="decode",
                execution="mtp-draft",
                request_ids=active_ids,
                input_token_ids=tokens,
                append_lengths=(1,) * len(active),
                logits_indices=None,
                cache=draft_cache,
            )
            step_hidden = torch.stack([hidden[index] for index in active]).unsqueeze(1)
            if (
                self._mtp_draft_graph is not None
                and len(active) == len(entries)
                and self._mtp_draft_graph.supports(len(active))
            ):
                result = self._mtp_draft_graph.run_positions(
                    tokens,
                    active_ids,
                    step_hidden,
                    rope_deltas=self._rope_deltas([entries[index] for index in active]),
                )
                step_logits = result.logits[:, -1]
                draft_hidden = result.hidden
            else:
                batch = draft_cache.prepare_batch(active_ids, (1,) * len(active))
                batch = replace(
                    batch,
                    mrope_positions=self._mrope_positions(
                        [
                            (entries[index], start, 1)
                            for index, start in zip(active, batch.starts)
                        ]
                    ),
                )
                logits, draft_hidden = self.model.forward_mtp(
                    torch.tensor(tokens, dtype=torch.long, device=self._device).view(-1, 1),
                    step_hidden,
                    kv_cache=draft_cache,
                    cache_batch=batch,
                )
                step_logits = logits[:, -1]
            self._speculative_draft_forwards += 1
            proposals = [
                self._propose(entries[index].request, step_logits[row])
                for row, index in enumerate(active)
            ]
            proposed = self._sync_token_ids(tuple(token for token, _ in proposals))
            for row, (index, token) in enumerate(zip(active, proposed)):
                drafts[index].append(token)
                probs[index].append(proposals[row][1])
                hidden[index] = draft_hidden[row, -1]
        return tuple(tuple(draft) for draft in drafts), tuple(tuple(q) for q in probs)

    def _mtp_catch_up(
        self,
        segments: list[_MTPSegment],
        target_hidden: torch.Tensor | None,
    ) -> None:
        """Run the MTP head over rows the target just processed; store the next drafts."""
        if not segments:
            return
        if target_hidden is None:
            raise RuntimeError("MTP catch-up requires the target hidden states")
        flat_hidden = target_hidden.reshape(-1, target_hidden.size(-1))
        request_ids = tuple(segment.entry.request.request_id for segment in segments)
        append_lengths = tuple(len(segment.tokens) for segment in segments)
        tokens = tuple(token for segment in segments for token in segment.tokens)
        rows = [
            row
            for segment in segments
            for row in range(segment.row_start, segment.row_start + len(segment.tokens))
        ]
        logits_indices: list[int] = []
        offset = 0
        for segment in segments:
            offset += len(segment.tokens)
            if segment.needs_draft:
                logits_indices.append(offset - 1)
        draft_cache = self._draft_cache
        self._validate_batch_plan(
            phase="decode",
            execution="mtp-catch-up",
            request_ids=request_ids,
            input_token_ids=tokens,
            append_lengths=append_lengths,
            logits_indices=tuple(logits_indices),
            cache=draft_cache,
        )
        batch = draft_cache.prepare_batch(request_ids, append_lengths)
        spans = [
            (segment.entry, start, len(segment.tokens))
            for segment, start in zip(segments, batch.starts)
        ]
        batch = replace(batch, mrope_positions=self._mrope_positions(spans))
        logits, mtp_hidden = self.model.forward_mtp(
            torch.tensor(tokens, dtype=torch.long, device=self._device),
            flat_hidden.index_select(0, torch.tensor(rows, device=flat_hidden.device)),
            kv_cache=draft_cache,
            cache_batch=batch,
            logits_indices=tuple(logits_indices),
            # 行 p 的输入 token 是 t_{p+1}；它若是图片占位符，同样换成视觉特征。
            embedding_overrides=self._vision_overrides(spans, shift=1),
        )
        self._speculative_draft_forwards += 1
        drafting = [segment for segment in segments if segment.needs_draft]
        proposals = [
            self._propose(segment.entry.request, logits[index])
            for index, segment in enumerate(drafting)
        ]
        next_drafts = self._sync_token_ids(tuple(token for token, _ in proposals))
        for segment, token, (_, q), row in zip(drafting, next_drafts, proposals, logits_indices):
            segment.entry.mtp_draft = token
            segment.entry.mtp_draft_prob = q
            segment.entry.mtp_hidden = mtp_hidden[row]

    def _mtp_catch_up_full_width(
        self,
        entries: list[_Entry],
        emitted: tuple[tuple[int, ...], ...],
        starts: tuple[int, ...],
        target_hidden: torch.Tensor,
    ) -> None:
        """Graph catch-up over every verified row, then drop rows past acceptance.

        Rows after a request's accepted prefix see arbitrary padding tokens, but causal
        attention keeps them from affecting earlier rows and the truncation below makes
        their draft K/V unreachable, exactly like rejected target K/V.
        """
        width = 1 + self._mtp_steps
        request_ids = tuple(entry.request.request_id for entry in entries)
        tokens = tuple(
            token
            for output in emitted
            for token in output + (output[-1],) * (width - len(output))
        )
        self._validate_batch_plan(
            phase="decode",
            execution="mtp-catch-up-graph",
            request_ids=request_ids,
            input_token_ids=tokens,
            append_lengths=(width,) * len(entries),
            logits_indices=None,
            cache=self._draft_cache,
        )
        result = self._mtp_catch_up_graph.run_positions(
            tokens, request_ids, target_hidden, rope_deltas=self._rope_deltas(entries)
        )
        self._speculative_draft_forwards += 1
        proposals = [
            (row[len(output) - 1], None)
            if entry.request.sampling_params.is_greedy
            else self._propose(entry.request, result.logits[index, len(output) - 1])
            for index, (entry, row, output) in enumerate(zip(entries, result.token_ids, emitted))
        ]
        drafts = self._sync_token_ids(tuple(token for token, _ in proposals))
        for index, (entry, output, start, token) in enumerate(
            zip(entries, emitted, starts, drafts)
        ):
            self._draft_cache.truncate(entry.request.request_id, start + len(output))
            entry.mtp_draft = token
            entry.mtp_draft_prob = proposals[index][1]
            entry.mtp_hidden = result.hidden[index, len(output) - 1].clone()

    @staticmethod
    def _rope_delta(entry: _Entry) -> int:
        multimodal = entry.request.multimodal
        return 0 if multimodal is None else multimodal.rope_delta

    def _rope_deltas(self, entries) -> tuple[int, ...]:
        return tuple(self._rope_delta(entry) for entry in entries)

    def _mrope_positions(self, rows) -> torch.Tensor | None:
        """``[3, tokens]`` RoPE positions when any row belongs to an image request.

        Prompt rows of an image request use its precomputed (t, h, w) positions; every
        other row uses its K/V index plus the request's ``rope_delta`` on all three axes.
        """
        if not any(entry.request.multimodal is not None for entry, _, _ in rows):
            return None
        parts = []
        for entry, start, length in rows:
            multimodal = entry.request.multimodal
            if multimodal is not None and start < len(entry.request.prompt_token_ids):
                parts.append(multimodal.mrope_positions[:, start : start + length])
            else:
                linear = torch.arange(start, start + length) + self._rope_delta(entry)
                parts.append(linear.expand(3, -1))
        return torch.cat(parts, dim=1).to(self._device)

    def _vision_overrides(self, rows, *, shift: int = 0):#覆写图片向量
        """Rows whose input token (prompt position ``start + i + shift``) is an image token."""
        row_indices: list[int] = []
        features: list[torch.Tensor] = []
        offset = 0
        for entry, start, length in rows:
            multimodal = entry.request.multimodal
            if multimodal is not None and multimodal.num_image_tokens:
                positions = multimodal.image_token_positions
                first = bisect_left(positions, start + shift)
                last = bisect_left(positions, start + shift + length)
                if first < last:
                    if entry.vision_features is None:
                        # 整个请求的图片只编码一次；后续 chunk 与 MTP 追赶复用。
                        entry.vision_features = self.model.encode_images(
                            multimodal.pixel_values, multimodal.image_grid_thw
                        )
                    row_indices.extend(
                        offset + positions[index] - start - shift for index in range(first, last)
                    )
                    features.append(entry.vision_features[first:last])
            offset += length
        if not row_indices:
            return None
        return (
            torch.tensor(row_indices, dtype=torch.long, device=self._device),
            torch.cat(features),
        )

    @staticmethod
    def _accept_greedy(draft: tuple[int, ...], target_tokens: tuple[int, ...]) -> tuple[int, ...]:
        accepted = 0
        while accepted < len(draft) and target_tokens[accepted] == draft[accepted]:
            accepted += 1
        return draft[:accepted] + (target_tokens[accepted],)

    @staticmethod
    def _split_hidden(result) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(result, tuple):
            return result
        return result, None

    @staticmethod
    def _accepted_prefix_length(output: tuple[int, ...], draft: tuple[int, ...]) -> int:
        accepted = 0
        while accepted < len(draft) and accepted < len(output):
            if output[accepted] != draft[accepted]:
                break
            accepted += 1
        return accepted

    @staticmethod
    def _truncate_speculative_output(
        request: Request,
        output: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Stop a multi-token result at the same boundary as repeated record_token calls."""
        remaining = request.sampling_params.max_new_tokens - request.num_output_tokens
        result = output[:remaining]
        stops = request.sampling_params.all_stop_token_ids
        for index, token_id in enumerate(result):
            if token_id in stops:
                return result[: index + 1]
        return result

    def _sync_speculative_outputs(
        self,
        outputs: tuple[tuple[int, ...], ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        counts = tuple(map(len, outputs))
        flattened = tuple(token for output in outputs for token in output)
        synchronized = self._sync_token_ids(counts + flattened)
        batch_size = len(outputs)
        authoritative_counts = synchronized[:batch_size]
        authoritative_tokens = synchronized[batch_size:]
        if any(
            count <= 0 or count > append_length
            for count, append_length in zip(authoritative_counts, append_lengths)
        ):
            raise RuntimeError("coordinator returned invalid speculative output lengths")
        if sum(authoritative_counts) != len(authoritative_tokens):
            raise RuntimeError("coordinator returned malformed speculative token IDs")
        rebuilt = []
        offset = 0
        for count in authoritative_counts:
            rebuilt.append(authoritative_tokens[offset : offset + count])
            offset += count
        return tuple(rebuilt)

    def _release(self, entry: _Entry, *, finished: bool) -> None:
        request = entry.request
        request_id = request.request_id
        entry.mtp_draft = None
        entry.mtp_hidden = None
        entry.mtp_draft_prob = None
        entry.vision_features = None
        if request_id in self.cache.allocator.request_ids:# 还持有页才释放
            if finished:
                self.cache.finish_request(request_id, request.all_token_ids) # 正常结束：把页交给 radix 缓存
            else:
                self.cache.release_request(request_id)  # abort：直接归还
        self._entries.pop(request_id, None) # 从总表里删除
        self._sampling_generators.pop(request_id, None)
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1

    @staticmethod
    def _max_request_tokens(request: Request) -> int:
        return len(request.prompt_token_ids) + request.sampling_params.max_new_tokens

    def _generator(self, request: Request) -> torch.Generator | None:
        params = request.sampling_params
        if params.is_greedy:
            return None
        generator = self._sampling_generators.get(request.request_id)
        if generator is None:
            generator = torch.Generator(device=self._device)
            if params.seed is None:
                generator.seed()
            else:
                generator.manual_seed(params.seed)
            self._sampling_generators[request.request_id] = generator
        return generator

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        return sample_token(
            logits,
            request.sampling_params,
            generator=self._generator(request),
            output_token_ids=request.output_token_ids,
        )

    def _validate_batch_plan(
        self,
        *,
        phase: str,
        execution: str,
        request_ids: tuple[str, ...],
        input_token_ids: tuple[int, ...],
        append_lengths: tuple[int, ...],
        logits_indices: tuple[int, ...] | None,
        cache: PagedKVCache | None = None,
    ) -> None:
        if self._coordinator is None:
            return
        cache = self.cache if cache is None else cache
        plan = SchedulerBatchPlan(
            phase=phase,
            execution=execution,
            request_ids=request_ids,
            input_token_ids=input_token_ids,
            append_lengths=append_lengths,
            logits_indices=logits_indices,
            starts=cache.lengths(request_ids),
            page_tables=tuple(
                cache.allocator.page_table(request_id) for request_id in request_ids
            ),
        )
        self._coordinator.validate_batch(plan)

    def _sync_token_ids(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        if self._coordinator is None:
            return token_ids
        result = self._coordinator.sync_token_ids(token_ids)
        if len(result) != len(token_ids):
            raise RuntimeError("coordinator returned the wrong number of sampled tokens")
        return result
