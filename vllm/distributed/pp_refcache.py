# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental PP RefCache transport helpers.

This module implements the initial PP RefCache transport:

* Phase 1 sends a small, tensorized match-plan header before the sender's
  forward pass finishes.
* Phase 2 sends raw INT8 activations for unmatched regions and INT8 deltas for
  matched prefill regions.
* Sender and receiver keep separate boundary-local caches and commit the
  reconstructed hidden states used for future references.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_refcache_kernels import (
    triton_decode_int8,
    triton_decode_int8_with_refs,
    triton_encode_int8_with_refs,
)

_VERSION = 1
_PHASE1_VERSION = 1
_MARKER_KEY = "__pp_refcache_version__"
_PHASE1_MARKER_KEY = "__pp_refcache_phase1_version__"
_PHASE1_PLAN_ID_KEY = "__pp_refcache_phase1_plan_id__"
_PHASE1_NUM_GLOBAL_TOKENS_KEY = "__pp_refcache_phase1_num_global_tokens__"
_PHASE1_TP_RANK_KEY = "__pp_refcache_phase1_tp_rank__"
_PHASE1_TP_SIZE_KEY = "__pp_refcache_phase1_tp_size__"
_PHASE1_TOKEN_SEGMENTS_KEY = "__pp_refcache_phase1_token_segments__"
_PHASE1_MATCH_SPANS_KEY = "__pp_refcache_phase1_match_spans__"
_PHASE1_SELF_REF_SPANS_KEY = "__pp_refcache_phase1_self_ref_spans__"
_CODEC_KEY = "__pp_refcache_codec__"
_TENSOR_NAME_KEY = "__pp_refcache_tensor_name__"
_ORIG_SHAPE_KEY = "__pp_refcache_orig_shape__"
_ORIG_DTYPE_KEY = "__pp_refcache_orig_dtype__"
_USED_ALL_GATHER_KEY = "__pp_refcache_used_all_gather__"
_LOCAL_TOKEN_START_KEY = "__pp_refcache_local_token_start__"
_LOCAL_TOKEN_COUNT_KEY = "__pp_refcache_local_token_count__"
_GROUP_SIZE_KEY = "__pp_refcache_group_size__"
_RAW_TENSORS_KEY = "__pp_refcache_raw_tensors__"
_Q_PAYLOAD_KEY = "__pp_refcache_q_payload__"
_SCALES_KEY = "__pp_refcache_scales__"
_DELTA_MATCH_SPANS_KEY = "__pp_refcache_delta_match_spans__"
_PACKET_STATS_KEY = "__pp_refcache_packet_stats__"
_DEFAULT_MAX_CACHE_TOKENS = 100000


def _profile_enabled() -> bool:
    return envs.VLLM_PP_REFCACHE_PROFILE


def _profile(label: str, start: float, **fields: Any) -> None:
    if not _profile_enabled():
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"PP_REFCACHE_PROFILE pid={os.getpid()} label={label} "
        f"ms={elapsed_ms:.3f} {extra}",
        flush=True,
    )


@dataclass(frozen=True)
class PPRefCacheConfig:
    enabled: bool
    codec: str
    min_hidden_bytes: int
    int8_group_size: int
    max_cache_tokens: int = _DEFAULT_MAX_CACHE_TOKENS
    min_match_rate: float = 0.0
    max_packet_ratio: float = 1.0


@dataclass(frozen=True)
class PPRefCachePhase1Plan:
    plan_id: int
    num_global_tokens: int
    tp_rank: int
    tp_size: int
    token_segments: torch.Tensor
    match_spans: torch.Tensor
    self_ref_spans: torch.Tensor
    token_fps: torch.Tensor | None = None


@dataclass
class _CachedSegment:
    token_fps: list[int]
    hiddens: torch.Tensor
    live_tokens: int
    ready_event: torch.cuda.Event | None = None

    def wait_ready(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()
            self.ready_event = None


@dataclass
class _PrefetchedRefs:
    plan_id: int
    positions: torch.Tensor
    refs: torch.Tensor


class _RefPrefetchBuffer:
    def __init__(self) -> None:
        self.max_bytes = 0
        self._positions: torch.Tensor | None = None
        self._refs: torch.Tensor | None = None
        self._prefetched: _PrefetchedRefs | None = None

    def clear(self) -> None:
        self._positions = None
        self._refs = None
        self._prefetched = None

    def set_max_bytes(self, max_bytes: int) -> None:
        max_bytes = max(0, max_bytes)
        if self.max_bytes == max_bytes:
            return
        self.max_bytes = max_bytes
        self.clear()

    def store(
        self,
        plan_id: int,
        positions: torch.Tensor,
        refs: torch.Tensor,
        device: torch.device,
    ) -> bool:
        if self.max_bytes <= 0 or positions.numel() == 0 or refs.numel() == 0:
            return False
        refs_bytes = refs.numel() * refs.element_size()
        pos_bytes = positions.numel() * positions.element_size()
        if refs_bytes + pos_bytes > self.max_bytes:
            self._prefetched = None
            return False
        if (
            self._positions is None
            or self._positions.device != device
            or self._positions.numel() < positions.numel()
        ):
            self._positions = torch.empty(
                positions.numel(),
                dtype=positions.dtype,
                device=device,
            )
        if (
            self._refs is None
            or self._refs.device != device
            or self._refs.dtype != refs.dtype
            or self._refs.numel() < refs.numel()
        ):
            self._refs = torch.empty(
                refs.numel(),
                dtype=refs.dtype,
                device=device,
            )
        pos_view = self._positions[: positions.numel()]
        refs_view = self._refs[: refs.numel()].reshape(refs.shape)
        pos_view.copy_(positions, non_blocking=True)
        refs_view.copy_(refs, non_blocking=True)
        self._prefetched = _PrefetchedRefs(plan_id, pos_view, refs_view)
        return True

    def store_runs(
        self,
        plan_id: int,
        positions: torch.Tensor,
        ref_runs: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        if self.max_bytes <= 0 or positions.numel() == 0 or not ref_runs:
            return False
        n_rows = sum(run.shape[0] for run in ref_runs)
        if n_rows == 0:
            return False
        row_shape = tuple(ref_runs[0].shape[1:])
        refs_numel = n_rows
        for dim in row_shape:
            refs_numel *= dim
        refs_bytes = refs_numel * torch.empty((), dtype=dtype).element_size()
        pos_bytes = positions.numel() * positions.element_size()
        if refs_bytes + pos_bytes > self.max_bytes:
            self._prefetched = None
            return False
        if (
            self._positions is None
            or self._positions.device != device
            or self._positions.numel() < positions.numel()
        ):
            self._positions = torch.empty(
                positions.numel(),
                dtype=positions.dtype,
                device=device,
            )
        refs_shape = (n_rows, *row_shape)
        if (
            self._refs is None
            or self._refs.device != device
            or self._refs.dtype != dtype
            or self._refs.numel() < refs_numel
        ):
            self._refs = torch.empty(refs_numel, dtype=dtype, device=device)
        pos_view = self._positions[: positions.numel()]
        refs_view = self._refs[:refs_numel].reshape(refs_shape)
        pos_view.copy_(positions, non_blocking=True)
        row_offset = 0
        for run in ref_runs:
            next_offset = row_offset + run.shape[0]
            refs_view[row_offset:next_offset].copy_(
                run.to(dtype=dtype), non_blocking=True
            )
            row_offset = next_offset
        self._prefetched = _PrefetchedRefs(plan_id, pos_view, refs_view)
        return True

    def pop(
        self,
        plan_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        prefetched = self._prefetched
        self._prefetched = None
        if prefetched is None or prefetched.plan_id != plan_id:
            return None
        if (
            prefetched.positions.device != device
            or prefetched.refs.device != device
            or prefetched.refs.dtype != dtype
        ):
            return None
        return prefetched.positions, prefetched.refs


class _BoundaryRefCache:
    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_CACHE_TOKENS,
        *,
        use_direct_match_spans: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self._use_direct_match_spans = use_direct_match_spans
        self._token_order: deque[tuple[int, int]] = deque()
        self._token_refs: dict[tuple[int, int], tuple[int, int]] = {}
        self._segments: dict[int, _CachedSegment] = {}
        self._bigram_index: dict[tuple[int, int], deque[tuple[int, int]]] = {}
        self._next_segment_id = 0
        self._has_evicted = False

    def clear(self) -> None:
        self._token_order.clear()
        self._token_refs.clear()
        self._segments.clear()
        self._bigram_index.clear()
        self._next_segment_id = 0
        self._has_evicted = False

    def set_max_tokens(self, max_tokens: int) -> None:
        self.max_tokens = max(0, max_tokens)
        self._evict_if_needed()

    def _has_token(self, key: tuple[int, int]) -> bool:
        return key in self._token_refs

    def _get_token_fp(self, key: tuple[int, int]) -> int | None:
        ref = self._token_refs.get(key)
        if ref is None:
            return None
        segment = self._segments.get(ref[0])
        if segment is None or ref[1] >= len(segment.token_fps):
            return None
        return segment.token_fps[ref[1]]

    def match_segment(
        self,
        batch_start: int,
        token_start: int,
        token_fps: Sequence[int],
        segment_idx: int,
    ) -> list[list[int]]:
        if self.max_tokens <= 0 or not self._token_refs:
            return []
        spans: list[list[int]] = []
        i = 1
        while i < len(token_fps):
            candidates = self._bigram_index.get((token_fps[i - 1], token_fps[i]))
            if not candidates:
                i += 1
                continue

            best_ref: tuple[int, int] | None = None
            best_token_ref: tuple[int, int] | None = None
            best_len = 0
            for ref_uid, ref_token_idx in candidates:
                ref_key = (ref_uid, ref_token_idx)
                token_ref = self._token_refs.get(ref_key)
                if token_ref is None:
                    continue
                segment = self._segments.get(token_ref[0])
                if segment is None:
                    continue
                segment_offset = token_ref[1]
                if segment.token_fps[segment_offset] != token_fps[i]:
                    continue
                match_len = 1
                if not self._has_evicted:
                    max_len = min(
                        len(token_fps) - i,
                        len(segment.token_fps) - segment_offset,
                    )
                    while (
                        match_len < max_len
                        and token_fps[i + match_len]
                        == segment.token_fps[segment_offset + match_len]
                    ):
                        match_len += 1
                else:
                    while i + match_len < len(token_fps):
                        key = (ref_uid, ref_token_idx + match_len)
                        if not self._has_token(key):
                            break
                        if self._get_token_fp(key) != token_fps[i + match_len]:
                            break
                        match_len += 1
                if match_len > best_len:
                    best_len = match_len
                    best_ref = (ref_uid, ref_token_idx)
                    best_token_ref = token_ref

            if best_ref is None or best_token_ref is None:
                i += 1
                continue
            spans.append(
                [
                    batch_start + i,
                    best_len,
                    best_ref[0],
                    best_ref[1],
                    best_token_ref[0],
                    best_token_ref[1],
                ]
            )
            i += best_len
        return spans

    def get_refs(
        self,
        match_spans: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions_cpu, refs_cpu = self.get_refs_cpu(match_spans, dtype)
        if positions_cpu.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty((0, 0), dtype=dtype, device=device),
            )
        return (
            positions_cpu.to(device=device, non_blocking=True),
            refs_cpu.to(device=device, dtype=dtype, non_blocking=True),
        )

    def get_refs_cpu(
        self,
        match_spans: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions_tensor, refs = self.get_ref_runs_cpu(match_spans)
        if not refs:
            return (
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 0), dtype=dtype),
            )
        return positions_tensor, torch.cat(refs, dim=0).to(dtype=dtype)

    def get_ref_runs_cpu(
        self,
        match_spans: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        positions: list[int] = []
        refs: list[torch.Tensor] = []
        for span in match_spans.detach().cpu().tolist():
            batch_start, length, ref_uid, ref_start, ref_segment_id, ref_offset = (
                int(span[0]),
                int(span[1]),
                int(span[2]),
                int(span[3]),
                int(span[4]),
                int(span[5]),
            )
            if self._use_direct_match_spans and ref_segment_id >= 0:
                segment = self._segments.get(ref_segment_id)
                if segment is not None and ref_offset + length <= segment.live_tokens:
                    segment.wait_ready()
                    positions.extend(range(batch_start, batch_start + length))
                    refs.append(segment.hiddens[ref_offset : ref_offset + length])
                    continue
            offset = 0
            while offset < length:
                token_ref = self._token_refs.get((ref_uid, ref_start + offset))
                if token_ref is None:
                    offset += 1
                    continue
                segment_id, segment_offset = token_ref
                segment = self._segments.get(segment_id)
                if segment is None:
                    offset += 1
                    continue
                segment.wait_ready()
                run_len = 1
                while offset + run_len < length:
                    next_ref = self._token_refs.get(
                        (ref_uid, ref_start + offset + run_len)
                    )
                    if next_ref != (segment_id, segment_offset + run_len):
                        break
                    run_len += 1
                positions.extend(
                    range(batch_start + offset, batch_start + offset + run_len)
                )
                refs.append(
                    segment.hiddens[segment_offset : segment_offset + run_len]
                )
                offset += run_len
        if not refs:
            return torch.empty(0, dtype=torch.int64), []
        return torch.tensor(positions, dtype=torch.int64), refs

    def has_refs(self, match_spans: torch.Tensor) -> bool:
        for span in match_spans.detach().cpu().tolist():
            _, length, ref_uid, ref_start, ref_segment_id, ref_offset = (
                int(span[0]),
                int(span[1]),
                int(span[2]),
                int(span[3]),
                int(span[4]),
                int(span[5]),
            )
            if self._use_direct_match_spans and ref_segment_id >= 0:
                segment = self._segments.get(ref_segment_id)
                if segment is not None and ref_offset + length <= segment.live_tokens:
                    continue
            for offset in range(length):
                if (ref_uid, ref_start + offset) not in self._token_refs:
                    return False
        return True

    def commit(
        self,
        plan: PPRefCachePhase1Plan | None,
        hidden_states: torch.Tensor,
        *,
        index_bigrams: bool = True,
        source_ready_event: torch.cuda.Event | None = None,
    ) -> None:
        if self.max_tokens <= 0:
            return
        if plan is None or plan.token_segments.numel() == 0:
            return
        if hidden_states.ndim < 2:
            return
        if source_ready_event is not None:
            source_ready_event.synchronize()

        token_fps: list[int] | None = None
        if (
            index_bigrams
            and plan.token_fps is not None
            and plan.token_fps.numel() > 0
        ):
            token_fps = [int(x) for x in plan.token_fps.detach().cpu().tolist()]
        fp_offset = 0
        for segment in plan.token_segments.detach().cpu().tolist():
            batch_start, length, token_start, req_uid = (
                int(segment[0]),
                int(segment[1]),
                int(segment[2]),
                int(segment[3]),
            )
            if token_fps is not None:
                segment_fps = token_fps[fp_offset : fp_offset + length]
                if len(segment_fps) != length:
                    segment_fps = [0] * length
            else:
                segment_fps = []
            fp_offset += length
            if batch_start + length > hidden_states.shape[0]:
                continue
            source = hidden_states[batch_start : batch_start + length].detach()
            ready_event = None
            if source.is_cuda:
                if source.dtype != torch.float16:
                    source = source.to(torch.float16)
                rows = torch.empty(
                    tuple(source.shape),
                    dtype=torch.float16,
                    device="cpu",
                    pin_memory=True,
                )
                rows.copy_(source, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(source.device))
            else:
                rows = source.cpu().to(torch.float16).contiguous()
            segment_id = self._next_segment_id
            self._next_segment_id += 1
            self._segments[segment_id] = _CachedSegment(
                token_fps=[int(fp) for fp in segment_fps],
                hiddens=rows,
                live_tokens=length,
                ready_event=ready_event,
            )
            token_refs = self._token_refs
            token_order = self._token_order
            bigram_index = self._bigram_index
            for offset in range(length):
                token_idx = token_start + offset
                key = (req_uid, token_idx)
                if key not in token_refs:
                    token_order.append(key)
                token_refs[key] = (segment_id, offset)
                if index_bigrams and offset > 0 and segment_fps:
                    bigram = (int(segment_fps[offset - 1]), int(segment_fps[offset]))
                    bigram_index.setdefault(bigram, deque(maxlen=32)).append(key)
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._token_refs) > self.max_tokens and self._token_order:
            key = self._token_order.popleft()
            token_ref = self._token_refs.pop(key, None)
            if token_ref is None:
                continue
            self._has_evicted = True
            segment = self._segments.get(token_ref[0])
            if segment is not None:
                segment.live_tokens -= 1
                if segment.live_tokens <= 0:
                    self._segments.pop(token_ref[0], None)


_SEND_CACHE = _BoundaryRefCache(use_direct_match_spans=True)
_RECV_CACHE = _BoundaryRefCache()
_SEND_PREFETCH = _RefPrefetchBuffer()
_RECV_PREFETCH = _RefPrefetchBuffer()
_DEFERRED_SEND_COMMITS: dict[int, tuple[PPRefCachePhase1Plan, torch.Tensor]] = {}
_COMMIT_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="pp-refcache-commit",
)
_PENDING_SEND_COMMITS: list[Future[None]] = []
_PENDING_RECV_COMMITS: list[Future[None]] = []


def _record_ready_event(tensor: torch.Tensor) -> torch.cuda.Event | None:
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _flush_pending_commits(
    pending: list[Future[None]],
    label: str,
) -> None:
    if not pending:
        return
    profile_start = time.perf_counter()
    futures = list(pending)
    pending.clear()
    for future in futures:
        future.result()
    _profile(label, profile_start, count=len(futures))


def _submit_cache_commit(
    pending: list[Future[None]],
    ref_cache: _BoundaryRefCache,
    plan: PPRefCachePhase1Plan | None,
    hidden_states: torch.Tensor,
    *,
    index_bigrams: bool,
) -> None:
    ready_event = _record_ready_event(hidden_states)
    pending.append(
        _COMMIT_EXECUTOR.submit(
            ref_cache.commit,
            plan,
            hidden_states,
            index_bigrams=index_bigrams,
            source_ready_event=ready_event,
        )
    )


def get_pp_refcache_config() -> PPRefCacheConfig:
    config = PPRefCacheConfig(
        enabled=envs.VLLM_PP_REFCACHE_ENABLE,
        codec=envs.VLLM_PP_REFCACHE_CODEC.lower(),
        min_hidden_bytes=envs.VLLM_PP_REFCACHE_MIN_HIDDEN_BYTES,
        int8_group_size=envs.VLLM_PP_REFCACHE_INT8_GROUP_SIZE,
        max_cache_tokens=envs.VLLM_PP_REFCACHE_MAX_TOKENS,
        min_match_rate=envs.VLLM_PP_REFCACHE_MIN_MATCH_RATE,
        max_packet_ratio=envs.VLLM_PP_REFCACHE_MAX_PACKET_RATIO,
    )
    _flush_pending_commits(_PENDING_SEND_COMMITS, "send_commit_flush")
    _flush_pending_commits(_PENDING_RECV_COMMITS, "recv_commit_flush")
    _SEND_CACHE.set_max_tokens(config.max_cache_tokens)
    _RECV_CACHE.set_max_tokens(config.max_cache_tokens)
    return config


def configure_pp_refcache_prefetch(max_activation_bytes: int) -> None:
    hard_cap = envs.VLLM_PP_REFCACHE_PREFETCH_MAX_BYTES
    max_bytes = min(max(0, max_activation_bytes), hard_cap)
    _SEND_PREFETCH.set_max_bytes(max_bytes)
    _RECV_PREFETCH.set_max_bytes(max_bytes)


def clear_pp_refcache_state() -> None:
    _flush_pending_commits(_PENDING_SEND_COMMITS, "send_commit_flush")
    _flush_pending_commits(_PENDING_RECV_COMMITS, "recv_commit_flush")
    _SEND_CACHE.clear()
    _RECV_CACHE.clear()
    _SEND_PREFETCH.clear()
    _RECV_PREFETCH.clear()
    _DEFERRED_SEND_COMMITS.clear()


def _prefetch_refs(
    ref_cache: _BoundaryRefCache,
    prefetch_cache: _RefPrefetchBuffer,
    plan: PPRefCachePhase1Plan | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    profile_start = time.perf_counter()
    if plan is None or plan.match_spans.numel() == 0:
        return
    if not ref_cache.has_refs(plan.match_spans):
        return
    positions, ref_runs = ref_cache.get_ref_runs_cpu(plan.match_spans)
    if positions.numel() > 0:
        prefetch_cache.store_runs(plan.plan_id, positions, ref_runs, device, dtype)
    _profile(
        "prefetch_refs",
        profile_start,
        positions=positions.numel(),
        bytes=sum(run.numel() * run.element_size() for run in ref_runs),
    )


def _pop_prefetched_refs(
    prefetch_cache: _RefPrefetchBuffer,
    plan: PPRefCachePhase1Plan | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if plan is None:
        return None
    return prefetch_cache.pop(plan.plan_id, device, dtype)


def _all_gather_slice_keeps_global_rows(
    used_all_gather: bool,
    local_token_start: int | None,
    local_token_count: int | None,
    plan: PPRefCachePhase1Plan | None,
) -> bool:
    if not used_all_gather:
        return True
    if plan is None:
        return False
    return local_token_start == 0 and local_token_count == plan.num_global_tokens


def prefetch_pp_refcache_send_refs(
    plan: PPRefCachePhase1Plan | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _flush_pending_commits(_PENDING_SEND_COMMITS, "send_commit_flush")
    _prefetch_refs(_SEND_CACHE, _SEND_PREFETCH, plan, device, dtype)


def prefetch_pp_refcache_recv_refs(
    plan: PPRefCachePhase1Plan | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _flush_pending_commits(_PENDING_RECV_COMMITS, "recv_commit_flush")
    _prefetch_refs(_RECV_CACHE, _RECV_PREFETCH, plan, device, dtype)


def _stable_request_uid(req_id: str) -> int:
    digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _empty_phase1_tensor(num_fields: int) -> torch.Tensor:
    return torch.empty((0, num_fields), dtype=torch.int64)


def _phase1_plan_id(
    num_global_tokens: int,
    tp_rank: int,
    tp_size: int,
    token_segments: torch.Tensor,
    match_spans: torch.Tensor,
    self_ref_spans: torch.Tensor,
) -> int:
    h = hashlib.blake2b(digest_size=8)
    for value in (num_global_tokens, tp_rank, tp_size):
        h.update(int(value).to_bytes(8, "little", signed=True))
    for tensor in (token_segments, match_spans, self_ref_spans):
        cpu_tensor = tensor.detach().cpu().contiguous()
        h.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        h.update(cpu_tensor.numpy().tobytes())
    return int.from_bytes(h.digest(), "little", signed=False) & ((1 << 63) - 1)


def _get_req_num_computed(scheduler_output: Any) -> dict[str, int]:
    num_computed: dict[str, int] = {}
    for req in scheduler_output.scheduled_new_reqs:
        num_computed[req.req_id] = req.num_computed_tokens
    cached_reqs = scheduler_output.scheduled_cached_reqs
    num_computed.update(
        zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens, strict=True)
    )
    return num_computed


def _get_new_req_prompt_lens(scheduler_output: Any) -> dict[str, int]:
    prompt_lens: dict[str, int] = {}
    for req in scheduler_output.scheduled_new_reqs:
        if req.prompt_token_ids is not None:
            prompt_lens[req.req_id] = len(req.prompt_token_ids)
    return prompt_lens


def _get_new_req_token_ids(scheduler_output: Any) -> dict[str, list[int]]:
    token_ids: dict[str, list[int]] = {}
    for req in scheduler_output.scheduled_new_reqs:
        ids = getattr(req, "prefill_token_ids", None) or req.prompt_token_ids
        if ids is not None:
            token_ids[req.req_id] = list(ids)
    return token_ids


def _get_cached_req_token_ids(scheduler_output: Any) -> dict[str, list[int]]:
    cached_reqs = scheduler_output.scheduled_cached_reqs
    token_ids = {req_id: list(ids) for req_id, ids in cached_reqs.all_token_ids.items()}
    for req_id, ids in zip(cached_reqs.req_ids, cached_reqs.new_token_ids):
        token_ids.setdefault(req_id, list(ids))
    return token_ids


def _fingerprint_tokens(token_ids: Sequence[int]) -> list[int]:
    return [int(token_id) & ((1 << 63) - 1) for token_id in token_ids]


def build_phase1_plan(
    scheduler_output: Any,
    tp_group: GroupCoordinator | None,
) -> PPRefCachePhase1Plan:
    """Build the sender-selected Phase 1 plan for the current PP batch.

    Milestone 2 only emits prefill token segments. Match spans and self-ref
    spans stay empty until the RefCache matcher lands in Milestone 3.
    """
    _flush_pending_commits(_PENDING_SEND_COMMITS, "send_commit_flush")
    profile_start = time.perf_counter()
    num_global_tokens = int(scheduler_output.total_num_scheduled_tokens)
    tp_rank = 0 if tp_group is None else tp_group.rank_in_group
    tp_size = 1 if tp_group is None else tp_group.world_size
    cached_reqs = scheduler_output.scheduled_cached_reqs
    cached_context_req_ids = {
        req_id for req_id in cached_reqs.req_ids if cached_reqs.is_context_phase(req_id)
    }
    num_computed = _get_req_num_computed(scheduler_output)
    new_req_prompt_lens = _get_new_req_prompt_lens(scheduler_output)
    new_req_token_ids = _get_new_req_token_ids(scheduler_output)
    cached_req_token_ids = _get_cached_req_token_ids(scheduler_output)

    rows: list[list[int]] = []
    token_fps: list[int] = []
    match_rows: list[list[int]] = []
    batch_start = 0
    req_ids = sorted(
        scheduler_output.num_scheduled_tokens,
        key=scheduler_output.num_scheduled_tokens.get,
    )
    for req_id in req_ids:
        scheduled_len = int(scheduler_output.num_scheduled_tokens[req_id])
        token_start = int(num_computed.get(req_id, 0))
        prefill_len = 0
        if req_id in new_req_prompt_lens:
            prefill_len = min(
                scheduled_len,
                max(0, new_req_prompt_lens[req_id] - token_start),
            )
        elif req_id in cached_context_req_ids:
            prefill_len = scheduled_len

        if prefill_len > 0:
            req_uid = _stable_request_uid(req_id)
            req_token_ids = (
                new_req_token_ids.get(req_id)
                if req_id in new_req_prompt_lens
                else cached_req_token_ids.get(req_id)
            )
            segment_fps = []
            if req_token_ids is not None:
                segment_token_ids = req_token_ids[
                    token_start : token_start + prefill_len
                ]
                segment_fps = _fingerprint_tokens(segment_token_ids)
            if len(segment_fps) != prefill_len:
                segment_fps = [0] * prefill_len
            segment_idx = len(rows)
            rows.append(
                [
                    batch_start,
                    prefill_len,
                    token_start,
                    req_uid,
                    0,
                ]
            )
            token_fps.extend(segment_fps)
            match_rows.extend(
                _SEND_CACHE.match_segment(
                    batch_start,
                    token_start,
                    segment_fps,
                    segment_idx,
                )
            )
        batch_start += scheduled_len

    token_segments = (
        torch.tensor(rows, dtype=torch.int64)
        if rows
        else _empty_phase1_tensor(5)
    )
    match_spans = (
        torch.tensor(match_rows, dtype=torch.int64)
        if match_rows
        else _empty_phase1_tensor(6)
    )
    self_ref_spans = _empty_phase1_tensor(4)
    token_fps_tensor = (
        torch.tensor(token_fps, dtype=torch.int64)
        if token_fps
        else torch.empty(0, dtype=torch.int64)
    )
    plan_id = _phase1_plan_id(
        num_global_tokens,
        tp_rank,
        tp_size,
        token_segments,
        match_spans,
        self_ref_spans,
    )
    plan = PPRefCachePhase1Plan(
        plan_id=plan_id,
        num_global_tokens=num_global_tokens,
        tp_rank=tp_rank,
        tp_size=tp_size,
        token_segments=token_segments,
        match_spans=match_spans,
        self_ref_spans=self_ref_spans,
        token_fps=token_fps_tensor,
    )
    _profile(
        "phase1_build",
        profile_start,
        tokens=num_global_tokens,
        segments=token_segments.shape[0],
        matched=_count_span_tokens(match_spans),
    )
    return plan


def _phase1_to_tensor_dict(
    plan: PPRefCachePhase1Plan,
) -> dict[str, torch.Tensor | Any]:
    return {
        _PHASE1_MARKER_KEY: _PHASE1_VERSION,
        _PHASE1_PLAN_ID_KEY: plan.plan_id,
        _PHASE1_NUM_GLOBAL_TOKENS_KEY: plan.num_global_tokens,
        _PHASE1_TP_RANK_KEY: plan.tp_rank,
        _PHASE1_TP_SIZE_KEY: plan.tp_size,
        _PHASE1_TOKEN_SEGMENTS_KEY: plan.token_segments,
        _PHASE1_MATCH_SPANS_KEY: plan.match_spans,
        _PHASE1_SELF_REF_SPANS_KEY: plan.self_ref_spans,
    }


def _phase1_from_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any] | None,
) -> PPRefCachePhase1Plan | None:
    if tensor_dict is None:
        return None
    if tensor_dict.get(_PHASE1_MARKER_KEY) != _PHASE1_VERSION:
        raise ValueError("Invalid PP RefCache Phase 1 packet")
    token_segments = tensor_dict[_PHASE1_TOKEN_SEGMENTS_KEY]
    match_spans = tensor_dict[_PHASE1_MATCH_SPANS_KEY]
    self_ref_spans = tensor_dict[_PHASE1_SELF_REF_SPANS_KEY]
    assert isinstance(token_segments, torch.Tensor)
    assert isinstance(match_spans, torch.Tensor)
    assert isinstance(self_ref_spans, torch.Tensor)
    return PPRefCachePhase1Plan(
        plan_id=int(tensor_dict[_PHASE1_PLAN_ID_KEY]),
        num_global_tokens=int(tensor_dict[_PHASE1_NUM_GLOBAL_TOKENS_KEY]),
        tp_rank=int(tensor_dict[_PHASE1_TP_RANK_KEY]),
        tp_size=int(tensor_dict[_PHASE1_TP_SIZE_KEY]),
        token_segments=token_segments,
        match_spans=match_spans,
        self_ref_spans=self_ref_spans,
    )


def _clip_phase1_plan_to_token_rows(
    plan: PPRefCachePhase1Plan | None,
    row_start: int,
    row_count: int,
) -> PPRefCachePhase1Plan | None:
    if plan is None or plan.token_segments.numel() == 0 or row_count <= 0:
        return None

    token_fps = (
        [int(x) for x in plan.token_fps.detach().cpu().tolist()]
        if plan.token_fps is not None
        else []
    )
    clipped_segments: list[list[int]] = []
    clipped_fps: list[int] = []
    fp_offset = 0
    for segment in plan.token_segments.detach().cpu().tolist():
        batch_start, length, token_start, req_uid, flags = (
            int(segment[0]),
            int(segment[1]),
            int(segment[2]),
            int(segment[3]),
            int(segment[4]),
        )
        segment_end = batch_start + length
        clip_start = max(batch_start, row_start)
        clip_end = min(segment_end, row_start + row_count)
        if clip_start < clip_end:
            rel_start = clip_start - batch_start
            clip_len = clip_end - clip_start
            clipped_segments.append(
                [
                    clip_start - row_start,
                    clip_len,
                    token_start + rel_start,
                    req_uid,
                    flags,
                ]
            )
            if token_fps:
                clipped_fps.extend(
                    token_fps[fp_offset + rel_start : fp_offset + rel_start + clip_len]
                )
        fp_offset += length

    clipped_matches: list[list[int]] = []
    for span in plan.match_spans.detach().cpu().tolist():
        batch_start, length, ref_uid, ref_start, segment_idx, flags = (
            int(span[0]),
            int(span[1]),
            int(span[2]),
            int(span[3]),
            int(span[4]),
            int(span[5]),
        )
        span_end = batch_start + length
        clip_start = max(batch_start, row_start)
        clip_end = min(span_end, row_start + row_count)
        if clip_start < clip_end:
            rel_start = clip_start - batch_start
            clipped_matches.append(
                [
                    clip_start - row_start,
                    clip_end - clip_start,
                    ref_uid,
                    ref_start + rel_start,
                    segment_idx,
                    flags + rel_start,
                ]
            )

    if not clipped_segments:
        return None
    token_segments = torch.tensor(clipped_segments, dtype=torch.int64)
    match_spans = (
        torch.tensor(clipped_matches, dtype=torch.int64)
        if clipped_matches
        else _empty_phase1_tensor(6)
    )
    token_fps_tensor = (
        torch.tensor(clipped_fps, dtype=torch.int64)
        if clipped_fps
        else torch.empty(0, dtype=torch.int64)
    )
    return PPRefCachePhase1Plan(
        plan_id=plan.plan_id,
        num_global_tokens=row_count,
        tp_rank=plan.tp_rank,
        tp_size=plan.tp_size,
        token_segments=token_segments,
        match_spans=match_spans,
        self_ref_spans=_empty_phase1_tensor(4),
        token_fps=token_fps_tensor,
    )


def _all_gather_token_row_slice(
    hidden_states: torch.Tensor,
    all_gather_group: GroupCoordinator | None,
) -> tuple[int, int] | None:
    if all_gather_group is None or hidden_states.ndim < 2:
        return None
    hidden_dim = hidden_states.shape[-1]
    if hidden_dim <= 0:
        return None
    tp_size = all_gather_group.world_size
    if hidden_states.numel() % tp_size != 0:
        return None
    chunk_elems = hidden_states.numel() // tp_size
    start_elem = all_gather_group.rank_in_group * chunk_elems
    end_elem = start_elem + chunk_elems
    if start_elem % hidden_dim != 0 or end_elem % hidden_dim != 0:
        return None
    return start_elem // hidden_dim, chunk_elems // hidden_dim


def _is_supported_hidden_tensor(
    hidden_states: torch.Tensor,
    config: PPRefCacheConfig,
) -> bool:
    if config.codec != "int8":
        return False
    if config.int8_group_size <= 0:
        return False
    if hidden_states.numel() == 0 or hidden_states.ndim < 1:
        return False
    if not hidden_states.is_floating_point():
        return False
    hidden_bytes = hidden_states.numel() * hidden_states.element_size()
    if hidden_bytes < config.min_hidden_bytes:
        return False
    return hidden_states.numel() % config.int8_group_size == 0


def _has_only_supported_tensors(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> bool:
    for key, value in tensor_dict.items():
        if key == "hidden_states":
            continue
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return False
    return isinstance(tensor_dict.get("hidden_states"), torch.Tensor)


def _should_use_all_gather(
    pp_group: GroupCoordinator,
    key: str,
    tensor: torch.Tensor,
    all_gather_group: GroupCoordinator | None,
    all_gather_tensors: dict[str, bool] | None,
) -> bool:
    return pp_group._should_use_all_gather(  # type: ignore[attr-defined]
        key,
        tensor.numel(),
        all_gather_group,
        all_gather_tensors,
    )


def _torch_per_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = x.shape
    x_2d = x.reshape(-1, group_size).to(torch.float32)
    scales = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
    q = torch.clamp(torch.round(x_2d / scales), -128, 127).to(torch.int8)
    return q.reshape(original_shape), scales.reshape(original_shape[:-1] + (-1,))


def _quantize_int8(
    tensor: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.contiguous()
    if tensor.is_cuda and tensor.numel() % group_size == 0:
        from vllm.model_executor.layers.quantization.utils.int8_utils import (
            per_token_group_quant_int8,
        )

        if tensor.ndim >= 2 and tensor.shape[-1] % group_size == 0:
            return per_token_group_quant_int8(tensor, group_size)
        q_payload, scales = per_token_group_quant_int8(
            tensor.reshape(1, -1),
            group_size,
        )
        return q_payload.reshape(tensor.shape), scales.reshape(-1)
    return _torch_per_group_quant_int8(tensor, group_size)


def _dequantize_int8(
    q_payload: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    triton_out = triton_decode_int8(
        q_payload.contiguous(),
        scales.contiguous(),
        group_size,
        dtype,
    )
    if triton_out is not None:
        return triton_out
    q_2d = q_payload.reshape(-1, group_size).to(torch.float32)
    scales_2d = scales.reshape(-1, 1).to(torch.float32)
    return (q_2d * scales_2d).reshape(q_payload.shape).to(dtype)


def _apply_delta_refs(
    encoded: torch.Tensor,
    match_spans: torch.Tensor,
    ref_cache: _BoundaryRefCache,
    prefetched_refs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if match_spans.numel() == 0:
        return encoded
    if prefetched_refs is None:
        positions, refs = ref_cache.get_refs(match_spans, encoded.device, encoded.dtype)
    else:
        positions, refs = prefetched_refs
    if positions.numel() == 0:
        return encoded
    reconstructed = encoded.clone()
    reconstructed[positions] = reconstructed[positions] + refs.reshape(
        reconstructed[positions].shape
    )
    return reconstructed


def _subtract_delta_refs(
    hidden_states: torch.Tensor,
    match_spans: torch.Tensor,
    prefetched_refs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if match_spans.numel() == 0:
        return hidden_states
    if prefetched_refs is None:
        positions, refs = _SEND_CACHE.get_refs(
            match_spans,
            hidden_states.device,
            hidden_states.dtype,
        )
    else:
        positions, refs = prefetched_refs
    if positions.numel() == 0:
        return hidden_states
    encoded = hidden_states.clone()
    encoded[positions] = encoded[positions] - refs.reshape(encoded[positions].shape)
    return encoded


def _encode_int8_with_refs(
    hidden_states: torch.Tensor,
    match_spans: torch.Tensor,
    group_size: int,
    prefetched_refs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if match_spans.numel() == 0:
        return None
    if prefetched_refs is None:
        positions, refs = _SEND_CACHE.get_refs(
            match_spans,
            hidden_states.device,
            hidden_states.dtype,
        )
    else:
        positions, refs = prefetched_refs
    if positions.numel() == 0:
        return None
    return triton_encode_int8_with_refs(
        hidden_states.contiguous(),
        positions,
        refs.contiguous(),
        group_size,
    )


def _decode_int8_with_refs(
    q_payload: torch.Tensor,
    scales: torch.Tensor,
    match_spans: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    prefetched_refs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    if match_spans.numel() == 0:
        return None
    if prefetched_refs is None:
        positions, refs = _RECV_CACHE.get_refs(match_spans, q_payload.device, dtype)
    else:
        positions, refs = prefetched_refs
    if positions.numel() == 0:
        return None
    return triton_decode_int8_with_refs(
        q_payload.contiguous(),
        scales.contiguous(),
        positions,
        refs.contiguous(),
        group_size,
        dtype,
    )


def _count_span_tokens(match_spans: torch.Tensor) -> int:
    if match_spans.numel() == 0:
        return 0
    return int(match_spans[:, 1].sum().item())


def _packet_payload_ratio(
    q_payload: torch.Tensor,
    scales: torch.Tensor,
    match_spans: torch.Tensor,
    raw_tensor: torch.Tensor,
) -> float:
    compressed_bytes = q_payload.numel() * q_payload.element_size()
    compressed_bytes += scales.numel() * scales.element_size()
    compressed_bytes += match_spans.numel() * match_spans.element_size()
    raw_bytes = raw_tensor.numel() * raw_tensor.element_size()
    if raw_bytes == 0:
        return 1.0
    return compressed_bytes / raw_bytes


def _make_send_tensor(
    pp_group: GroupCoordinator,
    tensor: torch.Tensor,
    all_gather_group: GroupCoordinator | None,
    all_gather_tensors: dict[str, bool] | None,
) -> tuple[torch.Tensor, bool]:
    use_all_gather = _should_use_all_gather(
        pp_group,
        "hidden_states",
        tensor,
        all_gather_group,
        all_gather_tensors,
    )
    if not use_all_gather:
        return tensor, False

    assert all_gather_group is not None
    tensor = tensor.reshape(all_gather_group.world_size, -1)[
        all_gather_group.rank_in_group
    ]
    return tensor, True


def _packet_all_gather_overrides(
    all_gather_tensors: dict[str, bool] | None,
) -> dict[str, bool]:
    overrides = dict(all_gather_tensors or {})
    overrides[_Q_PAYLOAD_KEY] = False
    overrides[_SCALES_KEY] = False
    overrides[_DELTA_MATCH_SPANS_KEY] = False
    return overrides


def _encode_packet(
    tensor_dict: dict[str, torch.Tensor | Any],
    pp_group: GroupCoordinator,
    all_gather_group: GroupCoordinator | None,
    all_gather_tensors: dict[str, bool] | None,
    config: PPRefCacheConfig,
    phase1_plan: PPRefCachePhase1Plan | None = None,
    defer_send_commit: bool = False,
) -> dict[str, torch.Tensor | Any] | None:
    _flush_pending_commits(_PENDING_SEND_COMMITS, "send_commit_flush")
    profile_start = time.perf_counter()
    if not config.enabled or not _has_only_supported_tensors(tensor_dict):
        return None

    hidden_states = tensor_dict["hidden_states"]
    assert isinstance(hidden_states, torch.Tensor)
    if not _is_supported_hidden_tensor(hidden_states, config):
        return None

    send_tensor, used_all_gather = _make_send_tensor(
        pp_group,
        hidden_states,
        all_gather_group,
        all_gather_tensors,
    )
    if send_tensor.numel() % config.int8_group_size != 0:
        return None

    match_spans = _empty_phase1_tensor(6)
    encode_tensor = send_tensor
    commit_plan = phase1_plan
    local_token_start: int | None = None
    local_token_count: int | None = None
    if used_all_gather:
        token_row_slice = _all_gather_token_row_slice(hidden_states, all_gather_group)
        if token_row_slice is not None:
            local_token_start, local_token_count = token_row_slice
            commit_plan = _clip_phase1_plan_to_token_rows(
                phase1_plan,
                local_token_start,
                local_token_count,
            )
            if commit_plan is not None:
                encode_tensor = send_tensor.reshape(local_token_count, -1)
        else:
            commit_plan = None

    can_use_delta = (
        commit_plan is not None
        and commit_plan.match_spans.numel() > 0
        and _SEND_CACHE.has_refs(commit_plan.match_spans)
        and encode_tensor.ndim >= 2
        and encode_tensor.shape[0] >= commit_plan.num_global_tokens
    )
    if can_use_delta and config.min_match_rate > 0.0:
        matched_tokens = _count_span_tokens(commit_plan.match_spans)
        can_use_delta = (
            matched_tokens / max(1, commit_plan.num_global_tokens)
        ) >= config.min_match_rate
    prefetched_refs = (
        _pop_prefetched_refs(
            _SEND_PREFETCH,
            commit_plan,
            encode_tensor.device,
            encode_tensor.dtype,
        )
        if can_use_delta
        and _all_gather_slice_keeps_global_rows(
            used_all_gather,
            local_token_start,
            local_token_count,
            phase1_plan,
        )
        else None
    )
    if can_use_delta:
        match_spans = commit_plan.match_spans
    fused_encoded = (
        _encode_int8_with_refs(
            encode_tensor,
            match_spans,
            config.int8_group_size,
            prefetched_refs,
        )
        if can_use_delta
        else None
    )
    if fused_encoded is not None:
        q_payload, scales, reconstructed = fused_encoded
        encode_path = "triton"
    else:
        if can_use_delta:
            encode_tensor = _subtract_delta_refs(
                encode_tensor,
                match_spans,
                prefetched_refs,
            )
        q_payload, scales = _quantize_int8(encode_tensor, config.int8_group_size)
        reconstructed = None
        if commit_plan is not None:
            reconstructed = _dequantize_int8(
                q_payload,
                scales,
                config.int8_group_size,
                send_tensor.dtype,
            )
            if can_use_delta:
                reconstructed = _apply_delta_refs(
                    reconstructed, match_spans, _SEND_CACHE, prefetched_refs
                )
        encode_path = "torch"

    packet_ratio = _packet_payload_ratio(q_payload, scales, match_spans, send_tensor)
    if packet_ratio > config.max_packet_ratio:
        return None

    if commit_plan is not None and reconstructed is not None and defer_send_commit:
        _DEFERRED_SEND_COMMITS[id(q_payload)] = (commit_plan, reconstructed)
    elif commit_plan is not None and reconstructed is not None:
        commit_start = time.perf_counter()
        _SEND_CACHE.commit(commit_plan, reconstructed)
        _profile("send_commit_sync", commit_start)

    raw_tensors = {
        key: value
        for key, value in tensor_dict.items()
        if key != "hidden_states"
    }
    packet = {
        _MARKER_KEY: _VERSION,
        _PHASE1_PLAN_ID_KEY: None if phase1_plan is None else phase1_plan.plan_id,
        _CODEC_KEY: config.codec,
        _TENSOR_NAME_KEY: "hidden_states",
        _ORIG_SHAPE_KEY: tuple(hidden_states.shape),
        _ORIG_DTYPE_KEY: hidden_states.dtype,
        _USED_ALL_GATHER_KEY: used_all_gather,
        _LOCAL_TOKEN_START_KEY: local_token_start,
        _LOCAL_TOKEN_COUNT_KEY: local_token_count,
        _GROUP_SIZE_KEY: config.int8_group_size,
        _RAW_TENSORS_KEY: raw_tensors,
        _Q_PAYLOAD_KEY: q_payload,
        _SCALES_KEY: scales,
        _DELTA_MATCH_SPANS_KEY: match_spans,
        _PACKET_STATS_KEY: {
            "matched_tokens": _count_span_tokens(match_spans),
            "packet_ratio": packet_ratio,
            "encode_path": encode_path,
        },
    }
    _profile(
        "encode_packet",
        profile_start,
        path=encode_path,
        delta=int(can_use_delta),
        matched=_count_span_tokens(match_spans),
        ratio=f"{packet_ratio:.3f}",
        defer=int(defer_send_commit),
    )
    return packet


def _finish_deferred_send_commit(packet: dict[str, torch.Tensor | Any]) -> None:
    q_payload = packet.get(_Q_PAYLOAD_KEY)
    if not isinstance(q_payload, torch.Tensor):
        return
    deferred = _DEFERRED_SEND_COMMITS.pop(id(q_payload), None)
    if deferred is None:
        return
    commit_plan, reconstructed = deferred
    profile_start = time.perf_counter()
    _submit_cache_commit(
        _PENDING_SEND_COMMITS,
        _SEND_CACHE,
        commit_plan,
        reconstructed,
        index_bigrams=True,
    )
    _profile("send_commit_deferred", profile_start, async_submit=1)


def is_pp_refcache_packet(tensor_dict: dict[str, torch.Tensor | Any]) -> bool:
    return tensor_dict.get(_MARKER_KEY) == _VERSION


def _decode_packet(
    tensor_dict: dict[str, torch.Tensor | Any],
    all_gather_group: GroupCoordinator | None,
    expected_phase1_plan: PPRefCachePhase1Plan | None = None,
) -> dict[str, torch.Tensor | Any]:
    profile_start = time.perf_counter()
    if not is_pp_refcache_packet(tensor_dict):
        return tensor_dict
    _flush_pending_commits(_PENDING_RECV_COMMITS, "recv_commit_flush")

    packet_plan_id = tensor_dict.get(_PHASE1_PLAN_ID_KEY)
    if (
        expected_phase1_plan is not None
        and packet_plan_id is not None
        and int(packet_plan_id) != expected_phase1_plan.plan_id
    ):
        raise ValueError(
            "PP RefCache Phase 2 packet does not match the received Phase 1 plan"
        )

    if tensor_dict[_CODEC_KEY] != "int8":
        raise ValueError(f"Unsupported PP RefCache codec: {tensor_dict[_CODEC_KEY]}")

    q_payload = tensor_dict[_Q_PAYLOAD_KEY]
    scales = tensor_dict[_SCALES_KEY]
    assert isinstance(q_payload, torch.Tensor)
    assert isinstance(scales, torch.Tensor)

    match_spans = tensor_dict.get(_DELTA_MATCH_SPANS_KEY, _empty_phase1_tensor(6))
    assert isinstance(match_spans, torch.Tensor)
    if match_spans.numel() > 0 and not _RECV_CACHE.has_refs(match_spans):
        raise RuntimeError("PP RefCache receiver is missing a referenced activation")
    prefetched_refs = (
        _pop_prefetched_refs(
            _RECV_PREFETCH,
            expected_phase1_plan,
            q_payload.device,
            tensor_dict[_ORIG_DTYPE_KEY],
        )
        if _all_gather_slice_keeps_global_rows(
            bool(tensor_dict[_USED_ALL_GATHER_KEY]),
            tensor_dict.get(_LOCAL_TOKEN_START_KEY),
            tensor_dict.get(_LOCAL_TOKEN_COUNT_KEY),
            expected_phase1_plan,
        )
        else None
    )
    hidden_states = _decode_int8_with_refs(
        q_payload,
        scales,
        match_spans,
        int(tensor_dict[_GROUP_SIZE_KEY]),
        tensor_dict[_ORIG_DTYPE_KEY],
        prefetched_refs,
    )
    if hidden_states is None:
        hidden_states = _dequantize_int8(
            q_payload,
            scales,
            int(tensor_dict[_GROUP_SIZE_KEY]),
            tensor_dict[_ORIG_DTYPE_KEY],
        )
        hidden_states = _apply_delta_refs(
            hidden_states, match_spans, _RECV_CACHE, prefetched_refs
        )
    commit_plan = expected_phase1_plan
    if tensor_dict[_USED_ALL_GATHER_KEY]:
        local_token_start = tensor_dict.get(_LOCAL_TOKEN_START_KEY)
        local_token_count = tensor_dict.get(_LOCAL_TOKEN_COUNT_KEY)
        if local_token_start is not None and local_token_count is not None:
            commit_plan = _clip_phase1_plan_to_token_rows(
                expected_phase1_plan,
                int(local_token_start),
                int(local_token_count),
            )
            if commit_plan is not None:
                commit_start = time.perf_counter()
                _submit_cache_commit(
                    _PENDING_RECV_COMMITS,
                    _RECV_CACHE,
                    commit_plan,
                    hidden_states,
                    index_bigrams=False,
                )
                _profile("recv_commit_ag", commit_start, async_submit=1)
        else:
            commit_plan = None

    if tensor_dict[_USED_ALL_GATHER_KEY]:
        if all_gather_group is None:
            raise ValueError("PP RefCache packet requires an all-gather group")
        hidden_states = all_gather_group.all_gather(hidden_states, dim=0)
    hidden_states = hidden_states.reshape(tensor_dict[_ORIG_SHAPE_KEY])
    if not tensor_dict[_USED_ALL_GATHER_KEY]:
        commit_start = time.perf_counter()
        _submit_cache_commit(
            _PENDING_RECV_COMMITS,
            _RECV_CACHE,
            commit_plan,
            hidden_states,
            index_bigrams=False,
        )
        _profile("recv_commit", commit_start, async_submit=1)

    decoded = dict(tensor_dict[_RAW_TENSORS_KEY])
    decoded[tensor_dict[_TENSOR_NAME_KEY]] = hidden_states
    _profile(
        "decode_packet",
        profile_start,
        matched=_count_span_tokens(match_spans),
        ag=int(tensor_dict[_USED_ALL_GATHER_KEY]),
    )
    return decoded


def send_pp_refcache_phase1_plan(
    pp_group: GroupCoordinator,
    plan: PPRefCachePhase1Plan,
    dst: int | None = None,
) -> None:
    profile_start = time.perf_counter()
    pp_group.send_tensor_dict(_phase1_to_tensor_dict(plan), dst=dst)
    _profile(
        "phase1_send",
        profile_start,
        tokens=plan.num_global_tokens,
        matched=_count_span_tokens(plan.match_spans),
    )


def recv_pp_refcache_phase1_plan(
    pp_group: GroupCoordinator,
    src: int | None = None,
) -> PPRefCachePhase1Plan | None:
    profile_start = time.perf_counter()
    plan = _phase1_from_tensor_dict(pp_group.recv_tensor_dict(src=src))
    _profile(
        "phase1_recv",
        profile_start,
        tokens=0 if plan is None else plan.num_global_tokens,
        matched=0 if plan is None else _count_span_tokens(plan.match_spans),
    )
    return plan


def isend_pp_refcache_tensor_dict(
    pp_group: GroupCoordinator,
    tensor_dict: dict[str, torch.Tensor | Any],
    dst: int | None = None,
    all_gather_group: GroupCoordinator | None = None,
    all_gather_tensors: dict[str, bool] | None = None,
    phase1_plan: PPRefCachePhase1Plan | None = None,
) -> list[Any]:
    config = get_pp_refcache_config()
    packet = _encode_packet(
        tensor_dict,
        pp_group,
        all_gather_group,
        all_gather_tensors,
        config,
        phase1_plan,
        defer_send_commit=True,
    )
    if packet is None:
        return pp_group.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )

    handles = pp_group.isend_tensor_dict(
        packet,
        dst=dst,
        all_gather_group=all_gather_group,
        all_gather_tensors=_packet_all_gather_overrides(all_gather_tensors),
    )
    _finish_deferred_send_commit(packet)
    return handles


def irecv_pp_refcache_tensor_dict(
    pp_group: GroupCoordinator,
    src: int | None = None,
    all_gather_group: GroupCoordinator | None = None,
    all_gather_tensors: dict[str, bool] | None = None,
    expected_phase1_plan: PPRefCachePhase1Plan | None = None,
) -> tuple[
    dict[str, torch.Tensor | Any] | None,
    list[Any],
    list[Callable[[], None]],
]:
    tensor_dict, handles, postprocess = pp_group.irecv_tensor_dict(
        src=src,
        all_gather_group=all_gather_group,
        all_gather_tensors=_packet_all_gather_overrides(all_gather_tensors),
    )
    if tensor_dict is None:
        return None, handles, postprocess

    def _decode_postprocess() -> None:
        decoded = _decode_packet(tensor_dict, all_gather_group, expected_phase1_plan)
        if decoded is not tensor_dict:
            tensor_dict.clear()
            tensor_dict.update(decoded)

    postprocess.append(_decode_postprocess)
    return tensor_dict, handles, postprocess
