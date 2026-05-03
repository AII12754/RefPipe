# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental PP RefCache transport helpers.

Milestone 2 implements the two-phase transport skeleton:

* Phase 1 sends a small, tensorized match-plan header before the sender's
  forward pass finishes.
* Phase 2 keeps the Milestone 1 raw INT8 activation packet and associates it
  with the sender-selected Phase 1 plan.

Actual reference matching and delta coding are intentionally left for later
milestones.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import GroupCoordinator

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
_GROUP_SIZE_KEY = "__pp_refcache_group_size__"
_RAW_TENSORS_KEY = "__pp_refcache_raw_tensors__"
_Q_PAYLOAD_KEY = "__pp_refcache_q_payload__"
_SCALES_KEY = "__pp_refcache_scales__"


@dataclass(frozen=True)
class PPRefCacheConfig:
    enabled: bool
    codec: str
    min_hidden_bytes: int
    int8_group_size: int


@dataclass(frozen=True)
class PPRefCachePhase1Plan:
    plan_id: int
    num_global_tokens: int
    tp_rank: int
    tp_size: int
    token_segments: torch.Tensor
    match_spans: torch.Tensor
    self_ref_spans: torch.Tensor


def get_pp_refcache_config() -> PPRefCacheConfig:
    return PPRefCacheConfig(
        enabled=envs.VLLM_PP_REFCACHE_ENABLE,
        codec=envs.VLLM_PP_REFCACHE_CODEC.lower(),
        min_hidden_bytes=envs.VLLM_PP_REFCACHE_MIN_HIDDEN_BYTES,
        int8_group_size=envs.VLLM_PP_REFCACHE_INT8_GROUP_SIZE,
    )


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


def build_phase1_plan(
    scheduler_output: Any,
    tp_group: GroupCoordinator | None,
) -> PPRefCachePhase1Plan:
    """Build the sender-selected Phase 1 plan for the current PP batch.

    Milestone 2 only emits prefill token segments. Match spans and self-ref
    spans stay empty until the RefCache matcher lands in Milestone 3.
    """
    num_global_tokens = int(scheduler_output.total_num_scheduled_tokens)
    tp_rank = 0 if tp_group is None else tp_group.rank_in_group
    tp_size = 1 if tp_group is None else tp_group.world_size
    cached_reqs = scheduler_output.scheduled_cached_reqs
    cached_context_req_ids = {
        req_id for req_id in cached_reqs.req_ids if cached_reqs.is_context_phase(req_id)
    }
    num_computed = _get_req_num_computed(scheduler_output)
    new_req_prompt_lens = _get_new_req_prompt_lens(scheduler_output)

    rows: list[list[int]] = []
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
            rows.append(
                [
                    batch_start,
                    prefill_len,
                    token_start,
                    _stable_request_uid(req_id),
                    0,
                ]
            )
        batch_start += scheduled_len

    token_segments = (
        torch.tensor(rows, dtype=torch.int64)
        if rows
        else _empty_phase1_tensor(5)
    )
    match_spans = _empty_phase1_tensor(6)
    self_ref_spans = _empty_phase1_tensor(4)
    plan_id = _phase1_plan_id(
        num_global_tokens,
        tp_rank,
        tp_size,
        token_segments,
        match_spans,
        self_ref_spans,
    )
    return PPRefCachePhase1Plan(
        plan_id=plan_id,
        num_global_tokens=num_global_tokens,
        tp_rank=tp_rank,
        tp_size=tp_size,
        token_segments=token_segments,
        match_spans=match_spans,
        self_ref_spans=self_ref_spans,
    )


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
    q_2d = q_payload.reshape(-1, group_size).to(torch.float32)
    scales_2d = scales.reshape(-1, 1).to(torch.float32)
    return (q_2d * scales_2d).reshape(q_payload.shape).to(dtype)


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
    return overrides


def _encode_packet(
    tensor_dict: dict[str, torch.Tensor | Any],
    pp_group: GroupCoordinator,
    all_gather_group: GroupCoordinator | None,
    all_gather_tensors: dict[str, bool] | None,
    config: PPRefCacheConfig,
    phase1_plan: PPRefCachePhase1Plan | None = None,
) -> dict[str, torch.Tensor | Any] | None:
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

    q_payload, scales = _quantize_int8(send_tensor, config.int8_group_size)
    raw_tensors = {
        key: value
        for key, value in tensor_dict.items()
        if key != "hidden_states"
    }
    return {
        _MARKER_KEY: _VERSION,
        _PHASE1_PLAN_ID_KEY: None if phase1_plan is None else phase1_plan.plan_id,
        _CODEC_KEY: config.codec,
        _TENSOR_NAME_KEY: "hidden_states",
        _ORIG_SHAPE_KEY: tuple(hidden_states.shape),
        _ORIG_DTYPE_KEY: hidden_states.dtype,
        _USED_ALL_GATHER_KEY: used_all_gather,
        _GROUP_SIZE_KEY: config.int8_group_size,
        _RAW_TENSORS_KEY: raw_tensors,
        _Q_PAYLOAD_KEY: q_payload,
        _SCALES_KEY: scales,
    }


def is_pp_refcache_packet(tensor_dict: dict[str, torch.Tensor | Any]) -> bool:
    return tensor_dict.get(_MARKER_KEY) == _VERSION


def _decode_packet(
    tensor_dict: dict[str, torch.Tensor | Any],
    all_gather_group: GroupCoordinator | None,
    expected_phase1_plan: PPRefCachePhase1Plan | None = None,
) -> dict[str, torch.Tensor | Any]:
    if not is_pp_refcache_packet(tensor_dict):
        return tensor_dict

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

    hidden_states = _dequantize_int8(
        q_payload,
        scales,
        int(tensor_dict[_GROUP_SIZE_KEY]),
        tensor_dict[_ORIG_DTYPE_KEY],
    )
    if tensor_dict[_USED_ALL_GATHER_KEY]:
        if all_gather_group is None:
            raise ValueError("PP RefCache packet requires an all-gather group")
        hidden_states = all_gather_group.all_gather(hidden_states, dim=0)
    hidden_states = hidden_states.reshape(tensor_dict[_ORIG_SHAPE_KEY])

    decoded = dict(tensor_dict[_RAW_TENSORS_KEY])
    decoded[tensor_dict[_TENSOR_NAME_KEY]] = hidden_states
    return decoded


def send_pp_refcache_phase1_plan(
    pp_group: GroupCoordinator,
    plan: PPRefCachePhase1Plan,
    dst: int | None = None,
) -> None:
    pp_group.send_tensor_dict(_phase1_to_tensor_dict(plan), dst=dst)


def recv_pp_refcache_phase1_plan(
    pp_group: GroupCoordinator,
    src: int | None = None,
) -> PPRefCachePhase1Plan | None:
    return _phase1_from_tensor_dict(pp_group.recv_tensor_dict(src=src))


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
    )
    if packet is None:
        return pp_group.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )

    return pp_group.isend_tensor_dict(
        packet,
        dst=dst,
        all_gather_group=all_gather_group,
        all_gather_tensors=_packet_all_gather_overrides(all_gather_tensors),
    )


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
