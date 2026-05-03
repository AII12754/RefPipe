# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental PP RefCache transport helpers.

Milestone 1 only implements a disabled-by-default raw INT8 transport packet for
PP boundary ``hidden_states``. It does not implement reference matching or
delta coding yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import GroupCoordinator

_VERSION = 1
_MARKER_KEY = "__pp_refcache_version__"
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


def get_pp_refcache_config() -> PPRefCacheConfig:
    return PPRefCacheConfig(
        enabled=envs.VLLM_PP_REFCACHE_ENABLE,
        codec=envs.VLLM_PP_REFCACHE_CODEC.lower(),
        min_hidden_bytes=envs.VLLM_PP_REFCACHE_MIN_HIDDEN_BYTES,
        int8_group_size=envs.VLLM_PP_REFCACHE_INT8_GROUP_SIZE,
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
) -> dict[str, torch.Tensor | Any]:
    if not is_pp_refcache_packet(tensor_dict):
        return tensor_dict

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


def isend_pp_refcache_tensor_dict(
    pp_group: GroupCoordinator,
    tensor_dict: dict[str, torch.Tensor | Any],
    dst: int | None = None,
    all_gather_group: GroupCoordinator | None = None,
    all_gather_tensors: dict[str, bool] | None = None,
) -> list[Any]:
    config = get_pp_refcache_config()
    packet = _encode_packet(
        tensor_dict,
        pp_group,
        all_gather_group,
        all_gather_tensors,
        config,
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
        decoded = _decode_packet(tensor_dict, all_gather_group)
        if decoded is not tensor_dict:
            tensor_dict.clear()
            tensor_dict.update(decoded)

    postprocess.append(_decode_postprocess)
    return tensor_dict, handles, postprocess
