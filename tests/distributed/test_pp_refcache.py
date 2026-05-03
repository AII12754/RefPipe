# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import pp_refcache


class FakePPGroup:
    def __init__(self, use_all_gather: bool = False):
        self.use_all_gather = use_all_gather

    def _should_use_all_gather(self, *args, **kwargs) -> bool:
        return self.use_all_gather


class FakeAllGatherGroup:
    world_size = 2
    rank_in_group = 0

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        return torch.cat([tensor, tensor], dim=dim)


def test_pp_refcache_encode_decode_int8_packet() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )
    hidden_states = torch.tensor(
        [[-1.0, -0.5, 0.25, 1.0], [2.0, -2.0, 0.0, 0.5]],
        dtype=torch.float16,
    )
    tensor_dict = {
        "hidden_states": hidden_states,
        "residual": None,
    }

    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        tensor_dict,
        FakePPGroup(),
        None,
        None,
        config,
    )

    assert packet is not None
    decoded = pp_refcache._decode_packet(packet, None)  # type: ignore[attr-defined]
    assert decoded["residual"] is None
    assert decoded["hidden_states"].shape == hidden_states.shape
    assert decoded["hidden_states"].dtype == hidden_states.dtype


def test_pp_refcache_encode_returns_none_for_small_tensor() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=1024,
        int8_group_size=4,
    )
    tensor_dict = {
        "hidden_states": torch.ones((1, 4), dtype=torch.float16),
        "residual": None,
    }

    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        tensor_dict,
        FakePPGroup(),
        None,
        None,
        config,
    )

    assert packet is None


def test_pp_refcache_encode_returns_none_for_extra_tensor() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )
    tensor_dict = {
        "hidden_states": torch.ones((1, 4), dtype=torch.float16),
        "residual": torch.ones((1, 4), dtype=torch.float16),
    }

    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        tensor_dict,
        FakePPGroup(),
        None,
        None,
        config,
    )

    assert packet is None


def test_pp_refcache_encode_decode_all_gather_slice() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )
    hidden_states = torch.arange(16, dtype=torch.float16).reshape(2, 8)
    tensor_dict = {
        "hidden_states": hidden_states,
    }

    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        tensor_dict,
        FakePPGroup(use_all_gather=True),
        FakeAllGatherGroup(),
        None,
        config,
    )

    assert packet is not None
    decoded = pp_refcache._decode_packet(  # type: ignore[attr-defined]
        packet,
        FakeAllGatherGroup(),
    )
    assert decoded["hidden_states"].shape == hidden_states.shape
