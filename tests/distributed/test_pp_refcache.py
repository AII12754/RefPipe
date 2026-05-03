# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed import pp_refcache


@pytest.fixture(autouse=True)
def clear_pp_refcache_state() -> None:
    pp_refcache.clear_pp_refcache_state()


class FakeNewReq:
    def __init__(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids
        self.prefill_token_ids = prompt_token_ids
        self.num_computed_tokens = num_computed_tokens


class FakeCachedReqs:
    req_ids = ["decode", "prefill"]
    new_token_ids = [[], [20, 21]]
    all_token_ids = {"prefill": [16, 17, 18, 19, 20, 21]}
    num_computed_tokens = [10, 4]

    def is_context_phase(self, req_id: str) -> bool:
        return req_id == "prefill"


class FakeSchedulerOutput:
    scheduled_new_reqs = [
        FakeNewReq("new", [1, 2, 3, 4, 5], 1),
    ]
    scheduled_cached_reqs = FakeCachedReqs()
    num_scheduled_tokens = {
        "decode": 1,
        "new": 3,
        "prefill": 2,
    }
    total_num_scheduled_tokens = 6


class SinglePrefillSchedulerOutput:
    scheduled_cached_reqs = FakeCachedReqs()
    total_num_scheduled_tokens = 4

    def __init__(self, req_id: str, token_ids: list[int]) -> None:
        self.scheduled_new_reqs = [FakeNewReq(req_id, token_ids, 0)]
        self.num_scheduled_tokens = {req_id: len(token_ids)}


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


def test_pp_refcache_phase1_plan_segments_prefill_only() -> None:
    plan = pp_refcache.build_phase1_plan(FakeSchedulerOutput(), None)

    assert plan.num_global_tokens == 6
    assert plan.tp_rank == 0
    assert plan.tp_size == 1
    assert plan.match_spans.shape == (0, 6)
    assert plan.self_ref_spans.shape == (0, 4)
    assert plan.token_segments.shape == (2, 5)
    # Batch order follows GPUModelRunner.prepare_inputs: decode first, then
    # smaller prefills before larger prefills.
    assert plan.token_segments[:, :3].tolist() == [
        [1, 2, 4],
        [3, 3, 1],
    ]


def test_pp_refcache_phase2_packet_carries_phase1_plan_id() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )
    plan = pp_refcache.build_phase1_plan(FakeSchedulerOutput(), None)
    hidden_states = torch.ones((2, 4), dtype=torch.float16)
    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        {"hidden_states": hidden_states},
        FakePPGroup(),
        None,
        None,
        config,
        plan,
    )

    assert packet is not None
    decoded = pp_refcache._decode_packet(  # type: ignore[attr-defined]
        packet,
        None,
        plan,
    )
    assert decoded["hidden_states"].shape == hidden_states.shape


def test_pp_refcache_phase2_rejects_mismatched_phase1_plan() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )
    plan = pp_refcache.build_phase1_plan(FakeSchedulerOutput(), None)
    wrong_plan = pp_refcache.PPRefCachePhase1Plan(
        plan_id=plan.plan_id + 1,
        num_global_tokens=plan.num_global_tokens,
        tp_rank=plan.tp_rank,
        tp_size=plan.tp_size,
        token_segments=plan.token_segments,
        match_spans=plan.match_spans,
        self_ref_spans=plan.self_ref_spans,
    )
    packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        {"hidden_states": torch.ones((2, 4), dtype=torch.float16)},
        FakePPGroup(),
        None,
        None,
        config,
        plan,
    )

    assert packet is not None
    with pytest.raises(ValueError, match="Phase 2 packet does not match"):
        pp_refcache._decode_packet(  # type: ignore[attr-defined]
            packet,
            None,
            wrong_plan,
        )


def test_pp_refcache_delta_roundtrip_uses_committed_refs() -> None:
    config = pp_refcache.PPRefCacheConfig(
        enabled=True,
        codec="int8",
        min_hidden_bytes=0,
        int8_group_size=4,
    )

    first_plan = pp_refcache.build_phase1_plan(
        SinglePrefillSchedulerOutput("first", [10, 11, 12, 13]),
        None,
    )
    first_hidden = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    first_packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        {"hidden_states": first_hidden},
        FakePPGroup(),
        None,
        None,
        config,
        first_plan,
    )
    assert first_packet is not None
    first_decoded = pp_refcache._decode_packet(  # type: ignore[attr-defined]
        first_packet,
        None,
        first_plan,
    )

    second_plan = pp_refcache.build_phase1_plan(
        SinglePrefillSchedulerOutput("second", [10, 11, 12, 13]),
        None,
    )
    first_req_uid = first_plan.token_segments[0, 3].item()
    assert second_plan.match_spans.tolist() == [
        [1, 3, first_req_uid, 1, 0, 0]
    ]
    second_hidden = first_decoded["hidden_states"] + torch.full(
        (4, 4),
        0.25,
        dtype=torch.float16,
    )
    second_packet = pp_refcache._encode_packet(  # type: ignore[attr-defined]
        {"hidden_states": second_hidden},
        FakePPGroup(),
        None,
        None,
        config,
        second_plan,
    )

    assert second_packet is not None
    second_decoded = pp_refcache._decode_packet(  # type: ignore[attr-defined]
        second_packet,
        None,
        second_plan,
    )
    assert second_decoded["hidden_states"].shape == second_hidden.shape
    assert torch.allclose(second_decoded["hidden_states"], second_hidden, atol=0.05)


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
