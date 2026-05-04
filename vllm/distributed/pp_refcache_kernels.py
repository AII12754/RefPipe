# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton


@triton.jit
def _pp_refcache_encode_int8_kernel(
    hidden_ptr,
    row_to_ref_ptr,
    refs_ptr,
    q_ptr,
    scales_ptr,
    recon_ptr,
    n_cols: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    cols = group * group_size + offsets
    mask = offsets < group_size

    hidden = tl.load(hidden_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    ref_idx = tl.load(row_to_ref_ptr + row)
    ref = tl.zeros((BLOCK,), dtype=tl.float32)
    if ref_idx >= 0:
        ref = tl.load(refs_ptr + ref_idx * n_cols + cols, mask=mask).to(tl.float32)
        hidden = hidden - ref

    absmax = tl.max(tl.abs(hidden), axis=0)
    scale = tl.maximum(absmax / 127.0, 1.0e-10)
    q_f32 = tldevice.nearbyint(hidden / scale)
    q_f32 = tl.minimum(tl.maximum(q_f32, -128.0), 127.0)
    q_i8 = q_f32.to(tl.int8)

    tl.store(q_ptr + row * n_cols + cols, q_i8, mask=mask)
    tl.store(scales_ptr + row * (n_cols // group_size) + group, scale)

    reconstructed = q_f32 * scale
    if ref_idx >= 0:
        reconstructed = reconstructed + ref
    tl.store(recon_ptr + row * n_cols + cols, reconstructed, mask=mask)


@triton.jit
def _pp_refcache_decode_int8_kernel(
    q_ptr,
    scales_ptr,
    row_to_ref_ptr,
    refs_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    cols = group * group_size + offsets
    mask = offsets < group_size

    q = tl.load(q_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    scale = tl.load(scales_ptr + row * (n_cols // group_size) + group).to(tl.float32)
    hidden = q * scale
    ref_idx = tl.load(row_to_ref_ptr + row)
    if ref_idx >= 0:
        ref = tl.load(refs_ptr + ref_idx * n_cols + cols, mask=mask).to(tl.float32)
        hidden = hidden + ref
    tl.store(out_ptr + row * n_cols + cols, hidden, mask=mask)


@triton.jit
def _pp_refcache_decode_raw_int8_kernel(
    q_ptr,
    scales_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    cols = group * group_size + offsets
    mask = offsets < group_size

    q = tl.load(q_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    scale = tl.load(scales_ptr + row * (n_cols // group_size) + group).to(tl.float32)
    hidden = q * scale
    tl.store(out_ptr + row * n_cols + cols, hidden, mask=mask)


def _can_use_triton_refcache(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    refs: torch.Tensor,
    group_size: int,
) -> bool:
    return (
        HAS_TRITON
        and tensor.is_cuda
        and positions.is_cuda
        and refs.is_cuda
        and tensor.ndim == 2
        and refs.ndim == 2
        and tensor.is_contiguous()
        and refs.is_contiguous()
        and tensor.shape[1] == refs.shape[1]
        and group_size > 0
        and tensor.shape[1] % group_size == 0
        and positions.numel() == refs.shape[0]
    )


def _row_to_ref_indices(
    num_rows: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    row_to_ref = torch.full(
        (num_rows,),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    row_to_ref[positions.to(torch.long)] = torch.arange(
        positions.numel(),
        dtype=torch.int32,
        device=positions.device,
    )
    return row_to_ref


def triton_encode_int8_with_refs(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    refs: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not _can_use_triton_refcache(hidden_states, positions, refs, group_size):
        return None

    n_rows, n_cols = hidden_states.shape
    q_payload = torch.empty_like(hidden_states, dtype=torch.int8)
    scales = torch.empty(
        (n_rows, n_cols // group_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    reconstructed = torch.empty_like(hidden_states)
    row_to_ref = _row_to_ref_indices(n_rows, positions)

    grid = (n_rows, n_cols // group_size)
    _pp_refcache_encode_int8_kernel[grid](
        hidden_states,
        row_to_ref,
        refs,
        q_payload,
        scales,
        reconstructed,
        n_cols,
        group_size,
        BLOCK=triton.next_power_of_2(group_size),
    )
    return q_payload, scales, reconstructed


def triton_decode_int8(
    q_payload: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not (
        HAS_TRITON
        and q_payload.is_cuda
        and scales.is_cuda
        and q_payload.ndim == 2
        and q_payload.is_contiguous()
        and scales.is_contiguous()
        and group_size > 0
        and q_payload.shape[1] % group_size == 0
    ):
        return None

    n_rows, n_cols = q_payload.shape
    hidden_states = torch.empty((n_rows, n_cols), dtype=dtype, device=q_payload.device)
    grid = (n_rows, n_cols // group_size)
    _pp_refcache_decode_raw_int8_kernel[grid](
        q_payload,
        scales,
        hidden_states,
        n_cols,
        group_size,
        BLOCK=triton.next_power_of_2(group_size),
    )
    return hidden_states


def triton_decode_int8_with_refs(
    q_payload: torch.Tensor,
    scales: torch.Tensor,
    positions: torch.Tensor,
    refs: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not _can_use_triton_refcache(q_payload, positions, refs, group_size):
        return None

    n_rows, n_cols = q_payload.shape
    hidden_states = torch.empty((n_rows, n_cols), dtype=dtype, device=q_payload.device)
    row_to_ref = _row_to_ref_indices(n_rows, positions)

    grid = (n_rows, n_cols // group_size)
    _pp_refcache_decode_int8_kernel[grid](
        q_payload,
        scales,
        row_to_ref,
        refs,
        hidden_states,
        n_cols,
        group_size,
        BLOCK=triton.next_power_of_2(group_size),
    )
    return hidden_states
