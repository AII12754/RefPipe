# PP RefCache Latency Optimization Analysis

This document analyzes latency bottlenecks in the current PP RefCache prototype
and proposes targeted optimizations. References to source code use line numbers
from `vllm/distributed/pp_refcache.py` and
`vllm/distributed/pp_refcache_kernels.py`.

## End-to-End Timeline

```
Sender (PP rank k):
  build_phase1_plan() [CPU]  →  send Phase 1 [network]  →  prefetch send refs [CPU→GPU]
                                                                   ‖ (overlap with forward)
                              execute_model() [GPU forward] ───────‖
                                                                   ↓
  _flush_pending_commits() [may block]  →  _encode_packet() [GPU/Triton]  →  isend Phase 2 [network]

Receiver (PP rank k+1):
  recv Phase 1 [network]  →  prefetch recv refs [CPU→GPU]
                                    ‖ (overlap with sender forward)
  irecv Phase 2 [post] ────────────‖
                                    ↓
  execute_model()  →  [Phase 2 arrives]  →  _decode_packet() [GPU/Triton]  →  forward
```

## Optimization 1: Eliminate plan_id Hashing (Priority: P0)

**Root cause:** `_phase1_plan_id()` (pp_refcache.py:726) computes a BLAKE2b
hash over the full byte content of `token_segments`, `match_spans`, and
`self_ref_spans` tensors. For a 100K-token prefill batch, these tensors can
contain tens of thousands of rows. The hashing walks every byte on the CPU
inside `build_phase1_plan()`, adding O(tokens) CPU work to the critical path
before Phase 1 send.

**Why it matters:** The plan_id only needs to pair a Phase 1 message with
its corresponding Phase 2 packet within a single PP transfer. It does not need
cryptographic guarantees or cross-batch uniqueness.

**Proposed fix:** Replace the hash with a monotonic counter combined with batch
identity metadata, e.g. `(batch_counter, tp_rank, num_global_tokens)`. The
sender increments the counter per batch. Determinism is satisfied because the
counter is local to the sender process.

**Expected impact:** Saves 10–30% of `build_phase1_plan()` CPU time depending
on batch size. Removes O(tokens) hashing from the hot path entirely.

## Optimization 2: Recoverable `_has_evicted` Flag (Priority: P1)

**Root cause:** `_BoundaryRefCache._has_evicted` (pp_refcache.py:270) is set to
`True` on the first eviction and never reset. This forces `match_segment()` to
use the slow matching path permanently: per-token hash lookups
(`_has_token` + `_get_token_fp`) instead of direct Python list comparisons.

The fast path is O(match_len) contiguous memory reads; the slow path is
O(match_len × log N) with two dict lookups per iteration.

**Why it matters:** In long-running serving, eviction is expected under
sustained load. After the first eviction, all subsequent matching degrades
indefinitely, even for segments that were never touched by eviction.

**Proposed fix:** After eviction completes, clear stale entries from
`_bigram_index` and reset `_has_evicted = False`. Only segments whose tokens
were partially evicted need special handling. Alternatively, maintain a
per-segment `valid` flag so the fast path can still be used for intact
segments.

**Expected impact:** Restores fast-path matching after eviction events,
reducing match latency by 2–5× under sustained load.

## Optimization 3: Fuse Triton 2D Grid into 1D (Priority: P1)

**Root cause:** The Triton encode/decode kernels use a 2D launch grid of
`(n_rows, n_cols // group_size)` (pp_refcache_kernels.py:158). For
hidden_dim=4096 and group_size=128, this launches 32 thread blocks per row.
For 1000 tokens, that is 32,000 blocks, each processing only 128 elements.
An A100 has 108 SMs, so most time is spent on block scheduling rather than
computation for small-to-medium batches.

**Proposed fix:** Use a 1D grid `(n_rows,)` where each program loops over all
groups in a row. This reduces launch overhead at the cost of slightly higher
register pressure per block. The `BLOCK` size can be tuned per group_size.

**Expected impact:** 30–50% reduction in encode/decode kernel latency for
batches under ~100 tokens. Minimal impact for large prefill batches that
are already compute-bound.

## Optimization 4: Enable Fused Kernel for TP Partial-Slice (Priority: P0)

**Root cause:** `_all_gather_slice_keeps_global_rows()` (pp_refcache.py:686)
guards whether prefetched refs and the fused Triton kernel can be used. When
the TP all-gather slice does not cover all token rows (i.e., `local_token_start
!= 0` or `local_token_count != num_global_tokens`), the guard returns `False`
and the code falls back to a 4-pass slow path:

```
_subtract_delta_refs() → _quantize_int8() → _dequantize_int8() → _apply_delta_refs()
```

Each pass reads and writes the full hidden_states tensor, consuming 4× the
memory bandwidth of the single fused kernel.

**Why it matters:** Any TP configuration where the flat send slice cuts a
non-zero offset into the batch hits this fallback. This is the common case
for sequence-parallel or uneven TP sharding.

**Proposed fix:** The `row_to_ref_indices` helper already handles partial
mappings by leaving unmatched rows at `-1`. The fused Triton kernel already
checks `ref_idx >= 0` before loading refs. The only missing piece is aligning
the `positions` tensor to the local token range. Extend the kernel interface to
accept local-row-relative positions directly.

**Expected impact:** 3–4× reduction in encode latency for TP partial-slice
scenarios by collapsing 4 GPU passes into 1.

## Optimization 5: Batch CPU-to-GPU Copies in store_runs (Priority: P1)

**Root cause:** `_RefPrefetchBuffer.store_runs()` (pp_refcache.py:227) copies
each ref run to the GPU staging buffer with a separate `copy_(non_blocking=True)`
call. Multiple small DMA transfers have higher aggregate launch overhead and
worse DMA engine utilization than a single large transfer.

**Proposed fix:** Concatenate all ref runs on the CPU side first
(`torch.cat(refs_cpu, dim=0)`), then issue a single `copy_` to the GPU staging
buffer. The CPU concatenation cost (a memory copy) is negligible compared to
PCIe DMA latency.

**Expected impact:** 20–40% reduction in prefetch latency for batches with
multiple discontiguous ref runs.

## Optimization 6: Decouple Commit Flush from Encode/Decode Hot Path (Priority: P2)

**Root cause:** `_encode_packet()` and `_decode_packet()` both call
`_flush_pending_commits()` at entry, which blocks until all previously
submitted async cache commits complete. A commit includes GPU→CPU copy
(hundreds of MB for large batches), bigram indexing (O(tokens) CPU work),
and FIFO eviction.

**Why it matters:** If the previous batch's commit is still in flight when
the current batch enters encode, the hot path blocks on it. With only 2
ThreadPoolExecutor workers, commits from multiple PP boundaries can queue up.

**Proposed fix:**
- Increase ThreadPoolExecutor worker count from 2 to 4–8, configurable.
- Separate bigram index updates from the critical commit path: commit
  hidden_states to the segment store synchronously (needed for future
  matching), but defer bigram index insertion to a lower-priority background
  task.
- Consider only flushing commits that affect the current plan's match spans,
  rather than all pending commits.

**Expected impact:** Eliminates tail-latency spikes caused by cross-batch
commit queueing.

## Optimization 7: Reduce CPU Dict Lookup Overhead in Hot Paths (Priority: P2)

**Root cause:** `match_segment()` and `get_ref_runs_cpu()` perform heavy
Python dict lookups on the CPU hot path:

- `match_segment()`: per-bigram candidate iteration over a `deque(maxlen=32)`
  from `_bigram_index`, with per-candidate `_token_refs.get()` calls.
- `get_ref_runs_cpu()`: per-matched-token `_token_refs.get()` + `_segments.get()`
  for ref resolution.
- Dict keys are `tuple[int, int]` which have non-trivial hash and allocation
  overhead compared to a single packed integer.

**Proposed fix:**
- Encode `(req_uid, token_idx)` as a single 64-bit integer key:
  `(req_uid << 32) | (token_idx & 0xFFFFFFFF)`. Avoids tuple allocation and
  hashing overhead on every lookup.
- In `get_ref_runs_cpu()`, detect contiguous runs in the same segment and
  batch them into a single slice rather than querying per-token.
- Consider Cython/C++ for the matching inner loop if Python overhead remains
  significant after the above fixes.

**Expected impact:** 30–50% reduction in `build_phase1_plan()` and prefetch
CPU time for high-match-rate batches (10K+ matched tokens).

## Optimization 8: Dedicated CUDA Stream for Commit Copies (Priority: P3)

**Root cause:** `_BoundaryRefCache.commit()` performs GPU→CPU copies on the
default CUDA stream (pp_refcache.py:522). This contends with the GPU forward
pass for memory bandwidth. For memory-bound models, the DMA transfer directly
slows down computation.

**Proposed fix:** Create a dedicated CUDA stream for cache commit copies.
Use `with torch.cuda.stream(commit_stream): rows.copy_(source, non_blocking=True)`.
The GPU copy engine can overlap DMA with compute on a separate stream.

**Expected impact:** Reduces commit interference with forward pass execution,
particularly beneficial for memory-bound model configurations.

## Optimization 9: Pre-allocate Commit and Prefetch Buffers (Priority: P3)

**Root cause:** `_BoundaryRefCache.commit()` allocates fresh CPU pinned memory
on every call (`torch.empty(..., pin_memory=True)`), and `_RefPrefetchBuffer`
reallocates GPU buffers when the required size exceeds capacity. Frequent CUDA
memory allocation has implicit synchronization overhead.

**Proposed fix:** Use a buffer pool keyed by tensor shape. Warmup can determine
the maximum expected buffer sizes. Reuse allocations across batches.

**Expected impact:** Eliminates allocation-induced synchronization points;
minor but consistent latency improvement across all batches.

## Optimization 10: Unify Ref Lookup Paths to Avoid Double Fetch (Priority: P3)

**Root cause:** `_encode_packet()` first attempts `_pop_prefetched_refs()`.
If that returns `None` (e.g., TP partial-slice), the fallback code in
`_encode_int8_with_refs()` calls `_SEND_CACHE.get_refs()` which does a
synchronous CPU→GPU copy on the encode critical path. This is a second,
slower ref resolution for the same data.

**Proposed fix:** Primary fix is Optimization 4 (enable fused kernel for
TP partial-slice), which eliminates the main reason prefetch is skipped. As a
secondary safeguard, ensure the prefetch buffer is sized large enough to cover
the maximum expected matched tokens so `store()` never fails due to capacity.

**Expected impact:** Eliminates the slow-path synchronous CPU→GPU copy from
the encode critical path for the common case.

## Priority Matrix

| Priority | Optimization | Affected Scenario | Complexity |
|----------|-------------|-------------------|------------|
| P0 | #1 Eliminate plan_id hashing | All batches | Low |
| P0 | #4 Fused kernel for TP partial-slice | TP with non-zero slice offset | Medium |
| P1 | #2 Recoverable `_has_evicted` | Sustained load with eviction | Medium |
| P1 | #3 Triton 1D grid fusion | Small batch (< 100 tokens) | Medium |
| P1 | #5 Batch CPU→GPU copies in store_runs | Multiple discontiguous ref runs | Low |
| P2 | #6 Decouple commit flush from hot path | Cross-batch tail latency | Medium |
| P2 | #7 Reduce CPU dict lookup overhead | High match rate, large batches | High |
| P3 | #8 Dedicated CUDA stream for commits | Memory-bound models | Medium |
| P3 | #9 Pre-allocate commit/prefetch buffers | All batches | Low |
| P3 | #10 Unify ref lookup paths | Covered by #4 | Low |
