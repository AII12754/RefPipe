# PP RefCache Design

## Introduction

Pipeline parallelism sends intermediate activations from one pipeline stage to
the next. For long prefill workloads, this tensor transfer can become a
significant bandwidth cost. PP RefCache is a proposed transport optimization
for prefill-time PP boundaries. It reuses historical boundary activations as
references and sends a quantized delta instead of the full activation when the
current token context matches a cached context.

This is a lossy, experimental transport optimization. It must be disabled by
default and guarded by runtime fallbacks.

## Current Status

The current branch implements the first end-to-end V1 prototype with the
following scope:

- Phase 1 and Phase 2 are both wired into the PP path.
- `hidden_states` is the only compressed tensor. `residual` is not sent
  separately and is not part of RefCache state.
- RefCache matching is prefill-only. Decode rows may still use INT8 transport
  when the batch-level gate enables compression, but they do not participate in
  matching or cache commit.
- The only supported codec is groupwise INT8.
- Long-lived RefCache state stays on CPU. GPU memory is used only for a bounded
  staging buffer that prefetches matched reference activations for the current
  batch.
- TP all-gather packets are supported in the current vLLM flattened-slice form.
  RefCache delta is enabled when the local PP slice aligns to whole token rows;
  otherwise the packet falls back to raw INT8 and skips cache commit.
- Sender and receiver both commit reconstructed hidden states, not original
  hidden states.
- Cache commit is asynchronous and flushed before later matching or reference
  prefetch so correctness remains fail-closed.

This means the document below describes both the stable design constraints and
the current prototype behavior. Where the original milestone plan and the
current implementation differ, the implementation should be treated as the
source of truth for this branch.

## Current PP Transfer Path

Today, a PP boundary has one logical intermediate tensor transfer:

1. The downstream PP rank posts an `irecv_tensor_dict()` before executing its
   stage.
2. The upstream PP rank runs forward and returns `IntermediateTensors`.
3. The upstream rank optionally consolidates `hidden_states + residual`.
4. The upstream rank sends `IntermediateTensors.tensors` through
   `isend_tensor_dict()`.

`send_tensor_dict()` internally sends tensor metadata before tensor payloads,
but this metadata only describes tensor shape, dtype, and device. It is not an
algorithm-level phase that can be used to prefetch reference activations.

PP RefCache therefore needs a real two-phase protocol:

1. Phase 1 sends the match plan before the upstream stage finishes forward.
2. Phase 2 sends the compressed activation packet after forward.

The Phase 1 window lets the downstream stage look up or prefetch reference
activations before Phase 2 arrives.

## Goals

- Reduce PP boundary communication during prefill.
- Support arbitrary PP size by maintaining state per adjacent PP boundary.
- Support TP with a performance-first order: shard first, quantize second,
  send compressed shards over the PP link.
- Support mixed batches by making compression decisions from total batch hidden
  bytes while applying RefCache matching only to prefill regions.
- Reuse existing vLLM quantization operators where they fit.
- Keep the receiver deterministic by executing a sender-selected match plan.

## Non-Goals

- This design does not address privacy. Downstream PP ranks may have request
  and token state.
- This design does not target decode latency. Decode regions may use raw
  quantized transport, but they do not participate in RefCache matching or
  commit.
- This design does not require bitwise-equivalent outputs. Quality controls and
  fallbacks are required.
- This design does not initially support INT4 delta compression. INT4 can be a
  later codec after the transport and cache protocol are proven.
- This design does not compress or transmit `residual` separately. PP RefCache
  assumes residual consolidation remains enabled so the boundary payload is
  `hidden_states`.

## Boundary State

Each adjacent PP boundary owns independent RefCache state:

```text
PP rank k -> PP rank k+1:
  sender:   BoundaryActivationTable[k]
  receiver: BoundaryRefStore[k]
```

Different PP boundaries are different layer ranges and must not share reference
activations.

The cache identity should be deterministic rather than a global opaque counter:

```text
TokenUID = (request_cache_id, token_index)
StoreKey = (boundary_id, tp_rank, shard_mode, TokenUID)
```

`request_cache_id` should be an integer assigned for the active request
lifetime. It avoids sending request ID strings in the hot path. A future version
can add a cache salt or lineage ID for resumed or migrated requests.

## Matching

The sender is the only rank that decides the match plan. The receiver may have
the same token state, but it must not independently rerun matching for the
packet. Independent matching would require cache contents, eviction order,
preemption handling, and TP shard visibility to remain identical on both sides.
Any mismatch would decode against the wrong reference.

The sender classifies prefill regions by token fingerprints using a bigram
index and maximal forward matching. The match plan is span encoded:

```text
MatchSpan:
  dst_start: int32
  ref_token_uid_start: int64 or packed int32 pair
  length: int32
  codec: uint8
```

For contiguous matches from the same cached request, all reference token IDs in
the span are derived as `ref_token_uid_start + offset`. This makes the plan
size `O(num_spans)` rather than `O(num_tokens)`.

The match plan communication cost is expected to be small. For a model with
hidden size 4096, raw fp16 hidden data is 8192 bytes per token. Even a naive
per-token `(dst_pos, ref_uid)` plan is usually below 0.5% of compressed FP8
payload size. Span encoding reduces this further.

## Mixed Batch Policy

vLLM can schedule decode and prefill work in the same batch. The design keeps
transport compression separate from RefCache matching:

```text
Compressed transport gate:
  Use total hidden bytes to decide whether compressed transport is worthwhile.

RefCache matching:
  Prefill matched tokens:   reference + quantized delta
  Prefill unmatched tokens: quantized hidden
  Decode tokens:            quantized hidden or raw fp16 fallback
```

Decode tokens do not participate in RefCache matching or cache commit. They may
still be included in the same compressed packet if the total batch is large
enough to justify quantization overhead. This is a performance decision rather
than a semantic limitation: decode usually has very few tokens per request, is
latency-critical, and adds more synchronization and cache-lifetime complexity
than it is expected to save in PP bandwidth.

Initial runtime gates should be conservative:

```text
min_hidden_bytes
min_total_tokens
min_matched_rate_for_delta
max_estimated_packet_ratio
```

If the estimated packet size is close to raw fp16 or the batch is too small, the
transport should fall back to the existing `IntermediateTensors` send path.

## Tensor Parallelism

The performance-first TP rule is:

```text
shard first -> quantize local shard -> PP send compressed shard
```

This avoids repeated full-tensor quantization on every TP rank.

### Replicated Hidden Path

When the PP boundary tensor is replicated across TP ranks, vLLM can currently
send slices of the replicated tensor and all-gather on the receiver. PP
RefCache should replace this with compressed local slices:

```text
sender TP rank t:
  full hidden -> local send slice -> quantize -> PP send compressed slice

receiver TP rank t:
  recv compressed slice -> decode local slice -> TP all-gather reconstructed hidden
```

The all-gather happens after decode so the PP link carries compressed data.

### Sequence Parallel Path

When sequence parallelism is enabled, some intermediate tensors may be token
sharded across TP ranks. In this case, the local token shard should be
compressed directly. The packet must carry enough shard metadata to make cache
commit unambiguous:

```text
shard_mode = sp_token_slice
global_num_tokens
local_token_indices or token range
tp_rank
tp_size
```

The cache key includes `tp_rank` for sequence-parallel local shards.

## Phase 1 Protocol

Phase 1 is a small tensorized message sent before the upstream PP stage's
forward completes. It should not use Python object lists in the hot path.

Example fields:

```text
Phase1RefPlan:
  batch_id
  boundary_id
  tp_rank
  tp_size
  shard_mode
  num_global_tokens
  hidden_dim
  token_segments: int32[num_segments, fields]
  match_spans: int64/int32[num_spans, fields]
  self_ref_spans: int32[num_self_ref_spans, fields]
  cache_epoch
```

The receiver uses Phase 1 to validate reference availability and prefetch
reference shards. The first implementation can avoid a reverse miss response by
only referencing entries known to have been committed by the receiver. Later
versions can add a miss bitmap and fallback selected spans to raw transport.

## Phase 2 Protocol

Phase 2 carries the compressed activation packet:

```text
Phase2RefCachePacket:
  packet_header
  region_descriptors
  q_payload
  scales
  zeros or scale metadata if needed
  raw_fp16_fallback regions
  commit_token_uids
```

The receiver decodes Phase 2 using the sender-selected Phase 1 plan.

Sender and receiver both commit reconstructed activations, not original
activations. This is required for future deltas to use numerically consistent
references:

```text
sender:
  quantize packet
  reconstruct hidden from packet
  commit reconstructed hidden

receiver:
  decode packet
  commit reconstructed hidden
```

If the sender cached original hidden while the receiver cached decoded hidden,
future deltas would be computed and applied against different references.

## Quantization Strategy

The initial codec should prioritize low latency and a low correctness risk.

Recommended order:

1. FP8 or INT8 raw activation transport without RefCache. This establishes the
   PP packet path and measures quantization overhead.
2. FP8 or INT8 delta transport for matched prefill regions.
3. INT4 delta with outlier preservation after the protocol is stable.

Existing vLLM quantization utilities can be reused for the first encode path:

- `scaled_fp8_quant`
- `per_token_group_quant_fp8`
- `per_token_group_quant_int8`
- `per_token_quant_int8`

KV-cache-specific reshape/cache operators should not be reused directly. They
are coupled to KV head layouts, cache blocks, slot mappings, and attention
backend conventions. PP boundary activations are local `[tokens, hidden]`
tensors or TP-local shards.

The long-term fast path should use dedicated fused kernels:

```text
encode matched:
  hidden, ref -> q_delta, scales, reconstructed_hidden

decode matched:
  q_delta, scales, ref -> reconstructed_hidden
```

Using separate subtract, quantize, dequantize, and add kernels will add large
intermediate reads and writes. This is acceptable for an MVP but not for the
performance target.

FP8 payloads should be sent as `uint8` tensors unless distributed dtype support
for the selected FP8 type is verified on all target backends.

## Integration Points

Primary V1 GPU worker path:

- Receive side: `vllm/v1/worker/gpu_worker.py` posts PP receives before model
  execution.
- Send side: `vllm/v1/worker/gpu_worker.py` sends `IntermediateTensors` after
  model execution.
- Model runner: non-first PP ranks copy received `IntermediateTensors` into
  local buffers before forward.
- `IntermediateTensors.consolidate_residual()` should remain enabled for the
  scope of this feature so the transport handles only `hidden_states`.

The feature should be guarded by an explicit configuration flag or environment
variable and should fall back to the current PP transfer path whenever an
unsupported model path or batch shape is detected.

## Implementation Status

### Completed in the Current Prototype

- Disabled-by-default feature flag and runtime fallback path.
- Tensorized Phase 1 packet with deterministic `plan_id`.
- Batch-level gating by total hidden bytes.
- Prefill token segment extraction from scheduler output.
- Sender-side bigram maximal-forward matching for prefill regions.
- Raw INT8 transport for unmatched rows.
- INT8 delta transport for matched prefill rows.
- Boundary-local CPU RefCache on sender and receiver.
- Deterministic fail-closed decode if a requested receiver reference is missing.
- Packet ratio gate and minimum matched-rate gate.
- Triton encode/decode kernels for matched INT8 delta packets.
- CPU-to-GPU reference prefetch staging with a bounded GPU buffer.
- TP all-gather support for whole-token-row slices.
- Unit tests for Phase 1/Phase 2 protocol, fallback behavior, TP row-slice
  behavior, CUDA kernel round-trip, and receiver miss failure.
- Benchmark harness for PP RefCache end-to-end timing.

### Remaining Work

- Broader TP coverage beyond the current whole-token-row all-gather case.
- Mixed-batch validation under more realistic scheduler patterns.
- Longer-running cache-lifetime validation, especially eviction behavior under
  sustained multi-request load.
- Cross-node benchmarking. Same-node benchmarks are useful for overhead
  measurement but do not capture the bandwidth savings that motivate the
  feature.
- Quality evaluation and acceptance thresholds for lossy INT8 transport.
- Possible future codecs such as FP8 raw transport or INT4 delta transport.

## Development Plan

### Milestone 1: Baseline Transport Packet

- Status: complete in the current branch.
- Delivered:
  - disabled-by-default PP RefCache feature flag;
  - tensorized packet metadata and fallback path;
  - batch-level gating by total hidden bytes;
  - raw INT8 transport for `hidden_states`;
  - PP size greater than one and current TP local-slice send behavior;
  - unit tests for packet encode/decode shape and fallback behavior.
- Notes:
  - the branch does not implement FP8 transport;
  - `hidden_states` remains the only compressed tensor.

Initial implementation flags:

```text
VLLM_PP_REFCACHE_ENABLE=0
VLLM_PP_REFCACHE_CODEC=int8
VLLM_PP_REFCACHE_MIN_HIDDEN_BYTES=1048576
VLLM_PP_REFCACHE_INT8_GROUP_SIZE=128
VLLM_PP_REFCACHE_MAX_TOKENS=100000
VLLM_PP_REFCACHE_MIN_MATCH_RATE=0.0
VLLM_PP_REFCACHE_MAX_PACKET_RATIO=1.0
```

### Milestone 2: Phase 1 Match Plan

- Status: complete, but the implementation went past the original milestone
  boundary.
- Delivered:
  - request token segments built from scheduler output and request state;
  - pre-forward Phase 1 send and receive for PP boundaries;
  - span-encoded `token_segments`, `match_spans`, and `self_ref_spans` fields;
  - deterministic `plan_id` carried in Phase 2 packets and validated on
    receive.
- Notes:
  - unlike the original plan, `match_spans` is not left empty in this branch;
    sender-side matching is already active.

### Milestone 3: RefCache Delta Codec

- Status: substantially complete for the current prototype scope.
- Delivered:
  - boundary-local activation tables and reference stores;
  - bigram maximal-forward matching for prefill regions;
  - matched `ref + quant(delta)` transport and unmatched raw INT8 transport;
  - reconstructed hidden commit on both sender and receiver;
  - deterministic cache capacity eviction;
  - fail-closed decode on receiver miss;
  - minimum matched-rate and packet-ratio gates.
- Notes:
  - cache state is CPU-resident;
  - current Triton kernels cover the matched INT8 path;
  - raw fp16 fallback regions and stronger numerical guardrails remain future
    codec-hardening work.

### Milestone 4: TP and Sequence Parallel Coverage

- Status: partially complete.
- Delivered:
  - local-slice compression for the current all-gather PP path when the flat
    slice aligns to whole token rows;
  - decode of local slices before TP all-gather;
  - raw INT8 fallback when a flat slice cuts through a token row.
- Remaining:
  - broader TP coverage and explicit sequence-parallel metadata contracts;
  - larger TP test matrix and more mixed prefill/decode scheduler coverage.

### Milestone 5: Fused Kernels and Tuning

- Status: partially complete.
- Delivered:
  - first Triton encode/decode kernels in
    `vllm/distributed/pp_refcache_kernels.py`;
  - threshold tuning to the extent needed to keep current-node overhead within
    a reasonable range;
  - same-node benchmark coverage for short and long prefill workloads.
- Remaining:
  - more aggressive kernel tuning;
  - cross-node benchmark validation;
  - quality regression tests and explicit accuracy criteria.

## Open Questions

- Exact `request_cache_id` lifetime under preemption, resumed requests, and
  disaggregated prefill needs to be finalized.
- The first supported codec should be selected after benchmarking FP8 and INT8
  encode/decode latency on target GPUs.
- Cache eviction can start with deterministic FIFO/LRU but may need receiver
  miss feedback for long-running workloads.
- Model Runner V2 support should be planned separately after V1 behavior is
  validated.
