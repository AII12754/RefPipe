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

## Development Plan

### Milestone 1: Baseline Transport Packet

- Add a disabled-by-default PP RefCache feature flag.
- Implement packet metadata structures with tensorized headers.
- Implement batch-level gating by total hidden bytes.
- Implement raw FP8 or INT8 transport for `hidden_states` without RefCache.
- Support PP size greater than one and TP local shard send.
- Fall back to existing PP transfer for unsupported tensors.
- Add unit tests for packet encode/decode shape and fallback behavior.

Initial implementation flags:

```text
VLLM_PP_REFCACHE_ENABLE=0
VLLM_PP_REFCACHE_CODEC=int8
VLLM_PP_REFCACHE_MIN_HIDDEN_BYTES=1048576
VLLM_PP_REFCACHE_INT8_GROUP_SIZE=128
```

### Milestone 2: Phase 1 Match Plan

- Build request token segments from `InputBatch` and request state.
- Add pre-forward Phase 1 send and receive for PP boundaries.
- Implement span-encoded match plans.
- Add receiver reference availability validation.
- Keep Phase 2 raw quantized transport until Phase 1 correctness is validated.

### Milestone 3: RefCache Delta Codec

- Implement boundary-local activation tables and reference stores.
- Add bigram maximal-forward matching for prefill regions.
- Encode matched regions as `ref + quant(delta)`.
- Encode unmatched regions as quantized raw hidden.
- Commit reconstructed hidden on both sender and receiver.
- Add cache capacity and deterministic eviction.
- Add numerical error checks and raw fp16 fallback regions.

### Milestone 4: TP and Sequence Parallel Coverage

- Implement replicated-hidden local slice compression.
- Decode local slices before TP all-gather.
- Add sequence-parallel token-shard packet metadata.
- Add tests for TP sizes greater than one.
- Add tests for mixed prefill/decode batches.

### Milestone 5: Fused Kernels and Tuning

- Add fused subtract-reference-and-quantize kernel.
- Add fused dequantize-and-add-reference kernel.
- Tune thresholds for hidden bytes, match rate, and estimated packet ratio.
- Benchmark against raw PP transfer on prefill-heavy workloads.
- Add quality regression tests for selected models and prompts.

## Open Questions

- Exact `request_cache_id` lifetime under preemption, resumed requests, and
  disaggregated prefill needs to be finalized.
- The first supported codec should be selected after benchmarking FP8 and INT8
  encode/decode latency on target GPUs.
- Cache eviction can start with deterministic FIFO/LRU but may need receiver
  miss feedback for long-running workloads.
- Model Runner V2 support should be planned separately after V1 behavior is
  validated.
