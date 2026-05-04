# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small end-to-end PP RefCache benchmark.

This is intentionally narrow: it exercises vLLM's real pipeline-parallel
intermediate tensor path with repeated prefill-heavy prompts, then prints a
single JSON result line that is easy to compare across baseline/refcache runs.
"""

import argparse
import json
import os
import statistics
import time
from typing import Any

import torch

from vllm import LLM, SamplingParams


def _make_shared_body(target_words: int) -> str:
    sentence = (
        "The meeting summary covers roadmap planning, release validation, "
        "customer feedback, performance tracking, deployment readiness, "
        "incident review, and follow-up ownership. "
    )
    words: list[str] = []
    sentence_words = sentence.split()
    while len(words) < target_words:
        words.extend(sentence_words)
    return " ".join(words[:target_words])


def _make_batch(batch_idx: int, batch_size: int, target_words: int) -> list[str]:
    shared_body = _make_shared_body(target_words)
    prompts = []
    for local_idx in range(batch_size):
        # Keep a small varying prefix so the workload is not fully identical,
        # while the large body remains reusable by the RefCache matcher.
        topic = (batch_idx + local_idx) % 8
        prompts.append(
            f"Request {batch_idx:03d}-{local_idx:02d}, topic {topic}. "
            f"Please continue the structured summary. {shared_body}"
        )
    return prompts


def _summarize_times(times: list[float]) -> dict[str, float]:
    if not times:
        return {}
    result = {
        "mean_s": statistics.fmean(times),
        "min_s": min(times),
        "max_s": max(times),
    }
    if len(times) > 1:
        result["stdev_s"] = statistics.stdev(times)
    else:
        result["stdev_s"] = 0.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--prompt-words", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        distributed_executor_backend="mp",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=True,
        seed=0,
        disable_log_stats=True,
        trust_remote_code=True,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    for warmup_idx in range(args.warmup_iters):
        llm.generate(
            _make_batch(warmup_idx, args.batch_size, args.prompt_words),
            sampling_params,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times: list[float] = []
    output_sample: list[dict[str, Any]] = []
    start_batch = args.warmup_iters
    total_input_words = 0
    total_outputs = 0
    for iter_idx in range(args.iters):
        batch_idx = start_batch + iter_idx
        prompts = _make_batch(batch_idx, args.batch_size, args.prompt_words)
        total_input_words += sum(len(prompt.split()) for prompt in prompts)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_outputs += len(outputs)

        if iter_idx == 0:
            for output in outputs[: min(2, len(outputs))]:
                completion = output.outputs[0]
                output_sample.append(
                    {
                        "prompt_prefix": output.prompt[:96],
                        "token_ids": list(completion.token_ids),
                        "text": completion.text,
                    }
                )

    total_time = sum(times)
    result = {
        "env_refcache_enabled": os.getenv("VLLM_PP_REFCACHE_ENABLE", "0"),
        "model": args.model,
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "prompt_words": args.prompt_words,
        "max_tokens": args.max_tokens,
        "timing": _summarize_times(times),
        "total_time_s": total_time,
        "batches_per_s": args.iters / total_time if total_time > 0 else 0.0,
        "requests_per_s": total_outputs / total_time if total_time > 0 else 0.0,
        "approx_input_words_per_s": (
            total_input_words / total_time if total_time > 0 else 0.0
        ),
        "output_sample": output_sample,
    }
    print("PP_REFCACHE_BENCH_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
