#!/usr/bin/env python3
"""Closed-loop/open-loop capacity experiment for an existing vLLM server.

The script uses only the Python standard library. It does not start a server,
an SSH tunnel, or a cloud instance.
"""

import argparse
import concurrent.futures
import json
import math
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from token_budget_benchmark import METRICS, histogram_delta, read_metrics, total


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return round(value, 4)


def make_prompt(run_id, index):
    unit = ("A serving benchmark must separate offered load, completed throughput, "
            "latency, and goodput under an explicit latency objective. ")
    return (f"Load experiment {run_id}, request {index}. " + unit * 6
            + "List consecutive positive integers, comma-separated, without explanation.")


def stream_request(base_url, model, run_id, index, output_tokens, timeout,
                   scheduled_at=None):
    prompt = make_prompt(run_id, index)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "min_tokens": output_tokens,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    content_times = []
    usage = None
    finish_reason = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line == "data: [DONE]":
                    done = True
                    break
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    if (choice.get("delta") or {}).get("content"):
                        content_times.append(time.perf_counter() - started)
        ended = time.perf_counter()
        ok = bool(done and usage and content_times and finish_reason == "length")
        return {
            "index": index,
            "ok": ok,
            "scheduled_at_s": scheduled_at,
            "client_release_lag_s": (round(started - scheduled_at, 6)
                                     if scheduled_at is not None else None),
            "started_at_s": started,
            "duration_s": round(ended - started, 4),
            "client_ttft_s": round(content_times[0], 6) if content_times else None,
            "max_content_gap_s": (round(max(b - a for a, b in zip(content_times, content_times[1:])), 6)
                                  if len(content_times) > 1 else None),
            "usage": usage,
            "finish_reason": finish_reason,
            "done": done,
            "error": None if ok else "missing content/usage/DONE or unexpected finish_reason",
        }
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "scheduled_at_s": scheduled_at,
            "client_release_lag_s": (round(started - scheduled_at, 6)
                                     if scheduled_at is not None else None),
            "started_at_s": started,
            "duration_s": round(time.perf_counter() - started, 4),
            "client_ttft_s": None,
            "usage": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def metrics_poller(base_url, origin, stopped, samples, interval):
    while not stopped.is_set():
        try:
            current = read_metrics(base_url)
            samples.append({
                "at_s": round(time.perf_counter() - origin, 4),
                "running": total(current, "vllm:num_requests_running"),
                "waiting": total(current, "vllm:num_requests_waiting"),
                "capacity_waiting": total(
                    current, "vllm:num_requests_waiting_by_reason", 'reason="capacity"'),
                "kv_cache_usage": total(current, "vllm:kv_cache_usage_perc"),
                "num_preemptions_total": total(current, "vllm:num_preemptions_total"),
            })
        except Exception as exc:
            samples.append({"at_s": round(time.perf_counter() - origin, 4),
                            "error": f"{type(exc).__name__}: {exc}"})
        stopped.wait(interval)


def run_closed_loop(args, run_id):
    next_index = 0
    lock = threading.Lock()

    def worker():
        nonlocal next_index
        completed = []
        while True:
            with lock:
                if next_index >= args.requests:
                    return completed
                index = next_index
                next_index += 1
            completed.append(stream_request(args.base_url, args.model, run_id, index,
                                            args.output_tokens, args.timeout))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        nested = [future.result() for future in
                  [pool.submit(worker) for _ in range(args.concurrency)]]
    return [item for group in nested for item in group]


def run_open_loop(args, run_id, origin):
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as pool:
        for index in range(args.requests):
            scheduled = origin + index / args.rate
            remaining = scheduled - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            futures.append(pool.submit(stream_request, args.base_url, args.model,
                                       run_id, index, args.output_tokens,
                                       args.timeout, scheduled))
        return [future.result() for future in futures]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=("closed", "open"), required=True)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--rate", type=float, help="Open-loop offered requests/second")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--slo-ttft", type=float, default=0.5)
    parser.add_argument("--slo-e2e", type=float, default=5.0)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.mode == "closed" and (args.concurrency is None or args.rate is not None):
        parser.error("closed mode requires --concurrency and no --rate")
    if args.mode == "open" and (args.rate is None or args.concurrency is not None):
        parser.error("open mode requires --rate and no --concurrency")
    if min(args.requests, args.output_tokens) < 1 or min(
            args.timeout, args.poll_interval, args.slo_ttft, args.slo_e2e) <= 0:
        parser.error("counts and time values must be positive")
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("concurrency must be positive")
    if args.rate is not None and args.rate <= 0:
        parser.error("rate must be positive")

    args.base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=10) as response:
        models = json.load(response)
    if not any(item["id"] == args.model for item in models["data"]):
        raise SystemExit(f"Model {args.model!r} is not registered")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{time.time_ns()}"
    before = read_metrics(args.base_url)
    origin = time.perf_counter()
    samples = []
    stopped = threading.Event()
    poller = threading.Thread(target=metrics_poller,
                              args=(args.base_url, origin, stopped, samples,
                                    args.poll_interval), daemon=True)
    poller.start()
    try:
        if args.mode == "closed":
            results = run_closed_loop(args, run_id)
        else:
            results = run_open_loop(args, run_id, origin)
    finally:
        stopped.set()
        poller.join(timeout=2)
    duration = time.perf_counter() - origin
    after = read_metrics(args.base_url)

    good = [result for result in results if result["ok"]]
    slo_good = [result for result in good
                if result["client_ttft_s"] <= args.slo_ttft
                and result["duration_s"] <= args.slo_e2e]
    completion_tokens = sum(result["usage"]["completion_tokens"] for result in good)
    valid_samples = [sample for sample in samples if "running" in sample]
    preemption_available = any(k == "vllm:num_preemptions_total" or
                               k.startswith("vllm:num_preemptions_total{") for k in before)
    preemption_available &= any(k == "vllm:num_preemptions_total" or
                                k.startswith("vllm:num_preemptions_total{") for k in after)
    summary = {
        "success_count": len(good),
        "request_count": len(results),
        "slo_good_count": len(slo_good),
        "slo_goodput_req_s": round(len(slo_good) / duration, 4),
        "slo_attainment_percent": round(100 * len(slo_good) / len(results), 2),
        "duration_s": round(duration, 4),
        "achieved_request_throughput_req_s": round(len(good) / duration, 4),
        "output_throughput_tok_s": round(completion_tokens / duration, 3),
        "client_ttft_p50_s": percentile([item["client_ttft_s"] for item in good], .5),
        "client_ttft_p95_s": percentile([item["client_ttft_s"] for item in good], .95),
        "client_e2e_p50_s": percentile([item["duration_s"] for item in good], .5),
        "client_e2e_p95_s": percentile([item["duration_s"] for item in good], .95),
        "client_release_lag_p95_s": percentile(
            [item["client_release_lag_s"] for item in results
             if item.get("client_release_lag_s") is not None], .95),
        "max_running": max((sample["running"] for sample in valid_samples), default=None),
        "max_waiting": max((sample["waiting"] for sample in valid_samples), default=None),
        "max_capacity_waiting": max((sample["capacity_waiting"] for sample in valid_samples), default=None),
        "max_kv_cache_usage": max((sample["kv_cache_usage"] for sample in valid_samples), default=None),
        "num_preemptions_delta": (total(after, "vllm:num_preemptions_total")
                                  - total(before, "vllm:num_preemptions_total")
                                  if preemption_available else None),
        "histogram_deltas": {
            name: histogram_delta(before, after, name)
            for name in ("vllm:request_queue_time_seconds",
                         "vllm:time_to_first_token_seconds",
                         "vllm:inter_token_latency_seconds")
        },
        "metric_poll_errors": sum("error" in sample for sample in samples),
    }
    report = {
        "run_id": run_id,
        "timestamp": stamp,
        "warmup": args.warmup,
        "mode": args.mode,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "offered_rate_req_s": args.rate,
        "requests": args.requests,
        "fixed_output_tokens": args.output_tokens,
        "slo": {"client_ttft_s_lte": args.slo_ttft,
                "client_e2e_s_lte": args.slo_e2e},
        "summary": summary,
        "results": sorted(results, key=lambda item: item["index"]),
        "samples": samples,
        "metrics_before": before,
        "metrics_after": after,
    }
    load = (f"c{args.concurrency}" if args.mode == "closed"
            else f"r{str(args.rate).replace('.', 'p')}")
    phase = "warmup" if args.warmup else "formal"
    output = (Path(__file__).resolve().parent / "results" /
              f"load_curve_{run_id}_{args.mode}_{load}_{phase}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "saved": str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
