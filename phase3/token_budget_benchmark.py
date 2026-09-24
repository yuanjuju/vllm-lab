#!/usr/bin/env python3
"""One-run, streaming token-budget experiment against an existing vLLM server.

Standard library only. No server, tunnel, or cloud instance is started here.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path


METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)
SAMPLE_NAMES = METRICS[:4]
HISTOGRAMS = METRICS[5:8]


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo), 4)


def read_metrics(base_url):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
        raw = response.read().decode()
    series = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(\{[^}]*\})?\s+([^\s]+)", line)
        if match:
            name, labels, number = match.groups()
            if any(name == metric or name.startswith(metric + "_") for metric in METRICS):
                try:
                    series[name + (labels or "")] = float(number)
                except ValueError:
                    pass
    return series


def total(series, name, label=None):
    return sum(value for key, value in series.items()
               if (key == name or key.startswith(name + "{"))
               and (label is None or label in key))


def has_series(series, name):
    return any(key == name or key.startswith(name + "{") for key in series)


def histogram_delta(before, after, name):
    if not has_series(before, name + "_count") or not has_series(after, name + "_count"):
        return None
    count = total(after, name + "_count") - total(before, name + "_count")
    summed = total(after, name + "_sum") - total(before, name + "_sum")
    buckets = {}
    for key in set(before) | set(after):
        if not key.startswith(name + "_bucket{"):
            continue
        match = re.search(r'le="([^"]+)"', key)
        if match:
            boundary = float(match.group(1))
            buckets[boundary] = buckets.get(boundary, 0) + after.get(key, 0) - before.get(key, 0)

    def bucket_upper_bound(p):
        if count <= 0:
            return None
        for boundary, cumulative in sorted(buckets.items()):
            if cumulative >= count * p:
                return boundary
        return None

    return {"count": int(count), "sum_s": round(summed, 4),
            "mean_s": round(summed / count, 4) if count > 0 else None,
            "p50_bucket_upper_s": bucket_upper_bound(.5),
            "p95_bucket_upper_s": bucket_upper_bound(.95)}


def prompt_for(shape, run_id, index):
    # The changing first user token limits cross-run prefix-cache reuse.
    prefix = f"Experiment {run_id}, request {index}. "
    unit = ("A model-serving scheduler shares each iteration's token budget "
            "between input processing and ongoing output generation. ")
    repeats = 6 if shape == "short" else 105
    return prefix + unit * repeats + "List consecutive positive integers, comma-separated, without explanation."


def stream_request(base_url, model, request_id, shape, prompt, output_tokens, gate):
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
    gate.wait()
    start = time.perf_counter()
    events, usage, finish_reason, done = [], None, None, False
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        events.append(round(time.perf_counter() - start, 6))
        duration = time.perf_counter() - start
        ok = done and usage is not None and len(events) > 0 and finish_reason == "length"
        gaps = [round(b - a, 6) for a, b in zip(events, events[1:])]
        return {"request_id": request_id, "shape": shape, "ok": ok,
                "duration_s": round(duration, 4),
                "client_ttft_s": events[0] if events else None,
                "content_event_times_s": events, "content_event_gaps_s": gaps,
                "content_events": len(events), "usage": usage,
                "finish_reason": finish_reason, "done": done,
                "error": None if ok else "missing content/usage/DONE or unexpected finish_reason"}
    except Exception as exc:
        return {"request_id": request_id, "shape": shape, "ok": False,
                "duration_s": round(time.perf_counter() - start, 4),
                "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--budget", type=int, required=True,
                        help="Declared server --max-num-batched-tokens; verify in server log")
    parser.add_argument("--shape", choices=("short", "long", "mixed"), required=True)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--warmup", action="store_true", help="Mark this run as warmup in its result file")
    parser.add_argument("--dry-run", action="store_true", help="Preview locally; no network or output file")
    args = parser.parse_args()
    if args.requests < 1 or args.output_tokens < 2 or args.budget < 1 or args.poll_interval <= 0:
        parser.error("requests >= 1, output-tokens >= 2, budget >= 1, poll-interval > 0 required")
    if args.budget < args.requests:
        parser.error("budget must be at least request count for the intended max-num-seqs")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{time.time_ns()}"
    shapes = [("short" if i % 2 == 0 else "long") if args.shape == "mixed" else args.shape
              for i in range(args.requests)]
    prompts = [prompt_for(shape, run_id, i) for i, shape in enumerate(shapes)]
    prompt_meta = [{"shape": shape, "characters": len(prompt),
                    "sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                   for shape, prompt in zip(shapes, prompts)]
    if args.dry_run:
        print(json.dumps({"budget_label": args.budget, "shape": args.shape,
                          "requests": args.requests, "output_tokens": args.output_tokens,
                          "prompts": prompt_meta,
                          "note": "Characters are not tokens; verify prompt_tokens online."}, indent=2))
        return

    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)
    if not any(item["id"] == args.model for item in models["data"]):
        raise SystemExit(f"Model {args.model!r} is not registered")
    before = read_metrics(base_url)
    gate = threading.Barrier(args.requests + 1)
    samples = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as pool:
        futures = [pool.submit(stream_request, base_url, args.model, f"{run_id}-{i}",
                               shapes[i], prompts[i], args.output_tokens, gate)
                   for i in range(args.requests)]
        gate.wait()
        while not all(f.done() for f in futures):
            try:
                snapshot = read_metrics(base_url)
                samples.append({"at_s": round(time.perf_counter() - started, 4),
                                "running": total(snapshot, SAMPLE_NAMES[0]),
                                "waiting": total(snapshot, SAMPLE_NAMES[1]),
                                "capacity_waiting": total(snapshot, SAMPLE_NAMES[2], 'reason="capacity"'),
                                "kv_cache_usage": total(snapshot, SAMPLE_NAMES[3]),
                                "num_preemptions_total": total(snapshot, METRICS[4])})
            except Exception as exc:
                samples.append({"at_s": round(time.perf_counter() - started, 4),
                                "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(args.poll_interval)
        results = [f.result() for f in futures]
    try:
        after = read_metrics(base_url)
        after_error = None
    except Exception as exc:
        after = {}
        after_error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - started
    good = [r for r in results if r["ok"]]
    gaps = [gap for r in good for gap in r["content_event_gaps_s"]]
    valid_samples = [s for s in samples if "running" in s]
    prompt_tokens = [r["usage"]["prompt_tokens"] for r in good]
    output_total = sum(r["usage"]["completion_tokens"] for r in good)
    preemptions_available = has_series(before, METRICS[4]) and has_series(after, METRICS[4])
    summary = {
        "success_count": len(good), "request_count": len(results),
        "duration_s": round(duration, 4), "completion_tokens": output_total,
        "output_throughput_tok_s": round(output_total / duration, 3),
        "prompt_tokens_min_max": [min(prompt_tokens), max(prompt_tokens)] if prompt_tokens else None,
        "client_ttft_p50_s": percentile([r["client_ttft_s"] for r in good], .5),
        "client_ttft_p95_s": percentile([r["client_ttft_s"] for r in good], .95),
        "client_event_gap_p50_s": percentile(gaps, .5),
        "client_event_gap_p95_s": percentile(gaps, .95),
        "max_running": max((s["running"] for s in valid_samples), default=None),
        "max_waiting": max((s["waiting"] for s in valid_samples), default=None),
        "max_capacity_waiting": max((s["capacity_waiting"] for s in valid_samples), default=None),
        "max_kv_cache_usage": max((s["kv_cache_usage"] for s in valid_samples), default=None),
        "num_preemptions_delta": (total(after, METRICS[4]) - total(before, METRICS[4])
                                  if preemptions_available else None),
        "histogram_deltas": {name: histogram_delta(before, after, name) for name in HISTOGRAMS},
        "prefix_cache_hits_delta": total(after, METRICS[8]) - total(before, METRICS[8]),
        "prefix_cache_queries_delta": total(after, METRICS[9]) - total(before, METRICS[9]),
        "metric_poll_errors": sum("error" in sample for sample in samples),
        "metrics_after_error": after_error,
    }
    summary["by_shape"] = {}
    for shape in sorted(set(shapes)):
        selected = [r for r in good if r["shape"] == shape]
        shape_gaps = [gap for r in selected for gap in r["content_event_gaps_s"]]
        shape_prompts = [r["usage"]["prompt_tokens"] for r in selected]
        summary["by_shape"][shape] = {
            "success_count": len(selected), "request_count": shapes.count(shape),
            "prompt_tokens_min_max": [min(shape_prompts), max(shape_prompts)] if shape_prompts else None,
            "client_ttft_p50_s": percentile([r["client_ttft_s"] for r in selected], .5),
            "client_ttft_p95_s": percentile([r["client_ttft_s"] for r in selected], .95),
            "client_event_gap_p50_s": percentile(shape_gaps, .5),
            "client_event_gap_p95_s": percentile(shape_gaps, .95),
        }
    report = {"run_id": run_id, "timestamp": stamp, "warmup": args.warmup,
              "base_url": base_url,
              "model": args.model, "declared_max_num_batched_tokens": args.budget,
              "shape": args.shape, "requests": args.requests,
              "fixed_output_tokens": args.output_tokens, "poll_interval_s": args.poll_interval,
              "prompt_metadata": prompt_meta, "summary": summary,
              "results": results, "samples": samples,
              "metrics_before": before, "metrics_after": after}
    phase = "warmup" if args.warmup else "formal"
    output = Path(__file__).resolve().parent / "results" / f"token_budget_{run_id}_{args.budget}_{args.shape}_{phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "saved": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
