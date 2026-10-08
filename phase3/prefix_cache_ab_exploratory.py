#!/usr/bin/env python3
"""Exploratory prefix-cache A/B run on an already-started local vLLM service.

Run once per server configuration with the same --experiment-id. Standard library
only; it does not start or stop the server. Formal runs are never mixed with the
separate warmup run.
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
from datetime import datetime, timezone
from pathlib import Path


MODEL = "Qwen/Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRIC_NAMES = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
)
HISTOGRAMS = METRIC_NAMES[-3:]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo, hi = math.floor(at), math.ceil(at)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo), 4)


def read_metrics(base_url):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
        raw = response.read().decode()
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        key = fields[0]
        name = key.split("{", 1)[0]
        if any(name == metric or name.startswith(metric + "_") for metric in METRIC_NAMES):
            try:
                values[key] = float(fields[1])
            except ValueError:
                pass
    return values


def has_metric(values, name):
    return any(key == name or key.startswith(name + "{") for key in values)


def metric_total(values, name):
    if not has_metric(values, name):
        return None
    return sum(value for key, value in values.items()
               if key == name or key.startswith(name + "{"))


def metric_delta(before, after, name):
    first, last = metric_total(before, name), metric_total(after, name)
    return round(last - first, 6) if first is not None and last is not None else None


def histogram_delta(before, after, name):
    count = metric_delta(before, after, name + "_count")
    summed = metric_delta(before, after, name + "_sum")
    if count is None or summed is None:
        return None
    buckets = {}
    for key in set(before) | set(after):
        if not key.startswith(name + "_bucket{"):
            continue
        match = re.search(r'le="([^"]+)"', key)
        if match:
            bound = float(match.group(1))
            buckets[bound] = buckets.get(bound, 0) + after.get(key, 0) - before.get(key, 0)

    def upper(fraction):
        if count <= 0:
            return None
        return next((bound for bound, value in sorted(buckets.items())
                     if value >= count * fraction), None)

    return {"count": int(count), "sum_s": round(summed, 4),
            "mean_s": round(summed / count, 4) if count > 0 else None,
            "p50_bucket_upper_s": upper(0.5), "p95_bucket_upper_s": upper(0.95)}


def stream_request(base_url, prompt, request_id, output_tokens, gate):
    payload = {
        "model": MODEL,
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
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    gate.wait()
    start = time.perf_counter()
    content_times, usage, finish, done = [], None, None, False
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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
                        finish = choice["finish_reason"]
                    if (choice.get("delta") or {}).get("content"):
                        content_times.append(round(time.perf_counter() - start, 6))
        elapsed = time.perf_counter() - start
        ok = done and usage is not None and bool(content_times) and finish == "length"
        return {"request_id": request_id, "ok": ok,
                "duration_s": round(elapsed, 4),
                "client_ttft_s": content_times[0] if content_times else None,
                "content_event_times_s": content_times,
                "usage": usage, "finish_reason": finish, "done": done,
                "error": None if ok else "missing usage/content/DONE or unexpected finish"}
    except Exception as exc:
        return {"request_id": request_id, "ok": False,
                "duration_s": round(time.perf_counter() - start, 4),
                "error": f"{type(exc).__name__}: {exc}"}


def prompt_set(experiment_id, label, requests):
    paragraph = (
        "vLLM使用调度器管理正在生成的请求，并将注意力的键和值放入KV缓存。"
        "共享前缀可能复用已经完成的块，从而减少后续请求的输入计算。"
        "这段文字仅用于构造稳定的实验输入，不要求模型解释它。"
    )
    marker = f"PrefixAB {experiment_id} {label}。"
    prefix = marker + paragraph * 24
    prime = prefix + "请连续列出自然数，用逗号隔开，不要解释。"
    probes = [prefix + f"并发请求编号{i}。请连续列出自然数，用逗号隔开，不要解释。"
              for i in range(requests)]
    changed = f"ChangedPrefixAB {experiment_id} {label}。" + paragraph * 24 + (
        "请连续列出自然数，用逗号隔开，不要解释。")
    all_prompts = [prime, *probes, changed]
    metadata = [{"characters": len(prompt),
                 "sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                for prompt in all_prompts]
    return prime, probes, changed, metadata


def run_repetition(base_url, experiment_id, label, requests, output_tokens, poll_interval):
    prime_prompt, probe_prompts, changed_prompt, prompt_meta = prompt_set(
        experiment_id, label, requests)
    prime = stream_request(base_url, prime_prompt, f"{label}-prime", 16,
                           threading.Barrier(1))
    if not prime["ok"]:
        raise RuntimeError(f"Prime request failed: {prime}")
    before = read_metrics(base_url)
    gate = threading.Barrier(requests + 1)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        futures = [pool.submit(stream_request, base_url, prompt, f"{label}-{i}",
                               output_tokens, gate)
                   for i, prompt in enumerate(probe_prompts)]
        gate.wait()
        started = time.perf_counter()
        while not all(future.done() for future in futures):
            try:
                snapshot = read_metrics(base_url)
                samples.append({"at_s": round(time.perf_counter() - started, 4),
                                "running": metric_total(snapshot, METRIC_NAMES[3]),
                                "waiting": metric_total(snapshot, METRIC_NAMES[4]),
                                "kv_usage": metric_total(snapshot, METRIC_NAMES[5])})
            except Exception as exc:
                samples.append({"error": f"{type(exc).__name__}: {exc}"})
            time.sleep(poll_interval)
        probes = [future.result() for future in futures]
        duration = time.perf_counter() - started
    after = read_metrics(base_url)
    changed = stream_request(base_url, changed_prompt, f"{label}-changed", 16,
                             threading.Barrier(1))
    changed_after = read_metrics(base_url)
    good = [result for result in probes if result["ok"]]
    output_total = sum(result["usage"]["completion_tokens"] for result in good)
    hits = metric_delta(before, after, METRIC_NAMES[1])
    queries = metric_delta(before, after, METRIC_NAMES[0])
    summary = {
        "success_count": len(good), "request_count": requests,
        "burst_duration_s": round(duration, 4),
        "output_tokens": output_total,
        "output_throughput_tok_s": round(output_total / duration, 3),
        "client_ttft_p50_s": percentile([result["client_ttft_s"] for result in good], .5),
        "client_ttft_p95_s": percentile([result["client_ttft_s"] for result in good], .95),
        "prompt_tokens_min_max": [min(result["usage"]["prompt_tokens"] for result in good),
                                  max(result["usage"]["prompt_tokens"] for result in good)]
        if good else None,
        "prefix_queries_delta": queries, "prefix_hits_delta": hits,
        "prefix_hit_ratio": round(hits / queries, 4)
        if hits is not None and queries is not None and queries > 0 else None,
        "preemptions_delta": metric_delta(before, after, METRIC_NAMES[2]),
        "max_running": max((sample["running"] for sample in samples
                            if sample.get("running") is not None), default=None),
        "max_waiting": max((sample["waiting"] for sample in samples
                            if sample.get("waiting") is not None), default=None),
        "max_kv_usage": max((sample["kv_usage"] for sample in samples
                             if sample.get("kv_usage") is not None), default=None),
        "metric_poll_errors": sum("error" in sample for sample in samples),
        "histogram_deltas": {name: histogram_delta(before, after, name)
                             for name in HISTOGRAMS},
        "changed_prefix_queries_delta": metric_delta(after, changed_after, METRIC_NAMES[0]),
        "changed_prefix_hits_delta": metric_delta(after, changed_after, METRIC_NAMES[1]),
    }
    return {"label": label, "prompt_metadata": prompt_meta,
            "prime": prime, "probes": probes, "changed_control": changed,
            "metrics_before": before, "metrics_after": after,
            "metrics_after_changed_control": changed_after,
            "samples": samples, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--condition", choices=("on", "off"), required=True)
    parser.add_argument("--experiment-id", required=True,
                        help="Pass exactly the same ID to the on and off runs")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--formal-repetitions", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.experiment_id):
        parser.error("experiment-id must be 1-64 ASCII letters, digits, _ or -")
    if args.requests < 1 or args.output_tokens < 2 or args.formal_repetitions < 3:
        parser.error("requests >= 1, output-tokens >= 2, formal-repetitions >= 3 required")
    if args.poll_interval <= 0:
        parser.error("poll-interval must be positive")
    plan = ["warmup", *[f"formal-{i}" for i in range(args.formal_repetitions)]]
    if args.dry_run:
        print(json.dumps({"experiment_id": args.experiment_id, "condition": args.condition,
                          "plan": plan, "requests": args.requests,
                          "fixed_output_tokens": args.output_tokens,
                          "first_prompt_metadata": prompt_set(args.experiment_id, plan[1],
                                                              args.requests)[3]},
                         ensure_ascii=False, indent=2))
        return
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in models):
        raise SystemExit(f"{MODEL} is not registered")
    runs = []
    for label in plan:
        run = run_repetition(base_url, args.experiment_id, label, args.requests,
                             args.output_tokens, args.poll_interval)
        runs.append(run)
        print(json.dumps({"label": label, "summary": run["summary"]}, ensure_ascii=False),
              flush=True)
    formal = [run["summary"] for run in runs[1:]]
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id, "condition": args.condition,
        "model": MODEL, "base_url": base_url, "transport": "cloud-local loopback",
        "requests_per_burst": args.requests, "fixed_probe_output_tokens": args.output_tokens,
        "formal_repetitions": args.formal_repetitions,
        "warmup_excluded_from_formal": True,
        "client_event_gap_note": "SSE content events are not individual tokens; use server ITL histogram.",
        "formal_medians": {
            key: percentile([run[key] for run in formal if run[key] is not None], .5)
            for key in ("client_ttft_p50_s", "client_ttft_p95_s",
                        "output_throughput_tok_s", "prefix_hit_ratio", "max_waiting")
        },
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"prefix_cache_ab_{args.experiment_id}_{args.condition}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
