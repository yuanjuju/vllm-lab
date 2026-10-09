#!/usr/bin/env python3
"""Cloud-local BF16 target vs N-gram speculative decoding comparison.

Start one matching server variant first. This script only sends requests and
records evidence; it never starts/stops a service or changes Python packages.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prefix_cache_ab_exploratory import (
    histogram_delta,
    metric_delta,
    metric_total,
    percentile,
)


MODEL = "Qwen/Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
POSITION_COUNTER = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
HISTOGRAMS = (
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
)
METRICS = (*SPEC_COUNTERS, POSITION_COUNTER, *GAUGES,
           "vllm:num_preemptions_total", *HISTOGRAMS)
SHAPES = ("repeat", "count")
CONCURRENCIES = (1, 8)


def prompt_for(experiment_id, shape, concurrency, label, index):
    marker = hashlib.sha256(
        f"{experiment_id}/{shape}/{concurrency}/{label}/{index}".encode()
    ).hexdigest()[:8].upper()
    if shape == "repeat":
        phrase = "蓝色风车缓缓转动，白色小船轻轻靠岸。"
        prompt = (
            f"实验编号 {marker}。请不加说明地从头逐字抄写下面正文，"
            "持续抄写直到输出长度限制。正文：\n" + phrase * 40
        )
    else:
        prompt = (
            f"实验编号 {marker}。请从 1 开始按升序连续输出自然数，"
            "只用英文逗号分隔，如 1,2,3,4。不要省略，不要解释，"
            "持续输出直到长度限制。"
        )
    return prompt


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
        if any(name == metric or name.startswith(metric + "_")
               for metric in METRICS):
            try:
                values[key] = float(fields[1])
            except ValueError:
                pass
    return values


def request(base_url, prompt, request_id, output_tokens, gate):
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
    http_request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    gate.wait()
    start = time.perf_counter()
    try:
        content_parts, content_times = [], []
        usage, finish, done = None, None, False
        with urllib.request.urlopen(http_request, timeout=180) as response:
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
                        finish = choice["finish_reason"]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        content_parts.append(content)
                        content_times.append(round(time.perf_counter() - start, 6))
        duration = time.perf_counter() - start
        content = "".join(content_parts)
        ok = (done and usage is not None and bool(content_times)
              and finish == "length"
              and usage.get("completion_tokens") == output_tokens)
        ttft = content_times[0] if content_times else None
        return {
            "request_id": request_id, "ok": ok,
            "duration_s": round(duration, 4), "client_ttft_s": ttft,
            "client_decode_tail_per_token_s": (
                round((duration - ttft) / (output_tokens - 1), 6)
                if ttft is not None else None
            ),
            "content_event_times_s": content_times,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "content": content, "usage": usage,
            "finish_reason": finish, "done": done,
            "error": None if ok else "incomplete stream or wrong output length",
        }
    except Exception as exc:
        return {"request_id": request_id, "ok": False,
                "duration_s": round(time.perf_counter() - start, 4),
                "error": f"{type(exc).__name__}: {exc}"}


def position_deltas(before, after):
    positions = {}
    for key, value in after.items():
        if not key.startswith(POSITION_COUNTER + "{"):
            continue
        match = re.search(r'position="([^"]+)"', key)
        if match:
            position = match.group(1)
            positions[position] = positions.get(position, 0) + value - before.get(key, 0)
    return dict(sorted(positions.items(), key=lambda item: int(item[0])))


def run_burst(base_url, experiment_id, shape, concurrency, label,
              output_tokens, poll_interval):
    prompts = [prompt_for(experiment_id, shape, concurrency, label, i)
               for i in range(concurrency)]
    hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    before = read_metrics(base_url)
    gate = threading.Barrier(concurrency + 1)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request, base_url, prompt,
                               f"{shape}-c{concurrency}-{label}-{i}",
                               output_tokens, gate)
                   for i, prompt in enumerate(prompts)]
        gate.wait()
        start = time.perf_counter()
        while not all(future.done() for future in futures):
            try:
                metrics = read_metrics(base_url)
                samples.append({
                    "at_s": round(time.perf_counter() - start, 4),
                    "running": metric_total(metrics, GAUGES[0]),
                    "waiting": metric_total(metrics, GAUGES[1]),
                    "kv_usage": metric_total(metrics, GAUGES[2]),
                })
            except Exception as exc:
                samples.append({"error": f"{type(exc).__name__}: {exc}"})
            time.sleep(poll_interval)
        outputs = [future.result() for future in futures]
        duration = time.perf_counter() - start
    after = read_metrics(base_url)
    good = [item for item in outputs if item["ok"]]
    output_total = sum(item["usage"]["completion_tokens"] for item in good)
    drafts = metric_delta(before, after, SPEC_COUNTERS[0])
    drafted = metric_delta(before, after, SPEC_COUNTERS[1])
    accepted = metric_delta(before, after, SPEC_COUNTERS[2])
    summary = {
        "success_count": len(good), "request_count": concurrency,
        "burst_duration_s": round(duration, 4),
        "output_tokens": output_total,
        "output_throughput_tok_s": round(output_total / duration, 3),
        "prompt_tokens_min_max": ([
            min(item["usage"]["prompt_tokens"] for item in good),
            max(item["usage"]["prompt_tokens"] for item in good),
        ] if good else None),
        "client_ttft_p50_s": percentile([x["client_ttft_s"] for x in good], .5),
        "client_ttft_p95_s": percentile([x["client_ttft_s"] for x in good], .95),
        "client_e2e_p50_s": percentile([x["duration_s"] for x in good], .5),
        "client_e2e_p95_s": percentile([x["duration_s"] for x in good], .95),
        "client_decode_tail_per_token_p50_s": percentile(
            [x["client_decode_tail_per_token_s"] for x in good], .5),
        "max_running": max((x["running"] for x in samples
                            if x.get("running") is not None), default=None),
        "max_waiting": max((x["waiting"] for x in samples
                            if x.get("waiting") is not None), default=None),
        "max_kv_usage": max((x["kv_usage"] for x in samples
                             if x.get("kv_usage") is not None), default=None),
        "preemptions_delta": metric_delta(
            before, after, "vllm:num_preemptions_total"),
        "spec_drafts_delta": drafts,
        "spec_draft_tokens_delta": drafted,
        "spec_accepted_tokens_delta": accepted,
        "spec_draft_acceptance_rate": (
            round(accepted / drafted, 4) if drafted else None),
        "spec_mean_acceptance_length": (
            round(1 + accepted / drafts, 4) if drafts else None),
        "spec_accepted_per_position": position_deltas(before, after),
        "metric_poll_errors": sum("error" in x for x in samples),
        "histogram_deltas": {name: histogram_delta(before, after, name)
                             for name in HISTOGRAMS},
    }
    return {"shape": shape, "concurrency": concurrency, "label": label,
            "prompt_sha256": hashes, "requests": outputs,
            "samples": samples, "metrics_before": before,
            "metrics_after": after, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("baseline", "ngram"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--formal-repetitions", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--server-profile", default="r570-vllm018-qwen3-8b-bf16-eager-prefix-off")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.experiment_id):
        parser.error("experiment-id must be 1-64 ASCII letters, digits, _ or -")
    if args.output_tokens < 2 or args.formal_repetitions < 3 or args.poll_interval <= 0:
        parser.error("Need >=2 output tokens, >=3 formal repetitions, positive polling")
    labels = ["warmup", *[f"formal-{i}" for i in range(args.formal_repetitions)]]
    if args.dry_run:
        print(json.dumps({
            "condition": args.condition, "labels": labels,
            "shapes": SHAPES, "concurrencies": CONCURRENCIES,
            "prompt_examples": {shape: {
                "characters": len(prompt_for(args.experiment_id, shape, 1,
                                              "formal-0", 0)),
                "sha256": hashlib.sha256(prompt_for(
                    args.experiment_id, shape, 1, "formal-0", 0).encode()).hexdigest(),
            } for shape in SHAPES},
        }, ensure_ascii=False, indent=2))
        return
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in models):
        raise SystemExit(f"{MODEL} not registered")
    runs = []
    for shape in SHAPES:
        for concurrency in CONCURRENCIES:
            for label in labels:
                run = run_burst(base_url, args.experiment_id, shape, concurrency,
                                label, args.output_tokens, args.poll_interval)
                runs.append(run)
                print(json.dumps({"shape": shape, "concurrency": concurrency,
                                  "label": label, "summary": run["summary"]},
                                 ensure_ascii=False), flush=True)
                if run["summary"]["success_count"] != concurrency:
                    break
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "condition": args.condition,
        "model": MODEL, "base_url": base_url,
        "transport": "cloud-local loopback",
        "server_profile": args.server_profile,
        "speculative_config_expected": ({
            "method": "ngram", "num_speculative_tokens": 4,
            "prompt_lookup_min": 1, "prompt_lookup_max": 4,
        } if args.condition == "ngram" else None),
        "shapes": SHAPES, "concurrencies": CONCURRENCIES,
        "fixed_output_tokens": args.output_tokens,
        "formal_repetitions_planned": args.formal_repetitions,
        "warmup_excluded_from_formal": True,
        "client_event_gap_note": (
            "SSE chunks may contain multiple accepted tokens under speculation; "
            "content-event gaps are not token ITL."
        ),
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"spec_ngram_ab_{args.experiment_id}_{args.condition}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
