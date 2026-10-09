#!/usr/bin/env python3
"""Cloud-local, single-variable BF16/auto vs FP8 KV cache comparison.

Requires prefix_cache_ab_exploratory.py in the same directory for shared
standard-library metrics helpers. Start one matching service condition first.
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
    HISTOGRAMS,
    METRIC_NAMES,
    histogram_delta,
    metric_delta,
    metric_total,
    percentile,
    read_metrics,
)


MODEL = "Qwen/Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FILLER = (
    "This sentence is ordinary context for a controlled KV cache capacity "
    "measurement and carries no instruction.\n"
)


def long_prompt(experiment_id, label, index, repeats, quality=False):
    marker = hashlib.sha256(
        f"{experiment_id}/{label}/{index}".encode()
    ).hexdigest()[:8].upper()
    task = ("请只输出开头的八位验证码，不要解释。" if quality else
            "请从 1 开始连续列出自然数，用逗号隔开，不要解释。")
    prompt = f"八位验证码：{marker}\n" + FILLER * repeats + task
    return prompt, marker


def request(base_url, prompt, request_id, output_tokens=None, gate=None):
    fixed = output_tokens is not None
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": output_tokens if fixed else 32,
        "stream": fixed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if fixed:
        payload["min_tokens"] = output_tokens
        payload["stream_options"] = {"include_usage": True}
    http_request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    if gate is not None:
        gate.wait()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=180) as response:
            if not fixed:
                body = json.load(response)
                choice = body["choices"][0]
                return {"request_id": request_id, "ok": True,
                        "duration_s": round(time.perf_counter() - start, 4),
                        "content": choice["message"].get("content") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "usage": body.get("usage")}
            event_times, usage, finish, done = [], None, None, False
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
                        event_times.append(round(time.perf_counter() - start, 6))
            ok = (done and usage is not None and bool(event_times)
                  and finish == "length"
                  and usage.get("completion_tokens") == output_tokens)
            return {"request_id": request_id, "ok": ok,
                    "duration_s": round(time.perf_counter() - start, 4),
                    "client_ttft_s": event_times[0] if event_times else None,
                    "content_event_times_s": event_times,
                    "usage": usage, "finish_reason": finish, "done": done,
                    "error": None if ok else "incomplete stream or wrong output length"}
    except Exception as exc:
        return {"request_id": request_id, "ok": False,
                "duration_s": round(time.perf_counter() - start, 4),
                "error": f"{type(exc).__name__}: {exc}"}


def quality_suite(base_url, experiment_id, repeats):
    cases = [
        ("math", "请只输出 15+27 的阿拉伯数字结果。", "42"),
        ("capital", "请只输出中国首都的中文城市名。", "北京"),
        ("translation", "请只输出英文单词 apple 的中文译词。", "苹果"),
        ("sequence", "请只输出数列 2,4,6,8 的下一项阿拉伯数字。", "10"),
    ]
    for index in range(2):
        prompt, marker = long_prompt(experiment_id, "quality", index,
                                     repeats, quality=True)
        cases.append((f"long-recall-{index}", prompt, marker))
    outputs = []
    for label, prompt, expected in cases:
        result = request(base_url, prompt, label)
        answer = result.get("content", "").strip().strip("。 .\n")
        result.update({"label": label, "expected": expected,
                       "exact_match": result["ok"] and answer == expected,
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        outputs.append(result)
    return outputs


def run_burst(base_url, experiment_id, label, requests, repeats,
              output_tokens, poll_interval):
    prompts = [long_prompt(experiment_id, label, i, repeats)[0]
               for i in range(requests)]
    prompt_hashes = [hashlib.sha256(prompt.encode()).hexdigest()
                     for prompt in prompts]
    before = read_metrics(base_url)
    gate = threading.Barrier(requests + 1)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        futures = [pool.submit(request, base_url, prompt, f"{label}-{i}",
                               output_tokens, gate)
                   for i, prompt in enumerate(prompts)]
        gate.wait()
        start = time.perf_counter()
        while not all(future.done() for future in futures):
            try:
                snapshot = read_metrics(base_url)
                samples.append({
                    "at_s": round(time.perf_counter() - start, 4),
                    "running": metric_total(snapshot, METRIC_NAMES[3]),
                    "waiting": metric_total(snapshot, METRIC_NAMES[4]),
                    "kv_usage": metric_total(snapshot, METRIC_NAMES[5]),
                })
            except Exception as exc:
                samples.append({"error": f"{type(exc).__name__}: {exc}"})
            time.sleep(poll_interval)
        outputs = [future.result() for future in futures]
        duration = time.perf_counter() - start
    after = read_metrics(base_url)
    good = [item for item in outputs if item["ok"]]
    output_total = sum(item["usage"]["completion_tokens"] for item in good)
    summary = {
        "success_count": len(good), "request_count": requests,
        "burst_duration_s": round(duration, 4),
        "output_tokens": output_total,
        "output_throughput_tok_s": round(output_total / duration, 3),
        "prompt_tokens_min_max": [
            min(item["usage"]["prompt_tokens"] for item in good),
            max(item["usage"]["prompt_tokens"] for item in good),
        ] if good else None,
        "client_ttft_p50_s": percentile(
            [item["client_ttft_s"] for item in good], .5),
        "client_ttft_p95_s": percentile(
            [item["client_ttft_s"] for item in good], .95),
        "preemptions_delta": metric_delta(before, after, METRIC_NAMES[2]),
        "max_running": max((x["running"] for x in samples
                            if x.get("running") is not None), default=None),
        "max_waiting": max((x["waiting"] for x in samples
                            if x.get("waiting") is not None), default=None),
        "max_kv_usage": max((x["kv_usage"] for x in samples
                             if x.get("kv_usage") is not None), default=None),
        "metric_poll_errors": sum("error" in x for x in samples),
        "histogram_deltas": {name: histogram_delta(before, after, name)
                             for name in HISTOGRAMS},
    }
    return {"label": label, "prompt_sha256": prompt_hashes,
            "requests": outputs, "samples": samples,
            "metrics_before": before, "metrics_after": after,
            "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("auto", "fp8"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--filler-repeats", type=int, default=180)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--formal-repetitions", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--server-profile", default="r570-vllm018-kv4g-eager-prefix-off-calculate-scales-on")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.experiment_id):
        parser.error("experiment-id must be 1-64 ASCII letters, digits, _ or -")
    if (args.requests < 1 or args.filler_repeats < 1 or args.output_tokens < 2
            or args.formal_repetitions < 3 or args.poll_interval <= 0):
        parser.error("positive workload fields and >=3 formal repetitions required")
    labels = ["warmup", *[f"formal-{i}" for i in range(args.formal_repetitions)]]
    if args.dry_run:
        prompt, _ = long_prompt(args.experiment_id, labels[1], 0,
                                args.filler_repeats)
        print(json.dumps({"condition": args.condition, "labels": labels,
                          "requests": args.requests,
                          "prompt_chars": len(prompt),
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "output_tokens": args.output_tokens}, indent=2))
        return
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in models):
        raise SystemExit(f"{MODEL} is not registered")
    quality = quality_suite(base_url, args.experiment_id, args.filler_repeats)
    runs = []
    for label in labels:
        run = run_burst(base_url, args.experiment_id, label, args.requests,
                        args.filler_repeats, args.output_tokens,
                        args.poll_interval)
        runs.append(run)
        print(json.dumps({"label": label, "summary": run["summary"]},
                         ensure_ascii=False), flush=True)
        if run["summary"]["success_count"] != args.requests:
            break
    formal = [run["summary"] for run in runs if run["label"].startswith("formal-")]
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id, "condition": args.condition,
        "model": MODEL, "base_url": base_url,
        "transport": "cloud-local loopback",
        "server_profile": args.server_profile,
        "requests_per_burst": args.requests,
        "filler_repeats": args.filler_repeats,
        "fixed_output_tokens": args.output_tokens,
        "formal_repetitions_planned": args.formal_repetitions,
        "warmup_excluded_from_formal": True,
        "client_event_gap_note": "SSE content events are not individual tokens; server histogram is used for ITL.",
        "quality_cases": quality, "runs": runs,
        "formal_medians": {
            key: percentile([row[key] for row in formal if row[key] is not None], .5)
            for key in ("client_ttft_p50_s", "client_ttft_p95_s",
                        "output_throughput_tok_s", "max_kv_usage")
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"kv_dtype_ab_{args.experiment_id}_{args.condition}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
