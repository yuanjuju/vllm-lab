#!/usr/bin/env python3
"""Observe vLLM's running/waiting queues during a small burst of requests."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path


METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "capacity_waiting": "vllm:num_requests_waiting_by_reason",
    "kv": "vllm:kv_cache_usage_perc",
}


def get_metrics(base_url):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as response:
        body = response.read().decode()
    values = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        for name, metric in METRICS.items():
            if line.startswith(metric + "{"):
                if name == "waiting" and "_by_reason" in line:
                    continue
                if name == "capacity_waiting" and 'reason="capacity"' not in line:
                    continue
                values[name] = float(line.rsplit(" ", 1)[1])
    return values


def send_request(base_url, request_id, tokens, gate):
    gate.wait()
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [{
            "role": "user",
            "content": (
                f"独立调度实验编号 {request_id}。"
                "从1开始连续列出自然数，数字之间用逗号分隔，持续列出，不要解释。"
            ),
        }],
        "temperature": 0,
        "min_tokens": tokens,
        "max_tokens": tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        return {
            "request_id": request_id,
            "ok": True,
            "seconds": round(time.perf_counter() - start, 3),
            "completion_tokens": result["usage"]["completion_tokens"],
            "finish_reason": result["choices"][0]["finish_reason"],
        }
    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "seconds": round(time.perf_counter() - start, 3),
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
        models = json.load(response)
    assert any(item["id"] == "Qwen/Qwen3-8B" for item in models["data"])

    initial = get_metrics(base_url)
    gate = threading.Barrier(args.requests + 1)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{time.time_ns()}"
    samples = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as pool:
        futures = [
            pool.submit(send_request, base_url, f"{run_id}-{i}", args.tokens, gate)
            for i in range(args.requests)
        ]
        gate.wait()
        while not all(future.done() for future in futures):
            try:
                values = get_metrics(base_url)
                samples.append({"at_s": round(time.perf_counter() - started, 3), **values})
            except Exception as exc:
                samples.append({"at_s": round(time.perf_counter() - started, 3), "error": str(exc)})
            time.sleep(args.poll_interval)
        results = [future.result() for future in futures]
    samples.append({"at_s": round(time.perf_counter() - started, 3), **get_metrics(base_url)})

    valid = [s for s in samples if "running" in s]
    summary = {
        "run_id": run_id,
        "url": base_url,
        "request_count": args.requests,
        "tokens_per_request": args.tokens,
        "initial_metrics": initial,
        "success_count": sum(bool(r["ok"]) for r in results),
        "max_running": max((s["running"] for s in valid), default=None),
        "max_waiting": max((s["waiting"] for s in valid), default=None),
        "max_capacity_waiting": max((s.get("capacity_waiting", 0) for s in valid), default=None),
        "max_kv_cache_usage": max((s["kv"] for s in valid), default=None),
        "duration_s": round(time.perf_counter() - started, 3),
        "results": results,
        "samples": samples,
    }
    output = Path(__file__).resolve().parent / "results" / f"scheduler_capacity_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, ensure_ascii=False, indent=2))
    print(f"samples: {len(samples)}; saved: {output}")


if __name__ == "__main__":
    main()
