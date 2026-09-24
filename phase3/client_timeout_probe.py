#!/usr/bin/env python3
"""Trigger a client-side timeout and verify that server work drains safely."""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from token_budget_benchmark import read_metrics, total


BASE_URL = "http://127.0.0.1:18000"
MODEL = "Qwen/Qwen3-8B"


def state():
    metrics = read_metrics(BASE_URL)
    return {
        "running": total(metrics, "vllm:num_requests_running"),
        "waiting": total(metrics, "vllm:num_requests_waiting"),
        "abort_total": total(metrics, "vllm:request_success_total",
                             'finished_reason="abort"'),
    }


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    before = state()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"Timeout probe {stamp}. List positive integers forever."}],
        "temperature": 0,
        "min_tokens": 2048,
        "max_tokens": 2048,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    started = time.perf_counter()
    error = None
    try:
        with urllib.request.urlopen(request, timeout=0.05) as response:
            response.read(1)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    samples = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = state()
        samples.append({"at_s": round(time.perf_counter() - started, 4), **current})
        if current["running"] == 0 and current["waiting"] == 0:
            break
        time.sleep(0.1)
    after = state()
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
        health_status = response.status
    report = {
        "timestamp": stamp,
        "base_url": BASE_URL,
        "model": MODEL,
        "client_timeout_s": 0.05,
        "retry_attempted": False,
        "client_error": error,
        "metrics_before": before,
        "samples": samples,
        "metrics_after": after,
        "abort_counter_delta": after["abort_total"] - before["abort_total"],
        "work_drained": after["running"] == 0 and after["waiting"] == 0,
        "health_status_after": health_status,
    }
    output = (Path(__file__).resolve().parent / "results" /
              f"client_timeout_{stamp}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": report, "saved": str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
