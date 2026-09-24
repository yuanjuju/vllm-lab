#!/usr/bin/env python3
"""Verify client cancellation and post-overload recovery on an existing server."""

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from token_budget_benchmark import read_metrics, total


BASE_URL = "http://127.0.0.1:18000"
MODEL = "Qwen/Qwen3-8B"


def gauges(metrics):
    return {
        "running": total(metrics, "vllm:num_requests_running"),
        "waiting": total(metrics, "vllm:num_requests_waiting"),
        "abort_total": total(metrics, "vllm:request_success_total",
                             'finished_reason="abort"'),
    }


def request(payload, timeout=30):
    return urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    before = gauges(read_metrics(BASE_URL))
    long_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"Cancellation probe {stamp}. List positive integers forever."}],
        "temperature": 0,
        "min_tokens": 2048,
        "max_tokens": 2048,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first_content_at = None
    started = time.perf_counter()
    with urllib.request.urlopen(request(long_payload), timeout=30) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if any((choice.get("delta") or {}).get("content")
                   for choice in chunk.get("choices", [])):
                first_content_at = round(time.perf_counter() - started, 6)
                break
        # Leaving the context closes the streaming response before [DONE].

    samples = []
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = gauges(read_metrics(BASE_URL))
        samples.append({"at_s": round(time.perf_counter() - started, 4), **state})
        if (state["abort_total"] >= before["abort_total"] + 1
                and state["running"] == 0 and state["waiting"] == 0):
            break
        time.sleep(0.1)
    after_cancel = gauges(read_metrics(BASE_URL))

    recovery_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word healthy."}],
        "temperature": 0,
        "min_tokens": 4,
        "max_tokens": 16,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    recovery_started = time.perf_counter()
    with urllib.request.urlopen(request(recovery_payload), timeout=30) as response:
        recovery = json.load(response)
    recovery_duration = round(time.perf_counter() - recovery_started, 4)
    final = gauges(read_metrics(BASE_URL))
    report = {
        "timestamp": stamp,
        "base_url": BASE_URL,
        "model": MODEL,
        "cancel_policy": "client closes stream after first non-empty content event; no retry",
        "first_content_at_s": first_content_at,
        "metrics_before": before,
        "metrics_after_cancel": after_cancel,
        "abort_counter_delta": after_cancel["abort_total"] - before["abort_total"],
        "cancel_recovered": bool(
            after_cancel["abort_total"] >= before["abort_total"] + 1
            and after_cancel["running"] == 0 and after_cancel["waiting"] == 0),
        "samples": samples,
        "recovery_request": {
            "duration_s": recovery_duration,
            "has_choice": bool(recovery.get("choices")),
            "finish_reason": ((recovery.get("choices") or [{}])[0].get("finish_reason")),
            "usage": recovery.get("usage"),
        },
        "final_metrics": final,
    }
    output = (Path(__file__).resolve().parent / "results" /
              f"overload_recovery_{stamp}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": report, "saved": str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
