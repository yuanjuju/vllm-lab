#!/usr/bin/env python3
"""Compare request usage with vLLM counters before and after cache reuse."""

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_URL = "http://127.0.0.1:18000"
MODEL = "Qwen/Qwen3-8B"
METRICS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:request_success_total",
)


def metrics_snapshot():
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as response:
        body = response.read().decode()
    raw = [
        line for line in body.splitlines()
        if any(
            line.startswith(metric + "{")
            or line.startswith("# HELP " + metric + " ")
            or line.startswith("# TYPE " + metric + " ")
            for metric in METRICS
        )
    ]
    values = {}
    for line in raw:
        if line.startswith("#"):
            continue
        metric, value = line.rsplit(" ", 1)
        values[metric] = float(value)
    return {"raw": raw, "values": values}


def send_request(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    return {"elapsed_s": round(time.perf_counter() - start, 4), "body": body}


def delta(before, after, metric):
    prior = [value for key, value in before["values"].items() if key.startswith(metric + "{")]
    later = [value for key, value in after["values"].items() if key.startswith(metric + "{")]
    return int(sum(later) - sum(prior))


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique = f"{stamp}-{time.time_ns()}"
    prompt = (
        f"指标实验编号：{unique}。"
        "vLLM服务接到提示词后会处理输入tokens，然后逐个生成输出tokens。"
        "服务端公开的指标是累计值，单次API响应的usage只描述当前请求。" * 8
        + "请仅回答：已完成。"
    )
    before = metrics_snapshot()
    first = send_request(prompt)
    middle = metrics_snapshot()
    second = send_request(prompt)
    after = metrics_snapshot()

    metric_deltas = {
        label: {
            name: delta(start, end, name)
            for name in METRICS if name != "vllm:request_success_total"
        }
        for label, start, end in (
            ("first", before, middle),
            ("second", middle, after),
        )
    }
    report = {
        "timestamp": stamp,
        "model": MODEL,
        "base_url": BASE_URL,
        "prompt": prompt,
        "before": before,
        "first": first,
        "middle": middle,
        "second": second,
        "after": after,
        "metric_deltas": metric_deltas,
    }
    output = Path(__file__).resolve().parent / "results" / f"metrics_counter_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("===== BEFORE (raw /metrics lines) =====")
    print("\n".join(before["raw"]))
    print("===== FIRST RESPONSE (raw /v1/chat/completions JSON) =====")
    print(json.dumps(first["body"], ensure_ascii=False, indent=2))
    print("===== MIDDLE (raw /metrics lines) =====")
    print("\n".join(middle["raw"]))
    print("===== SECOND RESPONSE (raw /v1/chat/completions JSON) =====")
    print(json.dumps(second["body"], ensure_ascii=False, indent=2))
    print("===== AFTER (raw /metrics lines) =====")
    print("\n".join(after["raw"]))
    print("===== DELTA =====")
    print(json.dumps(metric_deltas, ensure_ascii=False, indent=2))
    print(f"first_elapsed_s: {first['elapsed_s']}; second_elapsed_s: {second['elapsed_s']}; saved: {output}")


if __name__ == "__main__":
    main()
