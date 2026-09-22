#!/usr/bin/env python3
"""Compare cold, shared-prefix, and changed-prefix vLLM requests."""

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_URL = "http://127.0.0.1:18000"
MODEL = "Qwen/Qwen3-8B"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def cache_counts():
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        for key, metric in (
            ("queries", "vllm:prefix_cache_queries_total"),
            ("hits", "vllm:prefix_cache_hits_total"),
        ):
            if line.startswith(metric + "{"):
                values[key] = int(float(line.rsplit(" ", 1)[1]))
    if set(values) != {"queries", "hits"}:
        raise RuntimeError(f"Missing prefix-cache counters: {values}")
    return values


def ask(content):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return {
        "seconds": round(time.perf_counter() - start, 4),
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
        "finish_reason": result["choices"][0]["finish_reason"],
        "answer": result["choices"][0]["message"]["content"],
    }


def main():
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as response:
        model_list = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in model_list):
        raise RuntimeError(f"{MODEL} is not served")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{time.time_ns()}"
    paragraph = (
        "vLLM是面向大语言模型推理服务的引擎。"
        "它在处理请求时使用调度器组织运行序列，并将注意力计算需要的键和值存入KV缓存。"
        "如果后续请求有相同的前缀，自动前缀缓存可以复用已经计算好的部分。"
        "本段内容只用于构造较长且稳定的实验前缀。"
    )
    shared = f"实验编号：{run_id}。" + paragraph * 24
    changed = f"不同实验编号：{run_id}-CONTROL。" + paragraph * 24
    cases = [
        ("A_cold", shared + "请用一句话说明KV缓存的作用。"),
        ("B_shared_prefix", shared + "请用一句话说明前缀缓存的作用。"),
        ("C_changed_prefix", changed + "请用一句话说明前缀缓存的作用。"),
    ]

    records = []
    before = cache_counts()
    for label, content in cases:
        response = ask(content)
        after = cache_counts()
        queried = after["queries"] - before["queries"]
        hits = after["hits"] - before["hits"]
        records.append({
            "case": label,
            "same_shared_prefix_as_A": label == "B_shared_prefix",
            "queries_delta": queried,
            "hits_delta": hits,
            "hit_ratio": round(hits / queried, 4) if queried else None,
            **response,
        })
        before = after

    report = {
        "run_id": run_id,
        "model": MODEL,
        "base_url": BASE_URL,
        "paragraph_repetitions": 24,
        "note": "Elapsed time includes Mac client, SSH tunnel, and cloud server; single-run latency is not a speedup benchmark.",
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"prefix_partial_reuse_{stamp}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(
            f"{record['case']}: prompt={record['prompt_tokens']}, "
            f"queries={record['queries_delta']}, hits={record['hits_delta']}, "
            f"hit_ratio={record['hit_ratio']:.1%}, "
            f"time={record['seconds']:.4f}s, finish={record['finish_reason']}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
