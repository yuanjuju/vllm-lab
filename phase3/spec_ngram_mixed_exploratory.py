#!/usr/bin/env python3
"""Paired mixed-load probe for the isolated R570/vLLM 0.18.0 server.

Start the baseline or N-gram server with run_cloud_r570_vllm018_ngram_spec.sh
before this client. Each burst releases four repeat and four count prompts
simultaneously. The client never starts services or installs dependencies.
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

from prefix_cache_ab_exploratory import histogram_delta, metric_delta, metric_total, percentile
from spec_ngram_ab_exploratory import (
    GAUGES,
    HISTOGRAMS,
    MODEL,
    RESULTS_DIR,
    SPEC_COUNTERS,
    position_deltas,
    prompt_for,
    read_metrics,
    request,
)


def group_summary(items):
    good = [item for item in items if item["ok"]]
    return {
        "success_count": len(good),
        "request_count": len(items),
        "client_ttft_p50_s": percentile([x["client_ttft_s"] for x in good], .5),
        "client_ttft_p95_s": percentile([x["client_ttft_s"] for x in good], .95),
        "client_e2e_p50_s": percentile([x["duration_s"] for x in good], .5),
        "client_e2e_p95_s": percentile([x["duration_s"] for x in good], .95),
        "client_decode_tail_per_token_p50_s": percentile(
            [x["client_decode_tail_per_token_s"] for x in good], .5),
        "prompt_tokens_min_max": ([
            min(x["usage"]["prompt_tokens"] for x in good),
            max(x["usage"]["prompt_tokens"] for x in good),
        ] if good else None),
    }


def run_mixed_burst(base_url, experiment_id, label, output_tokens, poll_interval):
    prompts = [
        (shape, index, prompt_for(experiment_id, shape, 8, label, index))
        for shape in ("repeat", "count") for index in range(4)
    ]
    before = read_metrics(base_url)
    gate = threading.Barrier(len(prompts) + 1)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(request, base_url, prompt, f"{shape}-mixed-{label}-{index}",
                        output_tokens, gate)
            for shape, index, prompt in prompts
        ]
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
    records = [
        {"shape": shape, "index": index,
         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **output}
        for (shape, index, prompt), output in zip(prompts, outputs)
    ]
    good = [item for item in records if item["ok"]]
    output_total = sum(item["usage"]["completion_tokens"] for item in good)
    drafts = metric_delta(before, after, SPEC_COUNTERS[0])
    drafted = metric_delta(before, after, SPEC_COUNTERS[1])
    accepted = metric_delta(before, after, SPEC_COUNTERS[2])
    summary = {
        "success_count": len(good),
        "request_count": len(records),
        "burst_duration_s": round(duration, 4),
        "output_tokens": output_total,
        "output_throughput_tok_s": round(output_total / duration, 3),
        "by_shape": {shape: group_summary(
            [item for item in records if item["shape"] == shape])
            for shape in ("repeat", "count")},
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
        "metric_poll_errors": sum("error" in sample for sample in samples),
        "histogram_deltas": {name: histogram_delta(before, after, name)
                             for name in HISTOGRAMS},
    }
    return {"label": label, "requests": records, "samples": samples,
            "metrics_before": before, "metrics_after": after, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("baseline", "ngram"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--formal-repetitions", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
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
            "request_mix": {"repeat": 4, "count": 4},
            "example_hashes": {shape: [hashlib.sha256(prompt_for(
                args.experiment_id, shape, 8, "formal-0", i).encode()
            ).hexdigest() for i in range(4)] for shape in ("repeat", "count")},
        }, ensure_ascii=False, indent=2))
        return
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in models):
        raise SystemExit(f"{MODEL} not registered")
    runs = []
    for label in labels:
        run = run_mixed_burst(base_url, args.experiment_id, label,
                              args.output_tokens, args.poll_interval)
        runs.append(run)
        print(json.dumps({"label": label, "summary": run["summary"]},
                         ensure_ascii=False), flush=True)
        if run["summary"]["success_count"] != 8:
            break
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "condition": args.condition,
        "model": MODEL,
        "base_url": base_url,
        "transport": "cloud-local loopback",
        "server_profile": "r570-vllm018-qwen3-8b-bf16-eager-prefix-off",
        "request_mix": {"repeat": 4, "count": 4},
        "fixed_output_tokens": args.output_tokens,
        "formal_repetitions_planned": args.formal_repetitions,
        "warmup_excluded_from_formal": True,
        "aggregate_spec_counters_note": (
            "Prometheus speculation counters are burst-wide, not attributed "
            "to the two prompt shapes."
        ),
        "client_decode_tail_note": (
            "(E2E-TTFT)/(output_tokens-1) is an amortized end-to-end proxy, "
            "not per-token ITL; one SSE event may contain multiple tokens."
        ),
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"spec_ngram_mixed_{args.experiment_id}_{args.condition}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
