#!/usr/bin/env python3
"""Compare BF16 Qwen3-8B weights with official Qwen3-8B-AWQ weights.

Both service variants expose the same model ID. The script reuses the fixed
long-input/short-quality workload and metrics helpers from
kv_dtype_ab_exploratory.py; it does not start or stop the service.
"""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from kv_dtype_ab_exploratory import (
    MODEL,
    long_prompt,
    percentile,
    quality_suite,
    run_burst,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("bf16", "awq"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--filler-repeats", type=int, default=180)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--formal-repetitions", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--server-profile", default="r570-vllm018-weight-ab-bf16-activation-bf16-kv-eager-prefix-off")
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
        print(json.dumps({"variant": args.variant, "labels": labels,
                          "requests": args.requests,
                          "first_prompt_sha256": hashlib.sha256(
                              prompt.encode()).hexdigest(),
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
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "model_served_name": MODEL,
        "model_directory": ("models/Qwen3-8B" if args.variant == "bf16"
                            else "models/Qwen3-8B-AWQ"),
        "base_url": base_url,
        "transport": "cloud-local loopback",
        "server_profile": args.server_profile,
        "requests_per_burst": args.requests,
        "filler_repeats": args.filler_repeats,
        "fixed_output_tokens": args.output_tokens,
        "formal_repetitions_planned": args.formal_repetitions,
        "warmup_excluded_from_formal": True,
        "client_event_gap_note": (
            "SSE content events are not individual tokens; use server histogram "
            "deltas for ITL. Quality probes are separate from formal bursts."
        ),
        "quality_cases": quality,
        "runs": runs,
        "formal_medians": {
            key: percentile([row[key] for row in formal if row[key] is not None], .5)
            for key in ("client_ttft_p50_s", "client_ttft_p95_s",
                        "output_throughput_tok_s", "max_kv_usage")
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"weight_awq_ab_{args.experiment_id}_{args.variant}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
