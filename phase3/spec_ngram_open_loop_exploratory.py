#!/usr/bin/env python3
"""Bounded fixed-rate mixed-load probe for the R570/vLLM 0.18.0 profile.

Requests are dispatched on an absolute schedule independent of completion.
This is a deterministic open-loop arrival process, not a Poisson workload.
Only the N-gram server switch differs between paired conditions.
"""

import argparse
import hashlib
import json
import math
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


RATES = (0.5, 1.0, 1.5, 2.0)
FORMAL_DURATION_S = 20.0
WARMUP_DURATION_S = 8.0
OUTPUT_TOKENS = 256
TTFT_SLO_S = 1.0  # Exploratory teaching target, not a user production SLO.
E2E_SLO_S = 10.0


def selected_percentiles(items, field):
    values = [item[field] for item in items if item.get("ok") and item.get(field) is not None]
    return {"p50_s": percentile(values, .5), "p95_s": percentile(values, .95)}


def run_rate(base_url, experiment_id, rate, label, duration_s, poll_interval,
             max_drain_s):
    planned_count_float = rate * duration_s
    if not math.isclose(planned_count_float, round(planned_count_float), abs_tol=1e-9):
        raise ValueError("rate * duration must be an integer for a 50:50 mix")
    planned_count = round(planned_count_float)
    if planned_count < 2 or planned_count % 2:
        raise ValueError("need an even number of requests")
    before = read_metrics(base_url)
    ready = threading.Event()
    ready.set()
    stop_monitor = threading.Event()
    safety_stop = threading.Event()
    lock = threading.Lock()
    samples, records, threads = [], [], []
    active = 0
    max_client_inflight = 0
    safety_reason = None
    start = time.perf_counter()

    def monitor():
        nonlocal safety_reason
        while not stop_monitor.is_set():
            try:
                metrics = read_metrics(base_url)
                at_s = round(time.perf_counter() - start, 4)
                running = metric_total(metrics, GAUGES[0])
                waiting = metric_total(metrics, GAUGES[1])
                kv = metric_total(metrics, GAUGES[2])
                preempt = metric_delta(before, metrics, "vllm:num_preemptions_total")
                with lock:
                    samples.append({"at_s": at_s, "running": running,
                                    "waiting": waiting, "kv_usage": kv,
                                    "preemptions_delta": preempt})
                if kv is not None and kv >= .75:
                    safety_reason = "KV usage >= 75%"
                elif waiting is not None and waiting > 16:
                    safety_reason = "Waiting > 16"
                elif preempt is not None and preempt > 0:
                    safety_reason = "preemption counter increased"
                if safety_reason:
                    safety_stop.set()
            except Exception as exc:
                with lock:
                    samples.append({"at_s": round(time.perf_counter() - start, 4),
                                    "error": f"{type(exc).__name__}: {exc}"})
            stop_monitor.wait(poll_interval)

    def worker(record, prompt):
        nonlocal active, max_client_inflight
        worker_offset = time.perf_counter() - start
        with lock:
            record["worker_start_offset_s"] = round(worker_offset, 6)
            record["worker_start_lag_s"] = round(max(
                0, worker_offset - record["scheduled_offset_s"]), 6)
            active += 1
            max_client_inflight = max(max_client_inflight, active)
        response = request(base_url, prompt, record["request_id"], OUTPUT_TOKENS, ready)
        with lock:
            record.update(response)
            record["completion_offset_s"] = round(time.perf_counter() - start, 4)
            active -= 1

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    for index in range(planned_count):
        target = start + index / rate
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        if safety_stop.is_set():
            break
        shape = "repeat" if index % 2 == 0 else "count"
        shape_index = index // 2
        prompt = prompt_for(experiment_id, shape, 8,
                            f"rate{rate:g}-{label}", shape_index)
        now_offset = time.perf_counter() - start
        record = {
            "request_id": f"{shape}-r{rate:g}-{label}-{shape_index}",
            "shape": shape, "index": index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "scheduled_offset_s": round(index / rate, 6),
            "dispatch_offset_s": round(now_offset, 6),
            "dispatch_lag_s": round(max(0, now_offset - index / rate), 6),
            "ok": False, "error": "request did not complete",
        }
        records.append(record)
        thread = threading.Thread(target=worker, args=(record, prompt), daemon=True)
        threads.append(thread)
        thread.start()

    dispatched = len(records)
    # Drain only after the fixed arrival window; completion never gates arrivals.
    remaining_window = start + duration_s - time.perf_counter()
    if remaining_window > 0:
        time.sleep(remaining_window)
    arrival_end_offset = time.perf_counter() - start
    drain_deadline = time.perf_counter() + max_drain_s
    for thread in threads:
        thread.join(timeout=max(0, drain_deadline - time.perf_counter()))
    unfinished = sum(thread.is_alive() for thread in threads)
    stop_monitor.set()
    monitor_thread.join(timeout=2)
    end_offset = time.perf_counter() - start
    after = read_metrics(base_url)
    good = [item for item in records if item.get("ok")]
    output_total = sum(item["usage"]["completion_tokens"] for item in good)
    slo_good = [item for item in good
                if item["client_ttft_s"] <= TTFT_SLO_S
                and item["duration_s"] <= E2E_SLO_S]
    drafts = metric_delta(before, after, SPEC_COUNTERS[0])
    draft_tokens = metric_delta(before, after, SPEC_COUNTERS[1])
    accepted = metric_delta(before, after, SPEC_COUNTERS[2])
    by_shape = {}
    for shape in ("repeat", "count"):
        group = [item for item in records if item["shape"] == shape]
        by_shape[shape] = {
            "request_count": len(group),
            "success_count": sum(bool(item.get("ok")) for item in group),
            "ttft": selected_percentiles(group, "client_ttft_s"),
            "e2e": selected_percentiles(group, "duration_s"),
            "slo_good_count": sum(item in slo_good for item in group),
        }
    summary = {
        "planned_request_count": planned_count,
        "dispatched_request_count": dispatched,
        "success_count": len(good),
        "unfinished_count": unfinished,
        "offered_rate_req_s": rate,
        "actual_dispatch_rate_req_s": round(dispatched / duration_s, 4),
        "arrival_window_s": duration_s,
        "arrival_end_offset_s": round(arrival_end_offset, 4),
        "total_elapsed_s": round(end_offset, 4),
        "drain_after_arrivals_s": round(max(0, end_offset - duration_s), 4),
        "client_inflight_peak": max_client_inflight,
        "dispatch_lag_p95_s": percentile([r["dispatch_lag_s"] for r in records], .95),
        "worker_start_lag_p95_s": percentile([
            r["worker_start_lag_s"] for r in records
            if r.get("worker_start_lag_s") is not None], .95),
        "output_tokens": output_total,
        "output_throughput_tok_s": round(output_total / end_offset, 3),
        "success_rate": round(len(good) / dispatched, 4) if dispatched else None,
        "ttft": selected_percentiles(records, "client_ttft_s"),
        "e2e": selected_percentiles(records, "duration_s"),
        "slo_good_count": len(slo_good),
        "slo_good_fraction": round(len(slo_good) / dispatched, 4) if dispatched else None,
        "slo_goodput_req_s": round(len(slo_good) / end_offset, 4),
        "completed_before_arrival_end": sum(
            item.get("ok") and item.get("completion_offset_s", math.inf) <= duration_s
            for item in records),
        "by_shape": by_shape,
        "max_running": max((x["running"] for x in samples
                            if x.get("running") is not None), default=None),
        "max_waiting": max((x["waiting"] for x in samples
                            if x.get("waiting") is not None), default=None),
        "waiting_near_arrival_end": next((
            x["waiting"] for x in reversed(samples)
            if x.get("waiting") is not None and x["at_s"] <= duration_s), None),
        "max_kv_usage": max((x["kv_usage"] for x in samples
                             if x.get("kv_usage") is not None), default=None),
        "preemptions_delta": metric_delta(before, after, "vllm:num_preemptions_total"),
        "spec_drafts_delta": drafts,
        "spec_draft_tokens_delta": draft_tokens,
        "spec_accepted_tokens_delta": accepted,
        "spec_draft_acceptance_rate": (
            round(accepted / draft_tokens, 4) if draft_tokens else None),
        "spec_mean_acceptance_length": (
            round(1 + accepted / drafts, 4) if drafts else None),
        "spec_accepted_per_position": position_deltas(before, after),
        "metric_poll_errors": sum("error" in sample for sample in samples),
        "histogram_deltas": {name: histogram_delta(before, after, name)
                             for name in HISTOGRAMS},
        "safety_stop_reason": safety_reason,
    }
    return {"rate_req_s": rate, "label": label,
            "summary": summary, "requests": records, "samples": samples,
            "metrics_before": before, "metrics_after": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("baseline", "ngram"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--poll-interval", type=float, default=.2)
    parser.add_argument("--max-drain-s", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.experiment_id):
        parser.error("experiment-id must be 1-64 ASCII letters, digits, _ or -")
    if args.poll_interval <= 0 or args.max_drain_s <= 0:
        parser.error("poll interval and max drain must be positive")
    if args.dry_run:
        print(json.dumps({
            "rates_req_s": RATES,
            "formal_duration_s": FORMAL_DURATION_S,
            "warmup_duration_s": WARMUP_DURATION_S,
            "formal_repetitions": 3,
            "requests_per_formal_run": [round(r * FORMAL_DURATION_S) for r in RATES],
            "mix": "alternating repeat/count, 50:50",
            "output_tokens": OUTPUT_TOKENS,
            "illustrative_slo": {"ttft_s": TTFT_SLO_S, "e2e_s": E2E_SLO_S},
            "paired_example_prompt_sha256": hashlib.sha256(prompt_for(
                args.experiment_id, "repeat", 8, "rate1-formal-0", 0
            ).encode()).hexdigest(),
        }, ensure_ascii=False, indent=2))
        return
    base_url = args.base_url.rstrip("/")
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(item["id"] == MODEL for item in models):
        raise SystemExit(f"{MODEL} not registered")
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "condition": args.condition,
        "model": MODEL,
        "base_url": base_url,
        "transport": "cloud-local loopback",
        "server_profile": "r570-vllm018-qwen3-8b-bf16-eager-prefix-off",
        "arrival_process": "deterministic periodic open-loop; no client concurrency cap",
        "rates_req_s": RATES,
        "formal_duration_s": FORMAL_DURATION_S,
        "warmup_duration_s": WARMUP_DURATION_S,
        "formal_repetitions_planned": 3,
        "mix": "alternating repeat/count, 50:50",
        "fixed_output_tokens": OUTPUT_TOKENS,
        "illustrative_slo": {"ttft_s": TTFT_SLO_S, "e2e_s": E2E_SLO_S},
        "safety_stop": "stop dispatch if KV >=75%, Waiting >16, or preemptions >0; "
                       "stop later rates after any incomplete request",
        "warmup_excluded_from_formal": True,
        "throughput_note": "output tokens / total arrival-plus-drain wall time",
        "runs": [],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"spec_ngram_open_loop_{args.experiment_id}_{args.condition}.json"
    for rate in RATES:
        for label, duration_s in [
            ("warmup", WARMUP_DURATION_S),
            *[(f"formal-{i}", FORMAL_DURATION_S) for i in range(3)],
        ]:
            run = run_rate(base_url, args.experiment_id, rate, label,
                           duration_s, args.poll_interval, args.max_drain_s)
            result["runs"].append(run)
            staging = output.with_suffix(".json.tmp")
            staging.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            staging.replace(output)
            summary = run["summary"]
            print(json.dumps({"rate": rate, "label": label,
                              "success": summary["success_count"],
                              "dispatched": summary["dispatched_request_count"],
                              "ttft_p95_s": summary["ttft"]["p95_s"],
                              "e2e_p95_s": summary["e2e"]["p95_s"],
                              "slo_good_fraction": summary["slo_good_fraction"],
                              "waiting_peak": summary["max_waiting"],
                              "kv_peak": summary["max_kv_usage"],
                              "preemptions": summary["preemptions_delta"],
                              "safety_stop": summary["safety_stop_reason"]},
                             ensure_ascii=False), flush=True)
            if (summary["safety_stop_reason"] or summary["unfinished_count"]
                    or summary["success_count"] != summary["planned_request_count"]):
                print(f"Stopped escalation after {rate:g} req/s {label}; "
                      f"partial data saved: {output}", flush=True)
                return
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
