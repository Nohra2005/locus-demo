# -*- coding: utf-8 -*-
"""
Locus — MLops Metrics Exporter
================================
Reads evaluation results from MLflow AND Gemini judge JSON files every 30s,
then exposes all of them as Prometheus metrics on port 8003.

MLflow-based metrics (LoRA retraining pipeline):
  locus_lora_promoted_total       — cumulative adapter promotions
  locus_lora_runs_total           — total retraining pipeline runs
  locus_retrain_active            — 1 if a retrain run is currently RUNNING, else 0
  locus_retrain_step              — latest logged training step in the active run
  locus_lora_train_loss           — latest train_loss value in the active run
  locus_retrain_max_steps         — MAX_TRAIN_STEPS param from the active run
  locus_judge_old_avg             — old model avg judge score in latest completed run
  locus_judge_new_avg             — new model avg judge score in latest completed run
  locus_judge_delta               — judge score delta (new - old); positive = improvement

Gemini Judge metrics (from gemini_judge_results.json):
  locus_gemini_avg_vss5           — average VSS@5 across all evaluated queries
  locus_gemini_query_vss5{label}  — VSS@5 per individual query
  locus_gemini_cache_entries      — total entries in vss_cache.json
  locus_gemini_queries_evaluated  — how many queries were in the last eval run
  locus_gemini_queries_poor/fair/good/excellent — grade distribution
  locus_gemini_avg_top1           — average Top-1 VSS score
  locus_gemini_score_bucket{le}   — score distribution histogram buckets
"""

import json
import os
import time

from prometheus_client import Gauge, Counter, start_http_server

MLFLOW_URI            = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPORT_PORT           = int(os.getenv("METRICS_PORT", "8003"))
REFRESH_SECS          = int(os.getenv("REFRESH_SECS", "30"))
RESULTS_PATH          = os.getenv("GEMINI_RESULTS_PATH", "/mlops/gemini_judge_results.json")
CACHE_PATH            = os.getenv("VSS_CACHE_PATH",       "/mlops/vss_cache.json")
GOLDEN_DATASET_PATH   = os.getenv("GOLDEN_DATASET_PATH",  "/mlops/golden_dataset.json")

# ── Category inference ────────────────────────────────────────────────────────
# Ordered so more-specific terms match before broader ones.
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Jumpsuits",  ["jumpsuit", "romper", "dungaree", "overall"]),
    ("Activewear", ["sports bra", "gym", "yoga", "athletic"]),
    ("Dresses",    ["dress"]),
    ("Skirts",     ["skirt"]),
    ("Outerwear",  ["coat", "jacket", "blazer", "parka", "anorak", "puffer", "bomber", "windbreaker"]),
    ("Footwear",   ["boot", "sneaker", "sandal", "shoe", "heel", "loafer", "pump", "trainer", "mule"]),
    ("Bags",       ["bag", "tote", "purse", "clutch", "backpack", "crossbody", "satchel"]),
    ("Accessories",["hat", "cap", "beret", "scarf", "belt", "sunglasses", "watch"]),
    ("Tops",       ["top", "shirt", "blouse", "hoodie", "cardigan", "sweatshirt", "tank", "sweater", "pullover", "polo", "tee"]),
    ("Bottoms",    ["jeans", "trouser", "pant", "legging", "chino", "jogger", "short"]),
]


def _infer_category(label: str) -> str:
    lower = label.lower()
    for cat, keywords in _CATEGORY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return cat
    return "Other"

# ── Prometheus metrics ────────────────────────────────────────────────────────

lora_promoted = Gauge("locus_lora_promoted_total", "Cumulative adapter promotions")
lora_runs     = Gauge("locus_lora_runs_total",      "Total retraining pipeline runs (promoted + rejected)")

# Live training metrics — updated whenever a run is RUNNING
retrain_active    = Gauge("locus_retrain_active",     "1 if a retrain run is currently active, else 0")
retrain_step      = Gauge("locus_retrain_step",       "Latest logged training step in the active run")
lora_train_loss   = Gauge("locus_lora_train_loss",    "Latest train_loss value logged by the active run")
retrain_max_steps = Gauge("locus_retrain_max_steps",  "MAX_TRAIN_STEPS param of the active run")

# Judge benchmark metrics from the latest completed retrain run
judge_old_avg = Gauge("locus_judge_old_avg", "Old model avg judge score in latest retrain run (20q × top-3)")
judge_new_avg = Gauge("locus_judge_new_avg", "New model avg judge score in latest retrain run")
judge_delta   = Gauge("locus_judge_delta",   "Judge score delta: new - old (positive = improvement)")

# ── Gemini Judge metrics ──────────────────────────────────────────────────────
gemini_avg_vss5      = Gauge("locus_gemini_avg_vss5",         "Average VSS@5 across all queries in latest eval run")
gemini_query_vss5    = Gauge("locus_gemini_query_vss5",        "VSS@5 score per query", ["label"])
gemini_cache_entries = Gauge("locus_gemini_cache_entries",     "Total entries in vss_cache.json")
gemini_queries_eval  = Gauge("locus_gemini_queries_evaluated", "Queries evaluated in latest Gemini run")
gemini_queries_poor  = Gauge("locus_gemini_queries_poor",      "Queries with VSS@5 < 0.55")
gemini_queries_fair  = Gauge("locus_gemini_queries_fair",      "Queries with VSS@5 in [0.55, 0.70)")
gemini_queries_good  = Gauge("locus_gemini_queries_good",      "Queries with VSS@5 in [0.70, 0.85)")
gemini_queries_excel = Gauge("locus_gemini_queries_excellent", "Queries with VSS@5 >= 0.85")
gemini_top1_avg      = Gauge("locus_gemini_avg_top1",          "Average Top-1 VSS score across all queries")
gemini_score_bucket       = Gauge("locus_gemini_score_bucket",          "Count of result scores <= threshold", ["le"])
gemini_avg_vss5_by_cat    = Gauge("locus_gemini_avg_vss5_by_category", "Avg VSS@5 per clothing category",    ["category"])
golden_dataset_size       = Gauge("locus_golden_dataset_size",          "Number of queries in golden_dataset.json")

# ── LoRA run health (populated from MLflow run history) ───────────────────────
lora_failed_runs      = Gauge("locus_lora_failed_runs_total",        "Runs that ended with FAILED status")
lora_avg_duration     = Gauge("locus_lora_avg_run_duration_seconds", "Average wall-clock time of completed retrain runs")
lora_last_duration    = Gauge("locus_lora_last_run_duration_seconds","Duration of the most recent completed run in seconds")
lora_data_points      = Gauge("locus_lora_data_points_used",         "training_pairs count from the last training run")

# TODO: populate these from a post-train hit@5 eval step in retrain_clip.py
lora_new_hit5         = Gauge("locus_lora_new_hit5",                 "New model hit@5 (requires post-train eval in retrain_clip.py)")
lora_new_acs5         = Gauge("locus_lora_new_acs5",                 "New model ACS@5 (requires post-train eval in retrain_clip.py)")
lora_baseline_hit5    = Gauge("locus_baseline_hit5",                 "Baseline model hit@5 (requires post-train eval in retrain_clip.py)")
lora_last_error_code  = Gauge("locus_lora_last_error_code",          "Error code of last failed run (0=none; requires error tagging in pipeline)")

# ── Recall eval (populated from locus_recall_eval MLflow experiment) ──────────
recall_at_5           = Gauge("locus_recall_at_5",  "Recall@5 from latest locus_recall_eval MLflow run")
recall_at_10          = Gauge("locus_recall_at_10", "Recall@10 from latest locus_recall_eval MLflow run")
recall_at_25          = Gauge("locus_recall_at_25", "Recall@25 from latest locus_recall_eval MLflow run")


def _get_mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient()



def _refresh_retrain_metrics(client) -> None:
    """Pull retraining stats from 'locus_lora_retraining' experiment."""
    try:
        exp = client.get_experiment_by_name("locus_lora_retraining")
        if exp is None:
            lora_failed_runs.set(0)
            lora_avg_duration.set(0)
            lora_last_duration.set(0)
            lora_data_points.set(0)
            return

        # All runs for aggregate counts
        all_runs = client.search_runs(
            experiment_ids = [exp.experiment_id],
            max_results    = 1000,
        )
        if not all_runs:
            lora_failed_runs.set(0)
            lora_avg_duration.set(0)
            lora_last_duration.set(0)
            lora_data_points.set(0)
            return

        lora_runs.set(len(all_runs))

        promoted_count = sum(
            1 for r in all_runs
            if r.data.tags.get("promoted") == "true"
        )
        lora_promoted.set(promoted_count)

        # Latest completed run for judge metrics
        finished = [r for r in all_runs if r.info.status == "FINISHED"]
        if not finished:
            print(f"[exporter] Retrain — runs={len(all_runs)} promoted={promoted_count} (no finished runs yet)")
            return

        latest  = sorted(finished, key=lambda r: r.info.start_time, reverse=True)[0]
        outcome = latest.data.params.get("outcome", "unknown")

        j_old = latest.data.metrics.get("judge_old_avg")
        j_new = latest.data.metrics.get("judge_new_avg")
        j_d   = latest.data.metrics.get("judge_delta")

        if j_old is not None:
            judge_old_avg.set(round(float(j_old), 4))
        if j_new is not None:
            judge_new_avg.set(round(float(j_new), 4))
        if j_d is not None:
            judge_delta.set(round(float(j_d), 4))

        # ── Run health metrics ────────────────────────────────────────────────
        failed = [r for r in all_runs if r.info.status == "FAILED"]
        lora_failed_runs.set(len(failed))

        # Duration stats from FINISHED runs (MLflow timestamps are milliseconds)
        durations = [
            (r.info.end_time - r.info.start_time) / 1000.0
            for r in finished
            if r.info.end_time
        ]
        if durations:
            lora_last_duration.set(round(durations[0], 1))  # latest first (sorted above)
            lora_avg_duration.set(round(sum(durations) / len(durations), 1))

        # Data points used (training_pairs param)
        dp = latest.data.params.get("training_pairs")
        if dp is not None:
            try:
                lora_data_points.set(int(dp))
            except (ValueError, TypeError):
                pass

        print(
            f"[exporter] Retrain — runs={len(all_runs)} promoted={promoted_count} "
            f"failed={len(failed)} "
            f"latest_outcome={outcome} "
            f"judge_old={j_old} judge_new={j_new} judge_delta={j_d}"
        )

    except Exception as e:
        print(f"[exporter] Could not read locus_lora_retraining from MLflow: {e}")


def _refresh_live_training_metrics(client) -> None:
    """
    Poll for any RUNNING locus_lora_retraining run and export its live metrics.

    MLflow writes log_metric() calls synchronously to the backend, so the
    latest train_loss from an in-progress run is visible here between steps.
    When no run is active, all live gauges are zeroed.
    """
    try:
        exp = client.get_experiment_by_name("locus_lora_retraining")
        if exp is None:
            retrain_active.set(0)
            return

        running = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not running:
            retrain_active.set(0)
            retrain_step.set(0)
            lora_train_loss.set(0)
            print("[exporter] Live training: no active run")
            return

        run = running[0]
        retrain_active.set(1)

        # Latest train_loss — get_metric_history returns all logged points;
        # the last one has the highest step.
        try:
            history = client.get_metric_history(run.info.run_id, "train_loss")
            if history:
                latest = max(history, key=lambda m: m.step)
                lora_train_loss.set(round(latest.value, 6))
                retrain_step.set(latest.step)
            else:
                # Run started but hasn't logged a step yet (still in pair-build/init phase)
                lora_train_loss.set(0)
                retrain_step.set(0)
        except Exception as e:
            print(f"[exporter] Could not read train_loss history: {e}")

        # MAX_TRAIN_STEPS stored as a run param
        try:
            max_s = run.data.params.get("max_train_steps")
            if max_s is not None:
                retrain_max_steps.set(int(max_s))
        except Exception:
            pass

        print(
            f"[exporter] Live training — run={run.info.run_id[:8]} "
            f"step={int(retrain_step._value.get())} "
            f"loss={lora_train_loss._value.get():.4f}"
        )

    except Exception as e:
        print(f"[exporter] Could not read live training metrics: {e}")
        retrain_active.set(0)


def _refresh_gemini_metrics() -> None:
    """Read gemini_judge_results.json and results_cache.json, update Gemini gauges."""
    # Always update cache entry count — this is the judge health proxy.
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            gemini_cache_entries.set(len(cache))
        except Exception as e:
            print(f"[exporter] Could not read results cache: {e}")

    if not os.path.exists(RESULTS_PATH):
        print(f"[exporter] Gemini results not found: {RESULTS_PATH} — waiting for first eval run")
        return
    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except Exception as e:
        print(f"[exporter] Could not read Gemini results: {e}")
        return

    vss5_list  = [r["vss5"]       for r in results if "vss5"       in r]
    top1_list  = [r["top1_score"] for r in results if "top1_score" in r]
    all_scores = [s for r in results for s in r.get("scores", [])]

    if not vss5_list:
        return

    avg = sum(vss5_list) / len(vss5_list)
    gemini_avg_vss5.set(round(avg, 4))
    gemini_queries_eval.set(len(vss5_list))
    if top1_list:
        gemini_top1_avg.set(round(sum(top1_list) / len(top1_list), 4))

    gemini_queries_poor.set( sum(1 for s in vss5_list if s < 0.55))
    gemini_queries_fair.set( sum(1 for s in vss5_list if 0.55 <= s < 0.70))
    gemini_queries_good.set( sum(1 for s in vss5_list if 0.70 <= s < 0.85))
    gemini_queries_excel.set(sum(1 for s in vss5_list if s >= 0.85))

    # ── per-query VSS@5 ───────────────────────────────────────────────────────
    cat_scores: dict[str, list[float]] = {}
    for r in results:
        if "vss5" not in r:
            continue
        gemini_query_vss5.labels(label=r["label"][:50]).set(r["vss5"])
        cat = r.get("category") or _infer_category(r["label"])
        cat_scores.setdefault(cat, []).append(r["vss5"])

    for cat, scores in cat_scores.items():
        gemini_avg_vss5_by_cat.labels(category=cat).set(round(sum(scores) / len(scores), 4))

    for threshold in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gemini_score_bucket.labels(le=str(threshold)).set(
            sum(1 for s in all_scores if s <= threshold)
        )

    print(
        f"[exporter] Gemini — {len(vss5_list)} queries, avg_vss5={avg:.3f}, "
        f"categories={list(cat_scores.keys())} "
        f"poor={int(gemini_queries_poor._value.get())} "
        f"fair={int(gemini_queries_fair._value.get())} "
        f"good={int(gemini_queries_good._value.get())} "
        f"excellent={int(gemini_queries_excel._value.get())}"
    )


def _refresh_recall_metrics(client) -> None:
    """Pull recall@K scores from the latest locus_recall_eval MLflow run."""
    try:
        exp = client.get_experiment_by_name("locus_recall_eval")
        if exp is None:
            return
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return
        latest = runs[0]
        for k, gauge in ((5, recall_at_5), (10, recall_at_10), (25, recall_at_25)):
            val = latest.data.metrics.get(f"recall_at_{k}")
            if val is not None:
                gauge.set(round(float(val), 4))
        print(
            f"[exporter] Recall eval — "
            f"recall@5={latest.data.metrics.get('recall_at_5')} "
            f"recall@10={latest.data.metrics.get('recall_at_10')} "
            f"recall@25={latest.data.metrics.get('recall_at_25')}"
        )
    except Exception as e:
        print(f"[exporter] Could not read locus_recall_eval from MLflow: {e}")


def _refresh_golden_dataset_size() -> None:
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return
    try:
        with open(GOLDEN_DATASET_PATH) as f:
            data = json.load(f)
        golden_dataset_size.set(len(data))
        print(f"[exporter] Golden dataset — {len(data)} queries")
    except Exception as e:
        print(f"[exporter] Could not read golden dataset: {e}")


def refresh() -> None:
    try:
        client = _get_mlflow_client()
        _refresh_retrain_metrics(client)
        _refresh_live_training_metrics(client)
        _refresh_recall_metrics(client)
    except Exception as e:
        print(f"[exporter] MLflow connection error: {e}")
    _refresh_gemini_metrics()
    _refresh_golden_dataset_size()


if __name__ == "__main__":
    print(f"[exporter] Starting Locus MLops metrics exporter on port {EXPORT_PORT}")
    print(f"[exporter] MLflow URI    : {MLFLOW_URI}")
    print(f"[exporter] Refresh every : {REFRESH_SECS}s")

    start_http_server(EXPORT_PORT)
    print(f"[exporter] Prometheus metrics at http://0.0.0.0:{EXPORT_PORT}/metrics")

    while True:
        refresh()
        time.sleep(REFRESH_SECS)
