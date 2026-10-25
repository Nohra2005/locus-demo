"""
migrate_local_runs.py — Logs historical experiment runs into the running MLflow server.

The original mlops/mlflow.db is gone, so metrics are hardcoded from known results.

Run once:
    python mlops/migrate_local_runs.py
"""
import os
import mlflow

REMOTE_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(REMOTE_URI)


def log_run(experiment_name: str, run_name: str, params: dict, metrics: dict, note: str = "") -> None:
    if note:
        params = {**params, "note": note}
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    print(f"  ✓ {run_name}")


# ── Experiment 1: locus_visual_similarity ─────────────────────────────────────
# March 30 — establishing the cosine similarity baseline before any tuning.

print("\n[1] locus_visual_similarity")
log_run(
    "locus_visual_similarity",
    "cosine-similarity-baseline",
    params={"eval_method": "cosine_similarity", "n_queries": 30, "model": "patrickjohncyh/fashion-clip"},
    metrics={"recall_at_5": 0.700, "hit_at_5": 0.700},
    note="Baseline: raw Fashion-CLIP cosine similarity on 30 golden queries before any prompt or YOLO tuning.",
)


# ── Experiment 2: locus_search_accuracy ───────────────────────────────────────
# April 3 — iterating on CLIP prompt templates + gateway category fallback.

print("\n[2] locus_search_accuracy")
log_run(
    "locus_search_accuracy",
    "baseline-stale-crop",
    params={"eval_method": "recall_at_5", "n_queries": 30, "crop_strategy": "stale_bbox"},
    metrics={"hit_at_5": 0.800, "recall_at_5": 0.800},
    note="Baseline recall evaluation using stale bounding-box crops. hit@5=0.80.",
)
log_run(
    "locus_search_accuracy",
    "clip-prompts-v1-fix",
    params={"eval_method": "recall_at_5", "n_queries": 30, "clip_prompt_version": "v1"},
    metrics={"hit_at_5": 0.730, "recall_at_5": 0.730},
    note="Tweaked CLIP prompt templates (v1). hit@5 dropped to 0.73 — prompts overfit to label phrasing.",
)
log_run(
    "locus_search_accuracy",
    "clip-prompts-v2-plus-gateway-fallback",
    params={"eval_method": "recall_at_5", "n_queries": 30, "clip_prompt_version": "v2", "gateway_fallback": "true"},
    metrics={"hit_at_5": 0.967, "recall_at_5": 0.967},
    note="CLIP prompt v2 + added gateway category-fallback logic. hit@5 jumped to 0.967 — best result.",
)


# ── Experiment 3: locus_gemini_judge ──────────────────────────────────────────
# April 8-9 — switched evaluation metric to Gemini visual similarity score (vss5).
# Shows YOLO detector tuning and leggings category fix.

print("\n[3] locus_gemini_judge")
log_run(
    "locus_gemini_judge",
    "gemini-judge-initial",
    params={"judge_model": "gemini-2.0-flash-001", "n_queries": 30, "judge_prompt_version": "v1"},
    metrics={"avg_vss5": 0.635, "avg_top1": 0.801},
    note="First working Gemini VSS evaluation. avg_vss5=0.635, avg_top1=0.801. Establishes judge baseline.",
)
log_run(
    "locus_gemini_judge",
    "yolo-world-baseline",
    params={"judge_model": "gemini-2.0-flash-001", "n_queries": 30, "detector": "yolo-world"},
    metrics={"avg_vss5": 0.648, "cat_leggings_vss5": 0.370},
    note="YOLO-World detector integrated. avg_vss5=0.648. Per-category breakdown shows leggings worst at 0.37.",
)
log_run(
    "locus_gemini_judge",
    "yolo-world-v1-tier-analysis",
    params={"judge_model": "gemini-2.0-flash-001", "n_queries": 30, "detector": "yolo-world"},
    metrics={"avg_vss5": 0.648, "tier1_exact": 28, "tier2_alias": 1, "tier_fallback": 1},
    note="Full tier analysis: T1-exact=28/30, T2-alias=1/30, fallback=1/30. Confirms YOLO crop strategy is sound.",
)
log_run(
    "locus_gemini_judge",
    "phase1-leggings-fix",
    params={"judge_model": "gemini-2.0-flash-001", "n_queries": 30, "detector": "yolo-world"},
    metrics={"avg_vss5": 0.659, "cat_leggings_vss5": 0.712},
    note="Fixed leggings category alias in CLIP label mapping. cat/leggings_vss5: 0.37 → 0.712. avg_vss5=0.659.",
)
log_run(
    "locus_gemini_judge",
    "final-system-gt-vss",
    params={"judge_model": "gemini-2.0-flash-001", "n_queries": 30, "detector": "yolo-world", "judge_prompt_version": "v3"},
    metrics={"avg_vss5": 0.659, "avg_gt_vss": 0.735, "hit_at_5": 1.0},
    note="Final evaluation: avg_vss5=0.659, avg_gt_vss=0.735, hit@5=1.0 on 30 golden queries. System accepted.",
)

print("\nMigration complete. Open http://localhost:5000")
