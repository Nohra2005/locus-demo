"""
retrain_clip.py — Orchestrator for the automated LoRA retraining pipeline.

Pipeline:
  1. Check trigger conditions (feedback threshold OR quality drift)
  2. Build augmented training pairs from catalog images
  3. Train LoRA adapters on fashion-CLIP vision encoder
  4. Evaluate new adapter on golden dataset (offline, no service disruption)
  5. Promote if hit@5 >= baseline - 0.02, else rollback and log rejection

This script is designed to run:
  - On a schedule (every 48h) via the mlops_retrain Docker service
  - Manually via: python retrain_clip.py --force
  - Via gateway POST /trigger-retrain endpoint

All results are logged to MLflow under experiment "locus_lora_retraining".

Usage:
  python retrain_clip.py             # respects trigger thresholds
  python retrain_clip.py --force     # skip threshold checks, always train
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_HOST          = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT          = int(os.getenv("QDRANT_PORT", 6333))
FEEDBACK_COL         = "locus_feedback"
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI", "http://20.240.203.22:5000")
GATEWAY_URL          = os.getenv("GATEWAY_URL", "http://localhost:8000")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
EXPERIMENT_NAME      = "locus_lora_retraining"

# Trigger thresholds
FEEDBACK_RATE        = float(os.getenv("FEEDBACK_RATE", 0.10))    # retrain when N% of catalog is rated
MIN_ACS_THRESHOLD    = float(os.getenv("MIN_ACS_THRESHOLD", 0.70))

# Paths
SCRIPT_DIR           = Path(__file__).parent
RUNS_DIR             = SCRIPT_DIR / "lora_runs"
PAIRS_CACHE_DIR      = SCRIPT_DIR / "pairs_cache"
VISUAL_ENGINE_MODELS = Path(os.getenv(
    "VISUAL_ENGINE_MODELS",
    SCRIPT_DIR.parent / "visual_engine" / "models"
))
# ─────────────────────────────────────────────────────────────────────────────


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _count_feedback_and_catalog() -> tuple[int, int]:
    """Return (feedback_pairs, catalog_size) from Qdrant."""
    try:
        client = _get_qdrant_client()
        from qdrant_client.http import models as qmodels
        feedback = client.count(
            collection_name=FEEDBACK_COL,
            count_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key="training_signal",
                    match=qmodels.MatchAny(any=["positive", "negative"])
                )
            ]),
            exact=True,
        ).count
        catalog = client.count(collection_name="locus_items", exact=False).count
        return feedback, catalog
    except Exception as e:
        print(f"[RETRAIN] Could not count feedback/catalog: {e}")
        return 0, 0


def _cleanup_stale_runs() -> None:
    """
    On startup: terminate any zombie RUNNING runs, then delete all KILLED runs.
    Zombies are left by container crashes — MLflow never received an end signal.
    Deleting KILLED runs keeps the experiment view clean for inspection.
    """
    import mlflow, time as _time
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return

        # 1. Terminate zombie RUNNING runs
        stale = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=100,
        )
        if stale:
            now_ms = int(_time.time() * 1000)
            for r in stale:
                client.set_terminated(r.info.run_id, status="KILLED", end_time=now_ms)
            print(f"[RETRAIN] Terminated {len(stale)} zombie RUNNING run(s)")

        # 2. Delete all KILLED runs (includes the ones we just terminated)
        killed = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'KILLED'",
            max_results=100,
        )
        for r in killed:
            client.delete_run(r.info.run_id)
        if killed:
            print(f"[RETRAIN] Deleted {len(killed)} KILLED run(s)")
    except Exception as e:
        print(f"[RETRAIN] Could not clean up stale runs: {e}")



def _check_triggers(force: bool) -> tuple[bool, str]:
    """
    Returns (should_train, reason).
    """
    if force:
        return True, "forced via --force flag"

    n_pairs, catalog_size = _count_feedback_and_catalog()
    threshold = max(1, int(catalog_size * FEEDBACK_RATE))
    print(f"[RETRAIN] Feedback pairs: {n_pairs} / catalog: {catalog_size} / threshold: {threshold} ({FEEDBACK_RATE*100:.0f}%)")

    if catalog_size > 0 and n_pairs >= threshold:
        return True, f"feedback rate met ({n_pairs}/{catalog_size} = {n_pairs/catalog_size*100:.1f}% >= {FEEDBACK_RATE*100:.0f}%)"

    # Check ACS@5 drift via latest vss_cache.json written by metrics_exporter
    vss_cache = SCRIPT_DIR / "vss_cache.json"
    if vss_cache.exists():
        try:
            with open(vss_cache) as f:
                data = json.load(f)
            acs = data.get("acs_at_5", 1.0)
            if acs < MIN_ACS_THRESHOLD:
                return True, f"ACS@5 quality drift detected ({acs:.3f} < {MIN_ACS_THRESHOLD})"
        except Exception:
            pass

    return False, f"no trigger condition met (pairs={n_pairs}, threshold={threshold})"


def run_pipeline(force: bool = False, skip_promote: bool = False) -> dict:
    """
    Full retraining pipeline. Returns a results dict with outcome and metrics.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Create experiment if it doesn't exist
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"lora_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  Locus LoRA Retraining Pipeline")
    print(f"  Run: {run_name}")
    print(f"  Time: {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # ── Step 1: Check triggers ─────────────────────────────────────────────
    should_train, reason = _check_triggers(force)
    print(f"[RETRAIN] Trigger check: {reason}")
    if not should_train:
        print("[RETRAIN] Skipping — trigger conditions not met.")
        return {"outcome": "skipped", "reason": reason}

    if not OPENROUTER_API_KEY:
        print("[RETRAIN] ERROR: OPENROUTER_API_KEY not set — cannot run judge evaluation.")
        return {"outcome": "error", "reason": "OPENROUTER_API_KEY missing"}

    # Kill any zombie runs left by previous crashes, only now that we know we'll train.
    # Moving this after the trigger check prevents the 48h scheduler from accidentally
    # killing an in-progress GitHub Actions run when it wakes up and finds no trigger.
    _cleanup_stale_runs()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_param("trigger_reason",  reason)
        mlflow.log_param("lora_r",          4)
        mlflow.log_param("lora_alpha",       8)
        mlflow.log_param("learning_rate",    1e-4)
        mlflow.log_param("model_name",       "patrickjohncyh/fashion-clip")
        mlflow.log_param("eval_method",      "judge_avg_20q_top3")
        # Track active prompt versions so regressions can be correlated with prompt changes
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "attribute_tagger"))
            sys.path.insert(0, str(Path(__file__).parent.parent / "gateway"))
            from tagger import TAGGER_PROMPT_VERSION
            from judge import JUDGE_PROMPT_VERSION
            mlflow.log_param("tagger_prompt_version", TAGGER_PROMPT_VERSION)
            mlflow.log_param("judge_prompt_version",  JUDGE_PROMPT_VERSION)
        except Exception as _e:
            print(f"[RETRAIN] Warning: could not log prompt versions: {_e}")
        # Logged so the metrics_exporter can show a progress bar in Grafana
        mlflow.log_param("max_train_steps",  int(os.getenv("MAX_TRAIN_STEPS", 300)))

        # ── Step 2: Build training pairs ──────────────────────────────────
        print("\n[RETRAIN] Step 2/5: Building training pairs...")
        import random as _random
        from build_training_pairs import build_pairs, build_accessory_cross_pairs
        manifest_path = build_pairs()

        # Merge accessory cross-product pairs (oversampled 2× for effective weight ≈2×)
        acc_client = _get_qdrant_client()
        acc_pairs  = build_accessory_cross_pairs(PAIRS_CACHE_DIR, acc_client)
        print(f"[RETRAIN] {len(acc_pairs)} accessory cross-product pairs — merging (2× oversample)")
        if acc_pairs:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest.extend(acc_pairs)
            manifest.extend(acc_pairs)  # duplicate = effective weight ≈2× without changing trainer
            _random.shuffle(manifest)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            mlflow.log_param("accessory_cross_pairs", len(acc_pairs))

        with open(manifest_path) as f:
            n_pairs = len(json.load(f))
        mlflow.log_param("training_pairs", n_pairs)

        # ── Step 3: Train ─────────────────────────────────────────────────
        print("\n[RETRAIN] Step 3/5: Training LoRA adapters...")
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        from lora_trainer import train as train_lora
        train_result   = train_lora(
            manifest_path  = str(manifest_path),
            output_dir     = str(run_dir),
            mlflow_run_id  = mlflow_run_id,
        )
        gc.collect()
        adapter_path   = train_result["adapter_path"]
        initial_recall = train_result["initial_val_recall"]
        final_recall   = train_result["final_val_recall"]
        recall_delta   = final_recall - initial_recall
        print(f"[RETRAIN] val_recall@5: initial={initial_recall:.3f}  final={final_recall:.3f}  delta={recall_delta:+.3f}")
        mlflow.log_metric("val_recall_initial", initial_recall)
        mlflow.log_metric("val_recall_final",   final_recall)
        mlflow.log_metric("val_recall_delta",   recall_delta)

        # ── Step 4: Evaluate via judge benchmark ──────────────────────────
        print("\n[RETRAIN] Step 4/5: Judge benchmark — old vs new model (20 queries × top-3)...")
        from promote_model import evaluate_with_judge, PROMOTION_DELTA
        old_avg, new_avg = evaluate_with_judge(
            new_adapter_dir            = adapter_path,
            new_visual_projection_path = str(run_dir / "visual_projection.pt"),
            api_key                    = OPENROUTER_API_KEY,
        )
        delta = new_avg - old_avg
        gc.collect()
        print(f"[RETRAIN] OLD avg={old_avg:.4f}  NEW avg={new_avg:.4f}  delta={delta:+.4f}")
        print(f"[RETRAIN] Promotion threshold: new_avg > old_avg + {PROMOTION_DELTA}")
        mlflow.log_metric("judge_old_avg", old_avg)
        mlflow.log_metric("judge_new_avg", new_avg)
        mlflow.log_metric("judge_delta",   delta)

        # ── Step 5: Promote or rollback ───────────────────────────────────
        print("\n[RETRAIN] Step 5/5: Promote / rollback decision...")
        promoted = new_avg > old_avg + PROMOTION_DELTA

        # Signal outcome to GitHub Actions so the deploy step can be gated on it
        gha_output = os.getenv("GITHUB_OUTPUT")
        if gha_output:
            with open(gha_output, "a") as _f:
                _f.write(f"promoted={'true' if promoted else 'false'}\n")

        if promoted and not skip_promote:
            from promote_model import promote_adapter
            promote_adapter(
                adapter_dir             = adapter_path,
                visual_projection_path  = str(run_dir / "visual_projection.pt"),
                mlflow_run_id           = mlflow_run_id,
            )
            mlflow.log_param("outcome", "promoted")
            mlflow.set_tag("promoted", "true")
            print(f"\n✅ PROMOTED — new adapter deployed (judge_delta={delta:+.4f})")

            # Update CI baseline to reflect the new model's performance level
            gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8000")
            print(f"\n[BASELINE] Recomputing CI baseline against {gateway_url} ...")
            try:
                import subprocess
                result_bl = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "ci_eval.py"),
                     "--gateway-url", gateway_url, "--set-baseline"],
                    capture_output=True, text=True, timeout=600,
                )
                print(result_bl.stdout)
                if result_bl.returncode == 0:
                    print("[BASELINE] Baseline updated successfully.")
                else:
                    print(f"[BASELINE] Warning: baseline update failed:\n{result_bl.stderr}")
            except Exception as e:
                print(f"[BASELINE] Warning: could not update baseline: {e}")
        elif promoted and skip_promote:
            mlflow.log_param("outcome", "promoted-pending-deploy")
            mlflow.set_tag("promoted", "pending")
            print(f"\n✅ WOULD PROMOTE — adapter ready for deployment (judge_delta={delta:+.4f})")
            print(f"   Adapter saved at: {adapter_path}")
        else:
            mlflow.log_param("outcome", "rejected")
            mlflow.set_tag("promoted", "false")
            print(f"\n❌ REJECTED — judge_delta={delta:+.4f} does not exceed threshold +{PROMOTION_DELTA}")
            print(f"   Keeping current model. Adapter saved for inspection at: {adapter_path}")

        result = {
            "outcome":            "promoted" if promoted else "rejected",
            "val_recall_initial": initial_recall,
            "val_recall_final":   final_recall,
            "val_recall_delta":   recall_delta,
            "judge_old_avg":      old_avg,
            "judge_new_avg":      new_avg,
            "judge_delta":        delta,
            "mlflow_run_id":      mlflow_run_id,
            "adapter_path":       adapter_path,
        }

    print(f"\n{'='*60}")
    print(f"  Pipeline complete — outcome: {result['outcome']}")
    print(f"{'='*60}\n")
    return result


_LAST_RUN_FILE = SCRIPT_DIR / ".last_retrain_ts"


def _schedule_loop(interval_hours: float = 48.0):
    """Run the pipeline on a schedule, respecting the interval across container restarts."""
    interval_secs = interval_hours * 3600
    print(f"[RETRAIN] Scheduler started — running every {interval_hours}h")

    while True:
        # Check how long ago the last run finished (survives container restarts)
        now = time.time()
        if _LAST_RUN_FILE.exists():
            try:
                last_ts = float(_LAST_RUN_FILE.read_text().strip())
                elapsed = now - last_ts
                if elapsed < interval_secs:
                    wait = interval_secs - elapsed
                    print(
                        f"[RETRAIN] Last run was {elapsed/3600:.1f}h ago — "
                        f"next run in {wait/3600:.1f}h. Sleeping..."
                    )
                    time.sleep(wait)
                    continue
            except Exception:
                pass  # corrupt file — just run now

        try:
            result = run_pipeline(force=False)
            print(f"[RETRAIN] Scheduled run complete: {result['outcome']}")
        except Exception as e:
            print(f"[RETRAIN] Pipeline error: {e}")

        _LAST_RUN_FILE.write_text(str(time.time()))
        print(f"[RETRAIN] Next run in {interval_hours}h. Sleeping...")
        time.sleep(interval_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locus LoRA retraining pipeline")
    parser.add_argument("--force",      action="store_true", help="Skip trigger checks")
    parser.add_argument("--no-promote", action="store_true", help="Train and evaluate but skip promotion (for CI use)")
    parser.add_argument("--schedule",   type=float, default=0,
                        help="Run on schedule every N hours (0 = run once and exit)")
    args = parser.parse_args()

    if args.schedule > 0:
        _schedule_loop(interval_hours=args.schedule)
    else:
        result = run_pipeline(force=args.force, skip_promote=args.no_promote)
        sys.exit(0 if result["outcome"] != "error" else 1)
