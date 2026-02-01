"""
FastAPI backend for ML-TSSP optimization.
Runs ML + optimization outside Streamlit to reduce reruns.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from uuid import uuid4
from typing import Dict, Any
import threading
import time
import os

from api import run_optimization as api_run_optimization
from src.dashboard_integration import get_dashboard_pipeline
from shared_db import save_optimization_results, batch_save_sources, batch_save_assignments, log_audit

app = FastAPI()
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

PIPELINE = None


def _load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        pipeline = get_dashboard_pipeline()
        loaded = pipeline.load_models()
        if not loaded:
            raise RuntimeError("ML models failed to load.")
        PIPELINE = pipeline


def _run_optimization_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    _load_pipeline()
    return api_run_optimization(payload, ml_pipeline=PIPELINE)


def _run_with_timeout(payload: Dict[str, Any], timeout_s: int) -> tuple[Dict[str, Any] | None, str | None]:
    result_box: Dict[str, Any] = {"result": None, "error": None}

    def _runner():
        try:
            result_box["result"] = _run_optimization_core(payload)
        except Exception as exc:
            result_box["error"] = str(exc)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        return None, f"Optimization timed out after {timeout_s}s."
    if result_box["error"]:
        raise RuntimeError(result_box["error"])
    return result_box["result"], None


@app.get("/")
def root():
    """Root endpoint - confirms API is running."""
    return {"message": "ML-TSSP FastAPI backend is running", "docs": "/docs", "health": "/health", "optimize": "POST /optimize"}


@app.get("/health")
def health():
    try:
        _load_pipeline()
        models_loaded = PIPELINE is not None and PIPELINE.models_loaded
    except Exception:
        models_loaded = False
    return {"status": "ok", "models_loaded": models_loaded, "jobs": len(JOBS)}


@app.post("/optimize")
def submit_job(payload: Dict[str, Any], bg: BackgroundTasks):
    job_id = str(uuid4())
    timeout_s = int(payload.get("timeout_s") or os.environ.get("OPT_TIMEOUT_SECONDS", "90"))
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "result": None, "error": None, "created_at": time.time()}

    def run():
        try:
            result, timeout_error = _run_with_timeout(payload, timeout_s)
            with JOBS_LOCK:
                if timeout_error:
                    JOBS[job_id]["status"] = "timeout"
                    JOBS[job_id]["error"] = timeout_error
                    return
                JOBS[job_id]["result"] = result
                JOBS[job_id]["status"] = "done"

            # Persist results for shared access
            sources = payload.get("sources") or []
            username = payload.get("username") or "system"
            role = payload.get("role") or "system"
            try:
                save_optimization_results(result, sources, created_by=username)
                sources_map = {s.get("source_id"): s for s in sources if s.get("source_id") is not None}
                ml_policy = result.get("policies", {}).get("ml_tssp", [])
                sources_batch = []
                assignments_batch = []
                for assignment in ml_policy:
                    source_id = assignment.get("source_id")
                    if source_id:
                        src = sources_map.get(source_id)
                        if src:
                            sources_batch.append((source_id, src.get("features", {}), src.get("recourse_rules", {})))
                        assignments_batch.append((source_id, assignment))
                if sources_batch:
                    batch_save_sources(sources_batch, username)
                if assignments_batch:
                    batch_save_assignments(assignments_batch, "ml_tssp", username)
                log_audit(username, role, "run_optimization", "optimization", "batch",
                          {"n_sources": len(sources), "using_ml": result.get("_using_ml_models", False)})
            except Exception as e:
                with JOBS_LOCK:
                    JOBS[job_id]["error"] = f"Saved results with warning: {e}"
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(e)

    bg.add_task(run)
    return {"job_id": job_id}


@app.get("/optimize/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
