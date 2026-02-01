"""
Simplified FastAPI backend for ML-TSSP optimization.
Runs ML + TSSP synchronously - request blocks until optimization completes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Any, Optional

from api import run_optimization
from src.dashboard_integration import get_dashboard_pipeline

app = FastAPI(title="ML-TSSP Simple Backend", version="1.0.0")

PIPELINE = None


def _get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        pipeline = get_dashboard_pipeline()
        loaded = pipeline.load_models()
        if not loaded:
            raise RuntimeError("ML models failed to load. Ensure model files exist in models/.")
        PIPELINE = pipeline
    return PIPELINE


def _generate_demo_sources(n: int, recourse_rules: dict, seed: int = 42) -> List[Dict]:
    """Generate n synthetic demo sources."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sid = f"SRC_{i + 1:03d}"
        tsr = float(rng.uniform(0.45, 0.92))
        cor = float(rng.uniform(0.35, 0.88))
        tim = float(rng.uniform(0.40, 0.90))
        hc = float(rng.uniform(0.50, 0.90))
        dec = float(rng.uniform(0.05, 0.45))
        ci = int(rng.integers(0, 2))
        features = {
            "task_success_rate": tsr,
            "corroboration_score": cor,
            "report_timeliness": tim,
            "handler_confidence": hc,
            "deception_score": dec,
            "ci_flag": ci,
        }
        out.append({
            "source_id": sid,
            "features": features,
            "recourse_rules": recourse_rules,
        })
    return out


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sources: Optional[List[Dict]] = None
    n_sources: Optional[int] = None
    seed: int = 42
    recourse_rules: Optional[Dict[str, float]] = None


@app.get("/health")
def health():
    """Check backend and ML model status."""
    try:
        pipeline = _get_pipeline()
        models_loaded = pipeline is not None and pipeline.models_loaded
    except Exception as e:
        return {"status": "error", "models_loaded": False, "message": str(e)}
    return {"status": "ok", "models_loaded": models_loaded}


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    """
    Run ML-TSSP optimization synchronously.
    Accepts either pre-built sources or n_sources for demo data generation.
    """
    recourse_rules = request.recourse_rules or {
        "rel_disengage": 0.35,
        "rel_ci_flag": 0.50,
        "dec_disengage": 0.75,
        "dec_ci_flag": 0.60,
    }

    if request.sources:
        sources = request.sources
        for s in sources:
            if "recourse_rules" not in s or not s["recourse_rules"]:
                s["recourse_rules"] = recourse_rules
    elif request.n_sources and request.n_sources > 0:
        n = min(request.n_sources, 80)
        sources = _generate_demo_sources(n, recourse_rules, request.seed)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'sources' (list) or 'n_sources' (int) in the request body.",
        )

    pipeline = _get_pipeline()
    payload = {"sources": sources, "seed": request.seed}

    try:
        result = run_optimization(payload, ml_pipeline=pipeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
