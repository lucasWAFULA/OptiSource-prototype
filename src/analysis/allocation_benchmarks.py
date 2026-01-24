"""
Allocation benchmarks: deterministic (greedy) and uniform source–task assignments.

Deterministic and uniform are **Stage-1-only** baselines. They do not use ML
predictions or Stage 2 (ML-based performance forecasting, recourse). Allocation
is by fixed rules or equal allocation; cost is Stage 1 only. Used to compare
TSSP allocation efficiency vs non–ML baselines.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from src.optimization import TSSPModel


def _stage1_cost_from_assignments(
    assignments: Dict[Tuple[str, str], int],
    stage1_costs: Dict[Tuple[str, str], float],
) -> Tuple[float, int]:
    """Sum Stage 1 cost over assigned (s,t) and count assignments."""
    total = 0.0
    n = 0
    for (s, t), v in assignments.items():
        if v:
            total += stage1_costs.get((s, t), 0.0)
            n += 1
    return total, n


def alloc_deterministic_greedy(
    sources: List[str],
    tasks: List[str],
    stage1_costs: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], int]:
    """
    **Stage-1-only.** Greedy deterministic allocation: assign source–task pairs
    in ascending order of Stage 1 cost (fixed rule). Each task ≥ 1 source,
    each source ≤ 1 task. No ML, no Stage 2.
    """
    pairs = [(s, t) for s in sources for t in tasks]
    pairs.sort(key=lambda st: stage1_costs.get(st, float("inf")))

    assignments: Dict[Tuple[str, str], int] = {}
    covered_tasks: set = set()
    used_sources: set = set()

    for s, t in pairs:
        if t in covered_tasks or s in used_sources:
            continue
        assignments[(s, t)] = 1
        covered_tasks.add(t)
        used_sources.add(s)
        if len(covered_tasks) == len(tasks):
            break

    for s in sources:
        for t in tasks:
            if (s, t) not in assignments:
                assignments[(s, t)] = 0
    return assignments


def alloc_uniform(
    sources: List[str],
    tasks: List[str],
) -> Dict[Tuple[str, str], int]:
    """
    **Stage-1-only.** Uniform allocation: each task → one source, round-robin.
    Task i → source[i % n_sources]. Requires n_sources ≥ n_tasks. No ML, no Stage 2.
    """
    n_s = len(sources)
    n_t = len(tasks)
    if n_s < n_t:
        raise ValueError("alloc_uniform requires n_sources >= n_tasks")

    assignments: Dict[Tuple[str, str], int] = {}
    for i, t in enumerate(tasks):
        s = sources[i % n_s]
        assignments[(s, t)] = 1
    for s in sources:
        for t in tasks:
            if (s, t) not in assignments:
                assignments[(s, t)] = 0
    return assignments


def evaluate_allocation_efficiency(
    tssp_model: TSSPModel,
    tssp_inputs: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare TSSP (Stage 1 + Stage 2, ML-driven) vs deterministic and uniform
    **Stage-1-only** baselines.

    - **TSSP**: Full model. Stage 1 + Stage 2 (ML-based recourse). Reports
      stage1_cost, stage2_cost, total_cost.
    - **Deterministic / Uniform**: Stage 1 only. Fixed rules, no ML, no Stage 2.
      stage2_cost = 0, total_cost = stage1_cost.

    Allocation efficiency is compared on **Stage 1 cost** (apples-to-apples).
    """

    sources = tssp_inputs["sources"]
    tasks = tssp_inputs["tasks"]
    stage1_costs = tssp_inputs["stage1_costs"]

    rows: List[Dict[str, Any]] = []
    tssp_stage1 = np.nan
    tssp_stage2 = np.nan
    tssp_total = np.nan
    det_stage1 = np.nan
    unif_stage1 = np.nan

    # 1. TSSP optimal (Stage 1 + Stage 2, ML)
    if tssp_model.solution and tssp_model.solution.get("status") == "optimal":
        s1 = tssp_model.get_stage1_cost()
        s2 = tssp_model.get_stage2_expected_cost()
        tot = s1 + s2
        n_assign = len([v for v in tssp_model.solution.get("assignments", {}).values() if v])
        tssp_stage1, tssp_stage2, tssp_total = s1, s2, tot
        rows.append({
            "method": "TSSP (optimal)",
            "stage1_cost": s1,
            "stage2_cost": s2,
            "total_cost": tot,
            "n_assignments": n_assign,
            "success": True,
            "stage2_only": True,
        })
    else:
        rows.append({
            "method": "TSSP (optimal)",
            "stage1_cost": np.nan,
            "stage2_cost": np.nan,
            "total_cost": np.nan,
            "n_assignments": 0,
            "success": False,
            "stage2_only": True,
        })

    # 2. Deterministic (Stage 1 only, no ML, no Stage 2)
    try:
        det_assign = alloc_deterministic_greedy(sources, tasks, stage1_costs)
        s1, n_assign = _stage1_cost_from_assignments(det_assign, stage1_costs)
        det_stage1 = s1
        rows.append({
            "method": "Deterministic (greedy)",
            "stage1_cost": s1,
            "stage2_cost": 0.0,
            "total_cost": s1,
            "n_assignments": n_assign,
            "success": True,
            "stage2_only": False,
        })
    except Exception as e:
        rows.append({
            "method": "Deterministic (greedy)",
            "stage1_cost": np.nan,
            "stage2_cost": np.nan,
            "total_cost": np.nan,
            "n_assignments": 0,
            "success": False,
            "stage2_only": False,
            "message": str(e),
        })

    # 3. Uniform (Stage 1 only, no ML, no Stage 2)
    try:
        unif_assign = alloc_uniform(sources, tasks)
        s1, n_assign = _stage1_cost_from_assignments(unif_assign, stage1_costs)
        unif_stage1 = s1
        rows.append({
            "method": "Uniform (round-robin)",
            "stage1_cost": s1,
            "stage2_cost": 0.0,
            "total_cost": s1,
            "n_assignments": n_assign,
            "success": True,
            "stage2_only": False,
        })
    except Exception as e:
        rows.append({
            "method": "Uniform (round-robin)",
            "stage1_cost": np.nan,
            "stage2_cost": np.nan,
            "total_cost": np.nan,
            "n_assignments": 0,
            "success": False,
            "stage2_only": False,
            "message": str(e),
        })

    # Relative to TSSP Stage 1 (allocation efficiency)
    det_pct = np.nan
    unif_pct = np.nan
    if np.isfinite(tssp_stage1) and tssp_stage1 > 0:
        if np.isfinite(det_stage1):
            det_pct = 100.0 * (det_stage1 - tssp_stage1) / tssp_stage1
        if np.isfinite(unif_stage1):
            unif_pct = 100.0 * (unif_stage1 - tssp_stage1) / tssp_stage1

    out: Dict[str, Any] = {
        "comparison": rows,
        "tssp_stage1": tssp_stage1,
        "tssp_stage2": tssp_stage2,
        "tssp_total": tssp_total,
        "deterministic_stage1": det_stage1,
        "uniform_stage1": unif_stage1,
        "deterministic_vs_tssp_stage1_pct": det_pct,
        "uniform_vs_tssp_stage1_pct": unif_pct,
        "plot_path": None,
        "report_path": None,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            _save_allocation_comparison_plot(rows, output_dir)
            out["plot_path"] = output_dir / "allocation_efficiency_comparison.png"
        except Exception as e:
            out["plot_error"] = str(e)
        try:
            _save_allocation_report(out, output_dir)
            out["report_path"] = output_dir / "allocation_efficiency_report.txt"
        except Exception as e:
            out["report_error"] = str(e)

    return out


def _save_allocation_comparison_plot(rows: List[Dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in rows]
    stage1_plot = [r["stage1_cost"] if r["success"] else 0.0 for r in rows]
    # Stage 2 only for TSSP; baselines are Stage-1-only
    stage2_plot = [
        (r["stage2_cost"] if r["success"] else 0.0) if r.get("stage2_only") else 0.0
        for r in rows
    ]

    x = np.arange(len(methods))
    w = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, stage1_plot, w, label="Stage 1", color="tab:blue")
    ax.bar(x, stage2_plot, w, bottom=stage1_plot, label="Stage 2 (expected, TSSP only)", color="tab:orange")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Cost")
    ax.set_title("Allocation efficiency: TSSP (Stage 1+2) vs Stage-1-only baselines (deterministic, uniform)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    p = output_dir / "allocation_efficiency_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved allocation comparison plot to {p}")


def _save_allocation_report(result: Dict[str, Any], output_dir: Path) -> None:
    lines = [
        "=" * 60,
        "ALLOCATION EFFICIENCY: TSSP vs DETERMINISTIC vs UNIFORM",
        "=" * 60,
        "",
        "TSSP: Stage 1 + Stage 2 (ML-based recourse).",
        "Deterministic / Uniform: Stage 1 only (fixed rules, no ML, no Stage 2).",
        "",
    ]
    for r in result["comparison"]:
        lines.append(f"  {r['method']}")
        s1 = r["stage1_cost"]
        s2 = r["stage2_cost"]
        tot = r["total_cost"]
        lines.append(f"    Stage 1:    {s1:.2f}" if np.isfinite(s1) else "    Stage 1:    —")
        if r.get("stage2_only", False):
            lines.append(f"    Stage 2:   {s2:.2f}" if np.isfinite(s2) else "    Stage 2:   —")
        else:
            lines.append("    Stage 2:   N/A (Stage-1-only baseline)")
        lines.append(f"    Total:     {tot:.2f}" if np.isfinite(tot) else "    Total:     —")
        lines.append(f"    Assignments: {r['n_assignments']}")
        if not r["success"] and r.get("message"):
            lines.append(f"    Message:   {r['message']}")
        lines.append("")
    lines.append("Relative to TSSP Stage 1 (allocation efficiency):")
    if np.isfinite(result.get("deterministic_vs_tssp_stage1_pct")):
        p = result["deterministic_vs_tssp_stage1_pct"]
        lines.append(f"  Deterministic: {p:+.1f}%")
    else:
        lines.append("  Deterministic: —")
    if np.isfinite(result.get("uniform_vs_tssp_stage1_pct")):
        p = result["uniform_vs_tssp_stage1_pct"]
        lines.append(f"  Uniform:       {p:+.1f}%")
    else:
        lines.append("  Uniform:       —")
    lines.append("")
    lines.append("=" * 60)

    p = output_dir / "allocation_efficiency_report.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved allocation efficiency report to {p}")
