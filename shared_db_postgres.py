"""
PostgreSQL/Supabase backend for the shared database module.
Used when DATABASE_URL is set (e.g. Supabase connection string).
Provides the same API as shared_db.py for shared results across regions.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import threading

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None

DB_PATH = None  # Not used for Postgres; kept for API compatibility
DB_CONNECTED = False
DB_ENGINE = "postgres"
_db_lock = threading.Lock()


def get_db_connection():
    """Get a PostgreSQL connection (Supabase or any Postgres)."""
    global DB_CONNECTED
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 is required for PostgreSQL backend. Install with: pip install psycopg2-binary")
    try:
        # Supabase/cloud Postgres often need SSL
        if "sslmode" not in url and "supabase" in url.lower():
            url = url if "?" in url else url + "?sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
        DB_CONNECTED = True
        return conn
    except Exception as e:
        DB_CONNECTED = False
        raise RuntimeError(f"Database connection failed: {str(e)}") from e


def init_database():
    """Ensure PostgreSQL schema exists. Run scripts/init_postgres_schema.sql in Supabase for full init."""
    if not os.environ.get("DATABASE_URL") or not _PSYCOPG2_AVAILABLE:
        return
    with _db_lock:
        try:
            conn = get_db_connection()
        except Exception:
            return
        cur = conn.cursor()
        try:
            # Minimal table creation if not already run (optional; prefer running init_postgres_schema.sql)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    features TEXT NOT NULL,
                    recourse_rules TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    created_by TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS assignments (
                    assignment_id SERIAL PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    task TEXT,
                    action TEXT NOT NULL,
                    expected_risk REAL,
                    intrinsic_risk REAL,
                    risk_bucket TEXT,
                    reliability REAL,
                    deception REAL,
                    source_state TEXT,
                    score REAL,
                    policy_type TEXT DEFAULT 'ml_tssp',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    created_by TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    result_id SERIAL PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    sources_json TEXT,
                    n_sources INTEGER,
                    created_by TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    is_active INTEGER DEFAULT 1
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id SERIAL PRIMARY KEY,
                    "user" TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    details TEXT,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasking_requests (
                    request_id SERIAL PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    requested_task TEXT,
                    reason TEXT,
                    status TEXT DEFAULT 'pending',
                    submitted_by TEXT NOT NULL,
                    submitted_at TIMESTAMPTZ DEFAULT NOW(),
                    approved_by TEXT,
                    approved_at TIMESTAMPTZ,
                    assigned_task TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id SERIAL PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    reason TEXT,
                    status TEXT DEFAULT 'pending',
                    recommended_by TEXT NOT NULL,
                    recommended_at TIMESTAMPTZ DEFAULT NOW(),
                    reviewed_by TEXT,
                    reviewed_at TIMESTAMPTZ,
                    decision TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_actions (
                    action_id SERIAL PRIMARY KEY,
                    "user" TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target_entity TEXT,
                    target_id TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS threshold_settings (
                    setting_id SERIAL PRIMARY KEY,
                    threshold_type TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    approved_by TEXT,
                    approved_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(threshold_type, is_active)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS threshold_requests (
                    request_id SERIAL PRIMARY KEY,
                    threshold_type TEXT NOT NULL,
                    requested_value REAL NOT NULL,
                    current_value REAL,
                    reason TEXT,
                    status TEXT DEFAULT 'pending',
                    requested_by TEXT NOT NULL,
                    requested_at TIMESTAMPTZ DEFAULT NOW(),
                    approved_by TEXT,
                    approved_at TIMESTAMPTZ,
                    rejected_by TEXT,
                    rejected_at TIMESTAMPTZ,
                    rejection_reason TEXT
                )
            """)
            conn.commit()
        finally:
            cur.close()
            conn.close()


def log_audit(user: str, role: str, action: str, entity_type: str = None, entity_id: str = None, details: Dict = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log ("user", role, action, entity_type, entity_id, details)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user, role, action, entity_type, entity_id, json.dumps(details) if details else None))
        conn.commit()
        conn.close()


def save_source(source_id: str, features: Dict, recourse_rules: Dict = None, created_by: str = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sources (source_id, features, recourse_rules, created_by, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (source_id) DO UPDATE SET
                features = EXCLUDED.features,
                recourse_rules = EXCLUDED.recourse_rules,
                created_by = EXCLUDED.created_by,
                updated_at = NOW()
        """, (source_id, json.dumps(features), json.dumps(recourse_rules) if recourse_rules else None, created_by))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "save_source", "source", source_id, {"features": features})


def save_assignment(source_id: str, assignment: Dict, policy_type: str = "ml_tssp", created_by: str = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO assignments (
                source_id, task, action, expected_risk, intrinsic_risk, risk_bucket,
                reliability, deception, source_state, score, policy_type, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            source_id,
            assignment.get("task"),
            assignment.get("action"),
            assignment.get("expected_risk"),
            assignment.get("intrinsic_risk"),
            assignment.get("risk_bucket"),
            assignment.get("reliability"),
            assignment.get("deception"),
            assignment.get("source_state"),
            assignment.get("score"),
            policy_type,
            created_by
        ))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "save_assignment", "assignment", source_id, {"policy": policy_type})


def batch_save_sources(sources_data: List[tuple], created_by: str = None):
    if not sources_data:
        return
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        for source_id, features, recourse_rules in sources_data:
            cur.execute("""
                INSERT INTO sources (source_id, features, recourse_rules, created_by, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (source_id) DO UPDATE SET
                    features = EXCLUDED.features,
                    recourse_rules = EXCLUDED.recourse_rules,
                    created_by = EXCLUDED.created_by,
                    updated_at = NOW()
            """, (source_id, json.dumps(features), json.dumps(recourse_rules) if recourse_rules else None, created_by))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "batch_save_sources", "source", "batch", {"n_sources": len(sources_data)})


def batch_save_assignments(assignments_data: List[tuple], policy_type: str = "ml_tssp", created_by: str = None):
    if not assignments_data:
        return
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        for source_id, assignment in assignments_data:
            cur.execute("""
                INSERT INTO assignments (
                    source_id, task, action, expected_risk, intrinsic_risk, risk_bucket,
                    reliability, deception, source_state, score, policy_type, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                source_id,
                assignment.get("task"),
                assignment.get("action"),
                assignment.get("expected_risk"),
                assignment.get("intrinsic_risk"),
                assignment.get("risk_bucket"),
                assignment.get("reliability"),
                assignment.get("deception"),
                assignment.get("source_state"),
                assignment.get("score"),
                policy_type,
                created_by
            ))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "batch_save_assignments", "assignment", "batch",
                  {"n_assignments": len(assignments_data), "policy": policy_type})


def submit_tasking_request(source_id: str, requested_task: str, reason: str, submitted_by: str):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO tasking_requests (source_id, requested_task, reason, submitted_by)
            VALUES (%s, %s, %s, %s)
            RETURNING request_id
        """, (source_id, requested_task, reason, submitted_by))
        row = cur.fetchone()
        request_id = row["request_id"] if row else None
        conn.commit()
        conn.close()
        if request_id is not None:
            log_audit(submitted_by, "case_officer", "submit_tasking", "tasking_request", str(request_id),
                      {"source_id": source_id, "task": requested_task})
        return request_id


def get_user_tasking_requests(username: str, status: str = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        if status:
            cur.execute("""
                SELECT * FROM tasking_requests
                WHERE submitted_by = %s AND status = %s
                ORDER BY submitted_at DESC
            """, (username, status))
        else:
            cur.execute("""
                SELECT * FROM tasking_requests
                WHERE submitted_by = %s
                ORDER BY submitted_at DESC
            """, (username,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_latest_assignments(source_id: str = None, policy_type: str = "ml_tssp", limit: int = 100):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        if source_id:
            cur.execute("""
                SELECT * FROM assignments
                WHERE source_id = %s AND policy_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (source_id, policy_type, limit))
        else:
            cur.execute("""
                SELECT * FROM assignments
                WHERE policy_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (policy_type, limit))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_audit_log(user: str = None, role: str = None, limit: int = 100):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        if user:
            cur.execute("""
                SELECT * FROM audit_log
                WHERE "user" = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user, limit))
        elif role:
            cur.execute("""
                SELECT * FROM audit_log
                WHERE role = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (role, limit))
        else:
            cur.execute("""
                SELECT * FROM audit_log
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def save_optimization_results(results: Dict, sources: List[Dict] = None, created_by: str = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE optimization_results SET is_active = 0 WHERE is_active = 1")
        cur.execute("""
            INSERT INTO optimization_results (results_json, sources_json, n_sources, created_by)
            VALUES (%s, %s, %s, %s)
        """, (
            json.dumps(results),
            json.dumps(sources) if sources else None,
            len(sources) if sources else 0,
            created_by
        ))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "save_optimization_results", "optimization", "",
                  {"n_sources": len(sources) if sources else 0})
        return None  # result_id not returned for compatibility; callers use timestamp


def load_latest_optimization_results():
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT results_json, sources_json, created_at, created_by
            FROM optimization_results
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        if row:
            try:
                results = json.loads(row["results_json"])
                sources = json.loads(row["sources_json"]) if row["sources_json"] else None
                return results, sources
            except (json.JSONDecodeError, TypeError):
                return None, None
        return None, None


def get_optimization_results_timestamp():
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT created_at
            FROM optimization_results
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        return row["created_at"] if row else None


def clear_shared_db(clear_thresholds: bool = False) -> Dict[str, int]:
    tables = [
        "optimization_results", "assignments", "sources", "tasking_requests",
        "recommendations", "audit_log", "user_actions", "threshold_requests",
    ]
    if clear_thresholds:
        tables.append("threshold_settings")
    deleted_counts = {}
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                deleted_counts[table] = cur.fetchone()["count"]
                cur.execute(f"DELETE FROM {table}")
            conn.commit()
        finally:
            conn.close()
    log_audit("admin", "admin", "clear_shared_db", "database", "shared_state",
              {"tables": deleted_counts, "clear_thresholds": clear_thresholds})
    return deleted_counts


def clear_optimization_results() -> Dict[str, int]:
    tables = ["optimization_results", "assignments"]
    deleted_counts = {}
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                deleted_counts[table] = cur.fetchone()["count"]
                cur.execute(f"DELETE FROM {table}")
            conn.commit()
        finally:
            conn.close()
    log_audit("admin", "admin", "clear_optimization_results", "database", "optimization", {"tables": deleted_counts})
    return deleted_counts


def submit_threshold_request(threshold_type: str, requested_value: float, current_value: float, reason: str, requested_by: str) -> int:
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO threshold_requests (threshold_type, requested_value, current_value, reason, requested_by, status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
            RETURNING request_id
        """, (threshold_type, requested_value, current_value, reason, requested_by))
        row = cur.fetchone()
        request_id = row["request_id"] if row else None
        conn.commit()
        conn.close()
        if request_id is not None:
            log_audit(requested_by, "admin", "submit_threshold_request", "threshold", threshold_type, {
                "request_id": request_id, "requested_value": requested_value,
                "current_value": current_value, "reason": reason
            })
        return request_id


def approve_threshold_request(request_id: int, approved_by: str):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT threshold_type, requested_value, current_value, requested_by
            FROM threshold_requests
            WHERE request_id = %s AND status = 'pending'
        """, (request_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return False
        threshold_type = row["threshold_type"]
        requested_value = row["requested_value"]
        requested_by = row["requested_by"]
        cur.execute("""
            UPDATE threshold_settings
            SET is_active = 0, updated_at = NOW()
            WHERE threshold_type = %s AND is_active = 1
        """, (threshold_type,))
        cur.execute("""
            INSERT INTO threshold_settings (threshold_type, threshold_value, is_active, approved_by, approved_at)
            VALUES (%s, %s, 1, %s, NOW())
        """, (threshold_type, requested_value, approved_by))
        cur.execute("""
            UPDATE threshold_requests
            SET status = 'approved', approved_by = %s, approved_at = NOW()
            WHERE request_id = %s
        """, (approved_by, request_id))
        conn.commit()
        conn.close()
        log_audit(approved_by, "oversight", "approve_threshold_request", "threshold", threshold_type, {
            "request_id": request_id, "threshold_type": threshold_type,
            "approved_value": requested_value, "original_requested_by": requested_by
        })
        return True


def reject_threshold_request(request_id: int, rejected_by: str, rejection_reason: str = None):
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT threshold_type, requested_value, requested_by
            FROM threshold_requests
            WHERE request_id = %s AND status = 'pending'
        """, (request_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return False
        threshold_type = row["threshold_type"]
        requested_by = row["requested_by"]
        cur.execute("""
            UPDATE threshold_requests
            SET status = 'rejected', rejected_by = %s, rejected_at = NOW(), rejection_reason = %s
            WHERE request_id = %s
        """, (rejected_by, rejection_reason, request_id))
        conn.commit()
        conn.close()
        log_audit(rejected_by, "oversight", "reject_threshold_request", "threshold", threshold_type, {
            "request_id": request_id, "threshold_type": threshold_type,
            "rejection_reason": rejection_reason, "original_requested_by": requested_by
        })
        return True


def get_active_threshold_settings() -> Dict[str, float]:
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT threshold_type, threshold_value
            FROM threshold_settings
            WHERE is_active = 1
        """)
        rows = cur.fetchall()
        conn.close()
        return {r["threshold_type"]: r["threshold_value"] for r in rows}


def get_pending_threshold_requests(limit: int = 50) -> List[Dict]:
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT request_id, threshold_type, requested_value, current_value, reason, requested_by, requested_at
            FROM threshold_requests
            WHERE status = 'pending'
            ORDER BY requested_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_threshold_request_history(limit: int = 50) -> List[Dict]:
    with _db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT request_id, threshold_type, requested_value, current_value, reason, status,
                   requested_by, requested_at, approved_by, approved_at, rejected_by, rejected_at, rejection_reason
            FROM threshold_requests
            ORDER BY requested_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
