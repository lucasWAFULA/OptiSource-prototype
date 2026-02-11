"""
Shared Database Module for Multi-User Workflow
Provides persistent storage for cross-user actions and state.
Uses SQLite for local database operations.
"""

import os
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

# Database file path (SQLite)
DB_PATH = Path(__file__).parent / "shared_state.db"
DB_CONNECTED = False

# Thread lock for database operations
_db_lock = threading.Lock()


def get_db_connection():
    """Get a SQLite database connection with proper initialization."""
    global DB_CONNECTED
    try:
        # FLOW CONTROL: Add timeout to prevent hanging (5 seconds)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        DB_CONNECTED = True
        return conn
    except sqlite3.OperationalError as e:
        # Database locked or other operational error
        DB_CONNECTED = False
        raise RuntimeError(f"Database connection failed: {str(e)}") from e


def init_database():
    """Initialize the database schema."""
    with _db_lock:
        try:
            conn = get_db_connection()
        except Exception:
            global DB_CONNECTED
            DB_CONNECTED = False
            raise
        cursor = conn.cursor()

        schema_statements = [
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                features TEXT NOT NULL,
                recourse_rules TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                status TEXT DEFAULT 'active'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS assignments (
                assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tasking_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                requested_task TEXT,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                submitted_by TEXT NOT NULL,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_by TEXT,
                approved_at TIMESTAMP,
                assigned_task TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                recommended_by TEXT NOT NULL,
                recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_by TEXT,
                reviewed_at TIMESTAMP,
                decision TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                role TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_entity TEXT,
                target_id TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS optimization_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                results_json TEXT NOT NULL,
                sources_json TEXT,
                n_sources INTEGER,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS threshold_settings (
                setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                threshold_type TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                is_active INTEGER DEFAULT 1,
                approved_by TEXT,
                approved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(threshold_type, is_active) ON CONFLICT REPLACE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS threshold_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                threshold_type TEXT NOT NULL,
                requested_value REAL NOT NULL,
                current_value REAL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                requested_by TEXT NOT NULL,
                requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_by TEXT,
                approved_at TIMESTAMP,
                rejected_by TEXT,
                rejected_at TIMESTAMP,
                rejection_reason TEXT
            )
            """,
        ]

        for stmt in schema_statements:
            cursor.execute(stmt)

        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_assignments_source ON assignments(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_assignments_policy ON assignments(policy_type)",
            "CREATE INDEX IF NOT EXISTS idx_tasking_status ON tasking_requests(status)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)",
            "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user)",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_optimization_active ON optimization_results(is_active, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_threshold_settings_active ON threshold_settings(is_active, threshold_type)",
            "CREATE INDEX IF NOT EXISTS idx_threshold_requests_status ON threshold_requests(status)",
        ]
        for stmt in index_statements:
            cursor.execute(stmt)

        conn.commit()
        conn.close()


def log_audit(user: str, role: str, action: str, entity_type: str = None, entity_id: str = None, details: Dict = None):
    """Log an action to the audit trail."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (user, role, action, entity_type, entity_id, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user, role, action, entity_type, entity_id, json.dumps(details) if details else None))
        conn.commit()
        conn.close()


def save_source(source_id: str, features: Dict, recourse_rules: Dict = None, created_by: str = None):
    """Save or update a source in shared storage."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sources (source_id, features, recourse_rules, created_by, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (source_id, json.dumps(features), json.dumps(recourse_rules) if recourse_rules else None, created_by))
        conn.commit()
        conn.close()
        log_audit(created_by or "system", "system", "save_source", "source", source_id, {"features": features})


def save_assignment(source_id: str, assignment: Dict, policy_type: str = "ml_tssp", created_by: str = None):
    """Save an assignment result to shared storage."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assignments (
                source_id, task, action, expected_risk, intrinsic_risk, risk_bucket,
                reliability, deception, source_state, score, policy_type, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """
    Batch save multiple sources in a single transaction (much faster).
    
    Parameters:
    -----------
    sources_data : List[tuple]
        List of (source_id, features_dict, recourse_rules_dict) tuples
    created_by : str
        Username who created these sources
    """
    if not sources_data:
        return
    
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO sources (source_id, features, recourse_rules, created_by, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            (source_id, json.dumps(features), json.dumps(recourse_rules) if recourse_rules else None, created_by)
            for source_id, features, recourse_rules in sources_data
        ])
        
        conn.commit()
        conn.close()
        
        # Single audit log entry for batch operation
        log_audit(created_by or "system", "system", "batch_save_sources", "source", "batch", 
                 {"n_sources": len(sources_data)})


def batch_save_assignments(assignments_data: List[tuple], policy_type: str = "ml_tssp", created_by: str = None):
    """
    Batch save multiple assignments in a single transaction (much faster).
    
    Parameters:
    -----------
    assignments_data : List[tuple]
        List of (source_id, assignment_dict) tuples
    policy_type : str
        Type of policy (default: "ml_tssp")
    created_by : str
        Username who created these assignments
    """
    if not assignments_data:
        return
    
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Batch insert using executemany
        cursor.executemany("""
            INSERT INTO assignments (
                source_id, task, action, expected_risk, intrinsic_risk, risk_bucket,
                reliability, deception, source_state, score, policy_type, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
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
            )
            for source_id, assignment in assignments_data
        ])
        
        conn.commit()
        conn.close()
        
        # Single audit log entry for batch operation
        log_audit(created_by or "system", "system", "batch_save_assignments", "assignment", "batch",
                 {"n_assignments": len(assignments_data), "policy": policy_type})


def submit_tasking_request(source_id: str, requested_task: str, reason: str, submitted_by: str):
    """Case Officer submits a tasking request."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tasking_requests (source_id, requested_task, reason, submitted_by)
            VALUES (?, ?, ?, ?)
        """, (source_id, requested_task, reason, submitted_by))
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        log_audit(submitted_by, "case_officer", "submit_tasking", "tasking_request", str(request_id), {"source_id": source_id, "task": requested_task})
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM recommendations 
            WHERE status = 'pending'
            ORDER BY recommended_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_user_tasking_requests(username: str, status: str = None):
    """Get tasking requests for a specific user."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        if status:
            cursor.execute("""
                SELECT * FROM tasking_requests 
                WHERE submitted_by = ? AND status = ?
                ORDER BY submitted_at DESC
            """, (username, status))
        else:
            cursor.execute("""
                SELECT * FROM tasking_requests 
                WHERE submitted_by = ?
                ORDER BY submitted_at DESC
            """, (username,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_latest_assignments(source_id: str = None, policy_type: str = "ml_tssp", limit: int = 100):
    """Get latest assignments from shared storage."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        if source_id:
            cursor.execute("""
                SELECT * FROM assignments 
                WHERE source_id = ? AND policy_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (source_id, policy_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM assignments 
                WHERE policy_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (policy_type, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_audit_log(user: str = None, role: str = None, limit: int = 100):
    """Get audit log entries."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        if user:
            cursor.execute("""
                SELECT * FROM audit_log 
                WHERE user = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user, limit))
        elif role:
            cursor.execute("""
                SELECT * FROM audit_log 
                WHERE role = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (role, limit))
        else:
            cursor.execute("""
                SELECT * FROM audit_log 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def save_optimization_results(results: Dict, sources: List[Dict] = None, created_by: str = None):
    """
    Save full optimization results to shared storage.
    This allows all sections to access the same optimization results.
    """
    # #region agent log
    import json as _json_log
    try:
        with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
            _f.write(_json_log.dumps({"location": "shared_db.py:save_optimization_results:ENTRY", "message": "Saving results", "data": {"n_sources": len(sources) if sources else 0, "has_results": results is not None, "has_policies": "policies" in results if results else False, "created_by": created_by}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D,F"}) + "\n")
    except: pass
    # #endregion
    
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Deactivate previous results
        cursor.execute("UPDATE optimization_results SET is_active = 0 WHERE is_active = 1")
        
        # Insert new results
        cursor.execute("""
            INSERT INTO optimization_results (results_json, sources_json, n_sources, created_by)
            VALUES (?, ?, ?, ?)
        """, (
            json.dumps(results),
            json.dumps(sources) if sources else None,
            len(sources) if sources else 0,
            created_by
        ))
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # #region agent log
        try:
            with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                _f.write(_json_log.dumps({"location": "shared_db.py:save_optimization_results:SUCCESS", "message": "Results saved", "data": {"result_id": result_id, "n_sources": len(sources) if sources else 0}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D,F"}) + "\n")
        except: pass
        # #endregion
        
        log_audit(created_by or "system", "system", "save_optimization_results", "optimization", str(result_id),
                 {"n_sources": len(sources) if sources else 0})
        return result_id


def load_latest_optimization_results():
    """
    Load the latest active optimization results from shared storage.
    Returns (results_dict, sources_list) or (None, None) if no results found.
    """
    # #region agent log
    import json as _json_log
    try:
        with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
            _f.write(_json_log.dumps({"location": "shared_db.py:load_latest_optimization_results:ENTRY", "message": "Loading results", "data": {}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,C"}) + "\n")
    except: pass
    # #endregion
    
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT results_json, sources_json, created_at, created_by
            FROM optimization_results
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                results = json.loads(row["results_json"])
                sources = json.loads(row["sources_json"]) if row["sources_json"] else None
                # #region agent log
                try:
                    with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                        _f.write(_json_log.dumps({"location": "shared_db.py:load_latest_optimization_results:SUCCESS", "message": "Results loaded", "data": {"has_results": results is not None, "has_sources": sources is not None, "n_sources": len(sources) if sources else 0, "created_at": row["created_at"] if row else None}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,C"}) + "\n")
                except: pass
                # #endregion
                return results, sources
            except (json.JSONDecodeError, TypeError) as e:
                # Invalid JSON - return None
                # #region agent log
                try:
                    with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                        _f.write(_json_log.dumps({"location": "shared_db.py:load_latest_optimization_results:ERROR", "message": "JSON decode error", "data": {"error": str(e)}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + "\n")
                except: pass
                # #endregion
                return None, None
        # #region agent log
        try:
            with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                _f.write(_json_log.dumps({"location": "shared_db.py:load_latest_optimization_results:NO_DATA", "message": "No results found", "data": {}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,C"}) + "\n")
        except: pass
        # #endregion
        return None, None


def get_optimization_results_timestamp():
    """Get the timestamp of the latest optimization results."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT created_at
            FROM optimization_results
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        return row["created_at"] if row else None


def clear_shared_db(clear_thresholds: bool = False) -> Dict[str, int]:
    """
    Clear shared database tables used for optimization and workflow data.

    Args:
        clear_thresholds: If True, also clears threshold settings.
    Returns:
        Dict mapping table name to rows deleted.
    """
    tables = [
        "optimization_results",
        "assignments",
        "sources",
        "tasking_requests",
        "recommendations",
        "audit_log",
        "user_actions",
        "threshold_requests",
    ]
    if clear_thresholds:
        tables.append("threshold_settings")

    deleted_counts: Dict[str, int] = {}
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                cursor.execute(f"DELETE FROM {table}")
                deleted_counts[table] = int(count)
            conn.commit()
        finally:
            conn.close()

    log_audit("admin", "admin", "clear_shared_db", "database", "shared_state",
              {"tables": deleted_counts, "clear_thresholds": clear_thresholds})
    return deleted_counts


def clear_optimization_results() -> Dict[str, int]:
    """
    Clear only optimization outputs (results + assignments).
    Keeps sources, tasking requests, audit logs, and thresholds intact.
    """
    tables = ["optimization_results", "assignments"]
    deleted_counts: Dict[str, int] = {}
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                cursor.execute(f"DELETE FROM {table}")
                deleted_counts[table] = int(count)
            conn.commit()
        finally:
            conn.close()

    log_audit("admin", "admin", "clear_optimization_results", "database", "optimization",
              {"tables": deleted_counts})
    return deleted_counts


def submit_threshold_request(threshold_type: str, requested_value: float, current_value: float, reason: str, requested_by: str) -> int:
    """Submit a threshold change request for approval."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO threshold_requests (threshold_type, requested_value, current_value, reason, requested_by, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (threshold_type, requested_value, current_value, reason, requested_by))
        request_id = cursor.lastrowid
        conn.commit()
        conn.close()
        log_audit(requested_by, "admin", "submit_threshold_request", "threshold", threshold_type, {
            "request_id": request_id,
            "requested_value": requested_value,
            "current_value": current_value,
            "reason": reason
        })
        return request_id


def approve_threshold_request(request_id: int, approved_by: str):
    """Approve a threshold change request and apply it as active setting."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get request details
        cursor.execute("""
            SELECT threshold_type, requested_value, current_value, requested_by
            FROM threshold_requests
            WHERE request_id = ? AND status = 'pending'
        """, (request_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        threshold_type = row["threshold_type"]
        requested_value = row["requested_value"]
        requested_by = row["requested_by"]
        
        # Deactivate old active setting
        cursor.execute("""
            UPDATE threshold_settings
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE threshold_type = ? AND is_active = 1
        """, (threshold_type,))
        
        # Create new active setting
        cursor.execute("""
            INSERT INTO threshold_settings (threshold_type, threshold_value, is_active, approved_by, approved_at)
            VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
        """, (threshold_type, requested_value, approved_by))
        
        # Update request status
        cursor.execute("""
            UPDATE threshold_requests
            SET status = 'approved', approved_by = ?, approved_at = CURRENT_TIMESTAMP
            WHERE request_id = ?
        """, (approved_by, request_id))
        
        conn.commit()
        conn.close()
        
        log_audit(approved_by, "oversight", "approve_threshold_request", "threshold", threshold_type, {
            "request_id": request_id,
            "threshold_type": threshold_type,
            "approved_value": requested_value,
            "original_requested_by": requested_by
        })
        return True


def reject_threshold_request(request_id: int, rejected_by: str, rejection_reason: str = None):
    """Reject a threshold change request."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT threshold_type, requested_value, requested_by
            FROM threshold_requests
            WHERE request_id = ? AND status = 'pending'
        """, (request_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        threshold_type = row["threshold_type"]
        requested_by = row["requested_by"]
        
        cursor.execute("""
            UPDATE threshold_requests
            SET status = 'rejected', rejected_by = ?, rejected_at = CURRENT_TIMESTAMP, rejection_reason = ?
            WHERE request_id = ?
        """, (rejected_by, rejection_reason, request_id))
        
        conn.commit()
        conn.close()
        
        log_audit(rejected_by, "oversight", "reject_threshold_request", "threshold", threshold_type, {
            "request_id": request_id,
            "threshold_type": threshold_type,
            "rejection_reason": rejection_reason,
            "original_requested_by": requested_by
        })
        return True


def get_active_threshold_settings() -> Dict[str, float]:
    """Get all active threshold settings."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT threshold_type, threshold_value
            FROM threshold_settings
            WHERE is_active = 1
        """)
        rows = cursor.fetchall()
        conn.close()
        return {row["threshold_type"]: row["threshold_value"] for row in rows}


def get_pending_threshold_requests(limit: int = 50) -> List[Dict]:
    """Get pending threshold change requests."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT request_id, threshold_type, requested_value, current_value, reason, requested_by, requested_at
            FROM threshold_requests
            WHERE status = 'pending'
            ORDER BY requested_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_threshold_request_history(limit: int = 50) -> List[Dict]:
    """Get threshold request history (all statuses)."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT request_id, threshold_type, requested_value, current_value, reason, status,
                   requested_by, requested_at, approved_by, approved_at, rejected_by, rejected_at, rejection_reason
            FROM threshold_requests
            ORDER BY requested_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# Set DB_ENGINE to "sqlite" for compatibility with dashboard.py
DB_ENGINE = "sqlite"

# Use PostgreSQL/Supabase when DATABASE_URL is set (shared results across regions)
if os.environ.get("DATABASE_URL"):
    try:
        from shared_db_postgres import (
            get_db_connection,
            init_database,
            log_audit,
            save_source,
            save_assignment,
            batch_save_sources,
            batch_save_assignments,
            submit_tasking_request,
            get_user_tasking_requests,
            get_latest_assignments,
            get_audit_log,
            save_optimization_results,
            load_latest_optimization_results,
            get_optimization_results_timestamp,
            clear_shared_db,
            clear_optimization_results,
            submit_threshold_request,
            approve_threshold_request,
            reject_threshold_request,
            get_active_threshold_settings,
            get_pending_threshold_requests,
            get_threshold_request_history,
            DB_ENGINE,
            DB_CONNECTED,
            DB_PATH,
        )
    except Exception:
        pass  # Fall back to SQLite if Postgres import fails

# Initialize database on import
init_database()
