"""
Shared Database Module for Multi-User Workflow
Provides persistent storage for cross-user actions and state.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

# Database file path
DB_PATH = Path(__file__).parent / "shared_state.db"

# Thread lock for database operations
_db_lock = threading.Lock()


def get_db_connection():
    """Get a database connection with proper initialization."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database schema."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Sources table - shared source data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                features TEXT NOT NULL,
                recourse_rules TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Assignments table - ML-TSSP optimization results
        cursor.execute("""
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
        """)
        
        # Tasking requests - Case Officer submits, Commander approves
        cursor.execute("""
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
        """)
        
        # Recommendations - Evaluation Officer recommends disengagement
        cursor.execute("""
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
        """)
        
        # Audit log - track all actions
        cursor.execute("""
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
        """)
        
        # User actions - track pending actions per user
        cursor.execute("""
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
        """)
        
        # Optimization results - store full optimization results for cross-section sharing
        cursor.execute("""
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
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignments_source ON assignments(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignments_policy ON assignments(policy_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasking_status ON tasking_requests(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimization_active ON optimization_results(is_active, created_at)")
        
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
        
        # Batch insert using executemany
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
        log_audit(submitted_by, "case_officer", "submit_tasking", "tasking_request", str(request_id), 
                 {"source_id": source_id, "task": requested_task})
        return request_id


def approve_tasking_request(request_id: int, approved_by: str, assigned_task: str = None):
    """Tasking Coordinator approves a tasking request."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasking_requests 
            SET status = 'approved', approved_by = ?, approved_at = CURRENT_TIMESTAMP,
                assigned_task = COALESCE(?, requested_task)
            WHERE request_id = ? AND status = 'pending'
        """, (approved_by, assigned_task, request_id))
        conn.commit()
        
        # Get the source_id for audit log
        cursor.execute("SELECT source_id FROM tasking_requests WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()
        source_id = row["source_id"] if row else None
        conn.close()
        
        log_audit(approved_by, "tasking_coordinator", "approve_tasking", "tasking_request", str(request_id),
                 {"source_id": source_id, "assigned_task": assigned_task})
        return True


def reject_tasking_request(request_id: int, rejected_by: str, reason: str = None):
    """Tasking Coordinator rejects a tasking request."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE tasking_requests 
            SET status = 'rejected', approved_by = ?, approved_at = CURRENT_TIMESTAMP
            WHERE request_id = ? AND status = 'pending'
        """, (rejected_by, request_id))
        conn.commit()
        conn.close()
        log_audit(rejected_by, "tasking_coordinator", "reject_tasking", "tasking_request", str(request_id),
                 {"reason": reason})
        return True


def recommend_disengagement(source_id: str, reason: str, recommended_by: str):
    """Evaluation Officer recommends source disengagement."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recommendations (source_id, recommendation_type, reason, recommended_by)
            VALUES (?, 'disengagement', ?, ?)
        """, (source_id, reason, recommended_by))
        rec_id = cursor.lastrowid
        conn.commit()
        conn.close()
        log_audit(recommended_by, "evaluation_officer", "recommend_disengagement", "recommendation", str(rec_id),
                 {"source_id": source_id, "reason": reason})
        return rec_id


def get_pending_tasking_requests(limit: int = 50):
    """Get pending tasking requests for Tasking Coordinator review."""
    with _db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tasking_requests 
            WHERE status = 'pending'
            ORDER BY submitted_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_pending_recommendations(limit: int = 50):
    """Get pending recommendations for review."""
    with _db_lock:
        conn = get_db_connection()
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
        
        log_audit(created_by or "system", "system", "save_optimization_results", "optimization", str(result_id),
                 {"n_sources": len(sources) if sources else 0})
        return result_id


def load_latest_optimization_results():
    """
    Load the latest active optimization results from shared storage.
    Returns (results_dict, sources_list) or (None, None) if no results found.
    """
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
                return results, sources
            except (json.JSONDecodeError, TypeError) as e:
                # Invalid JSON - return None
                return None, None
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


# Initialize database on import
init_database()
