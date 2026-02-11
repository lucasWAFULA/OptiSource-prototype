-- ML-TSSP shared state schema for PostgreSQL / Supabase
-- Run this in Supabase SQL Editor (or any PostgreSQL client) once per project.

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    features TEXT NOT NULL,
    recourse_rules TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    status TEXT DEFAULT 'active'
);

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
);

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
);

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
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    "user" TEXT NOT NULL,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    details TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

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
);

CREATE TABLE IF NOT EXISTS optimization_results (
    result_id SERIAL PRIMARY KEY,
    results_json TEXT NOT NULL,
    sources_json TEXT,
    n_sources INTEGER,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active INTEGER DEFAULT 1
);

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
);

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
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_assignments_source ON assignments(source_id);
CREATE INDEX IF NOT EXISTS idx_assignments_policy ON assignments(policy_type);
CREATE INDEX IF NOT EXISTS idx_tasking_status ON tasking_requests(status);
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log("user");
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_optimization_active ON optimization_results(is_active, created_at);
CREATE INDEX IF NOT EXISTS idx_threshold_settings_active ON threshold_settings(is_active, threshold_type);
CREATE INDEX IF NOT EXISTS idx_threshold_requests_status ON threshold_requests(status);
