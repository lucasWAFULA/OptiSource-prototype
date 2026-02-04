import sys
sys.path.append('.')
from api import run_optimization

# Create test data with 39 sources to see if we can reproduce the issue
sources = []
for i in range(39):
    sources.append({
        'source_id': f'TEST_{i:03d}',
        'features': {
            'task_success_rate': 0.5 + (i % 5) * 0.1,  # Vary between 0.5-0.9
            'corroboration_score': 0.4 + (i % 4) * 0.1,  # Vary between 0.4-0.7
            'report_timeliness': 0.6 + (i % 3) * 0.1,    # Vary between 0.6-0.8
            'handler_confidence': 0.5 + (i % 6) * 0.08,  # Vary between 0.5-0.9
            'deception_score': 0.2 + (i % 4) * 0.1,      # Vary between 0.2-0.5
            'ci_flag': i % 3  # 0, 1, or 2
        },
        'recourse_rules': {
            'rel_disengage': 0.35,
            'rel_ci_flag': 0.50,
            'dec_disengage': 0.75,
            'dec_ci_flag': 0.60
        }
    })

test_payload = {'sources': sources}

try:
    from src.dashboard_integration import get_dashboard_pipeline
    pipeline = get_dashboard_pipeline()
    loaded = pipeline.load_models()
    if loaded:
        result = run_optimization(test_payload, ml_pipeline=pipeline)
        ml_policy = result.get('policies', {}).get('ml_tssp', [])

        print(f'Total sources: {len(ml_policy)}')

        # Count by risk bucket
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        escalated_count = 0
        for item in ml_policy:
            risk_bucket = item.get('risk_bucket', 'unknown')
            source_state = item.get('source_state', 'unknown')
            if risk_bucket in risk_counts:
                risk_counts[risk_bucket] += 1
            if source_state == 'assigned_escalated':
                escalated_count += 1

        print(f'Risk distribution: {risk_counts}')
        print(f'Escalated (medium risk) sources: {escalated_count}')

        # Check if all medium risk sources are being escalated
        medium_sources = [item for item in ml_policy if item.get('risk_bucket') == 'medium']
        escalated_medium = [item for item in medium_sources if item.get('source_state') == 'assigned_escalated']

        print(f'Medium risk sources: {len(medium_sources)}')
        print(f'Escalated medium risk sources: {len(escalated_medium)}')

        if len(medium_sources) > 0:
            escalation_rate = len(escalated_medium) / len(medium_sources) * 100
            print(f'Escalation rate for medium risk: {escalation_rate:.1f}%')

    else:
        print('ML models failed to load')
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()