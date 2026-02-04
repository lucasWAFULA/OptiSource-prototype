import sys
sys.path.append('.')
from api import run_optimization

# Test with the same data as before but let's examine the results more closely
test_payload = {
    'sources': [
        {
            'source_id': 'TEST_001',
            'features': {
                'task_success_rate': 0.8,
                'corroboration_score': 0.7,
                'report_timeliness': 0.9,
                'handler_confidence': 0.6,
                'deception_score': 0.2,
                'ci_flag': 0
            },
            'recourse_rules': {
                'rel_disengage': 0.35,
                'rel_ci_flag': 0.50,
                'dec_disengage': 0.75,
                'dec_ci_flag': 0.60
            }
        },
        {
            'source_id': 'TEST_002',
            'features': {
                'task_success_rate': 0.6,
                'corroboration_score': 0.8,
                'report_timeliness': 0.7,
                'handler_confidence': 0.8,
                'deception_score': 0.3,
                'ci_flag': 1
            },
            'recourse_rules': {
                'rel_disengage': 0.35,
                'rel_ci_flag': 0.50,
                'dec_disengage': 0.75,
                'dec_ci_flag': 0.60
            }
        }
    ]
}

try:
    from src.dashboard_integration import get_dashboard_pipeline
    pipeline = get_dashboard_pipeline()
    loaded = pipeline.load_models()
    if loaded:
        result = run_optimization(test_payload, ml_pipeline=pipeline)
        ml_policy = result.get('policies', {}).get('ml_tssp', [])

        print('ML-TSSP Policy Results:')
        escalated_count = 0
        for item in ml_policy:
            source_state = item.get('source_state', 'unknown')
            action = item.get('action', 'unknown')
            risk_bucket = item.get('risk_bucket', 'unknown')
            deception = item.get('deception', 0)
            reliability = item.get('reliability', 0)
            print(f'  {item["source_id"]}: state={source_state}, action={action}, risk={risk_bucket}, dec={deception:.3f}, rel={reliability:.3f}')
            if source_state == 'assigned_escalated':
                escalated_count += 1

        print(f'\nTotal escalated (medium risk) sources: {escalated_count}')

        # Check the optimization output
        opt_output = result.get('optimization_output', {})
        if 'escalated' in opt_output:
            print(f'Optimization output escalated count: {opt_output["escalated"]}')

    else:
        print('ML models failed to load')
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()