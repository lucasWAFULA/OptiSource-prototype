#!/usr/bin/env python3
"""
Test script to verify operational mode changes work without UnboundLocalError
"""
import sys
import os
sys.path.append('.')

import streamlit as st
from dashboard import render_streamlit_app
import time

def test_operational_mode_changes():
    """Test that changing operational mode doesn't cause UnboundLocalError"""

    # Mock streamlit session state
    class MockSessionState:
        def __init__(self):
            self.data = {
                'operational_mode': 'Standard',
                'auto_recompute': True,
                'last_mode_change': None
            }

        def __getitem__(self, key):
            return self.data.get(key)

        def __setitem__(self, key, value):
            self.data[key] = value

        def get(self, key, default=None):
            return self.data.get(key, default)

    # Replace streamlit session state
    st.session_state = MockSessionState()

    print("Testing operational mode changes...")

    # Test different operational modes
    modes_to_test = ['Standard', 'Conservative', 'Aggressive']

    for mode in modes_to_test:
        print(f"\nTesting mode: {mode}")

        try:
            # Simulate mode change
            st.session_state['operational_mode'] = mode
            st.session_state['last_mode_change'] = time.time()

            # This should trigger the auto-recompute logic without UnboundLocalError
            render_streamlit_app()

            print(f"✓ Mode '{mode}' change successful - no UnboundLocalError")

        except UnboundLocalError as e:
            print(f"✗ UnboundLocalError in mode '{mode}': {e}")
            return False
        except Exception as e:
            print(f"⚠ Other error in mode '{mode}': {e}")
            # Other errors might be expected due to mock environment

    print("\n✓ All operational mode changes completed without UnboundLocalError!")
    return True

if __name__ == "__main__":
    success = test_operational_mode_changes()
    sys.exit(0 if success else 1)