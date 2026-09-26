#!/usr/bin/env python3
"""
Test Runner for Business Entity Resolution Pipeline.
Runs all unit and integration tests across the modularized codebase.
"""

import os
import sys
import unittest
import time

# Ensure proper paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))


def run_all_tests():
    """Discovers and executes all tests in tests/."""
    print("=" * 70)
    print("RUNNING ALL MODULAR PIPELINE TESTS")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(BASE_DIR, "tests"),
        pattern="test_*.py"
    )

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Tests Executed: {result.testsRun}")
    print(f"  Passed:         {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures:       {len(result.failures)}")
    print(f"  Errors:         {len(result.errors)}")
    print(f"  Total Runtime:  {elapsed:.2f}s")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
