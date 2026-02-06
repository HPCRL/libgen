#!/usr/bin/env python3
"""
test_basic.py

Basic sanity tests for the cross-problem analysis framework.
Tests module imports, data loading, and basic functionality without running kernels.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...", end=" ")
    try:
        from config_manager import ConfigManager, ProblemShape, KernelConfig, BestConfig
        from kernel_runner import KernelRunner, PerformanceResult
        from ncu_profiler import NCUProfiler, NCUProfileResult
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_config_loading():
    """Test loading configurations from CSV"""
    print("Testing config loading...", end=" ")
    try:
        from config_manager import ConfigManager
        
        csv_path = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
        
        if not csv_path.exists():
            print(f"✗ SKIP: CSV not found at {csv_path}")
            return None
        
        config_mgr = ConfigManager(csv_path)
        problems = config_mgr.get_all_problems()
        
        if len(problems) == 0:
            print("✗ FAIL: No problems loaded")
            return False
        
        # Check first problem has valid config
        first_problem = problems[0]
        best_config = config_mgr.get_best_config(first_problem)
        
        if best_config is None:
            print("✗ FAIL: No best config for first problem")
            return False
        
        if best_config.max_gflops <= 0:
            print("✗ FAIL: Invalid GFLOPS value")
            return False
        
        print(f"✓ PASS (loaded {len(problems)} problems)")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_problem_filtering():
    """Test problem filtering functionality"""
    print("Testing problem filtering...", end=" ")
    try:
        from config_manager import ConfigManager
        
        csv_path = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
        
        if not csv_path.exists():
            print("✗ SKIP: CSV not found")
            return None
        
        config_mgr = ConfigManager(csv_path)
        
        # Test filtering by indices
        subset = config_mgr.get_problem_subset([0, 1, 2])
        if len(subset) != 3:
            print(f"✗ FAIL: Expected 3 problems, got {len(subset)}")
            return False
        
        # Test filtering by dimensions
        large = config_mgr.filter_problems(min_m=2048, min_n=2048)
        if not all(p.M >= 2048 and p.N >= 2048 for p in large):
            print("✗ FAIL: Dimension filter didn't work correctly")
            return False
        
        print(f"✓ PASS (subset: 3, large: {len(large)})")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_data_structures():
    """Test data structure operations"""
    print("Testing data structures...", end=" ")
    try:
        from config_manager import ProblemShape, KernelConfig
        
        # Test ProblemShape
        p1 = ProblemShape(128, 256, 512, 1)
        p2 = ProblemShape(128, 256, 512, 1)
        p3 = ProblemShape(256, 512, 1024, 1)
        
        if p1 != p2:
            print("✗ FAIL: Equal ProblemShapes not equal")
            return False
        
        if p1 == p3:
            print("✗ FAIL: Different ProblemShapes are equal")
            return False
        
        if str(p1) != "M128_N256_K512_L1":
            print(f"✗ FAIL: Unexpected string representation: {str(p1)}")
            return False
        
        # Test KernelConfig
        cfg = KernelConfig(64, 128, 32, 3, 2, 2, 1, "m", "n", "n")
        cfg_dict = cfg.to_dict()
        
        if cfg_dict["cta_m"] != 64 or cfg_dict["stages"] != 3:
            print("✗ FAIL: KernelConfig to_dict failed")
            return False
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_runner_initialization():
    """Test kernel runner initialization"""
    print("Testing runner initialization...", end=" ")
    try:
        from kernel_runner import KernelRunner
        
        run_script = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/run_one_config.py")
        
        if not run_script.exists():
            print(f"✗ SKIP: run_one_config.py not found at {run_script}")
            return None
        
        runner = KernelRunner(
            run_script_path=run_script,
            iterations=10,
            warmup=2,
        )
        
        if runner.iterations != 10 or runner.warmup != 2:
            print("✗ FAIL: Runner configuration incorrect")
            return False
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Cross-Problem Analysis - Basic Tests")
    print("="*80 + "\n")
    
    tests = [
        test_imports,
        test_config_loading,
        test_problem_filtering,
        test_data_structures,
        test_runner_initialization,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    
    print(f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("Status: ✗ SOME TESTS FAILED")
        return 1
    elif passed == total:
        print("Status: ✓ ALL TESTS PASSED")
        return 0
    else:
        print("Status: ⚠ SOME TESTS SKIPPED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
