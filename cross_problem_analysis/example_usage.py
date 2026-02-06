#!/usr/bin/env python3
"""
example_usage.py

Simple example demonstrating the cross-problem analysis API.
Shows how to use the modules programmatically.
"""

from pathlib import Path
import sys

# Add the analysis package to path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager, ProblemShape
from kernel_runner import KernelRunner
from ncu_profiler import NCUProfiler
from cross_problem_sweep import CrossProblemSweep


def example_basic_sweep():
    """Example: Basic 3x3 sweep with performance only"""
    print("="*80)
    print("Example 1: Basic 3x3 Performance Sweep")
    print("="*80)
    
    # Paths
    best_configs_csv = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
    run_script = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/run_one_config.py")
    output_dir = Path("/media/datassd/sina/libgen/cross_problem_analysis/example_output_3x3")
    
    # Load configurations
    print(f"\n1. Loading configurations from {best_configs_csv}")
    config_mgr = ConfigManager(best_configs_csv)
    print(f"   Found {len(config_mgr.get_all_problems())} problems with best configs")
    
    # Select first 3 problems
    print("\n2. Selecting first 3 problems")
    problems = config_mgr.get_problem_subset([0, 1, 2])
    for i, p in enumerate(problems):
        best = config_mgr.get_best_config(p)
        print(f"   [{i}] {p}: {best.max_gflops:.2f} GFLOPS")
    
    # Initialize kernel runner (performance only)
    print("\n3. Initializing kernel runner")
    runner = KernelRunner(
        run_script_path=run_script,
        iterations=20,  # Fewer iterations for quick test
        warmup=2,
        skip_ref_check=False,  # Enable correctness checking
    )
    
    # Create sweep (no NCU profiler)
    print("\n4. Creating sweep orchestrator")
    sweep = CrossProblemSweep(
        config_manager=config_mgr,
        kernel_runner=runner,
        ncu_profiler=None,  # Skip NCU for this example
        output_dir=output_dir,
    )
    
    # Run the sweep
    print("\n5. Running sweep...")
    sweep.run_sweep(
        problem_subset=problems,
        run_performance=True,
        run_ncu=False,
        verbose=True,
    )
    
    print(f"\n✓ Results saved to {output_dir}")


def example_filtered_problems():
    """Example: Filter problems and show configurations"""
    print("\n" + "="*80)
    print("Example 2: Filtering Problems by Dimensions")
    print("="*80)
    
    best_configs_csv = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
    
    config_mgr = ConfigManager(best_configs_csv)
    
    # Filter large problems
    print("\n1. Filtering problems with M >= 2048 and N >= 2048")
    large_problems = config_mgr.filter_problems(min_m=2048, min_n=2048)
    print(f"   Found {len(large_problems)} matching problems:")
    for p in large_problems[:5]:  # Show first 5
        best = config_mgr.get_best_config(p)
        print(f"   - {p}: {best.max_gflops:.2f} GFLOPS with {best.config}")
    
    # Filter small problems
    print("\n2. Filtering problems with M <= 512")
    small_problems = config_mgr.filter_problems(max_m=512)
    print(f"   Found {len(small_problems)} matching problems:")
    for p in small_problems:
        best = config_mgr.get_best_config(p)
        print(f"   - {p}: {best.max_gflops:.2f} GFLOPS")


def example_direct_kernel_run():
    """Example: Directly run a single kernel configuration"""
    print("\n" + "="*80)
    print("Example 3: Direct Single Kernel Execution")
    print("="*80)
    
    best_configs_csv = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
    run_script = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/run_one_config.py")
    
    # Load config manager
    config_mgr = ConfigManager(best_configs_csv)
    
    # Get first problem and its best config
    problem = config_mgr.get_all_problems()[0]
    best_config = config_mgr.get_best_config(problem)
    
    print(f"\n1. Running problem: {problem}")
    print(f"   Config: {best_config.config}")
    print(f"   Expected performance: {best_config.max_gflops:.2f} GFLOPS")
    
    # Initialize runner (with correctness checking enabled)
    runner = KernelRunner(run_script_path=run_script, iterations=10, warmup=2, skip_ref_check=False)
    
    # Run the kernel
    print(f"\n2. Executing kernel...")
    result = runner.run_single_config(problem, best_config.config)
    
    # Show result
    print(f"\n3. Result:")
    if result.success:
        print(f"   ✓ Success")
        print(f"   - Elapsed: {result.elapsed_us:.2f} us")
        print(f"   - GFLOPS: {result.gflops:.2f}")
    else:
        print(f"   ✗ Failed: {result.error}")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of cross-problem analysis")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3],
        help="Run specific example (1, 2, or 3). If not specified, shows all examples.",
    )
    args = parser.parse_args()
    
    examples = {
        1: ("Basic 3x3 Performance Sweep", example_basic_sweep),
        2: ("Filtering Problems", example_filtered_problems),
        3: ("Direct Kernel Execution", example_direct_kernel_run),
    }
    
    if args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}\n")
        func()
    else:
        print("\nAvailable examples:")
        print("  1. Basic 3x3 Performance Sweep (runs kernels)")
        print("  2. Filtering Problems (info only)")
        print("  3. Direct Kernel Execution (runs one kernel)")
        print("\nUsage:")
        print("  python example_usage.py --example 1")
        print("  python example_usage.py --example 2")
        print("  python example_usage.py --example 3")
        print("\nNote: Examples 1 and 3 will execute CUDA kernels.")


if __name__ == "__main__":
    main()
