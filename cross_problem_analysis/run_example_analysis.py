#!/usr/bin/env python3
"""
run_example_analysis.py

Run cross-problem analysis for specified problem shapes.
Collects performance and NCU profiles for all combinations of (problems × best configs).

Usage:
    # Use default problem indices (3, 7, 12, 17, 18)
    python run_example_analysis.py
    
    # Specify custom problem indices
    python run_example_analysis.py --problem_indices 0,1,2,3,4
    
    # Skip NCU profiling (performance only)
    python run_example_analysis.py --skip_ncu
    
    # Custom output directory
    python run_example_analysis.py --output_dir my_results
"""

import argparse
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from kernel_runner import KernelRunner
from ncu_profiler import NCUProfiler
from cross_problem_sweep import CrossProblemSweep


def parse_comma_separated_ints(s: str) -> list[int]:
    """Parse comma-separated integers"""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-problem kernel analysis with configurable problem selection"
    )
    
    # Problem selection
    parser.add_argument(
        "--problem_indices",
        type=parse_comma_separated_ints,
        default=[3, 7, 12, 17, 18],
        help="Comma-separated problem indices (0-based, default: 3,7,12,17,18 for rows 4,8,13,18,19)",
    )
    
    # Paths
    parser.add_argument(
        "--best_configs_csv",
        type=Path,
        default=Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv"),
        help="Path to CSV with best configs",
    )
    parser.add_argument(
        "--run_script",
        type=Path,
        default=Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/run_one_config.py"),
        help="Path to run_one_config.py script",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/media/datassd/sina/libgen/cross_problem_analysis/results_analysis"),
        help="Output directory for results",
    )
    
    # Performance settings
    parser.add_argument(
        "--skip_performance",
        action="store_true",
        help="Skip performance collection",
    )
    parser.add_argument(
        "--perf_iterations",
        type=int,
        default=50,
        help="Performance benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--perf_warmup",
        type=int,
        default=5,
        help="Performance warmup iterations (default: 5)",
    )
    
    # NCU settings
    parser.add_argument(
        "--skip_ncu",
        action="store_true",
        help="Skip NCU profiling",
    )
    parser.add_argument(
        "--ncu_sets",
        type=str,
        default="full",
        help="NCU metric sets, comma-separated (default: full)",
    )
    parser.add_argument(
        "--ncu_iterations",
        type=int,
        default=2,
        help="NCU profiling iterations (default: 2)",
    )
    
    # Other
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"Cross-Problem Kernel Analysis")
    print("="*80)
    
    # Configuration paths
    best_configs_csv = args.best_configs_csv
    run_script = args.run_script
    output_dir = args.output_dir
    
    print(f"\n1. Loading configurations from: {best_configs_csv.name}")
    config_mgr = ConfigManager(best_configs_csv)
    
    # Select specific problems by index
    # Note: CSV row N corresponds to index N-2 (row 1=header, row 2=index 0)
    problem_indices = args.problem_indices
    problems = config_mgr.get_problem_subset(problem_indices)
    
    print(f"\n2. Selected {len(problems)} problems:")
    for i, (idx, problem) in enumerate(zip(problem_indices, problems)):
        best = config_mgr.get_best_config(problem)
        print(f"   [{i}] Row {idx+2} (index {idx}): {problem}")
        print(f"       Best: {best.max_gflops:.2f} GFLOPS @ {best.avg_us:.2f} us")
        print(f"       Config: cta={best.config.cta_m}×{best.config.cta_n}×{best.config.cta_k}, "
              f"stages={best.config.stages}, atom={best.config.atom_m}×{best.config.atom_n}×{best.config.atom_k}")
    
    # Initialize kernel runner (with correctness checking)
    run_perf = not args.skip_performance
    print(f"\n3. Initializing kernel runner")
    print(f"   Correctness checking: ENABLED")
    print(f"   Iterations: {args.perf_iterations}, Warmup: {args.perf_warmup}")
    runner = KernelRunner(
        run_script_path=run_script,
        iterations=args.perf_iterations,
        warmup=args.perf_warmup,
        skip_ref_check=False,  # Correctness checking enabled
        use_cold_l2=False,
    ) if run_perf else None
    
    # Initialize NCU profiler
    run_ncu = not args.skip_ncu
    profiler = None
    if run_ncu:
        print(f"\n4. Initializing NCU profiler")
        ncu_output_dir = output_dir / "ncu_profiles"
        ncu_sets = [s.strip() for s in args.ncu_sets.split(",") if s.strip()]
        print(f"   Metric sets: {', '.join(ncu_sets)}")
        print(f"   Iterations: {args.ncu_iterations}")
        try:
            profiler = NCUProfiler(
                run_script_path=run_script,
                output_dir=ncu_output_dir,
                ncu_binary="ncu",
                ncu_sets=ncu_sets,
                iterations=args.ncu_iterations,
                warmup=1,
            )
            print(f"   ✓ NCU available")
        except RuntimeError as e:
            print(f"   ✗ NCU not available: {e}")
            print(f"   Continuing without NCU profiling...")
            profiler = None
            run_ncu = False
    else:
        print(f"\n4. NCU profiling: SKIPPED")
    
    # Create sweep orchestrator
    print(f"\n5. Creating sweep orchestrator")
    sweep = CrossProblemSweep(
        config_manager=config_mgr,
        kernel_runner=runner,
        ncu_profiler=profiler,
        output_dir=output_dir,
    )
    
    # Run the analysis
    print(f"\n6. Starting analysis...")
    print(f"   Output directory: {output_dir}")
    print()
    
    sweep.run_sweep(
        problem_subset=problems,
        run_performance=run_perf,
        run_ncu=run_ncu,
        verbose=not args.quiet,
    )
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    if run_perf:
        print(f"  - performance_results.csv: Performance data")
        print(f"  - summary.json: Statistics and metadata")
    if run_ncu and profiler:
        print(f"  - ncu_profiles/: NCU report files (.ncu-rep)")
    print()


if __name__ == "__main__":
    main()
