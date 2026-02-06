#!/usr/bin/env python3
"""
Profile cuBLAS kernels for cross-problem analysis.

This script profiles NVIDIA cuBLAS GEMM operations for the same problem shapes
used in cross-problem analysis, allowing comparison with custom CUTLASS kernels.
"""

import argparse
import sys
import os
import subprocess
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Direct execution: use absolute imports
    from config_manager import ConfigManager
    from cublas_profiler import CuBLASProfiler, CuBLASPerformanceResult, create_cublas_runner_script
    from ncu_metrics_extractor import NCUMetricsExtractor, PIPE_UTILIZATION_METRICS, MEMORY_METRICS, COMPUTE_METRICS
else:
    # Module execution: use relative imports
    from .config_manager import ConfigManager
    from .cublas_profiler import CuBLASProfiler, CuBLASPerformanceResult, create_cublas_runner_script
    from .ncu_metrics_extractor import NCUMetricsExtractor, PIPE_UTILIZATION_METRICS, MEMORY_METRICS, COMPUTE_METRICS


def main():
    parser = argparse.ArgumentParser(
        description="Profile cuBLAS GEMM kernels for cross-problem analysis (using direct cuBLAS API via C++)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect performance + NCU profiles for default problems (rows 4,8,13,18,19)
  python profile_cublas.py

  # Performance only (skip NCU profiling)
  python profile_cublas.py --skip-ncu --perf-iterations 100

  # NCU profiling only (skip performance)
  python profile_cublas.py --skip-performance

  # Profile for custom problem indices
  python profile_cublas.py --problem-indices 0 1 2 3 4

  # Extract metrics immediately after profiling
  python profile_cublas.py --extract-metrics

  # Custom performance and NCU settings
  python profile_cublas.py --perf-iterations 100 --ncu-sets detailed --ncu-iterations 3

Note: This script uses the C++ cuBLAS runner (cublas_gemm_runner) for clean
      profiling without PyTorch overhead. The runner will be built automatically
      if not already compiled. Generates performance_results.csv with GFLOPS
      and timing data, matching the format from cross_problem_sweep.py.
        """
    )
    
    parser.add_argument(
        "--best-configs-csv",
        default="/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv",
        help="Path to CSV with best configurations per problem"
    )
    
    parser.add_argument(
        "--problem-indices",
        type=int,
        nargs="+",
        default=[3, 7, 12, 17, 18],
        help="Problem indices to profile (0-indexed, default: 3 7 12 17 18 for rows 4,8,13,18,19)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="cublas_ncu_profiles",
        help="Output directory for NCU profiles (default: cublas_ncu_profiles)"
    )
    
    parser.add_argument(
        "--ncu-sets",
        default="full",
        help="NCU metric sets to collect (default: full)"
    )
    
    parser.add_argument(
        "--ncu-iterations",
        type=int,
        default=1,
        help="Number of NCU profiling iterations (default: 1)"
    )
    
    parser.add_argument(
        "--skip-ncu",
        action="store_true",
        help="Skip NCU profiling (performance only)"
    )
    
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance benchmarking (NCU profiling only)"
    )
    
    parser.add_argument(
        "--perf-iterations",
        type=int,
        default=50,
        help="Number of performance benchmark iterations (default: 50)"
    )
    
    parser.add_argument(
        "--perf-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for performance (default: 5)"
    )
    
    parser.add_argument(
        "--extract-metrics",
        action="store_true",
        help="Extract metrics to CSV after profiling"
    )
    
    parser.add_argument(
        "--metrics-output-dir",
        default="cublas_metrics_summary",
        help="Output directory for extracted metrics CSV (default: cublas_metrics_summary)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("cuBLAS Cross-Problem Profiling")
    print("="*80)
    
    # Load problem shapes
    print(f"\nLoading problem shapes from: {args.best_configs_csv}")
    config_manager = ConfigManager(args.best_configs_csv)
    
    try:
        problems = config_manager.get_problem_subset(args.problem_indices)
    except (IndexError, ValueError) as e:
        print(f"Error loading problems: {e}")
        return 1
    
    print(f"Loaded {len(problems)} problem shapes:")
    for i, p in enumerate(problems):
        print(f"  [{i}] {p}")
    
    # Setup cuBLAS runner (C++ direct API by default)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_runner_exe = os.path.join(script_dir, "cublas_gemm_runner")
    cpp_runner_src = os.path.join(script_dir, "cublas_gemm_runner.cu")
    makefile_path = os.path.join(script_dir, "Makefile")
    
    # Check if C++ runner exists, build if not
    if not os.path.exists(cpp_runner_exe):
        print(f"\nC++ cuBLAS runner not found at: {cpp_runner_exe}")
        print("Building C++ runner...")
        
        if not os.path.exists(cpp_runner_src):
            print(f"Error: cublas_gemm_runner.cu not found at {cpp_runner_src}")
            print("Please ensure cublas_gemm_runner.cu and Makefile are in place.")
            return 1
        
        if not os.path.exists(makefile_path):
            print(f"Error: Makefile not found at {makefile_path}")
            print("Please ensure cublas_gemm_runner.cu and Makefile are in place.")
            return 1
        
        # Build using Makefile
        try:
            result = subprocess.run(
                ["make", "cublas_gemm_runner"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ C++ runner built successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error building C++ runner: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return 1
    else:
        print(f"\n✓ Using C++ cuBLAS runner: {cpp_runner_exe}")
    
    # Create profiler with C++ runner
    profiler = CuBLASProfiler(
        cublas_runner_script=cpp_runner_exe,
        output_dir=args.output_dir,
        ncu_sets=args.ncu_sets,
        ncu_iterations=args.ncu_iterations,
        use_cpp_runner=True  # Use direct cuBLAS API via C++
    )
    
    # Collect performance data
    perf_results = []
    if not args.skip_performance:
        print("\n" + "="*80)
        print("PERFORMANCE COLLECTION")
        print("="*80 + "\n")
        
        perf_results = profiler.run_performance_problems(
            problems,
            iterations=args.perf_iterations,
            warmup=args.perf_warmup,
            verbose=True
        )
        
        # Save performance results to CSV
        perf_csv_path = os.path.join(args.output_dir, "performance_results.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(perf_csv_path, 'w', newline='') as f:
            if perf_results:
                fieldnames = list(perf_results[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in perf_results:
                    writer.writerow(result.to_dict())
        
        print(f"✓ Performance results saved to: {perf_csv_path}\n")
    
    # Profile all problems with NCU
    ncu_results = []
    if not args.skip_ncu:
        print("="*80)
        print("NCU PROFILING")
        print("="*80 + "\n")
        
        ncu_results = profiler.profile_problems(problems, verbose=True)
    
    # Check for failures
    failures = [r for r in ncu_results if not r.success]
    if failures:
        print("\nFailed profiles:")
        for f in failures:
            print(f"  {f.problem}: {f.error_message}")
    
    # Extract metrics if requested
    if args.extract_metrics:
        print("\n" + "="*80)
        print("Extracting NCU Metrics")
        print("="*80 + "\n")
        
        all_metrics = PIPE_UTILIZATION_METRICS + MEMORY_METRICS + COMPUTE_METRICS
        extractor = NCUMetricsExtractor(args.output_dir, args.metrics_output_dir)
        
        csv_path = extractor.extract_metrics_from_all_profiles(
            all_metrics,
            "cublas_comprehensive_metrics.csv"
        )
        
        if csv_path:
            print(f"\n✓ Metrics extracted to: {csv_path}")
        else:
            print("\n✗ Failed to extract metrics")
    
    print("\n" + "="*80)
    print("cuBLAS Profiling Complete!")
    print("="*80)
    
    if not args.skip_performance:
        print(f"\nPerformance results saved to: {args.output_dir}/performance_results.csv")
    
    print(f"NCU profiles saved to: {args.output_dir}/")
    
    if args.extract_metrics:
        print(f"Metrics CSV saved to: {args.metrics_output_dir}/")
    else:
        print(f"\nTo extract metrics, run:")
        print(f"  python extract_ncu_metrics.py --ncu-dir {args.output_dir} --metric-set all")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
