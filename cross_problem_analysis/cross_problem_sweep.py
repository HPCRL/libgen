"""
cross_problem_sweep.py

Main orchestration script for cross-problem kernel analysis.
Collects performance and NCU profiles for best configurations across problem subsets.

Usage examples:
    # Run first 5 problems
    python cross_problem_sweep.py --problem_indices 0,1,2,3,4 --output_dir results_5x5

    # Run specific problems by filtering
    python cross_problem_sweep.py --min_m 2048 --max_m 4096 --output_dir results_filtered

    # Run without NCU profiling (performance only)
    python cross_problem_sweep.py --problem_indices 0,1,2,3,4 --skip_ncu

    # Run with custom NCU metric sets
    python cross_problem_sweep.py --problem_indices 0,1,2 --ncu_sets full,memory
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(parent_dir))

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Direct execution: use absolute imports
    from config_manager import ConfigManager, ProblemShape, KernelConfig
    from kernel_runner import KernelRunner, PerformanceResult
    from ncu_profiler import NCUProfiler, NCUProfileResult
else:
    # Module execution: use relative imports
    from .config_manager import ConfigManager, ProblemShape, KernelConfig
    from .kernel_runner import KernelRunner, PerformanceResult
    from .ncu_profiler import NCUProfiler, NCUProfileResult


class CrossProblemSweep:
    """
    Orchestrates cross-problem kernel analysis.
    Runs best configs from each problem on all problems in a subset.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        kernel_runner: KernelRunner,
        ncu_profiler: NCUProfiler | None,
        output_dir: Path,
    ):
        self.config_manager = config_manager
        self.kernel_runner = kernel_runner
        self.ncu_profiler = ncu_profiler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_sweep(
        self,
        problem_subset: List[ProblemShape],
        run_performance: bool = True,
        run_ncu: bool = True,
        verbose: bool = True,
    ):
        """
        Run the complete cross-problem sweep.

        Args:
            problem_subset: List of problem shapes to analyze
            run_performance: Collect performance data
            run_ncu: Collect NCU profiles
            verbose: Print progress messages
        """
        # Get best configs for each problem in the subset
        best_configs = self.config_manager.get_all_best_configs_for_subset(problem_subset)
        configs = [bc.config for bc in best_configs]
        
        # Deduplicate configs (multiple problems may share the same best config)
        unique_configs = []
        seen_config_strs = set()
        config_mapping = {}  # Map original index to unique config index
        
        for i, config in enumerate(configs):
            config_str = str(config)
            if config_str not in seen_config_strs:
                seen_config_strs.add(config_str)
                config_mapping[i] = len(unique_configs)
                unique_configs.append(config)
            else:
                # Find which unique config this maps to
                for j, uc in enumerate(unique_configs):
                    if str(uc) == config_str:
                        config_mapping[i] = j
                        break

        # Use unique configs instead of all configs
        configs = unique_configs
        
        # Print header with counts
        n_problems = len(problem_subset)
        n_configs = len(configs)
        print(f"\n{'='*80}")
        print(f"Cross-Problem Kernel Analysis")
        print(f"{'='*80}")
        print(f"Problem subset size: {n_problems}")
        print(f"Unique configs: {n_configs}")
        print(f"Total runs: {n_problems * n_configs}")
        print(f"Performance collection: {'YES' if run_performance else 'NO'}")
        print(f"NCU profiling: {'YES' if run_ncu else 'NO'}")
        if run_performance and self.kernel_runner:
            correctness = "DISABLED" if self.kernel_runner.skip_ref_check else "ENABLED"
            print(f"Correctness checking: {correctness}")
            if self.kernel_runner.skip_ref_check:
                print("  ⚠ WARNING: Results may include incorrect kernels!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")

        if verbose:
            print("Problem subset:")
            original_configs = [bc.config for bc in best_configs]
            for i, (problem, best_config) in enumerate(zip(problem_subset, best_configs)):
                unique_idx = config_mapping[i]
                duplicate_marker = "" if i == list(config_mapping.values()).index(unique_idx) else " (duplicate config)"
                print(f"  [{i}] {problem} -> Config #{unique_idx}{duplicate_marker}")
                print(f"      Best perf: {best_config.max_gflops:.2f} GFLOPS @ {best_config.avg_us:.2f} us")
                print(f"      {best_config.config}")
            
            if len(configs) < len(original_configs):
                print(f"\n  ℹ Note: {len(original_configs)} problems, but only {len(configs)} unique configs")
                print(f"         Running {len(problem_subset)} × {len(configs)} = {len(problem_subset) * len(configs)} tests (instead of {len(problem_subset) * len(original_configs)})")
            print()

        # Run performance collection
        perf_results = []
        if run_performance:
            print("\n" + "="*80)
            print("PERFORMANCE COLLECTION")
            print("="*80 + "\n")
            perf_results = self.kernel_runner.run_cross_problem_matrix(
                problem_subset, configs, verbose=verbose
            )
            self._save_performance_results(perf_results)
            
            # Print summary of failures
            successful = [r for r in perf_results if r.success]
            failed = [r for r in perf_results if not r.success]
            if failed:
                correctness_fails = [r for r in failed if r.error and "[skip]" in r.error]
                runtime_fails = [r for r in failed if r.error and "[fail]" in r.error]
                print(f"\n⚠ Performance Summary:")
                print(f"  - Successful: {len(successful)}/{len(perf_results)}")
                print(f"  - Correctness failures: {len(correctness_fails)} (wrong results/assertions)")
                print(f"  - Runtime failures: {len(runtime_fails)} (compile/execution errors)")
                print(f"  - Other failures: {len(failed) - len(correctness_fails) - len(runtime_fails)}")
            
            print(f"\n✓ Performance results saved to {self.output_dir}/performance_results.csv")

        # Run NCU profiling
        ncu_results = []
        if run_ncu and self.ncu_profiler:
            print("\n" + "="*80)
            print("NCU PROFILING")
            print("="*80 + "\n")
            ncu_results = self.ncu_profiler.profile_cross_problem_matrix(
                problem_subset, configs, verbose=verbose
            )
            self._save_ncu_results(ncu_results)
            print(f"\n✓ NCU results saved to {self.output_dir}/ncu_results.csv")
            print(f"✓ NCU report files in {self.ncu_profiler.output_dir}")

        # Generate summary
        self._generate_summary(problem_subset, configs, best_configs, perf_results, ncu_results)
        print(f"\n✓ Summary saved to {self.output_dir}/summary.json")

        print(f"\n{'='*80}")
        print("SWEEP COMPLETE")
        print(f"{'='*80}\n")

    def _save_performance_results(self, results: List[PerformanceResult]):
        """Save performance results to CSV"""
        output_file = self.output_dir / "performance_results.csv"
        with open(output_file, "w", newline="") as f:
            if not results:
                return

            fieldnames = list(results[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

    def _save_ncu_results(self, results: List[NCUProfileResult]):
        """Save NCU profiling results to CSV"""
        output_file = self.output_dir / "ncu_results.csv"
        with open(output_file, "w", newline="") as f:
            if not results:
                return

            fieldnames = list(results[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

    def _generate_summary(
        self,
        problem_subset: List[ProblemShape],
        configs: List[KernelConfig],
        best_configs: List,
        perf_results: List[PerformanceResult],
        ncu_results: List[NCUProfileResult],
    ):
        """Generate summary JSON with analysis results"""
        summary = {
            "problem_count": len(problem_subset),
            "config_count": len(configs),
            "unique_configs": len(configs),
            "total_runs": len(problem_subset) * len(configs),
            "problems": [
                {
                    "index": i,
                    "shape": str(p),
                    "M": p.M,
                    "N": p.N,
                    "K": p.K,
                    "L": p.L,
                }
                for i, p in enumerate(problem_subset)
            ],
            "best_configs": [
                {
                    "problem_index": i,
                    "problem": str(bc.problem),
                    "config": str(bc.config),
                    "config_id": str(bc.config),  # For grouping duplicates
                    "original_gflops": bc.max_gflops,
                    "original_us": bc.avg_us,
                }
                for i, bc in enumerate(best_configs)
            ],
            "config_deduplication": {
                "total_configs": len(best_configs),
                "unique_configs": len(configs),
                "duplicates_removed": len(best_configs) - len(configs),
            },
        }

        # Add performance statistics
        if perf_results:
            successful = [r for r in perf_results if r.success]
            failed = [r for r in perf_results if not r.success]
            
            # Categorize failures
            correctness_fails = [r for r in failed if r.error and "[skip]" in r.error]
            runtime_fails = [r for r in failed if r.error and "[fail]" in r.error]
            other_fails = [r for r in failed if r not in correctness_fails and r not in runtime_fails]
            
            summary["performance"] = {
                "total_runs": len(perf_results),
                "successful": len(successful),
                "failed": len(failed),
                "correctness_failures": len(correctness_fails),
                "runtime_failures": len(runtime_fails),
                "other_failures": len(other_fails),
                "avg_gflops": sum(r.gflops for r in successful) / len(successful) if successful else 0,
                "max_gflops": max(r.gflops for r in successful) if successful else 0,
                "min_gflops": min(r.gflops for r in successful) if successful else 0,
            }

        # Add NCU statistics
        if ncu_results:
            successful_ncu = [r for r in ncu_results if r.success]
            summary["ncu_profiling"] = {
                "total_runs": len(ncu_results),
                "successful": len(successful_ncu),
                "failed": len(ncu_results) - len(successful_ncu),
            }

        output_file = self.output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)


def parse_comma_separated_ints(s: str) -> List[int]:
    """Parse comma-separated integers"""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-problem kernel analysis: run best configs on all problems"
    )

    # Input/output paths
    parser.add_argument(
        "--best_configs_csv",
        type=Path,
        default=Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv"),
        help="Path to CSV with best configs per problem",
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
        required=True,
        help="Output directory for results and NCU profiles",
    )

    # Problem subset selection
    subset_group = parser.add_mutually_exclusive_group(required=True)
    subset_group.add_argument(
        "--problem_indices",
        type=parse_comma_separated_ints,
        help="Comma-separated indices of problems to include (e.g., 0,1,2,3,4)",
    )
    subset_group.add_argument(
        "--filter_problems",
        action="store_true",
        help="Use dimension filters to select problems",
    )

    # Dimension filters (only used with --filter_problems)
    parser.add_argument("--min_m", type=int, help="Minimum M dimension")
    parser.add_argument("--max_m", type=int, help="Maximum M dimension")
    parser.add_argument("--min_n", type=int, help="Minimum N dimension")
    parser.add_argument("--max_n", type=int, help="Maximum N dimension")
    parser.add_argument("--min_k", type=int, help="Minimum K dimension")
    parser.add_argument("--max_k", type=int, help="Maximum K dimension")

    # Performance collection settings
    parser.add_argument(
        "--skip_performance",
        action="store_true",
        help="Skip performance collection (NCU only)",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip correctness checking (NOT RECOMMENDED: may collect wrong results)",
    )
    parser.add_argument(
        "--perf_iterations",
        type=int,
        default=50,
        help="Number of performance benchmark iterations",
    )
    parser.add_argument(
        "--perf_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for performance",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        help="Use cold L2 cache for performance benchmarking",
    )

    # NCU profiling settings
    parser.add_argument(
        "--skip_ncu",
        action="store_true",
        help="Skip NCU profiling (performance only)",
    )
    parser.add_argument(
        "--ncu_binary",
        type=str,
        default="ncu",
        help="Path to NCU binary",
    )
    parser.add_argument(
        "--ncu_sets",
        type=str,
        default="full",
        help="Comma-separated NCU metric sets (e.g., full,memory,launch)",
    )
    parser.add_argument(
        "--ncu_metrics",
        type=str,
        help="Comma-separated specific NCU metrics",
    )
    parser.add_argument(
        "--ncu_iterations",
        type=int,
        default=2,
        help="Number of iterations for NCU profiling",
    )
    parser.add_argument(
        "--ncu_output_dir",
        type=Path,
        help="Separate directory for NCU report files (default: output_dir/ncu_profiles)",
    )

    # General settings
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress messages",
    )

    args = parser.parse_args()

    # Load configurations
    print(f"Loading configurations from {args.best_configs_csv}...")
    config_manager = ConfigManager(args.best_configs_csv)

    # Select problem subset
    if args.problem_indices is not None:
        problem_subset = config_manager.get_problem_subset(args.problem_indices)
        print(f"Selected {len(problem_subset)} problems by indices: {args.problem_indices}")
    else:
        problem_subset = config_manager.filter_problems(
            min_m=args.min_m,
            max_m=args.max_m,
            min_n=args.min_n,
            max_n=args.max_n,
            min_k=args.min_k,
            max_k=args.max_k,
        )
        print(f"Selected {len(problem_subset)} problems by dimension filters")

    if not problem_subset:
        print("ERROR: No problems selected. Exiting.")
        sys.exit(1)

    # Initialize kernel runner
    kernel_runner = None
    if not args.skip_performance:
        kernel_runner = KernelRunner(
            run_script_path=args.run_script,
            iterations=args.perf_iterations,
            warmup=args.perf_warmup,
            skip_ref_check=args.skip_ref_check,
            use_cold_l2=args.use_cold_l2,
        )

    # Initialize NCU profiler
    ncu_profiler = None
    if not args.skip_ncu:
        ncu_output_dir = args.ncu_output_dir or (args.output_dir / "ncu_profiles")
        ncu_sets = [s.strip() for s in args.ncu_sets.split(",") if s.strip()]
        ncu_metrics = (
            [m.strip() for m in args.ncu_metrics.split(",") if m.strip()]
            if args.ncu_metrics
            else None
        )
        try:
            ncu_profiler = NCUProfiler(
                run_script_path=args.run_script,
                output_dir=ncu_output_dir,
                ncu_binary=args.ncu_binary,
                ncu_sets=ncu_sets,
                ncu_metrics=ncu_metrics,
                iterations=args.ncu_iterations,
                warmup=1,
            )
        except RuntimeError as e:
            print(f"WARNING: Could not initialize NCU profiler: {e}")
            print("Continuing without NCU profiling...")
            ncu_profiler = None

    # Create sweep orchestrator
    sweep = CrossProblemSweep(
        config_manager=config_manager,
        kernel_runner=kernel_runner,
        ncu_profiler=ncu_profiler,
        output_dir=args.output_dir,
    )

    # Run the sweep
    sweep.run_sweep(
        problem_subset=problem_subset,
        run_performance=not args.skip_performance,
        run_ncu=not args.skip_ncu and ncu_profiler is not None,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
