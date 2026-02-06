"""
kernel_runner.py

Executes kernel configurations and collects performance metrics.
Provides isolated execution using subprocess for robustness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

try:
    from .config_manager import ProblemShape, KernelConfig
except ImportError:
    from config_manager import ProblemShape, KernelConfig


@dataclass
class PerformanceResult:
    """Result from a single kernel execution"""
    problem: ProblemShape
    config: KernelConfig
    success: bool
    elapsed_us: float | None
    gflops: float | None
    error: str | None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = {
            "M": self.problem.M,
            "N": self.problem.N,
            "K": self.problem.K,
            "L": self.problem.L,
            "cta_m": self.config.cta_m,
            "cta_n": self.config.cta_n,
            "cta_k": self.config.cta_k,
            "stages": self.config.stages,
            "atom_m": self.config.atom_m,
            "atom_n": self.config.atom_n,
            "atom_k": self.config.atom_k,
            "a_major": self.config.a_major,
            "b_major": self.config.b_major,
            "c_major": self.config.c_major,
            "success": self.success,
            "elapsed_us": self.elapsed_us,
            "gflops": self.gflops,
            "error": self.error,
        }
        return result


class KernelRunner:
    """
    Executes kernel configurations and collects performance data.
    Uses subprocess isolation for robustness.
    """

    def __init__(
        self,
        run_script_path: Path,
        iterations: int = 50,
        warmup: int = 5,
        skip_ref_check: bool = False,
        use_cold_l2: bool = False,
    ):
        """
        Initialize the kernel runner.

        Args:
            run_script_path: Path to run_one_config.py script
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            skip_ref_check: Skip correctness checking (default: False, checking enabled)
            use_cold_l2: Use cold L2 cache during benchmarking
        """
        self.run_script_path = run_script_path
        self.iterations = iterations
        self.warmup = warmup
        self.skip_ref_check = skip_ref_check
        self.use_cold_l2 = use_cold_l2

    def run_single_config(
        self, problem: ProblemShape, config: KernelConfig
    ) -> PerformanceResult:
        """
        Execute a single kernel configuration.

        Args:
            problem: Problem shape to run
            config: Kernel configuration to use

        Returns:
            PerformanceResult with timing and success info
        """
        # Build command line arguments
        cmd = [
            sys.executable,
            str(self.run_script_path),
            "--M", str(problem.M),
            "--N", str(problem.N),
            "--K", str(problem.K),
            "--L", str(problem.L),
            "--a_major", config.a_major,
            "--b_major", config.b_major,
            "--c_major", config.c_major,
            "--cta_m", str(config.cta_m),
            "--cta_n", str(config.cta_n),
            "--cta_k", str(config.cta_k),
            "--stages", str(config.stages),
            "--atom_m", str(config.atom_m),
            "--atom_n", str(config.atom_n),
            "--atom_k", str(config.atom_k),
            "--iters", str(self.iterations),
            "--warmup", str(self.warmup),
        ]

        # Only skip ref check if explicitly requested
        if self.skip_ref_check:
            cmd.append("--skip_ref_check")
        if self.use_cold_l2:
            cmd.append("--use_cold_l2")

        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                check=False,
            )

            # Parse JSON output
            try:
                output = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return PerformanceResult(
                    problem=problem,
                    config=config,
                    success=False,
                    elapsed_us=None,
                    gflops=None,
                    error=f"Failed to parse JSON output: {result.stdout[:200]}",
                )

            if output.get("ok"):
                elapsed_us = output["elapsed_us"]
                # Calculate GFLOPS: (2*M*N*K*L) / (elapsed_us * 1000)
                gflops = (
                    2.0 * problem.M * problem.N * problem.K * problem.L
                ) / (elapsed_us * 1000.0)
                return PerformanceResult(
                    problem=problem,
                    config=config,
                    success=True,
                    elapsed_us=elapsed_us,
                    gflops=gflops,
                    error=None,
                )
            else:
                error_msg = output.get("error", "Unknown error")
                error_kind = output.get("kind", "fail")
                # Add kind prefix for clarity (skip = assertion/correctness, fail = runtime/compile)
                full_error = f"[{error_kind}] {error_msg}" if error_kind else error_msg
                return PerformanceResult(
                    problem=problem,
                    config=config,
                    success=False,
                    elapsed_us=None,
                    gflops=None,
                    error=full_error,
                )

        except subprocess.TimeoutExpired:
            return PerformanceResult(
                problem=problem,
                config=config,
                success=False,
                elapsed_us=None,
                gflops=None,
                error="Execution timeout (>120s)",
            )
        except Exception as e:
            return PerformanceResult(
                problem=problem,
                config=config,
                success=False,
                elapsed_us=None,
                gflops=None,
                error=f"Subprocess error: {type(e).__name__}: {e}",
            )

    def run_cross_problem_matrix(
        self,
        problem_subset: List[ProblemShape],
        configs: List[KernelConfig],
        verbose: bool = True,
    ) -> List[PerformanceResult]:
        """
        Run all configs on all problems (N x N matrix).

        Args:
            problem_subset: List of problem shapes
            configs: List of kernel configurations (one per problem)
            verbose: Print progress messages

        Returns:
            List of PerformanceResults for all combinations
        """
        results = []
        total = len(problem_subset) * len(configs)
        count = 0

        for problem in problem_subset:
            for config in configs:
                count += 1
                if verbose:
                    print(
                        f"[{count}/{total}] Running {problem} with config {config}...",
                        flush=True,
                    )

                result = self.run_single_config(problem, config)
                results.append(result)

                if verbose:
                    if result.success:
                        print(
                            f"  ✓ Success: {result.elapsed_us:.2f} us, {result.gflops:.2f} GFLOPS"
                        )
                    else:
                        print(f"  ✗ Failed: {result.error}")

        return results
