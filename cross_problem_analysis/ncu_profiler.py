"""
ncu_profiler.py

Wrapper for NVIDIA Nsight Compute (NCU) profiling.
Collects hardware counter profiles for kernel executions.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

try:
    from .config_manager import ProblemShape, KernelConfig
except ImportError:
    from config_manager import ProblemShape, KernelConfig


@dataclass
class NCUProfileResult:
    """Result from NCU profiling"""
    problem: ProblemShape
    config: KernelConfig
    success: bool
    output_file: Path | None
    error: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
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
            "output_file": str(self.output_file) if self.output_file else None,
            "error": self.error,
        }


class NCUProfiler:
    """
    Profiles kernel configurations using NVIDIA Nsight Compute.
    """

    def __init__(
        self,
        run_script_path: Path,
        output_dir: Path,
        ncu_binary: str = "ncu",
        ncu_sets: Optional[List[str]] = None,
        ncu_metrics: Optional[List[str]] = None,
        iterations: int = 2,
        warmup: int = 1,
    ):
        """
        Initialize the NCU profiler.

        Args:
            run_script_path: Path to run_one_config.py script
            output_dir: Directory to save NCU report files
            ncu_binary: Path or name of ncu executable
            ncu_sets: NCU metric sets to collect (e.g., ["full", "launch", "memory"])
            ncu_metrics: Specific NCU metrics to collect
            iterations: Number of iterations to profile
            warmup: Number of warmup iterations
        Note: Correctness checking is always enabled for NCU profiling
        """
        self.run_script_path = run_script_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ncu_binary = ncu_binary
        self.ncu_sets = ncu_sets or ["full"]
        self.ncu_metrics = ncu_metrics
        self.iterations = iterations
        self.warmup = warmup

        # Check if NCU is available
        self._check_ncu_available()

    def _check_ncu_available(self):
        """Check if NCU is available on the system"""
        try:
            result = subprocess.run(
                [self.ncu_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"NCU not available: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"NCU binary '{self.ncu_binary}' not found or not executable") from e

    def _build_ncu_command(
        self,
        problem: ProblemShape,
        config: KernelConfig,
        output_file: Path,
    ) -> List[str]:
        """Build the NCU command line"""
        # Base NCU command
        cmd = [
            self.ncu_binary,
            "--export", str(output_file),
            "--force-overwrite",
        ]

        # Add metric sets
        for metric_set in self.ncu_sets:
            cmd.extend(["--set", metric_set])

        # Add specific metrics if provided
        if self.ncu_metrics:
            for metric in self.ncu_metrics:
                cmd.extend(["--metrics", metric])

        # Target application and its arguments
        cmd.extend([
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
            # Note: Correctness checking enabled to ensure valid kernels
        ])

        return cmd

    def profile_single_config(
        self,
        problem: ProblemShape,
        config: KernelConfig,
        label: str | None = None,
    ) -> NCUProfileResult:
        """
        Profile a single kernel configuration with NCU.

        Args:
            problem: Problem shape
            config: Kernel configuration
            label: Optional label for the output file

        Returns:
            NCUProfileResult with profiling status and output file path
        """
        # Generate output filename
        if label:
            filename = f"{label}_{problem}_{config}"
        else:
            filename = f"{problem}_{config}"

        # Sanitize filename (remove special characters)
        filename = filename.replace("/", "_").replace(" ", "_")
        output_file = self.output_dir / f"{filename}.ncu-rep"

        # Build NCU command
        cmd = self._build_ncu_command(problem, config, output_file)

        try:
            # Run NCU with timeout (profiling can be slow)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False,
            )

            # Check if output file was created
            if output_file.exists():
                return NCUProfileResult(
                    problem=problem,
                    config=config,
                    success=True,
                    output_file=output_file,
                    error=None,
                )
            else:
                return NCUProfileResult(
                    problem=problem,
                    config=config,
                    success=False,
                    output_file=None,
                    error=f"NCU did not create output file. stderr: {result.stderr[:500]}",
                )

        except subprocess.TimeoutExpired:
            return NCUProfileResult(
                problem=problem,
                config=config,
                success=False,
                output_file=None,
                error="NCU profiling timeout (>10min)",
            )
        except Exception as e:
            return NCUProfileResult(
                problem=problem,
                config=config,
                success=False,
                output_file=None,
                error=f"NCU error: {type(e).__name__}: {e}",
            )

    def profile_cross_problem_matrix(
        self,
        problem_subset: List[ProblemShape],
        configs: List[KernelConfig],
        verbose: bool = True,
    ) -> List[NCUProfileResult]:
        """
        Profile all configs on all problems (N x N matrix) with NCU.

        Args:
            problem_subset: List of problem shapes
            configs: List of kernel configurations
            verbose: Print progress messages

        Returns:
            List of NCUProfileResults for all combinations
        """
        results = []
        total = len(problem_subset) * len(configs)
        count = 0

        for i, problem in enumerate(problem_subset):
            for j, config in enumerate(configs):
                count += 1
                label = f"prob{i}_cfg{j}"

                if verbose:
                    print(
                        f"[{count}/{total}] Profiling {problem} with config {config}...",
                        flush=True,
                    )

                result = self.profile_single_config(problem, config, label=label)
                results.append(result)

                if verbose:
                    if result.success:
                        print(f"  ✓ Profile saved: {result.output_file}")
                    else:
                        print(f"  ✗ Failed: {result.error}")

        return results
