"""
cuBLAS Profiler

This module provides functionality to profile NVIDIA cuBLAS GEMM operations
for given problem shapes and collect NCU profiles for comparison with custom kernels.
"""

import os
import subprocess
import tempfile
from typing import List, Optional
from dataclasses import dataclass

try:
    from .config_manager import ProblemShape
except ImportError:
    from config_manager import ProblemShape


@dataclass
class CuBLASProfileResult:
    """Result from a cuBLAS profile run"""
    problem: ProblemShape
    success: bool
    ncu_profile_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CuBLASPerformanceResult:
    """Result from a cuBLAS performance run"""
    problem: ProblemShape
    success: bool
    gflops: float = 0.0
    avg_time_us: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for CSV export"""
        return {
            'M': self.problem.M,
            'N': self.problem.N,
            'K': self.problem.K,
            'L': self.problem.L,
            'success': self.success,
            'gflops': self.gflops,
            'avg_time_us': self.avg_time_us,
            'error': self.error_message or ''
        }


class CuBLASProfiler:
    """Profiles cuBLAS GEMM kernels using NCU"""
    
    def __init__(
        self,
        cublas_runner_script: str,
        output_dir: str = "cublas_ncu_profiles",
        ncu_sets: str = "full",
        ncu_iterations: int = 1,
        use_cpp_runner: bool = True
    ):
        """
        Initialize the cuBLAS profiler.
        
        Args:
            cublas_runner_script: Path to Python script or C++ executable that runs cuBLAS GEMM
            output_dir: Directory to save NCU profile files
            ncu_sets: NCU metric sets to collect (e.g., "full", "detailed")
            ncu_iterations: Number of iterations for NCU profiling
            use_cpp_runner: If True, treat cublas_runner_script as C++ executable; if False, as Python script
        """
        self.cublas_runner_script = cublas_runner_script
        self.output_dir = output_dir
        self.ncu_sets = ncu_sets
        self.ncu_iterations = ncu_iterations
        self.use_cpp_runner = use_cpp_runner
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(cublas_runner_script):
            raise FileNotFoundError(f"cuBLAS runner not found: {cublas_runner_script}")
    
    def _build_profile_filename(self, problem: ProblemShape) -> str:
        """Generate NCU profile filename for a problem"""
        return f"cublas_M{problem.M}_N{problem.N}_K{problem.K}_L{problem.L}.ncu-rep"
    
    def profile_problem(
        self,
        problem: ProblemShape,
        verbose: bool = True
    ) -> CuBLASProfileResult:
        """
        Profile cuBLAS GEMM for a single problem shape.
        
        Args:
            problem: Problem shape to profile
            verbose: Whether to print progress information
            
        Returns:
            CuBLASProfileResult with profile path or error
        """
        profile_filename = self._build_profile_filename(problem)
        profile_path = os.path.join(self.output_dir, profile_filename)
        
        if verbose:
            print(f"Profiling cuBLAS for {problem}...")
        
        # Build NCU command based on runner type
        if self.use_cpp_runner:
            # C++ executable
            ncu_cmd = [
                "ncu",
                "--set", self.ncu_sets,
                "--kernel-name-base", "demangled",
                "--launch-skip", "0",
                "--launch-count", str(self.ncu_iterations),
                "-o", profile_path,
                "-f",  # Force overwrite
                self.cublas_runner_script,
                "--M", str(problem.M),
                "--N", str(problem.N),
                "--K", str(problem.K),
                "--batch", str(problem.L),
                "--iterations", "1",  # Single iteration for profiling
                "--warmup", "0"  # No warmup for profiling
            ]
        else:
            # Python script (legacy)
            ncu_cmd = [
                "ncu",
                "--set", self.ncu_sets,
                "--kernel-name-base", "demangled",
                "--launch-skip", "0",
                "--launch-count", str(self.ncu_iterations),
                "-o", profile_path,
                "-f",  # Force overwrite
                "python", self.cublas_runner_script,
                "--M", str(problem.M),
                "--N", str(problem.N),
                "--K", str(problem.K),
                "--batch", str(problem.L),
                "--iterations", "1",  # Single iteration for profiling
                "--warmup", "0"  # No warmup for profiling
            ]
        
        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"NCU profiling failed: {result.stderr}"
                if verbose:
                    print(f"  ✗ {error_msg}")
                return CuBLASProfileResult(
                    problem=problem,
                    success=False,
                    error_message=error_msg
                )
            
            # Check if profile file was created
            if not os.path.exists(profile_path):
                error_msg = "Profile file not created"
                if verbose:
                    print(f"  ✗ {error_msg}")
                return CuBLASProfileResult(
                    problem=problem,
                    success=False,
                    error_message=error_msg
                )
            
            if verbose:
                print(f"  ✓ Profile saved: {profile_filename}")
            
            return CuBLASProfileResult(
                problem=problem,
                success=True,
                ncu_profile_path=profile_path
            )
            
        except subprocess.TimeoutExpired:
            error_msg = "NCU profiling timeout (>5 minutes)"
            if verbose:
                print(f"  ✗ {error_msg}")
            return CuBLASProfileResult(
                problem=problem,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Exception during profiling: {str(e)}"
            if verbose:
                print(f"  ✗ {error_msg}")
            return CuBLASProfileResult(
                problem=problem,
                success=False,
                error_message=error_msg
            )
    
    def profile_problems(
        self,
        problems: List[ProblemShape],
        verbose: bool = True
    ) -> List[CuBLASProfileResult]:
        """
        Profile cuBLAS GEMM for multiple problem shapes.
        
        Args:
            problems: List of problem shapes to profile
            verbose: Whether to print progress information
            
        Returns:
            List of CuBLASProfileResult objects
        """
        results = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Profiling cuBLAS for {len(problems)} problem shapes")
            print(f"{'='*80}\n")
        
        for i, problem in enumerate(problems, 1):
            if verbose:
                print(f"[{i}/{len(problems)}] ", end="")
            
            result = self.profile_problem(problem, verbose=verbose)
            results.append(result)
        
        # Summary
        if verbose:
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            print(f"\n{'='*80}")
            print(f"cuBLAS Profiling Summary:")
            print(f"  Total problems:    {len(results)}")
            print(f"  Successful:        {successful}")
            print(f"  Failed:            {failed}")
            print(f"  Profile directory: {self.output_dir}")
            print(f"{'='*80}\n")
        
        return results
    
    def run_performance(
        self,
        problem: ProblemShape,
        iterations: int = 50,
        warmup: int = 5,
        verbose: bool = True
    ) -> CuBLASPerformanceResult:
        """
        Run cuBLAS GEMM for performance measurement.
        
        Args:
            problem: Problem shape to benchmark
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            verbose: Whether to print progress information
            
        Returns:
            CuBLASPerformanceResult with timing and GFLOPS
        """
        if verbose:
            print(f"Benchmarking cuBLAS for {problem}...")
        
        # Build command based on runner type
        if self.use_cpp_runner:
            # C++ executable
            cmd = [
                self.cublas_runner_script,
                "--M", str(problem.M),
                "--N", str(problem.N),
                "--K", str(problem.K),
                "--batch", str(problem.L),
                "--iterations", str(iterations),
                "--warmup", str(warmup)
            ]
        else:
            # Python script (legacy)
            cmd = [
                "python", self.cublas_runner_script,
                "--M", str(problem.M),
                "--N", str(problem.N),
                "--K", str(problem.K),
                "--batch", str(problem.L),
                "--iterations", str(iterations),
                "--warmup", str(warmup)
            ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Performance run failed: {result.stderr}"
                if verbose:
                    print(f"  ✗ {error_msg}")
                return CuBLASPerformanceResult(
                    problem=problem,
                    success=False,
                    error_message=error_msg
                )
            
            # Parse output for performance metrics
            gflops = None
            avg_time_us = None
            
            for line in result.stdout.split('\n'):
                if '[PERF] gflops:' in line:
                    gflops = float(line.split(':')[1].strip())
                elif '[PERF] avg_time_us:' in line:
                    avg_time_us = float(line.split(':')[1].strip())
            
            if gflops is None or avg_time_us is None:
                error_msg = "Could not parse performance metrics from output"
                if verbose:
                    print(f"  ✗ {error_msg}")
                return CuBLASPerformanceResult(
                    problem=problem,
                    success=False,
                    error_message=error_msg
                )
            
            if verbose:
                print(f"  ✓ {gflops:.2f} GFLOPS @ {avg_time_us:.2f} us")
            
            return CuBLASPerformanceResult(
                problem=problem,
                success=True,
                gflops=gflops,
                avg_time_us=avg_time_us
            )
            
        except subprocess.TimeoutExpired:
            error_msg = "Performance run timeout (>1 minute)"
            if verbose:
                print(f"  ✗ {error_msg}")
            return CuBLASPerformanceResult(
                problem=problem,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Exception during performance run: {str(e)}"
            if verbose:
                print(f"  ✗ {error_msg}")
            return CuBLASPerformanceResult(
                problem=problem,
                success=False,
                error_message=error_msg
            )
    
    def run_performance_problems(
        self,
        problems: List[ProblemShape],
        iterations: int = 50,
        warmup: int = 5,
        verbose: bool = True
    ) -> List[CuBLASPerformanceResult]:
        """
        Run performance benchmarks for multiple problem shapes.
        
        Args:
            problems: List of problem shapes to benchmark
            iterations: Number of benchmark iterations per problem
            warmup: Number of warmup iterations per problem
            verbose: Whether to print progress information
            
        Returns:
            List of CuBLASPerformanceResult objects
        """
        results = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"cuBLAS Performance Benchmarking")
            print(f"{'='*80}")
            print(f"Problems: {len(problems)}")
            print(f"Iterations: {iterations}")
            print(f"Warmup: {warmup}")
            print(f"{'='*80}\n")
        
        for i, problem in enumerate(problems, 1):
            if verbose:
                print(f"[{i}/{len(problems)}] ", end="")
            
            result = self.run_performance(problem, iterations, warmup, verbose=verbose)
            results.append(result)
        
        # Summary
        if verbose:
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"\n{'='*80}")
            print(f"cuBLAS Performance Summary:")
            print(f"  Total problems:    {len(results)}")
            print(f"  Successful:        {len(successful)}")
            print(f"  Failed:            {len(failed)}")
            if successful:
                avg_gflops = sum(r.gflops for r in successful) / len(successful)
                max_gflops = max(r.gflops for r in successful)
                min_gflops = min(r.gflops for r in successful)
                print(f"  Avg GFLOPS:        {avg_gflops:.2f}")
                print(f"  Max GFLOPS:        {max_gflops:.2f}")
                print(f"  Min GFLOPS:        {min_gflops:.2f}")
            print(f"{'='*80}\n")
        
        return results


def create_cublas_runner_script(output_path: str = "run_cublas_gemm.py") -> str:
    """
    Create a Python script that runs cuBLAS GEMM for profiling.
    
    Args:
        output_path: Where to save the script
        
    Returns:
        Path to the created script
    """
    script_content = '''#!/usr/bin/env python3
"""
cuBLAS GEMM Runner for NCU Profiling

This script runs a single cuBLAS GEMM operation with specified dimensions.
Used by the CuBLASProfiler for NCU profiling.
"""

import argparse
import numpy as np
import torch


def run_cublas_gemm(M, N, K, batch_size=1, iterations=1, warmup=1):
    """
    Run cuBLAS GEMM via PyTorch.
    
    Args:
        M, N, K: GEMM dimensions (C = A @ B, where A is MxK, B is KxN, C is MxN)
        batch_size: Number of batches
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
    """
    device = torch.device('cuda')
    
    # Create random matrices (FP16)
    if batch_size == 1:
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)
    else:
        A = torch.randn(batch_size, M, K, dtype=torch.float16, device=device)
        B = torch.randn(batch_size, K, N, dtype=torch.float16, device=device)
    
    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    # Timed iterations
    for _ in range(iterations):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    return C


def main():
    parser = argparse.ArgumentParser(description='Run cuBLAS GEMM for profiling')
    parser.add_argument('--M', type=int, required=True, help='M dimension')
    parser.add_argument('--N', type=int, required=True, help='N dimension')
    parser.add_argument('--K', type=int, required=True, help='K dimension')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (L dimension)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warmup iterations')
    
    args = parser.parse_args()
    
    print(f"Running cuBLAS GEMM: M={args.M}, N={args.N}, K={args.K}, L={args.batch}")
    
    result = run_cublas_gemm(
        args.M, args.N, args.K,
        batch_size=args.batch,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    print(f"Result shape: {result.shape}")
    print("cuBLAS GEMM completed successfully")


if __name__ == '__main__':
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"Created cuBLAS runner script: {output_path}")
    return output_path
