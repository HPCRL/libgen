#!/usr/bin/env python3
"""
Query nvMatmulHeuristics for optimal GEMM configurations and run them
through the CuTe tensorop_gemm_tunable.py kernel.

This script:
1. Loads problem sizes from a CSV file
2. Queries nvMatmulHeuristics for top-5 configs per problem
3. Runs each config through tensorop_gemm_tunable.py
4. Compares performance against cuBLAS baseline
5. Outputs results to CSV
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

try:
    from nvMatmulHeuristics import (
        NvMatmulHeuristicsInterface,
        NvMatmulHeuristicsNvidiaGpu,
    )
    NVMMH_AVAILABLE = True
except ImportError:
    NVMMH_AVAILABLE = False
    print("WARNING: nvidia-matmul-heuristics not installed.")
    print("Install with: pip install nvidia-matmul-heuristics")


def query_nvmmh_configs(
    m: int,
    n: int,
    k: int,
    precision: str = "HSH",  # FP16 input/output, FP32 accumulator
    gpu: int = NvMatmulHeuristicsNvidiaGpu.RTX_3090,
    count: int = 5,
) -> List[Dict]:
    """
    Query nvMatmulHeuristics for optimal configurations.
    
    Returns list of dicts with keys:
        - cta_m, cta_n, cta_k: CTA tile dimensions
        - stages: Number of pipeline stages
        - splitk: Split-K factor
        - estimated_time_ms: Predicted runtime
    """
    if not NVMMH_AVAILABLE:
        return []
    
    interface = NvMatmulHeuristicsInterface()
    
    # Query configurations
    # layout: 0 = NN (col-major A, col-major B)
    #         1 = NT (col-major A, row-major B)
    #         2 = TN (row-major A, col-major B)
    #         3 = TT (row-major A, row-major B)
    # For standard GEMM C = A * B where A is MxK and B is KxN:
    # We typically use layout=1 (NT): A is col-major (M-major), B is row-major (N-major)
    layout = 1  # NT layout
    
    # Load internal discovery set for better accuracy (with layout parameter)
    interface.loadInternalDiscoverySet(layout)
    
    # Create hardware descriptor for target GPU
    hw_descriptor = interface.createHardwareDescriptor()
    interface.setHardwarePredefinedGpu(hw_descriptor, gpu)
    
    # Query configurations
    # get_with_mnk signature: (m, n, k, matmulLayout, count, hardware_descriptor)
    configs = interface.get_with_mnk(
        m, n, k,
        layout,
        count,
        hw_descriptor,
    )
    
    result = []
    for cfg_dict in configs:
        # Extract the nvmmhKernelConfiguration object from the dict
        cfg = cfg_dict['nvmmhKernelConfiguration']
        
        # Extract configuration parameters
        cta_m, cta_n, cta_k = cfg.cta[0], cfg.cta[1], cfg.cta[2]
        stages = cfg.loadStages
        splitk = cfg.splitK
        
        # Runtime is already estimated and in the dict
        estimated_time = cfg_dict.get('runtime', -1.0)
        
        result.append({
            'cta_m': cta_m,
            'cta_n': cta_n,
            'cta_k': cta_k,
            'stages': stages,
            'splitk': splitk,
            'estimated_time_ms': estimated_time * 1000.0 if estimated_time > 0 else -1.0,  # Convert to ms
        })
    
    # Clean up hardware descriptor
    interface.destroyHardwareDescriptor(hw_descriptor)
    
    return result


def run_cute_kernel(
    m: int,
    n: int,
    k: int,
    cta_m: int,
    cta_n: int,
    cta_k: int,
    stages: int,
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    a_major: str = "m",
    b_major: str = "n",
    c_major: str = "n",
    kernel_script: str = "cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm_tunable.py",
    warmup_iterations: int = 2,
    iterations: int = 100,
) -> float:
    """
    Run the CuTe kernel with specified configuration.
    
    Returns execution time in microseconds (or -1 on failure).
    """
    cmd = [
        sys.executable,
        kernel_script,
        f"--mnkl={m},{n},{k},1",
        f"--cta_tiler={cta_m},{cta_n},{cta_k}",
        f"--num_stages={stages}",
        f"--atom_layout_mnk={atom_layout_mnk[0]},{atom_layout_mnk[1]},{atom_layout_mnk[2]}",
        f"--a_major={a_major}",
        f"--b_major={b_major}",
        f"--c_major={c_major}",
        "--ab_dtype=Float16",
        "--c_dtype=Float16",
        "--acc_dtype=Float32",
        f"--warmup_iterations={warmup_iterations}",
        f"--iterations={iterations}",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR running kernel: {result.stderr}")
            return -1.0
        
        # Parse output for execution time
        for line in result.stdout.split('\n'):
            if "Average execution time:" in line:
                # Format: "Average execution time: 123.45 us"
                time_str = line.split(':')[1].strip().split()[0]
                return float(time_str)
        
        print(f"WARNING: Could not parse execution time from output")
        return -1.0
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Kernel execution timed out")
        return -1.0
    except Exception as e:
        print(f"ERROR: {e}")
        return -1.0


def load_problems_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    """
    Load problem sizes from CSV file.
    Expected format: M,N,K (with or without header)
    """
    problems = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip header or empty rows
            if not row or row[0].strip().lower() in ['m', 'problem_m']:
                continue
            try:
                m = int(row[0].strip())
                n = int(row[1].strip())
                k = int(row[2].strip())
                problems.append((m, n, k))
            except (ValueError, IndexError):
                continue
    return problems


def run_benchmark_suite(
    problems: List[Tuple[int, int, int]],
    output_csv: str,
    kernel_script: str,
    num_configs: int = 5,
    warmup_iterations: int = 2,
    iterations: int = 100,
):
    """
    Run full benchmark suite: query nvMMH and test configs.
    """
    if not NVMMH_AVAILABLE:
        print("ERROR: nvidia-matmul-heuristics not available")
        return
    
    results = []
    
    # Check if output file exists to determine if we should write header
    write_header = not Path(output_csv).exists()
    
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow([
                'M', 'N', 'K',
                'cta_M', 'cta_N', 'cta_K',
                'stages', 'splitk',
                'nvmmh_rank',
                'nvmmh_estimated_ms',
                'cute_time_us',
                'cute_gflops',
            ])
        
        for idx, (m, n, k) in enumerate(problems, 1):
            print(f"\n[{idx}/{len(problems)}] Processing problem M={m}, N={n}, K={k}")
            
            # Query nvMatmulHeuristics
            configs = query_nvmmh_configs(m, n, k, count=num_configs)
            
            if not configs:
                print(f"  No configs returned from nvMMH")
                continue
            
            print(f"  Got {len(configs)} configs from nvMatmulHeuristics")
            
            # Test each config
            for rank, cfg in enumerate(configs, 1):
                cta_m = cfg['cta_m']
                cta_n = cfg['cta_n']
                cta_k = cfg['cta_k']
                stages = cfg['stages']
                splitk = cfg['splitk']
                estimated_ms = cfg['estimated_time_ms']
                
                print(f"    Rank {rank}: CTA({cta_m}x{cta_n}x{cta_k}) stages={stages} splitK={splitk} est={estimated_ms:.3f}ms")
                
                # Skip if split-K > 1 (current kernel doesn't support it)
                if splitk > 1:
                    print(f"      SKIP: split-K={splitk} not supported by base kernel")
                    writer.writerow([
                        m, n, k,
                        cta_m, cta_n, cta_k,
                        stages, splitk,
                        rank, estimated_ms,
                        -1, -1,
                    ])
                    f.flush()
                    continue
                
                # Run CuTe kernel
                time_us = run_cute_kernel(
                    m, n, k,
                    cta_m, cta_n, cta_k,
                    stages,
                    kernel_script=kernel_script,
                    warmup_iterations=warmup_iterations,
                    iterations=iterations,
                )
                
                if time_us > 0:
                    # Calculate GFLOPS
                    flops = 2.0 * m * n * k  # Multiply-add counts as 2 ops
                    gflops = (flops / time_us) / 1000.0  # us -> GFLOPS
                    print(f"      Result: {time_us:.2f} us, {gflops:.2f} GFLOPS")
                else:
                    gflops = -1
                    print(f"      Result: FAILED")
                
                # Write result
                writer.writerow([
                    m, n, k,
                    cta_m, cta_n, cta_k,
                    stages, splitk,
                    rank, estimated_ms,
                    time_us, gflops,
                ])
                f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Query nvMatmulHeuristics and run CuTe GEMM kernel"
    )
    parser.add_argument(
        "--problems",
        type=str,
        required=True,
        help="Path to CSV file with problem sizes (M,N,K per row)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nvmmh_cute_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--kernel_script",
        type=str,
        default="cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm_tunable.py",
        help="Path to tensorop_gemm_tunable.py",
    )
    parser.add_argument(
        "--num_configs",
        type=int,
        default=8,
        help="Number of top configs to get from nvMMH (default: 5)",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=5,
        help="Warmup iterations for kernel timing",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Benchmark iterations for kernel timing",
    )
    
    args = parser.parse_args()
    
    # Check if nvMMH is available
    if not NVMMH_AVAILABLE:
        print("\nERROR: nvidia-matmul-heuristics package not found!")
        print("Install with: pip install nvidia-matmul-heuristics")
        sys.exit(1)
    
    # Load problems
    print(f"Loading problems from {args.problems}...")
    problems = load_problems_csv(args.problems)
    print(f"Loaded {len(problems)} problems")
    
    if not problems:
        print("ERROR: No valid problems found in CSV")
        sys.exit(1)
    
    # Check kernel script exists
    if not Path(args.kernel_script).exists():
        print(f"ERROR: Kernel script not found: {args.kernel_script}")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Problems file: {args.problems}")
    print(f"  Output file: {args.output}")
    print(f"  Kernel script: {args.kernel_script}")
    print(f"  Configs per problem: {args.num_configs}")
    print(f"  Warmup iterations: {args.warmup_iterations}")
    print(f"  Benchmark iterations: {args.iterations}")
    
    # Run benchmark suite
    run_benchmark_suite(
        problems,
        args.output,
        args.kernel_script,
        args.num_configs,
        args.warmup_iterations,
        args.iterations,
    )
    
    print(f"\nDone! Results written to {args.output}")


if __name__ == "__main__":
    main()
