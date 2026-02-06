#!/usr/bin/env python3
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
