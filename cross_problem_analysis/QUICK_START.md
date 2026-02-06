# Quick Start: cuBLAS Direct API

## TL;DR

We replaced PyTorch-based cuBLAS profiling with direct cuBLAS API calls to get clean NCU profiles.

## Quick Start

```bash
# 1. Build (one-time)
cd /media/datassd/sina/libgen/cross_problem_analysis
make

# 2. Test
make test

# 3. Use
./cublas_gemm_runner --M 512 --N 512 --K 512

# 4. Profile with NCU
ncu --set full -o profile ./cublas_gemm_runner --M 512 --N 512 --K 512
```

## What Changed?

**Before:**
```python
# PyTorch-based (run_cublas_gemm.py)
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C = torch.matmul(A, B)  # Many kernels in profile!
```

**After:**
```cpp
// Direct cuBLAS API (cublas_gemm_runner.cu)
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K, &alpha,
             d_A, CUDA_R_16F, lda,
             d_B, CUDA_R_16F, ldb, &beta,
             d_C, CUDA_R_16F, ldc,
             CUBLAS_COMPUTE_16F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // One kernel!
```

## Result

NCU profiles now contain **only** cuBLAS GEMM kernels:
- ✅ `ampere_h16816gemm_64x64_ldg8_...`
- ❌ No random generation kernels
- ❌ No PyTorch overhead

## Usage

### Command Line
```bash
# Single GEMM
./cublas_gemm_runner --M 512 --N 512 --K 512

# Batched GEMM
./cublas_gemm_runner --M 256 --N 256 --K 256 --batch 8

# With iterations
./cublas_gemm_runner --M 1024 --N 1024 --K 1024 --iterations 10 --warmup 3
```

### Python API
```python
from cublas_profiler import CuBLASProfiler

# Uses C++ runner by default
profiler = CuBLASProfiler(
    cublas_runner_script="./cublas_gemm_runner"
)
```

## Files

- **`cublas_gemm_runner.cu`** - C++ CUDA program
- **`Makefile`** - Build system
- **`cublas_profiler.py`** - Updated profiler (backward compatible)
- **`run_cublas_gemm.py`** - Old PyTorch script (still works)

## Documentation

- **`CUBLAS_DIRECT_API.md`** - Full documentation
- **`CUBLAS_INTEGRATION_SUMMARY.md`** - Detailed summary

## Benefits

1. Clean NCU profiles (only cuBLAS kernels)
2. Accurate performance comparison
3. No PyTorch dependencies
4. Faster execution
5. Direct control over cuBLAS parameters
