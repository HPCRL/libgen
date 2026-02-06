# cuBLAS Direct API Integration

This directory now contains a C++ CUDA program that directly calls the cuBLAS library API for profiling, replacing the previous PyTorch-based approach.

## Problem Solved

Previously, the cuBLAS profiling was done through PyTorch's `torch.matmul()`, which resulted in NCU profiles containing random number generation kernels and other PyTorch overhead instead of pure cuBLAS GEMM kernels. This made it difficult to accurately compare cuBLAS performance with custom kernel implementations.

## New Implementation

### Files

1. **`cublas_gemm_runner.cu`** - C++ CUDA program that directly calls cuBLAS API
   - Uses `cublasGemmEx()` for single GEMM operations
   - Uses `cublasGemmStridedBatchedEx()` for batched operations
   - Supports FP16 computation with Tensor Core acceleration
   - No Python/PyTorch dependencies

2. **`Makefile`** - Build system for the C++ runner
   - Compiles with nvcc
   - Links against cuBLAS library
   - Configurable GPU architecture target

3. **`cublas_profiler.py`** - Updated profiler (backward compatible)
   - Now supports both C++ executable and Python script modes
   - Default: uses C++ executable for cleaner profiles
   - Legacy: can still use Python script if needed

## Building

```bash
cd cross_problem_analysis
make
```

This will compile `cublas_gemm_runner.cu` into an executable called `cublas_gemm_runner`.

### GPU Architecture Configuration

By default, the Makefile is configured for A100 GPUs (`-arch=sm_80`). To change this for your GPU:

```bash
# Edit Makefile and change the ARCH variable:
# For V100:        ARCH = -arch=sm_70
# For Turing:      ARCH = -arch=sm_75
# For A100:        ARCH = -arch=sm_80
# For RTX 30xx:    ARCH = -arch=sm_86
# For RTX 40xx:    ARCH = -arch=sm_89
```

## Testing

Test the runner with sample GEMM operations:

```bash
make test
```

Or run manually:

```bash
./cublas_gemm_runner --M 512 --N 512 --K 512
./cublas_gemm_runner --M 256 --N 256 --K 256 --batch 8
```

## Usage with Profiler

### Python API

```python
from cublas_profiler import CuBLASProfiler
from config_manager import ProblemShape

# Initialize profiler with C++ runner (recommended)
profiler = CuBLASProfiler(
    cublas_runner_script="./cublas_gemm_runner",
    output_dir="cublas_profiles",
    use_cpp_runner=True  # Default: True
)

# Profile problems
problems = [
    ProblemShape(M=512, N=512, K=512, L=1),
    ProblemShape(M=1024, N=1024, K=1024, L=1),
]
results = profiler.profile_problems(problems)
```

### Command Line

The profiler scripts automatically detect and use the C++ runner if available:

```bash
# Build the runner first
make

# Run profiling - will use C++ runner automatically
python analyze_and_plot.py
```

## Implementation Details

### cuBLAS API Calls

**Single GEMM:**
```cpp
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
    M, N, K,
    &alpha,                     // Scalar 1.0
    d_A, CUDA_R_16F, lda,      // Matrix A (FP16)
    d_B, CUDA_R_16F, ldb,      // Matrix B (FP16)
    &beta,                      // Scalar 0.0
    d_C, CUDA_R_16F, ldc,      // Matrix C (FP16)
    CUBLAS_COMPUTE_16F,         // FP16 computation
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Enable Tensor Cores
);
```

**Batched GEMM:**
```cpp
cublasGemmStridedBatchedEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, CUDA_R_16F, lda, strideA,
    d_B, CUDA_R_16F, ldb, strideB,
    &beta,
    d_C, CUDA_R_16F, ldc, strideC,
    batch_size,
    CUBLAS_COMPUTE_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

### Matrix Layout

- **Column-major storage** (cuBLAS default)
- Leading dimensions: `lda = M`, `ldb = K`, `ldc = M`
- Strides for batched: `strideA = M*K`, `strideB = K*N`, `strideC = M*N`

### Tensor Core Usage

- Math mode set to `CUBLAS_TENSOR_OP_MATH`
- Algorithm: `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- This enables automatic Tensor Core usage for FP16 operations

## Benefits of Direct cuBLAS Calls

1. **Clean Profiles**: NCU profiles contain only cuBLAS GEMM kernels
2. **No Overhead**: No PyTorch initialization, random generation, or Python overhead
3. **Explicit Control**: Direct control over cuBLAS parameters and settings
4. **Reproducible**: Consistent results without framework-specific behavior
5. **Accurate Comparison**: Fair comparison with custom CUTLASS/CuTe implementations

## Backward Compatibility

The old PyTorch-based `run_cublas_gemm.py` script is still available and can be used by setting `use_cpp_runner=False` in the profiler:

```python
profiler = CuBLASProfiler(
    cublas_runner_script="run_cublas_gemm.py",
    use_cpp_runner=False  # Use old Python/PyTorch approach
)
```

## References

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuBLAS GemmEx API](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm)
- [NVIDIA CUDA Samples - simpleCUBLAS](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/simpleCUBLAS)
