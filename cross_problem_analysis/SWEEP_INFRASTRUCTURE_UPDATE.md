# Sweep Infrastructure Update: C++ cuBLAS Runner Integration

## Overview

The sweep infrastructure (`profile_cublas.py`) has been updated to use the C++ direct cuBLAS API runner instead of PyTorch-based profiling. This eliminates extraneous kernels (random generation, etc.) from NCU profiles and provides clean cuBLAS GEMM profiling.

## Changes Made

### 1. Updated `profile_cublas.py`
**Location:** `/media/datassd/sina/libgen/cross_problem_analysis/profile_cublas.py`

**Key Changes:**
- Automatically detects C++ runner (`cublas_gemm_runner`)
- Builds runner if not already compiled using Makefile
- Passes C++ executable to `CuBLASProfiler` with `use_cpp_runner=True`
- Updated help text to indicate C++ API usage

**New Flow:**
```python
# Check if C++ runner exists
cpp_runner_exe = os.path.join(script_dir, "cublas_gemm_runner")

# Build if needed
if not os.path.exists(cpp_runner_exe):
    subprocess.run(["make", "cublas_gemm_runner"], cwd=script_dir, check=True)

# Create profiler with C++ runner
profiler = CuBLASProfiler(
    cublas_runner_script=cpp_runner_exe,
    output_dir=args.output_dir,
    ncu_sets=args.ncu_sets,
    ncu_iterations=args.ncu_iterations,
    use_cpp_runner=True  # Use direct cuBLAS API via C++
)
```

### 2. No Changes Needed to `cublas_profiler.py`
**Location:** `/media/datassd/sina/libgen/cross_problem_analysis/cublas_profiler.py`

Already supports C++ runner via:
- `use_cpp_runner` parameter in `__init__()` (default: `True`)
- Automatic command building for C++ executables vs Python scripts

### 3. Old PyTorch Runner Preserved
**Location:** `/media/datassd/sina/libgen/cross_problem_analysis/run_cublas_gemm.py`

- Still exists for backward compatibility
- No longer used by default sweep infrastructure
- Can be manually invoked if needed

## Usage

### Running Sweep with C++ Runner (Default)

```bash
# Profile default problems with performance + NCU (indices 3, 7, 12, 17, 18)
python profile_cublas.py

# Performance only (skip NCU profiling)
python profile_cublas.py --skip-ncu --perf-iterations 100

# NCU profiling only (skip performance)
python profile_cublas.py --skip-performance

# Profile custom problem indices
python profile_cublas.py --problem-indices 0 1 2 3 4

# Custom performance settings
python profile_cublas.py --perf-iterations 100 --perf-warmup 10

# Extract metrics immediately after profiling
python profile_cublas.py --extract-metrics

# Custom NCU settings
python profile_cublas.py --ncu-sets detailed --ncu-iterations 3
```

### Output Files

The sweep generates two main outputs:

1. **`performance_results.csv`** - Performance metrics (GFLOPS, timing)
   ```csv
   M,N,K,L,success,gflops,avg_time_us,error
   256,3072,8192,1,True,106275.0,121.242,
   256,8192,3072,1,True,102342.0,125.901,
   ```

2. **NCU Profile Files** - Detailed kernel metrics (`.ncu-rep` files)
   - `cublas_M256_N3072_K8192_L1.ncu-rep`
   - `cublas_M256_N8192_K3072_L1.ncu-rep`
   - etc.

### First Run Behavior
On the first run, if `cublas_gemm_runner` doesn't exist:
```
✓ Using C++ cuBLAS runner: /path/to/cublas_gemm_runner
```

Or if not compiled:
```
C++ cuBLAS runner not found at: /path/to/cublas_gemm_runner
Building C++ runner...
✓ C++ runner built successfully
```

### Verification
Check that profiles contain only cuBLAS kernels:
```bash
ncu --import cublas_ncu_profiles/cublas_M256_N3072_K8192_L1.ncu-rep --page raw --csv | grep "Kernel Name"
```

Expected: Only `ampere_h16816gemm_*` kernels, no PyTorch or cuRAND kernels.

## Benefits

### 1. Clean Profiles
**Before (PyTorch):**
- cuBLAS GEMM kernel
- PyTorch random generation kernels
- cuRAND kernels
- Extra memory operations

**After (C++ Direct API):**
- **Only** cuBLAS GEMM kernel
- Pure performance data

### 2. Accurate Metrics
- No contamination from random generation overhead
- Direct measurement of cuBLAS GEMM performance
- Comparable to custom CUTLASS kernel profiles

### 3. Performance Collection
- **NEW:** Collects GFLOPS and timing data like CUTLASS sweep
- Generates `performance_results.csv` with same format
- Enables direct performance comparison with custom kernels
- Configurable iterations and warmup periods

### 4. Automatic Build
- No manual compilation needed
- Makefile handles dependency checking
- Works seamlessly in sweep workflows

## Migration Guide

### For Existing Workflows

**No changes required!** Simply run `profile_cublas.py` as before:
```bash
python profile_cublas.py
```

The script will:
1. Detect C++ runner (or build it automatically)
2. Use C++ runner for profiling
3. Generate clean NCU profiles

### For Custom Scripts

If you're calling `CuBLASProfiler` directly:

**Old way (PyTorch):**
```python
profiler = CuBLASProfiler(
    cublas_runner_script="run_cublas_gemm.py",
    output_dir="profiles"
)
```

**New way (C++ Direct API):**
```python
profiler = CuBLASProfiler(
    cublas_runner_script="cublas_gemm_runner",  # C++ executable
    output_dir="profiles",
    use_cpp_runner=True  # Important!
)
```

## Re-profiling Old Data

If you have old cuBLAS profiles with PyTorch contamination, re-profile:

```bash
# Use the automated script
./reprofile_cublas.sh

# Or manually
python profile_cublas.py --problem-indices 3 7 12 17 18 --output-dir cublas_ncu_profiles_clean
```

## Technical Details

### C++ Runner Implementation
- **File:** `cublas_gemm_runner.cu`
- **API:** `cublasGemmEx` (single) and `cublasGemmStridedBatchedEx` (batched)
- **Precision:** FP16 with `CUBLAS_COMPUTE_16F` (Tensor Core)
- **Arguments:** `--M --N --K --batch --warmup --iterations`

### NCU Profiling Command
```bash
ncu --set full \
    --kernel-name-base demangled \
    --launch-skip 0 \
    --launch-count 1 \
    -o profile.ncu-rep \
    -f \
    ./cublas_gemm_runner --M 256 --N 3072 --K 8192 --batch 1
```

### Build System
```makefile
cublas_gemm_runner: cublas_gemm_runner.cu
	nvcc -O3 -std=c++11 -arch=sm_80 -lcublas $< -o $@
```

## Files Modified

| File | Type | Changes |
|------|------|---------|
| `profile_cublas.py` | Modified | Added C++ runner detection and build logic |
| `cublas_profiler.py` | No change | Already supports C++ runner via `use_cpp_runner` |
| `run_cublas_gemm.py` | Preserved | Old PyTorch runner kept for compatibility |

## Related Documentation

- **C++ Runner Details:** [CUBLAS_DIRECT_API.md](CUBLAS_DIRECT_API.md)
- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Integration Summary:** [CUBLAS_INTEGRATION_SUMMARY.md](CUBLAS_INTEGRATION_SUMMARY.md)
- **Main Documentation:** [README.md](README.md)

## Troubleshooting

### Build Errors
If `make cublas_gemm_runner` fails:
1. Check nvcc is available: `nvcc --version`
2. Verify CUDA toolkit installation
3. Check for cuBLAS library: `ldconfig -p | grep cublas`
4. Ensure correct GPU architecture in Makefile (`-arch=sm_80`)

### Runtime Errors
If profiling fails:
1. Verify executable exists: `ls -l cublas_gemm_runner`
2. Test manually: `./cublas_gemm_runner --M 128 --N 128 --K 128`
3. Check GPU access: `nvidia-smi`

### Profile Issues
If profiles still show extra kernels:
1. Verify `use_cpp_runner=True` in profiler
2. Check executable path is correct
3. Ensure not using old `run_cublas_gemm.py`

## Summary

The sweep infrastructure now uses direct cuBLAS API calls via C++ for clean, accurate profiling. The update is fully automatic - just run `profile_cublas.py` as usual and the system handles the rest. Old PyTorch-based profiles should be re-generated for consistency.
