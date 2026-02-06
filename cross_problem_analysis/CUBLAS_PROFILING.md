# cuBLAS Profiling Guide

This guide explains how to profile NVIDIA cuBLAS GEMM operations for the same problem shapes used in your cross-problem analysis, enabling direct comparison with custom CUTLASS kernels.

## Overview

The cuBLAS profiling functionality allows you to:
1. Profile NVIDIA's optimized cuBLAS library for your problem shapes
2. Collect NCU hardware counter profiles
3. Extract and compare metrics with your custom kernels
4. Understand performance differences and optimization opportunities

## Quick Start

### Basic cuBLAS Profiling

Profile cuBLAS for the default problem shapes (same as your cross-problem analysis):

```bash
python profile_cublas.py
```

This will:
- Create a cuBLAS runner script (if needed)
- Profile cuBLAS for 5 problem shapes
- Save NCU profiles to `cublas_ncu_profiles/`

### Profile and Extract Metrics

Profile cuBLAS and immediately extract metrics:

```bash
python profile_cublas.py --extract-metrics
```

This generates comprehensive metrics in `cublas_metrics_summary/cublas_comprehensive_metrics.csv`.

## Custom Problem Shapes

### Specify Custom Indices

Profile different problem shapes from your best_by_problem CSV:

```bash
# Profile problems at indices 0, 1, 2, 3, 4 (rows 1-5 in CSV)
python profile_cublas.py --problem-indices 0 1 2 3 4

# Profile problems at indices 10, 15, 20
python profile_cublas.py --problem-indices 10 15 20
```

### Custom Configuration File

Use a different CSV file with problem configurations:

```bash
python profile_cublas.py \
    --best-configs-csv /path/to/my_configs.csv \
    --problem-indices 0 1 2
```

## NCU Profiling Options

### Adjust NCU Collection

```bash
# Collect detailed metrics (more comprehensive but slower)
python profile_cublas.py --ncu-sets detailed

# Run multiple profiling iterations for better accuracy
python profile_cublas.py --ncu-iterations 3

# Custom output directory
python profile_cublas.py --output-dir my_cublas_profiles
```

## Complete Workflow Example

Here's a complete workflow to profile both custom kernels and cuBLAS, then compare:

```bash
# 1. Run your cross-problem analysis (custom CUTLASS kernels)
./run_example_analysis.sh

# 2. Profile cuBLAS for the same problems
python profile_cublas.py --extract-metrics

# 3. Extract metrics from both
python extract_ncu_metrics.py \
    --ncu-dir results_5x5_on_v2/ncu_profiles \
    --output-dir custom_metrics \
    --metric-set all

python extract_ncu_metrics.py \
    --ncu-dir cublas_ncu_profiles \
    --output-dir cublas_metrics \
    --metric-set all

# 4. Compare results (using Python, pandas, etc.)
```

## Understanding cuBLAS Profiles

### What Gets Profiled

Each cuBLAS profile captures:
- **Multiple kernels**: cuBLAS may launch several kernels for a single GEMM
- **Optimized implementations**: cuBLAS selects the best implementation based on problem size
- **Hardware counters**: Same metrics as your custom kernels

### Profile File Naming

cuBLAS profiles are named: `cublas_M{M}_N{N}_K{K}_L{L}.ncu-rep`

Example: `cublas_M256_N2048_K8192_L1.ncu-rep`

## Comparing with Custom Kernels

### Side-by-Side Metrics

After extracting metrics from both:

```python
import pandas as pd

# Load metrics
custom = pd.read_csv('custom_metrics/comprehensive_metrics.csv')
cublas = pd.read_csv('cublas_metrics/cublas_comprehensive_metrics.csv')

# Filter for main GEMM kernels
custom_gemm = custom[custom['Kernel_Name'].str.contains('tensorop_gemm')]
cublas_gemm = cublas[cublas['Kernel_Name'].str.contains('gemm', case=False)]

# Compare pipe utilization
print("Tensor Core Utilization:")
print(f"Custom: {custom_gemm['Tensor_Pipe_Active_%'].mean():.2f}%")
print(f"cuBLAS: {cublas_gemm['Tensor_Pipe_Active_%'].mean():.2f}%")
```

### Key Metrics to Compare

1. **Tensor Core Utilization** (`Tensor_Pipe_Active_%`)
   - Higher is better for tensor core kernels
   - Shows how well you're using tensor cores

2. **Memory Throughput** (`DRAM_Throughput_%`)
   - Indicates memory bottlenecks
   - Compare against peak bandwidth

3. **Kernel Duration** (`Duration_us`)
   - Direct performance comparison
   - Lower is better

4. **L1/L2 Hit Rates** (`L1_Hit_Rate_%`, `L2_Hit_Rate_%`)
   - Shows memory access efficiency
   - Higher hit rates mean better data reuse

## Troubleshooting

### "run_cublas_gemm.py not found"

The script is automatically created. If you see this error, ensure you have write permissions in the directory.

### "PyTorch/CUDA not available"

cuBLAS profiling requires:
- PyTorch with CUDA support
- CUDA toolkit installed

Install with: `pip install torch` (ensure CUDA version matches)

### "NCU profiling failed"

Common causes:
- NCU not in PATH
- Insufficient permissions (try `sudo` if needed)
- GPU in use by other processes

### Profiles are empty or have no data

Check:
- Problem sizes are valid (not too small)
- GPU has enough memory
- CUDA driver is compatible

## Advanced Usage

### Programmatic API

Use the cuBLAS profiler in your own scripts:

```python
from cublas_profiler import CuBLASProfiler, create_cublas_runner_script
from config_manager import ConfigManager, ProblemShape

# Create runner script
script = create_cublas_runner_script()

# Setup profiler
profiler = CuBLASProfiler(
    cublas_runner_script=script,
    output_dir="my_cublas_profiles",
    ncu_sets="full",
    ncu_iterations=1
)

# Profile specific problems
problems = [
    ProblemShape(M=256, N=256, K=256, L=1),
    ProblemShape(M=512, N=512, K=512, L=1),
]

results = profiler.profile_problems(problems, verbose=True)

# Check results
for result in results:
    if result.success:
        print(f"✓ {result.problem}: {result.ncu_profile_path}")
    else:
        print(f"✗ {result.problem}: {result.error_message}")
```

### Custom cuBLAS Runner

If you need custom GEMM behavior (different data types, transpose modes, etc.), modify `run_cublas_gemm.py`:

```python
# Example: Add transpose support
def run_cublas_gemm(M, N, K, batch_size=1, transA=False, transB=False):
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    if transA:
        A = A.T
    if transB:
        B = B.T
    
    C = torch.matmul(A, B)
    return C
```

## Integration with Existing Analysis

### Batch Processing

Profile cuBLAS for all problems in your dataset:

```bash
# Extract all problem indices from CSV
python -c "
from config_manager import ConfigManager
cm = ConfigManager('best_by_problem_v1.csv')
n = len(cm.get_all_best_configs())
print(' '.join(map(str, range(n))))
" | xargs python profile_cublas.py --problem-indices
```

### Automated Comparison Report

Create a comparison script:

```python
# compare_custom_vs_cublas.py
import pandas as pd

custom = pd.read_csv('custom_metrics/comprehensive_metrics.csv')
cublas = pd.read_csv('cublas_metrics/cublas_comprehensive_metrics.csv')

# Filter main kernels
custom_gemm = custom[custom['Kernel_Name'].str.contains('tensorop_gemm')]
cublas_gemm = cublas[cublas['Kernel_Name'].str.contains('gemm', case=False)]

# Compare
metrics = ['Tensor_Pipe_Active_%', 'Duration_us', 'DRAM_Throughput_%']

print("Custom vs cuBLAS Comparison:")
print("="*60)
for metric in metrics:
    custom_val = custom_gemm[metric].mean()
    cublas_val = cublas_gemm[metric].mean()
    diff = ((custom_val - cublas_val) / cublas_val) * 100
    print(f"{metric:30s}: {custom_val:8.2f} vs {cublas_val:8.2f} ({diff:+.1f}%)")
```

## Expected Output Structure

After running cuBLAS profiling:

```
cublas_ncu_profiles/
├── cublas_M256_N2048_K8192_L1.ncu-rep
├── cublas_M256_N8192_K2048_L1.ncu-rep
├── cublas_M2048_N2048_K16384_L1.ncu-rep
├── cublas_M4096_N128_K4096_L1.ncu-rep
└── cublas_M4096_N4096_K4096_L1.ncu-rep

cublas_metrics_summary/
└── cublas_comprehensive_metrics.csv
```

The CSV contains all kernels launched by cuBLAS (typically multiple per GEMM operation).

## Performance Tips

1. **Use same GPU state**: Profile both custom and cuBLAS on the same GPU configuration
2. **Multiple iterations**: Use `--ncu-iterations 3` for more stable metrics
3. **Problem size matters**: cuBLAS may use different kernels for different sizes
4. **Batch processing**: Profile multiple problems in one run to save setup overhead

## Further Reading

- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/)
- [PyTorch CUDA Operations](https://pytorch.org/docs/stable/torch.html#torch.matmul)
