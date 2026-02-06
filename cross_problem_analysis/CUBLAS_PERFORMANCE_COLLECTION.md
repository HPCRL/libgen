# cuBLAS Performance Collection

## Overview

The cuBLAS sweep infrastructure now collects both **performance metrics** (GFLOPS, timing) and **NCU profiles** (detailed hardware metrics), matching the functionality of the CUTLASS sweep infrastructure. This enables direct performance comparison between cuBLAS and custom CUTLASS kernels.

## Performance Data Collection

### What's Collected

For each problem shape (M×N×K×L):
- **GFLOPS**: Achieved compute throughput
- **avg_time_us**: Average execution time in microseconds
- **Success/Error**: Execution status and error messages

### Output Format

Results are saved to `performance_results.csv` with the same format as CUTLASS sweeps:

```csv
M,N,K,L,success,gflops,avg_time_us,error
256,3072,8192,1,True,106275.0,121.242,
256,8192,3072,1,True,102342.0,125.901,
4096,4096,4096,1,True,235891.0,142.356,
```

This matches the format from `cross_problem_sweep.py`, enabling unified analysis.

## Usage

### Basic Performance Collection

```bash
# Collect performance + NCU profiles
python profile_cublas.py

# Performance only (faster, no NCU overhead)
python profile_cublas.py --skip-ncu

# NCU only (no performance collection)
python profile_cublas.py --skip-performance
```

### Customizing Benchmark Parameters

```bash
# More iterations for stable results
python profile_cublas.py --perf-iterations 100 --perf-warmup 10

# Quick test with fewer iterations
python profile_cublas.py --perf-iterations 10 --perf-warmup 2

# High precision (many iterations)
python profile_cublas.py --perf-iterations 200 --perf-warmup 20
```

### Profile Specific Problems

```bash
# Profile problems at indices 3, 7, 12, 17, 18
python profile_cublas.py --problem-indices 3 7 12 17 18

# Custom output directory
python profile_cublas.py --output-dir cublas_results_v2
```

## Implementation Details

### C++ Runner Timing

The `cublas_gemm_runner` uses CUDA events for accurate timing:

```cpp
// Calculate GFLOPS (2*M*N*K*batch_size operations)
double gflops_per_iter = (2.0 * M * N * K * batch_size) / 1e9;

// Time the iterations
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (int i = 0; i < iterations; ++i) {
    cublasGemmEx(...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
float avg_time_ms = milliseconds / iterations;
double gflops = gflops_per_iter / (avg_time_ms / 1000.0);

// Output parseable format
std::cout << "[PERF] avg_time_us: " << (avg_time_ms * 1000.0) << std::endl;
std::cout << "[PERF] gflops: " << gflops << std::endl;
```

### Python Profiler Integration

The `CuBLASProfiler` class now has performance collection methods:

```python
# Run performance benchmark
result = profiler.run_performance(
    problem,
    iterations=50,
    warmup=5,
    verbose=True
)

# Batch processing
results = profiler.run_performance_problems(
    problems,
    iterations=50,
    warmup=5,
    verbose=True
)

# Save to CSV
with open('performance_results.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['M', 'N', 'K', 'L', 'success', 'gflops', 'avg_time_us', 'error'])
    writer.writeheader()
    for result in results:
        writer.writerow(result.to_dict())
```

## Comparison with CUTLASS Sweep

### Unified CSV Format

Both cuBLAS and CUTLASS sweeps now generate identical CSV formats:

**cuBLAS:** `profile_cublas.py` → `performance_results.csv`
**CUTLASS:** `cross_problem_sweep.py` → `performance_results.csv`

This enables:
- Direct GFLOPS comparison
- Unified analysis scripts
- Side-by-side performance plots
- Relative performance calculations

### Analysis Workflow

```python
import pandas as pd

# Load both results
cublas_df = pd.read_csv('cublas_results/performance_results.csv')
cutlass_df = pd.read_csv('cutlass_results/performance_results.csv')

# Merge on problem shape
merged = cublas_df.merge(
    cutlass_df,
    on=['M', 'N', 'K', 'L'],
    suffixes=('_cublas', '_cutlass')
)

# Calculate relative performance
merged['speedup'] = merged['gflops_cutlass'] / merged['gflops_cublas']
merged['efficiency_pct'] = (merged['gflops_cutlass'] / merged['gflops_cublas']) * 100

print(merged[['M', 'N', 'K', 'gflops_cublas', 'gflops_cutlass', 'speedup', 'efficiency_pct']])
```

## Performance Metrics

### GFLOPS Calculation

For GEMM operation: `C = α·A·B + β·C`

- **FLOPs per element**: 2 (multiply-add)
- **Total FLOPs**: `2 × M × N × K × L` (batch size L)
- **GFLOPS**: `Total FLOPs / (time_in_seconds × 10^9)`

### Timing Precision

- **Method**: CUDA events (`cudaEventElapsedTime`)
- **Granularity**: Microseconds
- **Overhead**: Minimal (event recording only)
- **Accuracy**: Hardware timer, precise to ~0.5 μs

### Warmup Importance

Warmup iterations are critical for:
- GPU frequency scaling stabilization
- Cache warming
- Driver/library initialization
- Consistent timing measurements

Recommended: 5-10 warmup iterations for stable results.

## Command-Line Reference

### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-performance` | False | Skip performance collection (NCU only) |
| `--perf-iterations` | 50 | Number of benchmark iterations |
| `--perf-warmup` | 5 | Number of warmup iterations |

### NCU Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-ncu` | False | Skip NCU profiling (performance only) |
| `--ncu-sets` | full | NCU metric sets to collect |
| `--ncu-iterations` | 1 | Number of NCU profiling iterations |

### General Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | cublas_ncu_profiles | Output directory |
| `--problem-indices` | [3,7,12,17,18] | Problem indices to profile |

## Example Workflows

### Quick Performance Check

```bash
# Fast performance-only run
python profile_cublas.py \
  --skip-ncu \
  --perf-iterations 20 \
  --perf-warmup 5 \
  --problem-indices 0 1 2 3 4
```

### Full Analysis with Metrics

```bash
# Comprehensive profiling
python profile_cublas.py \
  --perf-iterations 100 \
  --perf-warmup 10 \
  --ncu-sets full \
  --extract-metrics \
  --output-dir cublas_full_analysis
```

### NCU Deep Dive

```bash
# Detailed NCU profiling (skip perf for speed)
python profile_cublas.py \
  --skip-performance \
  --ncu-sets full \
  --ncu-iterations 3 \
  --output-dir cublas_ncu_detailed
```

## Troubleshooting

### Low GFLOPS

If GFLOPS are unexpectedly low:
1. Check GPU utilization: `nvidia-smi`
2. Increase iterations for stable measurements
3. Ensure no background processes competing for GPU
4. Verify problem sizes are reasonable (not too small)

### Timing Variability

If results vary between runs:
1. Increase warmup iterations (try 10-20)
2. Increase benchmark iterations (try 100-200)
3. Check for GPU frequency throttling
4. Ensure deterministic GPU clocks: `nvidia-smi -lgc <freq>`

### CSV Format Issues

If CSV doesn't match expected format:
1. Verify C++ runner outputs `[PERF]` lines
2. Check runner compilation: `make clean && make cublas_gemm_runner`
3. Test runner manually: `./cublas_gemm_runner --M 128 --N 128 --K 128`

## Related Documentation

- **Main Update Guide**: [SWEEP_INFRASTRUCTURE_UPDATE.md](SWEEP_INFRASTRUCTURE_UPDATE.md)
- **C++ Runner API**: [CUBLAS_DIRECT_API.md](CUBLAS_DIRECT_API.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Integration Summary**: [CUBLAS_INTEGRATION_SUMMARY.md](CUBLAS_INTEGRATION_SUMMARY.md)

## Summary

The cuBLAS sweep infrastructure now provides:
- ✅ Performance metrics collection (GFLOPS, timing)
- ✅ CSV output matching CUTLASS sweep format
- ✅ Configurable benchmark parameters
- ✅ Clean separation of performance and profiling runs
- ✅ Unified analysis workflow with custom kernels

This enables direct, fair comparison between cuBLAS and custom CUTLASS kernels using the same measurement methodology.
