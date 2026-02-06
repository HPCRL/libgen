# Cross-Problem Analysis - Quick Reference

## How to Run Scripts

All scripts can be run in **two ways**:

**Method 1: Direct execution (from the package directory)**
```bash
cd /path/to/cross_problem_analysis
python cross_problem_sweep.py [options]
python profile_cublas.py [options]
python extract_ncu_metrics.py [options]
```

**Method 2: As Python modules (from parent directory)**
```bash
cd /path/to/parent_directory
python -m cross_problem_analysis.cross_problem_sweep [options]
python -m cross_problem_analysis.profile_cublas [options]
python -m cross_problem_analysis.extract_ncu_metrics [options]
```

Use whichever method is more convenient for your workflow!

## Important: Correctness Checking

**By default, correctness checking is ENABLED** to ensure all collected results are valid.
- Wrong results are detected and marked as failures
- Use `--skip_ref_check` ONLY if you're certain all configs are correct
- Failed runs show `[skip]` for correctness errors, `[fail]` for runtime errors

## Common Use Cases

### Custom Kernel Analysis

#### 1. Run 5×5 sweep (Performance + NCU)
```bash
# From package directory:
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_5x5

# Or from parent directory:
python -m cross_problem_analysis.cross_problem_sweep \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_5x5
```

#### 2. Performance only (faster, no NCU)
```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --skip_ncu \
  --output_dir results_perf_only
```

#### 3. NCU only (already have performance)
```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --skip_performance \
  --output_dir results_ncu_only
```

#### 4. Filter large problems (M, N >= 2048)
```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --filter_problems \
  --min_m 2048 --min_n 2048 \
  --output_dir results_large_problems
```

#### 5. Custom NCU settings
```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --ncu_sets full,memory \
  --ncu_iterations 3 \
  --output_dir results_custom_ncu
```

#### 6. High-quality performance benchmark
```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --perf_iterations 100 \
  --perf_warmup 10 \
  --use_cold_l2 \
  --skip_ncu \
  --output_dir results_high_quality
  # Note: Correctness checking still enabled by default
```

#### 7. Skip correctness checking (NOT RECOMMENDED)
```bash
# Only use if you're CERTAIN all configs are correct
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --skip_ref_check \
  --output_dir results_no_check
  # WARNING: May collect wrong results!
```

### cuBLAS Profiling

#### 8. Profile cuBLAS (basic)
```bash
cd cross_problem_analysis
python profile_cublas.py
```

#### 9. Profile cuBLAS with metrics extraction
```bash
cd cross_problem_analysis
python profile_cublas.py --extract-metrics
```

#### 10. cuBLAS with custom problems
```bash
cd cross_problem_analysis
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_profiles \
  --extract-metrics \
  --metrics-output-dir cublas_metrics
```

#### 11. cuBLAS with detailed NCU
```bash
cd cross_problem_analysis
python profile_cublas.py \
  --ncu-sets detailed \
  --ncu-iterations 3 \
  --extract-metrics
```

### NCU Metrics Extraction

#### 12. Extract all metrics
```bash
cd cross_problem_analysis
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir metrics_summary \
  --metric-set all
```

#### 13. Extract pipe utilization only
```bash
cd cross_problem_analysis
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir pipe_metrics \
  --metric-set pipe
```

#### 14. Extract custom metrics
```bash
cd cross_problem_analysis
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir custom_metrics \
  --custom-metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed" "dram__throughput.avg.pct_of_peak_sustained_elapsed"
```

#### 15. List available metrics
```bash
cd cross_problem_analysis
python extract_ncu_metrics.py \
  --list-metrics results_5x5/ncu_profiles/prob0_cfg0_*.ncu-rep
```

### Complete Workflows

#### 16. Full comparison: Custom vs cuBLAS
```bash
cd cross_problem_analysis

# Step 1: Profile custom kernels
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir custom_results

# Step 2: Profile cuBLAS
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_results

# Step 3: Extract metrics from both
python extract_ncu_metrics.py \
  --ncu-dir custom_results/ncu_profiles \
  --output-dir custom_metrics \
  --metric-set all

python extract_ncu_metrics.py \
  --ncu-dir cublas_results \
  --output-dir cublas_metrics \
  --metric-set all

# Step 4: Compare CSVs
# custom_metrics/comprehensive_metrics.csv
# cublas_metrics/cublas_comprehensive_metrics.csv
```

#### 17. Pipe utilization comparison
```bash
cd cross_problem_analysis

# Custom kernels - pipe metrics only
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir custom_pipe

python extract_ncu_metrics.py \
  --ncu-dir custom_pipe/ncu_profiles \
  --metric-set pipe \
  --output-dir custom_pipe_metrics

# cuBLAS - pipe metrics only
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --extract-metrics

# Compare Tensor Core utilization in CSV files
```

## Output Files

### Custom Kernel Results
```
output_dir/
├── performance_results.csv    # All performance measurements
├── ncu_results.csv           # NCU profiling status
├── summary.json              # Statistics and metadata
└── ncu_profiles/             # NCU .ncu-rep files
```

### cuBLAS Profiling Results
```
cublas_ncu_profiles/          # Default output directory
├── cublas_M256_N2048_K8192_L1.ncu-rep
├── cublas_M256_N8192_K2048_L1.ncu-rep
└── ...

cublas_metrics_summary/       # If --extract-metrics used
└── cublas_comprehensive_metrics.csv
```

### Metrics Extraction Results
```
metrics_output_dir/
├── pipe_utilization_metrics.csv       # If --metric-set pipe
├── memory_metrics.csv                 # If --metric-set memory
├── compute_metrics.csv                # If --metric-set compute
└── comprehensive_metrics.csv          # If --metric-set all
```

## Analyzing Results

### View Performance Results
```bash
# Best performing config for each problem
python -c "
import pandas as pd
df = pd.read_csv('output_dir/performance_results.csv')
successful = df[df['success'] == True]
best = successful.loc[successful.groupby(['M','N','K','L'])['gflops'].idxmax()]
print(best[['M','N','K','L','gflops','cta_m','cta_n','cta_k','stages']])
"
```

### View Summary
```bash
cat output_dir/summary.json | python -m json.tool
```

### Open NCU Reports
```bash
ncu-ui output_dir/ncu_profiles/prob0_cfg0_*.ncu-rep
```

## Module Import (Python API)

```python
from cross_problem_analysis import (
    ConfigManager,
    KernelRunner,
    NCUProfiler,
    CrossProblemSweep
)

# Load and select problems
config_mgr = ConfigManager("../collected_data/best_by_problem_v1.csv")
problems = config_mgr.get_problem_subset([0, 1, 2])

# Run sweep
runner = KernelRunner("../run_one_config.py", iterations=50)
for problem in problems:
    best = config_mgr.get_best_config(problem)
    result = runner.run_single_config(problem, best.config)
    print(f"{problem}: {result.gflops:.2f} GFLOPS")
```

## Time Estimates

| Sweep Size | Performance Only | With NCU (full) |
|------------|------------------|-----------------|
| 3×3 (9 runs) | ~2-3 minutes | ~30-60 minutes |
| 5×5 (25 runs) | ~5-8 minutes | ~2-3 hours |
| 10×10 (100 runs) | ~20-30 minutes | ~8-12 hours |

**cuBLAS Profiling:**
- ~5-10 minutes per problem with NCU
- Metrics extraction: ~1-2 minutes per 25 profiles

*Times vary by hardware and problem sizes*

## Predefined Metric Sets

| Set | Metrics | Description |
|-----|---------|-------------|
| **pipe** | 12 metrics | Pipe utilization: Tensor, FMA, ALU, FP64, LSU, TEX (active %, elapsed %) |
| **memory** | 5 metrics | DRAM throughput, L1/L2 hit rates |
| **compute** | 4 metrics | Duration, warps, IPC, cycles |
| **all** | 21 metrics | All predefined metrics combined |

See `python extract_ncu_metrics.py --list-metrics <profile.ncu-rep>` for available metrics.

## Troubleshooting

**Import errors when running scripts:**
- **Solution 1**: Run from the package directory with `cd cross_problem_analysis && python script.py`
- **Solution 2**: Run as a module: `python -m cross_problem_analysis.script_name`
- Both methods are supported and work identically

**NCU not found:**
```bash
which ncu  # Check if in PATH
# or specify path:
--ncu_binary /usr/local/cuda/bin/ncu
```

**Timeout errors:**
- Reduce problem subset size
- Increase timeout in kernel_runner.py or ncu_profiler.py

**Out of memory:**
- Run smaller batches
- Use `--skip_performance` or `--skip_ncu` to split workload

**cuBLAS profiling fails:**
- Ensure PyTorch with CUDA support is installed: `pip install torch`
- Check CUDA driver compatibility
- Verify GPU is not in use by other processes

**Metrics extraction is slow:**
- Normal for large NCU profiles (1-2 min for 25 profiles)
- Use specific metric sets (`--metric-set pipe`) instead of `all`
- Process profiles in batches

## Quick Command Reference

```bash
# Most common commands (copy-paste ready)
# Run these from the cross_problem_analysis directory

cd cross_problem_analysis

# 1. Standard 5x5 analysis
python cross_problem_sweep.py --problem_indices 0,1,2,3,4 --output_dir results_5x5

# 2. Compare custom vs cuBLAS
python cross_problem_sweep.py --problem_indices 0,1,2,3,4 --output_dir custom_results
python profile_cublas.py --problem-indices 0,1,2,3,4 --extract-metrics

# 3. Extract pipe metrics from custom
python extract_ncu_metrics.py --ncu-dir results_5x5/ncu_profiles --output-dir metrics --metric-set pipe

# 4. Performance only (fast)
python cross_problem_sweep.py --problem_indices 0,1,2,3,4 --skip_ncu --output_dir perf_only

# 5. View NCU profile
ncu-ui results_5x5/ncu_profiles/prob0_cfg0_*.ncu-rep
```

## Documentation

- **README.md**: Complete package documentation
- **CUBLAS_PROFILING.md**: Detailed cuBLAS profiling guide
- **NCU_METRICS_EXTRACTION.md**: Metrics extraction documentation
- **IMPLEMENTATION.md**: Technical implementation details
- **USAGE_EXAMPLES.md**: Additional usage examples

## Examples

See `example_usage.py` for programmatic usage:
```bash
python cross_problem_analysis/example_usage.py --example 1  # 3x3 sweep
python cross_problem_analysis/example_usage.py --example 2  # Filter demo
python cross_problem_analysis/example_usage.py --example 3  # Single run
```
