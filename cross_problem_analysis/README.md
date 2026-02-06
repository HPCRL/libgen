# Cross-Problem Kernel Analysis

This package provides a modular framework for analyzing GEMM kernel configurations across multiple problem shapes using the CuTe DSL Ampere GEMM example.

## Overview

The system collects:
1. **Performance metrics**: Execution time and GFLOPS for all kernel configs on all problem shapes
2. **NCU profiles**: Hardware counter data using NVIDIA Nsight Compute

For a subset of N problem shapes, the system:
- Extracts the best kernel configuration for each problem
- Runs all N configs on all N problems (N×N matrix of experiments)
- Optionally profiles each execution with NCU

## Module Structure

```
cross_problem_analysis/
├── __init__.py                  # Package initialization
├── config_manager.py            # Load and manage kernel configurations
├── kernel_runner.py             # Execute kernels and collect performance
├── ncu_profiler.py              # NCU profiling wrapper
├── cublas_profiler.py           # cuBLAS profiling wrapper
├── ncu_metrics_extractor.py     # Extract metrics from NCU profiles
├── cross_problem_sweep.py       # Main orchestration script
├── profile_cublas.py            # cuBLAS profiling CLI
├── extract_ncu_metrics.py       # NCU metrics extraction CLI
└── README.md                    # This file
```

## How to Run

All scripts support **two execution methods**:

**Method 1: Direct execution (recommended for most users)**
```bash
cd cross_problem_analysis
python cross_problem_sweep.py [options]
python profile_cublas.py [options]
python extract_ncu_metrics.py [options]
```

**Method 2: As Python modules**
```bash
# From parent directory
python -m cross_problem_analysis.cross_problem_sweep [options]
python -m cross_problem_analysis.profile_cublas [options]
python -m cross_problem_analysis.extract_ncu_metrics [options]
```

Both methods work identically - use whichever fits your workflow!

## Quick Start

### 1. Basic Usage (First 5 Problems)

```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_5x5
```

This will:
- Select the first 5 problems from `best_by_problem_v1.csv`
- Run 5×5 = 25 performance benchmarks
- Collect 25 NCU profiles
- Save results to `results_5x5/`

### 2. Performance Only (Skip NCU)

```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_perf_only \
  --skip_ncu
```

### 3. Filter Problems by Dimensions

```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --filter_problems \
  --min_m 2048 --max_m 4096 \
  --output_dir results_filtered
```

### 4. Custom NCU Settings

```bash
cd cross_problem_analysis
python cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --output_dir results_3x3_custom \
  --ncu_sets full,memory,launch \
  --ncu_iterations 3
```

## Command-Line Arguments

### Required Arguments

- `--output_dir PATH`: Output directory for results
- Problem selection (one of):
  - `--problem_indices 0,1,2,...`: Comma-separated problem indices
  - `--filter_problems`: Use dimension filters (with --min_m, --max_m, etc.)

### Optional Arguments

**Input Paths:**
- `--best_configs_csv PATH`: Path to best configs CSV (default: `../collected_data/best_by_problem_v1.csv`)
- `--run_script PATH`: Path to `run_one_config.py` (default: `../run_one_config.py`)

**Performance Settings:**
- `--skip_performance`: Skip performance collection
- `--perf_iterations N`: Benchmark iterations (default: 50)
- `--perf_warmup N`: Warmup iterations (default: 5)
- `--use_cold_l2`: Use cold L2 cache

**NCU Settings:**
- `--skip_ncu`: Skip NCU profiling
- `--ncu_binary PATH`: NCU executable path (default: "ncu")
- `--ncu_sets SETS`: Metric sets, comma-separated (default: "full")
- `--ncu_metrics METRICS`: Specific metrics, comma-separated
- `--ncu_iterations N`: Profile iterations (default: 2)
- `--ncu_output_dir PATH`: NCU reports directory

**Dimension Filters** (with `--filter_problems`):
- `--min_m`, `--max_m`: M dimension range
- `--min_n`, `--max_n`: N dimension range
- `--min_k`, `--max_k`: K dimension range

**General:**
- `--quiet`: Suppress verbose output

## NCU Metrics Extraction

After collecting NCU profiles, extract hardware counter metrics:

```bash
cd cross_problem_analysis

# Extract all predefined metrics
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir metrics_summary \
  --metric-set all

# Extract specific metric sets
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir pipe_metrics \
  --metric-set pipe

# Extract custom metrics
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --output-dir custom_metrics \
  --custom-metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed" "dram__throughput.avg.pct_of_peak_sustained_elapsed"
```

Predefined metric sets:
- **pipe**: Tensor Core, FMA, ALU, FP64, LSU, TEX pipe utilization (12 metrics)
- **memory**: DRAM throughput, L1/L2 hit rates (5 metrics)
- **compute**: Duration, warps, IPC, cycles (4 metrics)
- **all**: All predefined metrics (21 metrics)

See [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md) for detailed documentation.

## cuBLAS Profiling

Profile NVIDIA's cuBLAS library for comparison with custom kernels:

```bash
cd cross_problem_analysis

# Profile cuBLAS for default problems
python profile_cublas.py

# Profile and extract metrics
python profile_cublas.py --extract-metrics

# Custom problem selection
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_profiles \
  --extract-metrics \
  --metrics-output-dir cublas_metrics

# Detailed NCU metrics
python profile_cublas.py \
  --ncu-sets detailed \
  --ncu-iterations 3 \
  --extract-metrics
```

This generates:
- NCU profiles for cuBLAS kernels in `cublas_ncu_profiles/`
- Metrics CSV in `cublas_metrics_summary/` (if `--extract-metrics` used)

See [CUBLAS_PROFILING.md](CUBLAS_PROFILING.md) for detailed guide on comparing cuBLAS vs custom kernels.

## Output Files

After a successful run, the output directory contains:

```
output_dir/
├── performance_results.csv      # Performance data for all runs
├── ncu_results.csv              # NCU profiling status for all runs
├── summary.json                 # Summary statistics and metadata
└── ncu_profiles/                # NCU report files (.ncu-rep)
    ├── prob0_cfg0_*.ncu-rep
    ├── prob0_cfg1_*.ncu-rep
    └── ...
```

### performance_results.csv

Columns:
- Problem shape: `M`, `N`, `K`, `L`
- Config params: `cta_m`, `cta_n`, `cta_k`, `stages`, `atom_m`, `atom_n`, `atom_k`, `a_major`, `b_major`, `c_major`
- Results: `success`, `elapsed_us`, `gflops`, `error`

### ncu_results.csv

Columns:
- Problem shape and config (same as above)
- Results: `success`, `output_file`, `error`

### summary.json

Contains:
- Problem subset information
- Best configuration details
- Performance statistics (avg/max/min GFLOPS)
- NCU profiling success/failure counts

## Python API Usage

You can also use the modules programmatically:

```python
from pathlib import Path
from cross_problem_analysis import (
    ConfigManager,
    KernelRunner,
    NCUProfiler,
    CrossProblemSweep
)

# Load configurations
config_mgr = ConfigManager(Path("../collected_data/best_by_problem_v1.csv"))

# Select problem subset
problems = config_mgr.get_problem_subset([0, 1, 2, 3, 4])

# Initialize runner and profiler
runner = KernelRunner(
    run_script_path=Path("../run_one_config.py"),
    iterations=50,
    warmup=5,
)

profiler = NCUProfiler(
    run_script_path=Path("../run_one_config.py"),
    output_dir=Path("ncu_profiles"),
)

# Create sweep
sweep = CrossProblemSweep(
    config_manager=config_mgr,
    kernel_runner=runner,
    ncu_profiler=profiler,
    output_dir=Path("results"),
)

# Run analysis
sweep.run_sweep(problems, run_performance=True, run_ncu=True)
```

## Module Details

### config_manager.py

**Classes:**
- `ProblemShape`: GEMM problem dimensions (M, N, K, L)
- `KernelConfig`: Kernel configuration parameters (CTA shape, stages, atom layout, majors)
- `BestConfig`: Best config for a problem with performance data
- `ConfigManager`: Load and query best configurations

**Key Methods:**
- `get_best_config(problem)`: Get best config for a problem
- `get_problem_subset(indices)`: Select problems by index
- `filter_problems(min_m, max_m, ...)`: Filter by dimensions

### kernel_runner.py

**Classes:**
- `PerformanceResult`: Result from kernel execution
- `KernelRunner`: Execute kernels via subprocess

**Key Methods:**
- `run_single_config(problem, config)`: Run one kernel
- `run_cross_problem_matrix(problems, configs)`: Run N×N matrix

### ncu_profiler.py

**Classes:**
- `NCUProfileResult`: Result from NCU profiling
- `NCUProfiler`: Profile kernels with NCU

**Key Methods:**
- `profile_single_config(problem, config)`: Profile one kernel
- `profile_cross_problem_matrix(problems, configs)`: Profile N×N matrix

## Example Workflows

### Workflow 1: Quick 3×3 Analysis

```bash
# Performance + NCU for first 3 problems
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --output_dir quick_3x3 \
  --perf_iterations 30
```

### Workflow 2: Large Problem Subset

```bash
# Filter to problems with M,N,K >= 2048
python cross_problem_analysis/cross_problem_sweep.py \
  --filter_problems \
  --min_m 2048 --min_n 2048 --min_k 2048 \
  --output_dir large_problems \
  --perf_iterations 100
```

### Workflow 3: NCU Only (Already Have Performance)

```bash
# Skip performance, only collect NCU profiles
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir ncu_only \
  --skip_performance \
  --ncu_sets full
```

### Workflow 4: Custom Metric Collection

```bash
# Collect specific NCU metrics
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --output_dir custom_metrics \
  --ncu_metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
```

### Workflow 5: Complete Custom + cuBLAS Comparison

```bash
# 1. Profile custom CUTLASS kernels
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir custom_results

# 2. Profile cuBLAS for same problems
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_results/ncu_profiles \
  --extract-metrics \
  --metrics-output-dir cublas_results/metrics

# 3. Extract metrics from custom kernels
python extract_ncu_metrics.py \
  --ncu-dir custom_results/ncu_profiles \
  --output-dir custom_results/metrics \
  --metric-set all

# 4. Compare results (CSV files in custom_results/metrics and cublas_results/metrics)
```

### Workflow 6: Pipe Utilization Analysis

```bash
cd cross_problem_analysis

# Collect only pipe utilization metrics for both custom and cuBLAS
python cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir custom_pipe

python extract_ncu_metrics.py \
  --ncu-dir custom_pipe/ncu_profiles \
  --output-dir custom_pipe/metrics \
  --metric-set pipe

python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_pipe \
  --extract-metrics

# Compare Tensor Core utilization between custom and cuBLAS
```

## Requirements

- Python 3.8+
- CUDA toolkit with `ncu` available
- CuTe DSL environment set up
- `run_one_config.py` and `tensorop_gemm_tunable.py` in parent directory

## Notes

- Each kernel execution is isolated in a subprocess to prevent CUDA context corruption
- NCU profiling can take significant time (10-15 minutes for full metric sets)
- Failed runs are logged but don't stop the sweep
- Use `--skip_ref_check` is automatically enabled for faster execution
- For very large sweeps, consider running performance and NCU separately

## Troubleshooting

**Problem: Import errors when running scripts**
```
Solution 1: Run from package directory with cd cross_problem_analysis && python script.py
Solution 2: Run as module: python -m cross_problem_analysis.script_name
Both methods are supported and work identically.
```

**Problem: NCU not found**
```
Solution: Ensure NCU is in PATH or use --ncu_binary /path/to/ncu
```

**Problem: Subprocess timeout**
```
Solution: Some configs may be very slow. Timeout is 120s for performance, 600s for NCU.
Check ncu_results.csv or performance_results.csv for timeout errors.
```

**Problem: Out of memory**
```
Solution: Reduce problem subset size or run in batches.
```

## Contact

For issues or questions about this analysis framework, refer to the main CuTe DSL documentation or the CUTLASS repository.
