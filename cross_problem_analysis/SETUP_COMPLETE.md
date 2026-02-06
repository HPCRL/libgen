# Cross-Problem Kernel Analysis - Setup Complete ✓

## What Was Created

A complete, modular framework for analyzing CuTe DSL GEMM kernel configurations across problem subsets. The system enables collecting performance metrics and NCU profiles for an N×N matrix of (problems × configurations).

## Directory Structure

```
cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/
├── run_one_config.py                    # (existing) Single config runner
├── tensorop_gemm_tunable.py             # (existing) Tunable GEMM kernel
├── collected_data/
│   ├── sweep_results_v1.csv             # (existing) All sweep results
│   └── best_by_problem_v1.csv           # (existing) Best configs per problem
└── cross_problem_analysis/              # ← NEW PACKAGE
    ├── __init__.py                      # Package exports
    ├── config_manager.py                # Load/manage configurations (191 lines)
    ├── kernel_runner.py                 # Execute kernels & collect perf (221 lines)
    ├── ncu_profiler.py                  # NCU profiling wrapper (240 lines)
    ├── cross_problem_sweep.py           # Main orchestration (392 lines)
    ├── example_usage.py                 # Usage examples (167 lines)
    ├── test_basic.py                    # Basic tests (194 lines)
    ├── README.md                        # Full documentation
    ├── QUICKREF.md                      # Quick reference
    └── IMPLEMENTATION.md                # Implementation details
```

## Quick Start

### Test the Installation
```bash
conda activate cutlass-pdsl
cd /media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere
python cross_problem_analysis/test_basic.py
```
**Status**: ✓ All 5 tests passed

### Run Your First 5×5 Analysis
```bash
# Performance + NCU for first 5 problems (25 runs)
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir my_first_5x5_analysis
```

### Quick Test (3×3, Performance Only)
```bash
# Faster test without NCU (9 runs, ~3-4 minutes)
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --skip_ncu \
  --output_dir quick_test_3x3
```

## Key Features

✓ **Modular Design**: Each component can be used independently  
✓ **Robust Execution**: Subprocess isolation prevents context corruption  
✓ **Flexible Selection**: Choose problems by index or dimension filters  
✓ **Comprehensive Output**: CSV results, JSON summary, NCU profiles  
✓ **CLI & API**: Use from command line or Python scripts  
✓ **Error Handling**: Graceful degradation, detailed error reporting  

## Common Use Cases

### 1. Analyze First 5 Problems
```bash
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_5x5
```

### 2. Large Problems Only
```bash
python cross_problem_analysis/cross_problem_sweep.py \
  --filter_problems --min_m 2048 --min_n 2048 \
  --output_dir results_large
```

### 3. High-Quality Performance Benchmark
```bash
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --perf_iterations 100 --perf_warmup 10 \
  --use_cold_l2 --skip_ncu \
  --output_dir results_hq_perf
```

### 4. NCU Only (Already Have Performance)
```bash
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --skip_performance \
  --output_dir results_ncu_only
```

## Python API Example

```python
from pathlib import Path
from cross_problem_analysis import ConfigManager, KernelRunner, CrossProblemSweep

# Load configurations
config_mgr = ConfigManager(Path("collected_data/best_by_problem_v1.csv"))

# Select problems
problems = config_mgr.get_problem_subset([0, 1, 2, 3, 4])

# Initialize runner
runner = KernelRunner(
    run_script_path=Path("run_one_config.py"),
    iterations=50,
    warmup=5
)

# Create and run sweep
sweep = CrossProblemSweep(config_mgr, runner, None, Path("output"))
sweep.run_sweep(problems, run_performance=True, run_ncu=False)
```

## Output Structure

After running a sweep, you'll get:

```
output_dir/
├── performance_results.csv       # All performance measurements
│   ├── Columns: M, N, K, L, cta_*, stages, atom_*, majors
│   ├── Results: success, elapsed_us, gflops, error
│   └── One row per (problem, config) combination
│
├── ncu_results.csv              # NCU profiling status
│   ├── Same problem/config columns
│   └── Results: success, output_file, error
│
├── summary.json                 # Statistics & metadata
│   ├── Problem information
│   ├── Best config details
│   ├── Performance statistics (avg/max/min GFLOPS)
│   └── Success/failure counts
│
└── ncu_profiles/                # NCU report files
    ├── prob0_cfg0_*.ncu-rep
    ├── prob0_cfg1_*.ncu-rep
    └── ...
```

## Time Estimates

| Configuration | Per Run | 3×3 Total | 5×5 Total | 10×10 Total |
|--------------|---------|-----------|-----------|-------------|
| Perf only (50 iters) | ~15s | ~2 min | ~6 min | ~25 min |
| NCU (full set) | ~3-5 min | ~45 min | ~2 hrs | ~8 hrs |

## Documentation

- **README.md**: Comprehensive documentation with examples
- **QUICKREF.md**: Quick reference for common commands
- **IMPLEMENTATION.md**: Architecture and design details
- **example_usage.py**: Interactive examples (`--example 1/2/3`)

## Validation

All modules tested and verified:
- ✓ Module imports
- ✓ Configuration loading (20 problems loaded)
- ✓ Problem filtering (subset selection works)
- ✓ Data structures (ProblemShape, KernelConfig)
- ✓ Runner initialization

## Next Steps

### Option 1: Quick Test Run
```bash
# Run a small 3×3 test (no NCU, fast)
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2 \
  --skip_ncu \
  --perf_iterations 20 \
  --output_dir test_3x3
```

### Option 2: Your 5×5 Analysis
```bash
# Full 5×5 with performance and NCU
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir analysis_5x5
```

### Option 3: Custom Subset
```bash
# Use dimension filters to select problems
python cross_problem_analysis/cross_problem_sweep.py \
  --filter_problems \
  --min_m 2048 --max_m 4096 \
  --output_dir analysis_custom
```

## Getting Help

```bash
# Full help
python cross_problem_analysis/cross_problem_sweep.py --help

# Run examples
python cross_problem_analysis/example_usage.py --example 2  # Info only
python cross_problem_analysis/example_usage.py --example 3  # Single kernel

# View available problems
python -c "
from cross_problem_analysis import ConfigManager
from pathlib import Path
mgr = ConfigManager(Path('collected_data/best_by_problem_v1.csv'))
for i, p in enumerate(mgr.get_all_problems()):
    print(f'{i}: {p}')
"
```

## System Requirements

- ✓ Python 3.8+ with conda environment `cutlass-pdsl`
- ✓ CUDA Toolkit with NCU (for profiling)
- ✓ CuTe DSL environment configured
- ✓ All dependencies already available in your environment

## Support Files

All key files are in place:
- ✓ `run_one_config.py` - Single config execution helper
- ✓ `tensorop_gemm_tunable.py` - Tunable GEMM kernel
- ✓ `collected_data/best_by_problem_v1.csv` - Best configurations

**Ready to use!** Start with a small test or jump into your full analysis.
