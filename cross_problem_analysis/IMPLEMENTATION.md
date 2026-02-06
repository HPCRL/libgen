# Cross-Problem Kernel Analysis - Implementation Summary

## Overview

A complete modular framework for analyzing CuTe DSL GEMM kernel configurations across multiple problem shapes. The system supports:

- **Performance benchmarking**: Collect execution time and GFLOPS for kernel configs
- **NCU profiling**: Gather hardware counter data using NVIDIA Nsight Compute
- **Cross-problem analysis**: Run best configs from N problems on all N problems (N×N matrix)

## Architecture

### Module Organization

```
cross_problem_analysis/
├── __init__.py                 # Package exports
├── config_manager.py           # Configuration & problem shape management
├── kernel_runner.py            # Kernel execution & performance collection
├── ncu_profiler.py            # NCU profiling wrapper
├── cross_problem_sweep.py     # Main orchestration script
├── example_usage.py           # Usage examples
├── README.md                  # Full documentation
└── QUICKREF.md                # Quick reference guide
```

### Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Robustness**: Subprocess isolation prevents CUDA context corruption
3. **Flexibility**: Easy to extend or use components independently
4. **Usability**: Both CLI and Python API interfaces

## Module Details

### 1. config_manager.py

**Purpose**: Load and manage kernel configurations and problem shapes

**Classes**:
- `ProblemShape`: Represents GEMM dimensions (M, N, K, L)
- `KernelConfig`: Kernel parameters (CTA shape, stages, atom layout, majors)
- `BestConfig`: Associates problem with its best configuration
- `ConfigManager`: Main interface for loading and querying configs

**Key Features**:
- Load best configurations from CSV
- Select problem subsets by indices or dimension filters
- Query best config for any problem shape

### 2. kernel_runner.py

**Purpose**: Execute kernel configurations and collect performance metrics

**Classes**:
- `PerformanceResult`: Execution result with timing and GFLOPS
- `KernelRunner`: Orchestrates kernel execution via subprocess

**Key Features**:
- Subprocess isolation for robustness
- JSON-based communication with `run_one_config.py`
- Automatic GFLOPS calculation
- Configurable iterations and warmup
- Cross-problem matrix execution (N×N runs)

### 3. ncu_profiler.py

**Purpose**: Profile kernels with NVIDIA Nsight Compute

**Classes**:
- `NCUProfileResult`: Profiling result with output file path
- `NCUProfiler`: Wrapper for NCU profiling

**Key Features**:
- Configurable metric sets and specific metrics
- Automatic output file naming
- Subprocess-based execution
- NCU availability checking
- Cross-problem matrix profiling

### 4. cross_problem_sweep.py

**Purpose**: Main orchestration script

**Features**:
- CLI interface with comprehensive options
- Coordinated performance and NCU collection
- Progress tracking and status reporting
- CSV and JSON output generation
- Summary statistics

## Usage Patterns

### Pattern 1: Command-Line Quick Start

```bash
# Basic 5×5 analysis
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir results_5x5
```

### Pattern 2: Performance-Only Analysis

```bash
# Skip NCU for faster execution
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --skip_ncu \
  --output_dir results_perf
```

### Pattern 3: Filtered Problem Selection

```bash
# Select problems by dimensions
python cross_problem_analysis/cross_problem_sweep.py \
  --filter_problems \
  --min_m 2048 --max_m 4096 \
  --output_dir results_filtered
```

### Pattern 4: Python API

```python
from cross_problem_analysis import ConfigManager, KernelRunner, CrossProblemSweep

config_mgr = ConfigManager("best_by_problem_v1.csv")
problems = config_mgr.get_problem_subset([0, 1, 2, 3, 4])

runner = KernelRunner("run_one_config.py", iterations=50)
sweep = CrossProblemSweep(config_mgr, runner, None, Path("output"))
sweep.run_sweep(problems, run_performance=True, run_ncu=False)
```

## Output Format

### Directory Structure
```
output_dir/
├── performance_results.csv    # All performance data
├── ncu_results.csv           # NCU profiling status
├── summary.json              # Statistics & metadata
└── ncu_profiles/             # NCU report files
    ├── prob0_cfg0_*.ncu-rep
    ├── prob0_cfg1_*.ncu-rep
    └── ...
```

### CSV Schemas

**performance_results.csv**:
- Problem: M, N, K, L
- Config: cta_m, cta_n, cta_k, stages, atom_m, atom_n, atom_k, a_major, b_major, c_major
- Results: success, elapsed_us, gflops, error

**ncu_results.csv**:
- Same problem and config fields
- Results: success, output_file, error

### JSON Summary

Contains:
- Problem subset information
- Best configuration details
- Performance statistics (avg/max/min GFLOPS)
- Success/failure counts

## Error Handling

### Graceful Degradation
- Failed kernel executions don't stop the sweep
- Errors are logged in result files
- Subprocess isolation prevents context corruption

### Timeout Protection
- Performance: 120 second timeout per run
- NCU: 600 second timeout per profile
- Configurable in source code if needed

## Extensibility

### Adding New Metrics

Extend `PerformanceResult` in `kernel_runner.py`:
```python
@dataclass
class PerformanceResult:
    # ... existing fields ...
    sm_efficiency: float | None  # New metric
```

### Custom Problem Filtering

Add method to `ConfigManager`:
```python
def filter_by_gflops(self, min_gflops: float):
    return [p for p, bc in self.best_configs.items() 
            if bc.max_gflops >= min_gflops]
```

### Additional NCU Metrics

Specify in CLI or API:
```bash
--ncu_metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
```

## Performance Considerations

### Time Estimates

| Configuration | Time per Run | 5×5 Total | 10×10 Total |
|--------------|--------------|-----------|-------------|
| Perf only (50 iters) | ~15-20s | ~6-8 min | ~25-35 min |
| NCU full set | ~3-5 min | ~1-2 hrs | ~5-8 hrs |
| NCU memory set | ~1-2 min | ~25-50 min | ~2-3 hrs |

### Optimization Tips

1. **Parallel execution**: Currently sequential; could parallelize across GPUs
2. **Batch profiling**: NCU supports multiple kernels in one invocation
3. **Metric selection**: Use specific metric sets instead of "full"
4. **Warm vs cold L2**: Skip `--use_cold_l2` for faster benchmarks

## Testing

Run examples to verify setup:
```bash
# Info-only example (no kernel execution)
python cross_problem_analysis/example_usage.py --example 2

# Single kernel test
python cross_problem_analysis/example_usage.py --example 3

# Small 3×3 sweep
python cross_problem_analysis/example_usage.py --example 1
```

## Requirements

- Python 3.8+
- CUDA Toolkit with NCU
- CuTe DSL environment
- Required files in parent directory:
  - `run_one_config.py`
  - `tensorop_gemm_tunable.py`
  - `collected_data/best_by_problem_v1.csv`

## Known Limitations

1. **Sequential execution**: Runs one kernel at a time
2. **Single GPU**: Doesn't distribute across multiple GPUs
3. **Fixed data types**: Uses fp16/fp32 (modifiable in source)
4. **Timeout limits**: Very slow configs may timeout

## Future Enhancements

Potential improvements:
- [ ] Multi-GPU support
- [ ] Parallel execution
- [ ] Real-time progress visualization
- [ ] Result analysis and plotting tools
- [ ] Automatic outlier detection
- [ ] Resume capability for interrupted sweeps
- [ ] Compressed NCU report storage

## Support

- Full documentation: `README.md`
- Quick reference: `QUICKREF.md`
- Examples: `example_usage.py`
- Main script help: `python cross_problem_sweep.py --help`

## License

Follows CUTLASS BSD-3-Clause license (same as parent project).
