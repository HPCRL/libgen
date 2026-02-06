# 5×5 Cross-Problem Analysis - Quick Start

This directory contains scripts to run your specific 5×5 analysis.

## Selected Problems

The analysis uses these 5 problems (from best_by_problem_v1.csv):

| Row | M    | N    | K     | Best GFLOPS | Description |
|-----|------|------|-------|-------------|-------------|
| 4   | 256  | 2048 | 8192  | 49,914      | Medium, varied |
| 8   | 256  | 8192 | 2048  | 59,850      | Medium, tall |
| 13  | 2048 | 2048 | 16384 | 64,900      | Large, square |
| 18  | 4096 | 128  | 4096  | 47,846      | Large, skinny |
| 19  | 4096 | 4096 | 4096  | 75,285      | Very large, square |

## What It Does

1. **Performance Collection**: Runs all 5 best configs on all 5 problems (25 runs)
   - Measures execution time and GFLOPS
   - **Correctness checking ENABLED** (validates results)
   - 50 iterations + 5 warmup per run
   - ~10-15 minutes total

2. **NCU Profiling**: Collects hardware counter profiles for all 25 runs
   - Full metric set
   - 2 iterations + 1 warmup per profile
   - ~2-3 hours total

## How to Run

### Option 1: Bash Script (Easiest)
```bash
cd /media/datassd/sina/libgen/cross_problem_analysis
./run_5x5_analysis.sh
```

### Option 2: Python Script Directly
```bash
cd /media/datassd/sina/libgen/cross_problem_analysis
conda activate cutlass-pdsl
python run_5x5_analysis.py
```

### Option 3: Skip NCU (Performance Only, ~15 min)
```bash
# Edit run_5x5_analysis.py and set:
#   ncu_sets=None  # or comment out NCU profiler initialization
python run_5x5_analysis.py
```

## Output

Results are saved to: `results_5x5_analysis/`

```
results_5x5_analysis/
├── performance_results.csv       # All 25 performance measurements
│   Columns: M, N, K, L, config params, success, elapsed_us, gflops, error
│
├── ncu_results.csv               # NCU profiling status
│   Columns: problem, config, success, output_file, error
│
├── summary.json                  # Statistics and metadata
│   - Problem list with best configs
│   - Performance stats (avg/max/min GFLOPS)
│   - Failure breakdown (correctness vs runtime)
│   - Success/failure counts
│
└── ncu_profiles/                 # NCU report files
    ├── prob0_cfg0_*.ncu-rep
    ├── prob0_cfg1_*.ncu-rep
    └── ... (25 files total)
```

## Viewing Results

### Performance Data
```bash
# View all results
cat results_5x5_analysis/performance_results.csv

# Only successful runs
grep ",True," results_5x5_analysis/performance_results.csv

# Sort by GFLOPS
python -c "
import pandas as pd
df = pd.read_csv('results_5x5_analysis/performance_results.csv')
print(df[df['success']].sort_values('gflops', ascending=False))
"
```

### Summary Statistics
```bash
cat results_5x5_analysis/summary.json | python -m json.tool
```

### NCU Reports
```bash
# List all NCU reports
ls -lh results_5x5_analysis/ncu_profiles/

# Open specific report in NCU UI
ncu-ui results_5x5_analysis/ncu_profiles/prob0_cfg0_*.ncu-rep

# Compare two reports
ncu-ui results_5x5_analysis/ncu_profiles/prob0_cfg0_*.ncu-rep \
       results_5x5_analysis/ncu_profiles/prob0_cfg1_*.ncu-rep
```

## Expected Runtime

| Component | Time | Notes |
|-----------|------|-------|
| Performance (25 runs) | ~10-15 min | 50 iters each, with correctness check |
| NCU Profiling (25 runs) | ~2-3 hours | Full metric set, 2 iters each |
| **Total** | **~2.5-3 hours** | For complete analysis |

## Troubleshooting

### NCU Not Found
```bash
# Check if NCU is available
which ncu

# If not in PATH, edit run_5x5_analysis.py:
profiler = NCUProfiler(
    ...
    ncu_binary="/usr/local/cuda/bin/ncu",  # Full path
)
```

### Some Runs Fail
- Check `error` column in performance_results.csv
- `[skip]`: Correctness failure (wrong results)
- `[fail]`: Runtime error (compile/crash)
- Review summary.json for failure breakdown

### Out of Memory
- NCU profiling uses extra memory
- Try closing other GPU applications
- Or run performance collection first, then NCU separately

## Customization

Edit `run_5x5_analysis.py` to customize:

```python
# Change iterations
runner = KernelRunner(
    iterations=100,  # More iterations for better accuracy
    warmup=10,
)

# Change NCU metric sets
profiler = NCUProfiler(
    ncu_sets=["memory", "compute"],  # Faster, specific metrics
)

# Skip NCU entirely
profiler = None  # Only collect performance

# Change output location
output_dir = Path("/path/to/my/results")
```

## Next Steps

After the analysis completes:

1. Check `summary.json` for overview
2. Analyze `performance_results.csv` for performance patterns
3. Use NCU UI to explore hardware counter data
4. Identify best configs for each problem
5. Look for configs that generalize well across problems
