# Quick Usage Examples for run_5x5_analysis.py

## Basic Usage

### 1. Run with default settings (indices 3,7,12,17,18)
```bash
./run_5x5_analysis.sh
# or
python run_5x5_analysis.py
```

### 2. Change problem indices
```bash
# Select first 5 problems (indices 0-4, rows 2-6)
./run_5x5_analysis.sh --problem_indices 0,1,2,3,4

# Select specific problems
python run_5x5_analysis.py --problem_indices 5,10,15
```

### 3. Performance only (skip NCU)
```bash
./run_5x5_analysis.sh --skip_ncu

# Faster with fewer iterations
./run_5x5_analysis.sh --skip_ncu --perf_iterations 20
```

### 4. NCU only (skip performance)
```bash
./run_5x5_analysis.sh --skip_performance
```

### 5. Custom output directory
```bash
./run_5x5_analysis.sh --problem_indices 0,1,2 --output_dir my_3x3_results
```

### 6. Lighter NCU profiling (faster)
```bash
# Use specific metric sets instead of "full"
./run_5x5_analysis.sh --ncu_sets memory,compute --ncu_iterations 1
```

### 7. List available problems
```bash
python -c "
from config_manager import ConfigManager
from pathlib import Path
mgr = ConfigManager(Path('/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv'))
for i, p in enumerate(mgr.get_all_problems()):
    bc = mgr.get_best_config(p)
    print(f'{i:2d} (row {i+2:2d}): {str(p):25s} {bc.max_gflops:8.1f} GFLOPS')
"
```

## Common Scenarios

### Small Quick Test (3 problems, no NCU)
```bash
./run_5x5_analysis.sh \
  --problem_indices 0,1,2 \
  --skip_ncu \
  --perf_iterations 20 \
  --output_dir quick_test
```
Time: ~2-3 minutes

### Medium Analysis (5 problems, performance only)
```bash
./run_5x5_analysis.sh \
  --problem_indices 3,7,12,17,18 \
  --skip_ncu \
  --output_dir perf_5x5
```
Time: ~5-10 minutes (with deduplication)

### Full Analysis with NCU
```bash
./run_5x5_analysis.sh \
  --problem_indices 3,7,12,17,18 \
  --output_dir full_5x5
```
Time: ~30-60 minutes (with deduplication, depends on unique configs)

### Large Sweep (10 problems)
```bash
./run_5x5_analysis.sh \
  --problem_indices 0,2,4,6,8,10,12,14,16,18 \
  --skip_ncu \
  --output_dir large_sweep_perf
```
Time: ~15-25 minutes

## Understanding Problem Indices

The CSV file structure:
- Row 1: Header
- Row 2: Index 0 (first problem)
- Row 3: Index 1
- ...
- Row N: Index N-2

Example:
```
--problem_indices 3,7,12,17,18
```
Corresponds to rows 4, 8, 13, 18, 19 in the CSV.

## Output Location

Default: `results_analysis/`

Structure:
```
results_analysis/
├── performance_results.csv
├── ncu_results.csv
├── summary.json
└── ncu_profiles/
    └── *.ncu-rep
```

## Tips

1. **Check for duplicates first:**
   ```bash
   python check_duplicates.py
   ```

2. **Start small:**
   Test with 2-3 problems before running large sweeps

3. **Monitor progress:**
   The script shows progress for each run

4. **Resume after failure:**
   Results are saved incrementally in CSV files
