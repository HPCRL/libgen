# Split-K GEMM Implementation for CuTe DSL

This directory contains a split-K parallelism implementation of the CuTe DSL tensorcore GEMM kernel, designed to improve performance for GEMM problems with small M and/or N dimensions but large K dimension.

## Files Created

### Core Implementation
- **tensorop_gemm_splitk.py**: Split-K version of the CuTe GEMM kernel
  - Parallelizes computation across the K dimension
  - Each thread block computes a partial result for a K-slice
  - Includes a reduction kernel to combine partial results
  - Compatible with the same tuning parameters as the base implementation

### Sweep Infrastructure
- **run_one_config_splitk.py**: Isolated runner for single split-K configurations
  - Returns JSON results for robust sweeping
  - Handles timeouts and errors gracefully

- **sweep_tensorop_gemm_splitk.py**: Parameter sweep script
  - Sweeps over CTA tiles, stages, atom layouts, layouts, AND split_k values
  - Records only configurations that pass correctness checks
  - Real-time CSV output for monitoring progress

- **tune_config_splitk.yaml**: Configuration file for parameter sweeps
  - Includes all CTA tile configurations
  - Split-K values: [1, 2, 4, 8, 16, 32, 64, 128]
  - Same stages, atom layouts, and layout options as base version

## Split-K Overview

### What is Split-K?
Split-K parallelism divides the K dimension (reduction dimension) across multiple thread blocks. Instead of having each thread block compute an entire M×N output tile, multiple blocks compute partial sums for different K-slices of the same output tile. A reduction phase then combines these partial results.

### When to Use Split-K?
Split-K is particularly beneficial for:
- **Small M and/or N**: When there aren't enough output tiles for good GPU utilization
- **Large K**: When the K dimension is large, providing many K-slices to parallelize over
- **Example**: 64×64×8192 GEMM (small M/N, large K) - shows ~5x speedup with split_k=16

### Performance Results (64×64×8192, fp16)
```
split_k=1:  103.01 us  (baseline - insufficient parallelism)
split_k=16:  20.07 us  (~5.1x speedup)
split_k=32:  21.30 us  (~4.8x speedup)
```

## Usage

### Run a single configuration
```bash
python tensorop_gemm_splitk.py \
  --mnkl 64,64,8192,1 \
  --split_k 16 \
  --cta_tiler 64,64,32 \
  --atom_layout_mnk 2,2,1 \
  --num_stages 3 \
  --iterations 100
```

### Run a parameter sweep
```bash
# Create a problems CSV file first (e.g., problems_small_mn.csv with small M/N, large K problems)
echo "m,n,k" > problems_small_mn.csv
echo "64,64,8192" >> problems_small_mn.csv
echo "128,128,4096" >> problems_small_mn.csv
echo "64,128,8192" >> problems_small_mn.csv

# Run sweep
python sweep_tensorop_gemm_splitk.py \
  --problems_csv problems_small_mn.csv \
  --config tune_config_splitk.yaml \
  --out sweep_results_splitk.csv
```

### Compare with cuBLAS
After running sweeps, you can:
1. Find the best split-K configuration for each problem size
2. Compare against cuBLAS results from `cublas_bench`
3. Select the best-performing CuTe kernels for your library

## Key Differences from Base Implementation

1. **Additional Parameter**: `split_k` controls parallelism degree
2. **Workspace Tensor**: Requires an additional workspace tensor for partial results (M×N×L×split_k)
3. **Two-Phase Execution**: 
   - Phase 1: Compute partial GEMMs
   - Phase 2: Reduce partial results and apply epilogue
4. **Grid Dimensions**: Z-dimension includes both batch (L) and split_k factor

## Implementation Details

### Split-K Kernel (`kernel_splitk`)
- Each thread block processes a subset of the K dimension
- K range: `[split_idx * k_per_split, min((split_idx+1) * k_per_split, K)]`
- Writes partial results to workspace tensor
- No epilogue applied at this stage (keeps accumulator precision)

### Reduction Kernel (`kernel_reduce`)
- Simple reduction across split_k dimension
- Each thread processes multiple output elements
- Applies epilogue fusion (currently identity, but extensible)
- Converts from accumulator precision to output precision

## Tuning Recommendations

For small M/N problems:
1. Start with `split_k` values in [8, 16, 32]
2. Use smaller CTA tiles (16×16, 32×32, 64×64) for better K-slice granularity
3. Balance split_k with CTA K-tile size (bK) to ensure each split has multiple K-tiles

## Future Enhancements

Potential improvements for production use:
- [ ] Atomic reduction option for very small M/N
- [ ] Stream-K parallelism (more dynamic work distribution)
- [ ] Heuristic to automatically choose split_k based on problem shape
- [ ] Fused reduction in shared memory to reduce global memory traffic
- [ ] Support for larger split_k values with efficient reduction trees

## Comparison with Original Implementation

The base `tensorop_gemm_tunable.py`:
- Best for problems with sufficient M×N parallelism
- No reduction overhead
- Simpler kernel structure

This split-K version:
- Better for small M/N, large K problems
- Adds reduction overhead (usually negligible compared to GEMM)
- More complex but more versatile for varied problem shapes

Choose based on your target problem distribution!
