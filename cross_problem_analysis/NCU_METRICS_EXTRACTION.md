# NCU Metrics Extraction Guide

This guide explains how to extract specific hardware counter metrics from NCU profile reports collected during cross-problem analysis.

## Overview

The NCU profiles collected during the analysis contain hundreds of hardware counter metrics. The `extract_ncu_metrics.py` script allows you to extract specific metrics of interest and generate a CSV summary that's easier to analyze.

## Quick Start

### Extract Pipe Utilization Metrics (Default)

```bash
python extract_ncu_metrics.py --metric-set pipe
```

This extracts 12 pipe utilization metrics including:
- ALU, FMA, FP64, Tensor pipe utilization
- LSU (Load/Store Unit) and TEX (Texture) pipe utilization
- Both "active" and "elapsed" percentages

### Extract All Predefined Metrics

```bash
python extract_ncu_metrics.py --metric-set all
```

This extracts comprehensive metrics including:
- Pipe utilization (12 metrics)
- Memory subsystem (5 metrics)
- Compute metrics (4 metrics)

### Extract Specific Metrics

```bash
python extract_ncu_metrics.py --metric-set memory
python extract_ncu_metrics.py --metric-set compute
```

## Custom Metrics

### Specify Your Own Metrics

```bash
python extract_ncu_metrics.py --custom-metrics \
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" \
    "dram__bytes_read.sum" \
    "l1tex__t_sector_hit_rate.pct"
```

### List All Available Metrics

To see all available metrics in your NCU profiles:

```bash
python extract_ncu_metrics.py --list-metrics results_5x5_on_v2/ncu_profiles/prob0_cfg0*.ncu-rep
```

This will display hundreds of available metrics. You can then select the ones you're interested in.

## Custom Directories

If your NCU profiles are in a different location:

```bash
python extract_ncu_metrics.py \
    --ncu-dir path/to/ncu_profiles \
    --output-dir my_custom_output \
    --metric-set pipe
```

## Understanding NCU Profile Files

Each `.ncu-rep` file can contain metrics for **multiple kernels**:
- The main GEMM kernel you're profiling
- Memory copy kernels (for data transfer)
- Other helper kernels launched during execution

The CSV output includes:
- `Profile_File`: Which .ncu-rep file the data came from
- `Kernel_ID`: Sequential ID of the kernel within that profile
- `Kernel_Name`: Name of the kernel
- `Grid_Size`, `Block_Size`: Launch configuration
- Requested metrics for each kernel

## Predefined Metric Sets

### Pipe Utilization Metrics (`--metric-set pipe`)

| Metric | Description |
|--------|-------------|
| `ALU_Pipe_Active_%` | ALU pipe utilization when SM is active |
| `ALU_Pipe_Elapsed_%` | ALU pipe utilization over total elapsed time |
| `FMA_Pipe_Active_%` | FMA pipe utilization when SM is active |
| `FMA_Pipe_Elapsed_%` | FMA pipe utilization over total elapsed time |
| `FP64_Pipe_Active_%` | FP64 pipe utilization when SM is active |
| `FP64_Pipe_Elapsed_%` | FP64 pipe utilization over total elapsed time |
| `Tensor_Pipe_Active_%` | Tensor Core pipe utilization when SM is active |
| `Tensor_Pipe_Elapsed_%` | Tensor Core pipe utilization over total elapsed time |
| `LSU_Pipe_Active_%` | Load/Store Unit pipe utilization when SM is active |
| `LSU_Pipe_Elapsed_%` | Load/Store Unit pipe utilization over total elapsed time |
| `TEX_Pipe_Active_%` | Texture unit pipe utilization when SM is active |
| `TEX_Pipe_Elapsed_%` | Texture unit pipe utilization over total elapsed time |

### Memory Metrics (`--metric-set memory`)

| Metric | Description |
|--------|-------------|
| `DRAM_Read_MB` | Total bytes read from DRAM (in MB) |
| `DRAM_Write_MB` | Total bytes written to DRAM (in MB) |
| `DRAM_Throughput_%` | DRAM throughput as % of peak |
| `L1_Hit_Rate_%` | L1 cache hit rate |
| `L2_Hit_Rate_%` | L2 cache hit rate |

### Compute Metrics (`--metric-set compute`)

| Metric | Description |
|--------|-------------|
| `SM_Active_Cycles` | Average number of cycles SM was active |
| `Warps_Active_%` | Percentage of warps active |
| `Inst_Per_Cycle` | Instructions executed per cycle |
| `Duration_us` | Kernel duration in microseconds |

## Programmatic Usage

You can also use the extractor as a Python module:

```python
from ncu_metrics_extractor import (
    NCUMetricsExtractor,
    PIPE_UTILIZATION_METRICS,
    create_custom_metric_set
)

# Extract predefined metrics
extractor = NCUMetricsExtractor(
    ncu_profiles_dir="results_5x5_on_v2/ncu_profiles",
    output_dir="my_metrics"
)

output_path = extractor.extract_metrics_from_all_profiles(
    PIPE_UTILIZATION_METRICS,
    "pipe_metrics.csv"
)

# Extract custom metrics
custom_metrics = create_custom_metric_set([
    "sm__cycles_active.avg",
    "gpu__time_duration.sum",
    "dram__bytes.sum"
])

output_path = extractor.extract_metrics_from_all_profiles(
    custom_metrics,
    "my_custom_metrics.csv"
)
```

## Output Format

The generated CSV has the following structure:

```csv
Profile_File,Kernel_ID,Kernel_Name,Grid_Size,Block_Size,ALU_Pipe_Active_%,FMA_Pipe_Active_%,...
prob0_cfg0_M256_N2048_K8192_L1_cta64x128x64_s4_atom2x2x1_mnn.ncu-rep,0,at::native::unrolled_elementwise_kernel,32768,128,31.06,48.22,...
prob0_cfg0_M256_N2048_K8192_L1_cta64x128x64_s4_atom2x2x1_mnn.ncu-rep,1,cutlass::Kernel<cutlass_80_simt_sgemm>,1024,256,5.94,8.33,...
prob0_cfg0_M256_N2048_K8192_L1_cta64x128x64_s4_atom2x2x1_mnn.ncu-rep,2,cutlass_kernel_tensorop_gemm,2048,128,13.41,9.18,...
...
```

Each row represents a kernel execution from the NCU profiles.

## Tips

1. **Focus on Main GEMM Kernels**: If you only care about the main GEMM kernel performance, filter the CSV by `Kernel_Name` containing "tensorop_gemm" or "simt"

2. **Compare Across Configurations**: Use the `Profile_File` column to identify which problem-config combination each kernel came from

3. **Active vs Elapsed**: 
   - "Active" percentages show utilization when the SM is active
   - "Elapsed" percentages show utilization over the entire kernel duration
   - For compute-bound kernels, focus on "Active" metrics
   - For memory-bound kernels, "Elapsed" may show idle time waiting for data

4. **Batch Processing**: You can extract different metric sets and combine them later using pandas or other tools

## Example Analysis Workflow

```bash
# 1. Extract comprehensive metrics
python extract_ncu_metrics.py --metric-set all --output-filename comprehensive.csv

# 2. Load in Python/Jupyter for analysis
import pandas as pd
df = pd.read_csv('ncu_metrics_summary/comprehensive.csv')

# 3. Filter for main GEMM kernels
gemm_kernels = df[df['Kernel_Name'].str.contains('tensorop_gemm')]

# 4. Analyze pipe utilization patterns
print(gemm_kernels[['Profile_File', 'Tensor_Pipe_Active_%', 'FMA_Pipe_Active_%']].describe())
```

## Troubleshooting

**Q: The script says "No NCU profile files found"**
A: Make sure you're pointing to the correct directory with `--ncu-dir`, and that it contains `.ncu-rep` files.

**Q: Some metrics show empty values**
A: Not all metrics are available for all kernels. This is expected, especially for kernels that don't use certain hardware units.

**Q: The extraction is slow**
A: NCU profile parsing can take time. For 25 profile files with multiple kernels each, expect 1-2 minutes of processing.

**Q: How do I find the right metric name?**
A: Use `--list-metrics` with a sample profile file to see all available metrics.

## Integration with Cross-Problem Analysis

After running your cross-problem analysis:

```bash
# Run your analysis
./run_5x5_analysis.sh

# Extract metrics from the collected NCU profiles
python extract_ncu_metrics.py --metric-set all

# Now you have:
# - results_5x5_on_v2/performance_results.csv (runtime and GFLOPS)
# - results_5x5_on_v2/ncu_results.csv (basic NCU metadata)
# - ncu_metrics_summary/comprehensive_metrics.csv (detailed hardware counters)
```

All three CSV files can be joined using the problem shape and configuration information.
