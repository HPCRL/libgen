# Cross-Problem Analysis - Documentation Index

Welcome! This document helps you navigate all available documentation.

## Getting Started

**New users start here:**

1. **[README.md](README.md)** - Main package documentation
   - Overview of the system
   - Module structure
   - Quick start guide
   - Command-line arguments reference
   - Output file formats

2. **[QUICKREF.md](QUICKREF.md)** - Quick reference guide
   - 17 common use cases with ready-to-run commands
   - Complete workflow examples
   - Time estimates
   - Quick troubleshooting

3. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Setup validation
   - Verify your environment is ready
   - Test basic functionality
   - Ensure dependencies are installed

## Feature-Specific Guides

### Core Features

**[RUN_5x5_GUIDE.md](RUN_5x5_GUIDE.md)** - Step-by-step 5×5 analysis guide
- Detailed walkthrough for first-time users
- What to expect at each step
- How to interpret results
- Common issues and solutions

**[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Practical examples
- Real-world usage scenarios
- Python API examples
- Advanced configurations
- Tips and tricks

### Advanced Features

**[NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md)** - Extract hardware counter metrics
- How to extract specific metrics from NCU profiles
- Predefined metric sets (pipe, memory, compute)
- Custom metric specification
- CSV output format
- Analysis examples

**[CUBLAS_PROFILING.md](CUBLAS_PROFILING.md)** - Profile NVIDIA cuBLAS library
- Profile cuBLAS for baseline comparison
- Side-by-side comparison with custom kernels
- Key metrics to compare
- Complete comparison workflow
- Troubleshooting cuBLAS profiling

## Technical Documentation

**[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Implementation details
- Module architecture
- Class relationships
- Error handling strategy
- Performance considerations
- Extension points

**[DEDUPLICATION.md](DEDUPLICATION.md)** - Configuration deduplication
- How duplicate detection works
- Performance savings
- check_duplicates.py tool usage

## Quick Navigation by Task

### I want to...

**Run my first analysis:**
→ Start with [QUICKREF.md](QUICKREF.md) Use Case #1, then read [RUN_5x5_GUIDE.md](RUN_5x5_GUIDE.md)

**Compare my kernels with cuBLAS:**
→ Read [CUBLAS_PROFILING.md](CUBLAS_PROFILING.md) + [QUICKREF.md](QUICKREF.md) Use Case #16

**Extract specific hardware counter metrics:**
→ Read [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md) + [QUICKREF.md](QUICKREF.md) Use Cases #12-15

**Understand pipe utilization (Tensor Cores, LSU, etc.):**
→ [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md) → Predefined Metrics section

**Analyze large problem subsets:**
→ [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) + [QUICKREF.md](QUICKREF.md) Use Case #4

**Use the package programmatically (Python API):**
→ [README.md](README.md) → Python API Usage section + [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)

**Understand performance bottlenecks:**
→ Run analysis, then read [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md) for metric interpretation

**Debug issues:**
→ [QUICKREF.md](QUICKREF.md) → Troubleshooting section + relevant feature guide

**Extend or modify the package:**
→ [IMPLEMENTATION.md](IMPLEMENTATION.md) → Extension points

## Documentation Features Summary

| Document | Length | Best For |
|----------|--------|----------|
| **QUICKREF.md** | 2-3 min read | Quick commands, common tasks |
| **README.md** | 10 min read | Understanding system architecture |
| **CUBLAS_PROFILING.md** | 15 min read | Baseline comparisons |
| **NCU_METRICS_EXTRACTION.md** | 15 min read | Hardware counter analysis |
| **RUN_5x5_GUIDE.md** | 5 min read | First-time walkthrough |
| **USAGE_EXAMPLES.md** | 10 min read | Practical recipes |
| **IMPLEMENTATION.md** | 20 min read | Technical deep-dive |

## Recommended Learning Path

### Beginner Path (30 minutes)
1. Read [README.md](README.md) → Overview section
2. Run commands from [QUICKREF.md](QUICKREF.md) Use Case #1
3. Follow [RUN_5x5_GUIDE.md](RUN_5x5_GUIDE.md)
4. Explore results with [QUICKREF.md](QUICKREF.md) analysis commands

### Intermediate Path (1 hour)
1. Complete Beginner Path
2. Read [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md)
3. Extract metrics from your profiles
4. Read [CUBLAS_PROFILING.md](CUBLAS_PROFILING.md)
5. Profile cuBLAS and compare

### Advanced Path (2 hours)
1. Complete Intermediate Path
2. Read [IMPLEMENTATION.md](IMPLEMENTATION.md)
3. Read [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) → Python API section
4. Experiment with custom configurations
5. Build your own analysis scripts

## Common Workflows with Documentation

### Workflow 1: Basic Performance Analysis
```bash
# Follow: QUICKREF.md Use Case #1
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 --output_dir results_5x5
# Then: RUN_5x5_GUIDE.md for result interpretation
```

### Workflow 2: Hardware Counter Deep-Dive
```bash
# Step 1 (QUICKREF.md Use Case #1)
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 --output_dir results_5x5

# Step 2 (NCU_METRICS_EXTRACTION.md)
python extract_ncu_metrics.py \
  --ncu-dir results_5x5/ncu_profiles \
  --metric-set all --output-dir metrics

# Analysis: NCU_METRICS_EXTRACTION.md → Analyzing Results section
```

### Workflow 3: Custom vs cuBLAS Comparison
```bash
# Complete workflow: CUBLAS_PROFILING.md → Complete Workflow Example
# Also see: QUICKREF.md Use Case #16

# Custom kernels
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 --output_dir custom_results

# cuBLAS baseline
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 --extract-metrics

# Metrics extraction
python extract_ncu_metrics.py \
  --ncu-dir custom_results/ncu_profiles \
  --metric-set all --output-dir custom_metrics

# Comparison: CUBLAS_PROFILING.md → Comparing with Custom Kernels
```

### Workflow 4: Pipe Utilization Study
```bash
# Focus on Tensor Core, LSU, ALU utilization
# Guide: NCU_METRICS_EXTRACTION.md → Predefined Metrics → pipe

# Custom
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 --output_dir custom_pipe

python extract_ncu_metrics.py \
  --ncu-dir custom_pipe/ncu_profiles \
  --metric-set pipe --output-dir custom_pipe_metrics

# cuBLAS
python profile_cublas.py \
  --problem-indices 0,1,2,3,4 \
  --output-dir cublas_pipe --extract-metrics

# Compare: CUBLAS_PROFILING.md → Side-by-Side Metrics
```

## Help & Support

### Quick Help Commands

```bash
# Get help for any script
python cross_problem_analysis/cross_problem_sweep.py --help
python profile_cublas.py --help
python extract_ncu_metrics.py --help

# List available metrics in a profile
python extract_ncu_metrics.py --list-metrics path/to/profile.ncu-rep

# Check for duplicate configs
python check_duplicates.py
```

### Troubleshooting Strategy

1. **Check [QUICKREF.md](QUICKREF.md) → Troubleshooting** first
2. For feature-specific issues:
   - cuBLAS: [CUBLAS_PROFILING.md](CUBLAS_PROFILING.md) → Troubleshooting
   - Metrics: [NCU_METRICS_EXTRACTION.md](NCU_METRICS_EXTRACTION.md) → Troubleshooting
3. For general issues: [README.md](README.md) → Troubleshooting
4. For technical details: [IMPLEMENTATION.md](IMPLEMENTATION.md)

## Updates & Version History

This documentation set was last updated for the package version that includes:
- ✅ Cross-problem kernel analysis (core feature)
- ✅ NCU profiling with customizable metric sets
- ✅ NCU metrics extraction with predefined sets
- ✅ cuBLAS profiling for baseline comparison
- ✅ Configuration deduplication
- ✅ Correctness checking (enabled by default)
- ✅ Full CLI and Python API

## Contributing

If you extend this package or find issues:
1. Update relevant documentation files
2. Add examples to [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
3. Update this index if you add new docs
4. Keep [QUICKREF.md](QUICKREF.md) concise and practical

## File Organization

```
cross_problem_analysis/
├── README.md                      # Main documentation (start here)
├── QUICKREF.md                    # Quick reference (most useful)
├── DOCUMENTATION_INDEX.md         # This file
├── SETUP_COMPLETE.md              # Setup validation
├── RUN_5x5_GUIDE.md              # First-time user guide
├── USAGE_EXAMPLES.md              # Practical examples
├── CUBLAS_PROFILING.md           # cuBLAS profiling guide
├── NCU_METRICS_EXTRACTION.md     # Metrics extraction guide
├── IMPLEMENTATION.md              # Technical details
├── DEDUPLICATION.md              # Config deduplication
└── [code files...]
```

## Quick Start for Impatient Users

**Just want to run something NOW?**

```bash
# Copy-paste this (takes ~5-10 minutes):
cd /media/datassd/sina/libgen
python cross_problem_analysis/cross_problem_sweep.py \
  --problem_indices 0,1,2,3,4 \
  --output_dir quick_test

# Then look at results:
cat quick_test/summary.json | python -m json.tool
```

**Want to compare with cuBLAS?**

```bash
python profile_cublas.py --extract-metrics
# Results in: cublas_ncu_profiles/ and cublas_metrics_summary/
```

**Want specific metrics?**

```bash
python extract_ncu_metrics.py \
  --ncu-dir quick_test/ncu_profiles \
  --metric-set pipe
# Results show Tensor Core, LSU, ALU utilization
```

That's it! For details, see [QUICKREF.md](QUICKREF.md) or [README.md](README.md).
