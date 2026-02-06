#!/usr/bin/env python3
"""
Example: Extract and analyze NCU metrics from cross-problem profiles.

This script demonstrates how to:
1. Extract specific NCU metrics
2. Load them with csv module
3. Perform basic analysis
"""

import sys
import os
import csv
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncu_metrics_extractor import (
    NCUMetricsExtractor,
    PIPE_UTILIZATION_METRICS,
    MEMORY_METRICS,
    COMPUTE_METRICS
)


def main():
    print("="*80)
    print("NCU Metrics Extraction Example")
    print("="*80)
    
    # Configuration
    ncu_dir = "results_5x5_on_v2/ncu_profiles"
    output_dir = "ncu_metrics_summary"
    
    if not os.path.exists(ncu_dir):
        print(f"\nError: NCU profiles directory not found: {ncu_dir}")
        print("Please run the cross-problem analysis first to generate NCU profiles.")
        return 1
    
    # Create extractor
    extractor = NCUMetricsExtractor(ncu_dir, output_dir)
    
    # Extract all important metrics
    print("\n1. Extracting comprehensive metrics...")
    all_metrics = PIPE_UTILIZATION_METRICS + MEMORY_METRICS + COMPUTE_METRICS
    csv_path = extractor.extract_metrics_from_all_profiles(
        all_metrics,
        "analysis_example.csv"
    )
    
    if not csv_path:
        print("Failed to extract metrics")
        return 1
    
    # Load and analyze with CSV module
    print("\n2. Loading and analyzing metrics...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nTotal rows: {len(rows)}")
    
    # Count unique profiles
    profiles = set(row['Profile_File'] for row in rows)
    print(f"Number of profiles: {len(profiles)}")
    
    # Filter for GEMM kernels
    gemm_rows = [row for row in rows if 'tensorop_gemm' in row.get('Kernel_Name', '')]
    print(f"GEMM kernel invocations: {len(gemm_rows)}")
    
    if len(gemm_rows) > 0:
        print("\n3. Analyzing pipe utilization (GEMM kernels only):")
        print("-" * 80)
        
        # Collect pipe utilization metrics
        pipe_metrics = {}
        pipe_cols = [col for col in gemm_rows[0].keys() if 'Pipe' in col and 'Active_%' in col]
        
        for col in pipe_cols:
            values = []
            for row in gemm_rows:
                try:
                    val = float(row[col])
                    values.append(val)
                except (ValueError, KeyError):
                    pass
            
            if values:
                pipe_metrics[col] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        # Display statistics
        for metric, stats in sorted(pipe_metrics.items()):
            print(f"\n{metric}:")
            print(f"  Min:   {stats['min']:6.2f}%")
            print(f"  Max:   {stats['max']:6.2f}%")
            print(f"  Avg:   {stats['avg']:6.2f}%")
            print(f"  Count: {stats['count']}")
        
        # Duration analysis
        print("\n4. Kernel duration analysis:")
        print("-" * 80)
        durations = []
        for row in gemm_rows:
            try:
                dur = float(row['Duration_us'])
                durations.append(dur)
            except (ValueError, KeyError):
                pass
        
        if durations:
            print(f"Min duration:  {min(durations):8.2f} us")
            print(f"Max duration:  {max(durations):8.2f} us")
            print(f"Mean duration: {sum(durations)/len(durations):8.2f} us")
    
    print("\n" + "="*80)
    print(f"✓ Analysis complete! Full results saved to: {csv_path}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
