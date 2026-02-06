#!/usr/bin/env python3
"""
Script to extract specific NCU metrics from collected profiles.

This script can extract predefined metric sets or custom metrics specified by the user.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Direct execution: use absolute imports
    from ncu_metrics_extractor import (
        NCUMetricsExtractor,
        NCUMetricConfig,
        PIPE_UTILIZATION_METRICS,
        MEMORY_METRICS,
        COMPUTE_METRICS,
        create_custom_metric_set
    )
else:
    # Module execution: use relative imports
    from .ncu_metrics_extractor import (
        NCUMetricsExtractor,
        NCUMetricConfig,
        PIPE_UTILIZATION_METRICS,
        MEMORY_METRICS,
        COMPUTE_METRICS,
        create_custom_metric_set
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific metrics from NCU profile reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract pipe utilization metrics from default location
  python extract_ncu_metrics.py --metric-set pipe

  # Extract all predefined metrics
  python extract_ncu_metrics.py --metric-set all

  # Extract custom metrics
  python extract_ncu_metrics.py --custom-metrics "sm__cycles_active.avg" "gpu__time_duration.sum"

  # Specify custom input/output directories
  python extract_ncu_metrics.py --ncu-dir results_5x5/ncu_profiles --output-dir my_metrics

  # List available metrics from a sample profile
  python extract_ncu_metrics.py --list-metrics results_5x5/ncu_profiles/prob0_cfg0*.ncu-rep
        """
    )
    
    parser.add_argument(
        "--ncu-dir",
        default="results_5x5_on_v2/ncu_profiles",
        help="Directory containing NCU .ncu-rep files (default: results_5x5_on_v2/ncu_profiles)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="ncu_metrics_summary",
        help="Output directory for CSV files (default: ncu_metrics_summary)"
    )
    
    parser.add_argument(
        "--metric-set",
        choices=["pipe", "memory", "compute", "all"],
        help="Predefined metric set to extract"
    )
    
    parser.add_argument(
        "--custom-metrics",
        nargs="+",
        metavar="METRIC",
        help="Custom list of metric names to extract"
    )
    
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Custom output filename (default: based on metric set)"
    )
    
    parser.add_argument(
        "--list-metrics",
        metavar="PROFILE_FILE",
        help="List all available metrics from a sample profile file"
    )
    
    args = parser.parse_args()
    
    # Handle list-metrics mode
    if args.list_metrics:
        if not os.path.exists(args.list_metrics):
            print(f"Error: Profile file not found: {args.list_metrics}")
            return 1
        
        extractor = NCUMetricsExtractor(".", ".")
        metrics = extractor.get_available_metrics(args.list_metrics)
        
        if not metrics:
            print("Failed to retrieve metrics")
            return 1
        
        print(f"\nAvailable metrics ({len(metrics)} total):")
        print("=" * 80)
        for i, metric in enumerate(metrics, 1):
            print(f"{i:4d}. {metric}")
        
        return 0
    
    # Determine which metrics to extract
    metrics_to_extract = []
    output_filename = args.output_filename
    
    if args.metric_set:
        if args.metric_set == "pipe":
            metrics_to_extract = PIPE_UTILIZATION_METRICS
            if not output_filename:
                output_filename = "pipe_utilization_metrics.csv"
        elif args.metric_set == "memory":
            metrics_to_extract = MEMORY_METRICS
            if not output_filename:
                output_filename = "memory_metrics.csv"
        elif args.metric_set == "compute":
            metrics_to_extract = COMPUTE_METRICS
            if not output_filename:
                output_filename = "compute_metrics.csv"
        elif args.metric_set == "all":
            metrics_to_extract = PIPE_UTILIZATION_METRICS + MEMORY_METRICS + COMPUTE_METRICS
            if not output_filename:
                output_filename = "comprehensive_metrics.csv"
    
    elif args.custom_metrics:
        metrics_to_extract = create_custom_metric_set(args.custom_metrics)
        if not output_filename:
            output_filename = "custom_metrics.csv"
    
    else:
        # Default to pipe utilization
        print("No metric set specified, defaulting to pipe utilization metrics")
        metrics_to_extract = PIPE_UTILIZATION_METRICS
        if not output_filename:
            output_filename = "pipe_utilization_metrics.csv"
    
    # Check if NCU profiles directory exists
    if not os.path.exists(args.ncu_dir):
        print(f"Error: NCU profiles directory not found: {args.ncu_dir}")
        return 1
    
    # Extract metrics
    print(f"Extracting {len(metrics_to_extract)} metrics from {args.ncu_dir}")
    print(f"Output will be saved to {args.output_dir}/{output_filename}")
    print()
    
    extractor = NCUMetricsExtractor(args.ncu_dir, args.output_dir)
    output_path = extractor.extract_metrics_from_all_profiles(
        metrics_to_extract,
        output_filename
    )
    
    if output_path:
        print(f"\n✓ Success! Metrics saved to: {output_path}")
        return 0
    else:
        print("\n✗ Failed to extract metrics")
        return 1


if __name__ == "__main__":
    sys.exit(main())
