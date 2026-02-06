#!/usr/bin/env python3
"""
Comprehensive analysis and plotting tool for cross-problem kernel analysis.

This script:
1. Extracts detailed metrics from NCU profiles (CuTe DSL, cuBLAS, CUTLASS C++)
2. Generates a summary CSV with occupancy, utilization, and instruction-level metrics
3. Creates visualization plots comparing performance across kernels and problem sizes
"""

import os
import sys
import csv
import subprocess
import re
from pathlib import Path
from collections import defaultdict
import argparse

try:
    from ncu_metrics_extractor import NCUMetricsExtractor
except ImportError:
    from .ncu_metrics_extractor import NCUMetricsExtractor


# Metric configurations for extraction
ANALYSIS_METRICS = [
    # Launch configuration
    ("launch__grid_size", "Grid_Size"),
    ("launch__block_size", "Block_Size"),
    ("launch__registers_per_thread", "Registers_Per_Thread"),
    ("launch__shared_mem_per_block_allocated", "Shared_Memory_Bytes"),
    
    # Occupancy
    ("launch__occupancy_limit_warps", "Theoretical_Occupancy_Warps"),
    ("launch__occupancy_limit_blocks", "Theoretical_Occupancy_Blocks"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved_Occupancy_%"),
    
    # Instructions
    ("sm__inst_executed.sum", "Instructions_Executed"),
    ("sm__inst_executed.avg.per_cycle_active", "Instructions_Per_Cycle"),
    
    # Throughput/Utilization
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM_Throughput_%"),
    
    # Performance
    ("gpu__time_duration.sum", "Duration_ns"),
]


class KernelAnalyzer:
    """Analyzes NCU profiles and extracts detailed metrics"""
    
    def __init__(self, output_dir="analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_problem_size_from_filename(self, filename):
        """Extract M, N, K, L from filename"""
        pattern = r'M(\d+)_N(\d+)_K(\d+)_L(\d+)'
        match = re.search(pattern, filename)
        if match:
            return {
                'M': int(match.group(1)),
                'N': int(match.group(2)),
                'K': int(match.group(3)),
                'L': int(match.group(4))
            }
        return None
    
    def extract_kernel_config_from_filename(self, filename):
        """Extract kernel configuration from filename"""
        # For CuTe DSL: cta64x128x64_s4_atom2x2x1_mnn
        pattern = r'cta(\d+)x(\d+)x(\d+)_s(\d+)_atom(\d+)x(\d+)x(\d+)_([a-z]+)'
        match = re.search(pattern, filename)
        if match:
            return f"cta{match.group(1)}x{match.group(2)}x{match.group(3)}_s{match.group(4)}"
        return None
    
    def extract_metrics_from_profile(self, profile_path):
        """Extract all metrics from a single NCU profile"""
        metrics_dict = {}
        
        # Build NCU command
        metric_names = [m[0] for m in ANALYSIS_METRICS]
        base_fields = ["ID", "Kernel Name"]
        
        cmd = [
            "ncu",
            "--import", profile_path,
            "--csv",
            "--page", "raw",
            "--metrics", ",".join(base_fields + metric_names)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"Warning: NCU extraction failed for {profile_path}")
                return None
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Determine profile type from filename
            filename = Path(profile_path).name
            is_cutedsl = 'cta' in filename and 'atom' in filename
            is_cutlass_cpp = 'cutlass_api2' in filename
            
            # Find the GEMM kernel (skip helper kernels)
            reader = csv.DictReader(lines)
            best_kernel = None
            best_duration = 0.0
            
            for row in reader:
                kernel_name = row.get('Kernel Name', '')
                kid = row.get('ID', '')
                if not kernel_name:
                    continue
                
                # Skip random number generation and other helper kernels
                skip_patterns = ['distribution_', 'reduce', 'elementwise', 'copy_kernel', 'transpose_', 'fill_']
                if any(pattern in kernel_name.lower() for pattern in skip_patterns):
                    continue
                
                # Look for main GEMM/MMA kernels
                if ('gemm' in kernel_name.lower() or 
                    'ampere' in kernel_name.lower() or
                    'mma' in kernel_name.lower() or
                    'xmma' in kernel_name.lower() or
                    'hgemm' in kernel_name.lower() or
                    'sm80' in kernel_name.lower() or
                    'tensorop' in kernel_name.lower()):
                    
                    # Extract metrics for this kernel
                    kernel_metrics = {}
                    for metric_name, friendly_name in ANALYSIS_METRICS:
                        value = row.get(metric_name, '')
                        kernel_metrics[friendly_name] = value
                    kernel_metrics['Kernel_Name'] = kernel_name
                    kernel_metrics['Kernel_ID'] = kid
                    
                    # For CuTe DSL: Take kernel with ID=2 (the main GEMM kernel)
                    if is_cutedsl:
                        if kid == '2':
                            best_kernel = kernel_metrics
                            break  # Found the right kernel
                        elif best_kernel is None:  # Fallback if ID=2 not found
                            best_kernel = kernel_metrics
                    
                    # For CUTLASS C++: Take the first GEMM kernel (skip the 59 repeats)
                    elif is_cutlass_cpp:
                        if best_kernel is None:  # Take first one only
                            best_kernel = kernel_metrics
                            break
                    
                    # For cuBLAS and others: Keep the kernel with longest duration
                    else:
                        try:
                            duration_str = kernel_metrics.get('Duration_ns', '0')
                            if duration_str:
                                duration = float(duration_str)
                                if duration > best_duration:
                                    best_duration = duration
                                    best_kernel = kernel_metrics
                        except:
                            if best_kernel is None:
                                best_kernel = kernel_metrics
            
            if best_kernel:
                metrics_dict = best_kernel
            
            return metrics_dict if metrics_dict else None
            
            return metrics_dict if metrics_dict else None
            
        except Exception as e:
            print(f"Error extracting from {profile_path}: {e}")
            return None
    
    def calculate_gpu_utilization(self, metrics, num_sms=108):
        """
        Calculate GPU utilization:
        GPU_Util = Grid_Size / (Theoretical_Occupancy_Blocks * num_SMs)
        """
        try:
            grid_size = float(metrics.get('Grid_Size', 0))
            theo_occ_blocks = float(metrics.get('Theoretical_Occupancy_Blocks', 1))
            if theo_occ_blocks > 0:
                utilization = (grid_size / (theo_occ_blocks * num_sms)) * 100
                return f"{utilization:.2f}"
            return "N/A"
        except:
            return "N/A"
    
    def analyze_profiles_in_directory(self, ncu_dir, kernel_type, num_sms=108):
        """Analyze all NCU profiles in a directory"""
        ncu_dir = Path(ncu_dir)
        results = []
        
        # Find all .ncu-rep files
        profile_files = list(ncu_dir.glob("*.ncu-rep"))
        
        print(f"\nAnalyzing {len(profile_files)} profiles in {ncu_dir}")
        
        for profile_path in sorted(profile_files):
            print(f"  Processing {profile_path.name}...")
            
            # Extract metrics
            metrics = self.extract_metrics_from_profile(str(profile_path))
            if not metrics:
                print(f"    Warning: Could not extract metrics")
                continue
            
            # Extract problem size
            problem = self.extract_problem_size_from_filename(profile_path.name)
            
            # Extract kernel config (for CuTe DSL)
            kernel_config = self.extract_kernel_config_from_filename(profile_path.name)
            
            # Build result row
            result = {
                'Kernel_Type': kernel_type,
                'Profile_File': profile_path.name,
                'M': problem['M'] if problem else 'N/A',
                'N': problem['N'] if problem else 'N/A',
                'K': problem['K'] if problem else 'N/A',
                'L': problem['L'] if problem else 'N/A',
                'Kernel_Config': kernel_config if kernel_config else 'N/A',
                **metrics,
                'GPU_Utilization_%': self.calculate_gpu_utilization(metrics, num_sms)
            }
            
            results.append(result)
        
        return results
    
    def generate_summary_csv(self, all_results, output_file="kernel_analysis_summary.csv"):
        """Generate comprehensive summary CSV"""
        output_path = self.output_dir / output_file
        
        if not all_results:
            print("No results to write")
            return
        
        # Define column order
        columns = [
            'Kernel_Type', 'Profile_File', 'M', 'N', 'K', 'L', 'Kernel_Config',
            'Kernel_Name', 'Kernel_ID', 'Grid_Size', 'Block_Size',
            'Theoretical_Occupancy_Warps', 'Theoretical_Occupancy_Blocks',
            'Achieved_Occupancy_%', 'GPU_Utilization_%',
            'Registers_Per_Thread', 'Shared_Memory_Bytes',
            'Instructions_Executed', 'Instructions_Per_Cycle',
            'SM_Throughput_%', 'Duration_ns'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Summary CSV saved to {output_path}")
        print(f"  Total kernels analyzed: {len(all_results)}")
        return output_path
    
    def create_plots(self, summary_csv):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("\nWarning: matplotlib/pandas not available. Skipping plots.")
            print("Install with: pip install matplotlib pandas")
            return
        
        # Read summary data
        df = pd.read_csv(summary_csv)
        
        # Convert numeric columns
        numeric_cols = ['M', 'N', 'K', 'Duration_ns', 'Achieved_Occupancy_%', 'SM_Throughput_%']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate GFLOPS
        df['GFLOPS'] = (2 * df['M'] * df['N'] * df['K']) / (df['Duration_ns'])
        
        # Create problem size labels
        df['Problem'] = df['M'].astype(str) + 'x' + df['N'].astype(str) + 'x' + df['K'].astype(str)
        
        self._plot_performance_by_problem(df)
        self._plot_performance_by_kernel(df)
        self._plot_occupancy_comparison(df)
        self._plot_throughput_comparison(df)
    
    def _plot_performance_by_problem(self, df):
        """Plot 1: Performance of different kernels on same input"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        problems = df[['M', 'N', 'K']].drop_duplicates().values
        
        for idx, (M, N, K) in enumerate(problems):
            if idx >= 6:
                break
            
            ax = axes[idx]
            problem_df = df[(df['M'] == M) & (df['N'] == N) & (df['K'] == K)]
            
            # Group by kernel type
            for kernel_type in ['CuTe_DSL', 'cuBLAS', 'CUTLASS_CPP']:
                type_df = problem_df[problem_df['Kernel_Type'] == kernel_type]
                if not type_df.empty:
                    if kernel_type == 'CuTe_DSL':
                        # Plot each config separately
                        for config in type_df['Kernel_Config'].unique():
                            config_df = type_df[type_df['Kernel_Config'] == config]
                            ax.bar(f"{kernel_type}\n{config}", config_df['GFLOPS'].values[0],
                                   alpha=0.7, label=config if idx == 0 else "")
                    else:
                        ax.bar(kernel_type, type_df['GFLOPS'].values[0],
                               alpha=0.7, label=kernel_type if idx == 0 else "")
            
            ax.set_title(f'M={int(M)}, N={int(N)}, K={int(K)}')
            ax.set_ylabel('GFLOPS')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'performance_by_problem.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")
    
    def _plot_performance_by_kernel(self, df):
        """Plot 2: Performance of each kernel on different inputs"""
        import matplotlib.pyplot as plt
        
        # Get unique CuTe DSL configs
        cutedsl_df = df[df['Kernel_Type'] == 'CuTe_DSL']
        configs = cutedsl_df['Kernel_Config'].unique()
        
        fig, axes = plt.subplots(len(configs) + 2, 1, figsize=(14, 4 * (len(configs) + 2)))
        
        # Plot each CuTe DSL config
        for idx, config in enumerate(configs):
            ax = axes[idx]
            config_df = cutedsl_df[cutedsl_df['Kernel_Config'] == config].sort_values('Problem')
            
            ax.plot(config_df['Problem'], config_df['GFLOPS'], 'o-', label=f'CuTe DSL: {config}', linewidth=2)
            ax.set_title(f'CuTe DSL Config: {config}')
            ax.set_ylabel('GFLOPS')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot cuBLAS
        ax = axes[len(configs)]
        cublas_df = df[df['Kernel_Type'] == 'cuBLAS'].sort_values('Problem')
        ax.plot(cublas_df['Problem'], cublas_df['GFLOPS'], 's-', label='cuBLAS', linewidth=2, color='red')
        ax.set_title('cuBLAS Performance')
        ax.set_ylabel('GFLOPS')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot CUTLASS C++
        ax = axes[len(configs) + 1]
        cutlass_df = df[df['Kernel_Type'] == 'CUTLASS_CPP'].sort_values('Problem')
        ax.plot(cutlass_df['Problem'], cutlass_df['GFLOPS'], '^-', label='CUTLASS C++', linewidth=2, color='green')
        ax.set_title('CUTLASS C++ Performance')
        ax.set_ylabel('GFLOPS')
        ax.set_xlabel('Problem Size')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / 'performance_by_kernel.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")
    
    def _plot_occupancy_comparison(self, df):
        """Plot occupancy comparison"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Achieved occupancy by kernel type
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax1.scatter(range(len(type_df)), type_df['Achieved_Occupancy_%'], 
                       label=kernel_type, alpha=0.7, s=100)
        
        ax1.set_xlabel('Kernel Index')
        ax1.set_ylabel('Achieved Occupancy (%)')
        ax1.set_title('Achieved Occupancy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SM Throughput by kernel type
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax2.scatter(range(len(type_df)), type_df['SM_Throughput_%'], 
                       label=kernel_type, alpha=0.7, s=100)
        
        ax2.set_xlabel('Kernel Index')
        ax2.set_ylabel('SM Throughput (%)')
        ax2.set_title('SM Throughput Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'occupancy_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")
    
    def _plot_throughput_comparison(self, df):
        """Plot IPC and resource usage"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Calculate the maximum kernel index across all kernel types
        max_index = max(len(df[df['Kernel_Type'] == kt]) for kt in df['Kernel_Type'].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Instructions per cycle
        ax = axes[0, 0]
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax.scatter(range(len(type_df)), 
                      pd.to_numeric(type_df['Instructions_Per_Cycle'], errors='coerce'),
                      label=kernel_type, alpha=0.7, s=100)
        ax.set_xlim(0, max_index)
        ax.set_ylabel('Instructions Per Cycle')
        ax.set_title('Instruction-Level Parallelism')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Register usage
        ax = axes[0, 1]
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax.scatter(range(len(type_df)), 
                      pd.to_numeric(type_df['Registers_Per_Thread'], errors='coerce'),
                      label=kernel_type, alpha=0.7, s=100)
        ax.set_xlim(0, max_index)
        ax.set_ylabel('Registers Per Thread')
        ax.set_title('Register Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Shared memory usage
        ax = axes[1, 0]
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax.scatter(range(len(type_df)), 
                      pd.to_numeric(type_df['Shared_Memory_Bytes'], errors='coerce') / 1024,
                      label=kernel_type, alpha=0.7, s=100)
        ax.set_xlim(0, max_index)
        ax.set_ylabel('Shared Memory (KB)')
        ax.set_title('Shared Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # GPU Utilization
        ax = axes[1, 1]
        for kernel_type in df['Kernel_Type'].unique():
            type_df = df[df['Kernel_Type'] == kernel_type]
            ax.scatter(range(len(type_df)), 
                      pd.to_numeric(type_df['GPU_Utilization_%'], errors='coerce'),
                      label=kernel_type, alpha=0.7, s=100)
        ax.set_xlim(0, max_index)
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'resource_usage.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NCU profiles and generate performance plots"
    )
    parser.add_argument(
        '--cutedsl-dir',
        default='results_5x5_on_v2_cutedsl/ncu_profiles',
        help='Directory with CuTe DSL NCU profiles'
    )
    parser.add_argument(
        '--cublas-dir',
        default='results_5_on_cublas_direct',
        help='Directory with cuBLAS NCU profiles'
    )
    parser.add_argument(
        '--cutlass-dir',
        default='results_5pz_cutlass_api2_padded_bench',
        help='Directory with CUTLASS C++ NCU profiles'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis_output',
        help='Output directory for CSV and plots'
    )
    parser.add_argument(
        '--num-sms',
        type=int,
        default=108,
        help='Number of SMs on GPU (default: 108 for A100)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    analyzer = KernelAnalyzer(args.output_dir)
    
    print("="*80)
    print("Kernel Analysis and Visualization Tool")
    print("="*80)
    
    all_results = []
    
    # Analyze CuTe DSL kernels
    if os.path.exists(args.cutedsl_dir):
        results = analyzer.analyze_profiles_in_directory(
            args.cutedsl_dir, 'CuTe_DSL', args.num_sms
        )
        all_results.extend(results)
    else:
        print(f"\nWarning: CuTe DSL directory not found: {args.cutedsl_dir}")
    
    # Analyze cuBLAS kernels
    if os.path.exists(args.cublas_dir):
        results = analyzer.analyze_profiles_in_directory(
            args.cublas_dir, 'cuBLAS', args.num_sms
        )
        all_results.extend(results)
    else:
        print(f"\nWarning: cuBLAS directory not found: {args.cublas_dir}")
    
    # Analyze CUTLASS C++ kernels
    if os.path.exists(args.cutlass_dir):
        results = analyzer.analyze_profiles_in_directory(
            args.cutlass_dir, 'CUTLASS_CPP', args.num_sms
        )
        all_results.extend(results)
    else:
        print(f"\nWarning: CUTLASS C++ directory not found: {args.cutlass_dir}")
    
    # Generate summary CSV
    if all_results:
        summary_csv = analyzer.generate_summary_csv(all_results)
        
        # Generate plots
        if not args.no_plots:
            print("\nGenerating plots...")
            analyzer.create_plots(summary_csv)
    else:
        print("\nNo results to analyze!")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
