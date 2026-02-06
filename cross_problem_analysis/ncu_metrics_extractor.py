"""
NCU Metrics Extractor

This module provides functionality to extract specific hardware counter metrics
from NCU profile reports and generate CSV summaries.
"""

import os
import subprocess
import csv
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class NCUMetricConfig:
    """Configuration for NCU metric extraction"""
    metric_name: str
    display_name: Optional[str] = None
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.metric_name


# Predefined metric sets for common use cases
PIPE_UTILIZATION_METRICS = [
    NCUMetricConfig("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", "ALU_Pipe_Active_%"),
    NCUMetricConfig("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed", "ALU_Pipe_Elapsed_%"),
    NCUMetricConfig("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", "FMA_Pipe_Active_%"),
    NCUMetricConfig("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed", "FMA_Pipe_Elapsed_%"),
    NCUMetricConfig("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "FP64_Pipe_Active_%"),
    NCUMetricConfig("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed", "FP64_Pipe_Elapsed_%"),
    NCUMetricConfig("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "Tensor_Pipe_Active_%"),
    NCUMetricConfig("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "Tensor_Pipe_Elapsed_%"),
    NCUMetricConfig("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", "LSU_Pipe_Active_%"),
    NCUMetricConfig("sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed", "LSU_Pipe_Elapsed_%"),
    NCUMetricConfig("sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active", "TEX_Pipe_Active_%"),
    NCUMetricConfig("sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_elapsed", "TEX_Pipe_Elapsed_%"),
]

MEMORY_METRICS = [
    NCUMetricConfig("dram__bytes_read.sum", "DRAM_Read_MB"),
    NCUMetricConfig("dram__bytes_write.sum", "DRAM_Write_MB"),
    NCUMetricConfig("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM_Throughput_%"),
    NCUMetricConfig("l1tex__t_sector_hit_rate.pct", "L1_Hit_Rate_%"),
    NCUMetricConfig("lts__t_sector_hit_rate.pct", "L2_Hit_Rate_%"),
]

COMPUTE_METRICS = [
    NCUMetricConfig("sm__cycles_active.avg", "SM_Active_Cycles"),
    NCUMetricConfig("sm__warps_active.avg.pct_of_peak_sustained_active", "Warps_Active_%"),
    NCUMetricConfig("sm__inst_executed.avg.per_cycle_active", "Inst_Per_Cycle"),
    NCUMetricConfig("gpu__time_duration.sum", "Duration_us"),
]


class NCUMetricsExtractor:
    """Extracts specific metrics from NCU profile reports"""
    
    def __init__(self, ncu_profiles_dir: str, output_dir: str = "ncu_metrics_summary"):
        """
        Initialize the metrics extractor.
        
        Args:
            ncu_profiles_dir: Directory containing .ncu-rep files
            output_dir: Directory to save extracted metrics CSV
        """
        self.ncu_profiles_dir = ncu_profiles_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_metrics_from_profile(
        self, 
        profile_path: str, 
        metrics: List[NCUMetricConfig]
    ) -> List[Dict[str, str]]:
        """
        Extract specified metrics from a single NCU profile.
        
        Args:
            profile_path: Path to .ncu-rep file
            metrics: List of metrics to extract
            
        Returns:
            List of dictionaries containing metric values for each kernel in the profile
        """
        # Build the ncu command to extract metrics in CSV format
        metric_names = [m.metric_name for m in metrics]
        
        # Add kernel identification metrics
        base_metrics = ["ID", "Kernel Name", "launch__grid_size", "launch__block_size"]
        all_metrics = base_metrics + metric_names
        
        cmd = [
            "ncu",
            "--import", profile_path,
            "--csv",
            "--page", "raw",
            "--metrics", ",".join(all_metrics)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Warning: NCU extraction failed for {profile_path}")
                print(f"Error: {result.stderr}")
                return []
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return []
            
            # Read CSV data
            reader = csv.DictReader(lines)
            results = []
            
            for row in reader:
                # Create result dict with display names
                result_dict = {
                    "Profile_File": os.path.basename(profile_path),
                    "Kernel_ID": row.get("ID", ""),
                    "Kernel_Name": row.get("Kernel Name", "").split('(')[0].strip(),  # Clean kernel name
                    "Grid_Size": row.get("launch__grid_size", ""),
                    "Block_Size": row.get("launch__block_size", ""),
                }
                
                # Add requested metrics with display names
                for metric in metrics:
                    value = row.get(metric.metric_name, "")
                    result_dict[metric.display_name] = value
                
                results.append(result_dict)
            
            return results
            
        except subprocess.TimeoutExpired:
            print(f"Warning: NCU extraction timeout for {profile_path}")
            return []
        except Exception as e:
            print(f"Error extracting metrics from {profile_path}: {e}")
            return []
    
    def extract_metrics_from_all_profiles(
        self,
        metrics: List[NCUMetricConfig],
        output_filename: str = "ncu_metrics.csv"
    ) -> str:
        """
        Extract metrics from all NCU profiles in the directory.
        
        Args:
            metrics: List of metrics to extract
            output_filename: Name of output CSV file
            
        Returns:
            Path to the generated CSV file
        """
        # Find all .ncu-rep files
        profile_files = []
        for root, dirs, files in os.walk(self.ncu_profiles_dir):
            for file in files:
                if file.endswith('.ncu-rep'):
                    profile_files.append(os.path.join(root, file))
        
        if not profile_files:
            print(f"No NCU profile files found in {self.ncu_profiles_dir}")
            return ""
        
        print(f"Found {len(profile_files)} NCU profile files")
        
        # Extract metrics from each profile
        all_results = []
        for i, profile_path in enumerate(profile_files, 1):
            print(f"Processing {i}/{len(profile_files)}: {os.path.basename(profile_path)}")
            results = self.extract_metrics_from_profile(profile_path, metrics)
            all_results.extend(results)
        
        if not all_results:
            print("No metrics extracted from any profiles")
            return ""
        
        # Write to CSV
        output_path = os.path.join(self.output_dir, output_filename)
        fieldnames = list(all_results[0].keys())
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Extracted {len(all_results)} kernel metrics to {output_path}")
        return output_path
    
    def get_available_metrics(self, sample_profile_path: str) -> List[str]:
        """
        Get list of all available metrics from a sample profile.
        
        Args:
            sample_profile_path: Path to any .ncu-rep file
            
        Returns:
            List of metric names available in NCU profiles
        """
        cmd = [
            "ncu",
            "--import", sample_profile_path,
            "--csv",
            "--page", "raw"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return []
            
            # Parse CSV header
            lines = result.stdout.strip().split('\n')
            if len(lines) < 1:
                return []
            
            # Split header by comma (note: some metric names may contain commas in quotes)
            reader = csv.reader([lines[0]])
            headers = next(reader)
            
            return headers
            
        except Exception as e:
            print(f"Error getting available metrics: {e}")
            return []


def create_custom_metric_set(metric_names: List[str], name_prefix: str = "") -> List[NCUMetricConfig]:
    """
    Helper function to create a custom metric configuration set.
    
    Args:
        metric_names: List of NCU metric names
        name_prefix: Optional prefix for display names
        
    Returns:
        List of NCUMetricConfig objects
    """
    configs = []
    for name in metric_names:
        # Create a simplified display name
        display_name = name.split('.')[-1] if '.' in name else name
        if name_prefix:
            display_name = f"{name_prefix}_{display_name}"
        configs.append(NCUMetricConfig(name, display_name))
    return configs


# Example usage function
def extract_pipe_utilization_metrics(ncu_profiles_dir: str, output_dir: str = "ncu_metrics_summary"):
    """
    Convenience function to extract pipe utilization metrics.
    
    Args:
        ncu_profiles_dir: Directory containing NCU profile files
        output_dir: Output directory for CSV
        
    Returns:
        Path to generated CSV file
    """
    extractor = NCUMetricsExtractor(ncu_profiles_dir, output_dir)
    return extractor.extract_metrics_from_all_profiles(
        PIPE_UTILIZATION_METRICS,
        "pipe_utilization_metrics.csv"
    )


def extract_all_important_metrics(ncu_profiles_dir: str, output_dir: str = "ncu_metrics_summary"):
    """
    Convenience function to extract a comprehensive set of important metrics.
    
    Args:
        ncu_profiles_dir: Directory containing NCU profile files
        output_dir: Output directory for CSV
        
    Returns:
        Path to generated CSV file
    """
    all_metrics = PIPE_UTILIZATION_METRICS + MEMORY_METRICS + COMPUTE_METRICS
    extractor = NCUMetricsExtractor(ncu_profiles_dir, output_dir)
    return extractor.extract_metrics_from_all_profiles(
        all_metrics,
        "comprehensive_metrics.csv"
    )
