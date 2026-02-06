"""
__init__.py

Cross-problem analysis package for CuTe DSL GEMM kernels.
"""

from .config_manager import ConfigManager, ProblemShape, KernelConfig, BestConfig
from .kernel_runner import KernelRunner, PerformanceResult
from .ncu_profiler import NCUProfiler, NCUProfileResult
from .ncu_metrics_extractor import (
    NCUMetricsExtractor,
    NCUMetricConfig,
    PIPE_UTILIZATION_METRICS,
    MEMORY_METRICS,
    COMPUTE_METRICS,
    create_custom_metric_set,
    extract_pipe_utilization_metrics,
    extract_all_important_metrics,
)
from .cublas_profiler import CuBLASProfiler, CuBLASProfileResult, create_cublas_runner_script

__all__ = [
    "ConfigManager",
    "ProblemShape",
    "KernelConfig",
    "BestConfig",
    "KernelRunner",
    "PerformanceResult",
    "NCUProfiler",
    "NCUProfileResult",
    "NCUMetricsExtractor",
    "NCUMetricConfig",
    "PIPE_UTILIZATION_METRICS",
    "MEMORY_METRICS",
    "COMPUTE_METRICS",
    "create_custom_metric_set",
    "extract_pipe_utilization_metrics",
    "extract_all_important_metrics",
    "CuBLASProfiler",
    "CuBLASProfileResult",
    "create_cublas_runner_script",
]
