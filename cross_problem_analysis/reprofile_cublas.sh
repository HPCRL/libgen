#!/bin/bash
# Re-profile cuBLAS with Direct API
# This script re-profiles all cuBLAS kernels using the new C++ runner

set -e  # Exit on error

echo "========================================="
echo "Re-profiling cuBLAS with Direct API"
echo "========================================="
echo ""

# Check if cublas_gemm_runner exists
if [ ! -f "cublas_gemm_runner" ]; then
    echo "Building cublas_gemm_runner..."
    make
    echo ""
fi

# Test the runner
echo "Testing runner..."
./cublas_gemm_runner --M 128 --N 128 --K 128 || {
    echo "Error: Runner test failed"
    exit 1
}
echo ""

# Backup old profiles if they exist
if [ -d "results_5_on_cublas" ]; then
    BACKUP_DIR="results_5_on_cublas_pytorch_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old PyTorch-based profiles to $BACKUP_DIR..."
    mv results_5_on_cublas "$BACKUP_DIR"
    echo ""
fi

# Create new profiles directory
mkdir -p results_5_on_cublas

# Profile each problem using the Python profiler
echo "Profiling cuBLAS kernels (this will take a few minutes)..."
echo ""

# Use Python to profile with the C++ runner
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from cublas_profiler import CuBLASProfiler
from config_manager import ProblemShape

# Define the 5 problem shapes
problems = [
    ProblemShape(M=512, N=512, K=512, L=1),
    ProblemShape(M=1024, N=1024, K=1024, L=1),
    ProblemShape(M=2048, N=2048, K=2048, L=1),
    ProblemShape(M=4096, N=4096, K=4096, L=1),
    ProblemShape(M=8192, N=8192, K=8192, L=1),
]

# Create profiler with C++ runner
profiler = CuBLASProfiler(
    cublas_runner_script="./cublas_gemm_runner",
    output_dir="results_5_on_cublas",
    ncu_sets="full",
    ncu_iterations=1,
    use_cpp_runner=True
)

# Profile all problems
results = profiler.profile_problems(problems, verbose=True)

# Print summary
successful = sum(1 for r in results if r.success)
failed = len(results) - successful

print("\n" + "="*80)
print("Re-profiling Complete!")
print(f"  Successful: {successful}/{len(results)}")
print(f"  Failed:     {failed}/{len(results)}")
print(f"  Profiles:   results_5_on_cublas/")
print("="*80)

if failed > 0:
    print("\nFailed problems:")
    for r in results:
        if not r.success:
            print(f"  - {r.problem}: {r.error_message}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Re-profiling successful!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run analysis: python analyze_and_plot.py"
    echo "  2. Check plots in: analysis_output/"
    echo "  3. Review CSV: analysis_output/kernel_analysis_summary.csv"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ Re-profiling failed!"
    echo "========================================="
    echo ""
    echo "Check the error messages above."
    exit 1
fi
