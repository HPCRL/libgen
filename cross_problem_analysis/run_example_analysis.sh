#!/bin/bash
# run_example_analysis.sh
#
# Convenience wrapper to run the cross-problem analysis.
# Automatically activates conda environment and runs the Python script.
#
# Usage:
#   ./run_example_analysis.sh                           # Use defaults (indices 3,7,12,17,18)
#   ./run_example_analysis.sh --problem_indices 0,1,2   # Custom problem indices
#   ./run_example_analysis.sh --skip_ncu                # Skip NCU profiling
#   ./run_example_analysis.sh --help                    # Show all options

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Cross-Problem Kernel Analysis"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate conda environment
echo -e "${YELLOW}Activating conda environment: cutlass-pdsl${NC}"
eval "$(conda shell.bash hook)"
conda activate cutlass-pdsl

# Change to script directory
cd "$SCRIPT_DIR"

# Run the Python script with all arguments passed through
echo -e "${GREEN}Starting analysis...${NC}"
echo ""
python run_example_analysis.py "$@"

# Exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Analysis completed successfully!${NC}"
    echo ""
    echo "View results:"
    echo "  - CSV data:    cat results_analysis/performance_results.csv"
    echo "  - Summary:     cat results_analysis/summary.json | python -m json.tool"
    echo "  - NCU reports: ls results_analysis/ncu_profiles/ 2>/dev/null || echo 'No NCU profiles'"
else
    echo ""
    echo -e "${YELLOW}⚠ Analysis completed with errors.${NC}"
    echo "Check results_analysis/ for partial results."
fi
