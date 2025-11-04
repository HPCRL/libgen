#!/bin/bash
#SBATCH -M notchpeak
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=11:59:00
#SBATCH --job-name=a40_cute_example
#SBATCH --output=/scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/sbatch/%j_%x.log
#SBATCH --error=/scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/sbatch/%j_%x.err

module load cuda/12.1.0
source /scratch/general/vast/sinamps/libgen/venv_cutlass_pdsl/bin/activate

cd /scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere
python sweep_tensorop_gemm.py --problems_csv problems_others.csv --out sweep_results_v2_otherProblems_A40.csv > sweep_v2_otherProblems_A40.log 2>&1