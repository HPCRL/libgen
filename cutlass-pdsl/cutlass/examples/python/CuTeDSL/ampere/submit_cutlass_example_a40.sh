#!/bin/bash
#SBATCH -M notchpeak
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=11:59:00
#SBATCH --job-name=a40_cute_example
#SBATCH --output=/scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/sbatch/%j_%x.log
#SBATCH --error=/scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/sbatch/%j_%x.err

module load cuda/12.1.0
source /scratch/general/vast/sinamps/libgen/venv_pdsl/bin/activate

cd /scratch/general/vast/sinamps/libgen/libgen_repo/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere
# Use unbuffered Python to ensure logs flush promptly into SLURM files
export PYTHONUNBUFFERED=1
python -u sweep_tensorop_gemm.py --problems_csv problems_others.csv --config tune_config.yaml --out sweep_results_v2_otherProblems_A40.csv --resume \
  --resume_from_log sbatch/7760574_a40_cute_example.log
