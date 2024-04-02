#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --output=/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/exp_out/online_gpu.out
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=40000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

source /home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/venv_gpr4dur/bin/activate
module load cuda
nvidia-smi


python3 /home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic/synthetic_GPR_browsing.py

deactivate

