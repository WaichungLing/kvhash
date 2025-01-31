#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wling@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-40:1

. .venv/bin/activate
# srun python run_longbench.py --n_latest=24 >kv48.out 2>kv48.err
# srun python run_longbench.py --n_latest=40 >kv40.out 2>kv40.err
# srun python run_longbench.py --n_latest=56 >kv56.out 2>kv56.err
# 8, 16, 24, 32, 40, 48, 56, 64
