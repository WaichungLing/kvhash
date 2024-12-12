#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:a100-80:1

. .venv/bin/activate
srun python run_longbench.py --cache_budget=0.4
