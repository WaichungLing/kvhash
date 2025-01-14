#!/bin/sh
#SBATCH --job-name=kv
#SBATCH --output=kv_%A.out
#SBATCH --error=kv_%A.err
#SBATCH --time=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@u.nus.edu
#SBATCH --gres=gpu:a100-40:1

. .venv/bin/activate
srun python run_longbench.py >kv.out 2>kv.err
