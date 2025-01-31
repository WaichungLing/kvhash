#!/bin/sh
#SBATCH --job-name=kv64
#SBATCH --output=kv64_%A.out
#SBATCH --error=kv64_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=64 >kv64.out 2>kv64.err
