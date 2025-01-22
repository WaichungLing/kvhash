#!/bin/sh
#SBATCH --job-name=kv40
#SBATCH --output=kv40_%A.out
#SBATCH --error=kv40_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=40 >kv40.out 2>kv40.err
