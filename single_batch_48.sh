#!/bin/sh
#SBATCH --job-name=kv48
#SBATCH --output=kv48_%A.out
#SBATCH --error=kv48_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=48 >kv48.out 2>kv48.err
