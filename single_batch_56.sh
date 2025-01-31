#!/bin/sh
#SBATCH --job-name=kv56
#SBATCH --output=kv56_%A.out
#SBATCH --error=kv56_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=56 >kv56.out 2>kv56.err
