#!/bin/sh
#SBATCH --job-name=kv0
#SBATCH --output=kv0_%A.out
#SBATCH --error=kv0_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=0 >kv0.out 2>kv0.err
