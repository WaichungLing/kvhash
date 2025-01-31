#!/bin/sh
#SBATCH --job-name=kv24
#SBATCH --output=kv24_%A.out
#SBATCH --error=kv24_%A.err
#SBATCH --time=900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinfan@comp.nus.edu.sg
#SBATCH --gres=gpu:h100-47:1

. .venv/bin/activate
srun python run_longbench.py --n_latest=24 >kv24.out 2>kv24.err
